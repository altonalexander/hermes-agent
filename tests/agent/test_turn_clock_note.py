"""The per-turn <current_time> note.

The system prompt carries a date-only line, built once per session and rebuilt
only after compaction, deliberately, so the provider's prefix KV cache stays
byte-stable. The cost is that the model has no idea what time it is, which zone
the date is in, or — past midnight on a long session — what day it is.

The clock therefore rides the current *user turn*, through the same api_content
sidecar the gateway already uses to keep volatile facts out of the system
prompt. These tests pin the two properties that make that safe: the system
prompt stays stable, and the note is computed once per turn so the persisted
sidecar can never disagree with the bytes on the wire.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

import hermes_time
from agent.turn_context import build_turn_context, compose_user_api_content

from tests.agent.test_gateway_turn_sidecar import _FakeAgent, _build


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


NOTE = "<current_time>Sunday, November 01, 2026 at 07:04 MST (America/Denver)</current_time>"


def _patch_clock(**kwargs):
    """Patch the clock on the module ``build_turn_context`` closes over.

    Not by dotted name: tests/agent/test_empty_tool_name_loop_dampening.py
    purges sys.modules of ``hermes_*``/``agent.*``, after which the name
    resolves to a different module object than this already-imported function
    uses, and the patch silently does nothing.
    """
    return patch.object(
        build_turn_context.__globals__["hermes_time"], "current_time_note", **kwargs
    )


class TestRendering:
    def test_names_both_abbreviation_and_iana_zone(self):
        """The model must never have to infer a zone or do offset arithmetic."""
        with patch.dict("os.environ", {"HERMES_TIMEZONE": "America/Denver"}):
            hermes_time.reset_cache()
            note = hermes_time.current_time_note()
        hermes_time.reset_cache()
        assert note.startswith("<current_time>")
        assert note.endswith("</current_time>")
        assert "America/Denver" in note
        assert "MDT" in note or "MST" in note

    def test_includes_day_of_week_and_time(self):
        with patch.dict("os.environ", {"HERMES_TIMEZONE": "UTC"}):
            hermes_time.reset_cache()
            note = hermes_time.current_time_note()
        hermes_time.reset_cache()
        assert " at " in note
        assert any(d in note for d in
                   ["Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"])

    def test_reflects_the_configured_zone_not_utc(self):
        """The whole point — a UTC host must still report the user's day."""
        with patch.dict("os.environ", {"HERMES_TIMEZONE": "Pacific/Kiritimati"}):
            hermes_time.reset_cache()
            ahead = hermes_time.current_time_note()
        with patch.dict("os.environ", {"HERMES_TIMEZONE": "Pacific/Midway"}):
            hermes_time.reset_cache()
            behind = hermes_time.current_time_note()
        hermes_time.reset_cache()
        # UTC+14 vs UTC-11: never the same wall clock.
        assert ahead != behind


class TestDeliveredOnTheTurn:
    def test_note_rides_the_api_content_sidecar(self):
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
             _patch_clock(return_value=NOTE):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        # Stored content stays clean; only the API copy carries the clock.
        assert msg["content"] == "hello"
        assert msg["api_content"] == "hello\n\n" + NOTE

    def test_sidecar_matches_what_conversation_loop_would_send(self):
        """The cache invariant: persisted sidecar == bytes on the wire."""
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
             _patch_clock(return_value=NOTE):
            ctx = _build(agent)
            msg = ctx.messages[ctx.current_turn_user_idx]
            rebuilt = compose_user_api_content(
                "hello", ctx.ext_prefetch_cache, ctx.plugin_user_context
            )
        assert msg["api_content"] == rebuilt

    def test_computed_once_per_turn(self):
        """Guard against moving the clock read into compose_user_api_content.

        That function runs twice per turn (the prologue's sidecar stamp and the
        api_messages build). A clock read inside it could straddle a minute
        boundary and produce two different strings for one turn, which would
        break the sidecar/wire invariant above.
        """
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
             _patch_clock(return_value=NOTE) as spy:
            ctx = _build(agent)
            # Re-composing must not consult the clock again.
            compose_user_api_content(
                "hello", ctx.ext_prefetch_cache, ctx.plugin_user_context
            )
        assert spy.call_count == 1

    def test_appended_as_text_part_on_multimodal_turns(self):
        """Image turns can't take the string sidecar — the clock must not drop."""
        agent = _FakeAgent()
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
        ]
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
             _patch_clock(return_value=NOTE):
            ctx = _build(agent, user_message=content)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["content"][-1] == {"type": "text", "text": NOTE}

    def test_clock_failure_does_not_break_the_turn(self):
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
             _patch_clock(side_effect=RuntimeError("boom")):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["content"] == "hello"
        assert "api_content" not in msg


def _reset_live_clock_cache() -> None:
    """Clear the zone cache on every live ``hermes_time`` module object.

    Another test module purges sys.modules of ``hermes_*``, so the object this
    file imported at collection time and the one ``build_system_prompt_parts``
    resolves at call time can be different, each with its own cache. Resetting
    only ours would leave the one that actually answers still holding a stale
    zone.
    """
    import importlib

    hermes_time.reset_cache()
    try:
        importlib.import_module("hermes_time").reset_cache()
    except Exception:
        pass


def _volatile_parts(zone: str) -> str:
    """Build just the volatile system-prompt block under a given timezone."""
    from agent.system_prompt import build_system_prompt_parts

    from tests.agent.test_system_prompt import _make_agent

    with patch.dict("os.environ", {"HERMES_TIMEZONE": zone}), \
         patch("run_agent.load_soul_md", return_value=""), \
         patch("run_agent.build_nous_subscription_prompt", return_value=""), \
         patch("run_agent.build_environment_hints", return_value=""), \
         patch("run_agent.build_context_files_prompt", return_value=""):
        _reset_live_clock_cache()
        parts = build_system_prompt_parts(_make_agent())
    _reset_live_clock_cache()
    return parts["volatile"]


class TestSystemPromptStaysStable:
    def test_prompt_names_the_zone(self):
        """Stable all day, so it costs nothing in prefix-cache terms."""
        assert "(America/Denver)" in _volatile_parts("America/Denver")

    def test_prompt_carries_no_clock(self):
        """A time here would churn the cached prefix on every rebuild."""
        volatile = _volatile_parts("America/Denver")
        assert "<current_time>" not in volatile
        # The date line must not have gained hours:minutes.
        started = [l for l in volatile.split("\n") if l.startswith("Conversation started:")]
        assert started, volatile
        assert ":" not in started[0].split("started:", 1)[1]

    def test_zone_label_tracks_the_configured_zone(self):
        assert "(Asia/Tokyo)" in _volatile_parts("Asia/Tokyo")
