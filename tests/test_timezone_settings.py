"""The timezone Settings field must actually take effect.

``hermes_time`` caches the resolved zone for the process lifetime, so before
this every write path — the dashboard's PUT /api/config, the desktop app,
``hermes config set`` — changed config.yaml while every long-lived process
(notably the gateway) kept serving the old zone until it restarted. The setting
looked broken.
"""

import os
from unittest.mock import patch

import pytest

import hermes_time
from hermes_cli.config import _invalidate_timezone_cache


@pytest.fixture(autouse=True)
def clean_timezone_state():
    previous = os.environ.get("HERMES_TIMEZONE")
    os.environ.pop("HERMES_TIMEZONE", None)
    hermes_time.reset_cache()
    yield
    if previous is None:
        os.environ.pop("HERMES_TIMEZONE", None)
    else:
        os.environ["HERMES_TIMEZONE"] = previous
    hermes_time.reset_cache()


class TestCacheInvalidation:
    def test_invalidate_clears_resolved_state(self):
        os.environ["HERMES_TIMEZONE"] = "America/Denver"
        assert str(hermes_time.get_timezone()) == "America/Denver"

        os.environ["HERMES_TIMEZONE"] = "Asia/Tokyo"
        assert str(hermes_time.get_timezone()) == "America/Denver", (
            "precondition: the resolved zone is cached"
        )

        _invalidate_timezone_cache()
        assert str(hermes_time.get_timezone()) == "Asia/Tokyo"

    def test_invalidate_never_raises(self):
        """A display concern must not be able to fail a config write."""
        with patch.object(hermes_time, "reset_cache", side_effect=RuntimeError("boom")):
            _invalidate_timezone_cache()  # must not propagate


class TestSaveConfigInvalidates:
    def test_save_config_picks_up_new_timezone(self, tmp_path):
        """Writing config.yaml makes the next now() use the new zone."""
        from hermes_cli.config import save_config

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config({"timezone": "America/Denver"})
            hermes_time.reset_cache()
            assert str(hermes_time.get_timezone()) == "America/Denver"

            # The write itself must clear the cache — no manual reset here.
            save_config({"timezone": "Asia/Tokyo"})
            assert str(hermes_time.get_timezone()) == "Asia/Tokyo"


class TestResolutionPrecedence:
    def test_env_beats_config(self, tmp_path):
        """HERMES_TIMEZONE wins — the reason compose must not set it."""
        from hermes_cli.config import save_config

        with patch.dict(
            os.environ, {"HERMES_HOME": str(tmp_path), "HERMES_TIMEZONE": "Asia/Tokyo"}
        ):
            save_config({"timezone": "America/Denver"})
            hermes_time.reset_cache()
            assert str(hermes_time.get_timezone()) == "Asia/Tokyo"

    def test_invalid_zone_falls_back_without_raising(self):
        os.environ["HERMES_TIMEZONE"] = "Not/AZone"
        assert hermes_time.get_timezone() is None
        assert hermes_time.now().tzinfo is not None


class TestPublishedTimezone:
    """The gateway publishes its resolved zone for containers that can't read config."""

    def test_reports_configured_zone(self):
        from gateway.platforms.api_server import _resolved_timezone_info

        os.environ["HERMES_TIMEZONE"] = "America/Denver"
        hermes_time.reset_cache()
        info = _resolved_timezone_info()
        assert info["name"] == "America/Denver"
        assert info["source"] == "environment"
        assert info["utc_offset"] in ("-06:00", "-07:00")  # MDT / MST
        assert info["abbreviation"] in ("MDT", "MST")

    def test_reports_scheduler_zone_separately(self):
        """Cron is UTC even when the display zone is not — clients must see both."""
        from gateway.platforms.api_server import _resolved_timezone_info

        os.environ["HERMES_TIMEZONE"] = "America/Denver"
        hermes_time.reset_cache()
        info = _resolved_timezone_info()
        assert info["scheduler"] == "UTC"
        assert info["name"] != info["scheduler"]

    def test_never_raises(self):
        from gateway.platforms.api_server import _resolved_timezone_info

        with patch.object(hermes_time, "now", side_effect=RuntimeError("boom")):
            info = _resolved_timezone_info()
        assert info["name"] == "UTC"
