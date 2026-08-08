"""Cron runs on machine time (UTC), independent of the user's display timezone.

This is the guard for the decoupling described in cron/clock.py. Before it,
cron read its clock from ``hermes_time.now()``, which follows the user-editable
``timezone`` key in config.yaml — so changing a *display preference* silently
reinterpreted every stored schedule and shifted every pending ``next_run_at``
by the offset, with no migration and no warning.

If someone re-points the scheduler at ``hermes_time`` these tests fail.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

import hermes_time
from cron import clock


def _reset_hermes_time_cache():
    hermes_time._cached_tz = None
    hermes_time._cached_tz_name = None
    hermes_time._cache_resolved = False


@pytest.fixture
def denver_display_zone():
    """Configure a DST-observing display timezone six-ish hours from UTC."""
    previous = os.environ.get("HERMES_TIMEZONE")
    os.environ["HERMES_TIMEZONE"] = "America/Denver"
    _reset_hermes_time_cache()
    yield
    if previous is None:
        os.environ.pop("HERMES_TIMEZONE", None)
    else:
        os.environ["HERMES_TIMEZONE"] = previous
    _reset_hermes_time_cache()


class TestSchedulerClockIsUTC:
    def test_now_is_utc_regardless_of_display_zone(self, denver_display_zone):
        assert clock.now().utcoffset() == timedelta(0)

    def test_now_ignores_configured_zone(self, denver_display_zone):
        """The whole point: the display zone must not move the scheduler."""
        assert hermes_time.now().utcoffset() != timedelta(0), (
            "precondition: the display zone should differ from UTC"
        )
        assert clock.now().utcoffset() == timedelta(0)

    def test_scheduler_tz_name_is_utc(self):
        assert clock.SCHEDULER_TZ_NAME == "UTC"


class TestComputeNextRunIgnoresDisplayZone:
    def test_daily_cron_stays_on_utc_hour(self, denver_display_zone):
        """`0 7 * * *` means 07:00 UTC, not 07:00 America/Denver."""
        from cron.jobs import compute_next_run

        last = datetime(2026, 6, 10, 7, 0, tzinfo=timezone.utc).isoformat()
        result = compute_next_run({"kind": "cron", "expr": "0 7 * * *"}, last_run_at=last)

        next_run = datetime.fromisoformat(result)
        assert next_run.astimezone(timezone.utc).hour == 7
        assert next_run.utcoffset() == timedelta(0)

    def test_result_identical_with_and_without_display_zone(self):
        """Setting a display zone must not change any computed instant."""
        from cron.jobs import compute_next_run

        schedule = {"kind": "cron", "expr": "0 7 * * *"}
        last = datetime(2026, 6, 10, 7, 0, tzinfo=timezone.utc).isoformat()

        os.environ.pop("HERMES_TIMEZONE", None)
        _reset_hermes_time_cache()
        without = compute_next_run(schedule, last_run_at=last)

        os.environ["HERMES_TIMEZONE"] = "America/Denver"
        _reset_hermes_time_cache()
        try:
            with_zone = compute_next_run(schedule, last_run_at=last)
        finally:
            os.environ.pop("HERMES_TIMEZONE", None)
            _reset_hermes_time_cache()

        assert datetime.fromisoformat(without) == datetime.fromisoformat(with_zone)

    @pytest.mark.parametrize(
        "label,last_run",
        [
            # The two US transitions. Pinned to UTC these are ordinary days;
            # the assertion is that they stay ordinary.
            ("fall-back", datetime(2026, 10, 31, 7, 0, tzinfo=timezone.utc)),
            ("spring-forward", datetime(2026, 3, 7, 7, 0, tzinfo=timezone.utc)),
        ],
    )
    def test_no_dst_drift_across_transitions(self, denver_display_zone, label, last_run):
        """A daily job advances exactly 24h across a display-zone DST boundary.

        With a DST-observing base, croniter's absolute-instant arithmetic would
        land an hour off here (08:00 after fall-back, 06:00 after spring-
        forward). UTC has no transitions, so this must be exact.
        """
        from cron.jobs import compute_next_run

        result = compute_next_run(
            {"kind": "cron", "expr": "0 7 * * *"}, last_run_at=last_run.isoformat()
        )
        next_run = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert next_run == last_run + timedelta(days=1), f"{label} drifted"


class TestScheduleCarriesTimezone:
    @pytest.mark.parametrize(
        "schedule", ["0 7 * * *", "every 30m", "30m", "2026-02-03T14:00:00"]
    )
    def test_parse_schedule_stamps_timezone(self, schedule):
        from cron.jobs import parse_schedule

        assert parse_schedule(schedule)["timezone"] == "UTC"

    def test_naive_iso_is_interpreted_as_utc(self, denver_display_zone):
        from cron.jobs import parse_schedule

        parsed = parse_schedule("2026-02-03T14:00:00")
        run_at = datetime.fromisoformat(parsed["run_at"])
        assert run_at.utcoffset() == timedelta(0)
        assert run_at.astimezone(timezone.utc).hour == 14


class TestDualFormatting:
    def test_labels_utc_and_local(self, denver_display_zone):
        # 2026-11-01 14:00Z is 07:00 MST (post fall-back).
        dt = datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc)
        rendered = clock.format_dual(dt)
        assert rendered.startswith("2026-11-01 14:00 UTC")
        assert "07:00" in rendered
        assert "MST" in rendered

    def test_omits_parenthetical_when_display_zone_is_utc(self):
        os.environ.pop("HERMES_TIMEZONE", None)
        _reset_hermes_time_cache()
        dt = datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc)
        assert clock.format_dual(dt) == "2026-11-01 14:00 UTC"

    def test_includes_date_when_local_day_differs(self):
        """A cross-midnight conversion must not hide the day change."""
        previous = os.environ.get("HERMES_TIMEZONE")
        os.environ["HERMES_TIMEZONE"] = "Asia/Tokyo"  # UTC+9, no DST
        _reset_hermes_time_cache()
        try:
            dt = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)  # 05:00 next day
            rendered = clock.format_dual(dt)
            assert "2026-11-02" in rendered.split("(", 1)[1]
        finally:
            if previous is None:
                os.environ.pop("HERMES_TIMEZONE", None)
            else:
                os.environ["HERMES_TIMEZONE"] = previous
            _reset_hermes_time_cache()

    def test_naive_input_treated_as_utc(self):
        os.environ.pop("HERMES_TIMEZONE", None)
        _reset_hermes_time_cache()
        assert clock.format_utc(datetime(2026, 11, 1, 14, 0)) == "2026-11-01 14:00 UTC"

    def test_describe_schedule_zone_mentions_utc(self, denver_display_zone):
        described = clock.describe_schedule_zone()
        assert "UTC" in described
        # Denver is UTC-6 (MDT) or UTC-7 (MST) depending on when tests run.
        assert "UTC-0" in described
