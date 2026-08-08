"""Scheduler clock for Hermes cron — deliberately machine time (UTC).

Why this module exists
----------------------
Cron used to read its clock from ``hermes_time.now()``, which follows the
user-configurable ``timezone`` key in ``config.yaml``. That coupling meant a
user changing their display timezone in Settings would silently reinterpret
every existing schedule: ``0 7 * * *`` created as 07:00 UTC would start meaning
07:00 America/Denver, shifting every pending ``next_run_at`` by the offset with
no warning and no migration.

Cron is therefore pinned to **UTC**, independent of the display timezone. The
trade-off is accepted deliberately: schedules are stable and never move under
the user, at the cost of the user having to think in UTC. To make that cost
payable, every surface that shows a cron time must render it through
``format_dual()`` so the UTC instant is stated explicitly *and* translated into
the user's zone for planning.

Consequences worth knowing:

- **No DST hazard.** UTC has no DST transitions, so the wall-clock drift that
  affects ``croniter`` when handed a DST-observing tz-aware base cannot occur
  here. See ``cron.jobs.compute_next_run`` for the details and for what would
  need fixing first if cron ever moves to a DST-observing zone.
- **Naive input is UTC.** A bare ``2026-02-03T14:00:00`` schedule means 14:00
  UTC. This matches the behaviour before the decoupling (the config key was
  empty, so the configured zone already resolved to the UTC host), so nothing
  changes for existing jobs.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

_NO_OFFSET = timedelta(0)

# The zone cron computes in. Not configurable by design — see module docstring.
SCHEDULER_TZ_NAME = "UTC"


def now() -> datetime:
    """Return the current scheduler time: timezone-aware UTC.

    This intentionally does *not* consult ``hermes_time`` / the user's
    configured timezone. Cron must not move when a display preference changes.
    """
    return datetime.now(timezone.utc)


def _display_tz():
    """Return the user's configured display zone, or None to mean 'same as UTC'.

    Fails open: any problem resolving the user's zone degrades to None, which
    makes the formatters emit the plain UTC form rather than raising.
    """
    try:
        from hermes_time import get_timezone

        tz = get_timezone()
    except Exception:
        return None
    if tz is None:
        # No zone configured — hermes_time falls back to server-local. Only
        # worth a parenthetical if the server actually differs from UTC.
        local = datetime.now().astimezone().tzinfo
        if local is None:
            return None
        offset = datetime.now(local).utcoffset()
        return None if offset in (None, _NO_OFFSET) else local
    # Equal offset means the rendered wall clock would be identical, so the
    # parenthetical would add nothing.
    return None if datetime.now(tz).utcoffset() == _NO_OFFSET else tz


def format_utc(dt: datetime, *, seconds: bool = False) -> str:
    """Render ``dt`` as an explicitly-labelled UTC wall clock."""
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    fmt = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return f"{aware.astimezone(timezone.utc).strftime(fmt)} UTC"


def format_dual(dt: datetime, *, seconds: bool = False) -> str:
    """Render ``dt`` as ``UTC (local)`` so a schedule can be read unambiguously.

    Example: ``2026-11-01 14:00 UTC (07:00 MDT)``. The parenthetical is omitted
    when the user's zone is UTC or unset, to avoid the noise of
    ``14:00 UTC (14:00 UTC)``.
    """
    base = format_utc(dt, seconds=seconds)
    tz = _display_tz()
    if tz is None:
        return base
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    local = aware.astimezone(tz)
    fmt = "%H:%M:%S" if seconds else "%H:%M"
    label = local.strftime("%Z") or str(tz)
    # Cross-midnight conversions need the date, or "14:00 UTC (21:00 JST)" hides
    # that the local day is different.
    if local.date() != aware.astimezone(timezone.utc).date():
        fmt = f"%Y-%m-%d {fmt}"
    return f"{base} ({local.strftime(fmt)} {label})"


def format_iso_dual(value, *, fallback: str = "?", seconds: bool = False) -> str:
    """Render a stored ISO timestamp as ``UTC (local)`` for display.

    Tolerant by design: display code must never crash on a missing or malformed
    stored timestamp, so unparseable input is returned as-is.
    """
    if value in (None, "", "?"):
        return fallback
    if isinstance(value, datetime):
        return format_dual(value, seconds=seconds)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value)
    return format_dual(parsed, seconds=seconds)


def describe_schedule_zone() -> str:
    """One-line statement of the zone cron expressions are interpreted in.

    Used in job-creation confirmations so the user is told up front rather than
    discovering it when a job fires at an unexpected hour.
    """
    tz = _display_tz()
    if tz is None:
        return "Cron schedules are interpreted in UTC."
    sample = datetime.now(timezone.utc).astimezone(tz)
    label = sample.strftime("%Z") or str(tz)
    offset = sample.utcoffset()
    if offset is None:
        return "Cron schedules are interpreted in UTC."
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return (
        f"Cron schedules are interpreted in UTC, not your local timezone "
        f"({label} is UTC{sign}{hours:02d}:{minutes:02d})."
    )


def local_zone_name() -> Optional[str]:
    """Return the display zone's name, or None when it matches UTC."""
    tz = _display_tz()
    return None if tz is None else str(tz)
