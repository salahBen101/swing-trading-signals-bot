"""A clock seam.

Everything that asks "what time is it" goes through a `Clock`. Live code gets
`SystemClock`; backtests, replays and tests get `SimulatedClock`, whose time only advances
when something advances it. Without this seam a test for "the position must flatten two
minutes before the close" has to either sleep or monkeypatch `datetime.now`, and a
backtest silently mixes wall-clock time into simulated decisions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("US/Eastern")


class Clock:
    """Interface. `now()` must always return a timezone-aware datetime."""

    def now(self) -> datetime:  # pragma: no cover - interface
        raise NotImplementedError

    def today(self):
        return self.now().date()


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(tz=MARKET_TZ)


class SimulatedClock(Clock):
    """Manually advanced clock.

    `set()` is used by the replay feed to pin simulated time to the bar being processed,
    so every downstream component — risk session rollover, staleness detection, the
    flatten-before-close rule — sees the same instant the bar belongs to.
    """

    def __init__(self, start: datetime) -> None:
        self.set(start)

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        if ts.tzinfo is None:
            raise ValueError("SimulatedClock requires a timezone-aware datetime")
        self._now = ts.astimezone(MARKET_TZ)

    def advance(self, delta: timedelta) -> datetime:
        self._now = self._now + delta
        return self._now


def ensure_aware(ts: datetime, *, tz: ZoneInfo = MARKET_TZ) -> datetime:
    """Boundary coercion for values arriving from outside the system.

    Naive datetimes are rejected rather than assumed to be in `tz`: a broker timestamp
    that quietly turns out to be UTC, read as Eastern, moves every decision by four hours
    and nothing in the output looks wrong.
    """
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime crossed a boundary: {ts!r}")
    return ts.astimezone(tz)
