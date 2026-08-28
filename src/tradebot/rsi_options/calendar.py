"""US equity trading sessions. The foundation of temporal correctness.

Everything in the formal trial that says "next session" resolves through this module. Naive
arithmetic like `date + timedelta(days=1)` is what produces a Saturday execution date, or an
entry booked on Thanksgiving, or a Friday signal that appears to fill before Monday's open. None
of those are visible in a P&L column - they just quietly make the numbers wrong.

The holiday table is explicit and versioned rather than pulled from a library, because a silent
dependency upgrade that shifts a holiday would retroactively change which session a historical
signal executed in. Half-days are listed separately: they are trading sessions, so a signal can
execute in one, but the close is 13:00 ET and anything that assumes 16:00 will mis-stamp them.

Sessions can also be derived from observed market data - if a price series has a bar for a date,
that date was a session. `from_observed` builds a calendar that way, which is the safest option
when the hardcoded table has aged past its last listed year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)

# NYSE/Nasdaq full closures. Extend deliberately; do not compute.
HOLIDAYS: frozenset[date] = frozenset(
    date(*d)
    for d in [
        # 2024
        (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 3, 29), (2024, 5, 27),
        (2024, 6, 19), (2024, 7, 4), (2024, 9, 2), (2024, 11, 28), (2024, 12, 25),
        # 2025
        (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18),
        (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1), (2025, 11, 27),
        (2025, 12, 25),
        # 2026
        (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25),
        (2026, 6, 19), (2026, 7, 3), (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
        # 2027
        (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31),
        (2027, 6, 18), (2027, 7, 5), (2027, 9, 6), (2027, 11, 25), (2027, 12, 24),
    ]
)

# Sessions that close at 13:00 ET. These ARE trading days.
HALF_DAYS: frozenset[date] = frozenset(
    date(*d)
    for d in [
        (2024, 7, 3), (2024, 11, 29), (2024, 12, 24),
        (2025, 7, 3), (2025, 11, 28), (2025, 12, 24),
        (2026, 11, 27), (2026, 12, 24),
        (2027, 11, 26),
    ]
)

LAST_CALENDAR_YEAR = 2027


class CalendarExpired(RuntimeError):
    """Raised when a date falls past the last year the holiday table covers.

    Fail closed. Guessing that a date past the table is a normal weekday would silently book
    trades on holidays the moment the table ages out.
    """


@dataclass(frozen=True, slots=True)
class Session:
    day: date
    is_half_day: bool

    @property
    def open_dt(self) -> datetime:
        return datetime.combine(self.day, REGULAR_OPEN, tzinfo=MARKET_TZ)

    @property
    def close_dt(self) -> datetime:
        return datetime.combine(
            self.day, HALF_DAY_CLOSE if self.is_half_day else REGULAR_CLOSE, tzinfo=MARKET_TZ)


class TradingCalendar:
    """Sessions from the explicit table, or from observed market data."""

    def __init__(self, observed: frozenset[date] | None = None) -> None:
        self._observed = observed

    @classmethod
    def from_observed(cls, days) -> TradingCalendar:
        """Build from dates that actually produced bars. A bar existed, so it was a session."""
        out = set()
        for d in days:
            if hasattr(d, "date"):
                d = d.date()
            out.add(d)
        return cls(observed=frozenset(out))

    def is_session(self, d: date | datetime) -> bool:
        if isinstance(d, datetime):
            d = d.date()
        if self._observed is not None:
            return d in self._observed
        if d.year > LAST_CALENDAR_YEAR:
            raise CalendarExpired(
                f"{d} is past the last year in the holiday table ({LAST_CALENDAR_YEAR}). "
                "Extend HOLIDAYS/HALF_DAYS or construct the calendar with from_observed()."
            )
        return d.weekday() < 5 and d not in HOLIDAYS

    def session(self, d: date | datetime) -> Session | None:
        if isinstance(d, datetime):
            d = d.date()
        if not self.is_session(d):
            return None
        return Session(day=d, is_half_day=d in HALF_DAYS)

    def next_session(self, d: date | datetime, offset: int = 1) -> date:
        """The Nth trading session strictly after `d`.

        This is the function that makes a Friday-close signal execute on Monday, and a
        Thursday-before-Thanksgiving signal execute on Friday's half day rather than on the
        holiday itself.
        """
        if offset < 1:
            raise ValueError("offset must be >= 1; offset 0 would be same-session execution")
        if isinstance(d, datetime):
            d = d.date()
        cur, found = d, 0
        for _ in range(offset * 12 + 30):        # generous bound; long holiday runs exist
            cur += timedelta(days=1)
            if self.is_session(cur):
                found += 1
                if found == offset:
                    return cur
        raise CalendarExpired(f"no trading session found within a reasonable window after {d}")

    def prev_session(self, d: date | datetime, offset: int = 1) -> date:
        if offset < 1:
            raise ValueError("offset must be >= 1")
        if isinstance(d, datetime):
            d = d.date()
        cur, found = d, 0
        for _ in range(offset * 12 + 30):
            cur -= timedelta(days=1)
            if self.is_session(cur):
                found += 1
                if found == offset:
                    return cur
        raise CalendarExpired(f"no trading session found within a reasonable window before {d}")

    def sessions_between(self, start: date | datetime, end: date | datetime) -> int:
        """Sessions strictly after `start`, up to and including `end`."""
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()
        if end <= start:
            return 0
        n, cur = 0, start
        while cur < end:
            cur += timedelta(days=1)
            if self.is_session(cur):
                n += 1
        return n

    def is_after_close(self, ts: datetime, d: date | None = None) -> bool:
        """Has the session for `d` finished as of `ts`?

        A daily signal is only knowable after its session closes. Anything that reads a partial
        bar and calls it a close is reading a number that will still change.
        """
        ts = ts.astimezone(MARKET_TZ)
        sess = self.session(d or ts.date())
        if sess is None:
            return True                      # not a session: nothing left to wait for
        return ts >= sess.close_dt
