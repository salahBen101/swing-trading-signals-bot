"""Session windows and the no-overnight rule.

Separated from the limit engine because "may I trade at this time" is a different question
from "may I trade given my P&L", and mixing them makes both harder to test.

The asymmetry here is deliberate and important: **entering** is gated tightly, while
**exiting** is never blocked by a time window. A rule that could refuse to flatten a
position at 15:59 would be a rule that creates overnight exposure, which PROJECT_SPEC §2
forbids outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pandas as pd

from ..config import SessionConfig
from ..core.clock import MARKET_TZ
from ..core.types import RejectReason


@dataclass(frozen=True, slots=True)
class SessionWindow:
    rth_start: time
    rth_end: time
    entry_open: time
    entry_close: time
    flatten_at: time


def _shift(base: time, minutes: int) -> time:
    anchor = datetime(2000, 1, 1, base.hour, base.minute, base.second)
    return (anchor + timedelta(minutes=minutes)).time()


class SessionGuard:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.window = SessionWindow(
            rth_start=config.start_time,
            rth_end=config.end_time,
            entry_open=_shift(config.start_time, config.entry_open_buffer_minutes),
            entry_close=_shift(config.end_time, -config.entry_close_buffer_minutes),
            flatten_at=_shift(config.end_time, -config.flatten_before_close_minutes),
        )
        self._blackouts = self._parse_blackouts(config.news_blackout_windows)

    @staticmethod
    def _parse_blackouts(raw) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        out = []
        for entry in raw or ():
            try:
                start = pd.Timestamp(entry["start"]).tz_convert(MARKET_TZ)
                end = pd.Timestamp(entry["end"]).tz_convert(MARKET_TZ)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"news_blackout_windows entry {entry!r} needs tz-aware 'start' and "
                    f"'end' ISO timestamps"
                ) from exc
            if end <= start:
                raise ValueError(f"news blackout {entry!r} ends before it starts")
            out.append((start, end))
        return out

    # -- entry ---------------------------------------------------------------------------
    def may_enter(self, ts: datetime) -> tuple[bool, RejectReason | None, str]:
        t = ts.astimezone(MARKET_TZ).time()
        w = self.window

        if not (w.rth_start <= t < w.rth_end):
            return False, RejectReason.OUTSIDE_TRADING_HOURS, f"{t} is outside RTH"
        if t < w.entry_open:
            return (
                False, RejectReason.OUTSIDE_TRADING_HOURS,
                f"{t} is inside the {self.config.entry_open_buffer_minutes}-minute "
                f"opening buffer (entries start at {w.entry_open})",
            )
        if t >= w.entry_close:
            return (
                False, RejectReason.OUTSIDE_TRADING_HOURS,
                f"{t} is inside the {self.config.entry_close_buffer_minutes}-minute "
                f"closing buffer; a trade opened now cannot reach its target before the "
                f"forced flatten at {w.flatten_at}",
            )

        blackout = self.in_blackout(ts)
        if blackout is not None:
            return (
                False, RejectReason.NEWS_BLACKOUT,
                f"{ts.isoformat()} falls in the news blackout {blackout[0]} to {blackout[1]}",
            )
        return True, None, ""

    def in_blackout(self, ts: datetime) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        moment = pd.Timestamp(ts).tz_convert(MARKET_TZ)
        for start, end in self._blackouts:
            if start <= moment < end:
                return (start, end)
        return None

    # -- exit ----------------------------------------------------------------------------
    def must_flatten(self, ts: datetime) -> bool:
        """True once a position must be closed. Never blocked by anything."""
        return ts.astimezone(MARKET_TZ).time() >= self.window.flatten_at

    def is_rth(self, ts: datetime) -> bool:
        t = ts.astimezone(MARKET_TZ).time()
        return self.window.rth_start <= t < self.window.rth_end

    def describe(self) -> str:
        w = self.window
        return (
            f"RTH {w.rth_start}-{w.rth_end}, entries {w.entry_open}-{w.entry_close}, "
            f"flat by {w.flatten_at}, {len(self._blackouts)} blackout window(s)"
        )
