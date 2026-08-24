"""Assemble the research frame: RTH bars plus every column the hypotheses read.

One function, called once per timeframe, so that every hypothesis sees exactly the same
inputs and a difference between two results is a difference between the *rules* rather
than between two slightly different feature builds.

All intraday columns come from `tradebot.features`, which is already covered by
prefix-equality tests. The session columns come from `sessions.py`. This module only joins
them and enforces the warmup boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.clock import MARKET_TZ
from ..data.store import resample
from ..features import indicators as ind
from .sessions import add_opening_range, build_session_table, minutes_since_open, rth_slice

# Session-level columns broadcast onto every bar.
SESSION_COLUMNS = [
    "rth_open", "prior_high", "prior_low", "prior_close", "prior_range_atr",
    "on_high", "on_low", "on_move_atr", "on_range_atr", "on_close_position",
    "gap_atr", "daily_atr", "vol_percentile",
    "or_high", "or_low", "or_range", "or_range_atr", "or_range_vs_trailing",
]


@dataclass(frozen=True, slots=True)
class ResearchFrame:
    bars: pd.DataFrame
    atr: np.ndarray
    bar_minutes: float
    warmup_sessions: int

    def __len__(self) -> int:
        return len(self.bars)

    def describe(self) -> str:
        sessions = self.bars["session"].nunique()
        return (
            f"{len(self.bars):,} bars over {sessions:,} sessions, "
            f"{self.bars.index[0].date()} -> {self.bars.index[-1].date()}, "
            f"{self.bar_minutes:g}-minute"
        )


def build_research_frame(
    eth_bars: pd.DataFrame,
    *,
    timeframe: str = "5min",
    opening_range_minutes: int = 30,
    atr_period: int = 14,
) -> ResearchFrame:
    """RTH bars at `timeframe`, with intraday and session features attached."""
    bar_minutes = float(pd.Timedelta(timeframe).total_seconds() / 60)

    rth = rth_slice(eth_bars)
    session_table = build_session_table(eth_bars)
    session_table = add_opening_range(rth, session_table, opening_range_minutes)

    if timeframe == "1min":
        bars = rth.drop(columns=["is_rth"]).copy()
    else:
        # Resample inside each session so a bucket never spans the overnight gap.
        parts = []
        for _, group in rth.groupby("session", sort=True):
            block = resample(group[["open", "high", "low", "close", "volume"]], timeframe)
            block["session"] = group["session"].iloc[0]
            parts.append(block)
        bars = pd.concat(parts).sort_index()

    bars.index = pd.DatetimeIndex(bars.index).tz_convert(MARKET_TZ)

    high, low, close, volume = (bars[c] for c in ("high", "low", "close", "volume"))
    bars["atr"] = ind.atr(high, low, close, atr_period)
    bars["rsi"] = ind.rsi(close, 14)
    bars["ema_fast"] = ind.ema(close, 9)
    bars["ema_slow"] = ind.ema(close, 21)
    bars["ema_trend"] = ind.ema(close, 50)
    adx, _, _ = ind.adx(high, low, close, 14)
    bars["adx"] = adx
    bars["efficiency_ratio"] = ind.efficiency_ratio(close, 20)
    bars["vwap"] = ind.session_vwap(high, low, close, volume)
    bars["session_high"] = ind.session_cumulative(high, "max")
    bars["session_low"] = ind.session_cumulative(low, "min")
    bars["volume_ratio"] = ind.volume_ratio(volume, 50)
    bars["minutes_since_open"] = minutes_since_open(bars.index)

    for column in SESSION_COLUMNS:
        bars[column] = bars["session"].map(session_table[column])

    # The first sessions cannot have a 14-session daily ATR or a 60-session opening-range
    # baseline. Dropping them here means no hypothesis has to guard for it.
    usable = bars["daily_atr"].notna() & bars["atr"].notna()
    first_usable = int(usable.to_numpy().argmax()) if usable.any() else len(bars)
    warmup_sessions = int(bars["session"].iloc[:first_usable].nunique())
    bars = bars.iloc[first_usable:].copy()

    return ResearchFrame(
        bars=bars,
        atr=bars["atr"].to_numpy(dtype="float64"),
        bar_minutes=bar_minutes,
        warmup_sessions=warmup_sessions,
    )


def split_frame(frame: ResearchFrame, split: str) -> ResearchFrame:
    """Slice a research frame to DEV / VALIDATION / HOLDOUT.

    Holdout access is deliberately awkward: it must be named in full, and the caller has
    to have decided to spend it.
    """
    sessions = pd.DatetimeIndex(frame.bars["session"])
    validation_start = pd.Timestamp("2023-01-01", tz=MARKET_TZ)
    holdout_start = pd.Timestamp("2025-01-01", tz=MARKET_TZ)

    key = split.strip().lower()
    if key == "dev":
        mask = sessions < validation_start
    elif key == "validation":
        mask = (sessions >= validation_start) & (sessions < holdout_start)
    elif key == "train":
        mask = sessions < holdout_start
    elif key == "holdout":
        mask = sessions >= holdout_start
    elif key == "all":
        mask = np.ones(len(sessions), dtype=bool)
    else:
        raise ValueError(f"unknown split {split!r}")

    bars = frame.bars[mask]
    return ResearchFrame(
        bars=bars,
        atr=bars["atr"].to_numpy(dtype="float64"),
        bar_minutes=frame.bar_minutes,
        warmup_sessions=frame.warmup_sessions,
    )
