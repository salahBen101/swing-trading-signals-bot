"""Per-session context, all of it knowable at the RTH open.

This is the layer that makes families A and C expressible. Every column here answers a
question a trader could actually have answered at 09:30 — what did the overnight session
do, where did yesterday settle, how far did we gap — and nothing here may look at the
session it describes.

The causality rule is strict and mechanical: a column describing session *T* may use bars
up to and including 09:29 on *T*, and nothing after. Columns describing session *T-1* are
built from that session's own bars and then shifted forward. `test_research_harness.py`
asserts this by rebuilding the table on truncated data.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from ..core.clock import MARKET_TZ

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def rth_slice(eth_bars: pd.DataFrame) -> pd.DataFrame:
    """The regular-hours bars, with the session label retained."""
    return eth_bars[eth_bars["is_rth"]].copy()


def overnight_slice(eth_bars: pd.DataFrame) -> pd.DataFrame:
    """Bars from the 18:00 open through 09:29 the next morning.

    Excludes the post-close 16:00-17:00 tail of the *previous* session, which belongs to
    that session's settlement rather than to tonight's inventory build.
    """
    local = eth_bars.index.tz_convert(MARKET_TZ)
    minutes = local.hour * 60 + local.minute
    is_evening = minutes >= 18 * 60
    is_morning = minutes < 9 * 60 + 30
    return eth_bars[(is_evening | is_morning) & ~eth_bars["is_rth"]].copy()


def build_session_table(eth_bars: pd.DataFrame) -> pd.DataFrame:
    """One row per session: prior-day structure, overnight structure, and the gap.

    Indexed by session date. Every value is available at 09:30 of that session.
    """
    rth = rth_slice(eth_bars)
    overnight = overnight_slice(eth_bars)

    rth_agg = rth.groupby("session").agg(
        rth_open=("open", "first"),
        rth_high=("high", "max"),
        rth_low=("low", "min"),
        rth_close=("close", "last"),
        rth_volume=("volume", "sum"),
        rth_bars=("close", "size"),
    )

    on_agg = overnight.groupby("session").agg(
        on_open=("open", "first"),
        on_high=("high", "max"),
        on_low=("low", "min"),
        on_close=("close", "last"),
        on_volume=("volume", "sum"),
        on_bars=("close", "size"),
    )

    table = rth_agg.join(on_agg, how="left")

    # --- yesterday, shifted so today can read it ---
    for column in ("rth_open", "rth_high", "rth_low", "rth_close", "rth_volume"):
        table[f"prior_{column[4:]}"] = table[column].shift(1)
    table["prior_range"] = table["prior_high"] - table["prior_low"]

    # --- the overnight move and the gap ---
    table["on_range"] = table["on_high"] - table["on_low"]
    # Drift accumulated overnight, measured from yesterday's settlement.
    table["on_move"] = table["on_close"] - table["prior_close"]
    table["gap"] = table["rth_open"] - table["prior_close"]

    # --- normalisers, trailing and causal ---
    # A 14-session average true range on daily bars, shifted so today is excluded.
    prior_close = table["prior_close"]
    true_range = pd.concat(
        [
            table["rth_high"] - table["rth_low"],
            (table["rth_high"] - prior_close).abs(),
            (table["rth_low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    table["daily_atr"] = true_range.shift(1).rolling(14, min_periods=14).mean()

    table["on_range_atr"] = table["on_range"] / table["daily_atr"]
    table["on_move_atr"] = table["on_move"] / table["daily_atr"]
    table["gap_atr"] = table["gap"] / table["daily_atr"]
    table["prior_range_atr"] = table["prior_range"] / table["daily_atr"]

    # Where the overnight close sits inside the overnight range: near 1.0 means the
    # session ran up into the open and held there, which is the inventory-imbalance case.
    span = table["on_range"].replace(0.0, np.nan)
    table["on_close_position"] = (table["on_close"] - table["on_low"]) / span

    # Trailing volatility percentile for regime conditioning (E1). Rank within the prior
    # 252 sessions only; a whole-sample quantile would rank today against its own future.
    table["vol_percentile"] = (
        table["daily_atr"].rolling(252, min_periods=60).rank(pct=True)
    )

    # Direction of the prior session, for continuation/reversal conditioning.
    table["prior_return"] = table["prior_close"] - table["prior_open"]

    return table


def add_opening_range(
    rth_bars: pd.DataFrame, session_table: pd.DataFrame, minutes: int = 30
) -> pd.DataFrame:
    """Opening-range extremes per session, plus their width relative to trailing history.

    The range is only *usable* after the window closes; that constraint is enforced at
    signal time by the harness, which will not generate an entry inside the window.
    """
    local = rth_bars.index.tz_convert(MARKET_TZ)
    elapsed = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    window = rth_bars[(elapsed >= 0) & (elapsed < minutes)]

    agg = window.groupby("session").agg(
        or_high=("high", "max"),
        or_low=("low", "min"),
        or_volume=("volume", "sum"),
    )
    agg["or_range"] = agg["or_high"] - agg["or_low"]

    out = session_table.join(agg, how="left")
    out["or_range_atr"] = out["or_range"] / out["daily_atr"]
    # Percentile of today's opening range within the trailing 60 sessions, shifted so the
    # comparison set excludes today.
    out["or_range_percentile"] = (
        out["or_range"].shift(1).rolling(60, min_periods=20).rank(pct=True)
    )
    # Today's own rank against that trailing distribution, computed causally.
    trailing_mean = out["or_range"].shift(1).rolling(60, min_periods=20).mean()
    out["or_range_vs_trailing"] = out["or_range"] / trailing_mean
    return out


def minutes_since_open(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(MARKET_TZ)
    return (local.hour * 60 + local.minute).to_numpy() - (9 * 60 + 30)


def attach_session_columns(
    rth_bars: pd.DataFrame, session_table: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Broadcast per-session values onto every bar of that session."""
    out = rth_bars.copy()
    for column in columns:
        out[column] = out["session"].map(session_table[column])
    return out
