"""Guards on the dynamic-exit and volume-profile research code.

Two properties carry the weight here:

* the dynamic exit does not read the future (poison test), and its trailing stop only ever
  tightens; and
* the **null property** — on random-walk prices, no exit rule produces positive net
  expectancy. This is the theoretical fact the whole dynamic-exit study rests on, so it is
  asserted directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.research.dynamic_exits import (
    DynamicExitRule,
    null_control,
    screen_dynamic,
    simulate_dynamic,
)
from tradebot.research.harness import ROUND_TRIP_POINTS
from tradebot.research.volume_profile import add_value_area, value_area_confirms


def _session_frame(rows: list[dict], session: str = "2021-03-01") -> pd.DataFrame:
    index = pd.date_range(f"{session} 09:30", periods=len(rows), freq="5min", tz=MARKET_TZ)
    frame = pd.DataFrame(rows, index=index)
    frame["session"] = pd.Timestamp(session, tz=MARKET_TZ)
    return frame


def _random_walk(n_sessions: int = 60, bars: int = 78, seed: int = 4) -> pd.DataFrame:
    """A pure random walk: by construction there is no edge to find."""
    rng = np.random.default_rng(seed)
    frames = []
    price = 18_000.0
    for d in range(n_sessions):
        day = pd.Timestamp("2021-03-01", tz=MARKET_TZ) + pd.Timedelta(days=d)
        index = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=bars,
                              freq="5min", tz=MARKET_TZ)
        closes = price + np.cumsum(rng.normal(0, 4.0, bars))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        highs = np.maximum(opens, closes) + rng.uniform(1, 6, bars)
        lows = np.minimum(opens, closes) - rng.uniform(1, 6, bars)
        frames.append(pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": rng.integers(100, 900, bars).astype(float), "session": day},
            index=index,
        ))
        price = float(closes[-1])
    out = pd.concat(frames).sort_index()
    out.index.name = "timestamp"
    return out


# --------------------------------------------------------------------------- fill rules


def test_the_trailing_stop_only_ever_tightens():
    # Long from 100; price runs to 120 then falls back. The stop must ratchet up and never
    # widen, so the exit is well above the initial stop.
    closes = [100, 104, 110, 118, 120, 108, 100]
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c} for c in closes]
    bars = _session_frame(rows)
    atr = np.full(len(bars), 4.0)
    direction = np.array([1, 0, 0, 0, 0, 0, 0], dtype="int8")

    rule = DynamicExitRule(initial_stop_atr=1.0, breakeven_at_atr=1.0, trail_atr=2.0,
                           early_cut_bars=99, max_bars=99, no_entry_within_bars=0)
    trades = simulate_dynamic(bars, direction, atr, rule)
    assert len(trades) == 1
    # Entered at 104 (next open). Best ~121; trail 2 ATR = 8 behind -> exit well above entry.
    assert trades[0].exit_price > trades[0].entry_price
    assert trades[0].exit_kind == "trail_stop"


def test_a_gap_through_the_trailed_stop_fills_at_the_open():
    closes = [100, 104, 112, 90]  # last bar gaps far below the trailed stop
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c} for c in closes]
    bars = _session_frame(rows)
    bars.iloc[3, bars.columns.get_loc("open")] = 92.0
    atr = np.full(len(bars), 4.0)
    direction = np.array([1, 0, 0, 0], dtype="int8")

    rule = DynamicExitRule(breakeven_at_atr=1.0, trail_atr=1.0, early_cut_bars=99,
                           max_bars=99, no_entry_within_bars=0)
    trades = simulate_dynamic(bars, direction, atr, rule)
    assert trades[0].exit_price == 92.0, "the gap is filled honestly, not at the stop level"


def test_the_early_cut_closes_a_trade_that_is_not_working():
    # Flat-to-drifting price that never gets 1 ATR in front: the early cut must fire.
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c} for c in closes]
    bars = _session_frame(rows)
    atr = np.full(len(bars), 5.0)
    direction = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="int8")

    rule = DynamicExitRule(early_cut_bars=3, early_cut_min_progress_atr=0.25,
                           breakeven_at_atr=1.0, max_bars=99, no_entry_within_bars=0)
    trades = simulate_dynamic(bars, direction, atr, rule)
    assert trades[0].exit_kind == "early_cut"
    # Entered at index 1; the cut fires on the third bar held (index 3), so the index span
    # exit - entry is 2.
    assert trades[0].bars_held == 2
    assert trades[0].exit_index == 3


def test_a_position_never_crosses_a_session_boundary():
    index = pd.date_range("2021-03-01 15:40", periods=4, freq="5min", tz=MARKET_TZ)
    bars = pd.DataFrame(
        {"open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4},
        index=index,
    )
    bars["session"] = [pd.Timestamp("2021-03-01", tz=MARKET_TZ)] * 2 + [
        pd.Timestamp("2021-03-02", tz=MARKET_TZ)
    ] * 2
    atr = np.full(4, 5.0)
    rule = DynamicExitRule(early_cut_bars=99, max_bars=99, no_entry_within_bars=0)
    trades = simulate_dynamic(bars, np.array([1, 0, 0, 0], dtype="int8"), atr, rule)
    assert trades[0].exit_index == 1


# --------------------------------------------------------------------------- causality


def test_the_dynamic_exit_does_not_read_the_future():
    bars = _random_walk()
    atr = bars["close"].rolling(14).std().bfill().to_numpy() + 1.0
    rng = np.random.default_rng(9)
    signal = rng.choice([-1, 0, 0, 1], size=len(bars)).astype("int8")
    rule = DynamicExitRule()

    cutoff = int(len(bars) * 0.6)
    poisoned = bars.copy()
    poisoned.iloc[cutoff:, [0, 1, 2, 3]] = 99_999.0
    poison_atr = atr.copy()

    clean = simulate_dynamic(bars, signal, atr, rule)
    dirty = simulate_dynamic(poisoned, signal, poison_atr, rule)

    clean_early = [t for t in clean if t.exit_index < cutoff]
    dirty_early = [t for t in dirty if t.exit_index < cutoff]
    assert len(clean_early) == len(dirty_early)
    for a, b in zip(clean_early, dirty_early, strict=True):
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_price == pytest.approx(b.exit_price)


# --------------------------------------------------------------------------- the null property


def test_on_a_random_walk_the_dynamic_exit_cannot_manufacture_an_edge():
    """The theoretical fact the whole study rests on.

    For random-walk prices no exit rule produces positive net expectancy, because the exit
    reshapes the distribution of wins and losses but not its mean, and costs then bite. The
    null-control mean must sit below zero.
    """
    bars = _random_walk(n_sessions=120, seed=11)
    atr = bars["close"].rolling(14).std().bfill().to_numpy() + 2.0
    result = null_control(bars, atr, DynamicExitRule(), n_trials=40, seed=3, bar_minutes=5)

    assert result["trials"] >= 30
    assert result["mean_net"] < 0, (
        f"a dynamic exit averaged {result['mean_net']:+.3f} net on random entries; if this "
        f"is positive the null control is broken and no dynamic-exit result can be trusted"
    )
    # Some individual trials clear zero by luck — that is the whole point of measuring the
    # 95th percentile as the bar a real entry must beat.
    assert result["p95"] > result["mean_net"]


def test_the_null_control_yardstick_is_actually_demanding():
    """The 95th percentile must be well above the mean, or it is not a real bar."""
    bars = _random_walk(n_sessions=120, seed=7)
    atr = bars["close"].rolling(14).std().bfill().to_numpy() + 2.0
    result = null_control(bars, atr, DynamicExitRule(), n_trials=40, seed=5, bar_minutes=5)
    assert result["p95"] - result["mean_net"] > 0.5


# --------------------------------------------------------------------------- volume profile


def test_the_developing_volume_profile_only_uses_past_bars():
    bars = _random_walk(n_sessions=20, seed=2)
    full = add_value_area(bars)

    cutoff = 40
    truncated = add_value_area(bars.iloc[:cutoff].copy())
    # The developing POC at each early bar must not change when later bars are added.
    pd.testing.assert_series_equal(
        truncated["vp_poc"].iloc[:cutoff].reset_index(drop=True),
        full["vp_poc"].iloc[:cutoff].reset_index(drop=True),
        check_names=False,
    )


def test_the_point_of_control_sits_inside_the_value_area():
    bars = _random_walk(n_sessions=10, seed=6)
    vp = add_value_area(bars).dropna(subset=["vp_poc", "vp_val", "vp_vah"])
    assert (vp["vp_val"] <= vp["vp_poc"] + 1e-9).all()
    assert (vp["vp_poc"] <= vp["vp_vah"] + 1e-9).all()


def test_value_area_confirmation_only_removes_signals_never_adds():
    bars = _random_walk(n_sessions=20, seed=8)
    vp = add_value_area(bars)
    rng = np.random.default_rng(1)
    signal = rng.choice([-1, 0, 1], size=len(bars)).astype("int8")
    confirmed = value_area_confirms(vp, signal)

    # Every confirmed signal was a signal, in the same direction; some are dropped.
    both = (confirmed != 0)
    assert (confirmed[both] == signal[both]).all()
    assert (confirmed != 0).sum() <= (signal != 0).sum()


def test_confirmation_requires_price_outside_the_value_area():
    vp = pd.DataFrame({
        "vp_position": np.array([1, -1, 0], dtype="int8"),
        "session": [pd.Timestamp("2021-03-01", tz=MARKET_TZ)] * 3,
    })
    signal = np.array([1, 1, 1], dtype="int8")  # all long
    confirmed = value_area_confirms(vp, signal)
    # Long kept only where price is above the value area (position +1).
    assert confirmed.tolist() == [1, 0, 0]


def test_costs_are_still_charged_under_the_dynamic_exit():
    bars = _random_walk(n_sessions=10, seed=3)
    atr = bars["close"].rolling(14).std().bfill().to_numpy() + 2.0
    signal = np.zeros(len(bars), dtype="int8")
    signal[10:200:20] = 1
    result = screen_dynamic(signal, bars, atr, DynamicExitRule(), hypothesis_id="t",
                            bar_minutes=5)
    if result.trades:
        assert result.net_points_per_trade == pytest.approx(
            result.gross_points_per_trade - ROUND_TRIP_POINTS
        )
