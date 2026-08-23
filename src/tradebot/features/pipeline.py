"""Assemble the feature frame a strategy reads.

One frame, computed once per backtest, aligned to the bar index. Strategies index into it
positionally — they never recompute an indicator, and they never see a raw price series
they could accidentally look ahead in.

`warmup_bars` is the number of leading bars where at least one feature is still NaN. The
engine refuses to trade before it, so a strategy cannot be handed a half-formed ATR and
size a position off it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators as ind
from . import levels as lv


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Which indicators to compute, and with what parameters.

    Strategies declare their own spec, so a run only pays for what it uses and the report
    can state exactly which indicator parameters produced a result.
    """

    atr_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    donchian_window: int = 20
    adx_period: int = 14
    efficiency_window: int = 40
    realized_vol_window: int = 50
    volume_window: int = 50
    vwap_band_std: float = 1.0
    opening_range_minutes: int = 30
    pivot_left: int = 3
    pivot_right: int = 3
    pivot_count: int = 4
    atr_percentile_window: int = 390  # ~one RTH session of 1-minute bars
    session_start: str = "09:30"
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """The computed features plus the warmup boundary."""

    frame: pd.DataFrame
    spec: FeatureSpec
    warmup_bars: int

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)


def build_features(bars: pd.DataFrame, spec: FeatureSpec | None = None) -> FeatureFrame:
    spec = spec or FeatureSpec()
    o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=bars.index)

    # --- price context ---
    f["close"] = c
    f["open"] = o
    f["high"] = h
    f["low"] = l
    f["volume"] = v
    f["bar_range"] = h - l
    f["prev_close"] = c.shift(1)

    # --- volatility ---
    f["atr"] = ind.atr(h, l, c, spec.atr_period)
    f["atr_pct"] = f["atr"] / c
    f["atr_percentile"] = ind.rolling_percentile(f["atr"], spec.atr_percentile_window)
    f["realized_vol"] = ind.realized_volatility(c, spec.realized_vol_window)

    # --- trend / momentum ---
    f["ema_fast"] = ind.ema(c, spec.ema_fast)
    f["ema_slow"] = ind.ema(c, spec.ema_slow)
    f["ema_trend"] = ind.ema(c, spec.ema_trend)
    f["rsi"] = ind.rsi(c, spec.rsi_period)
    adx_val, plus_di, minus_di = ind.adx(h, l, c, spec.adx_period)
    f["adx"] = adx_val
    f["plus_di"] = plus_di
    f["minus_di"] = minus_di
    f["efficiency_ratio"] = ind.efficiency_ratio(c, spec.efficiency_window)

    # --- bands and channels ---
    bb_lo, bb_mid, bb_hi = ind.bollinger(c, spec.bollinger_window, spec.bollinger_std)
    f["bb_lower"], f["bb_mid"], f["bb_upper"] = bb_lo, bb_mid, bb_hi
    dc_lo, dc_hi = ind.donchian(h, l, spec.donchian_window)
    f["donchian_lower"], f["donchian_upper"] = dc_lo, dc_hi

    # --- session anchored ---
    vwap_lo, vwap, vwap_hi = ind.session_vwap_bands(h, l, c, v, spec.vwap_band_std)
    f["vwap"] = vwap
    f["vwap_lower"] = vwap_lo
    f["vwap_upper"] = vwap_hi
    # Distance from VWAP in ATR units, so a "stretched" test means the same thing in a
    # quiet session and a violent one.
    f["vwap_distance_atr"] = (c - vwap) / f["atr"].replace(0.0, pd.NA)
    f["session_high"] = ind.session_cumulative(h, "max")
    f["session_low"] = ind.session_cumulative(l, "min")
    f["session_open"] = ind.session_cumulative(o, "first")
    f["bars_into_session"] = ind.bars_since_session_start(bars.index)
    f["prior_close"] = ind.prior_session_value(c, "last")
    f["prior_high"] = ind.prior_session_value(h, "max")
    f["prior_low"] = ind.prior_session_value(l, "min")

    or_lo, or_hi, or_done = ind.opening_range(h, l, spec.opening_range_minutes, spec.session_start)
    f["or_low"], f["or_high"], f["or_complete"] = or_lo, or_hi, or_done
    f["or_width"] = or_hi - or_lo
    # The opening range's width relative to its own recent history: the "quiet open"
    # condition that earlier research found to be the only ORB variant with a DEV-sample
    # signal (and which then failed the year-by-year test - see PROJECT_SPEC section 12).
    f["or_width_percentile"] = ind.rolling_percentile(f["or_width"], 60, min_periods=10)

    # --- volume ---
    f["volume_ratio"] = ind.volume_ratio(v, spec.volume_window)

    # --- structure ---
    highs = lv.recent_pivot_levels(h, spec.pivot_left, spec.pivot_right, "high", spec.pivot_count)
    lows = lv.recent_pivot_levels(l, spec.pivot_left, spec.pivot_right, "low", spec.pivot_count)
    f["swing_high"] = highs["level_0"]
    f["swing_low"] = lows["level_0"]
    f["resistance"] = lv.nearest_level_above(c, highs)
    f["support"] = lv.nearest_level_below(c, lows)
    f["resistance_distance_atr"] = (f["resistance"] - c) / f["atr"].replace(0.0, pd.NA)
    f["support_distance_atr"] = (c - f["support"]) / f["atr"].replace(0.0, pd.NA)

    warmup = _warmup_bars(f)
    return FeatureFrame(frame=f, spec=spec, warmup_bars=warmup)


# Columns that are legitimately NaN for long stretches and must not drive the warmup
# boundary. `support`/`resistance` are NaN whenever no confirmed pivot sits on that side of
# price, which can happen at any point in a session; `or_*` are NaN until the opening range
# closes, every session, by design. Letting either set the warmup would push the boundary
# to the end of the sample and silently disable trading.
_SPARSE_BY_DESIGN = frozenset(
    {
        "support", "resistance", "support_distance_atr", "resistance_distance_atr",
        "swing_high", "swing_low",
        "or_low", "or_high", "or_width", "or_width_percentile", "or_complete",
        "prior_close", "prior_high", "prior_low",
    }
)


def _warmup_bars(f: pd.DataFrame) -> int:
    """First index at which every dense feature is defined."""
    dense = [c for c in f.columns if c not in _SPARSE_BY_DESIGN]
    valid = f[dense].notna().all(axis=1)
    if not valid.any():
        return len(f)
    return int(valid.to_numpy().argmax())
