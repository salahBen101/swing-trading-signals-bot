"""Causal daily volatility-compression breakout features.

This module detects a *volatility squeeze*: Bollinger Bands contract inside Keltner
Channels, then price confirms a bullish breakout.  It is deliberately not called a
"short squeeze" detector.  OHLCV bars do not contain timely short interest, borrow
availability, positioning, or options-flow data, so they cannot establish that thesis.

All conditions at ``t`` use only the completed daily bar at ``t`` or earlier.  A scan
can therefore report a confirmed setup after the close; validation enters at ``t + 1``
open rather than assuming an executable fill at the signal close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SqueezeConfig:
    """Pre-specified daily squeeze definition; avoid tuning these per ticker."""

    band_period: int = 20
    band_stddev: float = 2.0
    keltner_period: int = 20
    keltner_atr_multiple: float = 1.5
    trend_period: int = 200
    breakout_period: int = 20
    volume_period: int = 20
    min_volume_ratio: float = 1.2
    bandwidth_lookback: int = 126
    max_bandwidth_percentile: float = 0.20
    min_squeeze_days: int = 3
    confirmation_window: int = 5


DEFAULT_CONFIG = SqueezeConfig()


@dataclass(frozen=True)
class SqueezeSignal:
    """Latest state of the long-only volatility-squeeze setup.

    ``status`` is one of ``none``, ``watch``, ``armed``, or ``confirmed``.  It is a
    condition label, not a probability estimate, recommendation, or order instruction.
    """

    status: str
    price: float
    squeeze_days: int
    recent_squeeze_days: int
    bandwidth_percentile: float | None
    volume_ratio: float | None
    breakout_level: float | None
    above_trend: bool
    close_above_breakout: bool

    @property
    def is_candidate(self) -> bool:
        return self.status in {"watch", "armed", "confirmed"}


def _require_positive(config: SqueezeConfig) -> None:
    if (
        config.band_period < 2
        or config.keltner_period < 2
        or config.trend_period < 2
        or config.breakout_period < 2
        or config.volume_period < 2
        or config.bandwidth_lookback < 5
        or config.min_squeeze_days < 1
        or config.confirmation_window < 1
        or config.band_stddev <= 0
        or config.keltner_atr_multiple <= 0
        or config.min_volume_ratio <= 0
        or not 0 < config.max_bandwidth_percentile <= 1
    ):
        raise ValueError("squeeze configuration contains an invalid period or threshold")


def _bandwidth_percentile(bandwidth: pd.Series, lookback: int) -> pd.Series:
    """Percentile of today's width against the preceding ``lookback`` widths.

    The final element in each rolling window is today's width; every earlier element
    is history.  This avoids the common look-ahead mistake of ranking a bar against
    observations that did not exist at the signal close.
    """

    def rank_current(values: np.ndarray) -> float:
        current = values[-1]
        history = values[:-1]
        if not np.isfinite(current) or not np.isfinite(history).all():
            return np.nan
        return float(np.mean(history <= current))

    return bandwidth.rolling(lookback + 1, min_periods=lookback + 1).apply(rank_current, raw=True)


def _consecutive_true(values: pd.Series) -> pd.Series:
    """Length of the current consecutive true run at each row."""

    flags = values.fillna(False).astype(bool)
    return flags.astype(int).groupby((~flags).cumsum()).cumsum().astype(int)


def build_squeeze_features(
    bars: pd.DataFrame,
    config: SqueezeConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return causal squeeze, trend, volume, and breakout features for daily OHLCV bars.

    Required columns are ``High``, ``Low``, ``Close``, and ``Volume``.  The input need
    not be adjusted because the setup is based on quoted market prices.  Return testing
    should use dividend-adjusted opens separately.
    """

    _require_positive(config)
    required = {"High", "Low", "Close", "Volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {', '.join(sorted(missing))}")

    frame = bars.sort_index().copy()
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / config.keltner_period,
        adjust=False,
        min_periods=config.keltner_period,
    ).mean()

    bb_mid = close.rolling(config.band_period, min_periods=config.band_period).mean()
    bb_std = close.rolling(config.band_period, min_periods=config.band_period).std(ddof=0)
    bb_upper = bb_mid + config.band_stddev * bb_std
    bb_lower = bb_mid - config.band_stddev * bb_std
    bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    kc_mid = close.ewm(span=config.keltner_period, adjust=False, min_periods=config.keltner_period).mean()
    kc_upper = kc_mid + config.keltner_atr_multiple * atr
    kc_lower = kc_mid - config.keltner_atr_multiple * atr
    squeeze_on = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False)
    squeeze_days = _consecutive_true(squeeze_on)
    bandwidth_percentile = _bandwidth_percentile(bandwidth, config.bandwidth_lookback)

    trend = close.rolling(config.trend_period, min_periods=config.trend_period).mean()
    above_trend = close > trend
    prior_high = high.shift(1).rolling(
        config.breakout_period,
        min_periods=config.breakout_period,
    ).max()
    volume_mean = volume.shift(1).rolling(
        config.volume_period,
        min_periods=config.volume_period,
    ).mean()
    volume_ratio = volume / volume_mean.replace(0, np.nan)

    compressed = (
        squeeze_on
        & (squeeze_days >= config.min_squeeze_days)
        & (bandwidth_percentile <= config.max_bandwidth_percentile)
    )
    # Compression must be visible before the breakout bar.  A rolling maximum here looks
    # only backwards because the source is shifted by one complete daily bar.
    prior_compressed = compressed.shift(1).rolling(
        config.confirmation_window,
        min_periods=1,
    ).max().fillna(0).astype(bool)
    prior_squeeze_days = squeeze_days.where(compressed).shift(1).rolling(
        config.confirmation_window,
        min_periods=1,
    ).max()

    close_above_breakout = close > prior_high
    volume_confirmed = volume_ratio >= config.min_volume_ratio
    raw_confirmed = prior_compressed & above_trend & close_above_breakout & volume_confirmed
    # A sustained move through the same 20-day high is one setup, not five independent ones.
    confirmed = raw_confirmed & ~raw_confirmed.shift(1, fill_value=False)
    watch = compressed & above_trend
    armed = prior_compressed & above_trend & ~raw_confirmed

    return pd.DataFrame(
        {
            "close": close,
            "sma_trend": trend,
            "above_trend": above_trend,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
            "bandwidth": bandwidth,
            "bandwidth_percentile": bandwidth_percentile,
            "squeeze_on": squeeze_on,
            "squeeze_days": squeeze_days,
            "compressed": compressed,
            "prior_compressed": prior_compressed,
            "recent_squeeze_days": prior_squeeze_days,
            "breakout_level": prior_high,
            "close_above_breakout": close_above_breakout,
            "volume_ratio": volume_ratio,
            "volume_confirmed": volume_confirmed,
            "watch": watch,
            "armed": armed,
            "raw_confirmed": raw_confirmed,
            "confirmed": confirmed,
        },
        index=frame.index,
    )


def latest_squeeze_signal(
    bars: pd.DataFrame,
    config: SqueezeConfig = DEFAULT_CONFIG,
) -> SqueezeSignal | None:
    """Classify the most recent completed daily bar, or return ``None`` if unusable."""

    features = build_squeeze_features(bars, config)
    if features.empty:
        return None
    row = features.iloc[-1]
    if bool(row["confirmed"]):
        status = "confirmed"
    elif bool(row["armed"]):
        status = "armed"
    elif bool(row["watch"]):
        status = "watch"
    else:
        status = "none"

    def optional_float(name: str) -> float | None:
        value = row[name]
        return float(value) if pd.notna(value) and np.isfinite(value) else None

    squeeze_days = int(row["squeeze_days"]) if pd.notna(row["squeeze_days"]) else 0
    recent_days = optional_float("recent_squeeze_days")
    return SqueezeSignal(
        status=status,
        price=float(row["close"]),
        squeeze_days=squeeze_days,
        recent_squeeze_days=int(recent_days) if recent_days is not None else 0,
        bandwidth_percentile=optional_float("bandwidth_percentile"),
        volume_ratio=optional_float("volume_ratio"),
        breakout_level=optional_float("breakout_level"),
        above_trend=bool(row["above_trend"]),
        close_above_breakout=bool(row["close_above_breakout"]),
    )
