"""Regression tests for the causal daily volatility-compression watch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scanner.squeeze_strategy import SqueezeConfig, build_squeeze_features, latest_squeeze_signal


def _small_config() -> SqueezeConfig:
    """Short periods make the signal construction observable in a compact fixture."""

    return SqueezeConfig(
        band_period=3,
        keltner_period=3,
        trend_period=5,
        breakout_period=3,
        volume_period=3,
        bandwidth_lookback=5,
        max_bandwidth_percentile=0.20,
        min_squeeze_days=2,
        confirmation_window=3,
    )


def _fixture_bars() -> pd.DataFrame:
    # Alternating history supplies wide prior bands.  The final tight range is a genuine
    # low-bandwidth squeeze; the last bar closes through the *prior* three-day high on 2x
    # normal volume.  It is deliberately not a same-close execution test.
    close = np.array(
        [
            100, 104, 99, 105, 98, 106, 99, 105, 100, 106, 101, 107,
            102, 108, 103, 109, 108, 108.02, 108.01, 108.03, 108.02,
            108.04, 108.03, 108.05, 112,
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.r_[np.full(len(close) - 1, 1_000_000), 2_000_000],
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="B"),
    )


def test_confirmed_breakout_requires_prior_compression_and_next_bar_volume() -> None:
    signal = latest_squeeze_signal(_fixture_bars(), _small_config())

    assert signal is not None
    assert signal.status == "confirmed"
    assert signal.recent_squeeze_days >= 2
    assert signal.close_above_breakout
    assert signal.breakout_level is not None
    assert signal.price > signal.breakout_level
    assert signal.volume_ratio == 2.0


def test_future_bar_cannot_change_existing_squeeze_features() -> None:
    bars = _fixture_bars()
    original = build_squeeze_features(bars, _small_config())
    future = bars.iloc[[-1]].copy()
    future.index = [bars.index[-1] + pd.offsets.BDay()]
    future.loc[:, ["Open", "High", "Low", "Close"]] = [200.0, 250.0, 150.0, 200.0]
    future.loc[:, "Volume"] = 20_000_000

    with_future = build_squeeze_features(pd.concat([bars, future]), _small_config())

    assert_frame_equal(original, with_future.loc[original.index], check_freq=False)
