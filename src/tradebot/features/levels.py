"""Support and resistance, derived causally.

Swing pivots are the place where look-ahead creeps into a "price action" system almost by
definition. A pivot high at bar *i* is defined by the bars on *both* sides of it, so it
cannot be known until `right` bars later. Charting packages draw it at bar *i*, which is
honest for a human reading a completed chart and catastrophic for a backtest reading the
same series as though it were available in real time.

Everything here therefore reports a level at the bar where it became **knowable**, not at
the bar it describes. `pivot_high(...)` places the confirmation `right` bars after the
pivot; `last_pivot_high(...)` forward-fills from there. `tests/tradebot/test_features.py`
asserts the confirmation lag explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _confirmed_pivots(series: pd.Series, left: int, right: int, kind: str) -> pd.Series:
    """The pivot value, placed at the bar where it becomes knowable (NaN elsewhere)."""
    if left < 1 or right < 1:
        raise ValueError("left and right must both be >= 1")
    width = left + right + 1

    # Trailing window ending at the current bar. The candidate sits `right` bars back, so
    # every bar the comparison touches is at or before the current one.
    window = series.rolling(width, min_periods=width)
    extreme = window.max() if kind == "high" else window.min()
    candidate = series.shift(right)

    is_pivot = candidate == extreme
    return candidate.where(is_pivot)


def pivot_high(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    return _confirmed_pivots(high, left, right, "high")


def pivot_low(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    return _confirmed_pivots(low, left, right, "low")


def last_pivot_high(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """The most recently *confirmed* swing high, carried forward."""
    return pivot_high(high, left, right).ffill()


def last_pivot_low(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    return pivot_low(low, left, right).ffill()


def recent_pivot_levels(
    series: pd.Series, left: int, right: int, kind: str, count: int = 4
) -> pd.DataFrame:
    """The `count` most recently confirmed pivots at each bar, newest first.

    Column `level_0` is the latest, `level_1` the one before it, and so on. Built by
    shifting the sequence of confirmations (not the price series), so each column is
    naturally causal: the k-th previous pivot is only visible once it has been superseded
    k times.
    """
    confirmed = _confirmed_pivots(series, left, right, kind)
    points = confirmed.dropna()

    out = {}
    for k in range(count):
        shifted = points.shift(k)
        out[f"level_{k}"] = shifted.reindex(series.index).ffill()
    return pd.DataFrame(out, index=series.index)


def _nearest(price: pd.Series, levels: pd.DataFrame, *, above: bool) -> pd.Series:
    """Closest level on one side of price, NaN when there is none on that side.

    A row with no qualifying level is the normal case, not an anomaly — price sitting
    above every confirmed swing high simply means there is no resistance to name. Rows are
    masked out before the reduction so numpy is never asked to reduce an all-NaN slice,
    which would otherwise emit a RuntimeWarning on ordinary data.
    """
    values = levels.to_numpy(dtype="float64")
    ref = price.to_numpy(dtype="float64")[:, None]

    result = np.full(len(price), np.nan)
    if values.size:
        side = np.where(values > ref if above else values < ref, values, np.nan)
        has_any = ~np.isnan(side).all(axis=1)
        if has_any.any():
            reducer = np.nanmin if above else np.nanmax
            result[has_any] = reducer(side[has_any], axis=1)
    return pd.Series(result, index=price.index)


def nearest_level_above(price: pd.Series, levels: pd.DataFrame) -> pd.Series:
    """The lowest level strictly above price, or NaN when there is none."""
    return _nearest(price, levels, above=True)


def nearest_level_below(price: pd.Series, levels: pd.DataFrame) -> pd.Series:
    """The highest level strictly below price, or NaN when there is none."""
    return _nearest(price, levels, above=False)


def touched_within(price: pd.Series, level: pd.Series, tolerance: pd.Series | float) -> pd.Series:
    """Whether price is within `tolerance` of a level. Tolerance is normally an ATR
    fraction, so the test widens in fast markets rather than being a fixed point count
    that is meaningless across volatility regimes."""
    return (price - level).abs() <= tolerance


def broke_above(close: pd.Series, level: pd.Series) -> pd.Series:
    """A close that crosses from at-or-below a level to above it.

    Uses closes, not highs: a wick through a level and back is not a break, and treating
    it as one produces a strategy that is stopped out by design.
    """
    prev_close, prev_level = close.shift(1), level.shift(1)
    return (close > level) & (prev_close <= prev_level)


def broke_below(close: pd.Series, level: pd.Series) -> pd.Series:
    prev_close, prev_level = close.shift(1), level.shift(1)
    return (close < level) & (prev_close >= prev_level)


def bars_since(condition: pd.Series) -> pd.Series:
    """How many bars ago `condition` was last True. NaN before the first occurrence.

    Used for "the breakout happened, has price come back within N bars" style rules.
    """
    idx = np.arange(len(condition))
    marks = pd.Series(np.where(condition.to_numpy(), idx, np.nan), index=condition.index)
    return pd.Series(idx, index=condition.index) - marks.ffill()
