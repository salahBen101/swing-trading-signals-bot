"""Bar-frame integrity checks.

Run on every load, not just on import. A corrupted price series does not announce itself:
a single stray tick printing 25000 while the market is at 18000 produces a plausible-looking
equity curve driven entirely by one fictional bar, and nothing downstream can tell.

The thresholds are calibrated against this project's real NQ data rather than guessed, and
they **scale with the bar duration**, because a bar's range grows roughly with the square
root of the time it covers. A flat 2% limit is right for 15-second bars and wrong for
5-minute ones.

Measured on the 9-year archive (898k 1-minute bars, 788k 15-second bars), the largest
*legitimate* ranges as a share of price were:

| timeframe | observed max | scaled threshold | headroom |
|---|---|---|---|
| 15s | 1.43% | 2.00% | 1.40x |
| 1m  | 3.35% | 4.00% | 1.20x |
| 5m  | 4.84% | 8.94% | 1.85x |
| 15m | 7.42% | 15.5% | 2.09x |

Every one of those extremes is a datable macro event — the 2020-03-16 COVID limit-down, the
2020-03-03 emergency Fed cut, and the April 2025 tariff sessions — not corruption. Genuine
corruption is an order of magnitude further out: a stray tick printing 25000 while the
market is at 18000 is a 28% range, caught at every timeframe above.

This frame-level check is an **import/batch** guard. The live path validates bar-by-bar via
`Bar.__post_init__` (OHLC consistency only), so an FOMC print cannot take a running session
down on a statistical threshold.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# The calibration point: 2% of price over a 15-second bar.
_REFERENCE_SECONDS = 15.0
_REFERENCE_RANGE_PCT = 0.02


def range_pct_threshold(timeframe_seconds: float) -> float:
    """Bad-tick threshold for a given bar duration, scaled as sqrt(time)."""
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    return _REFERENCE_RANGE_PCT * (timeframe_seconds / _REFERENCE_SECONDS) ** 0.5


class DataIntegrityError(ValueError):
    """Raised when a bar frame fails a check. The message names the offending rows."""


def validate_bars(
    df: pd.DataFrame,
    *,
    timeframe_seconds: float = _REFERENCE_SECONDS,
    max_range_pct_of_price: float | None = None,
    max_wick_ratio: float = 100.0,
    require_tz_aware: bool = True,
    min_periods: int = 20,
) -> None:
    if max_range_pct_of_price is None:
        max_range_pct_of_price = range_pct_threshold(timeframe_seconds)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataIntegrityError(f"missing columns: {missing}")
    if df.empty:
        raise DataIntegrityError("bar frame is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"index must be a DatetimeIndex, got {type(df.index).__name__}")
    if require_tz_aware and df.index.tz is None:
        raise DataIntegrityError(
            "bar index is timezone-naive; a UTC series read as Eastern shifts every "
            "decision by hours and nothing in the output looks wrong"
        )
    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError("timestamps are not monotonically increasing")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].tolist()[:5]
        raise DataIntegrityError(f"duplicate timestamps, e.g. {dupes}")

    ohlcv = df.loc[:, list(REQUIRED_COLUMNS)]
    if ohlcv.isna().any().any():
        bad = ohlcv[ohlcv.isna().any(axis=1)].index[:5].tolist()
        raise DataIntegrityError(f"missing values in OHLCV, e.g. {bad}")
    if (df["volume"] < 0).any():
        raise DataIntegrityError("negative volume present")

    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if bad_high.any() or bad_low.any():
        n = int((bad_high | bad_low).sum())
        first = df.loc[bad_high | bad_low].head(3)
        raise DataIntegrityError(
            f"{n} bars have inconsistent OHLC (high/low do not bound open/close):\n{first}"
        )

    rng = df["high"] - df["low"]

    absurd = rng > (df["close"].abs() * max_range_pct_of_price)
    if absurd.any():
        worst = df.loc[absurd, ["high", "low", "close"]].head(3)
        raise DataIntegrityError(
            f"{int(absurd.sum())} bars span more than {max_range_pct_of_price:.2%} of price "
            f"(threshold for a {timeframe_seconds:g}s bar) - almost certainly bad ticks. "
            f"If these are a real macro event, pass an explicit max_range_pct_of_price. "
            f"First offenders:\n{worst}"
        )

    median_rng = rng.rolling(200, min_periods=min_periods).median()
    spike = (rng > (median_rng * max_wick_ratio)) & median_rng.notna() & (median_rng > 0)
    if spike.any():
        worst_ratio = float((rng / median_rng)[spike].max())
        raise DataIntegrityError(
            f"{int(spike.sum())} bars span more than {max_wick_ratio}x the local rolling "
            f"median (worst: {worst_ratio:.1f}x) - inspect before use"
        )


def describe_bars(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    sessions = df.index.normalize().nunique()
    return (
        f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}  "
        f"{sessions:,} sessions  tz={df.index.tz}"
    )
