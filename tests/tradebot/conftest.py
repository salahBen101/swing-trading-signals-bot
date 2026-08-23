"""Shared fixtures and builders for the tradebot suite."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.instruments.registry import get_instrument
from tradebot.strategy.base import StrategyContext

MNQ = get_instrument("MNQ")


def make_session(
    day: str,
    closes,
    *,
    wick: float = 3.0,
    volume: float = 100.0,
    highs=None,
    lows=None,
    volumes=None,
    start: str = "09:30",
) -> pd.DataFrame:
    """One session of 1-minute bars built from a close path.

    Highs and lows default to a symmetric wick around the close, which keeps OHLC valid
    without the test having to spell out four numbers per bar.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    idx = pd.date_range(f"{day} {start}", periods=n, freq="1min", tz=MARKET_TZ)

    opens = np.concatenate([[closes[0]], closes[:-1]])
    hi = np.asarray(highs, dtype="float64") if highs is not None else np.maximum(opens, closes) + wick
    lo = np.asarray(lows, dtype="float64") if lows is not None else np.minimum(opens, closes) - wick
    vol = np.asarray(volumes, dtype="float64") if volumes is not None else np.full(n, volume)

    hi = np.maximum.reduce([hi, opens, closes])
    lo = np.minimum.reduce([lo, opens, closes])

    out = pd.DataFrame(
        {"open": opens, "high": hi, "low": lo, "close": closes, "volume": vol}, index=idx
    )
    out.index.name = "timestamp"
    return out


def warmup_sessions(
    days: list[str], bars_per_session: int = 140, level: float = 18000.0, seed: int = 3
) -> pd.DataFrame:
    """Quiet noise sessions, enough history for the 50-bar indicators to be defined."""
    rng = np.random.default_rng(seed)
    frames = []
    price = level
    for day in days:
        closes = price + np.cumsum(rng.normal(0, 1.2, bars_per_session))
        frames.append(make_session(day, closes, wick=2.5))
        price = float(closes[-1])
    return pd.concat(frames)


@pytest.fixture(scope="session")
def multi_session_bars() -> pd.DataFrame:
    """Ten realistic-ish RTH sessions, for integration sweeps across all strategies."""
    rng = np.random.default_rng(17)
    frames = []
    price = 18000.0
    for d in range(10):
        day = (pd.Timestamp("2024-04-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        n = 390
        # A regime per day: alternating drift and chop, so trend and fade rules both get
        # sessions they claim to work in.
        drift = (0.05 if d % 3 == 0 else -0.05 if d % 3 == 1 else 0.0)
        closes = price + np.cumsum(rng.normal(drift, 2.0, n))
        vols = rng.integers(60, 400, n).astype(float)
        frames.append(make_session(day, closes, wick=3.5, volumes=vols))
        price = float(closes[-1])
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out


def make_context(
    features: dict,
    *,
    bar: dict | None = None,
    previous_bar: dict | None = None,
    timestamp: datetime | None = None,
    trades_this_session: int = 0,
    equity: float = 50_000.0,
    i: int = 1,
) -> StrategyContext:
    """A hand-built context, so a rule can be tested against exact feature values.

    The frame is padded to `i + 1` rows: rules that read the prior close need at least one
    row behind the current bar, and rules that compare bar indices (the breakout/retest
    latch) need the current bar to sit at a realistic position rather than at index 1.
    Every padded row repeats the "previous" values, so only the final row differs.
    """
    timestamp = timestamp or datetime(2024, 4, 1, 11, 0, tzinfo=MARKET_TZ)
    close = float(features.get("close", 18000.0))
    bar = {"open": close, "high": close, "low": close, "close": close, "volume": 100.0, **(bar or {})}
    prev = {**bar, **(previous_bar or {})}

    n = i + 1
    index = pd.DatetimeIndex(
        [timestamp - timedelta(minutes=n - 1 - k) for k in range(n)], name="timestamp"
    )
    bars = pd.DataFrame([prev] * (n - 1) + [bar], index=index)

    row = {"close": close, **features}
    row.setdefault("open", bar["open"])
    row.setdefault("high", bar["high"])
    row.setdefault("low", bar["low"])
    row.setdefault("volume", bar["volume"])
    prev_row = {**row, "close": prev["close"], "open": prev["open"],
                "high": prev["high"], "low": prev["low"]}
    frame = pd.DataFrame([prev_row] * (n - 1) + [row], index=index)

    return StrategyContext(
        i=i,
        timestamp=timestamp,
        bar=bars.iloc[i],
        features=frame.iloc[i],
        instrument=MNQ,
        equity=equity,
        session_date=timestamp.date(),
        trades_this_session=trades_this_session,
        _frame=frame,
        _bars=bars,
    )


@pytest.fixture
def position_factory():
    """Build an open `Position` without going through the whole execution path."""
    from tradebot.core.models import Position
    from tradebot.core.types import Side

    def make(
        *,
        side: Side = Side.BUY,
        entry: float = 18000.0,
        stop: float = 17990.0,
        quantity: int = 1,
        risk_points: float | None = None,
        target: float | None = None,
        strategy: str = "test",
        entry_features: dict | None = None,
    ) -> Position:
        return Position(
            instrument="MNQ",
            side=side,
            quantity=quantity,
            entry_price=entry,
            entry_time=datetime(2024, 4, 1, 11, 0, tzinfo=MARKET_TZ),
            strategy=strategy,
            initial_stop=stop,
            stop_price=stop,
            target_price=target,
            risk_per_contract_points=(
                risk_points if risk_points is not None else abs(entry - stop)
            ),
            entry_features=entry_features or {},
        )

    return make
