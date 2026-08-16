from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.rsi2_strategy import backtest, portfolio_backtest


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=210)
    close = np.full(len(dates), 100.0)
    close[200:206] = [101.0, 102.0, 101.0, 100.5, 102.0, 102.5]
    open_ = close.copy()
    open_[203] = 90.0   # Executable entry after the completed signal close.
    open_[205] = 110.0  # Executable exit after the completed exit signal close.
    return pd.DataFrame({"Open": open_, "Close": close}, index=dates)


def test_next_open_backtest_never_fills_at_the_signal_close():
    bars = _bars()
    trades = backtest(bars, entry_rsi=99.0, cost_pct=0.0, execution="next_open")

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["signal_date"] == bars.index[202]
    assert trade["entry_date"] == bars.index[203]
    assert trade["entry"] == 90.0
    assert trade["exit_signal_date"] == bars.index[204]
    assert trade["exit_date"] == bars.index[205]
    assert trade["exit"] == 110.0


def test_same_close_is_explicitly_different_from_next_open():
    bars = _bars()
    close_fill = backtest(bars, entry_rsi=99.0, cost_pct=0.0, execution="same_close")
    next_open_fill = backtest(bars, entry_rsi=99.0, cost_pct=0.0, execution="next_open")

    assert close_fill.iloc[0]["entry_date"] == bars.index[202]
    assert next_open_fill.iloc[0]["entry_date"] == bars.index[203]
    assert close_fill.iloc[0]["entry"] != next_open_fill.iloc[0]["entry"]


def test_portfolio_uses_a_fixed_slot_limit_for_simultaneous_signals():
    bars = _bars()
    result = portfolio_backtest(
        {"AAA": bars, "BBB": bars},
        start=bars.index[0],
        entry_rsi=99.0,
        max_positions=1,
        cost_pct=0.0,
        calendar_ticker="AAA",
    )

    assert len(result.candidates) == 2
    assert len(result.selected_trades) == 1
    assert result.active_positions.max() == 1
    assert result.daily_returns.notna().all()


def test_backtest_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="execution"):
        backtest(_bars(), execution="magic")
