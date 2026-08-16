"""Executable research primitives for the daily RSI(2) pullback strategy.

The scanner itself only reports an indicator.  This module deliberately models the
implementation that an after-close scanner can actually trade: a signal is known at a
daily close and is filled at the *next* session's open.  The old research filled at the
same close that created the signal, which is useful as a diagnostic but not as the default
for a console checked after the market closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Wilder RSI, with deterministic values for zero-loss windows."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    boundary = pd.Series(np.where(avg_gain > 0, 100.0, 50.0), index=close.index)
    return out.where(avg_loss != 0, boundary)


def load_cached_bars(ticker: str, cache_dir: str | Path = "data_cache") -> pd.DataFrame | None:
    """Load a ticker's local daily history without silently downloading new data."""
    path = Path(cache_dir) / f"{ticker.replace('-', '')}_daily.parquet"
    if not path.exists():
        return None
    bars = pd.read_parquet(path).sort_index()
    required = {"Open", "Close"}
    if not required.issubset(bars.columns):
        return None
    bars = bars.dropna(subset=list(required))
    return bars if not bars.empty else None


def total_return_open(bars: pd.DataFrame) -> pd.Series:
    """Open prices adjusted for dividends when an adjusted close is available.

    Signals remain based on the actual quoted close, while P&L includes cash dividends
    received during a holding period. Scaling opens by ``Adj Close / Close`` is the standard
    way to construct an adjusted intraday price when the vendor supplies only adjusted close.
    """
    open_ = bars["Open"].astype(float)
    if "Adj Close" not in bars.columns or "Close" not in bars.columns:
        return open_
    close = bars["Close"].astype(float)
    adjusted_close = bars["Adj Close"].astype(float)
    factor = (adjusted_close / close).replace([np.inf, -np.inf], np.nan)
    return (open_ * factor).where(factor.notna(), open_)


def total_return_close(bars: pd.DataFrame) -> pd.Series:
    """Close series suitable for P&L; signals should still use the quoted close."""
    if "Adj Close" in bars.columns:
        return bars["Adj Close"].astype(float).fillna(bars["Close"].astype(float))
    return bars["Close"].astype(float)


def backtest(
    bars: pd.DataFrame,
    *,
    entry_rsi: float = 5.0,
    exit_rsi: float = 65.0,
    trend_filter: bool = True,
    trend_period: int = 200,
    max_hold: int = 10,
    cost_pct: float = 0.0004,
    execution: str = "next_open",
    entry_filter: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate non-overlapping RSI(2) trades from daily bars.

    ``execution='next_open'`` is the default and the only mode suitable for an
    after-close scan: signals and exits are decided from a completed close, then filled
    at the next available open.  ``same_close`` remains only to quantify the optimistic
    historical assumption used by the original scripts.

    ``cost_pct`` is a round-trip fraction of notional.  It intentionally applies a
    conservative four basis points by default rather than treating retail execution as
    free.
    """
    if execution not in {"next_open", "same_close"}:
        raise ValueError("execution must be 'next_open' or 'same_close'")
    if cost_pct < 0:
        raise ValueError("cost_pct cannot be negative")

    required = {"Open", "Close"}
    if not required.issubset(bars.columns):
        missing = ", ".join(sorted(required - set(bars.columns)))
        raise ValueError(f"bars missing required columns: {missing}")

    bars = bars.sort_index().dropna(subset=list(required))
    empty_columns = [
        "signal_date", "entry_date", "exit_signal_date", "exit_date", "entry", "exit",
        "days", "entry_rsi", "gross_pct", "net_pct", "year",
    ]
    if len(bars) < trend_period + 3:
        return pd.DataFrame(columns=empty_columns)

    close = bars["Close"].astype(float)
    open_ = bars["Open"].astype(float)
    return_open = total_return_open(bars)
    return_close = total_return_close(bars)
    r = rsi(close, 2)
    sma = close.rolling(trend_period).mean()
    sma5 = close.rolling(5).mean()
    allowed_entries = (
        pd.Series(True, index=bars.index)
        if entry_filter is None
        else entry_filter.reindex(bars.index, method="ffill").fillna(False).astype(bool)
    )

    trades: list[dict] = []
    i = 0
    n = len(bars)

    while i < n - 2:
        signal_date = bars.index[i]
        signal_price = float(close.iloc[i])
        if (
            pd.isna(r.iloc[i])
            or pd.isna(sma.iloc[i])
            or not bool(allowed_entries.iloc[i])
        ):
            i += 1
            continue

        trend_ok = (not trend_filter) or signal_price > float(sma.iloc[i])
        if float(r.iloc[i]) >= entry_rsi or not trend_ok:
            i += 1
            continue

        if execution == "same_close":
            entry_idx = i
            entry_price = signal_price
            entry_return_price = float(return_close.iloc[entry_idx])
            first_exit_idx = i + 1
        else:
            entry_idx = i + 1
            entry_price = float(open_.iloc[entry_idx])
            entry_return_price = float(return_open.iloc[entry_idx])
            first_exit_idx = entry_idx

        if (
            not np.isfinite(entry_price)
            or entry_price <= 0
            or not np.isfinite(entry_return_price)
            or entry_return_price <= 0
        ):
            i += 1
            continue

        exit_signal_idx: int | None = None
        # Reserve one later bar for a next-open exit fill.  For same-close execution the
        # close that supplies the exit signal is also the fill, so the final bar is usable.
        latest_exit_signal = n - 1 if execution == "same_close" else n - 2
        for j in range(first_exit_idx, latest_exit_signal + 1):
            sessions_held = j - entry_idx + (1 if execution == "next_open" else 0)
            exit_now = (
                float(r.iloc[j]) > exit_rsi
                or float(close.iloc[j]) > float(sma5.iloc[j])
                or sessions_held >= max_hold
            )
            if exit_now:
                exit_signal_idx = j
                break

        if exit_signal_idx is None:
            break

        exit_idx = exit_signal_idx if execution == "same_close" else exit_signal_idx + 1
        exit_price = float(close.iloc[exit_idx]) if execution == "same_close" else float(open_.iloc[exit_idx])
        exit_return_price = (
            float(return_close.iloc[exit_idx])
            if execution == "same_close"
            else float(return_open.iloc[exit_idx])
        )
        if (
            not np.isfinite(exit_price)
            or exit_price <= 0
            or not np.isfinite(exit_return_price)
            or exit_return_price <= 0
        ):
            i += 1
            continue

        gross = exit_return_price / entry_return_price - 1
        trades.append(
            {
                "signal_date": signal_date,
                "entry_date": bars.index[entry_idx],
                "exit_signal_date": bars.index[exit_signal_idx],
                "exit_date": bars.index[exit_idx],
                "entry": entry_price,
                "exit": exit_price,
                "days": exit_idx - entry_idx,
                "entry_rsi": float(r.iloc[i]),
                "gross_pct": gross * 100,
                "net_pct": (gross - cost_pct) * 100,
                "year": signal_date.year,
            }
        )
        # An exit at today's open still permits a fresh signal at today's close.
        i = exit_idx + 1 if execution == "same_close" else max(i + 1, exit_idx)

    return pd.DataFrame(trades, columns=empty_columns)


def trade_stats(trades: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Per-instrument trade statistics; do not use these as a portfolio equity curve."""
    if trades.empty:
        return {"n": 0}
    net = trades.sort_values("entry_date")["net_pct"].astype(float) / 100
    wins = net[net > 0]
    losses = net[net <= 0]
    gross_losses = abs(losses.sum())
    equity = (1 + net).cumprod()
    drawdown = float(((equity.cummax() - equity) / equity.cummax()).max())
    total_days = max(len(bars) - 1, 1)
    exposure = float(trades["days"].sum()) / total_days
    years = total_days / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    standard_error = net.std(ddof=1) / np.sqrt(len(net)) if len(net) > 1 else np.nan
    return {
        "n": len(net),
        "win_pct": round(float((net > 0).mean() * 100), 1),
        "avg_pct": round(float(net.mean() * 100), 3),
        "pf": round(float(wins.sum() / gross_losses), 3) if gross_losses > 0 else np.inf,
        "total_return_pct": round(float((equity.iloc[-1] - 1) * 100), 1),
        "cagr_pct": round(cagr * 100, 2),
        "maxDD_pct": round(drawdown * 100, 1),
        "exposure_pct": round(exposure * 100, 1),
        "avg_days": round(float(trades["days"].mean()), 1),
        # This is explicitly a trade-level diagnostic.  It does not assume positions across
        # tickers are independent; use portfolio_stats for the strategy-level t-statistic.
        "trade_t": round(float(net.mean() / standard_error), 2)
        if np.isfinite(standard_error) and standard_error > 0
        else 0.0,
        # Compatibility key for older reporting scripts. It remains deliberately a
        # trade-level diagnostic, not portfolio-level evidence.
        "t": round(float(net.mean() / standard_error), 2)
        if np.isfinite(standard_error) and standard_error > 0
        else 0.0,
    }


@dataclass(frozen=True)
class PortfolioResult:
    daily_returns: pd.Series
    active_positions: pd.Series
    candidates: pd.DataFrame
    selected_trades: pd.DataFrame


def _select_trades(
    candidates: pd.DataFrame,
    *,
    max_positions: int,
    sector_of: Callable[[str], str] | None,
    max_positions_per_sector: int | None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    if max_positions < 1:
        raise ValueError("max_positions must be at least one")
    if max_positions_per_sector is not None and max_positions_per_sector < 1:
        raise ValueError("max_positions_per_sector must be at least one")

    # Capacity is a risk-control decision, not another fitted signal. Alphabetical tie
    # breaking is intentionally arbitrary and fully known before the next-open fill.
    ordered = candidates.sort_values(["entry_date", "ticker", "entry_rsi"]).reset_index(drop=True)
    kept: list[dict] = []
    active: list[dict] = []
    for entry_date, group in ordered.groupby("entry_date", sort=True):
        # Positions exit at this open before any new entries are filled.
        active = [trade for trade in active if trade["exit_date"] > entry_date]
        for _, row in group.iterrows():
            if len(active) >= max_positions:
                continue
            record = row.to_dict()
            sector = sector_of(record["ticker"]) if sector_of is not None else None
            if max_positions_per_sector is not None and sector is not None:
                in_sector = sum(
                    1 for trade in active if sector_of is not None and sector_of(trade["ticker"]) == sector
                )
                if in_sector >= max_positions_per_sector:
                    continue
            kept.append(record)
            active.append(record)
    return pd.DataFrame(kept, columns=candidates.columns)


def portfolio_backtest(
    bars_by_ticker: Mapping[str, pd.DataFrame],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    entry_rsi: float = 5.0,
    exit_rsi: float = 65.0,
    trend_filter: bool = True,
    trend_period: int = 200,
    max_hold: int = 10,
    cost_pct: float = 0.0004,
    max_positions: int = 10,
    sector_of: Callable[[str], str] | None = None,
    max_positions_per_sector: int | None = None,
    calendar_ticker: str = "SPY",
) -> PortfolioResult:
    """Backtest a cash-constrained, equal-slot portfolio using next-open execution.

    Each active position receives ``1 / max_positions`` of capital, with the remainder in
    cash. Trades competing for capacity use an arbitrary alphabetical tie-break rather than
    another performance-tuned signal. Daily
    returns, rather than individual overlapping trades, are the unit of strategy inference.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    parts: list[pd.DataFrame] = []
    prepared: dict[str, pd.DataFrame] = {}
    for ticker, raw in bars_by_ticker.items():
        bars = raw.sort_index().dropna(subset=["Open", "Close"])
        prepared[ticker] = bars
        trades = backtest(
            bars,
            entry_rsi=entry_rsi,
            exit_rsi=exit_rsi,
            trend_filter=trend_filter,
            trend_period=trend_period,
            max_hold=max_hold,
            cost_pct=cost_pct,
            execution="next_open",
        )
        if not trades.empty:
            trades.insert(0, "ticker", ticker)
            parts.append(trades)

    columns = [
        "ticker", "signal_date", "entry_date", "exit_signal_date", "exit_date", "entry", "exit",
        "days", "entry_rsi", "gross_pct", "net_pct", "year",
    ]
    candidates = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if not candidates.empty:
        candidates = candidates[candidates["signal_date"] >= start_ts]
        if end_ts is not None:
            candidates = candidates[candidates["exit_date"] <= end_ts]
    selected = _select_trades(
        candidates,
        max_positions=max_positions,
        sector_of=sector_of,
        max_positions_per_sector=max_positions_per_sector,
    )

    calendar_source = prepared.get(calendar_ticker)
    if calendar_source is None:
        calendar_source = next(iter(prepared.values()), pd.DataFrame(index=pd.DatetimeIndex([])))
    sessions = pd.DatetimeIndex(calendar_source.index)
    sessions = sessions[sessions >= start_ts]
    if end_ts is not None:
        sessions = sessions[sessions <= end_ts]
    sessions = sessions.sort_values().unique()
    if len(sessions) < 2:
        empty = pd.Series(dtype=float)
        return PortfolioResult(empty, empty, candidates, selected)

    opens = {ticker: total_return_open(bars) for ticker, bars in prepared.items()}
    returns: list[float] = []
    active_counts: list[int] = []
    index: list[pd.Timestamp] = []
    side_cost = cost_pct / 2

    for date, next_date in zip(sessions[:-1], sessions[1:]):
        if selected.empty:
            active = selected
        else:
            active = selected[(selected["entry_date"] <= date) & (selected["exit_date"] > date)]
        gross = 0.0
        valid_positions = 0
        for _, trade in active.iterrows():
            series = opens[trade["ticker"]]
            p0 = series.get(date)
            p1 = series.get(next_date)
            if p0 is None or p1 is None or not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
                continue
            gross += float(p1 / p0 - 1) / max_positions
            valid_positions += 1

        entered = 0 if selected.empty else int((selected["entry_date"] == date).sum())
        exited = 0 if selected.empty else int((selected["exit_date"] == next_date).sum())
        returns.append(gross - (entered + exited) * side_cost / max_positions)
        active_counts.append(valid_positions)
        index.append(date)

    return PortfolioResult(
        daily_returns=pd.Series(returns, index=pd.DatetimeIndex(index), name="portfolio_return"),
        active_positions=pd.Series(active_counts, index=pd.DatetimeIndex(index), name="active_positions"),
        candidates=candidates.reset_index(drop=True),
        selected_trades=selected.reset_index(drop=True),
    )


def _newey_west_t(values: pd.Series, lags: int = 10) -> float:
    """A small dependency-free HAC t-statistic for a mean daily return."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    lag_count = min(max(lags, 0), n - 1)
    centered = x - x.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * covariance
    if long_run_variance <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(long_run_variance / n))


def portfolio_stats(result: PortfolioResult, *, hac_lags: int = 10) -> dict:
    """Strategy-level statistics from a daily, correlated portfolio return series."""
    daily = result.daily_returns.dropna()
    if daily.empty:
        return {"n_days": 0, "n_trades": 0, "n_candidates": len(result.candidates)}
    equity = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    drawdown = float(((equity.cummax() - equity) / equity.cummax()).max())
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    return {
        "n_days": len(daily),
        "n_trades": len(result.selected_trades),
        "n_candidates": len(result.candidates),
        "cagr_pct": round(cagr * 100, 2),
        "total_return_pct": round(float((equity.iloc[-1] - 1) * 100), 1),
        "maxDD_pct": round(drawdown * 100, 1),
        "annual_vol_pct": round(daily_std * np.sqrt(TRADING_DAYS) * 100, 2)
        if np.isfinite(daily_std)
        else np.nan,
        "sharpe_zero_rf": round(float(daily.mean() / daily_std * np.sqrt(TRADING_DAYS)), 2)
        if np.isfinite(daily_std) and daily_std > 0
        else np.nan,
        "mean_daily_bp": round(float(daily.mean() * 10_000), 3),
        "hac_t": round(_newey_west_t(daily, lags=hac_lags), 2),
        "avg_active_positions": round(float(result.active_positions.mean()), 2),
        "max_active_positions": int(result.active_positions.max()),
    }
