"""Validate RSI(2) long-only mean reversion on daily data.

The classic Connors rules: buy when RSI(2) is deeply oversold while price is above its
200-day average, exit when the bounce completes. Long only.

Two design choices make this test more informative than an arbitrary split:

  1. The strategy was popularized around 2008 ("Short Term Trading Strategies That Work").
     Everything before that is effectively in-sample for its original authors; everything
     after is a post-publication historical test. It is not a pristine forward test after the
     strategy, universe, and filters have been inspected in this repository.
  2. A long-only strategy in a market that rose for 30 years will look profitable no matter
     what. The only meaningful benchmark is buy-and-hold, adjusted for the fact that this
     strategy is in the market roughly 10-20% of the time.

This script reports one ticker at a time. For a correlation-aware, cash-constrained 120-name
portfolio result, use scripts/rsi2_portfolio_validation.py instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.rsi2_strategy import (
    TRADING_DAYS,
    backtest as executable_backtest,
    rsi as executable_rsi,
    trade_stats as executable_trade_stats,
)

PUBLICATION_YEAR = 2009


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    boundary = pd.Series(np.where(avg_gain > 0, 100.0, 50.0), index=close.index)
    return out.where(avg_loss != 0, boundary)


def backtest(
    bars: pd.DataFrame,
    *,
    entry_rsi: float = 5.0,
    exit_rsi: float = 65.0,
    trend_filter: bool = True,
    trend_period: int = 200,
    max_hold: int = 10,
    cost_pct: float = 0.0002,
) -> pd.DataFrame:
    """Enter at the close that triggers, exit at the close that satisfies the exit rule.

    Costs are charged round trip. On SPY the spread is about a cent on a ~$500 instrument, so
    0.02% is a fair retail assumption for shares.
    """
    close = bars["Close"]
    r = rsi(close, 2)
    sma = close.rolling(trend_period).mean()
    sma5 = close.rolling(5).mean()

    in_trade = False
    entry_price = 0.0
    entry_date = None
    days_held = 0
    trades = []

    for i in range(len(bars)):
        date = bars.index[i]
        px = float(close.iloc[i])

        if in_trade:
            days_held += 1
            exit_now = (
                float(r.iloc[i]) > exit_rsi
                or px > float(sma5.iloc[i])
                or days_held >= max_hold
            )
            if exit_now:
                gross = px / entry_price - 1
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry": entry_price,
                        "exit": px,
                        "days": days_held,
                        "gross_pct": gross * 100,
                        "net_pct": (gross - cost_pct) * 100,
                        "year": entry_date.year,
                    }
                )
                in_trade = False
            continue

        if pd.isna(sma.iloc[i]) or pd.isna(r.iloc[i]):
            continue
        trend_ok = (not trend_filter) or px > float(sma.iloc[i])
        if float(r.iloc[i]) < entry_rsi and trend_ok:
            in_trade = True
            entry_price = px
            entry_date = date
            days_held = 0

    return pd.DataFrame(trades)


def stats(trades: pd.DataFrame, bars: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    net = trades["net_pct"] / 100
    wins = net[net > 0]
    losses = net[net <= 0]
    gl = abs(losses.sum())

    equity = (1 + net).cumprod()
    dd = float(((equity.cummax() - equity) / equity.cummax()).max())

    days_in_market = trades["days"].sum()
    total_days = len(bars)
    exposure = days_in_market / total_days if total_days else 0.0
    years = total_days / TRADING_DAYS

    total_return = float(equity.iloc[-1] - 1)
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "n": len(net),
        "win_pct": round((net > 0).mean() * 100, 1),
        "avg_pct": round(net.mean() * 100, 3),
        "pf": round(wins.sum() / gl, 3) if gl > 0 else np.inf,
        "total_return_pct": round(total_return * 100, 1),
        "cagr_pct": round(cagr * 100, 2),
        "maxDD_pct": round(dd * 100, 1),
        "exposure_pct": round(exposure * 100, 1),
        "avg_days": round(trades["days"].mean(), 1),
        "t": round(net.mean() / (net.std() / np.sqrt(len(net))), 2) if net.std() > 0 else 0.0,
    }


# The original same-close research functions above are retained only so old notebooks can be
# inspected. All public names below deliberately point to the executable implementation: a
# completed daily signal fills at the following session's open, with a four-basis-point
# round-trip default. Use execution="same_close" only to measure the old optimistic assumption.
rsi = executable_rsi
backtest = executable_backtest
stats = executable_trade_stats


def buy_and_hold(bars: pd.DataFrame) -> dict:
    # Use vendor-adjusted closes when available so the benchmark includes dividends just as
    # the executable backtest now does for positions held through an ex-dividend date.
    close = bars["Adj Close"] if "Adj Close" in bars.columns else bars["Close"]
    years = len(bars) / TRADING_DAYS
    total = float(close.iloc[-1] / close.iloc[0] - 1)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
    equity = close / close.iloc[0]
    dd = float(((equity.cummax() - equity) / equity.cummax()).max())
    return {
        "total_return_pct": round(total * 100, 1),
        "cagr_pct": round(float(cagr) * 100, 2),
        "maxDD_pct": round(dd * 100, 1),
        "exposure_pct": 100.0,
    }


def report(label: str, bars: pd.DataFrame, **kw) -> dict:
    trades = backtest(bars, **kw)
    s = stats(trades, bars)
    bh = buy_and_hold(bars)
    print(f"\n{label}   ({bars.index[0].date()} -> {bars.index[-1].date()})")
    print("-" * 78)
    if not s.get("n"):
        print("  no trades")
        return s
    print(f"  trades {s['n']:>4}   win {s['win_pct']:>5}%   avg {s['avg_pct']:>+7.3f}%   "
          f"pf {s['pf']:>6}   trade-t={s['t']}")
    print(f"  strategy: {s['cagr_pct']:>6.2f}% CAGR   maxDD {s['maxDD_pct']:>5.1f}%   "
          f"exposure {s['exposure_pct']:>5.1f}%   avg hold {s['avg_days']} days")
    print(f"  buy&hold: {bh['cagr_pct']:>6.2f}% CAGR   maxDD {bh['maxDD_pct']:>5.1f}%   "
          f"exposure 100.0%")
    verdict = "BEATS" if s["cagr_pct"] > bh["cagr_pct"] else "TRAILS"
    print(f"  -> {verdict} buy-and-hold on strategy CAGR")
    if s["exposure_pct"] > 0:
        eff = s["cagr_pct"] / (s["exposure_pct"] / 100)
        print(f"  -> per unit of market exposure: {eff:.2f}% vs {bh['cagr_pct']:.2f}%")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    parser.add_argument("--cost-bps", type=float, default=4.0,
                        help="round-trip execution cost in basis points (default: 4)")
    args = parser.parse_args()

    bars = pd.read_parquet(f"data_cache/{args.ticker}_daily.parquet").dropna()
    print("=" * 78)
    print(f"RSI(2) LONG-ONLY MEAN REVERSION - {args.ticker}")
    print("=" * 78)
    print(f"Rules: signal when RSI(2) < {args.entry_rsi} and close > 200-day SMA")
    print("       enter/exit at the NEXT session open after a completed close signal")
    print(f"       exit on RSI(2) > 65, close > 5-day SMA, or 10 sessions; costs {args.cost_bps:g} bp RT")
    print(f"Data: {len(bars):,} daily bars, {bars.index[0].date()} -> {bars.index[-1].date()}")

    common = {"entry_rsi": args.entry_rsi, "cost_pct": args.cost_bps / 10_000}
    report("FULL PERIOD", bars, **common)

    print("\n" + "=" * 78)
    print(f"POST-PUBLICATION HISTORICAL CHECK (~{PUBLICATION_YEAR} onward)")
    print("=" * 78)
    pre = bars[bars.index.year < PUBLICATION_YEAR]
    post = bars[bars.index.year >= PUBLICATION_YEAR]
    pre_s = report(f"PRE-{PUBLICATION_YEAR} (in-sample for the original authors)", pre, **common)
    post_s = report(f"POST-{PUBLICATION_YEAR} (post-publication historical test)", post, **common)

    print("\n" + "=" * 78)
    print("BY 5-YEAR BLOCK: is the edge still present recently?")
    print("=" * 78)
    rows = []
    for start in range(1995, 2026, 5):
        block = bars[(bars.index.year >= start) & (bars.index.year < start + 5)]
        if len(block) < 250:
            continue
        t = backtest(block, **common)
        s = stats(t, block)
        if not s.get("n"):
            continue
        bh = buy_and_hold(block)
        rows.append(
            {
                "period": f"{start}-{min(start + 4, 2026)}",
                "trades": s["n"],
                "win%": s["win_pct"],
                "avg%": s["avg_pct"],
                "pf": s["pf"],
                "cagr%": s["cagr_pct"],
                "bh_cagr%": bh["cagr_pct"],
                "maxDD%": s["maxDD_pct"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("SENSITIVITY: does it depend on an exact threshold?")
    print("=" * 78)
    print(f"{'entry RSI':>10} {'trades':>8} {'win %':>8} {'avg %':>9} {'pf':>8} {'trade-t':>7}")
    print("-" * 60)
    for threshold in (2, 3, 5, 10, 15, 20):
        t = backtest(bars, entry_rsi=threshold, cost_pct=args.cost_bps / 10_000)
        s = stats(t, bars)
        if s.get("n"):
            print(f"{threshold:>10} {s['n']:>8} {s['win_pct']:>7}% {s['avg_pct']:>+8.3f}% "
                  f"{s['pf']:>8} {s['t']:>7}")
    print("\n  Smoothness across nearby thresholds is only a diagnostic. It does not prove an")
    print("  effect is live-tradeable or protect against universe selection and regime changes.")

    print("\n" + "=" * 78)
    print("TREND FILTER: is the 200-day SMA condition doing work?")
    print("=" * 78)
    for use_filter in (True, False):
        t = backtest(bars, **common, trend_filter=use_filter)
        s = stats(t, bars)
        label = "with 200-SMA filter" if use_filter else "no trend filter"
        if s.get("n"):
            print(f"  {label:<22} n={s['n']:>4}  win {s['win_pct']}%  avg {s['avg_pct']:+.3f}%  "
                  f"pf {s['pf']}  maxDD {s['maxDD_pct']}%")


if __name__ == "__main__":
    main()
