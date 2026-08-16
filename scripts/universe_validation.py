"""Validate RSI(2) across the whole tradeable universe, winners and losers together.

The universe is defined by liquidity in scanner/universe.py, never by backtest results, so
this reports the aggregate honestly rather than a curated list of names that happened to work.

Everything is measured post-2009, the live-forward window for a strategy popularized around
then. Names are ranked but nothing is filtered out - the losers are shown, because a universe
selected on performance is how a scanner ends up recommending whatever already went up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))

from rsi2_validation import backtest, buy_and_hold, stats  # noqa: E402
from universe import FULL_UNIVERSE, resolve  # noqa: E402


def ensure_data(ticker: str) -> pd.DataFrame | None:
    path = Path(f"data_cache/{ticker.replace('-', '')}_daily.parquet")
    if path.exists():
        return pd.read_parquet(path).dropna()
    try:
        df = yf.download(ticker, start="1990-01-01", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(path)
        return df.dropna()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    parser.add_argument("--since", type=int, default=2009)
    parser.add_argument("--universe", default="all")
    args = parser.parse_args()

    tickers = resolve(args.universe) if args.universe != "all" else FULL_UNIVERSE
    print(f"Validating RSI(2) < {args.entry_rsi:g} on {len(tickers)} tickers "
          f"since {args.since} ...\n")

    rows = []
    for i, ticker in enumerate(tickers, 1):
        bars = ensure_data(ticker)
        if bars is None or len(bars) < 500:
            continue
        bars = bars[bars.index.year >= args.since]
        if len(bars) < 500:
            continue
        trades = backtest(bars, entry_rsi=args.entry_rsi)
        s = stats(trades, bars)
        if not s.get("n") or s["n"] < 10:
            continue
        bh = buy_and_hold(bars)
        rows.append(
            {
                "ticker": ticker,
                "n": s["n"],
                "win%": s["win_pct"],
                "avg%": s["avg_pct"],
                "pf": s["pf"],
                "trade_t": s["trade_t"],
                "maxDD%": s["maxDD_pct"],
                "expo%": s["exposure_pct"],
                "cagr%": s["cagr_pct"],
                "bh%": bh["cagr_pct"],
            }
        )
        if i % 20 == 0:
            print(f"  {i}/{len(tickers)} ...", flush=True)

    df = pd.DataFrame(rows).sort_values("trade_t", ascending=False)
    if df.empty:
        print("No results.")
        return

    print("\n" + "=" * 92)
    print(f"ALL {len(df)} NAMES, RANKED BY TRADE-LEVEL t-STATISTIC (nothing filtered out)")
    print("=" * 92)
    print(df.to_string(index=False))

    print("\n" + "=" * 92)
    print("AGGREGATE")
    print("=" * 92)
    total_trades = int(df["n"].sum())
    print(f"  names tested                {len(df)}")
    print(f"  total trades                {total_trades:,}")
    print(f"  positive average trade      {(df['avg%'] > 0).sum()}/{len(df)}")
    print(f"  trade-level t > 2           {(df['trade_t'] > 2).sum()}/{len(df)}")
    print(f"  profit factor > 1.5         {(df['pf'] > 1.5).sum()}/{len(df)}")
    print(f"  median average trade        {df['avg%'].median():+.3f}%")
    print(f"  median win rate             {df['win%'].median():.1f}%")
    print(f"  median max drawdown         {df['maxDD%'].median():.1f}%")

    etf_mask = df["ticker"].isin(resolve("etfs"))
    for label, subset in (("ETFs", df[etf_mask]), ("single names", df[~etf_mask])):
        if subset.empty:
            continue
        print(f"\n  {label}: {len(subset)} tested, "
              f"{(subset['avg%'] > 0).sum()} positive, "
              f"median {subset['avg%'].median():+.3f}%/trade, "
              f"median maxDD {subset['maxDD%'].median():.1f}%")

    print("\n  These are per-name, next-open results. The trade-level t-statistic does not")
    print("  account for cross-ticker correlation; see rsi2_portfolio_validation.py for")
    print("  the cash-constrained daily portfolio view.")
    print("\n  ETFs versus single names is the comparison that matters: single-stock results")
    print("  carry company-specific risk that no technical rule can anticipate, and the")
    print("  drawdown difference is where that shows up.")

    worst = df.nsmallest(5, "avg%")
    print(f"\n  Worst 5 by average trade (shown deliberately):")
    print(worst[["ticker", "n", "win%", "avg%", "pf", "maxDD%"]].to_string(index=False))

    out = Path("data_cache/universe_validation.csv")
    df.to_csv(out, index=False)
    print(f"\n  Full table -> {out}")


if __name__ == "__main__":
    main()
