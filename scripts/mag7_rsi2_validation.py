"""Does the RSI(2) edge survive on individual Mag 7 stocks, or only on the index?

This matters before building a scanner that signals on single names. Index mean reversion
works because broad selloffs are usually liquidity- or sentiment-driven and revert. A single
stock can fall on an earnings miss or a guidance cut - information that does not revert, and
where "oversold" simply means the market repriced it.

So the edge cannot be assumed to transfer. It is tested per name, and against the same
buy-and-hold benchmark, with the post-2009 period reported separately since that is the
genuine live-forward window for a strategy popularized around then.
"""

from __future__ import annotations

import argparse

import pandas as pd

from rsi2_validation import backtest, buy_and_hold, stats  # noqa: E402

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def evaluate(ticker: str, entry_rsi: float, since: int | None = None) -> dict | None:
    bars = pd.read_parquet(f"data_cache/{ticker}_daily.parquet").dropna()
    if since:
        bars = bars[bars.index.year >= since]
    if len(bars) < 500:
        return None

    trades = backtest(bars, entry_rsi=entry_rsi)
    s = stats(trades, bars)
    if not s.get("n"):
        return None
    bh = buy_and_hold(bars)
    return {
        "ticker": ticker,
        "n": s["n"],
        "win%": s["win_pct"],
        "avg%": s["avg_pct"],
        "pf": s["pf"],
        "t": s["t"],
        "maxDD%": s["maxDD_pct"],
        "expo%": s["exposure_pct"],
        "cagr%": s["cagr_pct"],
        "bh_cagr%": bh["cagr_pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 86)
    print(f"RSI(2) < {args.entry_rsi} LONG-ONLY, ABOVE 200-DAY SMA - individual Mag 7 names")
    print("=" * 86)

    print("\nFULL AVAILABLE HISTORY PER NAME")
    print("-" * 86)
    rows = [r for t in MAG7 if (r := evaluate(t, args.entry_rsi))]
    full = pd.DataFrame(rows)
    print(full.to_string(index=False))

    print("\nPOST-2009 ONLY (live-forward window)")
    print("-" * 86)
    rows_post = [r for t in MAG7 if (r := evaluate(t, args.entry_rsi, since=2009))]
    post = pd.DataFrame(rows_post)
    print(post.to_string(index=False))

    print("\n" + "=" * 86)
    print("COMPARISON WITH THE INDEX")
    print("=" * 86)
    for index_ticker in ("SPY", "QQQ"):
        r = evaluate(index_ticker, args.entry_rsi, since=2009)
        if r:
            print(f"  {index_ticker:<6} n={r['n']:>4}  win {r['win%']:>5}%  avg {r['avg%']:>+7.3f}%  "
                  f"pf {r['pf']:>6}  t={r['t']:>5}  maxDD {r['maxDD%']:>5}%")

    print("\n" + "=" * 86)
    print("ASSESSMENT")
    print("=" * 86)
    if not post.empty:
        positive = post[post["avg%"] > 0]
        strong = post[(post["avg%"] > 0) & (post["t"] > 2)]
        print(f"  Names with positive average trade (post-2009): {len(positive)}/{len(post)}")
        print(f"  Names with t > 2:                              {len(strong)}/{len(post)}")
        print(f"  Median average trade across names: {post['avg%'].median():+.3f}%")
        print(f"  Median max drawdown:               {post['maxDD%'].median():.1f}%")
        print("\n  Single-name drawdowns are the thing to watch. An index that falls 5% usually")
        print("  bounces; a single stock that falls 15% on guidance often does not, and the")
        print("  strategy has no rule that distinguishes the two cases.")


if __name__ == "__main__":
    main()
