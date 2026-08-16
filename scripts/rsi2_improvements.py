"""Can RSI(2) be improved without overfitting it?

The reason this strategy survived when six others did not is that it was never tuned. Its
threshold curve degrades smoothly from RSI<2 to RSI<20, which is what a real effect looks
like. Scanning parameters for the best backtest would reliably produce something that scores
higher and works worse.

So only changes with an ex-ante economic argument are tested, each stated BEFORE its result is
seen, and each fitted on 2009-2018 then inspected once on the later 2019-2026 window:

  H1  Broad-market regime filter. Single-name mean reversion assumes a dip in an otherwise
      healthy market. When the whole market is below its own 200-day average, a falling stock
      is more likely repricing with everything else than bouncing. Trade only when SPY is
      above its 200-day average.

  H2  Exclude commodity-linked names. Energy and materials stocks track their underlying
      commodity rather than mean-reverting with equities, so "oversold" there reflects a
      commodity downtrend. This one is partly post-hoc - the pattern was noticed in the
      universe results - so it needs especially cautious interpretation.

  H3  Cap concurrent positions. This does not touch the signal at all and therefore cannot
      overfit it; it is pure risk management. Signals cluster on market-wide down days, so an
      uncapped version can be fully loaded into one correlated event.

Honest limitation: aggregate statistics across all years have already been looked at in this
project, so 2019-2026 is not a pristine holdout. Results are exploratory until a new locked
forward period is collected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.rsi2_strategy import backtest  # noqa: E402
from scanner.universe import ETFS, FULL_UNIVERSE  # noqa: E402

COMMODITY_LINKED = {
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "XLE",  # energy
    "FCX", "NEM", "LIN", "SHW", "XLB",  # materials
}

DEV_END = 2019  # fit below this year, test from it


def load(ticker: str) -> pd.DataFrame | None:
    path = Path(f"data_cache/{ticker.replace('-', '')}_daily.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path).dropna()
    return df if len(df) > 500 else None


def collect_trades(ticker: str, market_ok: pd.Series | None, entry_rsi: float = 5.0) -> pd.DataFrame:
    """Executable next-open trades for one ticker, tagged for portfolio analysis."""
    bars = load(ticker)
    if bars is None:
        return pd.DataFrame()
    trades = backtest(
        bars,
        entry_rsi=entry_rsi,
        cost_pct=0.0004,
        entry_filter=market_ok,
    )
    if trades.empty:
        return pd.DataFrame()
    trades = trades[trades["signal_date"].dt.year >= 2009].copy()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "entry_date": trades["entry_date"],
            "exit_date": trades["exit_date"],
            "year": trades["signal_date"].dt.year,
            "days": trades["days"],
            "ret_pct": trades["net_pct"],
        }
    )


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n": 0}
    r = trades["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    gl = abs(losses.sum())
    return {
        "label": label,
        "n": len(r),
        "win%": round((r > 0).mean() * 100, 1),
        "avg%": round(r.mean(), 3),
        "pf": round(wins.sum() / gl, 3) if gl > 0 else np.inf,
        "trade_t": round(r.mean() / (r.std() / np.sqrt(len(r))), 2) if r.std() > 0 else 0.0,
        "total%": round(r.sum(), 1),
    }


def apply_position_cap(trades: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep at most `cap` positions open at once, first-come on the entry date."""
    if trades.empty:
        return trades
    ordered = trades.sort_values(["entry_date", "ticker"]).copy()
    kept = []
    open_until: list[pd.Timestamp] = []
    for _, row in ordered.iterrows():
        open_until = [d for d in open_until if d > row["entry_date"]]
        if len(open_until) < cap:
            kept.append(row)
            open_until.append(row["exit_date"])
    return pd.DataFrame(kept)


def show(rows: list[dict]) -> None:
    df = pd.DataFrame([r for r in rows if r.get("n")])
    if not df.empty:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    args = parser.parse_args()

    spy = load("SPY")
    if spy is None:
        print("SPY data required.")
        return
    market_ok = spy["Close"] > spy["Close"].rolling(200).mean()

    print("Collecting baseline trades across the universe ...")
    baseline_parts, filtered_parts = [], []
    for ticker in FULL_UNIVERSE:
        baseline_parts.append(collect_trades(ticker, None, args.entry_rsi))
        filtered_parts.append(collect_trades(ticker, market_ok, args.entry_rsi))

    baseline = pd.concat([p for p in baseline_parts if not p.empty], ignore_index=True)
    market_filtered = pd.concat([p for p in filtered_parts if not p.empty], ignore_index=True)
    baseline = baseline[baseline["year"] >= 2009]
    market_filtered = market_filtered[market_filtered["year"] >= 2009]

    dev = baseline[baseline["year"] < DEV_END]
    hold = baseline[baseline["year"] >= DEV_END]

    print(f"\n{'=' * 78}\nBASELINE\n{'=' * 78}")
    show([summarize(dev, f"DEV 2009-{DEV_END - 1}"), summarize(hold, f"LATER {DEV_END}-2026")])

    print(f"\n{'=' * 78}\nH1: only trade when SPY is above its own 200-day average\n{'=' * 78}")
    mf_dev = market_filtered[market_filtered["year"] < DEV_END]
    mf_hold = market_filtered[market_filtered["year"] >= DEV_END]
    show(
        [
            summarize(dev, "baseline  DEV"),
            summarize(mf_dev, "H1        DEV"),
            summarize(hold, "baseline  LATER"),
            summarize(mf_hold, "H1        LATER"),
        ]
    )
    if not mf_dev.empty and not dev.empty:
        print(f"\n  DEV:     {dev['ret_pct'].mean():+.3f}% -> {mf_dev['ret_pct'].mean():+.3f}% "
              f"per trade, {len(dev)} -> {len(mf_dev)} trades ({len(mf_dev) / len(dev) - 1:+.0%})")
        print(f"  LATER:   {hold['ret_pct'].mean():+.3f}% -> {mf_hold['ret_pct'].mean():+.3f}% "
              f"per trade, {len(hold)} -> {len(mf_hold)} trades ({len(mf_hold) / len(hold) - 1:+.0%})")

    print(f"\n{'=' * 78}\nH2: exclude commodity-linked names\n{'=' * 78}")
    no_comm_dev = dev[~dev["ticker"].isin(COMMODITY_LINKED)]
    no_comm_hold = hold[~hold["ticker"].isin(COMMODITY_LINKED)]
    show(
        [
            summarize(dev, "baseline  DEV"),
            summarize(no_comm_dev, "H2        DEV"),
            summarize(hold, "baseline  LATER"),
            summarize(no_comm_hold, "H2        LATER"),
        ]
    )

    print(f"\n{'=' * 78}\nH3: cap concurrent positions (risk only, cannot overfit the signal)\n{'=' * 78}")
    print("Signals cluster on market-wide down days. Without a cap the strategy can be fully")
    print("loaded into a single correlated event.\n")
    daily_counts = baseline.groupby("entry_date").size()
    print(f"  median signals per signal-day: {daily_counts.median():.0f}")
    print(f"  busiest day: {daily_counts.max():.0f} simultaneous signals "
          f"on {daily_counts.idxmax().date()}")
    print(f"  days with >5 signals: {(daily_counts > 5).sum()} of {len(daily_counts)}\n")
    rows = [summarize(baseline, "uncapped")]
    for cap in (3, 5, 10):
        rows.append(summarize(apply_position_cap(baseline, cap), f"cap {cap}"))
    show(rows)

    print(f"\n{'=' * 78}\nCOMBINED (H1 + H2), inspected on the later window\n{'=' * 78}")
    combined = market_filtered[~market_filtered["ticker"].isin(COMMODITY_LINKED)]
    show(
        [
            summarize(baseline[baseline["year"] >= DEV_END], "baseline  LATER"),
            summarize(combined[combined["year"] >= DEV_END], "H1+H2     LATER"),
        ]
    )

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print("  An apparent improvement that helps on DEV and fades in the later window is weak")
    print("  evidence. Even one that survives remains exploratory because the later window is")
    print("  no longer untouched; do not retune it further without new forward data.")


if __name__ == "__main__":
    main()
