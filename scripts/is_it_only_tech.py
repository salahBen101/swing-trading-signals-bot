"""Is the RSI(2) edge broad, or is it really just QQQ and the Mag 7?

A fair challenge to the universe result. Two things could produce a healthy-looking aggregate
while the claim "it works broadly" is false:

  1. Concentration. A handful of large winners (tech, which rose enormously) could carry a
     pooled average while most names contribute nothing.
  2. Correlation. Mag 7 names, QQQ, XLK and SMH overlap heavily. Counting them as independent
     evidence inflates confidence in something that is close to one bet.

So the universe is split into a TECH/QQQ-ADJACENT group and everything else, and each is
tested on its own - pooled, and on the 2019-2026 later historical window separately. If the non-tech group has
no edge of its own, the scanner should not be signalling on those names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.rsi2_strategy import backtest  # noqa: E402
from scanner.universe import COMMODITY_LINKED, FULL_UNIVERSE, MEGA_CAP_TECH  # noqa: E402

# Anything that is effectively a bet on large-cap US tech.
TECH_ADJACENT = set(MEGA_CAP_TECH) | {"QQQ", "XLK", "SMH", "NFLX", "PYPL", "UBER", "ABNB", "BKNG"}

COST_BPS = 4.0


def load(ticker: str) -> pd.DataFrame | None:
    path = Path(f"data_cache/{ticker.replace('-', '')}_daily.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path).dropna()
    return df if len(df) > 500 else None


def collect(ticker: str, entry_rsi: float = 5.0, since: int = 2009) -> pd.DataFrame:
    bars = load(ticker)
    if bars is None:
        return pd.DataFrame()
    trades = backtest(bars, entry_rsi=entry_rsi, cost_pct=COST_BPS / 10_000)
    if trades.empty:
        return pd.DataFrame()
    trades = trades[trades["signal_date"].dt.year >= since].copy()
    if trades.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": trades["signal_date"],
            "year": trades["signal_date"].dt.year,
            "ret_pct": trades["net_pct"],
        }
    )


def stats(series: pd.Series, label: str) -> dict:
    if len(series) == 0:
        return {"group": label, "n": 0}
    wins, losses = series[series > 0], series[series <= 0]
    gl = abs(losses.sum())
    return {
        "group": label,
        "n": len(series),
        "win%": round((series > 0).mean() * 100, 1),
        "avg%": round(series.mean(), 3),
        "pf": round(wins.sum() / gl, 3) if gl > 0 else np.inf,
        "trade_t": round(series.mean() / (series.std() / np.sqrt(len(series))), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    args = parser.parse_args()

    print("Collecting trades ...")
    parts = [collect(t, args.entry_rsi) for t in FULL_UNIVERSE]
    trades = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    trades["tech"] = trades["ticker"].isin(TECH_ADJACENT)
    trades["commodity"] = trades["ticker"].isin(COMMODITY_LINKED)
    print(f"  {len(trades):,} trades across {trades['ticker'].nunique()} names\n")

    tech = trades[trades["tech"]]
    other = trades[~trades["tech"]]
    other_clean = other[~other["commodity"]]

    print("=" * 78)
    print("1. TECH/QQQ-ADJACENT  vs  EVERYTHING ELSE  (pooled, 2009-2026)")
    print("=" * 78)
    print(pd.DataFrame([
        stats(trades["ret_pct"], "whole universe"),
        stats(tech["ret_pct"], "tech / QQQ-adjacent"),
        stats(other["ret_pct"], "everything else"),
        stats(other_clean["ret_pct"], "everything else, ex-commodity"),
    ]).to_string(index=False))

    print("\n" + "=" * 78)
    print("2. THE SAME SPLIT ON THE LATER HISTORICAL WINDOW (2019-2026)")
    print("=" * 78)
    hold = trades[trades["year"] >= 2019]
    ht, ho = hold[hold["tech"]], hold[~hold["tech"]]
    ho_clean = ho[~ho["commodity"]]
    print(pd.DataFrame([
        stats(hold["ret_pct"], "whole universe"),
        stats(ht["ret_pct"], "tech / QQQ-adjacent"),
        stats(ho["ret_pct"], "everything else"),
        stats(ho_clean["ret_pct"], "everything else, ex-commodity"),
    ]).to_string(index=False))

    print("\n" + "=" * 78)
    print("3. IS THE AGGREGATE CARRIED BY A FEW NAMES?")
    print("=" * 78)
    per_name = trades.groupby("ticker")["ret_pct"].agg(["count", "mean", "sum"])
    per_name = per_name.sort_values("sum", ascending=False)
    total = per_name["sum"].sum()
    top10 = per_name.head(10)["sum"].sum()
    print(f"  total pooled return across all names: {total:,.0f}%")
    print(f"  contributed by the top 10 names:      {top10:,.0f}%  ({top10 / total:.0%})")
    print(f"  names with positive total:            {(per_name['sum'] > 0).sum()}/{len(per_name)}")
    print("\n  Top 10 contributors:")
    print(per_name.head(10).round(2).to_string())

    print("\n" + "=" * 78)
    print("4. NON-TECH SECTORS, EACH ON ITS OWN (later historical window 2019-2026)")
    print("=" * 78)
    from scanner.universe import (  # noqa: E402
        CONSUMER_INDUSTRIAL,
        ENERGY_MATERIALS_UTILITIES,
        ETFS,
        FINANCIALS,
        HEALTHCARE,
    )

    groups = {
        "financials": FINANCIALS,
        "healthcare": HEALTHCARE,
        "consumer/industrial": CONSUMER_INDUSTRIAL,
        "energy/materials/utils": ENERGY_MATERIALS_UTILITIES,
        "ETFs (all)": ETFS,
        "ETFs ex-QQQ/XLK/SMH": [e for e in ETFS if e not in {"QQQ", "XLK", "SMH"}],
    }
    rows = []
    for label, members in groups.items():
        subset = hold[hold["ticker"].isin(members)]
        if len(subset):
            rows.append(stats(subset["ret_pct"], label))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if len(ho_clean) and ho_clean["ret_pct"].mean() > 0:
        t_other = ho_clean["ret_pct"].mean() / (ho_clean["ret_pct"].std() / np.sqrt(len(ho_clean)))
        print(f"  Non-tech, ex-commodity, LATER WINDOW ONLY: {len(ho_clean):,} trades, "
              f"{ho_clean['ret_pct'].mean():+.3f}% per trade, trade-t={t_other:.2f}")
        if t_other > 3:
            print("  That is a positive trade-level result, but not independent evidence: signals")
            print("  overlap, share market exposure, and this later period informed the tier labels.")
        elif t_other > 2:
            print("  Present but weaker than the tech group. Broad, but not uniformly compelling.")
        else:
            print("  NOT significant on its own. The broad-universe claim does not hold and")
            print("  the scanner should be narrowed.")


if __name__ == "__main__":
    main()
