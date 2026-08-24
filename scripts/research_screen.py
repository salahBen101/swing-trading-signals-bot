"""Reproduce the strategy research screen from RESEARCH_PLAN.

Runs the sixteen pre-registered hypotheses through the pre-registered procedure and prints
the tables recorded in STRATEGY_REGISTRY.md. Deterministic: same data in, same numbers out.

    python scripts/research_screen.py                 # DEV gross + validation screen
    python scripts/research_screen.py --robustness A4_opening_reversal

**Holdout is not reachable from this script.** By design. Nothing here evaluates the
post-2025 data; spending it requires a deliberate, separate action once a candidate has
earned it, and in this phase none did.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.research.dataset import build_research_frame, split_frame  # noqa: E402
from tradebot.research.eth_data import load_or_build  # noqa: E402
from tradebot.research.harness import ROUND_TRIP_POINTS, screen  # noqa: E402
from tradebot.research.hypotheses import BY_ID, PRE_REGISTERED  # noqa: E402

DBN = ROOT / "data_cache" / "glbx-mdp3-20170630-20260729.ohlcv-1m.dbn"
CACHE = ROOT / "data_cache" / "nq_eth_1m.parquet"


def _load(timeframe: str):
    eth, report = load_or_build(DBN, CACHE)
    if report is not None:
        print(report.summary(), "\n")
    frame = build_research_frame(eth, timeframe=timeframe)
    return frame


def screen_all(timeframe: str) -> None:
    frame = _load(timeframe)
    dev = split_frame(frame, "dev")
    val = split_frame(frame, "validation")
    print(f"DEV : {dev.describe()}")
    print(f"VAL : {val.describe()}")
    print(f"\nround-trip cost {ROUND_TRIP_POINTS:.2f} pt, gross hurdle "
          f"{4 * ROUND_TRIP_POINTS:.2f} pt\n")

    rows = []
    for hypothesis in PRE_REGISTERED:
        rd = screen(hypothesis, dev.bars, dev.atr, bar_minutes=dev.bar_minutes)
        rv = screen(hypothesis, val.bars, val.atr, bar_minutes=val.bar_minutes)
        rows.append((hypothesis, rd, rv))

    print("STAGE 1/3 — DEV gross screen and validation, sorted by DEV gross")
    print("=" * 100)
    for hypothesis, rd, rv in sorted(rows, key=lambda x: -x[1].gross_points_per_trade):
        hurdle = "HURDLE" if rd.clears_cost_hurdle else "      "
        flip = "  FLIP" if (rd.gross_points_per_trade > 0) != (
            rv.gross_points_per_trade > 0
        ) else ""
        print(f"{hurdle}  {rd.headline()}   | VAL gross={rv.gross_points_per_trade:+6.2f} "
              f"n={rv.trades:>4}{flip}")

    cleared = [h.id for h, rd, _ in rows if rd.clears_cost_hurdle]
    print(f"\ncleared the gross cost hurdle on DEV: {cleared or 'none'}")


def robustness(hypothesis_id: str, timeframe: str) -> None:
    frame = _load(timeframe)
    dev = split_frame(frame, "dev")
    hypothesis = BY_ID[hypothesis_id]
    result = screen(hypothesis, dev.bars, dev.atr, bar_minutes=dev.bar_minutes,
                    keep_trades=True)

    print(f"{hypothesis_id} — DEV robustness\n" + "=" * 60)

    total = sum(v["net_usd"] for v in result.by_year.values())
    print("\nyear by year (net USD):")
    for year in sorted(result.by_year):
        v = result.by_year[year]
        share = v["net_usd"] / total * 100 if total else 0.0
        print(f"  {year}  n={v['trades']:>3}  ${v['net_usd']:>9,.0f}  ({share:+5.0f}%)")
    positive = sum(1 for v in result.by_year.values() if v["net_usd"] > 0)
    print(f"  positive years {positive}/{len(result.by_year)}, total ${total:,.0f}")

    print(f"\ndirection:  long n={result.long_trades} gross={result.long_net_points:+.3f}"
          f"   short n={result.short_trades} gross={result.short_net_points:+.3f}")

    nets = np.array([t.gross_points for t in result.trade_list]) - ROUND_TRIP_POINTS
    k = max(1, int(len(nets) * 0.05))
    trimmed = np.sort(nets)[k:-k]
    print(f"concentration:  full mean {nets.mean():+.3f}pt  "
          f"trimmed(5%) {trimmed.mean():+.3f}pt")

    stressed = screen(hypothesis, dev.bars, dev.atr, bar_minutes=dev.bar_minutes,
                      slippage_ticks=2.0)
    print(f"cost stress:  1-tick net {result.net_points_per_trade:+.3f}pt  "
          f"2-tick net {stressed.net_points_per_trade:+.3f}pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="5min")
    parser.add_argument("--robustness", metavar="HYPOTHESIS_ID",
                        help="run the DEV robustness battery for one family")
    args = parser.parse_args()

    if not CACHE.exists() and not DBN.exists():
        raise SystemExit(
            "neither the ETH parquet cache nor the source DBN is present; this script "
            "needs the licensed Databento archive, which is gitignored."
        )

    if args.robustness:
        robustness(args.robustness, args.timeframe)
    else:
        screen_all(args.timeframe)


if __name__ == "__main__":
    main()
