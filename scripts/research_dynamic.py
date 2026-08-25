"""Reproduce Amendment 1: dynamic exits, volume-profile confirmation, and the null control.

    python scripts/research_dynamic.py

Prints the null-control distribution (dynamic exit on random entries) and the momentum
entries measured against it, on DEV and validation. Deterministic. Holdout is unreachable.

The finding: a dynamic exit cannot manufacture expectancy from entries with no directional
persistence, and volume-profile confirmation degrades rather than sharpens every entry.
See RESEARCH_PLAN.md §12.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.research.dataset import build_research_frame, split_frame  # noqa: E402
from tradebot.research.dynamic_exits import (  # noqa: E402
    DynamicExitRule,
    null_control,
    screen_dynamic,
)
from tradebot.research.eth_data import load_or_build  # noqa: E402
from tradebot.research.hypotheses import BY_ID  # noqa: E402
from tradebot.research.volume_profile import add_value_area, value_area_confirms  # noqa: E402

DBN = ROOT / "data_cache" / "glbx-mdp3-20170630-20260729.ohlcv-1m.dbn"
CACHE = ROOT / "data_cache" / "nq_eth_1m.parquet"
MOMENTUM = [
    "A1_opening_drive", "C1_prior_extreme_break", "D1_strong_session_pullback",
    "D2_breakout_pullback", "D3_volatility_adjusted_trend", "A4_opening_reversal",
]


def main() -> None:
    if not CACHE.exists() and not DBN.exists():
        raise SystemExit("the licensed Databento archive is required and is gitignored.")

    eth, _ = load_or_build(DBN, CACHE)
    frame = build_research_frame(eth, timeframe="5min")
    dev = split_frame(frame, "dev")
    val = split_frame(frame, "validation")
    rule = DynamicExitRule()

    print("NULL CONTROL — dynamic exit on RANDOM entries (must centre below zero)")
    print("=" * 78)
    for name, fr, seed in (("DEV", dev, 1), ("VALIDATION", val, 2)):
        nc = null_control(fr.bars, fr.atr, rule, n_trials=40, seed=seed, bar_minutes=5)
        print(f"  {name:<11} mean net {nc['mean_net']:+.3f}  95th {nc['p95']:+.3f}  "
              f"best {nc['max_net']:+.3f}  ({nc['trials']} trials)")

    dev_null = null_control(dev.bars, dev.atr, rule, n_trials=40, seed=1, bar_minutes=5)
    bar = dev_null["p95"]

    print(f"\nMOMENTUM ENTRIES + dynamic exit on DEV (must beat null 95th = {bar:+.2f})")
    print("=" * 78)
    for hid in MOMENTUM:
        result = screen_dynamic(BY_ID[hid].signal(dev.bars), dev.bars, dev.atr, rule,
                                hypothesis_id=hid, bar_minutes=5)
        beats = "  BEATS NULL" if result.net_points_per_trade > bar else ""
        t = "n/a" if result.t_stat is None else f"{result.t_stat:+.2f}"
        print(f"  {hid:<28} n={result.trades:>4}  net={result.net_points_per_trade:+6.2f}  "
              f"t={t:>6}{beats}")

    print("\nVOLUME-PROFILE CONFIRMATION + dynamic exit on DEV (a filter should sharpen)")
    print("=" * 78)
    dev_vp = add_value_area(dev.bars)
    for hid in ["A1_opening_drive", "C1_prior_extreme_break",
                "D1_strong_session_pullback", "A4_opening_reversal"]:
        signal = BY_ID[hid].signal(dev_vp)
        confirmed = value_area_confirms(dev_vp, signal)
        r0 = screen_dynamic(signal, dev_vp, dev.atr, rule, hypothesis_id=hid, bar_minutes=5)
        r1 = screen_dynamic(confirmed, dev_vp, dev.atr, rule, hypothesis_id=hid, bar_minutes=5)
        print(f"  {hid:<28} no-VP net={r0.net_points_per_trade:+6.2f} n={r0.trades:>4}   "
              f"VP net={r1.net_points_per_trade:+6.2f} n={r1.trades:>4}")

    print("\nbid/ask: NOT tested. No quote data exists over DEV/validation; the only "
          "order-flow\nproxy is holdout-only. See RESEARCH_PLAN.md §12.")


if __name__ == "__main__":
    main()
