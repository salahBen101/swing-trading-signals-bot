"""Does raising the exit from RSI(2) > 65 to > 70 actually improve returns?

A higher exit threshold means HOLDING LONGER - waiting for a deeper overbought reading before
selling. That is a real trade-off, not a free improvement:

    holding longer  ->  captures more of the bounce when the bounce continues
    holding longer  ->  gives back more when it rolls over, and ties up a position slot,
                        which on a 5-slot account means missing other signals

So it cannot be settled by reasoning. It is measured here across SEVERAL SEPARATE PERIODS rather
than one, because a threshold that only wins on the most recent two years is a fitted number, not
a better rule. If 70 beats 65 in one window and loses in the others, the honest answer is that
the difference is noise and 65 should stay.

The 5-day-SMA exit and the 10-session stop are unchanged throughout - only the RSI level moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scanner"))

from scanner.portfolio_sim import Config, load_universe, simulate, stats  # noqa: E402
from scanner.universe import PRESETS  # noqa: E402

THRESHOLDS = [60, 65, 70, 75, 80]


def main() -> None:
    names = sorted(set(PRESETS["strong"]))
    print(f"loading {len(names)} names ...")
    data = load_universe(names)
    ref = data["SPY"] if "SPY" in data else data.get("QQQ")
    last = max(df.index[-1] for df in data.values())
    print(f"  usable {len(data)}, data to {last.date()}\n")

    # Separate, non-overlapping windows. A rule that only works in one of them is not a rule.
    windows = [
        ("2018-2020 (incl. COVID)", pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31")),
        ("2021-2022 (bear)", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
        ("2023-2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
        ("2025-now", pd.Timestamp("2025-01-01"), last),
    ]

    rows = []
    for wname, ws, we in windows:
        for thr in THRESHOLDS:
            cfg = Config(start_equity=20_000.0, exit_rsi=thr)
            res = simulate(data, cfg, ws, we, ref)
            s = stats(res, cfg)
            if not s or s["trades"] < 20:
                continue
            rows.append({"window": wname, "exit_rsi": thr, "final": s["final"],
                         "total_pct": s["total_pct"], "cagr": s["cagr"],
                         "max_dd": s["max_dd"], "sharpe": s["sharpe"],
                         "trades": s["trades"], "win": s["win_rate"],
                         "avg_pct": s["avg_pct"], "avg_days": s["avg_days"]})
        print(f"  done {wname}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("EXIT THRESHOLD, WINDOW BY WINDOW  ($20,000 start, 5 positions, 22% each)")
    print("=" * 100)
    for w in df["window"].unique():
        sub = df[df["window"] == w]
        print(f"\n  {w}")
        print(f"  {'exit':>6} {'final $':>11} {'total %':>9} {'CAGR %':>8} {'maxDD':>8} "
              f"{'Sharpe':>7} {'trades':>7} {'win %':>7} {'avg %':>7} {'days':>6}")
        print("  " + "-" * 88)
        best = sub["final"].max()
        for _, r in sub.iterrows():
            mark = "  <-- best" if r["final"] == best else ""
            print(f"  {int(r['exit_rsi']):>6} {r['final']:>11,.0f} {r['total_pct']:>9.1f} "
                  f"{r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['sharpe']:>7.2f} "
                  f"{int(r['trades']):>7} {r['win']:>7.1f} {r['avg_pct']:>7.2f} "
                  f"{r['avg_days']:>6.1f}{mark}")

    print("\n" + "=" * 100)
    print("VERDICT — how often does each threshold win, across independent windows?")
    print("=" * 100)
    wins = df.loc[df.groupby("window")["final"].idxmax()]
    tally = wins["exit_rsi"].value_counts().to_dict()
    n_win = df["window"].nunique()
    for thr in THRESHOLDS:
        bar = "#" * tally.get(thr, 0)
        print(f"  exit RSI > {thr:<3} best in {tally.get(thr, 0)}/{n_win} windows  {bar}")

    piv = df.pivot_table(index="exit_rsi", values=["cagr", "sharpe", "avg_pct", "avg_days",
                                                   "max_dd", "trades"], aggfunc="mean")
    print("\n  Averaged across all windows:")
    print(piv.round(2).to_string())

    c65 = df[df["exit_rsi"] == 65]["cagr"].mean()
    c70 = df[df["exit_rsi"] == 70]["cagr"].mean()
    print(f"\n  65 -> mean CAGR {c65:+.2f}%   |   70 -> mean CAGR {c70:+.2f}%   "
          f"|   difference {c70 - c65:+.2f} points")
    n70 = tally.get(70, 0)
    if c70 > c65 and n70 >= n_win - 1:
        print("  70 is better and it is CONSISTENT across windows. Worth adopting.")
    elif c70 > c65:
        print("  70 looks better on average but does NOT win consistently across windows.")
        print("  That pattern is what an overfitted threshold looks like. Keep 65.")
    else:
        print("  70 does not beat 65. Keep 65.")

    out = ROOT / "output" / "exit_rsi_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n  written -> {out}")


if __name__ == "__main__":
    main()
