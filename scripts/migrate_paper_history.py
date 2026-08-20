"""Recover the trade history from paper_options_state.json into the $50k paper account.

The account file was created fresh today and therefore started empty, while the earlier tracker
had already recorded four days of real decisions - three closed winners and an open AVGO
position. Those are genuine recorded trades at genuine quotes; they should not be thrown away
just because the program around them was rebuilt.

The migration preserves each trade EXACTLY as it was executed, including the original contract
count. It does not retro-size old trades to the larger account - that would be inventing a
history that never happened. Only trades taken from here on are sized against $50,000.

Cash is rebuilt by replaying the actual flows in date order: subtract the cost when a position
opened, add the proceeds when it closed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "output" / "paper_options_state.json"
NEW = ROOT / "output" / "paper_account.json"
START_EQUITY = 50_000.0

# Fields the new dataclass has that the old one lacks, and vice versa.
DROP = {"entry_oi"}


def main() -> None:
    if not OLD.exists():
        raise SystemExit(f"nothing to migrate: {OLD} not found")
    old = json.loads(OLD.read_text())

    if NEW.exists():
        backup = NEW.with_suffix(".json.bak")
        shutil.copy2(NEW, backup)
        print(f"backed up existing account -> {backup.name}")

    positions, cash = [], START_EQUITY
    for src in sorted(old.get("positions", []), key=lambda x: x["opened"]):
        p = {k: v for k, v in src.items() if k not in DROP}
        n = int(p.get("contracts", 1))
        p["cost"] = round(float(p["entry_ask"]) * 100 * n, 2)
        p["proceeds"] = (round(float(p["exit_bid"]) * 100 * n, 2)
                         if p.get("exit_bid") is not None and p["status"] == "CLOSED" else 0.0)
        p.setdefault("feat", {})
        p.setdefault("ml_score", None)
        p.setdefault("stale", False)
        p.setdefault("marks", [])
        positions.append(p)
        cash -= p["cost"]
        cash += p["proceeds"]

    closed = [p for p in positions if p["status"] == "CLOSED"]
    openp = [p for p in positions if p["status"] == "OPEN"]
    held = sum(p["last_mid"] * 100 * int(p.get("contracts", 1)) for p in openp)

    state = {
        "started": old.get("started", "2026-08-15"),
        "start_equity": START_EQUITY,
        "cash": round(cash, 2),
        "positions": positions,
        "equity": [{"date": max(p.get("exit_date") or p["opened"] for p in positions),
                    "equity": round(cash + held, 2), "cash": round(cash, 2),
                    "open": len(openp),
                    "realised": round(sum(p["proceeds"] - p["cost"] for p in closed), 2)}],
        "missed": old.get("missed", []),
        "migrated_from": OLD.name,
    }
    NEW.write_text(json.dumps(state, indent=2, default=str))

    print(f"\nrecovered {len(positions)} positions ({len(closed)} closed, {len(openp)} open)")
    print(f"{'':6}{'ticker':<7}{'contract':<22}{'opened':<12}{'in':>8}{'out':>8}{'P&L':>10}")
    print("  " + "-" * 74)
    for p in positions:
        n = int(p.get("contracts", 1))
        pnl = ((p["proceeds"] - p["cost"]) if p["status"] == "CLOSED"
               else p["last_mid"] * 100 * n - p["cost"])
        out = f"{p['exit_bid']:.2f}" if p.get("exit_bid") is not None else "-"
        print(f"  {p['status']:<6}{p['ticker']:<7}"
              f"{f'{p['strike']:g}C {p['expiry']}':<22}{p['opened']:<12}"
              f"{p['entry_ask']:>8.2f}{out:>8}{pnl:>+10,.0f}")

    realised = sum(p["proceeds"] - p["cost"] for p in closed)
    unreal = sum(p["last_mid"] * 100 * int(p.get("contracts", 1)) - p["cost"] for p in openp)
    print(f"\n  realised   ${realised:+,.2f}")
    print(f"  unrealised ${unreal:+,.2f}")
    print(f"  cash       ${cash:,.2f}")
    print(f"  equity     ${cash + held:,.2f}  ({(cash + held) / START_EQUITY - 1:+.2%})")
    if closed:
        w = [p for p in closed if p["proceeds"] > p["cost"]]
        print(f"  closed win rate {len(w)}/{len(closed)} = {len(w) / len(closed) * 100:.0f}%")
    print(f"\nwritten -> {NEW}")


if __name__ == "__main__":
    main()
