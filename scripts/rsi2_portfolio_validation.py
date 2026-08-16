"""Validate RSI(2) as an executable, cash-constrained portfolio.

This is the companion to the per-ticker reports. It deliberately avoids treating thousands
of overlapping stock trades as independent observations: trades are filled next session's
open, capped by available portfolio slots, and evaluated as one daily return stream with a
Newey-West adjusted t-statistic.

It is historical evidence, not a profitability guarantee. The current watchlist is not a
point-in-time constituent database, so survivorship bias remains and a forward paper-trading
period is still required before risking capital.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.rsi2_strategy import (  # noqa: E402
    load_cached_bars,
    portfolio_backtest,
    portfolio_stats,
)
from scanner.universe import FULL_UNIVERSE, PRESETS, sector_of  # noqa: E402


def load_universe(name: str) -> dict[str, pd.DataFrame]:
    if name == "all":
        tickers = FULL_UNIVERSE
    elif name in PRESETS:
        tickers = PRESETS[name]
    else:
        tickers = [part.strip().upper() for part in name.split(",") if part.strip()]

    bars_by_ticker: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        bars = load_cached_bars(ticker)
        if bars is None or len(bars) < 250:
            missing.append(ticker)
        else:
            bars_by_ticker[ticker] = bars
    if missing:
        print(f"  Skipping {len(missing)} ticker(s) without usable local history: {', '.join(missing)}")
    return bars_by_ticker


def render(label: str, result) -> dict:
    stats = portfolio_stats(result)
    print(f"\n{label}")
    print("-" * 88)
    if not stats.get("n_days"):
        print("  no completed portfolio observations")
        return stats
    print(f"  candidate signals {stats['n_candidates']:,}  selected trades {stats['n_trades']:,}")
    print(f"  daily observations {stats['n_days']:,}  average active slots "
          f"{stats['avg_active_positions']:.2f}  maximum {stats['max_active_positions']}")
    print(f"  CAGR {stats['cagr_pct']:+.2f}%   total {stats['total_return_pct']:+.1f}%   "
          f"max drawdown {stats['maxDD_pct']:.1f}%")
    print(f"  annual volatility {stats['annual_vol_pct']:.2f}%   zero-rate Sharpe "
          f"{stats['sharpe_zero_rf']:.2f}")
    print(f"  mean daily return {stats['mean_daily_bp']:+.3f} bp   HAC t-stat {stats['hac_t']:.2f}")
    return stats


def run_period(label: str, bars_by_ticker: dict[str, pd.DataFrame], args, start: str, end: str | None):
    result = portfolio_backtest(
        bars_by_ticker,
        start=start,
        end=end,
        entry_rsi=args.entry_rsi,
        cost_pct=args.cost_bps / 10_000,
        max_positions=args.max_positions,
        sector_of=sector_of if args.max_per_sector else None,
        max_positions_per_sector=args.max_per_sector or None,
    )
    return render(label, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="all", help="preset, 'all', or comma-separated symbols")
    parser.add_argument("--entry-rsi", type=float, default=5.0)
    parser.add_argument("--cost-bps", type=float, default=4.0,
                        help="round-trip shares execution cost (default: 4)")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-per-sector", type=int, default=2,
                        help="sector concentration cap; use 0 to disable")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--holdout-start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    if args.cost_bps < 0:
        parser.error("--cost-bps cannot be negative")
    if args.max_positions < 1:
        parser.error("--max-positions must be at least one")
    if args.max_per_sector < 0:
        parser.error("--max-per-sector cannot be negative")

    bars_by_ticker = load_universe(args.universe)
    print("=" * 88)
    print("RSI(2) EXECUTABLE PORTFOLIO VALIDATION")
    print("=" * 88)
    print(f"  universe: {len(bars_by_ticker)} locally cached names ({args.universe})")
    print(f"  signals: RSI(2) < {args.entry_rsi:g}, close above 200-day SMA")
    print("  execution: signal/exit confirmed at close, filled next session open; returns include dividends")
    sector_text = str(args.max_per_sector) if args.max_per_sector else "disabled"
    print(f"  allocation: {args.max_positions} equal capital slots, sector cap {sector_text}, "
          f"cost {args.cost_bps:g} bp round trip")
    print("  inference: daily portfolio returns, Newey-West HAC t-stat (10 lags)")

    run_period("FULL POST-PUBLICATION HISTORY", bars_by_ticker, args, args.start, args.end)
    if pd.Timestamp(args.holdout_start) > pd.Timestamp(args.start):
        run_period("LATER HISTORICAL WINDOW (NOT A PRISTINE HOLDOUT AFTER RETIERING)",
                   bars_by_ticker, args, args.holdout_start, args.end)

    print("\nInterpretation")
    print("  Positive results here show a historical, executable-looking effect under the")
    print("  stated portfolio rules. They do not validate the selected 120 names prospectively,")
    print("  account for delisted stocks, taxes, market impact, or corporate-event gaps beyond dividends.")
    print("  Treat this as a paper-trading candidate until a locked, out-of-sample log exists.")


if __name__ == "__main__":
    main()
