"""Historical check for the daily volatility-compression breakout watch.

The scanner can see a completed daily bar only after the close.  This report therefore
records a signal at that close, enters at the next session's dividend-adjusted open, and
marks a fixed-horizon exit at a later open.  It compares a confirmed compression breakout
with the same trend/volume/Donchian breakout *without* recent compression, so the squeeze
filter has to add information rather than merely inherit ordinary momentum.

This is a reproducible research check, not a probability estimate or a claim of live
profitability.  The 120-name cache is a present-day survivor universe, events overlap, and
the date blocks do not undo selection or implementation bias.  Do not tune the parameters
after inspecting this report; forward paper trading is the next evidence-generating step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.rsi2_strategy import load_cached_bars, total_return_open  # noqa: E402
from scanner.squeeze_strategy import DEFAULT_CONFIG, build_squeeze_features  # noqa: E402
from scanner.universe import FULL_UNIVERSE, PRESETS  # noqa: E402


def newey_west_t(values: pd.Series, lags: int) -> float:
    """Dependency-free HAC t-statistic for correlated date-level basket returns."""

    x = values.dropna().to_numpy(dtype=float)
    if len(x) < 3:
        return float("nan")
    lag_count = min(max(lags, 0), len(x) - 1)
    centered = x - x.mean()
    long_run_variance = float(np.dot(centered, centered) / len(x))
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / len(x))
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * covariance
    return float(x.mean() / np.sqrt(long_run_variance / len(x))) if long_run_variance > 0 else float("nan")


def universe_from_arg(name: str) -> list[str]:
    if name == "all":
        return list(FULL_UNIVERSE)
    if name in PRESETS:
        return list(PRESETS[name])
    return [part.strip().upper() for part in name.split(",") if part.strip()]


def signal_starts(mask: pd.Series) -> pd.Series:
    """One signal per contiguous breakout; avoids counting a sustained break repeatedly."""

    flags = mask.fillna(False).astype(bool)
    return flags & ~flags.shift(1, fill_value=False)


def events_from_mask(
    ticker: str,
    bars: pd.DataFrame,
    mask: pd.Series,
    *,
    hold_days: int,
    cost_pct: float,
    kind: str,
) -> pd.DataFrame:
    """Create executable next-open fixed-horizon events from close-known signals."""

    return_open = total_return_open(bars).reindex(mask.index)
    rows: list[dict] = []
    for signal_index in np.flatnonzero(mask.to_numpy(dtype=bool)):
        entry_index = signal_index + 1
        exit_index = entry_index + hold_days
        if exit_index >= len(mask):
            continue
        entry = float(return_open.iloc[entry_index])
        exit_ = float(return_open.iloc[exit_index])
        if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0 and exit_ > 0):
            continue
        gross = exit_ / entry - 1
        rows.append(
            {
                "ticker": ticker,
                "kind": kind,
                "signal_date": mask.index[signal_index],
                "entry_date": mask.index[entry_index],
                "exit_date": mask.index[exit_index],
                "gross_return": gross,
                "net_return": gross - cost_pct,
            }
        )
    return pd.DataFrame(rows)


def collect_events(
    tickers: list[str], *, hold_days: int, cost_pct: float
) -> tuple[pd.DataFrame, list[str]]:
    """Load cached daily data and produce compression and no-compression comparisons."""

    events: list[pd.DataFrame] = []
    missing: list[str] = []
    for ticker in tickers:
        bars = load_cached_bars(ticker)
        if bars is None or not {"High", "Low", "Close", "Volume"}.issubset(bars.columns):
            missing.append(ticker)
            continue
        features = build_squeeze_features(bars)
        if features.empty:
            missing.append(ticker)
            continue

        # The identical trend, volume, and prior-20-day close breakout conditions define
        # both groups.  Only prior compression differs.
        momentum_breakout = (
            features["above_trend"]
            & features["close_above_breakout"]
            & features["volume_confirmed"]
        )
        compressed_breakout = features["confirmed"]
        ordinary_breakout = signal_starts(momentum_breakout & ~features["prior_compressed"])
        events.append(
            events_from_mask(
                ticker, bars, compressed_breakout,
                hold_days=hold_days, cost_pct=cost_pct, kind="compression",
            )
        )
        events.append(
            events_from_mask(
                ticker, bars, ordinary_breakout,
                hold_days=hold_days, cost_pct=cost_pct, kind="no_compression",
            )
        )
    usable = [frame for frame in events if not frame.empty]
    columns = ["ticker", "kind", "signal_date", "entry_date", "exit_date", "gross_return", "net_return"]
    return (pd.concat(usable, ignore_index=True) if usable else pd.DataFrame(columns=columns), missing)


def summarize(events: pd.DataFrame, *, hold_days: int) -> dict[str, float | int]:
    if events.empty:
        return {"n": 0}
    returns = events["net_return"].astype(float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    loss_total = abs(float(losses.sum()))
    by_entry_date = returns.groupby(events["entry_date"]).mean().sort_index()
    return {
        "n": int(len(events)),
        "dates": int(len(by_entry_date)),
        "tickers": int(events["ticker"].nunique()),
        "win_pct": float((returns > 0).mean() * 100),
        "mean_bp": float(returns.mean() * 10_000),
        "median_bp": float(returns.median() * 10_000),
        "pf": float(wins.sum() / loss_total) if loss_total > 0 else float("inf"),
        "basket_hac_t": newey_west_t(by_entry_date, lags=hold_days),
    }


def print_summary(label: str, events: pd.DataFrame, *, hold_days: int) -> None:
    stats = summarize(events, hold_days=hold_days)
    print(f"\n  {label}")
    print("  " + "-" * 80)
    if not stats.get("n"):
        print("  no mature signals")
        return
    print(f"  signals {stats['n']:,} across {stats['dates']:,} entry dates / {stats['tickers']} tickers")
    print(f"  net {hold_days}-session return: mean {stats['mean_bp']:+.1f} bp, "
          f"median {stats['median_bp']:+.1f} bp, win rate {stats['win_pct']:.1f}%, PF {stats['pf']:.2f}")
    print(f"  date-basket Newey-West t-stat (lag {hold_days}): {stats['basket_hac_t']:.2f}")


def run_window(label: str, events: pd.DataFrame, start: str, end: str | None, *, hold_days: int) -> None:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else None
    window = events[events["signal_date"] >= start_ts]
    if end_ts is not None:
        window = window[window["signal_date"] <= end_ts]
    print("\n" + "=" * 88)
    print(label)
    print("=" * 88)
    print_summary("CONFIRMED COMPRESSION BREAKOUT", window[window["kind"] == "compression"], hold_days=hold_days)
    print_summary("SAME BREAKOUT CONDITIONS, NO RECENT COMPRESSION", window[window["kind"] == "no_compression"], hold_days=hold_days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="all", help="preset, 'all', or comma-separated symbols")
    parser.add_argument("--cost-bps", type=float, default=20.0,
                        help="round-trip execution cost in basis points (default: 20)")
    parser.add_argument("--hold-days", type=int, default=10,
                        help="fixed exit horizon after the next-open entry (default: 10)")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--dev-end", default="2018-12-31")
    parser.add_argument("--validation-end", default="2024-12-31")
    args = parser.parse_args()
    if args.cost_bps < 0:
        parser.error("--cost-bps cannot be negative")
    if args.hold_days < 1:
        parser.error("--hold-days must be at least one")

    config = DEFAULT_CONFIG
    print("=" * 88)
    print("DAILY VOLATILITY-COMPRESSION BREAKOUT CHECK")
    print("=" * 88)
    print(f"  universe: {args.universe}; historical present-day survivor universe")
    print(f"  compression: BB({config.band_period},{config.band_stddev:g}) inside "
          f"KC({config.keltner_period},{config.keltner_atr_multiple:g} ATR), {config.min_squeeze_days}+ days,")
    print(f"               BB width <= {config.max_bandwidth_percentile:.0%} of prior "
          f"{config.bandwidth_lookback} sessions")
    print(f"  trigger: prior-{config.breakout_period}-day high close + {config.min_volume_ratio:.1f}x "
          f"prior-{config.volume_period}-day volume + above {config.trend_period}-day SMA")
    print(f"  execution: next-open entry, fixed {args.hold_days}-session open exit, "
          f"{args.cost_bps:g} bp round-trip cost")
    print("  comparison: identical trend/volume breakouts with no qualifying recent compression")

    events, missing = collect_events(
        universe_from_arg(args.universe), hold_days=args.hold_days, cost_pct=args.cost_bps / 10_000,
    )
    if missing:
        print(f"  skipped {len(missing)} name(s) without usable daily OHLCV: {', '.join(missing[:10])}")

    run_window("DEVELOPMENT WINDOW", events, args.start, args.dev_end, hold_days=args.hold_days)
    validation_start = (pd.Timestamp(args.dev_end) + pd.Timedelta(days=1)).date().isoformat()
    run_window("LATER HISTORICAL WINDOW", events, validation_start, args.validation_end, hold_days=args.hold_days)
    final_start = (pd.Timestamp(args.validation_end) + pd.Timedelta(days=1)).date().isoformat()
    run_window("FINAL DATE BLOCK (REPORT ONLY; NOT PROSPECTIVE LIVE EVIDENCE)",
               events, final_start, None, hold_days=args.hold_days)

    print("\nInterpretation")
    print("  A compression setup is not a short-squeeze probability. This data lacks short interest,")
    print("  borrow, options-position, earnings, spreads, and delisted constituents. Date-level HAC")
    print("  statistics reduce—but do not remove—correlation from simultaneous/overlapping signals.")
    print("  Keep the rules frozen and record every scanner candidate in a forward paper-trading log")
    print("  before enabling an alert, position, or automated execution workflow.")


if __name__ == "__main__":
    main()
