"""RSI(2) pullback scanner - Mag 7.

Reports what the indicator currently says for each name. It places no orders and makes no
recommendation; every decision stays with the person reading the output.

Timing note, which matters if you check at 15:30 ET:
    RSI(2) is a daily-bar indicator, and the daily bar is not final until 16:00. Before the
    close this scanner computes a PROVISIONAL reading from the current price, then reports
    the closing price at which the signal would flip. That flip level is the useful part -
    it tells you whether a signal is comfortably established or hanging on half a percent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# Support both ``python scanner/rsi2_scanner.py`` and
# ``python -m scanner.rsi2_scanner``.  Do not use a broad ImportError fallback here: an
# import failure *inside* one of these modules should be visible rather than being mistaken
# for the wrong invocation mode.
if __package__:
    from .market_context import ContextGate, DEFAULT_CONTEXT_DIR, evaluate_context, load_context
    from .notify import notify
    from .options_fit import fetch_candidates, render_options
    from .squeeze_strategy import SqueezeSignal, latest_squeeze_signal
    from .universe import COMMODITY_LINKED, PRESETS, is_core, resolve, sector_of, tier_of
else:
    from market_context import ContextGate, DEFAULT_CONTEXT_DIR, evaluate_context, load_context
    from notify import notify
    from options_fit import fetch_candidates, render_options
    from squeeze_strategy import SqueezeSignal, latest_squeeze_signal
    from universe import COMMODITY_LINKED, PRESETS, is_core, resolve, sector_of, tier_of

MAG7 = PRESETS["mag7"]
MARKET_TZ = ZoneInfo("America/New_York")

# Validated defaults. See scripts/rsi2_validation.py and scripts/mag7_rsi2_validation.py.
ENTRY_RSI = 5.0
EXIT_RSI = 65.0
TREND_PERIOD = 200


def wilder_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    boundary = pd.Series(np.where(avg_gain > 0, 100.0, 50.0), index=close.index)
    return out.where(avg_loss != 0, boundary)


def rsi_for_hypothetical_close(history: pd.Series, candidate_close: float, period: int = 2) -> float:
    """RSI(2) as it would read if today closed at `candidate_close`."""
    series = pd.concat([history, pd.Series([candidate_close], index=[pd.Timestamp("2099-01-01")])])
    return float(wilder_rsi(series, period).iloc[-1])


def find_flip_price(history: pd.Series, current: float, target_rsi: float) -> float | None:
    """Closing price at which RSI(2) would cross `target_rsi`. Binary search, since RSI is
    monotone in today's close when the prior history is fixed."""
    lo, hi = current * 0.80, current * 1.20
    if rsi_for_hypothetical_close(history, lo) > target_rsi:
        return None
    if rsi_for_hypothetical_close(history, hi) < target_rsi:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if rsi_for_hypothetical_close(history, mid) < target_rsi:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def latest_expected_weekday(as_of: date) -> date:
    """Conservative expected daily-bar date without an exchange-calendar dependency.

    On a market holiday this intentionally expects a bar that will not exist and therefore
    fails closed for alerts rather than treating Friday's price as current Monday data.
    """
    expected = as_of
    while expected.weekday() >= 5:
        expected = expected.fromordinal(expected.toordinal() - 1)
    return expected


@dataclass
class Reading:
    ticker: str
    price: float
    prev_close: float
    rsi2: float
    sma200: float
    sma5: float
    above_trend: bool
    signal: bool
    flip_price: float | None
    stale: bool
    error: str = ""
    squeeze: SqueezeSignal | None = None
    squeeze_context: ContextGate | None = None

    @property
    def pct_from_prev(self) -> float:
        return (self.price / self.prev_close - 1) * 100 if self.prev_close else 0.0

    @property
    def pct_to_flip(self) -> float | None:
        if self.flip_price is None:
            return None
        return (self.flip_price / self.price - 1) * 100


def scan_ticker(
    ticker: str,
    entry_rsi: float = ENTRY_RSI,
    *,
    as_of: date | None = None,
) -> Reading:
    try:
        raw = yf.download(ticker, period="2y", progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            raise ValueError("no data returned")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close = raw["Close"].dropna()
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        history = close.iloc[:-1]  # everything before today

        rsi_now = float(wilder_rsi(close, 2).iloc[-1])
        sma200 = float(close.rolling(TREND_PERIOD).mean().iloc[-1])
        sma5 = float(close.rolling(5).mean().iloc[-1])
        above_trend = price > sma200
        signal = rsi_now < entry_rsi and above_trend

        flip = find_flip_price(history, price, entry_rsi)

        # Alerting fails closed if this scan lacks the current weekday's bar. This may also
        # suppress a holiday run, which is safer than acting on a stale Friday close.
        last_date = close.index[-1].date()
        today = as_of or datetime.now(MARKET_TZ).date()
        stale = last_date < latest_expected_weekday(today)

        # A squeeze feature is informative only.  Keep an OHLCV or calculation issue from
        # turning a sound RSI reading into a failed scan.
        squeeze = None
        if {"High", "Low", "Close", "Volume"}.issubset(raw.columns):
            try:
                squeeze = latest_squeeze_signal(raw)
            except (TypeError, ValueError, ArithmeticError):
                squeeze = None

        return Reading(
            ticker=ticker,
            price=price,
            prev_close=prev_close,
            rsi2=rsi_now,
            sma200=sma200,
            sma5=sma5,
            above_trend=above_trend,
            signal=signal,
            flip_price=flip,
            stale=stale,
            squeeze=squeeze,
        )
    except Exception as exc:
        return Reading(ticker, 0, 0, float("nan"), 0, 0, False, False, None, True, str(exc))


def render_options_for_ticker(ticker: str, spot: float) -> None:
    """Render an options panel without allowing a live-chain failure to abort the scan."""
    try:
        render_options(ticker, spot, fetch_candidates(ticker, spot))
    except Exception as exc:
        # External option-chain responses are not reliable enough to make the scanner fail.
        # Report only the exception class so an upstream response cannot inject noisy output.
        print(f"    (options chain unavailable for {ticker}: {type(exc).__name__})")


def apply_squeeze_context_gate(
    readings: list[Reading],
    *,
    context_dir: str | Path = DEFAULT_CONTEXT_DIR,
    now: datetime | None = None,
) -> None:
    """Attach a fail-closed gate to completed-close technical squeeze candidates.

    The scanner never reaches out to a broker or data vendor here.  It only reads the
    dated snapshots prepared before the scan, so a missing or corrupt cache cannot turn
    into an optimistic signal during a 120-ticker run.
    """

    for reading in readings:
        reading.squeeze_context = None
        if (
            reading.error
            or reading.stale
            or reading.squeeze is None
            or reading.squeeze.status != "confirmed"
        ):
            continue
        try:
            reading.squeeze_context = evaluate_context(
                load_context(reading.ticker, context_dir),
                now=now,
            )
        except ValueError as exc:
            reading.squeeze_context = ContextGate(False, (str(exc),))


def render_squeeze_watch(readings: list[Reading], *, provisional: bool) -> None:
    """Render separate volatility-compression research candidates, never alerts."""

    evaluated = [
        reading for reading in readings
        if not reading.error and not reading.stale and reading.squeeze is not None
    ]
    candidates = [
        reading for reading in evaluated if reading.squeeze is not None and reading.squeeze.is_candidate
    ]
    if not candidates:
        if evaluated:
            print("\n  VOLATILITY-COMPRESSION WATCH: no technical candidates in "
                  f"{len(evaluated)} fresh OHLCV reading(s).")
        return

    state_order = {"confirmed": 0, "armed": 1, "watch": 2}
    candidates.sort(
        key=lambda reading: (
            state_order.get(reading.squeeze.status, 9) if reading.squeeze else 9,
            reading.squeeze.bandwidth_percentile
            if reading.squeeze is not None and reading.squeeze.bandwidth_percentile is not None
            else 9.0,
            reading.ticker,
        )
    )
    confirmed = [reading for reading in candidates if reading.squeeze and reading.squeeze.status == "confirmed"]
    armed = [reading for reading in candidates if reading.squeeze and reading.squeeze.status == "armed"]
    watching = [reading for reading in candidates if reading.squeeze and reading.squeeze.status == "watch"]

    print("\n  " + "=" * 80)
    print("  VOLATILITY-COMPRESSION WATCH (research; strict data gate, never an alert)")
    print("  " + "=" * 80)
    print("  Setup: 3+ daily closes with Bollinger Bands inside Keltner Channels and")
    print("  BB width in the lowest 20% of its prior 126 sessions. Confirmation requires")
    print("  a completed-close break above the prior 20-day high, 1.2x prior volume, and")
    print("  price above its 200-day average. It is a condition score, not a probability.")
    if provisional:
        print("  *** MARKET OPEN - this panel is PROVISIONAL until the 16:00 close ***")
    print()
    print(f"  {'ticker':<7} {'state':<10} {'data':<11} {'sqz days':>8} {'width pct':>10} {'volume':>8}  trigger")
    print("  " + "-" * 80)

    qualified = sum(
        1
        for reading in confirmed
        if not provisional and reading.squeeze_context is not None and reading.squeeze_context.eligible
    )
    suppressed = len(confirmed) - qualified
    for reading in candidates[:15]:
        squeeze = reading.squeeze
        assert squeeze is not None
        days = squeeze.squeeze_days if squeeze.status == "watch" else squeeze.recent_squeeze_days
        width = "-" if squeeze.bandwidth_percentile is None else f"{squeeze.bandwidth_percentile * 100:.0f}%"
        volume = "-" if squeeze.volume_ratio is None else f"{squeeze.volume_ratio:.2f}x"
        if squeeze.status == "confirmed":
            gate = reading.squeeze_context or ContextGate(False, ("market-context gate not evaluated",))
            if provisional:
                gate = ContextGate(False, ("daily price bar is provisional until the close",))
            data_label = gate.label
            trigger = f"confirmed close > {squeeze.breakout_level:,.2f}" if squeeze.breakout_level else "confirmed"
        elif squeeze.status == "armed":
            data_label = "PENDING"
            trigger = f"watch for close > {squeeze.breakout_level:,.2f}" if squeeze.breakout_level else "await breakout"
        else:
            data_label = "PENDING"
            trigger = f"compressing; prior high {squeeze.breakout_level:,.2f}" if squeeze.breakout_level else "compressing"
        print(
            f"  {reading.ticker:<7} {squeeze.status.upper():<10} {data_label:<11} "
            f"{days:>8} {width:>10} {volume:>8}  {trigger}"
        )
        if squeeze.status == "confirmed" and not gate.eligible:
            detail = "; ".join(gate.reasons[:2])
            more = " …" if len(gate.reasons) > 2 else ""
            print(f"    data gate: {detail}{more}")

    extra = len(candidates) - min(len(candidates), 15)
    if extra:
        print(f"  ... plus {extra} additional lower-priority watch candidate(s).")
    print(f"\n  {len(confirmed)} technical confirmed / {qualified} data-qualified / "
          f"{suppressed} suppressed / {len(armed)} armed / {len(watching)} compressing.")
    print("  A data-qualified result only means its source/freshness checks passed. It is not a")
    print("  probability estimate, profitable claim, order, or alert; it still needs forward paper trading.")


def render(
    readings: list[Reading],
    entry_rsi: float,
    show_options: bool = False,
    send_notifications: bool = True,
    allow_research_signals: bool = False,
    scan_started_at: datetime | None = None,
    show_squeeze: bool = True,
) -> None:
    now = datetime.now(MARKET_TZ)
    snapshot_time = scan_started_at or now
    market_open = (
        snapshot_time.replace(hour=9, minute=30, second=0)
        <= snapshot_time
        < snapshot_time.replace(hour=16, minute=0, second=0)
    )
    # A long sequential scan that starts before the close can contain mixed snapshots even if
    # it finishes after 16:00. Treat the whole run as provisional and fail closed for alerts.
    provisional = market_open and snapshot_time.weekday() < 5

    scanned = [r for r in readings if not r.error]
    sectors = sorted({sector_of(r.ticker) for r in scanned})
    scope = f"{len(scanned)} names" + (f", {len(sectors)} sectors" if len(sectors) > 1 else "")
    print("=" * 98)
    print(f"  RSI(2) PULLBACK SCAN - {scope}        {now:%Y-%m-%d %H:%M} ET")
    print("=" * 98)
    print(f"  Signal condition: RSI(2) < {entry_rsi:g}  AND  price above its {TREND_PERIOD}-day average")
    if provisional:
        print("  *** MARKET OPEN - readings are PROVISIONAL until the 16:00 close ***")
    print()

    ranked = sorted(readings, key=lambda x: (not x.signal, x.rsi2 if not np.isnan(x.rsi2) else 999))
    signals = [r for r in ranked if r.signal]
    actionable = [
        r for r in signals
        if not r.stale and (is_core(r.ticker) or allow_research_signals)
    ]
    errors = [r for r in ranked if r.error]
    ok = [r for r in ranked if not r.error]

    # With a large universe, printing every name buries the point. Show signals plus the
    # nearest misses; the rest is noise on any given day.
    large = len(ok) > 20
    display = signals + [r for r in ok if not r.signal][: (10 if large else len(ok))]

    if large:
        print(f"  Scanned {len(ok)} names. Showing {len(signals)} signal(s) and the "
              f"closest non-signals.\n")

    print(f"  {'ticker':<7} {'sector':<13} {'price':>9} {'chg':>7} {'RSI(2)':>8} "
          f"{'trend':>7} {'evidence':>9} {'signal':>7}  flip level")
    print("  " + "-" * 98)

    for r in display:
        trend = "above" if r.above_trend else "BELOW"
        mark = "  YES" if r.signal else "   -"
        if r.flip_price is not None and r.pct_to_flip is not None:
            direction = "below" if r.pct_to_flip < 0 else "above"
            flip_txt = f"RSI {entry_rsi:g} at {r.flip_price:,.2f} ({abs(r.pct_to_flip):.2f}% {direction})"
        else:
            flip_txt = "-"
        print(
            f"  {r.ticker:<7} {sector_of(r.ticker):<13} {r.price:>9,.2f} "
            f"{r.pct_from_prev:>+6.2f}% {r.rsi2:>8.2f} {trend:>7} {tier_of(r.ticker):>9} "
            f"{mark:>7}  {flip_txt}"
        )

    below_trend = sum(1 for r in ok if not r.above_trend)
    if large:
        print(f"\n  {below_trend} of {len(ok)} are below their 200-day average and are "
              f"excluded regardless of RSI.")
    if errors:
        print(f"  {len(errors)} ticker(s) failed to load: "
              f"{', '.join(r.ticker for r in errors[:8])}")

    if show_squeeze:
        render_squeeze_watch(readings, provisional=provisional)

    print()
    if signals:
        print("  " + "=" * 80)
        print(f"  {len(signals)} SIGNAL(S): " + ", ".join(r.ticker for r in signals))
        print("  " + "=" * 80)
        # Several signals in one sector is a sector event, not independent confirmation.
        sector_counts = Counter(sector_of(r.ticker) for r in signals)
        crowded = [(s, n) for s, n in sector_counts.items() if n >= 3]
        if crowded:
            detail = ", ".join(f"{n} in {s}" for s, n in crowded)
            print(f"  Concentrated: {detail}. That is one sector move, not "
                  f"{len(signals)} independent setups.")

        for r in signals:
            tier = tier_of(r.ticker)
            status = "CORE PAPER-TRADE" if is_core(r.ticker) else "RESEARCH ONLY"
            print(f"\n  {r.ticker}  @ {r.price:,.2f}   [{sector_of(r.ticker)} | "
                  f"{status} | historical label: {tier.upper()}]")
            if tier == "no edge":
                print("    Historical research did not find a reliable share-trading effect in this")
                print("    group. It is displayed for observation, not default alerts.")
            elif tier == "unconfirmed":
                print("    This group was not independently confirmed. It is displayed for research,")
                print("    not default alerts; the later historical period informed these labels.")
            elif not is_core(r.ticker):
                print("    Retrospective context only. The default alert universe is the pre-specified")
                print("    diversified ETF core, not a performance-selected stock subset.")
            print(f"    RSI(2) {r.rsi2:.2f}, {TREND_PERIOD}-day avg {r.sma200:,.2f} "
                  f"({(r.price / r.sma200 - 1) * 100:+.1f}% above)")
            print(f"    Research exit: confirm RSI(2) > {EXIT_RSI:g}, or close > the 5-day "
                  f"average (now {r.sma5:,.2f}); the revised test fills next session open.")
            if provisional and r.flip_price is not None and r.pct_to_flip is not None:
                margin = abs(r.pct_to_flip)
                minutes_left = max(
                    0,
                    int((now.replace(hour=16, minute=0, second=0) - now).total_seconds() // 60),
                )
                # Typical remaining move, measured on 2,259 sessions of 1-minute data:
                # 0.215% average from 15:30, 0.085% from 15:55 (90th pct 0.47% / 0.18%).
                typical = 0.215 if minutes_left > 15 else 0.085
                p90 = 0.471 if minutes_left > 15 else 0.178

                print(f"    Signal cancels on a close above {r.flip_price:,.2f} "
                      f"({margin:.2f}% away, {minutes_left} min to the bell).")
                if margin > p90:
                    print(f"      SETTLED - a move that big happens in under 10% of sessions "
                          f"from here.")
                elif margin > typical:
                    print(f"      LIKELY HOLDS - bigger than the {typical:.2f}% typical move "
                          f"from this point, but not out of reach.")
                else:
                    print(f"      MARGINAL - inside the {typical:.2f}% a session typically "
                          f"still moves from here. Could easily vanish by the close.")
            if show_options and r in actionable:
                render_options_for_ticker(r.ticker, r.price)
            elif show_options and r not in actionable:
                print("    Options scenario withheld: this is not in the enabled alert universe.")
    else:
        print("  No signals. Closest by RSI(2):")
        ranked = [r for r in readings if not r.error and not np.isnan(r.rsi2)]
        for r in sorted(ranked, key=lambda x: x.rsi2)[:5]:
            blocker = "" if r.above_trend else "  (also below its 200-day average)"
            print(f"    {r.ticker:<6} {sector_of(r.ticker):<13} RSI(2) {r.rsi2:>6.2f}"
                  f"  [{tier_of(r.ticker)}]{blocker}")

    if send_notifications and actionable and not provisional:
        try:
            notify(actionable, provisional=False)
        except Exception as exc:
            print(f"\n  (notifications unavailable: {type(exc).__name__})")
    elif send_notifications and signals:
        reason = "the scan began before the close" if provisional else "no non-stale enabled alert signals"
        print(f"\n  Alerts suppressed: {reason}.")

    if any(r.stale and not r.error for r in readings):
        print("\n  WARNING: some data looks stale - check the feed before acting on it.")

    if signals and len(signals) >= 5:
        print(f"\n  NOTE: {len(signals)} simultaneous signals usually means a market-wide")
        print(f"  selloff rather than {len(signals)} independent setups. Treat it as one")
        print("  correlated bet and apply the portfolio's position/sector limits.")

    print("\n" + "-" * 84)
    print("  Research status:")
    print("    - Shares only: signals/exits are evaluated at a completed close and filled")
    print("      next session open in scripts/rsi2_portfolio_validation.py.")
    print("    - Portfolio evidence is historical, uses a current-survivor universe, and is")
    print("      not a prospective profitability claim. The later period is not pristine holdout.")
    print("    - Default alerts are core ETF paper-trade candidates only; use")
    print("      --include-research-signals only to observe broader research alerts.")
    print("  This tool reports an indicator, places no orders, and is not investment advice.")
    print("-" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSI(2) pullback scanner",
        epilog="presets: " + ", ".join(f"{k} ({len(v)})" for k, v in PRESETS.items()),
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["all"],
        help="preset name (core, mag7, etfs, stocks, index, tech, all) or explicit tickers",
    )
    parser.add_argument("--entry-rsi", type=float, default=ENTRY_RSI)
    parser.add_argument("--options", action="store_true",
                        help="show illustrative option scenarios for enabled alert signals")
    parser.add_argument("--no-options", action="store_true",
                        help="deprecated compatibility flag; options are already off by default")
    parser.add_argument("--no-notify", action="store_true", help="do not send notifications")
    parser.add_argument(
        "--no-squeeze",
        action="store_true",
        help="hide the separate volatility-compression watch panel",
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=DEFAULT_CONTEXT_DIR,
        help="dated short-interest/borrow/options/earnings snapshots for strict squeeze qualification",
    )
    parser.add_argument(
        "--include-research-signals",
        action="store_true",
        help="allow non-core, retrospective-research signals to alert/show option scenarios",
    )
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="send a test notification and exit, to verify .env setup",
    )
    parser.add_argument(
        "--force-options",
        metavar="TICKER",
        help="show the options fit for a ticker even without a signal (for inspection)",
    )
    args = parser.parse_args()

    if args.test_notify:
        from dataclasses import replace as _replace

        sample = Reading(
            ticker="TEST", price=100.0, prev_close=102.0, rsi2=3.21, sma200=90.0, sma5=101.0,
            above_trend=True, signal=True, flip_price=101.5, stale=False,
        )
        print("\nSending a test notification ...")
        notify([sample], provisional=True)
        print("\nIf nothing arrived, check .env against .env.example.")
        return

    tickers = resolve(args.tickers)
    scan_started_at = datetime.now(MARKET_TZ)
    scan_as_of = scan_started_at.date()
    print(f"\nScanning {len(tickers)} tickers ...", flush=True)
    readings = []
    for n, ticker in enumerate(tickers, 1):
        readings.append(scan_ticker(ticker, args.entry_rsi, as_of=scan_as_of))
        if len(tickers) > 20 and n % 20 == 0:
            print(f"  {n}/{len(tickers)} ...", flush=True)
    print()
    if not args.no_squeeze:
        apply_squeeze_context_gate(readings, context_dir=args.context_dir, now=scan_started_at)
    render(
        readings,
        args.entry_rsi,
        show_options=args.options and not args.no_options,
        send_notifications=not args.no_notify,
        allow_research_signals=args.include_research_signals,
        scan_started_at=scan_started_at,
        show_squeeze=not args.no_squeeze,
    )

    if args.force_options:
        target = args.force_options.upper()
        match = next((r for r in readings if r.ticker == target and not r.error), None)
        if match is None:
            match = scan_ticker(target, args.entry_rsi, as_of=scan_as_of)
        if match.error:
            print(f"\n  Could not load {target}: {match.error}")
        else:
            print(f"\n\n  [--force-options] {target} has no active signal; "
                  f"showing the chain for inspection only.")
            render_options_for_ticker(target, match.price)


if __name__ == "__main__":
    main()
