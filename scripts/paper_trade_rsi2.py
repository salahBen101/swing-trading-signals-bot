"""Paper-trading journal for the RSI(2) scanner signals.

Runs the EXACT rules the live scanner uses - no re-derivation, no quiet variation:

    ENTRY   RSI(2) < 5.0  AND  close > 200-day SMA        filled at the signal's close
    EXIT    RSI(2) > 65   OR   close > 5-day SMA          filled at that day's close
            OR 10 sessions held (time stop)

Backfills the trailing year from real daily history so results exist immediately, then keeps
accumulating each time it is run. State lives in output/paper_trades.json, so re-running does not
duplicate trades or lose history.

ON THE OPTION NUMBERS - READ THIS

Share returns here are EXACT: real closes, real dates. The option column is MODELLED, using
Black-Scholes with implied volatility estimated from each name's own trailing realised volatility.
It is an approximation, and it is optimistic in three specific ways that matter:

  * no bid/ask spread is crossed (real short-dated equity options often cost 2-5% of premium
    per side, and far more on wide markets)
  * implied volatility is held constant from entry to exit, so there is no IV crush - the single
    most common way a directionally-correct option trade still loses money
  * fills are assumed at the mid

Treat the share column as the truth and the option column as an indication of leverage, not a
prediction of your fill. The strategy's validated edge (+0.509%/trade, t=8.70 on holdout) is a
SHARE return. Nothing about options was ever validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scanner"))

import yfinance as yf  # noqa: E402

from scanner.options_fit import RISK_FREE, bs_call, call_delta  # noqa: E402
from scanner.rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi  # noqa: E402
from scanner.universe import PRESETS, tier_of  # noqa: E402

STATE = ROOT / "output" / "paper_trades.json"
REPORT = ROOT / "output" / "PAPER_TRADING.md"
MAX_HOLD = 10
OPT_DTE = 21            # days to expiry at entry, matching the scanner's usual window
OPT_OTM_PCT = 3.0       # strike this far above spot
CONTRACTS = 1
IV_PREMIUM = 1.15       # implied vol typically prints above trailing realised


# --------------------------------------------------------------------------- data
def fetch(tickers: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    print(f"Fetching {len(tickers)} names ...")
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=False,
                      group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])
            if len(df) > TREND_PERIOD + 30:
                out[t] = df
        except (KeyError, TypeError):
            continue
    print(f"  usable: {len(out)}")
    return out


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["close"] = df["Close"].astype(float)
    d["rsi2"] = wilder_rsi(d["close"], 2)
    d["sma200"] = d["close"].rolling(TREND_PERIOD).mean()
    d["sma5"] = d["close"].rolling(5).mean()
    # trailing realised vol, annualised - the input to the modelled option price
    d["rv"] = d["close"].pct_change().rolling(20).std() * np.sqrt(252)
    return d


# --------------------------------------------------------------------------- options
def model_option(spot: float, iv: float, dte: int, strike: float | None = None) -> dict:
    """Black-Scholes call, struck OTM_PCT above spot unless a strike is supplied."""
    k = strike if strike is not None else round(spot * (1 + OPT_OTM_PCT / 100), 0)
    T = max(dte, 1) / 365.0
    iv = float(np.clip(iv, 0.12, 1.50))
    px = bs_call(spot, k, T, RISK_FREE, iv)
    return {"strike": float(k), "dte": dte, "iv": round(iv, 3),
            "premium": round(px, 2),
            "delta": round(call_delta(spot, k, T, RISK_FREE, iv), 3)}


# --------------------------------------------------------------------------- engine
def build_trades(data: dict[str, pd.DataFrame], start: pd.Timestamp) -> list[dict]:
    trades = []
    for tkr, df in data.items():
        d = indicators(df)
        d = d.dropna(subset=["sma200", "rsi2", "sma5"])
        if d.empty:
            continue
        idx = d.index
        i = 0
        while i < len(d):
            row = d.iloc[i]
            if not (row["rsi2"] < ENTRY_RSI and row["close"] > row["sma200"]):
                i += 1
                continue
            if idx[i] < start:
                i += 1
                continue

            entry_px = float(row["close"])
            iv = float(row["rv"]) * IV_PREMIUM if np.isfinite(row["rv"]) else 0.35
            opt_in = model_option(entry_px, iv, OPT_DTE)

            exit_i, reason = None, "open"
            for j in range(i + 1, min(i + 1 + MAX_HOLD, len(d))):
                rj = d.iloc[j]
                if rj["rsi2"] > EXIT_RSI:
                    exit_i, reason = j, "rsi>65"; break
                if rj["close"] > rj["sma5"]:
                    exit_i, reason = j, "close>sma5"; break
                if j - i >= MAX_HOLD:
                    exit_i, reason = j, "10-day stop"; break

            rec = {
                "ticker": tkr, "sector_tier": tier_of(tkr),
                "signal_date": idx[i].strftime("%Y-%m-%d"),
                "entry": round(entry_px, 2),
                "entry_rsi2": round(float(row["rsi2"]), 2),
                "opt_strike": opt_in["strike"], "opt_dte_in": OPT_DTE,
                "opt_iv": opt_in["iv"], "opt_delta": opt_in["delta"],
                "opt_premium_in": opt_in["premium"],
            }
            if exit_i is not None:
                ex = d.iloc[exit_i]
                held = exit_i - i
                opt_out = model_option(float(ex["close"]), iv, max(OPT_DTE - held, 1),
                                       strike=opt_in["strike"])
                rec |= {
                    "status": "closed",
                    "exit_date": idx[exit_i].strftime("%Y-%m-%d"),
                    "exit": round(float(ex["close"]), 2),
                    "days": held, "exit_reason": reason,
                    "share_pct": round((float(ex["close"]) / entry_px - 1) * 100, 3),
                    "opt_premium_out": opt_out["premium"],
                    "opt_pct": round((opt_out["premium"] / opt_in["premium"] - 1) * 100, 1)
                    if opt_in["premium"] > 0.01 else None,
                }
                i = exit_i + 1
            else:
                last = d.iloc[-1]
                held = len(d) - 1 - i
                opt_now = model_option(float(last["close"]), iv, max(OPT_DTE - held, 1),
                                       strike=opt_in["strike"])
                rec |= {
                    "status": "OPEN", "exit_date": None, "exit": None,
                    "days": held, "exit_reason": None,
                    "mark": round(float(last["close"]), 2),
                    "share_pct": round((float(last["close"]) / entry_px - 1) * 100, 3),
                    "opt_premium_out": opt_now["premium"],
                    "opt_pct": round((opt_now["premium"] / opt_in["premium"] - 1) * 100, 1)
                    if opt_in["premium"] > 0.01 else None,
                }
                i = len(d)
            trades.append(rec)
    return sorted(trades, key=lambda r: r["signal_date"])


def daily_curve(trades: list[dict], data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight $1,000 per signal, marked daily. Shares, not options."""
    closed = [t for t in trades if t["status"] == "closed"]
    if not closed:
        return pd.DataFrame()
    all_days = sorted({d for df in data.values() for d in df.index})
    curve = pd.Series(0.0, index=pd.DatetimeIndex(all_days))
    for t in closed:
        s = pd.Timestamp(t["signal_date"]); e = pd.Timestamp(t["exit_date"])
        df = data[t["ticker"]]
        seg = df.loc[(df.index > s) & (df.index <= e), "Close"].astype(float)
        if seg.empty:
            continue
        prev = float(t["entry"])
        for dt, px in seg.items():
            curve.loc[dt] += (px / prev - 1) * 1000.0
            prev = px
    out = pd.DataFrame({"pnl": curve})
    out["cum"] = out["pnl"].cumsum()
    return out[out["pnl"] != 0]


# --------------------------------------------------------------------------- report
def write_report(trades: list[dict], curve: pd.DataFrame, universe: str) -> None:
    closed = [t for t in trades if t["status"] == "closed"]
    openp = [t for t in trades if t["status"] == "OPEN"]
    L = []
    A = L.append
    A("# RSI(2) Paper Trading Journal\n")
    A(f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Universe: **{universe}**. "
      f"Rules: entry RSI(2) < {ENTRY_RSI:g} and close > {TREND_PERIOD}-day SMA; "
      f"exit RSI(2) > {EXIT_RSI:g}, or close > 5-day SMA, or {MAX_HOLD} sessions._\n")
    A("> Share returns are exact. **Option values are modelled** (Black-Scholes, IV from trailing "
      "realised vol, no spread crossed, no IV crush). The validated edge is a share return; "
      "options were never validated.\n")

    if closed:
        s = pd.DataFrame(closed)
        wins = s[s["share_pct"] > 0]
        losses = s[s["share_pct"] <= 0]
        pf = wins["share_pct"].sum() / abs(losses["share_pct"].sum()) if len(losses) else np.inf
        A("## Summary — closed trades\n")
        A(f"| metric | shares | options (modelled) |")
        A("|---|---|---|")
        A(f"| trades | {len(s)} | {len(s)} |")
        A(f"| win rate | **{len(wins)/len(s)*100:.1f}%** | "
          f"{(s['opt_pct'] > 0).mean()*100:.1f}% |")
        A(f"| avg per trade | **{s['share_pct'].mean():+.3f}%** | "
          f"{s['opt_pct'].dropna().mean():+.1f}% |")
        A(f"| avg winner | {wins['share_pct'].mean():+.3f}% | — |")
        A(f"| avg loser | {losses['share_pct'].mean():+.3f}% | — |")
        A(f"| profit factor | **{pf:.2f}** | — |")
        A(f"| total (sum of %) | {s['share_pct'].sum():+.2f}% | "
          f"{s['opt_pct'].dropna().sum():+.0f}% |")
        A(f"| avg days held | {s['days'].mean():.1f} | |")
        A(f"| best / worst | {s['share_pct'].max():+.2f}% / {s['share_pct'].min():+.2f}% | |\n")

        if not curve.empty:
            A(f"**Equity ($1,000 per signal, equal weight):** "
              f"`${curve['cum'].iloc[-1]:+,.0f}` over {len(curve)} active days, "
              f"peak `${curve['cum'].max():+,.0f}`, "
              f"worst drawdown `${(curve['cum'] - curve['cum'].cummax()).min():+,.0f}`\n")

    A("## Open positions\n")
    if openp:
        A("| ticker | tier | signal date | entry | mark | days | share % | option | opt % |")
        A("|---|---|---|---|---|---|---|---|---|")
        for t in openp:
            A(f"| **{t['ticker']}** | {t['sector_tier']} | {t['signal_date']} | "
              f"{t['entry']:.2f} | {t.get('mark', float('nan')):.2f} | {t['days']} | "
              f"**{t['share_pct']:+.2f}%** | {t['opt_strike']:.0f}C | "
              f"{t['opt_pct'] if t['opt_pct'] is not None else 0:+.0f}% |")
    else:
        A("_None._")
    A("")

    A("## Closed trades\n")
    if closed:
        A("| ticker | signal | entry | exit date | exit | days | share % | opt % | reason |")
        A("|---|---|---|---|---|---|---|---|---|")
        for t in sorted(closed, key=lambda r: r["signal_date"], reverse=True)[:80]:
            A(f"| {t['ticker']} | {t['signal_date']} | {t['entry']:.2f} | {t['exit_date']} | "
              f"{t['exit']:.2f} | {t['days']} | **{t['share_pct']:+.2f}%** | "
              f"{t['opt_pct'] if t['opt_pct'] is not None else 0:+.0f}% | {t['exit_reason']} |")
        if len(closed) > 80:
            A(f"\n_showing 80 of {len(closed)}; full list in paper_trades.json_")
    A("")

    if not curve.empty:
        A("## Daily P&L (last 30 active days)\n")
        A("| date | day P&L | cumulative |")
        A("|---|---|---|")
        for dt, r in curve.tail(30).iterrows():
            A(f"| {dt:%Y-%m-%d} | ${r['pnl']:+,.2f} | ${r['cum']:+,.2f} |")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="mag7_qqq",
                    help="preset name, or 'strong' for the validated tier")
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()

    if args.universe not in PRESETS:
        raise SystemExit(f"unknown universe '{args.universe}'. "
                         f"Choose from: {', '.join(sorted(PRESETS))}")
    tickers = list(PRESETS[args.universe])
    tickers = sorted(set(tickers))

    data = fetch(tickers)
    if not data:
        raise SystemExit("no data fetched")
    start = pd.Timestamp.now(tz=None).normalize() - pd.DateOffset(months=args.months)
    trades = build_trades(data, start)
    curve = daily_curve(trades, data)

    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"generated": datetime.now().isoformat(),
                                 "universe": args.universe, "trades": trades}, indent=2))
    write_report(trades, curve, args.universe)

    closed = [t for t in trades if t["status"] == "closed"]
    openp = [t for t in trades if t["status"] == "OPEN"]
    print(f"\n{len(trades)} signals over {args.months} months "
          f"({len(closed)} closed, {len(openp)} open)")
    if closed:
        s = pd.DataFrame(closed)
        wins = (s["share_pct"] > 0).mean() * 100
        print(f"  share win rate {wins:.1f}%  avg {s['share_pct'].mean():+.3f}%/trade  "
              f"total {s['share_pct'].sum():+.1f}%")
    if openp:
        print("  OPEN: " + ", ".join(f"{t['ticker']} {t['share_pct']:+.1f}%" for t in openp))
    print(f"\n  {REPORT}\n  {STATE}")


if __name__ == "__main__":
    main()
