"""Forward paper-trading program for RSI(2) signals, traded as call options.

Run it once a day after the close. It is idempotent - running twice in one day will not
duplicate a position or double-count a day.

WHAT IT DOES ON EACH RUN
  1. reads every open paper position and re-prices it against the LIVE option chain for that
     exact contract
  2. applies the scanner's exit rules and closes anything that triggered
  3. scans the universe for new RSI(2) signals and opens a position in the best-executing
     contract
  4. appends a daily equity snapshot and rewrites the report

FILLS ARE MODELLED THE WAY THEY ACTUALLY HAPPEN
  buys cross the spread at the ASK, sells hit the BID. This is the single biggest difference
  between a paper option result and a real one, and it is why the earlier Black-Scholes backfill
  overstated things: a 6% spread on a short-dated call costs 6% of the premium the instant you
  enter, before the stock has moved at all.

  Everything else is real market data: real contracts, real strikes, real quotes.

STARTS EMPTY. The first run opens positions for whatever is signalling today and nothing else.
There is no backfill - this tracks forward from the day you start it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scanner"))

try:
    from scanner.options_fit import fetch_candidates
    from scanner.rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi
    from scanner.universe import PRESETS, sector_of, tier_of
except ImportError:                                   # running from inside scanner/
    from options_fit import fetch_candidates
    from rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi
    from universe import PRESETS, sector_of, tier_of

MARKET_TZ = ZoneInfo("America/New_York")
STATE = ROOT / "output" / "paper_options_state.json"
REPORT = ROOT / "output" / "PAPER_OPTIONS.md"
REPORT_HTML = ROOT / "output" / "PAPER_OPTIONS.html"
MAX_HOLD = 10
DEFAULT_CONTRACTS = 1


@dataclass
class Position:
    ticker: str
    sector: str
    tier: str
    opened: str                 # date the signal fired
    entry_spot: float
    entry_rsi2: float
    expiry: str
    strike: float
    dte_at_entry: int
    contracts: int
    entry_ask: float            # what you paid, per share of the contract
    entry_spread_pct: float
    entry_delta: float
    entry_oi: int
    status: str = "OPEN"
    last_spot: float = 0.0
    last_mid: float = 0.0
    last_bid: float = 0.0
    days_held: int = 0
    exit_date: str | None = None
    exit_bid: float | None = None
    exit_reason: str | None = None
    marks: list = field(default_factory=list)   # [[date, spot, mid]]

    @property
    def cost(self) -> float:
        return self.entry_ask * 100 * self.contracts

    def value(self, price: float) -> float:
        return price * 100 * self.contracts

    @property
    def pnl(self) -> float:
        if self.status == "CLOSED":
            return self.value(self.exit_bid or 0.0) - self.cost
        return self.value(self.last_mid) - self.cost

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost * 100 if self.cost else 0.0

    @property
    def spot_pct(self) -> float:
        return (self.last_spot / self.entry_spot - 1) * 100 if self.entry_spot else 0.0


# --------------------------------------------------------------------------- state
def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started": datetime.now(MARKET_TZ).strftime("%Y-%m-%d"),
            "positions": [], "equity": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, default=str))


# --------------------------------------------------------------------------- market
def readings(tickers: list[str]) -> dict[str, dict]:
    raw = yf.download(tickers, period="1y", progress=False, auto_adjust=False,
                      group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            c = df["Close"].astype(float).dropna()
            if len(c) < TREND_PERIOD + 5:
                continue
            r = wilder_rsi(c, 2)
            out[t] = {
                "date": c.index[-1].strftime("%Y-%m-%d"),
                "close": float(c.iloc[-1]),
                "rsi2": float(r.iloc[-1]),
                "sma200": float(c.rolling(TREND_PERIOD).mean().iloc[-1]),
                "sma5": float(c.rolling(5).mean().iloc[-1]),
            }
        except (KeyError, TypeError, IndexError):
            continue
    return out


def quote_contract(ticker: str, expiry: str, strike: float) -> tuple[float, float] | None:
    """Live (bid, mid) for one specific contract. None if it cannot be quoted."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry).calls
    except Exception:
        return None
    if chain is None or chain.empty:
        return None
    row = chain.loc[np.isclose(pd.to_numeric(chain["strike"], errors="coerce"), strike)]
    if row.empty:
        return None
    r = row.iloc[0]
    bid = max(0.0, float(r.get("bid") or 0.0))
    ask = max(0.0, float(r.get("ask") or 0.0))
    if ask <= 0 and bid <= 0:
        last = float(r.get("lastPrice") or 0.0)
        return (last, last) if last > 0 else None
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)
    return bid, mid


def pick_contract(ticker: str, spot: float, max_cost: float, target_delta: float):
    """Same ranking the scanner shows: execution quality first, inside a budget.

    The budget matters more than it looks. Without it the ranker gravitates to near-the-money
    contracts, which quote tightest and carry the most open interest - and on a $390 stock that
    is $2,000+ per contract. Capping the cost pushes the selection out of the money, which is
    cheaper per contract but a materially different trade: lower delta, so the same 1.5% move in
    the stock produces a smaller move in the option, and more of the premium is time value that
    decays. Cheaper is not better here, it is just affordable.
    """
    try:
        cands = [c for c in fetch_candidates(ticker, spot, min_dte=10, max_dte=35)
                 if c.tradeable]
    except Exception:
        return None
    if not cands:
        return None
    affordable = [c for c in cands if c.cost_per_contract <= max_cost]
    if not affordable:
        cheapest = min(cands, key=lambda c: c.cost_per_contract)
        print(f"    {ticker}: nothing under ${max_cost:,.0f} "
              f"(cheapest tradeable ${cheapest.cost_per_contract:,.0f}) — skipped")
        return None

    # The scanner's display ranks by (spread, OI, delta). That is right for "what is cheapest to
    # get in and out of", but it never actually reaches the delta term: spread_pct is a float, so
    # it decides the ordering on its own and delta is dead weight in the key.
    #
    # For measuring whether the STRATEGY works through options, delta is the term that matters -
    # it sets how much of the validated share move the contract captures. So: keep only contracts
    # whose execution is good enough not to distort the result, then choose on delta.
    liquid = [c for c in affordable if c.spread_pct <= 5.0] or affordable
    return sorted(liquid, key=lambda c: (abs(c.delta - target_delta),
                                         c.spread_pct, -c.open_interest))[0]


# --------------------------------------------------------------------------- engine
def run(universe: str, contracts: int, dry: bool,
        max_cost: float = float("inf"), target_delta: float = 0.70) -> None:
    tickers = sorted(set(PRESETS[universe]))
    st = load_state()
    today = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
    positions = [Position(**{k: v for k, v in p.items()}) for p in st["positions"]]

    print(f"RSI(2) OPTIONS PAPER TRADER — {today}")
    print(f"universe {universe} ({len(tickers)} names) | tracking since {st['started']}\n")

    reads = readings(tickers)
    if not reads:
        raise SystemExit("no market data")
    bar_date = max(r["date"] for r in reads.values())
    print(f"latest completed bar: {bar_date}\n")

    # ---------------------------------------------------------------- 1. mark & exit
    open_pos = [p for p in positions if p.status == "OPEN"]
    if open_pos:
        print(f"marking {len(open_pos)} open position(s)")
    for p in open_pos:
        rd = reads.get(p.ticker)
        if not rd:
            continue
        q = quote_contract(p.ticker, p.expiry, p.strike)
        p.last_spot = rd["close"]
        if q:
            p.last_bid, p.last_mid = q
        if not any(m[0] == bar_date for m in p.marks):
            p.marks.append([bar_date, p.last_spot, p.last_mid])
            p.days_held = len(p.marks)

        reason = None
        if rd["rsi2"] > EXIT_RSI:
            reason = f"RSI(2) {rd['rsi2']:.1f} > {EXIT_RSI:g}"
        elif rd["close"] > rd["sma5"]:
            reason = f"close {rd['close']:.2f} > 5-day SMA {rd['sma5']:.2f}"
        elif p.days_held >= MAX_HOLD:
            reason = f"{MAX_HOLD}-session time stop"
        if reason and not dry:
            p.status, p.exit_date = "CLOSED", bar_date
            p.exit_bid, p.exit_reason = p.last_bid, reason
            print(f"  CLOSE {p.ticker} {p.strike:g}C {p.expiry} @ bid {p.last_bid:.2f} "
                  f"-> {p.pnl:+,.0f} ({p.pnl_pct:+.1f}%)  [{reason}]")
        elif reason:
            print(f"  would close {p.ticker}: {reason}")
        else:
            print(f"  hold  {p.ticker} {p.strike:g}C  spot {p.last_spot:.2f} "
                  f"({p.spot_pct:+.1f}%)  opt {p.last_mid:.2f} ({p.pnl_pct:+.1f}%)  "
                  f"day {p.days_held}")

    # ---------------------------------------------------------------- 2. new signals
    held = {p.ticker for p in positions if p.status == "OPEN"}
    fired = [t for t, r in reads.items()
             if r["rsi2"] < ENTRY_RSI and r["close"] > r["sma200"] and t not in held]
    print(f"\n{len(fired)} new signal(s): {', '.join(fired) if fired else '—'}")

    for t in fired:
        rd = reads[t]
        c = pick_contract(t, rd["close"], max_cost, target_delta)
        if c is None:
            print(f"  {t}: signal, but no tradeable contract (wide spread / thin book) — skipped")
            continue
        p = Position(
            ticker=t, sector=sector_of(t), tier=tier_of(t), opened=bar_date,
            entry_spot=rd["close"], entry_rsi2=rd["rsi2"], expiry=c.expiry,
            strike=c.strike, dte_at_entry=c.dte, contracts=contracts,
            entry_ask=c.ask, entry_spread_pct=c.spread_pct, entry_delta=c.delta,
            entry_oi=c.open_interest, last_spot=rd["close"], last_mid=c.mid,
            last_bid=c.bid, days_held=0, marks=[[bar_date, rd["close"], c.mid]],
        )
        print(f"  OPEN  {t} {c.strike:g}C {c.expiry} ({c.dte}d)  ask {c.ask:.2f} "
              f"x{contracts} = ${p.cost:,.0f}  spread {c.spread_pct:.1f}%  delta {c.delta:.2f}")
        print(f"        spot {rd['close']:.2f}, RSI(2) {rd['rsi2']:.2f}, "
              f"immediate spread cost {(c.ask - c.mid) * 100 * contracts:,.0f}")
        if not dry:
            positions.append(p)

    # ---------------------------------------------------------------- 3. snapshot
    realised = sum(p.pnl for p in positions if p.status == "CLOSED")
    unreal = sum(p.pnl for p in positions if p.status == "OPEN")
    deployed = sum(p.cost for p in positions if p.status == "OPEN")
    if not dry:
        st["equity"] = [e for e in st.get("equity", []) if e["date"] != bar_date]
        st["equity"].append({"date": bar_date, "realised": round(realised, 2),
                             "unrealised": round(unreal, 2),
                             "total": round(realised + unreal, 2),
                             "deployed": round(deployed, 2),
                             "open": len([p for p in positions if p.status == "OPEN"])})
        st["equity"].sort(key=lambda e: e["date"])
        st["positions"] = [asdict(p) for p in positions]
        save_state(st)
        write_report(st, positions, universe)

    print(f"\n  realised ${realised:+,.0f} | unrealised ${unreal:+,.0f} | "
          f"total ${realised + unreal:+,.0f} | deployed ${deployed:,.0f}")
    if not dry:
        print(f"\n  {REPORT}\n  {STATE}")


# --------------------------------------------------------------------------- report
def write_report(st: dict, positions: list[Position], universe: str) -> None:
    closed = [p for p in positions if p.status == "CLOSED"]
    openp = [p for p in positions if p.status == "OPEN"]
    eq = pd.DataFrame(st.get("equity", []))
    L, A = [], None
    A = L.append

    A("# RSI(2) Options — Paper Trading (live, forward)\n")
    A(f"_Tracking since **{st['started']}**. Last run {datetime.now(MARKET_TZ):%Y-%m-%d %H:%M} ET. "
      f"Universe **{universe}**._\n")
    A(f"_Entry: RSI(2) < {ENTRY_RSI:g} and close > {TREND_PERIOD}-day SMA. "
      f"Exit: RSI(2) > {EXIT_RSI:g}, or close > 5-day SMA, or {MAX_HOLD} sessions._\n")
    A("> Buys fill at the **ask**, sells at the **bid** — real quotes, real contracts, spread paid "
      "both ways. This is paper, not advice, and no orders are ever sent.\n")

    realised = sum(p.pnl for p in closed)
    unreal = sum(p.pnl for p in openp)
    A("## Running total\n")
    A("| | |")
    A("|---|---|")
    A(f"| **Total P&L** | **${realised + unreal:+,.2f}** |")
    A(f"| realised (closed) | ${realised:+,.2f} |")
    A(f"| unrealised (open) | ${unreal:+,.2f} |")
    A(f"| capital deployed now | ${sum(p.cost for p in openp):,.2f} |")
    A(f"| positions | {len(closed)} closed, {len(openp)} open |")
    if closed:
        w = [p for p in closed if p.pnl > 0]
        l = [p for p in closed if p.pnl <= 0]
        pf = (sum(p.pnl for p in w) / abs(sum(p.pnl for p in l))) if l and sum(p.pnl for p in l) else float("inf")
        A(f"| **win rate** | **{len(w)/len(closed)*100:.1f}%** ({len(w)}/{len(closed)}) |")
        A(f"| profit factor | {pf:.2f} |")
        A(f"| avg winner | ${np.mean([p.pnl for p in w]):+,.2f} |" if w else "| avg winner | — |")
        A(f"| avg loser | ${np.mean([p.pnl for p in l]):+,.2f} |" if l else "| avg loser | — |")
        A(f"| avg days held | {np.mean([p.days_held for p in closed]):.1f} |")
    A("")

    A("## Open positions\n")
    if openp:
        A("| ticker | contract | opened | days | entry ask | now (mid) | spot % | **P&L $** | P&L % |")
        A("|---|---|---|---|---|---|---|---|---|")
        for p in sorted(openp, key=lambda x: x.opened):
            A(f"| **{p.ticker}** | {p.strike:g}C {p.expiry} | {p.opened} | {p.days_held} | "
              f"{p.entry_ask:.2f} | {p.last_mid:.2f} | {p.spot_pct:+.2f}% | "
              f"**${p.pnl:+,.0f}** | {p.pnl_pct:+.1f}% |")
    else:
        A("_None._")
    A("")

    A("## Closed positions\n")
    if closed:
        A("| ticker | contract | opened | closed | days | in | out | **P&L $** | P&L % | reason |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for p in sorted(closed, key=lambda x: x.exit_date or "", reverse=True):
            A(f"| {p.ticker} | {p.strike:g}C {p.expiry} | {p.opened} | {p.exit_date} | "
              f"{p.days_held} | {p.entry_ask:.2f} | {p.exit_bid:.2f} | "
              f"**${p.pnl:+,.0f}** | {p.pnl_pct:+.1f}% | {p.exit_reason} |")
    else:
        A("_None yet._")
    A("")

    if not eq.empty:
        A("## Daily equity\n")
        A("| date | day change | realised | unrealised | **total** | open |")
        A("|---|---|---|---|---|---|")
        eq = eq.sort_values("date")
        eq["chg"] = eq["total"].diff().fillna(eq["total"])
        for _, r in eq.tail(40).iterrows():
            A(f"| {r['date']} | ${r['chg']:+,.2f} | ${r['realised']:+,.2f} | "
              f"${r['unrealised']:+,.2f} | **${r['total']:+,.2f}** | {int(r['open'])} |")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    write_html(st, positions, universe)


def _cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def write_html(st: dict, positions: list[Position], universe: str) -> None:
    """Same journal as HTML. Markdown has no reliable file association on Windows; a browser
    always exists, so this is what scan.bat actually opens."""
    closed = [p for p in positions if p.status == "CLOSED"]
    openp = [p for p in positions if p.status == "OPEN"]
    realised = sum(p.pnl for p in closed)
    unreal = sum(p.pnl for p in openp)
    total = realised + unreal
    eq = pd.DataFrame(st.get("equity", []))

    h = []
    a = h.append
    a("<!doctype html><meta charset='utf-8'>")
    a("<title>RSI(2) Options Paper Trading</title>")
    a("""<style>
:root{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e3e0dc;--card:#fff;
      --pos:#0a7a45;--neg:#c0392b;--accent:#2d5f8a}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#eceae5;--mut:#9a958d;
      --line:#33312c;--card:#1e1d18;--pos:#4ade80;--neg:#f87171;--accent:#7cb3e0}}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
     font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px}
h1{font-size:23px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 10px;
   text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.hero{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:20px 24px;margin-bottom:8px}
.big{font-size:40px;font-weight:650;letter-spacing:-.02em}
.grid{display:flex;flex-wrap:wrap;gap:26px;margin-top:14px}
.grid div{font-size:13px;color:var(--mut)}
.grid b{display:block;font-size:19px;color:var(--fg);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-bottom:6px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:11.5px;
   text-transform:uppercase;letter-spacing:.05em;padding:8px 10px;
   border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}
.flat{color:var(--mut)}
.tk{font-weight:650}
.note{background:var(--card);border-left:3px solid var(--accent);padding:12px 16px;
      border-radius:0 7px 7px 0;color:var(--mut);font-size:13px;margin:14px 0}
.empty{color:var(--mut);font-style:italic;padding:10px 0}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);
      border-radius:10px;padding:4px 6px}
</style>""")
    a(f"<h1>RSI(2) Options — Paper Trading</h1>")
    a(f"<div class='sub'>Tracking since <b>{st['started']}</b> &middot; "
      f"updated {datetime.now(MARKET_TZ):%Y-%m-%d %H:%M} ET &middot; universe <b>{universe}</b><br>"
      f"Entry RSI(2) &lt; {ENTRY_RSI:g} and close &gt; {TREND_PERIOD}-day SMA &middot; "
      f"exit RSI(2) &gt; {EXIT_RSI:g}, close &gt; 5-day SMA, or {MAX_HOLD} sessions</div>")

    a("<div class='hero'>")
    a(f"<div class='big {_cls(total)}'>${total:+,.2f}</div>")
    a("<div class='grid'>")
    a(f"<div>realised<b class='{_cls(realised)}'>${realised:+,.0f}</b></div>")
    a(f"<div>unrealised<b class='{_cls(unreal)}'>${unreal:+,.0f}</b></div>")
    a(f"<div>deployed<b>${sum(p.cost for p in openp):,.0f}</b></div>")
    a(f"<div>positions<b>{len(closed)} closed / {len(openp)} open</b></div>")
    if closed:
        w = [p for p in closed if p.pnl > 0]
        l = [p for p in closed if p.pnl <= 0]
        pf = sum(p.pnl for p in w) / abs(sum(p.pnl for p in l)) if l and sum(p.pnl for p in l) else 0
        a(f"<div>win rate<b>{len(w)/len(closed)*100:.0f}% ({len(w)}/{len(closed)})</b></div>")
        a(f"<div>profit factor<b>{pf:.2f}</b></div>")
        a(f"<div>avg hold<b>{np.mean([p.days_held for p in closed]):.1f}d</b></div>")
    a("</div></div>")

    a("<div class='note'>Buys fill at the <b>ask</b>, sells at the <b>bid</b> — real contracts, "
      "real quotes, spread paid both ways. Paper only; no orders are ever sent. The validated "
      "edge (+0.509%/trade, t=8.70) is a <b>share</b> return — this measures what survives the "
      "options wrapper.</div>")

    a("<h2>Open positions</h2>")
    if openp:
        a("<div class='wrap'><table><tr><th>Ticker</th><th>Contract</th><th>Opened</th>"
          "<th class='num'>Days</th><th class='num'>Entry</th><th class='num'>Now</th>"
          "<th class='num'>Spot %</th><th class='num'>P&amp;L $</th><th class='num'>P&amp;L %</th></tr>")
        for p in sorted(openp, key=lambda x: x.opened):
            a(f"<tr><td class='tk'>{p.ticker}</td><td>{p.strike:g}C {p.expiry}</td>"
              f"<td>{p.opened}</td><td class='num'>{p.days_held}</td>"
              f"<td class='num'>{p.entry_ask:.2f}</td><td class='num'>{p.last_mid:.2f}</td>"
              f"<td class='num {_cls(p.spot_pct)}'>{p.spot_pct:+.2f}%</td>"
              f"<td class='num {_cls(p.pnl)}'>${p.pnl:+,.0f}</td>"
              f"<td class='num {_cls(p.pnl)}'>{p.pnl_pct:+.1f}%</td></tr>")
        a("</table></div>")
    else:
        a("<div class='empty'>No open positions.</div>")

    a("<h2>Closed positions</h2>")
    if closed:
        a("<div class='wrap'><table><tr><th>Ticker</th><th>Contract</th><th>Opened</th>"
          "<th>Closed</th><th class='num'>Days</th><th class='num'>In</th><th class='num'>Out</th>"
          "<th class='num'>P&amp;L $</th><th class='num'>P&amp;L %</th><th>Reason</th></tr>")
        for p in sorted(closed, key=lambda x: x.exit_date or "", reverse=True):
            a(f"<tr><td class='tk'>{p.ticker}</td><td>{p.strike:g}C {p.expiry}</td>"
              f"<td>{p.opened}</td><td>{p.exit_date}</td><td class='num'>{p.days_held}</td>"
              f"<td class='num'>{p.entry_ask:.2f}</td><td class='num'>{(p.exit_bid or 0):.2f}</td>"
              f"<td class='num {_cls(p.pnl)}'>${p.pnl:+,.0f}</td>"
              f"<td class='num {_cls(p.pnl)}'>{p.pnl_pct:+.1f}%</td>"
              f"<td>{p.exit_reason or ''}</td></tr>")
        a("</table></div>")
    else:
        a("<div class='empty'>Nothing closed yet.</div>")

    if not eq.empty:
        a("<h2>Daily equity</h2><div class='wrap'><table>"
          "<tr><th>Date</th><th class='num'>Day</th><th class='num'>Realised</th>"
          "<th class='num'>Unrealised</th><th class='num'>Total</th><th class='num'>Open</th></tr>")
        eq = eq.sort_values("date")
        eq["chg"] = eq["total"].diff().fillna(eq["total"])
        for _, r in eq.tail(45).iloc[::-1].iterrows():
            a(f"<tr><td>{r['date']}</td>"
              f"<td class='num {_cls(r['chg'])}'>${r['chg']:+,.2f}</td>"
              f"<td class='num'>${r['realised']:+,.2f}</td>"
              f"<td class='num'>${r['unrealised']:+,.2f}</td>"
              f"<td class='num {_cls(r['total'])}'>${r['total']:+,.2f}</td>"
              f"<td class='num'>{int(r['open'])}</td></tr>")
        a("</table></div>")

    REPORT_HTML.write_text("\n".join(h), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="strong",
                    help=f"one of: {', '.join(sorted(PRESETS))}")
    ap.add_argument("--contracts", type=int, default=DEFAULT_CONTRACTS)
    ap.add_argument("--max-cost", type=float, default=float("inf"),
                    help="cap per contract in dollars; default uncapped, so the pick is the "
                         "contract that best tracks the strategy rather than the affordable one")
    ap.add_argument("--target-delta", type=float, default=0.70,
                    help="preferred delta. 0.70 keeps the option close to the share return, "
                         "which is the thing that was actually validated")
    ap.add_argument("--dry-run", action="store_true", help="show actions, write nothing")
    ap.add_argument("--reset", action="store_true", help="erase history and start fresh today")
    args = ap.parse_args()

    if args.universe not in PRESETS:
        raise SystemExit(f"unknown universe. choose from: {', '.join(sorted(PRESETS))}")
    if args.reset and STATE.exists():
        STATE.unlink()
        print("state cleared — starting fresh\n")
    run(args.universe, args.contracts, args.dry_run, args.max_cost, args.target_delta)


if __name__ == "__main__":
    main()
