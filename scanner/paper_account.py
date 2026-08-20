"""The scanner's own paper-trading ACCOUNT. Starts at $20,000, runs forward, learns.

This is not a backtest. Every run records what the live scanner actually signalled that day,
sizes the position against the account's current equity, and marks open positions off the real
option chain. Run it daily and after a year you have a real, unhindsighted track record.

    day 1        account = $20,000, no positions
    every run    mark open positions -> apply exit rules -> take new signals with whatever
                 cash the account has left -> append one equity snapshot
    output       every trade with its return, the running equity curve, win rate, and what
                 the model has learned so far

POSITION SIZING
    Each position gets a fixed fraction of CURRENT equity, so size compounds up as the account
    grows and shrinks automatically after losses. Contracts are whole numbers, so a signal the
    account cannot afford is simply missed - and those misses are recorded, because a $20k
    account genuinely cannot take every signal and pretending otherwise is how paper results
    stop resembling real ones.

FILLS
    Buy at the ASK, sell at the BID. Real contracts, real quotes, spread paid both directions.

THE LEARNING LAYER
    Trains only on trades this account has already closed, and only once there are enough of
    them. It scores signals the rules already produced; it never invents one. Until the minimum
    is reached it does nothing at all and says so. Its running verdict - whether the trades it
    liked actually beat the trades it disliked - is printed every run, so it can be judged
    instead of trusted.
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
except ImportError:
    from options_fit import fetch_candidates
    from rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi
    from universe import PRESETS, sector_of, tier_of

MARKET_TZ = ZoneInfo("America/New_York")
STATE = ROOT / "output" / "paper_account.json"
REPORT = ROOT / "output" / "PAPER_ACCOUNT.html"
MAX_HOLD = 10
START_EQUITY = 50_000.0
ALLOC_PCT = 0.12          # of current equity per position
MAX_POSITIONS = 10
ML_MIN_TRADES = 40        # before the model is allowed any opinion at all
MIN_DTE = 21              # > MAX_HOLD sessions in calendar days, so nothing expires while held


@dataclass
class Pos:
    ticker: str
    sector: str
    tier: str
    opened: str
    entry_spot: float
    entry_rsi2: float
    expiry: str
    strike: float
    dte_at_entry: int
    contracts: int
    entry_ask: float
    entry_spread_pct: float
    entry_delta: float
    cost: float
    feat: dict = field(default_factory=dict)
    ml_score: float | None = None
    status: str = "OPEN"
    last_spot: float = 0.0
    last_mid: float = 0.0
    last_bid: float = 0.0
    days_held: int = 0
    exit_date: str | None = None
    exit_bid: float | None = None
    exit_reason: str | None = None
    proceeds: float = 0.0
    stale: bool = False
    marks: list = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.last_mid * 100 * self.contracts

    @property
    def pnl(self) -> float:
        return (self.proceeds - self.cost) if self.status == "CLOSED" else (self.value - self.cost)

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.cost * 100 if self.cost else 0.0

    @property
    def spot_pct(self) -> float:
        return (self.last_spot / self.entry_spot - 1) * 100 if self.entry_spot else 0.0


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started": datetime.now(MARKET_TZ).strftime("%Y-%m-%d"),
            "start_equity": START_EQUITY, "cash": START_EQUITY,
            "positions": [], "equity": [], "missed": []}


def readings(tickers: list[str]) -> tuple[dict[str, dict], pd.DatetimeIndex]:
    """Returns per-ticker readings AND the real trading-session calendar.

    The calendar matters: holding period must be counted in SESSIONS THE MARKET HELD, not in
    the number of times this script happened to be run. Counting runs means that skipping a few
    days makes a 10-session stop fire on day 30, or never.
    """
    raw = yf.download(tickers, period="2y", progress=False, auto_adjust=False,
                      group_by="ticker", threads=True)
    out = {}
    sessions = pd.DatetimeIndex([])
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            c = df["Close"].astype(float).dropna()
            if len(c) < TREND_PERIOD + 5:
                continue
            r = wilder_rsi(c, 2)
            out[t] = {
                "date": c.index[-1].strftime("%Y-%m-%d"),
                "close": float(c.iloc[-1]), "rsi2": float(r.iloc[-1]),
                "sma200": float(c.rolling(TREND_PERIOD).mean().iloc[-1]),
                "sma5": float(c.rolling(5).mean().iloc[-1]),
                "vol20": float(c.pct_change().rolling(20).std().iloc[-1] * 100),
                "ret5": float(c.pct_change(5).iloc[-1] * 100),
                "hi252": float(c.rolling(252).max().iloc[-1]),
            }
        except (KeyError, TypeError, IndexError):
            continue
        if len(c.index) > len(sessions):
            sessions = c.index
    return out, sessions


def sessions_between(sessions: pd.DatetimeIndex, opened: str, bar: str) -> int:
    """Trading sessions strictly after `opened`, up to and including `bar`."""
    if len(sessions) == 0:
        return 0
    lo, hi = pd.Timestamp(opened), pd.Timestamp(bar)
    return int(((sessions > lo) & (sessions <= hi)).sum())


def features(r: dict, breadth: int, mkt: float) -> dict:
    return {
        "rsi2": r["rsi2"],
        "above_200": (r["close"] / r["sma200"] - 1) * 100,
        "below_5": (r["close"] / r["sma5"] - 1) * 100,
        "vol20": r["vol20"], "ret5": r["ret5"],
        "off_high": (r["close"] / r["hi252"] - 1) * 100,
        "breadth": float(breadth), "mkt_above_200": mkt,
    }


def quote_contract(ticker: str, expiry: str, strike: float) -> tuple[float, float] | None:
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
    return bid, ((bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask))


def pick_contract(ticker: str, spot: float, budget: float, target_delta: float = 0.70):
    try:
        # min_dte must exceed the 10-SESSION hold (~14 calendar days) or a contract can
        # expire while the position is still open.
        cands = [c for c in fetch_candidates(ticker, spot, min_dte=MIN_DTE, max_dte=45)
                 if c.tradeable]
    except Exception:
        return None
    if not cands:
        return None
    fits = [c for c in cands if c.cost_per_contract <= budget]
    if not fits:
        return None
    liquid = [c for c in fits if c.spread_pct <= 6.0] or fits
    return sorted(liquid, key=lambda c: (abs(c.delta - target_delta),
                                         c.spread_pct, -c.open_interest))[0]


class Learner:
    """Scores signals using only trades this account has already closed."""

    def __init__(self):
        self.model = None
        self.cols: list[str] = []
        self.n = 0
        self.verdict = "not enough closed trades yet"

    def fit(self, closed: list[Pos]) -> None:
        rows = [p.feat | {"y": p.pnl_pct} for p in closed if p.feat]
        if len(rows) < ML_MIN_TRADES:
            self.verdict = (f"dormant - {len(rows)}/{ML_MIN_TRADES} closed trades. "
                            "It will not score anything until it has a real sample.")
            return
        from sklearn.ensemble import GradientBoostingRegressor
        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < ML_MIN_TRADES:
            return
        self.cols = [c for c in df.columns if c != "y"]
        self.model = GradientBoostingRegressor(n_estimators=60, max_depth=2,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=7)
        self.model.fit(df[self.cols], df["y"])
        self.n = len(df)

        # Honest self-assessment: did the trades it liked beat the ones it disliked?
        scored = [(p.ml_score, p.pnl_pct) for p in closed if p.ml_score is not None]
        if len(scored) >= 20:
            s = pd.DataFrame(scored, columns=["score", "ret"])
            hi = s[s["score"] >= s["score"].median()]["ret"].mean()
            lo = s[s["score"] < s["score"].median()]["ret"].mean()
            self.verdict = (f"trained on {self.n} trades. On {len(s)} it scored live: "
                            f"top half returned {hi:+.1f}%, bottom half {lo:+.1f}% "
                            f"({'ADDING VALUE' if hi > lo else 'NOT adding value'})")
        else:
            self.verdict = (f"trained on {self.n} trades, but only {len(scored)} were scored "
                            "live - too few to judge it yet")

    def score(self, feat: dict) -> float | None:
        if self.model is None:
            return None
        x = pd.DataFrame([{c: feat.get(c, np.nan) for c in self.cols}])
        if x.isna().any(axis=None):
            return None
        return float(self.model.predict(x)[0])


# --------------------------------------------------------------------------- engine
def run(universe: str, dry: bool) -> None:
    tickers = sorted(set(PRESETS[universe]))
    st = load_state()
    positions = [Pos(**p) for p in st["positions"]]
    cash = float(st["cash"])
    start_eq = float(st["start_equity"])

    print(f"RSI(2) PAPER ACCOUNT  |  started {st['started']}  |  universe {universe}")
    reads, sessions = readings(tickers)
    if not reads:
        raise SystemExit("no market data")
    bar = max(r["date"] for r in reads.values())

    # An incomplete bar is not a bar. yfinance happily returns a partial candle for a session
    # that is still trading, and acting on it means entering on an RSI that will change before
    # the close. Refuse to record anything unless the last bar is a finished session.
    now = datetime.now(MARKET_TZ)
    if pd.Timestamp(bar).date() == now.date() and now.hour < 16:
        raise SystemExit(
            f"the {bar} bar is still forming (it is {now:%H:%M} ET). "
            "Run after 16:00 ET so the daily close is final.")
    print(f"latest completed bar: {bar}\n")

    closed = [p for p in positions if p.status == "CLOSED"]
    learner = Learner()
    learner.fit(closed)

    # ---------------------------------------------------------------- mark & exit
    for p in [x for x in positions if x.status == "OPEN"]:
        rd = reads.get(p.ticker)
        if not rd:
            continue
        q = quote_contract(p.ticker, p.expiry, p.strike)
        p.last_spot = rd["close"]
        if q:
            p.last_bid, p.last_mid = q
            p.stale = False
        else:
            # A failed quote used to leave the previous price in place silently, so the account
            # reported an equity built on a stale mark and looked fine while being wrong.
            p.stale = True
            print(f"  WARN {p.ticker} {p.strike:g}C {p.expiry}: no quote returned; "
                  f"mark held at {p.last_mid:.2f} from {p.marks[-1][0] if p.marks else 'entry'}")

        # Sessions the MARKET held, not runs of this script.
        p.days_held = sessions_between(sessions, p.opened, bar)
        if not any(m[0] == bar for m in p.marks):
            p.marks.append([bar, p.last_spot, p.last_mid])

        reason = None
        # Expiry first: an expired contract is worth its intrinsic value and nothing else.
        # Without this it simply stops quoting and sits in the book forever at a stale price.
        if pd.Timestamp(bar).date() >= pd.Timestamp(p.expiry).date():
            intrinsic = max(0.0, p.last_spot - p.strike)
            p.last_bid = p.last_mid = intrinsic
            reason = f"expired {p.expiry} (settled at intrinsic {intrinsic:.2f})"
        elif rd["rsi2"] > EXIT_RSI:
            reason = f"RSI(2) {rd['rsi2']:.1f} > {EXIT_RSI:g}"
        elif rd["close"] > rd["sma5"]:
            reason = "close > 5-day SMA"
        elif p.days_held >= MAX_HOLD:
            reason = f"{MAX_HOLD}-session stop"

        if reason and not dry:
            p.status, p.exit_date = "CLOSED", bar
            p.exit_bid, p.exit_reason = p.last_bid, reason
            p.proceeds = p.last_bid * 100 * p.contracts
            cash += p.proceeds
            print(f"  SELL {p.ticker} {p.strike:g}C  {p.exit_bid:.2f} x{p.contracts}  "
                  f"{p.pnl:+,.0f} ({p.pnl_pct:+.1f}%)   [{reason}]")
        elif not reason:
            print(f"  hold {p.ticker} {p.strike:g}C  spot {p.spot_pct:+.1f}%  "
                  f"opt {p.pnl_pct:+.1f}%  day {p.days_held}")

    open_now = [p for p in positions if p.status == "OPEN"]
    equity = cash + sum(p.value for p in open_now)

    # ---------------------------------------------------------------- new signals
    held = {p.ticker for p in open_now}
    fired = [t for t, r in reads.items()
             if r["rsi2"] < ENTRY_RSI and r["close"] > r["sma200"] and t not in held]
    spy = reads.get("SPY") or reads.get("QQQ")
    mkt = ((spy["close"] / spy["sma200"] - 1) * 100) if spy else 0.0

    print(f"\n{len(fired)} signal(s): {', '.join(fired) if fired else '-'}")
    slots = MAX_POSITIONS - len(open_now)
    cands = []
    for t in fired:
        f = features(reads[t], len(fired), mkt)
        cands.append((learner.score(f), t, f))
    # deepest oversold first when the model has no opinion yet
    cands.sort(key=lambda c: (-(c[0] if c[0] is not None else 0), reads[c[1]]["rsi2"]))

    for score, t, f in cands:
        if slots <= 0:
            st.setdefault("missed", []).append({"date": bar, "ticker": t, "why": "no slots"})
            print(f"  MISS {t} - all {MAX_POSITIONS} slots full")
            continue
        rd = reads[t]
        budget = min(equity * ALLOC_PCT, cash)
        c = pick_contract(t, rd["close"], budget)
        if c is None:
            st.setdefault("missed", []).append({"date": bar, "ticker": t, "why": "no contract in budget"})
            print(f"  MISS {t} - no tradeable contract under ${budget:,.0f}")
            continue
        n = max(int(budget // c.cost_per_contract), 0)
        if n < 1:
            st.setdefault("missed", []).append({"date": bar, "ticker": t, "why": "cannot afford 1"})
            print(f"  MISS {t} - one contract is ${c.cost_per_contract:,.0f}, budget ${budget:,.0f}")
            continue
        cost = c.ask * 100 * n
        if cost > cash:
            st.setdefault("missed", []).append({"date": bar, "ticker": t, "why": "out of cash"})
            print(f"  MISS {t} - out of cash")
            continue

        p = Pos(ticker=t, sector=sector_of(t), tier=tier_of(t), opened=bar,
                entry_spot=rd["close"], entry_rsi2=rd["rsi2"], expiry=c.expiry,
                strike=c.strike, dte_at_entry=c.dte, contracts=n, entry_ask=c.ask,
                entry_spread_pct=c.spread_pct, entry_delta=c.delta, cost=cost,
                feat=f, ml_score=score, last_spot=rd["close"], last_mid=c.mid,
                last_bid=c.bid, marks=[[bar, rd["close"], c.mid]])
        tag = f"  model {score:+.1f}%" if score is not None else ""
        print(f"  BUY  {t} {c.strike:g}C {c.expiry} ({c.dte}d) x{n} @ {c.ask:.2f} "
              f"= ${cost:,.0f}  delta {c.delta:.2f}  spread {c.spread_pct:.1f}%{tag}")
        if not dry:
            cash -= cost
            positions.append(p)
            slots -= 1

    # ---------------------------------------------------------------- snapshot
    open_now = [p for p in positions if p.status == "OPEN"]
    closed = [p for p in positions if p.status == "CLOSED"]
    equity = cash + sum(p.value for p in open_now)

    print(f"\n  equity ${equity:,.2f}   cash ${cash:,.2f}   "
          f"invested ${sum(p.cost for p in open_now):,.0f}   "
          f"return {(equity/start_eq-1)*100:+.2f}%")
    print(f"  model: {learner.verdict}")

    if dry:
        return
    st["cash"] = round(cash, 2)
    st["positions"] = [asdict(p) for p in positions]
    st["equity"] = [e for e in st.get("equity", []) if e["date"] != bar]
    st["equity"].append({"date": bar, "equity": round(equity, 2), "cash": round(cash, 2),
                         "open": len(open_now),
                         "realised": round(sum(p.pnl for p in closed), 2)})
    st["equity"].sort(key=lambda e: e["date"])
    st["ml_verdict"] = learner.verdict
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, default=str))
    write_report(st, positions, universe, learner)
    print(f"\n  {REPORT}")


# --------------------------------------------------------------------------- report
GO_LIVE_T = 2.0
GO_LIVE_MIN_TRADES = 96      # from the power calculation, not chosen for convenience


def decision(closed: list[Pos]) -> dict:
    """Pre-registered go/no-go, computed on the SHARE-equivalent return of each signal.

    Why shares decide it and options do not, measured on this strategy's own trades:

        shares   mean +0.759%/trade, sd  3.72%  ->  signal/noise 0.204  ->    96 trades for t=2
        options  mean +0.590%/trade, sd 27.39%  ->  signal/noise 0.022  -> 8,635 trades for t=2

    The 7.4x leverage of a 0.70-delta call multiplies the edge and the noise by the same factor,
    but the ~5% round-trip spread subtracts a fixed amount that consumes most of the levered
    edge. Signal-to-noise collapses by an order of magnitude, and 8,635 trades is 66 years.

    So the option P&L measures what execution COSTS. It cannot establish whether an edge exists,
    and it must never be the thing that authorises real money. The share track answers that in
    roughly 96 trades - about eight months at this signal rate.

    One year of share trades has only ~35% power to detect the validated +0.509% edge, so
    "inconclusive" is the most likely outcome and is NOT a failure. The false-positive rate at
    t > 2.0 is 2.5%, which is why the bar sits there.
    """
    done = [p for p in closed if p.entry_spot and p.last_spot]
    n = len(done)
    if n < 5:
        return {"n": n, "status": "TOO EARLY", "t": 0.0, "mean": 0.0,
                "detail": f"{n} closed trades. Need {GO_LIVE_MIN_TRADES} for a verdict."}
    r = np.array([(x.last_spot / x.entry_spot - 1) * 100 for x in done])
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else 0.0
    if n < GO_LIVE_MIN_TRADES:
        status = "COLLECTING"
        detail = (f"{n}/{GO_LIVE_MIN_TRADES} trades. Nothing is decided yet - at this sample "
                  f"even a strong-looking number is inside the range of luck.")
    elif t > GO_LIVE_T and r.mean() > 0:
        status = "PASS"
        detail = (f"share edge {r.mean():+.3f}%/trade, t={t:.2f} over {n} trades. Clears the "
                  f"pre-registered bar. Compare against the validated +0.509% before sizing.")
    else:
        status = "NOT PROVEN"
        detail = (f"share edge {r.mean():+.3f}%/trade, t={t:.2f} over {n} trades - below the "
                  f"t>{GO_LIVE_T} bar. Most likely outcome after one year; keep collecting "
                  f"rather than concluding either way.")
    return {"n": n, "status": status, "t": t, "mean": r.mean(), "sd": sd, "detail": detail}


def _c(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def curve_svg(eq: list[dict], w: int = 1040, h: int = 240) -> str:
    if len(eq) < 2:
        return ("<div style='color:var(--mut);font-size:13px;padding:20px 4px'>"
                "The curve appears once there are at least two trading days recorded.</div>")
    v = np.array([e["equity"] for e in eq], dtype=float)
    lo, hi = min(v.min(), START_EQUITY), max(v.max(), START_EQUITY)
    rng = (hi - lo) or 1.0
    pad = 40
    xs = np.linspace(pad, w - 12, len(v))
    ys = h - pad - (v - lo) / rng * (h - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    base_y = h - pad - (START_EQUITY - lo) / rng * (h - 2 * pad)
    col = "var(--pos)" if v[-1] >= START_EQUITY else "var(--neg)"
    out = [f"<svg viewBox='0 0 {w} {h}' style='width:100%;height:auto'>"]
    for f in (0.0, 0.5, 1.0):
        y = h - pad - f * (h - 2 * pad)
        out.append(f"<line x1='{pad}' y1='{y:.1f}' x2='{w-12}' y2='{y:.1f}' "
                   f"stroke='var(--line)' stroke-dasharray='2,4'/>"
                   f"<text x='4' y='{y+4:.1f}' font-size='10' fill='var(--mut)'>"
                   f"${(lo+rng*f)/1000:.1f}k</text>")
    out.append(f"<polygon points='{xs[0]:.1f},{h-pad:.1f} {pts} {xs[-1]:.1f},{h-pad:.1f}' "
               f"fill='{col}' opacity='.10'/>")
    out.append(f"<line x1='{pad}' y1='{base_y:.1f}' x2='{w-12}' y2='{base_y:.1f}' "
               f"stroke='var(--mut)' stroke-dasharray='5,5'/>")
    out.append(f"<text x='{w-96}' y='{base_y-5:.1f}' font-size='10' fill='var(--mut)'>"
               f"start ${START_EQUITY/1000:.0f}k</text>")
    out.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.2'/>")
    out.append(f"<text x='{pad}' y='{h-10}' font-size='10' fill='var(--mut)'>{eq[0]['date']}</text>")
    out.append(f"<text x='{w-76}' y='{h-10}' font-size='10' fill='var(--mut)'>{eq[-1]['date']}</text>")
    out.append("</svg>")
    return "".join(out)


CSS = """<style>
:root{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e3e0dc;--card:#fff;
 --pos:#0a7a45;--neg:#c0392b;--accent:#2d5f8a}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#eceae5;--mut:#9a958d;
 --line:#33312c;--card:#1e1d18;--pos:#4ade80;--neg:#f87171;--accent:#7cb3e0}}
*{box-sizing:border-box}
body{margin:0;padding:26px;background:var(--bg);color:var(--fg);
 font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:14px;margin:30px 0 10px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:20px 24px;margin-bottom:12px}
.big{font-size:44px;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.grid{display:flex;flex-wrap:wrap;gap:24px;margin-top:14px}
.grid div{font-size:12.5px;color:var(--mut)}
.grid b{display:block;font-size:19px;color:var(--fg);font-weight:600;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;
 letter-spacing:.05em;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.flat{color:var(--mut)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:12px 16px;
 border-radius:0 7px 7px 0;color:var(--mut);font-size:13px;margin:12px 0}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:4px 8px}
.empty{color:var(--mut);font-style:italic;padding:12px 4px}
</style>"""


def write_report(st: dict, positions: list[Pos], universe: str, learner: Learner) -> None:
    openp = [p for p in positions if p.status == "OPEN"]
    closed = [p for p in positions if p.status == "CLOSED"]
    eq = st.get("equity", [])
    equity = eq[-1]["equity"] if eq else st["start_equity"]
    start = st["start_equity"]
    realised = sum(p.pnl for p in closed)
    unreal = sum(p.pnl for p in openp)

    h = []
    a = h.append
    a("<!doctype html><meta charset='utf-8'><title>RSI(2) Paper Account</title>")
    a(CSS)
    a("<h1>RSI(2) Scanner &mdash; Paper Account</h1>")
    a(f"<div class='sub'>Live since <b>{st['started']}</b> &middot; updated "
      f"{datetime.now(MARKET_TZ):%Y-%m-%d %H:%M} ET &middot; universe <b>{universe}</b> &middot; "
      f"max {MAX_POSITIONS} positions at {ALLOC_PCT:.0%} of equity each</div>")

    a("<div class='card'>")
    a(f"<div class='big {_c(equity-start)}'>${equity:,.2f}</div>")
    a("<div class='grid'>")
    a(f"<div>started<b>${start:,.0f}</b></div>")
    a(f"<div>return<b class='{_c(equity-start)}'>{(equity/start-1)*100:+.2f}%</b></div>")
    a(f"<div>cash<b>${st['cash']:,.0f}</b></div>")
    a(f"<div>realised<b class='{_c(realised)}'>${realised:+,.0f}</b></div>")
    a(f"<div>unrealised<b class='{_c(unreal)}'>${unreal:+,.0f}</b></div>")
    a(f"<div>positions<b>{len(closed)} closed / {len(openp)} open</b></div>")
    if closed:
        w = [p for p in closed if p.pnl > 0]
        l = [p for p in closed if p.pnl <= 0]
        pf = sum(p.pnl for p in w) / abs(sum(p.pnl for p in l)) if l and sum(p.pnl for p in l) else 0
        a(f"<div>win rate<b>{len(w)/len(closed)*100:.0f}%</b></div>")
        a(f"<div>profit factor<b>{pf:.2f}</b></div>")
        a(f"<div>avg hold<b>{np.mean([p.days_held for p in closed]):.1f}d</b></div>")
    a("</div></div>")

    a("<h2>Equity curve</h2>")
    a("<div class='card'>" + curve_svg(eq) + "</div>")

    d = decision(closed)
    tone = {"PASS": "pos", "NOT PROVEN": "neg", "COLLECTING": "flat", "TOO EARLY": "flat"}[d["status"]]
    a("<h2>Go / no-go for real money</h2>")
    a("<div class='card'>")
    a(f"<div style='font-size:26px;font-weight:650' class='{tone}'>{d['status']}</div>")
    a(f"<div style='margin-top:8px;color:var(--mut);font-size:13.5px'>{d['detail']}</div>")
    if d["n"] >= 5:
        pctdone = min(d["n"] / GO_LIVE_MIN_TRADES * 100, 100)
        a(f"<div style='margin-top:14px;height:7px;background:var(--line);border-radius:4px'>"
          f"<div style='width:{pctdone:.0f}%;height:7px;background:var(--accent);"
          f"border-radius:4px'></div></div>"
          f"<div style='color:var(--mut);font-size:11.5px;margin-top:5px'>"
          f"{d['n']} of {GO_LIVE_MIN_TRADES} trades needed &middot; current t = {d['t']:.2f} "
          f"&middot; bar is t &gt; {GO_LIVE_T}</div>")
    a("</div>")
    a("<div class='note'>The decision is made on the <b>share-equivalent</b> return of each "
      "signal, never on the option P&amp;L. Options carry 7.4x leverage on both the edge and "
      "the noise while the spread subtracts a fixed cost, so their signal-to-noise ratio is "
      "0.022 against 0.204 for shares &mdash; <b>8,635 option trades (66 years) versus 96 share "
      "trades (~8 months)</b> to reach the same confidence. A profitable option year proves "
      "nothing: with a <b>zero</b> edge, one year still ends above $20,000 about 37% of the "
      "time and reaches $49,659 at the 95th percentile.</div>")
    a(f"<div class='note'><b>Learning layer:</b> {learner.verdict}</div>")
    a("<div class='note'>Buys fill at the <b>ask</b>, sells at the <b>bid</b> &mdash; real "
      "contracts, real quotes. Paper only; no orders are ever sent. The validated edge "
      "(+0.509%/trade, t=8.70 on a 7-year holdout) is a <b>share</b> return; this measures what "
      "survives the options wrapper on a real-sized account.</div>")

    a("<h2>Open positions</h2>")
    if openp:
        a("<div class='wrap'><table><tr><th>Ticker</th><th>Contract</th><th>Opened</th>"
          "<th class='num'>Qty</th><th class='num'>Days</th><th class='num'>Cost</th>"
          "<th class='num'>Value</th><th class='num'>Spot %</th>"
          "<th class='num'>P&amp;L $</th><th class='num'>P&amp;L %</th></tr>")
        for p in sorted(openp, key=lambda x: x.opened):
            a(f"<tr><td><b>{p.ticker}</b></td><td>{p.strike:g}C {p.expiry}</td>"
              f"<td>{p.opened}</td><td class='num'>{p.contracts}</td>"
              f"<td class='num'>{p.days_held}</td><td class='num'>${p.cost:,.0f}</td>"
              f"<td class='num'>${p.value:,.0f}</td>"
              f"<td class='num {_c(p.spot_pct)}'>{p.spot_pct:+.2f}%</td>"
              f"<td class='num {_c(p.pnl)}'>${p.pnl:+,.0f}</td>"
              f"<td class='num {_c(p.pnl)}'>{p.pnl_pct:+.1f}%</td></tr>")
        a("</table></div>")
    else:
        a("<div class='empty'>No open positions.</div>")

    a(f"<h2>All closed trades ({len(closed)})</h2>")
    if closed:
        a("<div class='wrap'><table><tr><th>Ticker</th><th>Sector</th><th>Contract</th>"
          "<th>In</th><th>Out</th><th class='num'>Days</th><th class='num'>Qty</th>"
          "<th class='num'>Cost</th><th class='num'>P&amp;L $</th><th class='num'>%</th>"
          "<th>Exit</th></tr>")
        for p in sorted(closed, key=lambda x: x.exit_date or "", reverse=True):
            a(f"<tr><td><b>{p.ticker}</b></td><td>{p.sector}</td>"
              f"<td>{p.strike:g}C {p.expiry}</td><td>{p.opened}</td><td>{p.exit_date}</td>"
              f"<td class='num'>{p.days_held}</td><td class='num'>{p.contracts}</td>"
              f"<td class='num'>${p.cost:,.0f}</td>"
              f"<td class='num {_c(p.pnl)}'>${p.pnl:+,.0f}</td>"
              f"<td class='num {_c(p.pnl)}'>{p.pnl_pct:+.1f}%</td>"
              f"<td>{p.exit_reason}</td></tr>")
        a("</table></div>")
    else:
        a("<div class='empty'>Nothing closed yet.</div>")

    missed = st.get("missed", [])
    if missed:
        a(f"<h2>Signals missed ({len(missed)})</h2>")
        a("<div class='note'>A $20k account cannot take every signal. These are the ones it had "
          "to pass on &mdash; and they cluster on broad selloff days, which is exactly when the "
          "setup fires most.</div>")
        a("<div class='wrap'><table><tr><th>Date</th><th>Ticker</th><th>Reason</th></tr>")
        for m in missed[-40:][::-1]:
            a(f"<tr><td>{m['date']}</td><td><b>{m['ticker']}</b></td><td>{m['why']}</td></tr>")
        a("</table></div>")

    if eq:
        a("<h2>Daily equity</h2>")
        a("<div class='wrap'><table><tr><th>Date</th><th class='num'>Equity</th>"
          "<th class='num'>Change</th><th class='num'>Cash</th>"
          "<th class='num'>Realised</th><th class='num'>Open</th></tr>")
        prev = None
        for e in eq[::-1][:60]:
            idx = eq.index(e)
            chg = e["equity"] - (eq[idx-1]["equity"] if idx > 0 else start)
            a(f"<tr><td>{e['date']}</td><td class='num'>${e['equity']:,.2f}</td>"
              f"<td class='num {_c(chg)}'>${chg:+,.2f}</td>"
              f"<td class='num'>${e['cash']:,.0f}</td>"
              f"<td class='num {_c(e['realised'])}'>${e['realised']:+,.0f}</td>"
              f"<td class='num'>{e['open']}</td></tr>")
        a("</table></div>")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(h), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="strong")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="wipe and restart at $20,000 today")
    args = ap.parse_args()
    if args.universe not in PRESETS:
        raise SystemExit(f"unknown universe. choose from: {', '.join(sorted(PRESETS))}")
    if args.reset and STATE.exists():
        STATE.unlink()
        print("account reset\n")
    run(args.universe, args.dry_run)


if __name__ == "__main__":
    main()
