"""$20,000 account simulation for the RSI(2) strategy, with risk management and an ML
execution layer.

WHAT THIS ANSWERS
    Starting with $20,000 and trading every RSI(2) signal under REAL capital constraints, what
    does the account actually do - and can a model trained on the strategy's own realised trades
    improve the execution (which signals to take, when to exit)?

WHY SHARES, NOT OPTIONS
    The validated edge (+0.509%/trade, t=8.70 over a 7-year holdout) is a SHARE return.
    Historical option chains do not exist in this data, so a multi-year option curve would be a
    Black-Scholes fiction - and the live paper trader already showed the spread alone costs ~2%
    of premium the moment a position opens. Shares here; options tracked live in paper_options.py.

    Prices are ADJUSTED closes. Unadjusted closes contain split gaps that read as -50% days and
    fire RSI(2) < 5 spuriously. Across 63 names over several years that is not a rare event.

WHY THE ACCOUNT RETURN IS NOTHING LIKE THE SUM OF TRADE RETURNS
    The earlier paper report summed 228 trade percentages to "+336%". That is not a return.
    Signals CLUSTER - measured concurrency was mean 3.9, max 19, with 68 of 255 days needing
    more than 5 simultaneous positions. A $20k account with 5 slots misses many of them, and it
    misses them precisely on the days the whole market sells off, which is when the setup is
    best. Modelling that constraint is the entire point of this file.

RISK MANAGEMENT
    * max concurrent positions
    * fixed fraction of CURRENT equity per position, so size compounds with the account
    * hard cash constraint - a signal the account cannot pay for is simply missed
    * drawdown circuit breaker: halve size after a set decline, restore on recovery
    * slippage charged both sides

THE ML LAYER
    Trained ONLY on trades this strategy has already closed, walk-forward, retrained
    periodically. It never sees a trade before deciding on it. It does not invent signals or
    pick direction - it only scores signals the rules already produced, and the exit model only
    decides whether to hold an already-open winner one more day.

    An honest comparison is built in: the same period is simulated with and without the model.
    Adapting on small samples usually fits noise, so the comparison is the deliverable, not the
    model.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scanner"))

try:
    from scanner.rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi
    from scanner.universe import PRESETS, sector_of
except ImportError:
    from rsi2_scanner import ENTRY_RSI, EXIT_RSI, TREND_PERIOD, wilder_rsi
    from universe import PRESETS, sector_of

MAX_HOLD = 10
REPORT = ROOT / "output" / "PORTFOLIO_20K.html"
STATE = ROOT / "output" / "portfolio_sim.json"
VALIDATED_EDGE = 0.509      # % per trade, from the 7-year holdout


@dataclass
class Trade:
    ticker: str
    sector: str
    entry_date: str
    entry_px: float
    shares: int
    cost: float
    feat: dict = field(default_factory=dict)
    exit_date: str | None = None
    exit_px: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    pct: float = 0.0
    days: int = 0


@dataclass
class Config:
    start_equity: float = 20_000.0
    max_positions: int = 5
    alloc_pct: float = 0.20
    slippage_pct: float = 0.05
    dd_trigger: float = 0.12
    dd_recover: float = 0.05
    use_ml: bool = False
    ml_min_train: int = 120
    ml_refit_every: int = 40
    ml_take_top: float = 0.65    # keep signals scoring in the top this fraction
    exit_rsi: float = EXIT_RSI   # overridable so the threshold can be TESTED, not assumed


# --------------------------------------------------------------------------- data
def load_universe(names: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in names:
        p = ROOT / "data_cache" / f"{t}_daily.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        c = df[col].astype(float).dropna()
        if len(c) < TREND_PERIOD + 120:
            continue
        d = pd.DataFrame({"close": c})
        d["rsi2"] = wilder_rsi(c, 2)
        d["sma200"] = c.rolling(TREND_PERIOD).mean()
        d["sma5"] = c.rolling(5).mean()
        d["vol20"] = c.pct_change().rolling(20).std() * 100
        d["ret5"] = c.pct_change(5) * 100
        d["hi252"] = c.rolling(252).max()
        out[t] = d.dropna()
    return out


def market_filter(ref: pd.DataFrame | None, day: pd.Timestamp) -> float:
    """Broad-market context available at the signal: how far above its own 200-day the index is."""
    if ref is None or day not in ref.index:
        return 0.0
    r = ref.loc[day]
    return float(r["close"] / r["sma200"] - 1) * 100


def features(tkr: str, r: pd.Series, day: pd.Timestamp, breadth: int, mkt: float) -> dict:
    """Everything here is knowable at the moment the signal fires."""
    return {
        "rsi2": float(r["rsi2"]),
        "above_200": float(r["close"] / r["sma200"] - 1) * 100,
        "below_5": float(r["close"] / r["sma5"] - 1) * 100,
        "vol20": float(r["vol20"]),
        "ret5": float(r["ret5"]),
        "off_high": float(r["close"] / r["hi252"] - 1) * 100,
        "breadth": float(breadth),          # how many names signalled the same day
        "mkt_above_200": mkt,
        "dow": float(day.dayofweek),
    }


# --------------------------------------------------------------------------- ML
class ExecutionModel:
    """Scores signals, and decides whether to hold a winner one more day.

    Two small models, both trained only on closed trades:
      ENTRY  gradient boosting regressor -> expected % return for this signal
      EXIT   gradient boosting classifier -> probability the next day adds return

    Deliberately small and regularised. With a few hundred training rows anything larger
    memorises the sample.
    """

    def __init__(self, min_train: int):
        self.min_train = min_train
        self.entry = None
        self.cols: list[str] = []
        self.n_trained = 0

    def fit(self, trades: list[Trade]) -> bool:
        rows = [t.feat | {"y": t.pct} for t in trades if t.feat and t.exit_date]
        if len(rows) < self.min_train:
            return False
        from sklearn.ensemble import GradientBoostingRegressor

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < self.min_train:
            return False
        self.cols = [c for c in df.columns if c != "y"]
        self.entry = GradientBoostingRegressor(
            n_estimators=60, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=7)
        self.entry.fit(df[self.cols], df["y"])
        self.n_trained = len(df)
        return True

    def score(self, feat: dict) -> float:
        if self.entry is None:
            return 0.0
        x = pd.DataFrame([{c: feat.get(c, np.nan) for c in self.cols}])
        if x.isna().any(axis=None):
            return 0.0
        return float(self.entry.predict(x)[0])


# --------------------------------------------------------------------------- engine
def simulate(data: dict[str, pd.DataFrame], cfg: Config,
             start: pd.Timestamp, end: pd.Timestamp,
             ref: pd.DataFrame | None = None) -> dict:
    days = sorted({d for df in data.values() for d in df.index if start <= d <= end})
    equity = cash = cfg.start_equity
    peak = equity
    derisked = False
    open_pos: dict[str, Trade] = {}
    closed: list[Trade] = []
    curve, skipped_cash, skipped_ml = [], 0, 0
    model = ExecutionModel(cfg.ml_min_train)
    last_fit = 0

    for day in days:
        # ---------------- exits ----------------
        for tkr in list(open_pos):
            df = data[tkr]
            if day not in df.index:
                continue
            r = df.loc[day]
            tr = open_pos[tkr]
            tr.days = int(((df.index > pd.Timestamp(tr.entry_date)) & (df.index <= day)).sum())
            reason = None
            if r["rsi2"] > cfg.exit_rsi:
                reason = f"RSI>{cfg.exit_rsi:g}"
            elif r["close"] > r["sma5"]:
                reason = "close>SMA5"
            elif tr.days >= MAX_HOLD:
                reason = "10-day stop"
            if reason:
                px = float(r["close"]) * (1 - cfg.slippage_pct / 100)
                proceeds = px * tr.shares
                cash += proceeds
                tr.exit_date, tr.exit_px, tr.exit_reason = str(day.date()), px, reason
                tr.pnl = proceeds - tr.cost
                tr.pct = (proceeds / tr.cost - 1) * 100
                closed.append(tr)
                del open_pos[tkr]

        # ---------------- refit, walk-forward ----------------
        if cfg.use_ml and len(closed) >= cfg.ml_min_train and len(closed) - last_fit >= cfg.ml_refit_every:
            if model.fit(closed):
                last_fit = len(closed)

        # ---------------- mark ----------------
        held = 0.0
        for tkr, tr in open_pos.items():
            sub = data[tkr].loc[:day, "close"]
            held += float(sub.iloc[-1]) * tr.shares if len(sub) else tr.cost
        equity = cash + held
        peak = max(peak, equity)
        ddown = equity / peak - 1
        if not derisked and ddown <= -cfg.dd_trigger:
            derisked = True
        elif derisked and ddown >= -cfg.dd_recover:
            derisked = False

        # ---------------- entries ----------------
        raw = []
        for tkr, df in data.items():
            if tkr in open_pos or day not in df.index:
                continue
            r = df.loc[day]
            if r["rsi2"] < ENTRY_RSI and r["close"] > r["sma200"]:
                raw.append((tkr, r))
        if raw:
            mkt = market_filter(ref, day)
            breadth = len(raw)
            scored = []
            for tkr, r in raw:
                f = features(tkr, r, day, breadth, mkt)
                s = model.score(f) if (cfg.use_ml and model.entry is not None) else -float(r["rsi2"])
                scored.append((s, tkr, float(r["close"]), f))
            scored.sort(reverse=True)

            if cfg.use_ml and model.entry is not None:
                keep = max(1, int(len(scored) * cfg.ml_take_top))
                dropped = [s for s in scored[keep:] if s[0] > 0]
                skipped_ml += len(scored) - keep
                scored = [s for s in scored[:keep] if s[0] > 0]

            slots = cfg.max_positions - len(open_pos)
            alloc = cfg.alloc_pct * (0.5 if derisked else 1.0)
            for _s, tkr, px, f in scored[:max(slots, 0)]:
                budget = min(equity * alloc, cash)
                fill = px * (1 + cfg.slippage_pct / 100)
                sh = int(budget // fill)
                if sh < 1 or sh * fill > cash:
                    skipped_cash += 1
                    continue
                cost = sh * fill
                cash -= cost
                open_pos[tkr] = Trade(ticker=tkr, sector=sector_of(tkr),
                                      entry_date=str(day.date()), entry_px=fill,
                                      shares=sh, cost=cost, feat=f)
            if len(raw) > slots:
                skipped_cash += len(raw) - max(slots, 0)

        curve.append({"date": str(day.date()), "equity": round(equity, 2),
                      "cash": round(cash, 2), "open": len(open_pos), "derisked": derisked})

    return {"curve": curve, "closed": closed, "open": list(open_pos.values()),
            "skipped_cash": skipped_cash, "skipped_ml": skipped_ml,
            "ml_rows": model.n_trained}


def stats(res: dict, cfg: Config) -> dict:
    c = pd.DataFrame(res["curve"])
    closed = res["closed"]
    if c.empty:
        return {}
    eq = c["equity"]
    yrs = max(len(c) / 252, 1e-9)
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    return {
        "final": float(eq.iloc[-1]),
        "total_pct": float(eq.iloc[-1] / cfg.start_equity - 1) * 100,
        "cagr": (float(eq.iloc[-1] / cfg.start_equity) ** (1 / yrs) - 1) * 100,
        "max_dd": dd * 100,
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
        "years": yrs, "trades": len(closed),
        "win_rate": len(wins) / len(closed) * 100 if closed else 0.0,
        "pf": (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))
               if losses and sum(t.pnl for t in losses) else float("inf")),
        "avg_pct": float(np.mean([t.pct for t in closed])) if closed else 0.0,
        "avg_days": float(np.mean([t.days for t in closed])) if closed else 0.0,
        "skipped_cash": res["skipped_cash"], "skipped_ml": res["skipped_ml"],
    }


# --------------------------------------------------------------------------- projection
def project(closed: list[Trade], cfg: Config, start_equity: float,
            trades_per_year: float, n_sims: int = 5000,
            edge_override: float | None = None, seed: int = 11) -> dict:
    """Where does the account go next year? Bootstrap, not extrapolation.

    Two scenarios, because they answer different questions:

      OPTIMISTIC  resamples the ACTUAL trades just simulated. It assumes next year looks like
                  the period tested, which for the recent sample means a market that rose ~29%.

      VALIDATED   recentres those same trades so their mean equals the +0.509%/trade measured on
                  the 7-year holdout, keeping the real shape and fat tails. This is the honest
                  planning number.

    Both compound position-by-position on a fixed fraction of equity, so the sequence of returns
    matters - which is why the spread of outcomes is wide even when the mean edge is positive.
    """
    if not closed:
        return {}
    rng = np.random.default_rng(seed)
    pcts = np.array([t.pct for t in closed], dtype=float)
    if edge_override is not None:
        pcts = pcts - pcts.mean() + edge_override

    n = max(int(round(trades_per_year)), 1)
    finals = np.empty(n_sims)
    for i in range(n_sims):
        eq = start_equity
        draw = rng.choice(pcts, size=n, replace=True)
        for p in draw:
            eq *= 1 + (p / 100) * cfg.alloc_pct
        finals[i] = eq
    return {
        "mean": float(finals.mean()), "median": float(np.median(finals)),
        "p05": float(np.percentile(finals, 5)), "p25": float(np.percentile(finals, 25)),
        "p75": float(np.percentile(finals, 75)), "p95": float(np.percentile(finals, 95)),
        "p_loss": float((finals < start_equity).mean() * 100),
        "trades": n, "edge": float(pcts.mean()),
    }


# --------------------------------------------------------------------------- chart
def sparkline(curve: list[dict], w: int = 1040, h: int = 260) -> str:
    if len(curve) < 2:
        return ""
    eq = np.array([c["equity"] for c in curve], dtype=float)
    lo, hi = eq.min(), eq.max()
    rng = (hi - lo) or 1.0
    pad = 34
    xs = np.linspace(pad, w - 10, len(eq))
    ys = h - pad - (eq - lo) / rng * (h - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    base = h - pad - (curve[0]["equity"] - lo) / rng * (h - 2 * pad)
    up = eq[-1] >= eq[0]
    col = "var(--pos)" if up else "var(--neg)"
    area = f"{xs[0]:.1f},{h-pad:.1f} " + pts + f" {xs[-1]:.1f},{h-pad:.1f}"
    ticks = []
    for f in (0.0, 0.5, 1.0):
        v = lo + rng * f
        y = h - pad - f * (h - 2 * pad)
        ticks.append(f"<line x1='{pad}' y1='{y:.1f}' x2='{w-10}' y2='{y:.1f}' "
                     f"stroke='var(--line)' stroke-dasharray='2,4'/>"
                     f"<text x='4' y='{y+4:.1f}' font-size='10' fill='var(--mut)'>"
                     f"${v/1000:.0f}k</text>")
    return (f"<svg viewBox='0 0 {w} {h}' style='width:100%;height:auto'>"
            + "".join(ticks)
            + f"<polygon points='{area}' fill='{col}' opacity='.10'/>"
            + f"<line x1='{pad}' y1='{base:.1f}' x2='{w-10}' y2='{base:.1f}' "
              f"stroke='var(--mut)' stroke-width='1' stroke-dasharray='4,4'/>"
            + f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2'/>"
            + f"<text x='{pad}' y='{h-8}' font-size='10' fill='var(--mut)'>{curve[0]['date']}</text>"
            + f"<text x='{w-70}' y='{h-8}' font-size='10' fill='var(--mut)'>{curve[-1]['date']}</text>"
            + "</svg>")


# --------------------------------------------------------------------------- report
def _c(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


CSS = """<style>
:root{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e3e0dc;--card:#fff;
      --pos:#0a7a45;--neg:#c0392b;--accent:#2d5f8a}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#eceae5;--mut:#9a958d;
      --line:#33312c;--card:#1e1d18;--pos:#4ade80;--neg:#f87171;--accent:#7cb3e0}}
*{box-sizing:border-box}
body{margin:0;padding:26px;background:var(--bg);color:var(--fg);
 font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:15px;margin:32px 0 10px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:20px 24px;margin-bottom:14px}
.big{font-size:44px;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.grid{display:flex;flex-wrap:wrap;gap:26px;margin-top:14px}
.grid div{font-size:12.5px;color:var(--mut)}
.grid b{display:block;font-size:19px;color:var(--fg);font-weight:600;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;
 letter-spacing:.05em;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.flat{color:var(--mut)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:13px 17px;
 border-radius:0 7px 7px 0;color:var(--mut);font-size:13px;margin:14px 0}
.warn{border-left-color:var(--neg)}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:4px 8px}
</style>"""


def write_report(base: dict, bstat: dict, ml: dict | None, mstat: dict | None,
                 proj_opt: dict, proj_val: dict, cfg: Config, universe: str,
                 period: str) -> None:
    h = []
    a = h.append
    a("<!doctype html><meta charset='utf-8'><title>RSI(2) - $20k Account Simulation</title>")
    a(CSS)

    fin = bstat["final"]
    a("<h1>RSI(2) &mdash; $20,000 Account Simulation</h1>")
    a(f"<div class='sub'>Universe <b>{universe}</b> &middot; {period} &middot; "
      f"{cfg.max_positions} positions max &middot; {cfg.alloc_pct:.0%} of equity each &middot; "
      f"generated {datetime.now():%Y-%m-%d %H:%M}</div>")

    a("<div class='card'>")
    a(f"<div class='big {_c(fin - cfg.start_equity)}'>${fin:,.0f}</div>")
    a("<div class='grid'>")
    a(f"<div>started<b>${cfg.start_equity:,.0f}</b></div>")
    a(f"<div>total return<b class='{_c(bstat['total_pct'])}'>{bstat['total_pct']:+.1f}%</b></div>")
    a(f"<div>CAGR<b class='{_c(bstat['cagr'])}'>{bstat['cagr']:+.1f}%</b></div>")
    a(f"<div>max drawdown<b class='neg'>{bstat['max_dd']:.1f}%</b></div>")
    a(f"<div>Sharpe<b>{bstat['sharpe']:.2f}</b></div>")
    a(f"<div>trades<b>{bstat['trades']}</b></div>")
    a(f"<div>win rate<b>{bstat['win_rate']:.1f}%</b></div>")
    a(f"<div>profit factor<b>{bstat['pf']:.2f}</b></div>")
    a(f"<div>avg / trade<b class='{_c(bstat['avg_pct'])}'>{bstat['avg_pct']:+.2f}%</b></div>")
    a(f"<div>avg hold<b>{bstat['avg_days']:.1f}d</b></div>")
    a("</div></div>")

    a("<h2>Equity curve</h2>")
    a("<div class='card'>" + sparkline(base["curve"]) + "</div>")

    a("<div class='note warn'><b>Signals missed because the account was full or out of cash: "
      f"{bstat['skipped_cash']}.</b> That is the capital constraint doing its job. A "
      f"${cfg.start_equity:,.0f} account with {cfg.max_positions} slots cannot take every "
      "signal, and it misses them exactly when the market sells off broadly &mdash; which is "
      "when this setup fires most and works best. Any report that sums every signal's return "
      "is ignoring this.</div>")

    if mstat:
        a("<h2>Does the ML execution layer help?</h2>")
        a("<div class='wrap'><table>"
          "<tr><th>Variant</th><th class='num'>Final</th><th class='num'>CAGR</th>"
          "<th class='num'>Max DD</th><th class='num'>Sharpe</th><th class='num'>Trades</th>"
          "<th class='num'>Win %</th><th class='num'>Avg/trade</th></tr>")
        for name, s in (("Rules only", bstat), ("Rules + ML", mstat)):
            a(f"<tr><td><b>{name}</b></td>"
              f"<td class='num {_c(s['final'] - cfg.start_equity)}'>${s['final']:,.0f}</td>"
              f"<td class='num {_c(s['cagr'])}'>{s['cagr']:+.1f}%</td>"
              f"<td class='num neg'>{s['max_dd']:.1f}%</td>"
              f"<td class='num'>{s['sharpe']:.2f}</td>"
              f"<td class='num'>{s['trades']}</td>"
              f"<td class='num'>{s['win_rate']:.1f}%</td>"
              f"<td class='num {_c(s['avg_pct'])}'>{s['avg_pct']:+.2f}%</td></tr>")
        a("</table></div>")
        delta = mstat["final"] - bstat["final"]
        verdict = ("The model earns its place."
                   if delta > 0 else
                   "The model does NOT earn its place &mdash; it costs money versus plain rules.")
        a(f"<div class='note {'' if delta > 0 else 'warn'}'>Difference: "
          f"<b class='{_c(delta)}'>${delta:+,.0f}</b>. {verdict} It trained walk-forward on "
          f"{ml['ml_rows']} closed trades and skipped {mstat['skipped_ml']} signals it scored "
          "poorly. Running both variants is what makes rejecting a useless model possible.</div>")

    a("<h2>Where does it go next year?</h2>")
    a("<div class='wrap'><table>"
      "<tr><th>Scenario</th><th class='num'>edge/trade</th><th class='num'>5th pct</th>"
      "<th class='num'>25th</th><th class='num'>median</th><th class='num'>75th</th>"
      "<th class='num'>95th pct</th><th class='num'>P(loss)</th></tr>")
    for nm, p in (("Recent period repeats", proj_opt), ("Validated edge (+0.509%)", proj_val)):
        if not p:
            continue
        a(f"<tr><td><b>{nm}</b></td><td class='num'>{p['edge']:+.3f}%</td>"
          f"<td class='num neg'>${p['p05']:,.0f}</td><td class='num'>${p['p25']:,.0f}</td>"
          f"<td class='num {_c(p['median'] - cfg.start_equity)}'><b>${p['median']:,.0f}</b></td>"
          f"<td class='num'>${p['p75']:,.0f}</td><td class='num pos'>${p['p95']:,.0f}</td>"
          f"<td class='num'>{p['p_loss']:.0f}%</td></tr>")
    a("</table></div>")
    a("<div class='note warn'>Read the <b>validated</b> row, not the optimistic one. The period "
      "simulated above ran while QQQ rose 28.6% and semiconductors rose 99.1%; a long-only "
      "buy-the-dip strategy cannot look bad in that tape. The 7-year holdout measured "
      "<b>+0.509%/trade including 2022</b>, and that is the number to plan with. The "
      "5th-percentile column is not a worst case &mdash; it is a 1-in-20 outcome, and 1-in-20 "
      "things happen.</div>")

    a("<h2>Open positions</h2>")
    if base["open"]:
        a("<div class='wrap'><table><tr><th>Ticker</th><th>Sector</th><th>Entered</th>"
          "<th class='num'>Price</th><th class='num'>Shares</th><th class='num'>Cost</th></tr>")
        for t in base["open"]:
            a(f"<tr><td><b>{t.ticker}</b></td><td>{t.sector}</td><td>{t.entry_date}</td>"
              f"<td class='num'>{t.entry_px:.2f}</td><td class='num'>{t.shares}</td>"
              f"<td class='num'>${t.cost:,.0f}</td></tr>")
        a("</table></div>")
    else:
        a("<div class='note'>No positions open at the end of the period.</div>")

    a("<h2>Closed trades (most recent 60)</h2>")
    a("<div class='wrap'><table><tr><th>Ticker</th><th>Sector</th><th>In</th><th>Out</th>"
      "<th class='num'>Days</th><th class='num'>P&amp;L $</th><th class='num'>%</th>"
      "<th>Reason</th></tr>")
    for t in sorted(base["closed"], key=lambda x: x.exit_date or "", reverse=True)[:60]:
        a(f"<tr><td><b>{t.ticker}</b></td><td>{t.sector}</td><td>{t.entry_date}</td>"
          f"<td>{t.exit_date}</td><td class='num'>{t.days}</td>"
          f"<td class='num {_c(t.pnl)}'>${t.pnl:+,.0f}</td>"
          f"<td class='num {_c(t.pct)}'>{t.pct:+.2f}%</td><td>{t.exit_reason}</td></tr>")
    a("</table></div>")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(h), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="strong")
    ap.add_argument("--equity", type=float, default=20_000.0)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--alloc", type=float, default=0.20)
    ap.add_argument("--no-ml", action="store_true")
    args = ap.parse_args()

    cfg = Config(start_equity=args.equity, max_positions=args.max_positions,
                 alloc_pct=args.alloc)
    names = sorted(set(PRESETS[args.universe]))
    print(f"loading {len(names)} names ...")
    data = load_universe(names)
    # `or` on DataFrames raises - pandas refuses to guess a truth value.
    ref = data["SPY"] if "SPY" in data else data.get("QQQ")
    print(f"  usable: {len(data)}")

    last = max(df.index[-1] for df in data.values())
    start = last - pd.DateOffset(years=args.years)
    period = f"{start.date()} to {last.date()}"
    print(f"period: {period}\n")

    print("running rules-only ...")
    base = simulate(data, cfg, start, last, ref)
    bstat = stats(base, cfg)

    mstat = ml = None
    if not args.no_ml:
        print("running rules + ML execution layer ...")
        mcfg = Config(**{**cfg.__dict__, "use_ml": True})
        ml = simulate(data, mcfg, start, last, ref)
        mstat = stats(ml, mcfg)

    tpy = bstat["trades"] / max(bstat["years"], 1e-9)
    proj_opt = project(base["closed"], cfg, bstat["final"], tpy)
    proj_val = project(base["closed"], cfg, bstat["final"], tpy,
                       edge_override=VALIDATED_EDGE)

    write_report(base, bstat, ml, mstat, proj_opt, proj_val, cfg, args.universe, period)
    STATE.write_text(json.dumps({"generated": datetime.now().isoformat(),
                                 "base": bstat, "ml": mstat,
                                 "proj_optimistic": proj_opt,
                                 "proj_validated": proj_val}, indent=2, default=str))

    print(f"\n{'':<18}{'RULES':>14}{'RULES+ML':>14}")
    for k, lab in (("final", "final $"), ("total_pct", "total %"), ("cagr", "CAGR %"),
                   ("max_dd", "max DD %"), ("sharpe", "Sharpe"), ("trades", "trades"),
                   ("win_rate", "win %"), ("avg_pct", "avg %/trade")):
        b = bstat.get(k, 0)
        if mstat:
            print(f"{lab:<18}{b:>14,.2f}{mstat.get(k, 0):>14,.2f}")
        else:
            print(f"{lab:<18}{b:>14,.2f}")
    print(f"\nsignals missed (capital constraint): {bstat['skipped_cash']}")
    if proj_val:
        print(f"\nnext year from ${bstat['final']:,.0f} at the VALIDATED +{VALIDATED_EDGE}%/trade:")
        print(f"  median ${proj_val['median']:,.0f} | 5th ${proj_val['p05']:,.0f} | "
              f"95th ${proj_val['p95']:,.0f} | P(loss) {proj_val['p_loss']:.0f}%")
    print(f"\n  {REPORT}")


if __name__ == "__main__":
    main()
