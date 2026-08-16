"""Earnings and news context for a signal.

This exists to address a known blind spot rather than to add decoration. The strategy assumes
an oversold dip reverts, which holds when the cause is a market-wide flush and fails when the
cause is company-specific news that repriced the stock permanently. Nothing in RSI(2) can tell
those apart. Earnings dates and recent headlines can.

Two things are surfaced:

  EARNINGS PROXIMITY - the trade holds 3-4 sessions, so an earnings report inside that window
  is an event the backtest never modelled. An earnings gap routinely exceeds the entire
  expected move in either direction, and no stop protects against it.

  PAST EARNINGS - beat/miss history with surprise percentages, which says whether the company
  has been delivering or disappointing.

On news, a deliberate limitation: this does NOT score sentiment. Automated sentiment on
financial headlines is unreliable enough to be actively misleading - "CVS falls on strong
guidance" and "CVS falls despite strong guidance" score identically to a keyword model, and a
confident-looking label would be worse than no label. Instead the headlines are shown with
their timestamps, and only unambiguous risk language (investigation, guidance cut, downgrade,
probe, recall) is flagged. The judgement stays with the reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# Terms whose appearance in a headline is reliably bad news for a long position. Chosen to be
# unambiguous - words like "falls" or "drops" are excluded because they describe the move we
# already know about rather than telling us anything about its cause.
RISK_TERMS = [
    "investigation", "investigating", "probe", "subpoena", "lawsuit", "sues", "litigation",
    "guidance cut", "cuts guidance", "lowers guidance", "slashes", "warns", "warning",
    "downgrade", "downgrades", "downgraded",
    "recall", "fda rejects", "fails trial", "halts", "suspended",
    "misses", "miss", "shortfall", "disappoints",
    "resigns", "steps down", "ceo departs", "cfo departs",
    "fraud", "accounting", "restatement", "sec charges",
    "bankruptcy", "default", "going concern",
]


@dataclass
class EarningsContext:
    next_date: pd.Timestamp | None = None
    days_until: int | None = None
    in_trade_window: bool = False
    past: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def label(self) -> str:
        if self.error or self.days_until is None:
            return "?"
        if self.days_until < 0:
            return "past"
        if self.days_until == 0:
            return "TODAY"
        return f"{self.days_until}d"

    def beat_record(self, n: int = 4) -> str:
        recent = [p for p in self.past if p.get("surprise") is not None][:n]
        if not recent:
            return "-"
        beats = sum(1 for p in recent if p["surprise"] > 0)
        return f"{beats}/{len(recent)} beat"


@dataclass
class NewsContext:
    headlines: list[dict] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    fresh_count: int = 0  # published within the last 48 hours
    error: str = ""

    @property
    def label(self) -> str:
        if self.error:
            return "?"
        if self.risk_flags:
            return f"RISK({len(self.risk_flags)})"
        if self.fresh_count == 0:
            return "quiet"
        return f"{self.fresh_count} fresh"


def get_earnings(ticker: str, hold_days: int = 4) -> EarningsContext:
    ctx = EarningsContext()
    try:
        tk = yf.Ticker(ticker)
        today = datetime.now(timezone.utc).date()

        try:
            frame = tk.earnings_dates
        except Exception:
            frame = None

        if frame is not None and not frame.empty:
            for idx, row in frame.iterrows():
                date = idx.date() if hasattr(idx, "date") else idx
                reported = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                surprise = row.get("Surprise(%)")
                if pd.notna(reported):
                    ctx.past.append(
                        {
                            "date": date,
                            "estimate": float(estimate) if pd.notna(estimate) else None,
                            "reported": float(reported),
                            "surprise": float(surprise) if pd.notna(surprise) else None,
                        }
                    )
            ctx.past.sort(key=lambda p: p["date"], reverse=True)

            # Next scheduled report: the nearest future date with no reported figure yet.
            future = [
                (idx.date() if hasattr(idx, "date") else idx)
                for idx, row in frame.iterrows()
                if pd.isna(row.get("Reported EPS"))
                and (idx.date() if hasattr(idx, "date") else idx) >= today
            ]
            if future:
                ctx.next_date = min(future)

        # The calendar endpoint is sometimes fresher than earnings_dates.
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date") or []
                upcoming = [d for d in dates if d >= today]
                if upcoming and (ctx.next_date is None or min(upcoming) < ctx.next_date):
                    ctx.next_date = min(upcoming)
        except Exception:
            pass

        if ctx.next_date is not None:
            ctx.days_until = (ctx.next_date - today).days
            ctx.in_trade_window = 0 <= ctx.days_until <= hold_days
    except Exception as exc:
        ctx.error = type(exc).__name__
    return ctx


def get_news(ticker: str, limit: int = 6) -> NewsContext:
    ctx = NewsContext()
    try:
        items = yf.Ticker(ticker).news or []
        now = datetime.now(timezone.utc)
        for item in items[:limit]:
            content = item.get("content") or item
            title = content.get("title") or ""
            if not title:
                continue
            published = content.get("pubDate") or content.get("displayTime")
            when = None
            if published:
                try:
                    when = pd.Timestamp(published).tz_convert("UTC")
                except Exception:
                    try:
                        when = pd.Timestamp(published, unit="s", tz="UTC")
                    except Exception:
                        when = None
            age_h = (now - when.to_pydatetime()).total_seconds() / 3600 if when is not None else None

            provider = content.get("provider") or {}
            source = provider.get("displayName") if isinstance(provider, dict) else str(provider)

            lowered = title.lower()
            hits = [term for term in RISK_TERMS if term in lowered]
            if hits:
                ctx.risk_flags.extend(hits)
            if age_h is not None and age_h <= 48:
                ctx.fresh_count += 1

            ctx.headlines.append(
                {
                    "title": title,
                    "source": source or "?",
                    "age_hours": age_h,
                    "risk_terms": hits,
                    "summary": (content.get("summary") or "")[:200],
                }
            )
        ctx.risk_flags = sorted(set(ctx.risk_flags))
    except Exception as exc:
        ctx.error = type(exc).__name__
    return ctx


def render_context(ticker: str, earnings: EarningsContext, news: NewsContext) -> None:
    print(f"\n    EARNINGS")
    if earnings.error:
        print(f"      unavailable ({earnings.error})")
    else:
        if earnings.next_date is not None:
            when = f"{earnings.next_date} ({earnings.label} away)"
            print(f"      next report: {when}")
            if earnings.in_trade_window:
                print("      *** INSIDE THE EXPECTED HOLD WINDOW ***")
                print("          This trade normally lasts 3-4 sessions, so it would be open")
                print("          through the report. An earnings gap routinely exceeds the whole")
                print("          expected move, and no stop executes across it. The backtest")
                print("          never modelled this - it treats every trade as news-free.")
        else:
            print("      next report: unknown")

        if earnings.past:
            print(f"      last 4 reports ({earnings.beat_record()}):")
            for p in earnings.past[:4]:
                if p["surprise"] is None:
                    print(f"        {p['date']}  reported {p['reported']:.2f}")
                else:
                    verdict = "BEAT" if p["surprise"] > 0 else "MISS"
                    est = f"{p['estimate']:.2f}" if p["estimate"] is not None else "?"
                    print(f"        {p['date']}  est {est:>6}  actual {p['reported']:>6.2f}  "
                          f"{verdict} {p['surprise']:+.1f}%")

    print(f"\n    RECENT NEWS")
    if news.error:
        print(f"      unavailable ({news.error})")
    elif not news.headlines:
        print("      none returned")
    else:
        if news.risk_flags:
            print(f"      FLAGGED TERMS: {', '.join(news.risk_flags)}")
            print("      Headlines containing these are worth reading before assuming the dip")
            print("      reverts - company-specific bad news often does not.")
        for h in news.headlines[:5]:
            age = f"{h['age_hours']:.0f}h ago" if h["age_hours"] is not None else "?"
            mark = " [!]" if h["risk_terms"] else ""
            print(f"      - ({age:>8}, {h['source']}){mark} {h['title'][:88]}")
        print("\n      Headlines are shown, not scored. Automated sentiment on financial news")
        print("      is unreliable enough to mislead, so the read is yours.")
