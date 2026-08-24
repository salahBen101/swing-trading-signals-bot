"""Hypothesis screening harness.

Fast enough to sweep a parameter neighbourhood over nine years, and pessimistic in exactly
the same places as the production engine, so that a result here means what it will mean
there.

**Why a separate harness at all.** The production `ExecutionEngine` runs every bar through
the full risk/guard/journal path, which is right for a deployment candidate and far too
slow for a screen that must run hundreds of parameter variants. This harness shares the
assumptions that matter — next-bar fill, stop-before-target, gap fills at the open,
identical cost model — and drops only the machinery that cannot change a fill price. A
candidate that survives here is re-run through the real engine before anything is believed.

**Gross before net.** `screen()` reports gross and net separately because that distinction
decided the previous body of work: eight of nine 5-minute families had no gross edge at
all, which is a different and worse finding than being killed by fees. A rule with no gross
signal is rejected without costing it, and without tinkering.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# MNQ economics. One place, because everything downstream is denominated in them.
POINT_VALUE = 2.0
TICK_SIZE = 0.25
COMMISSION_ROUND_TRIP = 1.24
SLIPPAGE_TICKS_PER_SIDE = 1.0

#: Round-trip cost in index points at the default slippage assumption: 1.12 points.
ROUND_TRIP_POINTS = (
    COMMISSION_ROUND_TRIP / POINT_VALUE + 2 * SLIPPAGE_TICKS_PER_SIDE * TICK_SIZE
)


@dataclass(frozen=True, slots=True)
class ExitRule:
    """How a trade is closed. Deliberately small: three levers, no discretion.

    Distances are in ATR multiples so a rule means the same thing in a quiet August and on
    an FOMC afternoon. `max_bars` caps the hold; a session-end flatten always overrides.
    """

    stop_atr: float = 1.0
    target_atr: float = 2.0
    max_bars: int = 24
    # Bars before the RTH close after which no new entry is taken, so a trade has room to
    # resolve rather than being flattened at random.
    no_entry_within_bars: int = 3


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A pre-registered economic claim and its deterministic implementation."""

    id: str
    family: str
    rationale: str
    rules: str
    signal: Callable[[pd.DataFrame], np.ndarray]
    exit_rule: ExitRule = field(default_factory=ExitRule)
    params: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trade:
    entry_index: int
    exit_index: int
    direction: int
    entry_price: float
    exit_price: float
    gross_points: float
    bars_held: int
    exit_kind: str
    session: pd.Timestamp
    atr: float


@dataclass(frozen=True, slots=True)
class ScreenResult:
    hypothesis_id: str
    trades: int
    gross_points_per_trade: float
    gross_ticks_per_trade: float
    net_points_per_trade: float
    gross_pnl_usd: float
    commission_usd: float
    slippage_usd: float
    net_pnl_usd: float
    t_stat: float | None
    win_rate: float
    profit_factor: float
    avg_r: float
    median_hold_minutes: float
    max_drawdown_usd: float
    long_trades: int
    short_trades: int
    long_net_points: float
    short_net_points: float
    cost_share_of_gross: float | None
    by_year: dict = field(default_factory=dict)
    by_exit_kind: dict = field(default_factory=dict)
    trade_list: list = field(default_factory=list)

    @property
    def has_gross_edge(self) -> bool:
        """Does the entry predict anything before costs are charged?"""
        return self.gross_points_per_trade > 0

    @property
    def clears_cost_hurdle(self) -> bool:
        """Gross at least 4x the round trip, per RESEARCH_PLAN section 7."""
        return self.gross_points_per_trade >= 4 * ROUND_TRIP_POINTS

    def headline(self) -> str:
        t = "n/a" if self.t_stat is None else f"{self.t_stat:+.2f}"
        return (
            f"{self.hypothesis_id:<22} n={self.trades:>5}  "
            f"gross={self.gross_points_per_trade:+7.3f}pt "
            f"({self.gross_ticks_per_trade:+6.2f}tk)  "
            f"net={self.net_points_per_trade:+7.3f}pt  "
            f"t={t:>7}  win={self.win_rate:5.1%}  "
            f"hold={self.median_hold_minutes:>5.0f}m"
        )


def simulate(
    bars: pd.DataFrame,
    direction: np.ndarray,
    atr: np.ndarray,
    exit_rule: ExitRule,
) -> list[Trade]:
    """Walk each signal forward to its exit.

    Ordering, matching the production engine exactly:

    * a signal on bar *i* fills at bar *i+1*'s **open** — never on the bar that produced it
    * a bar that gaps through the stop fills at that bar's open, not at the stop level
    * when one bar's range contains both stop and target, the **stop** is taken; the
      intra-bar path is not recoverable from OHLCV, so the pessimistic branch is assumed
    * a position never crosses a session boundary
    """
    opens = bars["open"].to_numpy(dtype="float64")
    highs = bars["high"].to_numpy(dtype="float64")
    lows = bars["low"].to_numpy(dtype="float64")
    closes = bars["close"].to_numpy(dtype="float64")
    sessions = bars["session"].to_numpy()

    n = len(bars)
    trades: list[Trade] = []
    blocked_until = -1

    signal_indices = np.flatnonzero(direction != 0)
    for i in signal_indices:
        if i <= blocked_until or i + 1 >= n:
            continue
        if sessions[i + 1] != sessions[i]:
            continue  # the fill bar belongs to the next session
        bar_atr = atr[i]
        if not np.isfinite(bar_atr) or bar_atr <= 0:
            continue

        side = int(direction[i])
        entry_index = i + 1
        entry = opens[entry_index]
        stop_distance = exit_rule.stop_atr * bar_atr
        target_distance = exit_rule.target_atr * bar_atr
        stop = entry - side * stop_distance
        target = entry + side * target_distance

        # The window closes at whichever comes first: the hold cap or the session end.
        session = sessions[entry_index]
        last = entry_index
        limit = min(n - 1, entry_index + exit_rule.max_bars)
        while last < limit and sessions[last + 1] == session:
            last += 1

        exit_index, exit_price, kind = _walk(
            entry_index, last, side, stop, target, opens, highs, lows, closes
        )
        gross = (exit_price - entry) * side

        trades.append(
            Trade(
                entry_index=entry_index,
                exit_index=exit_index,
                direction=side,
                entry_price=entry,
                exit_price=exit_price,
                gross_points=gross,
                bars_held=exit_index - entry_index,
                exit_kind=kind,
                session=session,
                atr=bar_atr,
            )
        )
        # One position at a time, matching the deployment constraint.
        blocked_until = exit_index

    return trades


def _walk(
    entry_index: int,
    last: int,
    side: int,
    stop: float,
    target: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> tuple[int, float, str]:
    for j in range(entry_index, last + 1):
        if side > 0:
            if lows[j] <= stop:
                # A gap below the stop fills at the open, which is worse than the level.
                return j, min(stop, opens[j]), "stop"
            if highs[j] >= target:
                return j, target, "target"
        else:
            if highs[j] >= stop:
                return j, max(stop, opens[j]), "stop"
            if lows[j] <= target:
                return j, target, "target"
    return last, closes[last], "timeout"


def screen(
    hypothesis: Hypothesis,
    bars: pd.DataFrame,
    atr: np.ndarray,
    *,
    bar_minutes: float,
    slippage_ticks: float = SLIPPAGE_TICKS_PER_SIDE,
    keep_trades: bool = False,
) -> ScreenResult:
    """Run one hypothesis and compute the full metric set."""
    direction = hypothesis.signal(bars)
    direction = _mask_session_tail(direction, bars, hypothesis.exit_rule)
    trades = simulate(bars, direction, atr, hypothesis.exit_rule)
    return summarize(
        hypothesis.id, trades, bars,
        bar_minutes=bar_minutes, slippage_ticks=slippage_ticks, keep_trades=keep_trades,
    )


def _mask_session_tail(
    direction: np.ndarray, bars: pd.DataFrame, exit_rule: ExitRule
) -> np.ndarray:
    """Suppress entries too close to the close to resolve.

    Without this a rule's statistics are dominated by trades that were flattened at the
    bell rather than by the behaviour the hypothesis is about.
    """
    out = direction.copy()
    sessions = bars["session"].to_numpy()
    # Index of each bar counted backwards from its session's last bar.
    order = np.arange(len(bars))
    frame = pd.DataFrame({"session": sessions, "order": order})
    from_end = frame.groupby("session")["order"].transform("max").to_numpy() - order
    out[from_end < exit_rule.no_entry_within_bars] = 0
    return out


def summarize(
    hypothesis_id: str,
    trades: list[Trade],
    bars: pd.DataFrame,
    *,
    bar_minutes: float,
    slippage_ticks: float = SLIPPAGE_TICKS_PER_SIDE,
    keep_trades: bool = False,
) -> ScreenResult:
    if not trades:
        return ScreenResult(
            hypothesis_id, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0,
            0.0, 0.0, 0, 0, 0.0, 0.0, None,
        )

    gross = np.array([t.gross_points for t in trades])
    slippage_points = 2 * slippage_ticks * TICK_SIZE
    net = gross - slippage_points - COMMISSION_ROUND_TRIP / POINT_VALUE

    gross_usd = gross * POINT_VALUE
    net_usd = net * POINT_VALUE
    commission_usd = COMMISSION_ROUND_TRIP * len(trades)
    slippage_usd = slippage_points * POINT_VALUE * len(trades)

    mean = float(net.mean())
    t_stat = None
    if len(net) >= 2:
        sd = float(net.std(ddof=1))
        if sd > 0:
            t_stat = mean / (sd / math.sqrt(len(net)))

    wins = net[net > 0]
    losses = net[net < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0
        else float("inf")
    )

    # R multiple against the planned risk, which is what the risk engine actually sized on.
    planned_risk = np.array([t.atr for t in trades])
    r_multiples = np.divide(net, planned_risk, out=np.zeros_like(net),
                            where=planned_risk > 0)

    equity = np.concatenate([[0.0], np.cumsum(net_usd)])
    peak = np.maximum.accumulate(equity)
    max_drawdown = float((peak - equity).max())

    longs = [t for t in trades if t.direction > 0]
    shorts = [t for t in trades if t.direction < 0]
    long_net = float(np.mean([t.gross_points for t in longs])) if longs else 0.0
    short_net = float(np.mean([t.gross_points for t in shorts])) if shorts else 0.0

    years: dict[int, dict] = {}
    for trade, value in zip(trades, net_usd, strict=True):
        year = pd.Timestamp(trade.session).year
        bucket = years.setdefault(year, {"trades": 0, "net_usd": 0.0})
        bucket["trades"] += 1
        bucket["net_usd"] += float(value)

    kinds: dict[str, int] = {}
    for trade in trades:
        kinds[trade.exit_kind] = kinds.get(trade.exit_kind, 0) + 1

    gross_total = float(gross_usd.sum())
    return ScreenResult(
        hypothesis_id=hypothesis_id,
        trades=len(trades),
        gross_points_per_trade=float(gross.mean()),
        gross_ticks_per_trade=float(gross.mean() / TICK_SIZE),
        net_points_per_trade=mean,
        gross_pnl_usd=gross_total,
        commission_usd=commission_usd,
        slippage_usd=slippage_usd,
        net_pnl_usd=float(net_usd.sum()),
        t_stat=t_stat,
        win_rate=float((net > 0).mean()),
        profit_factor=profit_factor,
        avg_r=float(r_multiples.mean()),
        median_hold_minutes=float(np.median([t.bars_held for t in trades]) * bar_minutes),
        max_drawdown_usd=max_drawdown,
        long_trades=len(longs),
        short_trades=len(shorts),
        long_net_points=long_net,
        short_net_points=short_net,
        cost_share_of_gross=(
            (commission_usd + slippage_usd) / abs(gross_total) if gross_total else None
        ),
        by_year=years,
        by_exit_kind=kinds,
        trade_list=trades if keep_trades else [],
    )


def selection_bar(configurations: int) -> float:
    """The |t| a result must clear given how many configurations were evaluated."""
    from ..analytics.metrics import selection_bar as _bar

    return _bar(configurations)
