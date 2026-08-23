"""Trade statistics.

The full set PROJECT_SPEC §6 requires, computed from a list of `Trade` objects and nothing
else — this module never re-derives a trade, never re-runs a strategy, and never touches
prices. That separation is what lets the same function report on a backtest and on a live
journal and be trusted to mean the same thing.

Two things are deliberately reported that most summaries omit:

* **Cost as a share of gross.** Prior work in this repository killed a genuine +3.65-tick
  order-flow effect on a 3.2-tick fee. A strategy's gross edge and its net edge are
  different findings and the difference is the interesting part.
* **The search count.** A t-statistic quoted without how many configurations were tried is
  not evidence. `selection_bar` turns the count into the threshold it implies.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..core.models import Trade
from ..core.types import ExitReason, Side

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True, slots=True)
class Bucket:
    """Performance for one slice — an hour, a weekday, a side, an exit reason."""

    label: str
    trades: int
    net_pnl: float
    win_rate: float
    avg_r: float

    @property
    def avg_pnl(self) -> float:
        return self.net_pnl / self.trades if self.trades else 0.0


@dataclass(frozen=True, slots=True)
class Metrics:
    trades: int
    wins: int
    losses: int
    scratches: int
    win_rate: float
    gross_pnl: float
    commission: float
    net_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    expectancy: float
    profit_factor: float
    avg_r: float
    total_r: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float | None
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_bars_held: float
    cost_share_of_gross: float | None
    by_hour: tuple[Bucket, ...] = ()
    by_weekday: tuple[Bucket, ...] = ()
    by_side: tuple[Bucket, ...] = ()
    by_exit_reason: tuple[Bucket, ...] = ()
    equity_curve: tuple[float, ...] = ()
    equity_times: tuple[datetime, ...] = ()
    starting_equity: float = 0.0
    configurations_tried: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.starting_equity

    # t on mean net P&L per trade. None when the sample cannot support one (fewer than
    # two trades, or zero dispersion).
    t_statistic: float | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["by_hour"] = [asdict(b) for b in self.by_hour]
        out["by_weekday"] = [asdict(b) for b in self.by_weekday]
        out["by_side"] = [asdict(b) for b in self.by_side]
        out["by_exit_reason"] = [asdict(b) for b in self.by_exit_reason]
        out["equity_times"] = [t.isoformat() for t in self.equity_times]
        out["selection_bar"] = selection_bar(self.configurations_tried)
        out["clears_selection_bar"] = self.clears_selection_bar
        return out

    @property
    def clears_selection_bar(self) -> bool | None:
        if self.t_statistic is None:
            return None
        # A precisely estimated losing strategy is useful evidence, but it is not a
        # positive edge and must never be labelled as clearing the promotion bar.
        return self.t_statistic >= selection_bar(self.configurations_tried)


def selection_bar(configurations: int) -> float:
    """The positive t-stat a result must clear after configuration search.

    A Bonferroni-style two-sided correction at alpha = 0.05. It is deliberately crude —
    the point is not precision, it is that the bar *moves* when you search harder, so a
    number cannot be quoted without its search count attached.
    """
    configurations = max(1, int(configurations))
    if configurations == 1:
        return 1.96
    # Inverse-normal approximation for the two-sided alpha/n quantile.
    alpha = 0.05 / configurations
    return abs(_inverse_normal_cdf(alpha / 2.0))


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation. Accurate to ~1e-9, no scipy dependency."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _bucket(label: str, trades: list[Trade]) -> Bucket:
    if not trades:
        return Bucket(label, 0, 0.0, 0.0, 0.0)
    net = sum(t.net_pnl_usd for t in trades)
    wins = sum(1 for t in trades if t.net_pnl_usd > 0)
    return Bucket(
        label=label,
        trades=len(trades),
        net_pnl=net,
        win_rate=wins / len(trades),
        avg_r=sum(t.r_multiple for t in trades) / len(trades),
    )


def _max_streak(trades: list[Trade], winning: bool) -> int:
    best = current = 0
    for trade in trades:
        hit = trade.net_pnl_usd > 0 if winning else trade.net_pnl_usd < 0
        current = current + 1 if hit else 0
        best = max(best, current)
    return best


def _drawdown(curve: list[float]) -> tuple[float, float]:
    """Largest peak-to-trough fall, in dollars and as a fraction of the peak."""
    peak = curve[0] if curve else 0.0
    worst_abs = worst_pct = 0.0
    for value in curve:
        peak = max(peak, value)
        drop = peak - value
        worst_abs = max(worst_abs, drop)
        if peak > 0:
            worst_pct = max(worst_pct, drop / peak)
    return worst_abs, worst_pct


def _sharpe(r_multiples: list[float]) -> float | None:
    """Per-trade R-multiple Sharpe, unannualised.

    Deliberately not annualised: scaling by sqrt(trades per year) on an intraday strategy
    with a variable trade rate produces a number that looks authoritative and means very
    little. Returns None below 30 trades, where the estimate is dominated by noise.

    Dollar P&L silently gives larger-sized trades more statistical weight. R multiples
    put each result on its planned initial-risk scale, which is the stable unit for this
    variable-size system.
    """
    if len(r_multiples) < 30:
        return None
    mean = sum(r_multiples) / len(r_multiples)
    variance = sum((r - mean) ** 2 for r in r_multiples) / (len(r_multiples) - 1)
    sd = math.sqrt(variance)
    return mean / sd if sd > 0 else None


def compute_metrics(
    trades: list[Trade],
    *,
    starting_equity: float = 50_000.0,
    configurations_tried: int = 1,
    notes: tuple[str, ...] = (),
    marked_equity_curve: Iterable[float] | None = None,
) -> Metrics:
    trades = sorted(trades, key=lambda t: t.exit_time)

    if not trades:
        return Metrics(
            trades=0, wins=0, losses=0, scratches=0, win_rate=0.0, gross_pnl=0.0,
            commission=0.0, net_pnl=0.0, avg_winner=0.0, avg_loser=0.0,
            largest_winner=0.0, largest_loser=0.0, expectancy=0.0, profit_factor=0.0,
            avg_r=0.0, total_r=0.0, max_drawdown=0.0, max_drawdown_pct=0.0, sharpe=None,
            max_consecutive_wins=0, max_consecutive_losses=0, avg_bars_held=0.0,
            cost_share_of_gross=None, starting_equity=starting_equity,
            equity_curve=(starting_equity,), configurations_tried=configurations_tried,
            notes=notes,
        )

    winners = [t for t in trades if t.net_pnl_usd > 0]
    losers = [t for t in trades if t.net_pnl_usd < 0]
    scratches = [t for t in trades if t.net_pnl_usd == 0]

    gross = sum(t.gross_pnl_usd for t in trades)
    commission = sum(t.commission_usd for t in trades)
    net = sum(t.net_pnl_usd for t in trades)

    gross_profit = sum(t.net_pnl_usd for t in winners)
    gross_loss = abs(sum(t.net_pnl_usd for t in losers))

    equity = [starting_equity]
    for trade in trades:
        equity.append(equity[-1] + trade.net_pnl_usd)
    if marked_equity_curve is None:
        drawdown_curve = equity
    else:
        marked = [float(value) for value in marked_equity_curve]
        if any(not math.isfinite(value) for value in marked):
            raise ValueError("marked equity curve must contain only finite values")
        drawdown_curve = [starting_equity, *marked]
    dd_abs, dd_pct = _drawdown(drawdown_curve)

    pnls = [t.net_pnl_usd for t in trades]
    mean = net / len(trades)
    t_stat = None
    if len(trades) >= 2:
        variance = sum((p - mean) ** 2 for p in pnls) / (len(trades) - 1)
        sd = math.sqrt(variance)
        if sd > 0:
            t_stat = mean / (sd / math.sqrt(len(trades)))

    by_hour = tuple(
        _bucket(f"{hour:02d}:00", [t for t in trades if t.entry_time.hour == hour])
        for hour in sorted({t.entry_time.hour for t in trades})
    )
    by_weekday = tuple(
        _bucket(WEEKDAYS[day], [t for t in trades if t.entry_time.weekday() == day])
        for day in sorted({t.entry_time.weekday() for t in trades})
    )
    by_side = tuple(
        _bucket(side.name, [t for t in trades if t.side is side])
        for side in (Side.BUY, Side.SELL)
        if any(t.side is side for t in trades)
    )
    reasons: Counter[ExitReason] = Counter(t.exit_reason for t in trades)
    by_exit = tuple(
        _bucket(reason.value, [t for t in trades if t.exit_reason is reason])
        for reason, _ in reasons.most_common()
    )

    return Metrics(
        trades=len(trades),
        wins=len(winners),
        losses=len(losers),
        scratches=len(scratches),
        win_rate=len(winners) / len(trades),
        gross_pnl=gross,
        commission=commission,
        net_pnl=net,
        avg_winner=(gross_profit / len(winners)) if winners else 0.0,
        avg_loser=(-gross_loss / len(losers)) if losers else 0.0,
        largest_winner=max((t.net_pnl_usd for t in winners), default=0.0),
        largest_loser=min((t.net_pnl_usd for t in losers), default=0.0),
        expectancy=mean,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        avg_r=sum(t.r_multiple for t in trades) / len(trades),
        total_r=sum(t.r_multiple for t in trades),
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        sharpe=_sharpe([t.r_multiple for t in trades]),
        max_consecutive_wins=_max_streak(trades, winning=True),
        max_consecutive_losses=_max_streak(trades, winning=False),
        avg_bars_held=sum(t.bars_held for t in trades) / len(trades),
        cost_share_of_gross=(commission / abs(gross)) if gross != 0 else None,
        by_hour=by_hour,
        by_weekday=by_weekday,
        by_side=by_side,
        by_exit_reason=by_exit,
        equity_curve=tuple(equity),
        equity_times=tuple([trades[0].entry_time] + [t.exit_time for t in trades]),
        starting_equity=starting_equity,
        configurations_tried=configurations_tried,
        notes=notes,
        t_statistic=t_stat,
    )


def by_year(trades: list[Trade]) -> list[Bucket]:
    """Year-by-year breakdown.

    Its own function because it is the test that killed the one candidate strategy this
    project ever produced: a rule drawing 140% of its net profit from a single year is not
    an edge, and an aggregate number hides that completely.
    """
    years = sorted({t.entry_time.year for t in trades})
    return [
        _bucket(str(year), [t for t in trades if t.entry_time.year == year])
        for year in years
    ]
