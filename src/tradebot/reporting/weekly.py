"""Pure weekly research reporting with survival-first hypothesis controls.

The report consumes already-journalled, terminal trade and rejection records.  It has no
broker, risk-engine, strategy, configuration, or deployment dependency and therefore
cannot apply any finding.  Candidate improvements are deliberately called *research
hypotheses*, are selected using the explicit :class:`HypothesisPolicy`, and are capped at
two.  They still require the repository's normal DEV -> validation -> untouched holdout
-> paper -> comparison -> human-approval path.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Iterable
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
_DIRECTIONS = frozenset(("LONG", "SHORT"))


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _count(value: int, name: str, *, minimum: int = 0) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        integer = minimum - 1
    if isinstance(value, bool) or integer != value or integer < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return integer


def _label(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class WeeklyTradeRecord:
    """One completed trade whose signal passed all gates."""

    trade_id: str
    entry_time: datetime
    instrument: str
    strategy: str
    setup: str
    direction: str
    pnl_usd: float
    r_multiple: float
    planned_risk_usd: float
    initial_stop_points: float
    slippage_usd: float
    fees_usd: float

    def __post_init__(self) -> None:
        for name in ("trade_id", "instrument", "strategy", "setup"):
            object.__setattr__(self, name, _label(getattr(self, name), f"weekly trade {name}"))
        _aware(self.entry_time, "weekly trade entry time")
        direction = _label(self.direction, "weekly trade direction").upper()
        if direction not in _DIRECTIONS:
            raise ValueError("weekly trade direction must be LONG or SHORT")
        object.__setattr__(self, "direction", direction)
        for name in (
            "pnl_usd",
            "r_multiple",
            "planned_risk_usd",
            "initial_stop_points",
            "slippage_usd",
            "fees_usd",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), f"weekly trade {name}"),
            )
        if self.planned_risk_usd <= 0:
            raise ValueError("weekly trade planned risk must be positive")
        if self.initial_stop_points <= 0:
            raise ValueError("weekly trade initial stop distance must be positive")
        if self.slippage_usd < 0 or self.fees_usd < 0:
            raise ValueError("weekly trade slippage and fees must be non-negative")


@dataclass(frozen=True, slots=True)
class RejectedSignalRecord:
    """A deterministic setup rejected before becoming a completed trade."""

    signal_id: str
    timestamp: datetime
    instrument: str
    strategy: str
    setup: str
    direction: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("signal_id", "instrument", "strategy", "setup", "reason"):
            object.__setattr__(
                self,
                name,
                _label(getattr(self, name), f"rejected signal {name}"),
            )
        _aware(self.timestamp, "rejected signal timestamp")
        direction = _label(self.direction, "rejected signal direction").upper()
        if direction not in _DIRECTIONS:
            raise ValueError("rejected signal direction must be LONG or SHORT")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class PropSurvivalBaseline:
    """Provided Monte Carlo/fresh-account results used as survival context.

    Runs may be censored at the configured horizon.  Hard failures and internal locks are
    treated as disjoint terminal counts.  Evaluation passes are validated independently
    because a journey can pass evaluation and later stop in its funded stage.
    """

    simulations: int
    evaluation_passes: int
    hard_failures: int
    internal_locks: int
    first_payouts: int
    median_days_to_pass: float | None
    average_drawdown_usd: float
    tail_drawdown_usd: float
    expected_lifetime_sessions: float
    expected_profit_per_account_usd: float
    consecutive_losses_survived: int

    def __post_init__(self) -> None:
        simulations = _count(self.simulations, "survival simulations", minimum=1)
        object.__setattr__(self, "simulations", simulations)
        for name in (
            "evaluation_passes",
            "hard_failures",
            "internal_locks",
            "first_payouts",
            "consecutive_losses_survived",
        ):
            object.__setattr__(
                self,
                name,
                _count(getattr(self, name), f"survival {name}"),
            )
        for name in ("evaluation_passes", "hard_failures", "internal_locks", "first_payouts"):
            if getattr(self, name) > simulations:
                raise ValueError(f"survival {name} cannot exceed simulations")
        if self.hard_failures + self.internal_locks > simulations:
            raise ValueError("survival account-stop counts cannot exceed simulations")
        if self.evaluation_passes:
            if self.median_days_to_pass is None:
                raise ValueError("median days to pass is required when simulations pass")
            median_days = _finite(
                self.median_days_to_pass, "survival median days to pass"
            )
            if median_days <= 0:
                raise ValueError("survival median days to pass must be positive")
            object.__setattr__(self, "median_days_to_pass", median_days)
        elif self.median_days_to_pass is not None:
            raise ValueError("median days to pass must be null when no simulations pass")
        for name in (
            "average_drawdown_usd",
            "tail_drawdown_usd",
            "expected_lifetime_sessions",
            "expected_profit_per_account_usd",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), f"survival {name}"),
            )
        if self.average_drawdown_usd < 0 or self.tail_drawdown_usd < 0:
            raise ValueError("survival drawdowns must be non-negative")
        if self.tail_drawdown_usd < self.average_drawdown_usd:
            raise ValueError("survival tail drawdown cannot be below average drawdown")
        if self.expected_lifetime_sessions < 0:
            raise ValueError("survival expected lifetime must be non-negative")

    @property
    def pass_rate(self) -> float:
        return self.evaluation_passes / self.simulations

    @property
    def hard_failure_rate(self) -> float:
        return self.hard_failures / self.simulations

    @property
    def internal_lock_rate(self) -> float:
        return self.internal_locks / self.simulations

    @property
    def first_payout_probability(self) -> float:
        return self.first_payouts / self.simulations

    @property
    def account_stop_rate(self) -> float:
        """Hard failures plus conservative internal locks."""
        return (self.hard_failures + self.internal_locks) / self.simulations

    def to_dict(self) -> dict:
        return {
            "simulations": self.simulations,
            "counts": {
                "evaluation_passes": self.evaluation_passes,
                "hard_failures": self.hard_failures,
                "internal_locks": self.internal_locks,
                "first_payouts": self.first_payouts,
            },
            "rates": {
                "pass_rate": self.pass_rate,
                "hard_failure_rate": self.hard_failure_rate,
                "internal_lock_rate": self.internal_lock_rate,
                "account_stop_rate": self.account_stop_rate,
                "first_payout_probability": self.first_payout_probability,
            },
            "median_days_to_pass": self.median_days_to_pass,
            "average_drawdown_usd": self.average_drawdown_usd,
            "tail_drawdown_usd": self.tail_drawdown_usd,
            "expected_lifetime_sessions": self.expected_lifetime_sessions,
            "expected_profit_per_account_usd": self.expected_profit_per_account_usd,
            "consecutive_losses_survived": self.consecutive_losses_survived,
        }


@dataclass(frozen=True, slots=True)
class HypothesisPolicy:
    """Predeclared evidence thresholds and deterministic candidate priority."""

    minimum_trades: int = 20
    minimum_slice_trades: int = 5
    weak_slice_average_r_at_most: float = -0.25
    weak_slice_loss_rate_at_least: float = 0.60
    account_stop_rate_at_least: float = 0.20
    average_cost_to_risk_at_least: float = 0.05
    losing_stop_ratio_at_least: float = 1.50
    minimum_outcome_group_trades: int = 5
    minimum_rejections: int = 10
    dominant_rejection_share_at_least: float = 0.60
    max_hypotheses: int = 2

    def __post_init__(self) -> None:
        for name in (
            "minimum_trades",
            "minimum_slice_trades",
            "minimum_outcome_group_trades",
            "minimum_rejections",
        ):
            object.__setattr__(
                self,
                name,
                _count(getattr(self, name), f"hypothesis {name}", minimum=1),
            )
        maximum = _count(self.max_hypotheses, "hypothesis max hypotheses", minimum=1)
        if maximum not in (1, 2):
            raise ValueError("hypothesis max hypotheses must be one or two")
        object.__setattr__(self, "max_hypotheses", maximum)
        for name in (
            "weak_slice_average_r_at_most",
            "weak_slice_loss_rate_at_least",
            "account_stop_rate_at_least",
            "average_cost_to_risk_at_least",
            "losing_stop_ratio_at_least",
            "dominant_rejection_share_at_least",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), f"hypothesis {name}"),
            )
        if self.weak_slice_average_r_at_most >= 0:
            raise ValueError("weak-slice average-R threshold must be negative")
        for name in (
            "weak_slice_loss_rate_at_least",
            "account_stop_rate_at_least",
            "dominant_rejection_share_at_least",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"hypothesis {name} must be between zero and one")
        if self.average_cost_to_risk_at_least <= 0:
            raise ValueError("cost-to-risk threshold must be positive")
        if self.losing_stop_ratio_at_least <= 1:
            raise ValueError("losing-stop ratio threshold must exceed one")

    def to_dict(self) -> dict:
        return {
            "minimum_trades": self.minimum_trades,
            "minimum_slice_trades": self.minimum_slice_trades,
            "weak_slice_average_r_at_most": self.weak_slice_average_r_at_most,
            "weak_slice_loss_rate_at_least": self.weak_slice_loss_rate_at_least,
            "account_stop_rate_at_least": self.account_stop_rate_at_least,
            "average_cost_to_risk_at_least": self.average_cost_to_risk_at_least,
            "losing_stop_ratio_at_least": self.losing_stop_ratio_at_least,
            "minimum_outcome_group_trades": self.minimum_outcome_group_trades,
            "minimum_rejections": self.minimum_rejections,
            "dominant_rejection_share_at_least": self.dominant_rejection_share_at_least,
            "max_hypotheses": self.max_hypotheses,
            "candidate_priority": [
                "WEAK_SLICE_SURVIVAL_EXPERIMENT",
                "EXECUTION_COST_STRESS_EXPERIMENT",
                "STOP_DISTANCE_EXPERIMENT",
                "REJECTION_CAUSE_AUDIT",
            ],
        }


@dataclass(frozen=True, slots=True)
class WeeklyReportInput:
    period_start: date
    period_end: date
    signals_generated: int
    survival_baseline: PropSurvivalBaseline
    accepted_trades: tuple[WeeklyTradeRecord, ...] = ()
    rejected_signals: tuple[RejectedSignalRecord, ...] = ()
    hypothesis_policy: HypothesisPolicy = field(default_factory=HypothesisPolicy)

    def __post_init__(self) -> None:
        if self.period_end < self.period_start:
            raise ValueError("weekly report period end cannot precede start")
        if (self.period_end - self.period_start).days > 6:
            raise ValueError("weekly report period cannot exceed seven calendar days")
        object.__setattr__(
            self,
            "signals_generated",
            _count(self.signals_generated, "weekly generated signals"),
        )
        trades = tuple(self.accepted_trades)
        rejections = tuple(self.rejected_signals)
        if any(not isinstance(item, WeeklyTradeRecord) for item in trades):
            raise TypeError("accepted trades must contain WeeklyTradeRecord values")
        if any(not isinstance(item, RejectedSignalRecord) for item in rejections):
            raise TypeError("rejected signals must contain RejectedSignalRecord values")
        if self.signals_generated != len(trades) + len(rejections):
            raise ValueError(
                "generated signals must equal accepted trades plus rejected signals"
            )
        trade_ids = [trade.trade_id for trade in trades]
        if len(trade_ids) != len(set(trade_ids)):
            raise ValueError("weekly accepted trade ids must be unique")
        signal_ids = [signal.signal_id for signal in rejections]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("weekly rejected signal ids must be unique")
        for record_time, kind in (
            *((trade.entry_time, "accepted trade") for trade in trades),
            *((signal.timestamp, "rejected signal") for signal in rejections),
        ):
            eastern_date = record_time.astimezone(EASTERN).date()
            if not self.period_start <= eastern_date <= self.period_end:
                raise ValueError(f"weekly {kind} timestamp falls outside report period")
        if not isinstance(self.survival_baseline, PropSurvivalBaseline):
            raise TypeError("weekly report requires a PropSurvivalBaseline")
        if not isinstance(self.hypothesis_policy, HypothesisPolicy):
            raise TypeError("weekly report requires a HypothesisPolicy")
        object.__setattr__(self, "accepted_trades", trades)
        object.__setattr__(self, "rejected_signals", rejections)


def _profit_factor(records: Iterable[WeeklyTradeRecord]) -> float | None:
    records = tuple(records)
    gross_profit = math.fsum(max(0.0, item.pnl_usd) for item in records)
    gross_loss = -math.fsum(min(0.0, item.pnl_usd) for item in records)
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


@dataclass(frozen=True, slots=True)
class PerformanceSlice:
    label: str
    trades: int
    wins: int
    losses: int
    net_pnl_usd: float
    expectancy_usd: float
    average_r: float
    profit_factor: float | None

    @property
    def loss_rate(self) -> float:
        return self.losses / self.trades if self.trades else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "loss_rate": self.loss_rate,
            "net_pnl_usd": self.net_pnl_usd,
            "expectancy_usd": self.expectancy_usd,
            "average_r": self.average_r,
            "profit_factor": self.profit_factor,
        }


@dataclass(frozen=True, slots=True)
class CountSlice:
    label: str
    count: int
    share: float

    def to_dict(self) -> dict:
        return {"label": self.label, "count": self.count, "share": self.share}


@dataclass(frozen=True, slots=True)
class StopDistanceSummary:
    trades: int
    mean_points: float
    median_points: float
    minimum_points: float
    maximum_points: float
    winning_mean_points: float | None
    losing_mean_points: float | None
    winning_median_points: float | None
    losing_median_points: float | None

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "mean_points": self.mean_points,
            "median_points": self.median_points,
            "minimum_points": self.minimum_points,
            "maximum_points": self.maximum_points,
            "winning_mean_points": self.winning_mean_points,
            "losing_mean_points": self.losing_mean_points,
            "winning_median_points": self.winning_median_points,
            "losing_median_points": self.losing_median_points,
        }


@dataclass(frozen=True, slots=True)
class CostSummary:
    slippage_usd: float
    fees_usd: float
    total_cost_usd: float
    average_slippage_usd: float
    average_fees_usd: float
    average_cost_to_planned_risk: float

    def to_dict(self) -> dict:
        return {
            "slippage_usd": self.slippage_usd,
            "fees_usd": self.fees_usd,
            "total_cost_usd": self.total_cost_usd,
            "average_slippage_usd": self.average_slippage_usd,
            "average_fees_usd": self.average_fees_usd,
            "average_cost_to_planned_risk": self.average_cost_to_planned_risk,
        }


@dataclass(frozen=True, slots=True)
class LosingSequenceSummary:
    sequence_lengths: tuple[int, ...]
    sequence_count: int
    maximum_consecutive_losses: int
    average_sequence_length: float
    baseline_consecutive_losses_survived: int
    survival_margin_losses: int

    def to_dict(self) -> dict:
        return {
            "sequence_lengths": list(self.sequence_lengths),
            "sequence_count": self.sequence_count,
            "maximum_consecutive_losses": self.maximum_consecutive_losses,
            "average_sequence_length": self.average_sequence_length,
            "baseline_consecutive_losses_survived": self.baseline_consecutive_losses_survived,
            "survival_margin_losses": self.survival_margin_losses,
        }


@dataclass(frozen=True, slots=True)
class ResearchHypothesis:
    code: str
    title: str
    rationale: str
    evidence: tuple[str, ...]
    status: str = "RESEARCH_HYPOTHESIS_ONLY"
    production_change_permitted: bool = False
    required_next_step: str = (
        "Test in isolation on DEV, then validation and untouched holdout; compare in paper "
        "and require explicit human approval before production."
    )

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "status": self.status,
            "production_change_permitted": self.production_change_permitted,
            "required_next_step": self.required_next_step,
        }


@dataclass(frozen=True, slots=True)
class WeeklyResearchReport:
    period_start: date
    period_end: date
    signals_generated: int
    signals_accepted: int
    signals_rejected: int
    overall: PerformanceSlice
    by_setup: tuple[PerformanceSlice, ...]
    by_time_of_day: tuple[PerformanceSlice, ...]
    by_direction: tuple[PerformanceSlice, ...]
    rejections_by_reason: tuple[CountSlice, ...]
    rejections_by_setup: tuple[CountSlice, ...]
    rejections_by_time_of_day: tuple[CountSlice, ...]
    rejections_by_direction: tuple[CountSlice, ...]
    stop_distances: StopDistanceSummary
    costs: CostSummary
    losing_sequences: LosingSequenceSummary
    survival_baseline: PropSurvivalBaseline
    hypothesis_policy: HypothesisPolicy
    research_hypotheses: tuple[ResearchHypothesis, ...]

    def to_dict(self) -> dict:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "signals": {
                "generated": self.signals_generated,
                "accepted": self.signals_accepted,
                "rejected": self.signals_rejected,
            },
            "performance": {
                "overall": self.overall.to_dict(),
                "by_setup": [item.to_dict() for item in self.by_setup],
                "by_time_of_day": [item.to_dict() for item in self.by_time_of_day],
                "by_direction": [item.to_dict() for item in self.by_direction],
            },
            "rejections": {
                "by_reason": [item.to_dict() for item in self.rejections_by_reason],
                "by_setup": [item.to_dict() for item in self.rejections_by_setup],
                "by_time_of_day": [item.to_dict() for item in self.rejections_by_time_of_day],
                "by_direction": [item.to_dict() for item in self.rejections_by_direction],
            },
            "stop_distances": self.stop_distances.to_dict(),
            "costs": self.costs.to_dict(),
            "losing_sequences": self.losing_sequences.to_dict(),
            "prop_survival_baseline": self.survival_baseline.to_dict(),
            "hypothesis_policy": self.hypothesis_policy.to_dict(),
            "research_hypotheses": [item.to_dict() for item in self.research_hypotheses],
            "production_changes": [],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    def to_text(self) -> str:
        profit_factor = (
            "N/A (no losing trades)"
            if self.overall.profit_factor is None
            else f"{self.overall.profit_factor:.3f}"
        )
        rejection_text = _format_counts(self.rejections_by_reason)
        hypotheses = [
            f"{index}. [{item.code}] {item.title} — {item.rationale}"
            for index, item in enumerate(self.research_hypotheses, start=1)
        ]
        if not hypotheses:
            hypotheses = ["NONE — no predeclared evidence threshold was met."]
        return "\n".join(
            (
                f"WEEKLY RESEARCH REPORT  {self.period_start.isoformat()} to "
                f"{self.period_end.isoformat()}",
                "PRODUCTION CHANGES: NONE — findings are research hypotheses only.",
                "",
                f"Signals: {self.signals_generated} generated / "
                f"{self.signals_accepted} accepted / {self.signals_rejected} rejected",
                f"Expectancy: ${self.overall.expectancy_usd:+,.2f} per trade",
                f"Profit factor: {profit_factor}",
                f"Average R: {self.overall.average_r:+.3f}",
                f"Rejected reasons: {rejection_text}",
                "",
                f"Stops: median {self.stop_distances.median_points:.3f} points; "
                f"range {self.stop_distances.minimum_points:.3f}-"
                f"{self.stop_distances.maximum_points:.3f}",
                f"Slippage: ${self.costs.slippage_usd:,.2f}",
                f"Fees: ${self.costs.fees_usd:,.2f}",
                f"Maximum consecutive losses: "
                f"{self.losing_sequences.maximum_consecutive_losses}",
                "",
                f"Prop pass rate: {self.survival_baseline.pass_rate:.1%}",
                f"Prop hard-failure rate: {self.survival_baseline.hard_failure_rate:.1%}",
                f"Prop internal-lock rate: {self.survival_baseline.internal_lock_rate:.1%}",
                f"First-payout probability: "
                f"{self.survival_baseline.first_payout_probability:.1%}",
                "",
                "Research hypotheses (maximum two):",
                *hypotheses,
            )
        )


def _performance(label: str, records: Iterable[WeeklyTradeRecord]) -> PerformanceSlice:
    records = tuple(records)
    count = len(records)
    net = math.fsum(item.pnl_usd for item in records)
    return PerformanceSlice(
        label=label,
        trades=count,
        wins=sum(item.pnl_usd > 0 for item in records),
        losses=sum(item.pnl_usd < 0 for item in records),
        net_pnl_usd=net,
        expectancy_usd=net / count if count else 0.0,
        average_r=(math.fsum(item.r_multiple for item in records) / count if count else 0.0),
        profit_factor=_profit_factor(records),
    )


def _performance_groups(
    records: Iterable[WeeklyTradeRecord],
    key: Callable[[WeeklyTradeRecord], str],
) -> tuple[PerformanceSlice, ...]:
    groups: dict[str, list[WeeklyTradeRecord]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return tuple(_performance(label, groups[label]) for label in sorted(groups))


def _count_groups(
    records: Iterable[RejectedSignalRecord],
    key: Callable[[RejectedSignalRecord], str],
) -> tuple[CountSlice, ...]:
    records = tuple(records)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[key(record)] += 1
    total = len(records)
    return tuple(
        CountSlice(label=label, count=count, share=count / total if total else 0.0)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _eastern_hour(value: datetime) -> str:
    return value.astimezone(EASTERN).strftime("%H:00 ET")


def _stop_summary(records: tuple[WeeklyTradeRecord, ...]) -> StopDistanceSummary:
    values = [item.initial_stop_points for item in records]
    winners = [item.initial_stop_points for item in records if item.pnl_usd > 0]
    losers = [item.initial_stop_points for item in records if item.pnl_usd < 0]
    if not values:
        return StopDistanceSummary(0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)
    return StopDistanceSummary(
        trades=len(values),
        mean_points=statistics.fmean(values),
        median_points=statistics.median(values),
        minimum_points=min(values),
        maximum_points=max(values),
        winning_mean_points=statistics.fmean(winners) if winners else None,
        losing_mean_points=statistics.fmean(losers) if losers else None,
        winning_median_points=statistics.median(winners) if winners else None,
        losing_median_points=statistics.median(losers) if losers else None,
    )


def _cost_summary(records: tuple[WeeklyTradeRecord, ...]) -> CostSummary:
    count = len(records)
    slippage = math.fsum(item.slippage_usd for item in records)
    fees = math.fsum(item.fees_usd for item in records)
    ratios = [
        (item.slippage_usd + item.fees_usd) / item.planned_risk_usd for item in records
    ]
    return CostSummary(
        slippage_usd=slippage,
        fees_usd=fees,
        total_cost_usd=slippage + fees,
        average_slippage_usd=slippage / count if count else 0.0,
        average_fees_usd=fees / count if count else 0.0,
        average_cost_to_planned_risk=statistics.fmean(ratios) if ratios else 0.0,
    )


def _losing_sequences(
    records: tuple[WeeklyTradeRecord, ...], baseline: PropSurvivalBaseline
) -> LosingSequenceSummary:
    lengths: list[int] = []
    current = 0
    for item in sorted(records, key=lambda record: (record.entry_time, record.trade_id)):
        if item.pnl_usd < 0:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    maximum = max(lengths, default=0)
    return LosingSequenceSummary(
        sequence_lengths=tuple(lengths),
        sequence_count=len(lengths),
        maximum_consecutive_losses=maximum,
        average_sequence_length=statistics.fmean(lengths) if lengths else 0.0,
        baseline_consecutive_losses_survived=baseline.consecutive_losses_survived,
        survival_margin_losses=baseline.consecutive_losses_survived - maximum,
    )


def _format_counts(items: tuple[CountSlice, ...]) -> str:
    if not items:
        return "NONE"
    return ", ".join(f"{item.label}={item.count}" for item in items)


def _select_hypotheses(
    *,
    overall: PerformanceSlice,
    slices: tuple[tuple[str, PerformanceSlice], ...],
    rejections: tuple[CountSlice, ...],
    stops: StopDistanceSummary,
    costs: CostSummary,
    baseline: PropSurvivalBaseline,
    policy: HypothesisPolicy,
) -> tuple[ResearchHypothesis, ...]:
    """Apply the fixed policy priority; dollar profit never ranks candidates."""
    candidates: list[ResearchHypothesis] = []

    weak_slices = [
        (category, item)
        for category, item in slices
        if item.trades >= policy.minimum_slice_trades
        and item.average_r <= policy.weak_slice_average_r_at_most
        and item.loss_rate >= policy.weak_slice_loss_rate_at_least
    ]
    if (
        overall.trades >= policy.minimum_trades
        and baseline.account_stop_rate >= policy.account_stop_rate_at_least
        and weak_slices
    ):
        category, weakest = min(
            weak_slices,
            key=lambda pair: (pair[1].average_r, -pair[1].loss_rate, pair[0], pair[1].label),
        )
        candidates.append(
            ResearchHypothesis(
                code="WEAK_SLICE_SURVIVAL_EXPERIMENT",
                title=f"Test isolating the weak {category} slice {weakest.label}",
                rationale=(
                    "Determine whether an independently specified filter improves account "
                    "survival without degrading validation stability."
                ),
                evidence=(
                    f"slice_trades={weakest.trades}",
                    f"slice_average_r={weakest.average_r:.6f}",
                    f"slice_loss_rate={weakest.loss_rate:.6f}",
                    f"prop_account_stop_rate={baseline.account_stop_rate:.6f}",
                ),
            )
        )

    if (
        overall.trades >= policy.minimum_trades
        and costs.average_cost_to_planned_risk >= policy.average_cost_to_risk_at_least
    ):
        candidates.append(
            ResearchHypothesis(
                code="EXECUTION_COST_STRESS_EXPERIMENT",
                title="Stress-test execution cost sensitivity",
                rationale=(
                    "Test whether the measured expectancy and prop survival remain stable "
                    "under worse fills before considering any execution change."
                ),
                evidence=(
                    f"trades={overall.trades}",
                    f"average_cost_to_planned_risk="
                    f"{costs.average_cost_to_planned_risk:.6f}",
                ),
            )
        )

    winner_count = overall.wins
    loser_count = overall.losses
    if (
        overall.trades >= policy.minimum_trades
        and winner_count >= policy.minimum_outcome_group_trades
        and loser_count >= policy.minimum_outcome_group_trades
        and stops.winning_median_points is not None
        and stops.losing_median_points is not None
        and stops.losing_median_points / stops.winning_median_points
        >= policy.losing_stop_ratio_at_least
    ):
        ratio = stops.losing_median_points / stops.winning_median_points
        candidates.append(
            ResearchHypothesis(
                code="STOP_DISTANCE_EXPERIMENT",
                title="Test stop-distance normalization as an isolated variant",
                rationale=(
                    "Check whether the losing-trade stop asymmetry is stable out of sample "
                    "and whether normalization improves survival rather than only net profit."
                ),
                evidence=(
                    f"winning_trades={winner_count}",
                    f"losing_trades={loser_count}",
                    f"losing_to_winning_median_stop_ratio={ratio:.6f}",
                ),
            )
        )

    if len(rejections) >= 1:
        dominant = rejections[0]
        rejected_count = sum(item.count for item in rejections)
        if (
            rejected_count >= policy.minimum_rejections
            and dominant.share >= policy.dominant_rejection_share_at_least
        ):
            candidates.append(
                ResearchHypothesis(
                    code="REJECTION_CAUSE_AUDIT",
                    title=f"Audit the dominant rejection cause {dominant.label}",
                    rationale=(
                        "Replay and classify this concentration to distinguish expected safety "
                        "behavior from a data or configuration defect; never weaken a risk gate."
                    ),
                    evidence=(
                        f"rejections={rejected_count}",
                        f"dominant_reason_count={dominant.count}",
                        f"dominant_reason_share={dominant.share:.6f}",
                    ),
                )
            )

    return tuple(candidates[: policy.max_hypotheses])


def generate_weekly_report(source: WeeklyReportInput) -> WeeklyResearchReport:
    """Build one deterministic weekly research report from explicit inputs."""
    trades = tuple(sorted(source.accepted_trades, key=lambda item: (item.entry_time, item.trade_id)))
    rejections = tuple(
        sorted(source.rejected_signals, key=lambda item: (item.timestamp, item.signal_id))
    )
    overall = _performance("ALL", trades)
    by_setup = _performance_groups(trades, lambda item: item.setup)
    by_hour = _performance_groups(trades, lambda item: _eastern_hour(item.entry_time))
    by_direction = _performance_groups(trades, lambda item: item.direction)
    rejection_reasons = _count_groups(rejections, lambda item: item.reason)
    rejection_setups = _count_groups(rejections, lambda item: item.setup)
    rejection_hours = _count_groups(rejections, lambda item: _eastern_hour(item.timestamp))
    rejection_directions = _count_groups(rejections, lambda item: item.direction)
    stops = _stop_summary(trades)
    costs = _cost_summary(trades)
    sequences = _losing_sequences(trades, source.survival_baseline)
    candidate_slices = tuple(
        (category, item)
        for category, group in (
            ("setup", by_setup),
            ("time-of-day", by_hour),
            ("direction", by_direction),
        )
        for item in group
    )
    hypotheses = _select_hypotheses(
        overall=overall,
        slices=candidate_slices,
        rejections=rejection_reasons,
        stops=stops,
        costs=costs,
        baseline=source.survival_baseline,
        policy=source.hypothesis_policy,
    )
    return WeeklyResearchReport(
        period_start=source.period_start,
        period_end=source.period_end,
        signals_generated=source.signals_generated,
        signals_accepted=len(trades),
        signals_rejected=len(rejections),
        overall=overall,
        by_setup=by_setup,
        by_time_of_day=by_hour,
        by_direction=by_direction,
        rejections_by_reason=rejection_reasons,
        rejections_by_setup=rejection_setups,
        rejections_by_time_of_day=rejection_hours,
        rejections_by_direction=rejection_directions,
        stop_distances=stops,
        costs=costs,
        losing_sequences=sequences,
        survival_baseline=source.survival_baseline,
        hypothesis_policy=source.hypothesis_policy,
        research_hypotheses=hypotheses,
    )
