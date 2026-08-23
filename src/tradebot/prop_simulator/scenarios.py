"""Paired, auditable survival-scenario comparison without profit ranking."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import date
import math
from typing import Sequence

from tradebot.prop_firms import AccountRuleSet, PropFirmProfile

from .journey_monte_carlo import JourneyMonteCarloReport, run_journey_session_bootstrap
from .models import HistoricalPropSession, PropSimulationPolicy


SCENARIO_COMPARISON_LIMITATIONS = (
    "Scenario output is descriptive survival evidence, not an optimizer, ranking, or "
    "claim of positive expectancy.",
    "Added commission, fee, and slippage amounts are deterministic dollar adjustments per "
    "source signal; they do not invent a spread, order-book, partial-fill, or impact model.",
    "Policy and internal-buffer perturbations change risk acceptance only. Strategy, stop, "
    "target, or signal-parameter perturbations require a separately generated backtest trade "
    "set and are deliberately unsupported here.",
    "All scenarios are paired on identical sampled source-session indices, but bootstrap "
    "results remain conditional on the supplied historical blocks and configured horizon.",
)


@dataclass(frozen=True, slots=True)
class TradeOutcomeAdjustment:
    """Explicit adverse dollars added to every source signal outcome and planned risk."""

    additional_commission_usd_per_trade: float = 0.0
    additional_fees_usd_per_trade: float = 0.0
    additional_slippage_usd_per_trade: float = 0.0

    @property
    def total_adverse_adjustment_usd(self) -> float:
        return (
            self.additional_commission_usd_per_trade
            + self.additional_fees_usd_per_trade
            + self.additional_slippage_usd_per_trade
        )

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{field.name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TradeAdjustmentAudit:
    """One immutable source-to-scenario trade transformation record."""

    session_date: date
    trade_id: str
    original_net_pnl_usd: float
    adjusted_net_pnl_usd: float
    original_minimum_intratrade_pnl_usd: float
    adjusted_minimum_intratrade_pnl_usd: float
    original_planned_risk_usd: float
    adjusted_planned_risk_usd: float
    added_commission_usd: float
    added_fees_usd: float
    added_slippage_usd: float


@dataclass(frozen=True, slots=True)
class ScenarioParameterChange:
    """A named simulator/policy value changed relative to baseline."""

    parameter: str
    baseline_value: float | int
    scenario_value: float | int


@dataclass(frozen=True, slots=True)
class JourneyScenario:
    """One named, explicit outcome and/or safety-policy perturbation."""

    name: str
    description: str
    outcome_adjustment: TradeOutcomeAdjustment = TradeOutcomeAdjustment()
    policy: PropSimulationPolicy = PropSimulationPolicy()
    internal_safety_buffer_usd: float = 400.0

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.name.strip().casefold() == "baseline":
            raise ValueError("scenario name 'baseline' is reserved")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("scenario description must not be empty")
        if not isinstance(self.outcome_adjustment, TradeOutcomeAdjustment):
            raise ValueError("outcome_adjustment must be a TradeOutcomeAdjustment")
        if not isinstance(self.policy, PropSimulationPolicy):
            raise ValueError("scenario policy must be a PropSimulationPolicy")
        self.outcome_adjustment.validate()
        self.policy.validate()
        if (
            isinstance(self.internal_safety_buffer_usd, bool)
            or not isinstance(self.internal_safety_buffer_usd, (int, float))
            or not math.isfinite(float(self.internal_safety_buffer_usd))
            or self.internal_safety_buffer_usd < 0
        ):
            raise ValueError("internal_safety_buffer_usd must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class JourneyScenarioOutcome:
    """One scenario's inputs, transformation ledger, and survival metrics."""

    name: str
    description: str
    outcome_adjustment: TradeOutcomeAdjustment
    policy: PropSimulationPolicy
    internal_safety_buffer_usd: float
    parameter_changes: tuple[ScenarioParameterChange, ...]
    trade_adjustments: tuple[TradeAdjustmentAudit, ...]
    report: JourneyMonteCarloReport


@dataclass(frozen=True, slots=True)
class JourneyScenarioComparison:
    """Baseline and ordered paired scenarios; intentionally contains no ranking."""

    simulations: int
    seed: int
    horizon_sessions: int
    baseline: JourneyScenarioOutcome
    scenarios: tuple[JourneyScenarioOutcome, ...]
    limitations: tuple[str, ...]


def apply_trade_outcome_adjustment(
    sessions: Sequence[HistoricalPropSession],
    adjustment: TradeOutcomeAdjustment,
) -> tuple[tuple[HistoricalPropSession, ...], tuple[TradeAdjustmentAudit, ...]]:
    """Return transformed immutable blocks and a complete source-trade audit ledger."""

    if not isinstance(adjustment, TradeOutcomeAdjustment):
        raise ValueError("adjustment must be a TradeOutcomeAdjustment")
    adjustment.validate()
    source = tuple(sessions)
    if not source:
        raise ValueError("at least one historical session is required")
    for session in source:
        if not isinstance(session, HistoricalPropSession):
            raise ValueError("every input item must be a HistoricalPropSession")
        session.validate()
    total = adjustment.total_adverse_adjustment_usd
    if total == 0:
        return source, ()

    transformed_sessions: list[HistoricalPropSession] = []
    audit: list[TradeAdjustmentAudit] = []
    for session in source:
        adjusted_trades = []
        for ordinal, trade in enumerate(session.trades, start=1):
            adjusted = replace(
                trade,
                net_pnl_usd=trade.net_pnl_usd - total,
                minimum_intratrade_pnl_usd=(
                    trade.minimum_intratrade_pnl_usd - total
                ),
                planned_risk_usd=trade.planned_risk_usd + total,
                commission_usd=(
                    trade.commission_usd
                    + adjustment.additional_commission_usd_per_trade
                ),
                fees_usd=trade.fees_usd + adjustment.additional_fees_usd_per_trade,
                slippage_usd=(
                    trade.slippage_usd + adjustment.additional_slippage_usd_per_trade
                ),
            )
            adjusted_trades.append(adjusted)
            audit.append(
                TradeAdjustmentAudit(
                    session_date=session.session_date,
                    trade_id=trade.trade_id or f"{session.session_date.isoformat()}:{ordinal}",
                    original_net_pnl_usd=trade.net_pnl_usd,
                    adjusted_net_pnl_usd=adjusted.net_pnl_usd,
                    original_minimum_intratrade_pnl_usd=(
                        trade.minimum_intratrade_pnl_usd
                    ),
                    adjusted_minimum_intratrade_pnl_usd=(
                        adjusted.minimum_intratrade_pnl_usd
                    ),
                    original_planned_risk_usd=trade.planned_risk_usd,
                    adjusted_planned_risk_usd=adjusted.planned_risk_usd,
                    added_commission_usd=(
                        adjustment.additional_commission_usd_per_trade
                    ),
                    added_fees_usd=adjustment.additional_fees_usd_per_trade,
                    added_slippage_usd=adjustment.additional_slippage_usd_per_trade,
                )
            )
        transformed_sessions.append(replace(session, trades=tuple(adjusted_trades)))
    return tuple(transformed_sessions), tuple(audit)


def _parameter_changes(
    baseline_policy: PropSimulationPolicy,
    baseline_buffer_usd: float,
    scenario: JourneyScenario,
) -> tuple[ScenarioParameterChange, ...]:
    changes: list[ScenarioParameterChange] = []
    for field in fields(PropSimulationPolicy):
        baseline_value = getattr(baseline_policy, field.name)
        scenario_value = getattr(scenario.policy, field.name)
        if scenario_value != baseline_value:
            changes.append(
                ScenarioParameterChange(
                    parameter=f"policy.{field.name}",
                    baseline_value=baseline_value,
                    scenario_value=scenario_value,
                )
            )
    if scenario.internal_safety_buffer_usd != baseline_buffer_usd:
        changes.append(
            ScenarioParameterChange(
                parameter="internal_safety_buffer_usd",
                baseline_value=baseline_buffer_usd,
                scenario_value=scenario.internal_safety_buffer_usd,
            )
        )
    return tuple(changes)


def compare_journey_scenarios(
    profile: PropFirmProfile,
    evaluation_rules: AccountRuleSet,
    funded_rules: AccountRuleSet,
    sessions: Sequence[HistoricalPropSession],
    *,
    scenarios: Sequence[JourneyScenario],
    simulations: int,
    seed: int,
    horizon_sessions: int | None = None,
    baseline_policy: PropSimulationPolicy = PropSimulationPolicy(),
    baseline_internal_safety_buffer_usd: float = 400.0,
) -> JourneyScenarioComparison:
    """Compare paired scenarios in caller order without selecting a winner."""

    if not isinstance(baseline_policy, PropSimulationPolicy):
        raise ValueError("baseline_policy must be a PropSimulationPolicy")
    baseline_policy.validate()
    scenario_list = tuple(scenarios)
    if not scenario_list:
        raise ValueError("at least one comparison scenario is required")
    names: set[str] = set()
    for scenario in scenario_list:
        if not isinstance(scenario, JourneyScenario):
            raise ValueError("every scenario must be a JourneyScenario")
        scenario.validate()
        normalized = scenario.name.strip().casefold()
        if normalized in names:
            raise ValueError(f"duplicate scenario name {scenario.name!r}")
        names.add(normalized)

    source = tuple(sessions)
    baseline_report = run_journey_session_bootstrap(
        profile,
        evaluation_rules,
        funded_rules,
        source,
        simulations=simulations,
        seed=seed,
        horizon_sessions=horizon_sessions,
        internal_safety_buffer_usd=baseline_internal_safety_buffer_usd,
        policy=baseline_policy,
    )
    baseline = JourneyScenarioOutcome(
        name="baseline",
        description="Unmodified source outcomes and baseline personal-risk policy.",
        outcome_adjustment=TradeOutcomeAdjustment(),
        policy=baseline_policy,
        internal_safety_buffer_usd=float(baseline_internal_safety_buffer_usd),
        parameter_changes=(),
        trade_adjustments=(),
        report=baseline_report,
    )

    outcomes: list[JourneyScenarioOutcome] = []
    for scenario in scenario_list:
        adjusted_sessions, audit = apply_trade_outcome_adjustment(
            source, scenario.outcome_adjustment
        )
        report = run_journey_session_bootstrap(
            profile,
            evaluation_rules,
            funded_rules,
            adjusted_sessions,
            simulations=simulations,
            seed=seed,
            horizon_sessions=horizon_sessions,
            internal_safety_buffer_usd=scenario.internal_safety_buffer_usd,
            policy=scenario.policy,
        )
        if report.sampled_source_indices != baseline_report.sampled_source_indices:
            raise RuntimeError("paired scenario session samples diverged from baseline")
        outcomes.append(
            JourneyScenarioOutcome(
                name=scenario.name,
                description=scenario.description,
                outcome_adjustment=scenario.outcome_adjustment,
                policy=scenario.policy,
                internal_safety_buffer_usd=float(scenario.internal_safety_buffer_usd),
                parameter_changes=_parameter_changes(
                    baseline_policy,
                    float(baseline_internal_safety_buffer_usd),
                    scenario,
                ),
                trade_adjustments=audit,
                report=report,
            )
        )

    return JourneyScenarioComparison(
        simulations=simulations,
        seed=seed,
        horizon_sessions=baseline_report.horizon_sessions,
        baseline=baseline,
        scenarios=tuple(outcomes),
        limitations=SCENARIO_COMPARISON_LIMITATIONS,
    )


__all__ = [
    "JourneyScenario",
    "JourneyScenarioComparison",
    "JourneyScenarioOutcome",
    "SCENARIO_COMPARISON_LIMITATIONS",
    "ScenarioParameterChange",
    "TradeAdjustmentAudit",
    "TradeOutcomeAdjustment",
    "apply_trade_outcome_adjustment",
    "compare_journey_scenarios",
]
