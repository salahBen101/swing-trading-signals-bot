"""Fresh-account prop simulation and whole-session Monte Carlo tools."""

from .engine import simulate_evaluation_to_funded, simulate_prop_phase
from .journey_monte_carlo import JourneyMonteCarloReport, run_journey_session_bootstrap
from .models import (
    HistoricalPropSession,
    HistoricalPropTrade,
    HoldDurationCompliance,
    JourneyStage,
    PropJourneyResult,
    PropSimulationResult,
    PropSimulationPolicy,
    SimulationEvent,
    SimulationEventKind,
    SimulationTermination,
)
from .monte_carlo import MonteCarloReport, run_session_bootstrap
from .scenarios import (
    JourneyScenario,
    JourneyScenarioComparison,
    JourneyScenarioOutcome,
    ScenarioParameterChange,
    TradeAdjustmentAudit,
    TradeOutcomeAdjustment,
    apply_trade_outcome_adjustment,
    compare_journey_scenarios,
)

__all__ = [
    "HistoricalPropSession",
    "HistoricalPropTrade",
    "HoldDurationCompliance",
    "JourneyStage",
    "JourneyMonteCarloReport",
    "JourneyScenario",
    "JourneyScenarioComparison",
    "JourneyScenarioOutcome",
    "MonteCarloReport",
    "PropJourneyResult",
    "PropSimulationResult",
    "PropSimulationPolicy",
    "SimulationEvent",
    "SimulationEventKind",
    "SimulationTermination",
    "ScenarioParameterChange",
    "TradeAdjustmentAudit",
    "TradeOutcomeAdjustment",
    "apply_trade_outcome_adjustment",
    "compare_journey_scenarios",
    "run_journey_session_bootstrap",
    "run_session_bootstrap",
    "simulate_evaluation_to_funded",
    "simulate_prop_phase",
]
