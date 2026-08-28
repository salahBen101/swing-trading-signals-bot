"""RSI(2) options paper-trial infrastructure.

Built on the existing tradebot broker/journal/risk foundations rather than beside them. Paper
execution only - there is no live path in this package and none may be added without a separate
engineering decision.
"""

from .calendar import MARKET_TZ, CalendarExpired, TradingCalendar
from .experiment import (
    ExperimentConfig,
    ExperimentState,
    ExperimentStatus,
    OptionsPolicy,
    Provenance,
    RiskLimits,
    TradingMode,
    default_config,
    require_paper_mode,
)
from .quotes import OptionQuote, validate_contract_terms, validate_quote
from .risk import PortfolioSnapshot, ProposedTrade, RiskDecision, RiskEngine
from .states import ExitState, RejectReason, Rejection, SignalState, assert_transition

__all__ = [
    "MARKET_TZ", "CalendarExpired", "TradingCalendar",
    "ExperimentConfig", "ExperimentState", "ExperimentStatus", "OptionsPolicy",
    "Provenance", "RiskLimits", "TradingMode", "default_config", "require_paper_mode",
    "OptionQuote", "validate_quote", "validate_contract_terms",
    "PortfolioSnapshot", "ProposedTrade", "RiskDecision", "RiskEngine",
    "ExitState", "RejectReason", "Rejection", "SignalState", "assert_transition",
]
