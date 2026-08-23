"""Immutable inputs and outputs for prop-account historical simulation.

The simulator consumes already-closed strategy trades.  ``net_pnl_usd`` is the amount
that changes account balance and therefore already includes the explicitly recorded
commission, fee, and slippage fields.  Keeping those costs separately makes assumptions
auditable without ever adding them to balance a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math

from tradebot.prop_firms import AccountPhase
from tradebot.risk.prop import MarketDayStatus, PropAccountState


_EPSILON = 1e-9


class SimulationTermination(str, Enum):
    """Why a phase simulation stopped."""

    HORIZON_EXHAUSTED = "horizon_exhausted"
    EVALUATION_PASSED = "evaluation_passed"
    PAYOUT_ELIGIBLE = "payout_eligible"
    ACCOUNT_FAILED = "account_failed"
    INTERNAL_SAFETY_LOCK = "internal_safety_lock"


class JourneyStage(str, Enum):
    """Account stage active when an evaluation-to-funded journey stopped."""

    EVALUATION = "evaluation"
    FUNDED = "funded"


class SimulationEventKind(str, Enum):
    """Machine-readable event types retained in a simulation audit trace."""

    SESSION_STARTED = "session_started"
    TRADE_EXECUTED = "trade_executed"
    TRADE_REJECTED_DAILY_LOSS = "trade_rejected_daily_loss"
    TRADE_REJECTED_CONTRACT_LIMIT = "trade_rejected_contract_limit"
    TRADE_REJECTED_PERSONAL_RISK = "trade_rejected_personal_risk"
    TRADE_REJECTED_TRADE_QUOTA = "trade_rejected_trade_quota"
    TRADE_REJECTED_DAILY_STRATEGY_LOSS = "trade_rejected_daily_strategy_loss"
    TRADE_REJECTED_POST_LOSS_RISK_INCREASE = "trade_rejected_post_loss_risk_increase"
    TRADE_REJECTED_PRETRADE_SAFETY = "trade_rejected_pretrade_safety"
    TRADE_REJECTED_MARKET_DAY = "trade_rejected_market_day"
    HARD_ACCOUNT_BREACH = "hard_account_breach"
    INTERNAL_SAFETY_LOCK = "internal_safety_lock"
    SESSION_CLOSED = "session_closed"
    EVALUATION_PASSED = "evaluation_passed"
    PAYOUT_ELIGIBLE = "payout_eligible"


@dataclass(frozen=True, slots=True)
class HistoricalPropTrade:
    """One completed historical trade expressed in account-dollar outcomes.

    ``minimum_intratrade_pnl_usd`` is the worst marked P&L seen while the position was
    open, relative to equity immediately before entry.  It is non-positive and net of
    the entry/forced-liquidation assumptions used by the source backtest.  The simulator
    applies this value to real-time net liquidation *before* applying the closing P&L.
    """

    opened_at: datetime
    closed_at: datetime
    net_pnl_usd: float
    minimum_intratrade_pnl_usd: float
    planned_risk_usd: float
    commission_usd: float = 0.0
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
    minis: int = 0
    micros: int = 1
    trade_id: str = ""

    @property
    def hold_seconds(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds()

    @property
    def gross_pnl_before_explicit_costs_usd(self) -> float:
        """Consistency P&L before commissions/fees, but after execution slippage."""

        return self.net_pnl_usd + self.commission_usd + self.fees_usd

    def validate(self) -> None:
        for label, moment in (("opened_at", self.opened_at), ("closed_at", self.closed_at)):
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError(f"{label} must be timezone-aware")
        if self.closed_at < self.opened_at:
            raise ValueError("trade closed_at cannot precede opened_at")
        for label in (
            "net_pnl_usd",
            "minimum_intratrade_pnl_usd",
            "planned_risk_usd",
            "commission_usd",
            "fees_usd",
            "slippage_usd",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be a finite number")
        if self.minimum_intratrade_pnl_usd > _EPSILON:
            raise ValueError("minimum_intratrade_pnl_usd cannot be positive")
        if self.minimum_intratrade_pnl_usd > min(0.0, self.net_pnl_usd) + _EPSILON:
            raise ValueError(
                "minimum_intratrade_pnl_usd must be no greater than the closing net P&L"
            )
        if self.planned_risk_usd <= _EPSILON:
            raise ValueError("planned_risk_usd must be strictly positive")
        if self.commission_usd < 0 or self.fees_usd < 0 or self.slippage_usd < 0:
            raise ValueError("commission, fees, and slippage must be non-negative")
        for label in ("minis", "micros"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.minis + self.micros <= 0:
            raise ValueError("a trade must contain at least one mini or micro contract")
        if not isinstance(self.trade_id, str):
            raise ValueError("trade_id must be a string")


@dataclass(frozen=True, slots=True)
class HistoricalPropSession:
    """A complete Tradeify session block, including zero-trade elapsed sessions."""

    session_date: date
    market_day_status: MarketDayStatus = MarketDayStatus.REGULAR
    trades: tuple[HistoricalPropTrade, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.session_date, date) or isinstance(self.session_date, datetime):
            raise ValueError("session_date must be a date")
        if not isinstance(self.trades, tuple):
            raise ValueError("session trades must be supplied as an immutable tuple")
        if not isinstance(self.market_day_status, MarketDayStatus):
            raise ValueError("market_day_status must be a MarketDayStatus")
        for trade in self.trades:
            if not isinstance(trade, HistoricalPropTrade):
                raise ValueError("every session item must be a HistoricalPropTrade")
            trade.validate()


@dataclass(frozen=True, slots=True)
class PropSimulationPolicy:
    """Personal safety policy applied before any historical trade can affect equity.

    Values may be tightened for stress research but cannot exceed the project's pinned
    maximums.  The historical input is sequential and rejects overlapping positions, so
    only one open position is supported.
    """

    max_planned_risk_per_trade_usd: float = 200.0
    max_trades_per_session: int = 1
    max_daily_strategy_loss_usd: float = 200.0
    max_open_positions: int = 1

    def validate(self) -> None:
        for label, value, ceiling in (
            ("max_planned_risk_per_trade_usd", self.max_planned_risk_per_trade_usd, 200.0),
            ("max_daily_strategy_loss_usd", self.max_daily_strategy_loss_usd, 200.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                or value > ceiling + _EPSILON
            ):
                raise ValueError(f"{label} must be positive and no greater than {ceiling:.0f}")
        if (
            not isinstance(self.max_trades_per_session, int)
            or isinstance(self.max_trades_per_session, bool)
            or self.max_trades_per_session != 1
        ):
            raise ValueError("max_trades_per_session must be exactly 1")
        if (
            not isinstance(self.max_open_positions, int)
            or isinstance(self.max_open_positions, bool)
            or self.max_open_positions != 1
        ):
            raise ValueError("max_open_positions must be exactly 1")


@dataclass(frozen=True, slots=True)
class HoldDurationCompliance:
    """Funded payout hold-duration ratios from executed trades only."""

    threshold_seconds: float | None
    total_trades: int
    qualifying_trades: int
    trade_fraction: float | None
    total_positive_profit_usd: float
    qualifying_positive_profit_usd: float
    positive_profit_fraction: float | None
    trade_fraction_required: float | None
    positive_profit_fraction_required: float | None
    compliant: bool


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """Compact immutable audit event for one phase run."""

    kind: SimulationEventKind
    session_date: date
    trade_id: str | None
    balance_usd: float
    net_liquidation_usd: float
    detail: str


@dataclass(frozen=True, slots=True)
class PropSimulationResult:
    """Survival-first result of a fresh-account phase simulation."""

    profile_id: str
    rule_set_name: str
    phase: AccountPhase
    termination: SimulationTermination
    initial_balance_usd: float
    final_balance_usd: float
    net_profit_usd: float
    maximum_drawdown_usd: float
    minimum_distance_to_failure_usd: float
    sessions_elapsed: int
    account_lifetime_sessions: int
    trading_days: int
    executed_trades: int
    rejected_trades: int
    rejected_daily_loss_trades: int
    rejected_contract_limit_trades: int
    rejected_personal_risk_trades: int
    rejected_trade_quota_trades: int
    rejected_daily_strategy_loss_trades: int
    rejected_post_loss_risk_increase_trades: int
    rejected_pretrade_safety_trades: int
    rejected_market_day_trades: int
    total_commission_usd: float
    total_fees_usd: float
    total_slippage_usd: float
    evaluation_passed: bool
    sessions_to_pass: int | None
    account_failed: bool
    internal_safety_locked: bool
    payout_eligible: bool
    sessions_to_payout: int | None
    hold_duration: HoldDurationCompliance
    max_consecutive_losses: int
    final_state: PropAccountState
    events: tuple[SimulationEvent, ...]


@dataclass(frozen=True, slots=True)
class PropJourneyResult:
    """A fresh evaluation followed by a separate fresh funded account."""

    profile_id: str
    evaluation_rule_set_name: str
    funded_rule_set_name: str
    terminal_stage: JourneyStage
    failure_stage: JourneyStage | None
    termination: SimulationTermination
    total_sessions_elapsed: int
    account_lifetime_sessions: int
    sessions_to_evaluation_pass: int | None
    sessions_to_payout: int | None
    evaluation_passed: bool
    funded_started: bool
    payout_eligible: bool
    account_failed: bool
    internal_safety_locked: bool
    combined_net_profit_usd: float
    evaluation: PropSimulationResult
    funded: PropSimulationResult | None


__all__ = [
    "HistoricalPropSession",
    "HistoricalPropTrade",
    "HoldDurationCompliance",
    "JourneyStage",
    "PropJourneyResult",
    "PropSimulationResult",
    "PropSimulationPolicy",
    "SimulationEvent",
    "SimulationEventKind",
    "SimulationTermination",
]
