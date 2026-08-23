"""Fail-closed composition of the three independent entry-risk layers.

``ThreeLayerRiskEngine`` is the entry approval boundary.  It deliberately owns no
strategy, prop-account, or broker state: current external facts arrive through an
immutable context supplied afresh for both approval and broker-side verification.

The ordering is security-significant:

* Strategy validation runs before personal sizing can mint an approval.
* Prop validation runs after sizing, against the exact sized quantity and risk.
* At broker verification, strategy and prop facts are refreshed before the personal
  token is verified/spent.  A failed outer gate therefore cannot consume a token.

Risk-reducing orders are intentionally different.  They delegate straight to the
personal engine's token verification so an entry-only prop rule can never trap exposure.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Callable

from ..core.models import Order, OrderIntent, Position, Rejection, Trade
from ..core.types import OrderPurpose, RejectReason
from ..deployment.stages import DeploymentStage, StageAuthorization
from ..prop_firms.models import AccountRuleSet, PropFirmProfile
from .broker_state import AuthoritativeBrokerSnapshot, BrokerRiskAccount
from .limits import Approval, RiskDecision, RiskEngine
from .prop import (
    MarketDayStatus,
    PropAccountState,
    PropGateDecision,
    PropGateReason,
    PropOrderAction,
    PropPreTradeRequest,
    evaluate_prop_order,
)
from .strategy_gate import StrategyGate, StrategyGateDecision
from .tokens import RiskToken


class ContractKind(str, Enum):
    """How one strategy contract maps to the prop profile's contract limits."""

    MINI = "mini"
    MICRO = "micro"


@dataclass(frozen=True, slots=True)
class RuntimeRiskContext:
    """Authoritative, point-in-time facts required by the outer risk layers.

    Optional annotations are intentional: a provider can represent an unavailable fact,
    but the coordinator will refuse the entry rather than inventing a value.  Stage 0-2
    may omit ``stage_authorization``; Stage 3-4 never may.
    """

    prop_state: PropAccountState | None
    market_data_timestamp: datetime | None
    strategy_permitted: bool | None
    session_permitted: bool | None
    market_day_status: MarketDayStatus | None
    rule_verification_as_of: datetime | None
    stage_authorization: StageAuthorization | None = None


RuntimeContextProvider = Callable[[datetime], RuntimeRiskContext]


@dataclass(frozen=True, slots=True)
class RuntimeContextTrace:
    """Journal-safe result of obtaining and binding one runtime context."""

    evaluated: bool
    allowed: bool
    reason_codes: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    facts: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def skipped(cls, reason: str) -> "RuntimeContextTrace":
        return cls(False, False, ("not_evaluated",), (reason,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "details": list(self.details),
            "facts": {key: _json_safe(value) for key, value in self.facts},
        }


@dataclass(frozen=True, slots=True)
class StrategyLayerTrace:
    """Layer-1 result, including an explicit marker when short-circuited."""

    decision: StrategyGateDecision | None = None
    skipped_reason: str | None = None

    @property
    def evaluated(self) -> bool:
        return self.decision is not None

    @property
    def allowed(self) -> bool:
        return self.decision is not None and self.decision.approved

    @classmethod
    def skipped(cls, reason: str) -> "StrategyLayerTrace":
        return cls(skipped_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        if self.decision is None:
            return {
                "layer": "STRATEGY",
                "evaluated": False,
                "allowed": False,
                "skipped_reason": self.skipped_reason,
            }
        payload = self.decision.to_dict()
        payload.update({"evaluated": True, "allowed": self.decision.approved})
        return _json_safe(payload)


@dataclass(frozen=True, slots=True)
class PersonalLayerTrace:
    """A deliberately token-free Layer-2 trace.

    The raw :class:`RiskDecision` is never retained here.  That prevents an approval
    minted by the personal engine from escaping when the independent prop layer refuses.
    """

    evaluated: bool
    allowed: bool
    reason_code: str | None = None
    detail: str = ""
    stage: str | None = None
    risk_usd: float | None = None
    contracts: int | None = None
    skipped_reason: str | None = None

    @classmethod
    def skipped(cls, reason: str) -> "PersonalLayerTrace":
        return cls(False, False, skipped_reason=reason)

    @classmethod
    def from_decision(cls, decision: RiskDecision) -> "PersonalLayerTrace":
        if decision.approval is not None:
            return cls(
                True,
                True,
                risk_usd=decision.approval.risk_usd,
                contracts=decision.approval.contracts,
                detail=decision.approval.detail,
            )
        rejection = decision.rejection
        return cls(
            True,
            False,
            reason_code=(rejection.reason.value if rejection is not None else "invalid_decision"),
            detail=(rejection.detail if rejection is not None else "personal gate returned no result"),
            stage=(rejection.stage if rejection is not None else "RISK"),
        )

    @classmethod
    def from_verification(cls, rejection: Rejection | None) -> "PersonalLayerTrace":
        if rejection is None:
            return cls(True, True, detail="personal token and current limits verified")
        return cls(
            True,
            False,
            reason_code=rejection.reason.value,
            detail=rejection.detail,
            stage=rejection.stage,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "layer": "PERSONAL",
                "evaluated": self.evaluated,
                "allowed": self.allowed,
                "reason_code": self.reason_code,
                "detail": self.detail,
                "stage": self.stage,
                "risk_usd": self.risk_usd,
                "contracts": self.contracts,
                "skipped_reason": self.skipped_reason,
            }
        )


@dataclass(frozen=True, slots=True)
class PropLayerTrace:
    """Layer-3 result, retaining official reason codes but no account object."""

    decision: PropGateDecision | None = None
    skipped_reason: str | None = None

    @property
    def evaluated(self) -> bool:
        return self.decision is not None

    @property
    def allowed(self) -> bool:
        return self.decision is not None and self.decision.allowed

    @classmethod
    def skipped(cls, reason: str) -> "PropLayerTrace":
        return cls(skipped_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        decision = self.decision
        return {
            "layer": "PROP_FIRM",
            "evaluated": decision is not None,
            "allowed": decision is not None and decision.allowed,
            "reason_codes": (
                [reason.value for reason in decision.reason_codes] if decision else []
            ),
            "details": list(decision.details) if decision else [],
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True, slots=True)
class ThreeLayerDecision:
    """One complete, journal-safe account of a coordinated risk decision."""

    operation: str
    allowed: bool
    context_trace: RuntimeContextTrace
    strategy_trace: StrategyLayerTrace
    personal_trace: PersonalLayerTrace
    prop_trace: PropLayerTrace
    approval: Approval | None = None
    rejection: Rejection | None = None

    @property
    def approved(self) -> bool:
        """Compatibility with callers that consume a normal ``RiskDecision``."""
        return self.allowed

    def to_dict(self) -> dict[str, Any]:
        approval: dict[str, Any] | None = None
        if self.approval is not None:
            order = self.approval.order
            # Deliberately omit the RiskToken, its id, and its signature.
            approval = {
                "risk_usd": self.approval.risk_usd,
                "contracts": self.approval.contracts,
                "detail": self.approval.detail,
                "order": {
                    "order_id": order.order_id,
                    "intent_id": order.intent_id,
                    "timestamp": order.timestamp,
                    "instrument": order.instrument,
                    "side": order.side,
                    "quantity": order.quantity,
                    "order_type": order.order_type,
                    "purpose": order.purpose,
                },
            }
        rejection: dict[str, Any] | None = None
        if self.rejection is not None:
            rejection = {
                "timestamp": self.rejection.timestamp,
                "reason": self.rejection.reason,
                "detail": self.rejection.detail,
                "stage": self.rejection.stage,
                "instrument": self.rejection.instrument,
                "strategy": self.rejection.strategy,
                "intent_id": self.rejection.intent_id,
                "order_id": self.rejection.order_id,
                "context": self.rejection.context,
            }
        return _json_safe(
            {
                "operation": self.operation,
                "allowed": self.allowed,
                "context": self.context_trace.to_dict(),
                "strategy": self.strategy_trace.to_dict(),
                "personal": self.personal_trace.to_dict(),
                "prop_firm": self.prop_trace.to_dict(),
                "approval": approval,
                "rejection": rejection,
            }
        )


class ThreeLayerRiskEngine:
    """The sole entry approval API across strategy, personal, and prop risk."""

    def __init__(
        self,
        *,
        strategy_gate: StrategyGate,
        personal_risk: RiskEngine,
        profile: PropFirmProfile,
        rules: AccountRuleSet,
        deployment_stage: DeploymentStage,
        contract_kind: ContractKind,
        context_provider: RuntimeContextProvider,
    ) -> None:
        if not isinstance(strategy_gate, StrategyGate):
            raise TypeError("strategy_gate must be a StrategyGate")
        if not isinstance(personal_risk, RiskEngine):
            raise TypeError("personal_risk must be a RiskEngine")
        if not isinstance(profile, PropFirmProfile):
            raise TypeError("profile must be a PropFirmProfile")
        if not isinstance(rules, AccountRuleSet):
            raise TypeError("rules must be an AccountRuleSet")
        if not isinstance(deployment_stage, DeploymentStage):
            raise TypeError("deployment_stage must be a DeploymentStage")
        if not isinstance(contract_kind, ContractKind):
            raise TypeError("contract_kind must be a ContractKind")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")

        profile.validate()
        rules.validate()
        if rules not in profile.phases:
            raise ValueError("rules must belong to the fixed prop profile")
        if strategy_gate.instrument.symbol != personal_risk.instrument.symbol:
            raise ValueError("strategy and personal gates must use the same instrument")

        self._strategy_gate = strategy_gate
        self._personal = personal_risk
        self._profile = profile
        self._rules = rules
        self._deployment_stage = deployment_stage
        self._contract_kind = contract_kind
        self._context_provider = context_provider
        self._pending_intents: dict[str, OrderIntent] = {}
        self._pending_account_ids: dict[str, str | None] = {}
        self._pending_lock = RLock()
        self._last_verify_decision: ThreeLayerDecision | None = None

    # ----------------------------------------------------------- entry approval

    def evaluate_entry(
        self,
        intent: OrderIntent,
        *,
        equity: float | None = None,
        position: Position | None = None,
        now: datetime | None = None,
    ) -> ThreeLayerDecision:
        now = now or self.clock.now()
        context, context_trace = self._runtime_context(now)
        if context is None:
            rejection = self._context_rejection(now, intent=intent, trace=context_trace)
            return ThreeLayerDecision(
                "ENTRY_EVALUATE",
                False,
                context_trace,
                StrategyLayerTrace.skipped("runtime context rejected"),
                PersonalLayerTrace.skipped("runtime context rejected"),
                PropLayerTrace.skipped("runtime context rejected"),
                rejection=rejection,
            )

        strategy_decision = self._strategy_gate.evaluate(
            intent,
            now=now,
            market_data_timestamp=context.market_data_timestamp,
            strategy_permitted=context.strategy_permitted,
            session_permitted=context.session_permitted,
        )
        strategy_trace = StrategyLayerTrace(strategy_decision)
        if not strategy_decision.approved:
            rejection = self._strategy_rejection(now, intent, strategy_decision)
            return ThreeLayerDecision(
                "ENTRY_EVALUATE",
                False,
                context_trace,
                strategy_trace,
                PersonalLayerTrace.skipped("Layer 1 rejected"),
                PropLayerTrace.skipped("Layer 1 rejected"),
                rejection=rejection,
            )

        personal_decision = self._personal.evaluate_entry(
            intent, equity=equity, position=position, now=now
        )
        personal_trace = PersonalLayerTrace.from_decision(personal_decision)
        approval = personal_decision.approval
        if approval is None:
            rejection = personal_decision.rejection or self._rejection(
                now,
                RejectReason.INVALID_ORDER,
                "personal risk gate returned neither approval nor rejection",
                "RISK",
                intent=intent,
            )
            return ThreeLayerDecision(
                "ENTRY_EVALUATE",
                False,
                context_trace,
                strategy_trace,
                personal_trace,
                PropLayerTrace.skipped("Layer 2 rejected"),
                rejection=rejection,
            )

        prop_decision = self._evaluate_prop(
            now=now,
            context=context,
            quantity=approval.order.quantity,
            risk_usd=approval.risk_usd,
        )
        prop_trace = PropLayerTrace(prop_decision)
        if not prop_decision.allowed:
            # The Layer-2 Approval (and its token) is intentionally not returned.
            rejection = self._prop_rejection(
                now,
                prop_decision,
                intent=intent,
                order=approval.order,
            )
            return ThreeLayerDecision(
                "ENTRY_EVALUATE",
                False,
                context_trace,
                strategy_trace,
                personal_trace,
                prop_trace,
                rejection=rejection,
            )

        with self._pending_lock:
            self._pending_intents[approval.order.order_id] = deepcopy(intent)
            authorization = context.stage_authorization
            self._pending_account_ids[approval.order.order_id] = (
                authorization.account_id
                if (
                    self._deployment_stage.requires_human_approval
                    and isinstance(authorization, StageAuthorization)
                )
                else None
            )
        return ThreeLayerDecision(
            "ENTRY_EVALUATE",
            True,
            context_trace,
            strategy_trace,
            personal_trace,
            prop_trace,
            approval=approval,
        )

    # ------------------------------------------------------------ broker guard

    def verify(
        self,
        order: Order,
        token: RiskToken,
        *,
        now: datetime | None = None,
        broker_snapshot: AuthoritativeBrokerSnapshot | None = None,
    ) -> Rejection | None:
        """Revalidate an order and return the broker guard's expected shape."""
        now = now or self.clock.now()
        decision = self._verify_decision(
            order, token, now=now, broker_snapshot=broker_snapshot
        )
        self._last_verify_decision = decision
        return decision.rejection

    def _verify_decision(
        self,
        order: Order,
        token: RiskToken,
        *,
        now: datetime,
        broker_snapshot: AuthoritativeBrokerSnapshot | None,
    ) -> ThreeLayerDecision:
        if order.purpose is not OrderPurpose.ENTRY:
            personal_rejection = self._personal.verify(
                order,
                token,
                now=now,
                broker_snapshot=broker_snapshot,
            )
            return ThreeLayerDecision(
                "RISK_REDUCING_VERIFY",
                personal_rejection is None,
                RuntimeContextTrace.skipped("entry-only context bypassed for risk reduction"),
                StrategyLayerTrace.skipped("entry-only gate bypassed for risk reduction"),
                PersonalLayerTrace.from_verification(personal_rejection),
                PropLayerTrace.skipped("entry-only gate bypassed for risk reduction"),
                rejection=personal_rejection,
            )

        with self._pending_lock:
            stored_intent = self._pending_intents.get(order.order_id)
            stored_intent = deepcopy(stored_intent) if stored_intent is not None else None
            stored_account_id = self._pending_account_ids.get(order.order_id)
        if stored_intent is None:
            rejection = self._rejection(
                now,
                RejectReason.INVALID_TOKEN,
                "entry order has no coordinated three-layer approval",
                "COORDINATOR",
                order=order,
            )
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                RuntimeContextTrace.skipped("unknown entry order"),
                StrategyLayerTrace.skipped("no stored exact intent"),
                PersonalLayerTrace.skipped("unknown entry rejected before token verification"),
                PropLayerTrace.skipped("no stored exact intent"),
                rejection=rejection,
            )

        context, context_trace = self._runtime_context(now)
        if context is None:
            rejection = self._context_rejection(
                now, intent=stored_intent, order=order, trace=context_trace
            )
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                context_trace,
                StrategyLayerTrace.skipped("runtime context rejected"),
                PersonalLayerTrace.skipped("outer gates rejected before token verification"),
                PropLayerTrace.skipped("runtime context rejected"),
                rejection=rejection,
            )

        strategy_decision = self._strategy_gate.evaluate(
            stored_intent,
            now=now,
            market_data_timestamp=context.market_data_timestamp,
            strategy_permitted=context.strategy_permitted,
            session_permitted=context.session_permitted,
        )
        strategy_trace = StrategyLayerTrace(strategy_decision)
        if not strategy_decision.approved:
            rejection = self._strategy_rejection(
                now, stored_intent, strategy_decision, order=order
            )
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                context_trace,
                strategy_trace,
                PersonalLayerTrace.skipped("Layer 1 rejected before token verification"),
                PropLayerTrace.skipped("Layer 1 rejected"),
                rejection=rejection,
            )

        prop_decision = self._evaluate_prop(
            now=now,
            context=context,
            quantity=order.quantity,
            risk_usd=token.risk_usd,
        )
        prop_trace = PropLayerTrace(prop_decision)
        if not prop_decision.allowed:
            rejection = self._prop_rejection(
                now, prop_decision, intent=stored_intent, order=order
            )
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                context_trace,
                strategy_trace,
                PersonalLayerTrace.skipped("Layer 3 rejected before token verification"),
                prop_trace,
                rejection=rejection,
            )

        identity_rejection = self._authorized_broker_identity_rejection(
            now=now,
            context=context,
            broker_snapshot=broker_snapshot,
            stored_account_id=stored_account_id,
            order=order,
        )
        if identity_rejection is not None:
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                context_trace,
                strategy_trace,
                PersonalLayerTrace.skipped(
                    "broker identity rejected before personal token verification"
                ),
                prop_trace,
                rejection=identity_rejection,
            )

        personal_rejection = self._personal.verify(
            order, token, now=now, broker_snapshot=broker_snapshot
        )
        personal_trace = PersonalLayerTrace.from_verification(personal_rejection)
        if personal_rejection is not None:
            return ThreeLayerDecision(
                "ENTRY_VERIFY",
                False,
                context_trace,
                strategy_trace,
                personal_trace,
                prop_trace,
                rejection=personal_rejection,
            )

        with self._pending_lock:
            self._pending_intents.pop(order.order_id, None)
            self._pending_account_ids.pop(order.order_id, None)
        return ThreeLayerDecision(
            "ENTRY_VERIFY",
            True,
            context_trace,
            strategy_trace,
            personal_trace,
            prop_trace,
        )

    @property
    def last_verify_decision(self) -> ThreeLayerDecision | None:
        return self._last_verify_decision

    # ------------------------------------------------------ explicit delegation

    @property
    def clock(self):
        return self._personal.clock

    @property
    def kill_switch(self):
        return self._personal.kill_switch

    @property
    def state(self):
        return self._personal.state

    @property
    def config(self):
        return self._personal.config

    @property
    def instrument(self):
        return self._personal.instrument

    @property
    def session(self):
        return self._personal.session

    @property
    def floor_breached_at(self):
        return self._personal.floor_breached_at

    def evaluate_exit(
        self, position: Position, *, reason: str = "EXIT", now: datetime | None = None
    ) -> RiskDecision:
        return self._personal.evaluate_exit(position, reason=reason, now=now)

    def evaluate_protective(
        self, position: Position, order: Order, *, now: datetime | None = None
    ) -> RiskDecision:
        return self._personal.evaluate_protective(position, order, now=now)

    def reconcile_broker_snapshot(
        self,
        snapshot: AuthoritativeBrokerSnapshot | None,
        *,
        now: datetime | None = None,
        order: Order | None = None,
    ) -> Rejection | None:
        return self._personal.reconcile_broker_snapshot(
            snapshot, now=now, order=order
        )

    def roll_session(self, now: datetime) -> bool:
        rolled = self._personal.roll_session(now)
        if rolled:
            with self._pending_lock:
                self._pending_intents.clear()
                self._pending_account_ids.clear()
        return rolled

    def on_position_opened(self, position: Position) -> None:
        self._personal.on_position_opened(position)

    def on_trade_closed(self, trade: Trade) -> None:
        self._personal.on_trade_closed(trade)

    def forced_exit_reason(self, position: Position, mark: float, now: datetime):
        personal_reason = self._personal.forced_exit_reason(position, mark, now)
        if personal_reason is not None:
            return personal_reason

        # An entry-only rule must never trap exposure, but an *open* position still has
        # to react to the authoritative prop-account state. Refresh the same runtime
        # context used by the outer gate after the broker has marked this bar. Unknown or
        # invalid account/calendar/stage facts cause a fail-safe flatten, not permission
        # to keep carrying risk.
        context, trace = self._runtime_context(now)
        if context is None:
            return (
                RejectReason.BROKER_ERROR,
                "prop-account risk context is unavailable while exposure is open; "
                f"fail-safe flatten ({', '.join(trace.reason_codes)})",
            )

        state = context.prop_state
        assert isinstance(state, PropAccountState)  # guaranteed by _runtime_context
        status = context.market_day_status
        if status in (MarketDayStatus.CLOSED, MarketDayStatus.UNKNOWN):
            return (
                RejectReason.OUTSIDE_TRADING_HOURS,
                f"market-day status is {status.value}; open exposure must be flattened",
            )
        if self._profile.trading_window.position_must_be_flat(
            now, holiday=status is MarketDayStatus.EARLY_CLOSE
        ):
            deadline = (
                self._profile.trading_window.holiday_flatten_by
                if status is MarketDayStatus.EARLY_CLOSE
                else self._profile.trading_window.flatten_by
            )
            return (
                RejectReason.OUTSIDE_TRADING_HOURS,
                f"prop-firm flatten deadline {deadline} reached; no overnight exposure",
            )
        if state.hard_breached:
            return (
                RejectReason.TRAILING_DRAWDOWN,
                "prop account is hard-breached: "
                + (", ".join(state.hard_breach_reasons) or "unknown hard breach"),
            )
        if (
            state.real_time_net_liquidation_usd
            <= state.internal_safety_threshold_usd + 1e-9
        ):
            return (
                RejectReason.TRAILING_DRAWDOWN,
                f"real-time net liquidation ${state.real_time_net_liquidation_usd:,.2f} "
                "is at/below the internal prop safety threshold "
                f"${state.internal_safety_threshold_usd:,.2f}",
            )
        if state.soft_daily_loss_locked or (
            state.firm_daily_loss_limit_usd is not None
            and state.daily_pnl_usd <= -state.firm_daily_loss_limit_usd + 1e-9
        ):
            return (
                RejectReason.MAX_DAILY_LOSS,
                f"prop daily P&L ${state.daily_pnl_usd:,.2f} has reached the "
                f"firm limit ${state.firm_daily_loss_limit_usd:,.2f}",
            )
        return None

    def limits_snapshot(self) -> dict:
        return self._personal.limits_snapshot()

    def cushion_fraction(self) -> float:
        return self._personal.cushion_fraction()

    # -------------------------------------------------------------- internals

    def _runtime_context(
        self, now: datetime
    ) -> tuple[RuntimeRiskContext | None, RuntimeContextTrace]:
        if not _aware(now):
            trace = RuntimeContextTrace(
                True,
                False,
                ("invalid_current_time",),
                ("coordinator time must be timezone-aware",),
            )
            return None, trace
        try:
            context = self._context_provider(now)
        except Exception as exc:  # external reconciliation must fail closed
            trace = RuntimeContextTrace(
                True,
                False,
                ("provider_exception",),
                (f"runtime context provider raised {type(exc).__name__}",),
            )
            return None, trace

        if not isinstance(context, RuntimeRiskContext):
            trace = RuntimeContextTrace(
                True,
                False,
                ("context_missing_or_invalid",),
                ("provider must return RuntimeRiskContext",),
            )
            return None, trace

        reasons: list[str] = []
        details: list[str] = []
        state = context.prop_state
        if not isinstance(state, PropAccountState):
            reasons.append("prop_state_missing_or_invalid")
            details.append("context must contain an immutable PropAccountState")
        else:
            if state.profile_id != self._profile.profile_id:
                reasons.append("prop_profile_mismatch")
                details.append("runtime account state does not match the fixed prop profile")
            if state.rule_set_name != self._rules.name:
                reasons.append("prop_rule_set_mismatch")
                details.append("runtime account state does not match the fixed rule set")
            if state.phase is not self._rules.phase:
                reasons.append("prop_phase_mismatch")
                details.append("runtime account phase does not match the fixed rule set")

        if not isinstance(context.market_data_timestamp, datetime):
            reasons.append("market_data_timestamp_missing_or_invalid")
            details.append("context must supply a market-data timestamp")
        elif not _aware(context.market_data_timestamp):
            reasons.append("market_data_timestamp_naive")
            details.append("market-data timestamp must be timezone-aware")
        if not isinstance(context.strategy_permitted, bool):
            reasons.append("strategy_permission_missing_or_invalid")
            details.append("strategy permission must be an explicit boolean")
        if not isinstance(context.session_permitted, bool):
            reasons.append("session_permission_missing_or_invalid")
            details.append("session permission must be an explicit boolean")
        if not isinstance(context.market_day_status, MarketDayStatus):
            reasons.append("market_day_status_missing_or_invalid")
            details.append("context must supply a MarketDayStatus")
        if not isinstance(context.rule_verification_as_of, datetime):
            reasons.append("rule_verification_as_of_missing_or_invalid")
            details.append("context must supply a rule-verification as-of timestamp")
        elif not _aware(context.rule_verification_as_of):
            reasons.append("rule_verification_as_of_naive")
            details.append("rule-verification as-of timestamp must be timezone-aware")

        authorization = context.stage_authorization
        if authorization is None:
            if self._deployment_stage.requires_human_approval:
                reasons.append("stage_authorization_missing")
                details.append("Stage 3/4 requires a fresh StageAuthorization")
        elif not isinstance(authorization, StageAuthorization):
            reasons.append("stage_authorization_invalid")
            details.append("stage authorization must be a StageAuthorization object")
        else:
            if authorization.stage is not self._deployment_stage:
                reasons.append("stage_authorization_mismatch")
                details.append("authorization stage does not match the fixed deployment stage")
            if authorization.allowed is not True:
                reasons.append("stage_authorization_denied")
                details.append("runtime stage authorization is denied")
            if self._deployment_stage.requires_human_approval and (
                not isinstance(authorization.account_id, str)
                or not authorization.account_id.strip()
            ):
                reasons.append("stage_authorized_account_missing")
                details.append(
                    "Stage 3/4 authorization must bind a non-empty manifest account id"
                )

        facts = (
            ("profile_id", state.profile_id if isinstance(state, PropAccountState) else None),
            ("rule_set_name", state.rule_set_name if isinstance(state, PropAccountState) else None),
            ("account_phase", state.phase if isinstance(state, PropAccountState) else None),
            (
                "prop_account_state",
                state.to_dict() if isinstance(state, PropAccountState) else None,
            ),
            ("market_data_timestamp", context.market_data_timestamp),
            ("strategy_permitted", context.strategy_permitted),
            ("session_permitted", context.session_permitted),
            ("market_day_status", context.market_day_status),
            ("rule_verification_as_of", context.rule_verification_as_of),
            ("deployment_stage", self._deployment_stage),
            ("stage_authorization_present", authorization is not None),
            (
                "stage_authorized",
                authorization.allowed if isinstance(authorization, StageAuthorization) else None,
            ),
            (
                "authorized_account_id_present",
                bool(
                    isinstance(authorization, StageAuthorization)
                    and isinstance(authorization.account_id, str)
                    and authorization.account_id.strip()
                ),
            ),
            (
                "authorized_execution_route",
                authorization.execution_route
                if isinstance(authorization, StageAuthorization)
                else None,
            ),
        )
        trace = RuntimeContextTrace(
            True,
            not reasons,
            tuple(reasons),
            tuple(details),
            facts,
        )
        return (context if trace.allowed else None), trace

    def _authorized_broker_identity_rejection(
        self,
        *,
        now: datetime,
        context: RuntimeRiskContext,
        broker_snapshot: AuthoritativeBrokerSnapshot | None,
        stored_account_id: str | None,
        order: Order,
    ) -> Rejection | None:
        """Bind a Stage 3/4 token to manifest identity before personal verification."""
        if not self._deployment_stage.requires_human_approval:
            return None

        authorization = context.stage_authorization
        authorized_account_id = (
            authorization.account_id
            if isinstance(authorization, StageAuthorization)
            else None
        )
        # _runtime_context already validates the type/non-empty invariant. Keep this
        # defensive check local so a future caller cannot weaken the ordering guarantee.
        if not isinstance(authorized_account_id, str) or not authorized_account_id.strip():
            return self._rejection(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "Stage 3/4 broker identity is not bound by the authorization manifest",
                "BROKER_IDENTITY",
                order=order,
            )
        if stored_account_id != authorized_account_id:
            return self._rejection(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "authorized broker account identity changed after entry approval",
                "BROKER_IDENTITY",
                order=order,
            )
        if not isinstance(broker_snapshot, AuthoritativeBrokerSnapshot):
            return self._rejection(
                now,
                RejectReason.BROKER_ERROR,
                "Stage 3/4 entry verification requires an authoritative broker snapshot",
                "BROKER_IDENTITY",
                order=order,
            )
        account = broker_snapshot.account
        if (
            not isinstance(account, BrokerRiskAccount)
            or not isinstance(account.account_id, str)
            or not account.account_id
        ):
            return self._rejection(
                now,
                RejectReason.BROKER_ERROR,
                "authoritative broker snapshot has no valid account identity",
                "BROKER_IDENTITY",
                order=order,
            )
        if account.account_id != authorized_account_id:
            return self._rejection(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "authoritative broker account does not match the approved manifest",
                "BROKER_IDENTITY",
                order=order,
            )
        return None

    def _evaluate_prop(
        self,
        *,
        now: datetime,
        context: RuntimeRiskContext,
        quantity: int,
        risk_usd: float,
    ) -> PropGateDecision:
        requested_minis = quantity if self._contract_kind is ContractKind.MINI else 0
        requested_micros = quantity if self._contract_kind is ContractKind.MICRO else 0
        authorization = context.stage_authorization
        request = PropPreTradeRequest(
            action=PropOrderAction.ENTRY,
            now=now,
            rule_verification_as_of=context.rule_verification_as_of,
            requested_minis=requested_minis,
            requested_micros=requested_micros,
            worst_case_loss_usd=risk_usd,
            deployment_stage=self._deployment_stage,
            stage_authorized=(
                authorization is not None
                and authorization.stage is self._deployment_stage
                and authorization.allowed is True
            ),
            market_day_status=context.market_day_status,
            increases_exposure=True,
            averaging_down=False,
            automated=True,
        )
        return evaluate_prop_order(
            request,
            profile=self._profile,
            rules=self._rules,
            state=context.prop_state,
        )

    def _context_rejection(
        self,
        now: datetime,
        *,
        trace: RuntimeContextTrace,
        intent: OrderIntent | None = None,
        order: Order | None = None,
    ) -> Rejection:
        reason = (
            RejectReason.BROKER_ERROR
            if "provider_exception" in trace.reason_codes
            else RejectReason.LIVE_TRADING_DISABLED
            if any(code.startswith("stage_authorization") for code in trace.reason_codes)
            else RejectReason.INVALID_ORDER
        )
        return self._rejection(
            now,
            reason,
            "; ".join(trace.details) or "runtime risk context rejected",
            "COORDINATOR",
            intent=intent,
            order=order,
            context={"context_reason_codes": list(trace.reason_codes)},
        )

    def _strategy_rejection(
        self,
        now: datetime,
        intent: OrderIntent,
        decision: StrategyGateDecision,
        *,
        order: Order | None = None,
    ) -> Rejection:
        codes = decision.failure_codes
        if any(code in {"signal_current", "data_fresh"} for code in codes):
            reason = RejectReason.STALE_MARKET_DATA
        elif "session_permission" in codes:
            reason = RejectReason.OUTSIDE_TRADING_HOURS
        else:
            reason = RejectReason.INVALID_ORDER
        return self._rejection(
            now,
            reason,
            f"strategy gate rejected: {', '.join(codes)}",
            "STRATEGY",
            intent=intent,
            order=order,
            context={"strategy_failure_codes": list(codes)},
        )

    def _prop_rejection(
        self,
        now: datetime,
        decision: PropGateDecision,
        *,
        intent: OrderIntent | None = None,
        order: Order | None = None,
    ) -> Rejection:
        reasons = set(decision.reason_codes)
        if reasons & {
            PropGateReason.HARD_ACCOUNT_BREACH,
            PropGateReason.INTERNAL_SAFETY_THRESHOLD,
        }:
            reason = RejectReason.TRAILING_DRAWDOWN
        elif reasons & {
            PropGateReason.DAILY_LOSS_LIMIT,
            PropGateReason.SOFT_DAILY_LOSS_LOCK,
        }:
            reason = RejectReason.MAX_DAILY_LOSS
        elif PropGateReason.CONTRACT_LIMIT in reasons:
            reason = RejectReason.MAX_POSITION_SIZE
        elif PropGateReason.EXISTING_EXPOSURE in reasons:
            reason = RejectReason.POSITION_ALREADY_OPEN
        elif reasons & {
            PropGateReason.MARKET_CLOSED,
            PropGateReason.OUTSIDE_PERMITTED_HOURS,
            PropGateReason.HOLIDAY_STATUS_UNKNOWN,
            PropGateReason.HOLIDAY_SCHEDULE_UNKNOWN,
        }:
            reason = RejectReason.OUTSIDE_TRADING_HOURS
        elif any(
            item.value.startswith(("profile_", "rule_set_", "state_", "stage_"))
            or item is PropGateReason.AMBIGUOUS_RULES
            for item in reasons
        ):
            reason = RejectReason.LIVE_TRADING_DISABLED
        else:
            reason = RejectReason.INVALID_ORDER
        codes = [item.value for item in decision.reason_codes]
        return self._rejection(
            now,
            reason,
            "; ".join(decision.details) or "prop-firm risk gate rejected",
            "PROP_FIRM",
            intent=intent,
            order=order,
            context={"prop_reason_codes": codes},
        )

    def _rejection(
        self,
        now: datetime,
        reason: RejectReason,
        detail: str,
        stage: str,
        *,
        intent: OrderIntent | None = None,
        order: Order | None = None,
        context: dict[str, Any] | None = None,
    ) -> Rejection:
        return Rejection(
            timestamp=now,
            reason=reason,
            detail=detail,
            stage=stage,
            instrument=(
                intent.instrument
                if intent is not None
                else order.instrument
                if order is not None
                else self.instrument.symbol
            ),
            strategy=(
                intent.strategy
                if intent is not None
                else order.strategy
                if order is not None
                else ""
            ),
            intent_id=(
                intent.intent_id
                if intent is not None
                else order.intent_id
                if order is not None
                else None
            ),
            order_id=order.order_id if order is not None else None,
            context=context or {},
        )


def _aware(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _json_safe(value: Any) -> Any:
    """Convert trace data to JSON-compatible primitives without calling user hooks."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)
