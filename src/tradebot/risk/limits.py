"""The risk engine.

This is the one component that must never be bypassable. It owns the answer to "may we
open a position at all", and it answers **no by default** whenever a limit is breached.

Structure, per PROJECT_SPEC §3.1:

* `evaluate_entry` turns a strategy's `OrderIntent` into either a rejection or an
  `Approval` carrying a concrete `Order` and a single-use `RiskToken`. The strategy never
  chooses the quantity; the sizer does.
* `evaluate_exit` mints a token for a closing order. **Exits are never blocked by a limit.**
  Refusing to flatten because the daily loss cap was hit would be the exact opposite of
  what the cap is for.
* `verify` is called by `GuardedBroker` immediately before an order goes out. It checks the
  token *and independently re-runs the current-state limits*, so an approval issued sixty
  seconds ago cannot execute after a fresh breach.

Every refusal carries a `RejectReason` code and a human sentence, and is returned rather
than logged-and-swallowed, because the dashboard has to show what did not happen.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import CostConfig, RiskConfig, SessionConfig
from ..core.clock import Clock, SystemClock
from ..core.models import Order, OrderIntent, Position, Rejection, Trade, new_id
from ..core.types import (
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)
from ..instruments.registry import InstrumentSpec
from .broker_state import (
    AuthoritativeBrokerSnapshot,
    BrokerRiskAccount,
    BrokerRiskOrder,
    BrokerRiskPosition,
)
from .killswitch import KillSwitch
from .session import SessionGuard
from .sizing import VolatilityTargetSizer
from .state_store import (
    RiskState,
    RiskStateBinding,
    RiskStateStore,
    StoredRiskState,
)
from .tokens import AuthorizationKind, RiskToken, TokenError, TokenMint


_MAX_BROKER_SNAPSHOT_AGE = timedelta(seconds=1)
_MAX_BROKER_SNAPSHOT_READ_DURATION = timedelta(seconds=2)


def _aware_datetime(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _risk_policy_sha256(
    config: RiskConfig,
    session: SessionConfig,
    instrument: InstrumentSpec,
    costs: CostConfig,
    *,
    starting_equity_usd: float,
    enforce_drawdown_floor: bool,
    minimum_expected_rr: float | None,
) -> str:
    """Fingerprint every material personal-risk rule, excluding local file paths."""
    payload = {
        "schema_version": 3,
        # Constructor overrides are effective policy. Fingerprinting only the source
        # config would let the same durable state be reopened under different limits.
        "starting_equity_usd": starting_equity_usd,
        "enforce_drawdown_floor": enforce_drawdown_floor,
        "minimum_expected_rr": minimum_expected_rr,
        "max_open_positions": config.max_open_positions,
        "per_trade": {
            "risk_pct_of_equity": config.per_trade.risk_pct_of_equity,
            "max_risk_per_trade_usd": config.per_trade.max_risk_per_trade_usd,
            "max_contracts": config.per_trade.max_contracts,
            "min_contracts": config.per_trade.min_contracts,
            "max_entry_gap_ticks": config.per_trade.max_entry_gap_ticks,
            "max_stop_gap_ticks": config.per_trade.max_stop_gap_ticks,
        },
        "daily": {
            "max_daily_loss_usd": config.daily.max_daily_loss_usd,
            "max_daily_loss_r": config.daily.max_daily_loss_r,
            "max_trades_per_day": config.daily.max_trades_per_day,
        },
        "streaks": {
            "max_consecutive_losses": config.streaks.max_consecutive_losses,
            "cooldown_minutes": config.streaks.cooldown_minutes,
        },
        "drawdown": {
            "trailing_drawdown_pct": config.drawdown.trailing_drawdown_pct,
            "size_reduction_cushion_pct": config.drawdown.size_reduction_cushion_pct,
        },
        "kill_switch_max_consecutive_errors": config.kill_switch.max_consecutive_errors,
        "token_ttl_seconds": config.token_ttl_seconds,
        "session": {
            "timezone": session.timezone,
            "rth_start": session.rth_start,
            "rth_end": session.rth_end,
            "entry_open_buffer_minutes": session.entry_open_buffer_minutes,
            "entry_close_buffer_minutes": session.entry_close_buffer_minutes,
            "flatten_before_close_minutes": session.flatten_before_close_minutes,
            "news_blackout_windows": list(session.news_blackout_windows),
        },
        "instrument": {
            "symbol": instrument.symbol,
            "multiplier": instrument.multiplier,
            "tick_size": instrument.tick_size,
            "currency": instrument.currency,
        },
        "cost_reserve": {
            "commission_round_trip_usd": costs.commission_round_trip_usd,
            "slippage_ticks_per_side": costs.slippage_ticks_per_side,
            "slippage_stress_multiplier": costs.slippage_stress_multiplier,
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _append_bounded_id(existing: tuple[str, ...], identifier: str) -> tuple[str, ...]:
    if identifier in existing:
        return existing
    return (*existing, identifier)[-256:]


def _intent_fingerprint_key(intent: OrderIntent) -> str:
    """Return a stable JSON-safe identity for durable duplicate detection."""

    payload = {
        "timestamp": intent.timestamp.isoformat(timespec="microseconds"),
        "instrument": intent.instrument,
        "side": intent.side.name,
        "strategy": intent.strategy,
        "stop_price": round(intent.stop_price, 6),
        "target_price": (
            None if intent.target_price is None else round(intent.target_price, 6)
        ),
        "order_type": intent.order_type.value,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class Approval:
    order: Order
    token: RiskToken
    risk_usd: float
    contracts: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approval: Approval | None = None
    rejection: Rejection | None = None

    @property
    def approved(self) -> bool:
        return self.approval is not None


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig,
        instrument: InstrumentSpec,
        session_config: SessionConfig,
        *,
        clock: Clock | None = None,
        kill_switch: KillSwitch | None = None,
        starting_equity: float | None = None,
        enforce_drawdown_floor: bool = True,
        expected_broker_account_id: str | None = None,
        expected_broker_name: str | None = None,
        expected_broker_is_paper: bool | None = None,
        expected_broker_route: str | None = None,
        risk_state_store: RiskStateStore | None = None,
        bootstrap_risk_state_store: bool = False,
        risk_state_context_id: str | None = None,
        cost_config: CostConfig | None = None,
        minimum_expected_rr: float | None = None,
    ) -> None:
        for label, value in (
            ("expected_broker_account_id", expected_broker_account_id),
            ("expected_broker_name", expected_broker_name),
            ("expected_broker_route", expected_broker_route),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{label} must be null or a non-empty string")
        if expected_broker_is_paper is not None and not isinstance(
            expected_broker_is_paper, bool
        ):
            raise ValueError("expected_broker_is_paper must be null or boolean")
        if type(bootstrap_risk_state_store) is not bool:
            raise ValueError("bootstrap_risk_state_store must be boolean")
        if risk_state_context_id is not None and (
            not isinstance(risk_state_context_id, str)
            or not risk_state_context_id.strip()
        ):
            raise ValueError("risk_state_context_id must be null or non-empty")
        if risk_state_store is not None and not isinstance(
            risk_state_store, RiskStateStore
        ):
            raise TypeError("risk_state_store must implement RiskStateStore")
        if minimum_expected_rr is not None and (
            isinstance(minimum_expected_rr, bool)
            or not isinstance(minimum_expected_rr, (int, float))
            or not math.isfinite(minimum_expected_rr)
            or minimum_expected_rr <= 0
        ):
            raise ValueError("minimum_expected_rr must be null or finite and positive")
        self.config = config
        self.instrument = instrument
        self.cost_config = cost_config or CostConfig()
        self.minimum_expected_rr = (
            None if minimum_expected_rr is None else float(minimum_expected_rr)
        )
        self.cost_config.validate("costs")
        self.clock = clock or SystemClock()
        self.session = SessionGuard(session_config)
        self._session_timezone = ZoneInfo(session_config.timezone)
        self.kill_switch = kill_switch or KillSwitch(
            config.kill_switch.flag_file,
            max_consecutive_errors=config.kill_switch.max_consecutive_errors,
            clock=self.clock,
        )
        self.sizer = VolatilityTargetSizer(config.per_trade, instrument)
        # The mint is deliberately private.  Execution may request a narrowly validated
        # authorization, but it cannot manufacture one for an arbitrary Order.
        self._tokens = TokenMint(ttl_seconds=config.token_ttl_seconds)

        equity = starting_equity if starting_equity is not None else config.starting_equity_usd
        state_now = self.clock.now()
        self.state = RiskState(
            equity=equity,
            peak_equity=equity,
            session_start_equity=equity,
            updated_at=state_now,
        )
        # Breaching the trailing floor is terminal: peak equity never falls, so the account
        # can never trade back. That mirrors a real prop account being closed, but it
        # silently truncates a research backtest, so it can be disabled for measurement and
        # the breach is recorded either way.
        self.enforce_drawdown_floor = enforce_drawdown_floor
        self.floor_breached_at: datetime | None = None
        # Explicit pins avoid trust-on-first-use where a runner knows its exact identity.
        # When omitted, the first structurally valid read remains a conservative binding
        # for backwards-compatible Stage 0-2 construction.
        self._broker_account_id = expected_broker_account_id
        self._broker_name = expected_broker_name
        self._broker_is_paper = expected_broker_is_paper
        self._broker_route = (
            expected_broker_route.strip().casefold()
            if expected_broker_route is not None
            else None
        )
        self._risk_policy_sha256 = _risk_policy_sha256(
            config,
            session_config,
            instrument,
            self.cost_config,
            starting_equity_usd=equity,
            enforce_drawdown_floor=enforce_drawdown_floor,
            minimum_expected_rr=self.minimum_expected_rr,
        )
        self._risk_state_context_id = (
            risk_state_context_id.strip() if risk_state_context_id is not None else None
        )
        self._deployment_context_verified = False
        self._risk_state_store = risk_state_store
        self._risk_state_bootstrapped_this_process = (
            risk_state_store is not None and bootstrap_risk_state_store
        )
        self._pending_entry_fingerprints: dict[str, str] = {}
        self._risk_state_store_error: str | None = None
        self._risk_state_recovery_required = risk_state_store is not None
        self._risk_state_recovery_reason = (
            "fresh broker reconciliation is required after startup"
            if risk_state_store is not None
            else ""
        )

        if risk_state_store is not None:
            try:
                restored = risk_state_store.load()
                if restored is None:
                    if not bootstrap_risk_state_store:
                        raise ValueError(
                            "personal-risk state is missing; explicit bootstrap is required"
                        )
                    if not self._initialize_risk_state(state_now):
                        # _initialize_risk_state recorded the durable error.
                        return
                else:
                    self._restore_risk_state(restored)
            except Exception as exc:
                self._risk_state_store_error = (
                    "could not restore personal-risk state: "
                    f"{type(exc).__name__}: {exc}"
                )

    @property
    def risk_state_store_error(self) -> str | None:
        return self._risk_state_store_error

    @property
    def risk_state_recovery_required(self) -> bool:
        return self._risk_state_recovery_required

    @property
    def durable_state_enabled(self) -> bool:
        return self._risk_state_store is not None

    @property
    def durable_state_certified(self) -> bool:
        return bool(getattr(self._risk_state_store, "is_durable", False))

    @property
    def risk_state_bootstrapped_this_process(self) -> bool:
        return self._risk_state_bootstrapped_this_process

    @property
    def risk_state_context_id(self) -> str | None:
        return self._risk_state_context_id

    @property
    def deployment_context_verified(self) -> bool:
        return self._deployment_context_verified

    @property
    def broker_identity_is_pinned(self) -> bool:
        return (
            self._broker_account_id is not None
            and self._broker_name is not None
            and self._broker_is_paper is not None
            and self._broker_route is not None
        )

    def verify_canonical_deployment_context(self, expected_context_id: str) -> None:
        """Mark a structured coordinator binding as verified, or refuse construction."""

        if not isinstance(expected_context_id, str) or not expected_context_id.strip():
            raise ValueError("canonical deployment context must be non-empty")
        if self._risk_state_context_id != expected_context_id:
            raise ValueError(
                "durable personal-risk state is not bound to the coordinator's canonical "
                "deployment context"
            )
        self._deployment_context_verified = True

    def _current_risk_state_binding(self) -> RiskStateBinding:
        return RiskStateBinding(
            instrument=self.instrument.symbol,
            risk_policy_sha256=self._risk_policy_sha256,
            broker_account_id=self._broker_account_id,
            broker_name=self._broker_name,
            broker_is_paper=self._broker_is_paper,
            broker_execution_route=self._broker_route,
            deployment_context_id=self._risk_state_context_id,
        )

    def _restore_risk_state(self, restored: StoredRiskState) -> None:
        if not isinstance(restored, StoredRiskState):
            raise TypeError("risk state store returned an invalid object")
        restored.validate()
        binding = restored.binding
        if binding.instrument != self.instrument.symbol:
            raise ValueError("persisted risk state belongs to a different instrument")
        if binding.risk_policy_sha256 != self._risk_policy_sha256:
            raise ValueError("persisted risk state uses a different risk policy")
        if binding.deployment_context_id != self._risk_state_context_id:
            raise ValueError("persisted risk state belongs to a different deployment context")
        for label, configured, persisted in (
            ("account", self._broker_account_id, binding.broker_account_id),
            ("broker", self._broker_name, binding.broker_name),
            ("route", self._broker_route, binding.broker_execution_route),
        ):
            if configured is not None and persisted is not None and configured != persisted:
                raise ValueError(f"persisted risk state {label} binding differs from runtime")
        if (
            self._broker_is_paper is not None
            and binding.broker_is_paper is not None
            and self._broker_is_paper is not binding.broker_is_paper
        ):
            raise ValueError("persisted risk state paper/live binding differs from runtime")

        self.state = restored.state
        self._broker_account_id = self._broker_account_id or binding.broker_account_id
        self._broker_name = self._broker_name or binding.broker_name
        self._broker_route = self._broker_route or binding.broker_execution_route
        if self._broker_is_paper is None:
            self._broker_is_paper = binding.broker_is_paper

    def _persist_risk_state(
        self,
        now: datetime,
        *,
        exposure_possible: bool = False,
    ) -> bool:
        if self._risk_state_store is None:
            return True
        if self._risk_state_store_error is not None:
            if exposure_possible:
                self._trip_kill_switch_best_effort(
                    "durable personal-risk state is unavailable while exposure exists"
                )
            return False
        previous_revision = self.state.revision
        previous_updated_at = self.state.updated_at
        self.state.revision += 1
        self.state.updated_at = now
        try:
            self._risk_state_store.save(
                StoredRiskState(
                    binding=self._current_risk_state_binding(),
                    state=self.state,
                )
            )
        except Exception as exc:
            self.state.revision = previous_revision
            self.state.updated_at = previous_updated_at
            self._risk_state_store_error = (
                "could not persist personal-risk state: "
                f"{type(exc).__name__}: {exc}"
            )
            if exposure_possible:
                self._trip_kill_switch_best_effort(
                    "personal-risk state persistence failed while exposed"
                )
            return False
        return True

    def _trip_kill_switch_best_effort(self, reason: str) -> None:
        """Latch the durable kill flag without ever sacrificing a proven risk exit.

        A disk or permission failure may take out both state files at once. The in-memory
        durable-state error remains authoritative, so a failed flag write must not raise
        through broker reconciliation or authorization of an exposure-reducing order.
        """

        try:
            self.kill_switch.trip(reason)
        except Exception:
            # There is intentionally nothing else to persist here: persistence is the
            # failed subsystem. forced_exit_reason() continues to fail closed in memory.
            pass

    def _initialize_risk_state(self, now: datetime) -> bool:
        """Perform the store's one-time fresh-account initialization operation."""

        if self._risk_state_store is None:
            return True
        previous_revision = self.state.revision
        previous_updated_at = self.state.updated_at
        self.state.revision += 1
        self.state.updated_at = now
        try:
            self._risk_state_store.initialize(
                StoredRiskState(
                    binding=self._current_risk_state_binding(),
                    state=self.state,
                )
            )
        except Exception as exc:
            self.state.revision = previous_revision
            self.state.updated_at = previous_updated_at
            self._risk_state_store_error = (
                "could not initialize personal-risk state: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        return True

    def _risk_state_entry_rejection(
        self,
        now: datetime,
        *,
        intent: OrderIntent | None = None,
        order: Order | None = None,
    ) -> Rejection | None:
        if self._risk_state_store_error is not None:
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                "durable personal-risk state is unavailable; entries remain locked",
                intent=intent,
                order=order,
                stage="RISK_STATE",
            )
        if self._risk_state_recovery_required:
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                self._risk_state_recovery_reason
                or "durable personal-risk state requires reconciliation",
                intent=intent,
                order=order,
                stage="RISK_STATE",
            )
        return None

    # ================================================================== entry

    def _entry_price_bound(
        self,
        intent: OrderIntent,
        *,
        now: datetime,
    ) -> tuple[float | None, Rejection | None]:
        """Return the worst executable entry price under a venue-enforced limit."""

        if self.minimum_expected_rr is not None:
            for label, price in (
                ("signal reference", intent.reference_price),
                ("protective stop", intent.stop_price),
            ):
                if (
                    isinstance(price, bool)
                    or not isinstance(price, (int, float))
                    or not math.isfinite(price)
                    or price <= 0
                    or abs(self.instrument.round_to_tick(price) - price) > 1e-9
                ):
                    return None, self._reject(
                        now,
                        RejectReason.INVALID_ORDER,
                        f"{label} must be finite, positive, and tick-aligned",
                        intent=intent,
                    )

        if intent.order_type is OrderType.MARKET:
            # Futures orders execute on whole ticks. Flooring the configured allowance is
            # conservative for fractional values and can only make the bound tighter.
            allowed_ticks = math.floor(
                self.config.per_trade.max_entry_gap_ticks + 1e-12
            )
            gap_points = allowed_ticks * self.instrument.tick_size
            raw_bound = (
                intent.reference_price + gap_points
                if intent.side is Side.BUY
                else intent.reference_price - gap_points
            )
            bound = self.instrument.round_to_tick(raw_bound)
        elif intent.order_type is OrderType.LIMIT:
            bound = intent.limit_price
        else:
            return None, self._reject(
                now,
                RejectReason.INVALID_ORDER,
                "entry STOP and STOP_LIMIT orders are disabled because their gap and "
                "trigger semantics are not yet bounded by the personal-risk envelope",
                intent=intent,
            )

        if (
            bound is None
            or isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(bound)
            or bound <= 0
        ):
            return None, self._reject(
                now,
                RejectReason.INVALID_ORDER,
                "entry price bound must be finite and positive",
                intent=intent,
            )
        bound = float(bound)
        if abs(self.instrument.round_to_tick(bound) - bound) > 1e-9:
            return None, self._reject(
                now,
                RejectReason.INVALID_ORDER,
                f"entry limit {bound} is not aligned to the "
                f"{self.instrument.tick_size}-point tick",
                intent=intent,
            )
        stop_is_protective = (
            intent.side is Side.BUY and intent.stop_price < bound
        ) or (
            intent.side is Side.SELL and intent.stop_price > bound
        )
        if not stop_is_protective:
            return None, self._reject(
                now,
                RejectReason.INVALID_ORDER,
                "protective stop is on the wrong side of the worst executable entry price",
                intent=intent,
                entry_price_bound=bound,
            )

        # The generic Stage-0 runner has no prop-account coordinator, but it must not
        # produce research from geometry that the real Layer-1 gate would refuse. This
        # optional defense-in-depth policy is enabled by that runner; coordinated prop
        # paths leave it unset and use StrategyGate's complete trace instead.
        if self.minimum_expected_rr is not None:
            target = intent.target_price
            if (
                target is None
                or isinstance(target, bool)
                or not isinstance(target, (int, float))
                or not math.isfinite(target)
                or target <= 0
                or abs(self.instrument.round_to_tick(target) - target) > 1e-9
            ):
                return None, self._reject(
                    now,
                    RejectReason.INVALID_ORDER,
                    "an explicit finite tick-aligned target is required for research",
                    intent=intent,
                    entry_price_bound=bound,
                )
            reward = (
                target - bound
                if intent.side is Side.BUY
                else bound - target
            )
            stop_risk = abs(bound - intent.stop_price)
            expected_rr = reward / stop_risk if stop_risk > 0 else float("nan")
            if (
                reward <= 0
                or not math.isfinite(expected_rr)
                or expected_rr + 1e-12 < self.minimum_expected_rr
            ):
                return None, self._reject(
                    now,
                    RejectReason.INVALID_ORDER,
                    f"signed execution reward:risk {expected_rr:.4f} is below the "
                    f"minimum {self.minimum_expected_rr:.4f}",
                    intent=intent,
                    entry_price_bound=bound,
                    expected_reward_risk=expected_rr,
                    minimum_expected_reward_risk=self.minimum_expected_rr,
                )
        return bound, None

    def _fixed_risk_reserve_per_contract_usd(self) -> float:
        stressed_exit_slippage = (
            self.cost_config.slippage_ticks_per_side
            * self.cost_config.slippage_stress_multiplier
            * self.instrument.tick_size
            * self.instrument.multiplier
        )
        planned_stop_gap = (
            self.config.per_trade.max_stop_gap_ticks
            * self.instrument.tick_size
            * self.instrument.multiplier
        )
        return (
            self.cost_config.commission_round_trip_usd
            + stressed_exit_slippage
            + planned_stop_gap
        )

    def evaluate_entry(
        self,
        intent: OrderIntent,
        *,
        equity: float | None = None,
        position: Position | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or self.clock.now()
        if equity is not None:
            self.state.equity = equity
            self.state.peak_equity = max(self.state.peak_equity, equity)
        self.roll_session(now)

        blocked = self._entry_blocked(now, position, intent)
        if blocked is not None:
            return RiskDecision(rejection=blocked)

        entry_price_bound, bound_rejection = self._entry_price_bound(intent, now=now)
        if bound_rejection is not None:
            return RiskDecision(rejection=bound_rejection)
        assert entry_price_bound is not None
        stop_distance = abs(entry_price_bound - intent.stop_price)
        # With the floor disabled for research, the cushion throttle must be disabled too.
        # Peak equity never falls, so a fully-consumed cushion multiplies the risk budget
        # by zero and vetoes every subsequent trade -- a second, silent path to exactly the
        # deadlock that turning the floor off was meant to remove, and one that truncates a
        # whole-sample run to a handful of trades without saying why.
        cushion = self.cushion_fraction() if self.enforce_drawdown_floor else 1.0
        sizing = self.sizer.size(
            equity=self.state.equity,
            stop_distance_points=stop_distance,
            fixed_risk_per_contract_usd=(
                self._fixed_risk_reserve_per_contract_usd()
            ),
            cushion_fraction=cushion,
            cushion_threshold=self.config.drawdown.size_reduction_cushion_pct / 100.0,
        )
        if not sizing.approved:
            return RiskDecision(
                rejection=self._reject(
                    now, sizing.reason or RejectReason.SIZE_BELOW_MINIMUM, sizing.detail,
                    intent=intent, stop_distance=stop_distance,
                )
            )

        if self.state.last_trade_was_loss and (
            sizing.contracts > self.state.last_trade_quantity
            or sizing.risk_usd > self.state.last_trade_approved_risk_usd + 1e-9
        ):
            return RiskDecision(
                rejection=self._reject(
                    now,
                    RejectReason.MAX_RISK_PER_TRADE,
                    "post-loss policy forbids increasing contract quantity or approved risk",
                    intent=intent,
                    proposed_contracts=sizing.contracts,
                    previous_contracts=self.state.last_trade_quantity,
                    proposed_risk_usd=sizing.risk_usd,
                    previous_risk_usd=self.state.last_trade_approved_risk_usd,
                )
            )

        # The per-trade dollar cap is re-asserted on the sized result, not just on the
        # budget: rounding down to a whole number of contracts can only reduce risk, but
        # the assertion costs nothing and would catch a future change to the rounding.
        if sizing.risk_usd > self.config.per_trade.max_risk_per_trade_usd + 1e-9:
            return RiskDecision(
                rejection=self._reject(
                    now, RejectReason.MAX_RISK_PER_TRADE,
                    f"sized risk ${sizing.risk_usd:,.2f} exceeds the per-trade cap "
                    f"${self.config.per_trade.max_risk_per_trade_usd:,.2f}",
                    intent=intent,
                )
            )

        order = Order(
            order_id=new_id("ord"),
            timestamp=now,
            instrument=intent.instrument,
            side=intent.side,
            quantity=sizing.contracts,
            # Even a strategy MARKET intent becomes a marketable-or-resting LIMIT at the
            # signed adverse-gap boundary. A missed entry is safe; an unbounded fill is not.
            order_type=OrderType.LIMIT,
            limit_price=entry_price_bound,
            stop_price=None,
            # Entry intents have no signed expiry/revalidation policy yet. Give the venue
            # exactly one opportunity at the next eligible open; a stale resting setup is
            # more dangerous than a missed trade.
            time_in_force=TimeInForce.IOC,
            intent_id=intent.intent_id,
            strategy=intent.strategy,
            purpose=OrderPurpose.ENTRY,
        )
        token = self._tokens.issue(
            order,
            now=now,
            risk_usd=sizing.risk_usd,
            stop_price=intent.stop_price,
            authorization_kind=AuthorizationKind.ENTRY,
        )
        fingerprint = _intent_fingerprint_key(intent)
        self.state.recent_fingerprints[fingerprint] = now

        if self._risk_state_store is not None:
            # Layer 2 may be followed by an independent prop-layer refusal. Keep this
            # approval process-local until final broker-side verification has passed all
            # three layers; durable exposure reservation happens in ``verify`` immediately
            # before the token is spent and the guard can submit.
            self._pending_entry_fingerprints[order.order_id] = fingerprint
            while len(self._pending_entry_fingerprints) > 256:
                self._pending_entry_fingerprints.pop(
                    next(iter(self._pending_entry_fingerprints))
                )

        return RiskDecision(
            approval=Approval(
                order=order,
                token=token,
                risk_usd=sizing.risk_usd,
                contracts=sizing.contracts,
                detail=(
                    f"{sizing.contracts} contract(s), ${sizing.risk_usd:,.2f} all-in "
                    f"risk at bounded entry {entry_price_bound:.2f}"
                    + (f", size throttled to {sizing.throttle:.0%}" if sizing.throttle < 1 else "")
                ),
            )
        )

    def _entry_blocked(
        self, now: datetime, position: Position | None, intent: OrderIntent
    ) -> Rejection | None:
        """Every reason an entry may not proceed, in order of severity."""
        cfg = self.config

        if self.kill_switch.is_active():
            return self._reject(
                now, RejectReason.KILL_SWITCH_ACTIVE,
                f"kill switch is engaged: {self.kill_switch.state().reason}", intent=intent,
            )

        durable_rejection = self._risk_state_entry_rejection(now, intent=intent)
        if durable_rejection is not None:
            return durable_rejection

        if self.state.active_entry_order_id is not None:
            return self._reject(
                now,
                RejectReason.POSITION_ALREADY_OPEN,
                "a durable entry reservation is still active",
                intent=intent,
            )

        open_positions = self._open_position_count(position)
        if open_positions >= cfg.max_open_positions:
            return self._reject(
                now, RejectReason.POSITION_ALREADY_OPEN,
                f"{open_positions} open position(s); the cap is {cfg.max_open_positions}",
                intent=intent,
            )

        if self.state.halted:
            return self._reject(
                now, RejectReason.HALTED_FOR_SESSION,
                f"halted for the session: {self.state.halt_reason}", intent=intent,
            )

        allowed, reason, detail = self.session.may_enter(now)
        if not allowed:
            return self._reject(now, reason, detail, intent=intent)

        if self.state.trades_today >= cfg.daily.max_trades_per_day:
            return self._reject(
                now, RejectReason.MAX_TRADES_PER_DAY,
                f"{self.state.trades_today} trades today; the cap is "
                f"{cfg.daily.max_trades_per_day}", intent=intent,
            )

        if self.state.daily_realized_pnl <= -abs(cfg.daily.max_daily_loss_usd):
            self._halt("daily_loss_cap_usd")
            return self._reject(
                now, RejectReason.MAX_DAILY_LOSS,
                f"daily loss ${self.state.daily_realized_pnl:,.2f} has reached the "
                f"${cfg.daily.max_daily_loss_usd:,.2f} cap", intent=intent,
            )

        if self.state.daily_r <= -abs(cfg.daily.max_daily_loss_r):
            self._halt("daily_loss_cap_r")
            return self._reject(
                now, RejectReason.MAX_DAILY_LOSS_R,
                f"daily result {self.state.daily_r:.2f}R has reached the "
                f"-{cfg.daily.max_daily_loss_r:.2f}R cap", intent=intent,
            )

        if self._floor_breached():
            if self.floor_breached_at is None:
                self.floor_breached_at = now
            if self.enforce_drawdown_floor:
                self._halt("trailing_drawdown_floor")
                return self._reject(
                    now, RejectReason.TRAILING_DRAWDOWN,
                    f"equity ${self.state.equity:,.2f} is at or below the trailing "
                    f"drawdown floor ${self._floor_level():,.2f}", intent=intent,
                )

        if self.state.cooldown_until is not None and now < self.state.cooldown_until:
            return self._reject(
                now, RejectReason.CONSECUTIVE_LOSS_COOLDOWN,
                f"cooling off after {cfg.streaks.max_consecutive_losses} consecutive "
                f"losses until {self.state.cooldown_until.isoformat()}", intent=intent,
            )

        seen = self.state.recent_fingerprints.get(_intent_fingerprint_key(intent))
        if seen is not None:
            return self._reject(
                now, RejectReason.DUPLICATE_ORDER,
                f"an identical intent was already approved at {seen.isoformat()}",
                intent=intent,
            )

        return None

    # ================================================================== exit

    def evaluate_exit(
        self, position: Position, *, reason: str = "EXIT", now: datetime | None = None
    ) -> RiskDecision:
        """Approve a closing order.

        Deliberately unconditional. The only checks are structural (there is a position,
        and the quantity is sane); no limit may refuse a flatten, because every limit in
        this engine exists in order to *cause* one.
        """
        now = now or self.clock.now()
        order = Order(
            order_id=new_id("ord"),
            timestamp=now,
            instrument=position.instrument,
            side=position.side.opposite,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            strategy=position.strategy,
            purpose=(
                OrderPurpose.FLATTEN if reason == "FLATTEN" else OrderPurpose.EXIT
            ),
        )
        token = self._tokens.issue(
            order,
            now=now,
            risk_usd=0.0,
            stop_price=position.stop_price,
            authorization_kind=AuthorizationKind.EXIT,
        )
        return RiskDecision(
            approval=Approval(order=order, token=token, risk_usd=0.0,
                              contracts=position.quantity, detail=reason)
        )

    def evaluate_snapshot_flatten(
        self,
        snapshot: AuthoritativeBrokerSnapshot,
        *,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Authorize an exact emergency close from broker truth after local-state loss.

        Restart recovery cannot safely recreate a strategy ``Position`` without original
        fill and stop provenance. It can, however, prove the signed venue quantity and
        mint a one-use FLATTEN order that exactly opposes it. The ordinary broker guard
        re-reads and re-verifies the quantity immediately before submission.
        """

        now = now or self.clock.now()
        if not isinstance(snapshot, AuthoritativeBrokerSnapshot):
            return RiskDecision(
                rejection=self._reject(
                    now,
                    RejectReason.BROKER_ERROR,
                    "emergency flatten requires an authoritative broker snapshot",
                    stage="BROKER_RECOVERY",
                )
            )
        matching = tuple(
            position
            for position in snapshot.positions
            if isinstance(position, BrokerRiskPosition)
            and position.instrument == self.instrument.symbol
            and position.quantity != 0
        )
        aggregate_quantity = sum(position.quantity for position in matching)
        if not matching or aggregate_quantity == 0:
            return RiskDecision(
                rejection=self._reject(
                    now,
                    RejectReason.INVALID_ORDER,
                    "broker snapshot has no non-zero recoverable exposure in the configured "
                    "instrument",
                    stage="BROKER_RECOVERY",
                )
            )
        order = Order(
            order_id=new_id("ord"),
            timestamp=now,
            instrument=self.instrument.symbol,
            side=Side.SELL if aggregate_quantity > 0 else Side.BUY,
            quantity=abs(aggregate_quantity),
            order_type=OrderType.MARKET,
            strategy="broker-recovery-flatten",
            purpose=OrderPurpose.FLATTEN,
        )
        rejection = self.reconcile_broker_snapshot(
            snapshot,
            now=now,
            order=order,
        )
        if rejection is not None:
            return RiskDecision(rejection=rejection)
        token = self._tokens.issue(
            order,
            now=now,
            risk_usd=0.0,
            stop_price=0.0,
            authorization_kind=AuthorizationKind.EXIT,
        )
        return RiskDecision(
            approval=Approval(
                order=order,
                token=token,
                risk_usd=0.0,
                contracts=order.quantity,
                detail="exact broker-snapshot emergency flatten",
            )
        )

    def evaluate_protective(
        self, position: Position, order: Order, *, now: datetime | None = None
    ) -> RiskDecision:
        """Authorize one exact risk-reducing STOP or TARGET order.

        Execution constructs the venue order because it owns OCO lifecycle management, but
        only this method can authorize it.  Every relationship to the open position is
        checked before a protective token is minted; arbitrary non-entry orders do not get
        an exit-shaped bypass.
        """
        now = now or self.clock.now()
        invalid = self._protective_order_error(position, order)
        if invalid is not None:
            return RiskDecision(
                rejection=self._reject(
                    now, RejectReason.INVALID_ORDER, invalid, order=order
                )
            )

        protective_stop = (
            order.stop_price
            if order.purpose is OrderPurpose.STOP
            else position.stop_price
        )
        token = self._tokens.issue(
            order,
            now=now,
            risk_usd=0.0,
            stop_price=protective_stop,
            authorization_kind=AuthorizationKind.PROTECTIVE,
        )
        return RiskDecision(
            approval=Approval(
                order=order,
                token=token,
                risk_usd=0.0,
                contracts=order.quantity,
                detail=f"validated {order.purpose.value.lower()} protection",
            )
        )

    def _protective_order_error(self, position: Position, order: Order) -> str | None:
        if order.purpose not in (OrderPurpose.STOP, OrderPurpose.TARGET):
            return f"{order.purpose.value} is not a protective order purpose"
        if order.instrument != self.instrument.symbol or order.instrument != position.instrument:
            return (
                f"protective instrument {order.instrument} does not match open "
                f"position {position.instrument}"
            )
        if order.side is not position.side.opposite:
            return (
                f"protective side {order.side.name} does not reduce the "
                f"{position.side.name} position"
            )
        if order.quantity != position.quantity:
            return (
                f"protective quantity {order.quantity} must exactly cover the "
                f"{position.quantity} open contracts"
            )
        if order.strategy != position.strategy:
            return "protective strategy identity does not match the open position"
        if not isinstance(order.oco_group, str) or not order.oco_group.strip():
            return "protective orders require a non-empty OCO group"

        if order.purpose is OrderPurpose.STOP:
            if order.order_type is not OrderType.STOP:
                return "STOP protection must use a stop-market order"
            if order.stop_price is None or order.limit_price is not None:
                return "STOP protection requires only a stop price"
            price = order.stop_price
            if position.side is Side.BUY and price < position.stop_price - 1e-9:
                return "a long protective stop may not be widened lower"
            if position.side is Side.SELL and price > position.stop_price + 1e-9:
                return "a short protective stop may not be widened higher"
        else:
            if order.order_type is not OrderType.LIMIT:
                return "TARGET protection must use a limit order"
            if order.limit_price is None or order.stop_price is not None:
                return "TARGET protection requires only a limit price"
            price = order.limit_price
            if position.target_price is None or abs(price - position.target_price) > 1e-9:
                return "target price does not match the open position's approved target"
            if position.side is Side.BUY and price <= position.entry_price:
                return "a long profit target must be above the entry price"
            if position.side is Side.SELL and price >= position.entry_price:
                return "a short profit target must be below the entry price"

        if not math.isfinite(price) or price <= 0:
            return "protective price must be finite and positive"
        if abs(self.instrument.round_to_tick(price) - price) > 1e-9:
            return (
                f"protective price {price} is not aligned to the "
                f"{self.instrument.tick_size}-point tick"
            )
        return None

    # ================================================================== broker-side gate

    def verify(
        self,
        order: Order,
        token: RiskToken,
        *,
        now: datetime | None = None,
        broker_snapshot: AuthoritativeBrokerSnapshot | None = None,
    ) -> Rejection | None:
        """Final check, called by `GuardedBroker` immediately before the order goes out.

        Returns a `Rejection` to refuse, or None to allow. This is layer 3: it re-runs the
        limits against *current* state rather than trusting the approval, so a token minted
        before a breach cannot be spent after one.
        """
        now = now or self.clock.now()

        try:
            authorization_kind = self._tokens.verify(order, token, now=now)
        except TokenError as exc:
            return self._reject(now, exc.reason, exc.detail, order=order, stage="GUARD")

        if self.kill_switch.is_active():
            # Exits stay permitted: the kill switch flattens, it does not trap.
            if authorization_kind is AuthorizationKind.ENTRY:
                return self._reject(
                    now, RejectReason.KILL_SWITCH_ACTIVE,
                    f"kill switch engaged since approval: {self.kill_switch.state().reason}",
                    order=order, stage="GUARD",
                )

        # The privilege comes from the signed authorization kind, never from the Order's
        # purpose string.  EXIT and PROTECTIVE tokens are minted only by their separately
        # validated RiskEngine paths above.
        if authorization_kind not in (
            AuthorizationKind.ENTRY,
            AuthorizationKind.EXIT,
            AuthorizationKind.PROTECTIVE,
        ):
            return self._reject(
                now,
                RejectReason.INVALID_TOKEN,
                f"unsupported authorization kind {authorization_kind!r}",
                order=order,
                stage="GUARD",
            )

        snapshot_rejection = self.reconcile_broker_snapshot(
            broker_snapshot, now=now, order=order
        )
        if snapshot_rejection is not None:
            return snapshot_rejection

        assert broker_snapshot is not None  # established by reconciliation above
        if authorization_kind in (AuthorizationKind.EXIT, AuthorizationKind.PROTECTIVE):
            reducing_error = self._risk_reducing_snapshot_error(
                order, authorization_kind, broker_snapshot
            )
            if reducing_error is not None:
                return self._reject(
                    now,
                    RejectReason.INVALID_ORDER,
                    reducing_error,
                    order=order,
                    stage="BROKER_SNAPSHOT",
                )
            self._tokens.spend(token)
            return None

        durable_rejection = self._risk_state_entry_rejection(now, order=order)
        if durable_rejection is not None:
            return durable_rejection

        # A single open broker position already consumes the personal one-position cap.
        # Any working venue order is treated conservatively as possible exposure: without
        # a venue-level reduce-only guarantee, even an apparently closing order can open a
        # reverse position if local state is stale.
        nonflat = broker_snapshot.nonflat_positions
        if nonflat:
            return self._reject(
                now,
                RejectReason.POSITION_ALREADY_OPEN,
                f"authoritative broker snapshot reports {len(nonflat)} non-flat position(s)",
                order=order,
                stage="GUARD",
                broker_positions=[position.to_dict() for position in nonflat],
            )
        working = broker_snapshot.working_orders
        if working:
            return self._reject(
                now,
                RejectReason.POSITION_ALREADY_OPEN,
                f"authoritative broker snapshot reports {len(working)} working order(s)",
                order=order,
                stage="GUARD",
                broker_working_orders=[item.to_dict() for item in working],
            )

        if self._floor_breached():
            if self.floor_breached_at is None:
                self.floor_breached_at = now
            if self.enforce_drawdown_floor:
                return self._reject(
                    now,
                    RejectReason.TRAILING_DRAWDOWN,
                    f"broker equity ${self.state.equity:,.2f} is at or below the trailing "
                    f"drawdown floor ${self._floor_level():,.2f}",
                    order=order,
                    stage="GUARD",
                )

        open_positions = self._open_position_count()
        if open_positions >= self.config.max_open_positions:
            return self._reject(
                now, RejectReason.POSITION_ALREADY_OPEN,
                f"{open_positions} tracked open position(s) since approval; the cap is "
                f"{self.config.max_open_positions}",
                order=order, stage="GUARD",
            )

        if order.quantity > self.config.per_trade.max_contracts:
            return self._reject(
                now, RejectReason.MAX_POSITION_SIZE,
                f"{order.quantity} contracts exceeds the cap of "
                f"{self.config.per_trade.max_contracts}", order=order, stage="GUARD",
            )

        if token.risk_usd > self.config.per_trade.max_risk_per_trade_usd + 1e-9:
            return self._reject(
                now, RejectReason.MAX_RISK_PER_TRADE,
                f"approved risk ${token.risk_usd:,.2f} exceeds the per-trade cap "
                f"${self.config.per_trade.max_risk_per_trade_usd:,.2f}",
                order=order, stage="GUARD",
            )

        # Approval-time sizing is not authority after the account mark changes. Reapply
        # both the equity percentage and the drawdown-cushion throttle to the fresh broker
        # equity imported above; checking only the absolute dollar ceiling would allow a
        # stale full-size token to execute immediately beside the personal floor.
        cushion = self.cushion_fraction() if self.enforce_drawdown_floor else 1.0
        current_budget, current_throttle = self.sizer.risk_budget(
            equity=self.state.equity,
            cushion_fraction=cushion,
            cushion_threshold=(
                self.config.drawdown.size_reduction_cushion_pct / 100.0
            ),
        )
        if token.risk_usd > current_budget + 1e-9:
            return self._reject(
                now,
                RejectReason.MAX_RISK_PER_TRADE,
                f"approved risk ${token.risk_usd:,.2f} exceeds the freshly recomputed "
                f"budget ${current_budget:,.2f}",
                order=order,
                stage="GUARD",
                broker_equity=self.state.equity,
                current_risk_budget_usd=current_budget,
                current_size_throttle=current_throttle,
            )

        if self.state.last_trade_was_loss and (
            order.quantity > self.state.last_trade_quantity
            or token.risk_usd > self.state.last_trade_approved_risk_usd + 1e-9
        ):
            return self._reject(
                now,
                RejectReason.MAX_RISK_PER_TRADE,
                "post-loss quantity or approved-risk ceiling changed since approval",
                order=order,
                stage="GUARD",
                previous_contracts=self.state.last_trade_quantity,
                previous_risk_usd=self.state.last_trade_approved_risk_usd,
            )

        if self.state.halted:
            return self._reject(
                now, RejectReason.HALTED_FOR_SESSION,
                f"halted since approval: {self.state.halt_reason}", order=order, stage="GUARD",
            )

        if self.state.daily_realized_pnl <= -abs(self.config.daily.max_daily_loss_usd):
            return self._reject(
                now, RejectReason.MAX_DAILY_LOSS,
                "the daily loss cap was reached between approval and submission",
                order=order, stage="GUARD",
            )

        if self.state.trades_today >= self.config.daily.max_trades_per_day:
            return self._reject(
                now, RejectReason.MAX_TRADES_PER_DAY,
                "the daily trade cap was reached between approval and submission",
                order=order, stage="GUARD",
            )

        allowed, reason, detail = self.session.may_enter(now)
        if not allowed:
            return self._reject(now, reason, detail, order=order, stage="GUARD")

        durable_reservation = self._reserve_verified_entry(order, token, now=now)
        if durable_reservation is not None:
            return durable_reservation

        self._tokens.spend(token)
        return None

    def _reserve_verified_entry(
        self,
        order: Order,
        token: RiskToken,
        *,
        now: datetime,
    ) -> Rejection | None:
        """Persist the exact entry only after every final pre-submit gate passed."""

        if self._risk_state_store is None:
            return None
        fingerprint = self._pending_entry_fingerprints.get(order.order_id)
        if fingerprint is None:
            return self._reject(
                now,
                RejectReason.INVALID_TOKEN,
                "entry approval has no process-local durable reservation provenance",
                order=order,
                stage="RISK_STATE",
            )
        if self.state.active_entry_order_id is not None:
            return self._reject(
                now,
                RejectReason.POSITION_ALREADY_OPEN,
                "another durable entry reservation is already active",
                order=order,
                stage="RISK_STATE",
            )
        self.state.active_entry_order_id = order.order_id
        self.state.active_entry_intent_id = order.intent_id
        self.state.active_entry_fingerprint = fingerprint
        self.state.active_entry_risk_usd = token.risk_usd
        self.state.active_approved_quantity = order.quantity
        self.state.entry_ever_filled = False
        if not self._persist_risk_state(now):
            return self._risk_state_entry_rejection(now, order=order)
        self._pending_entry_fingerprints.pop(order.order_id, None)
        return None

    @staticmethod
    def _risk_reducing_snapshot_error(
        order: Order,
        authorization_kind: AuthorizationKind,
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> str | None:
        """Prove the signed order exactly closes/covers current venue exposure."""
        if authorization_kind is AuthorizationKind.EXIT:
            if order.purpose not in (OrderPurpose.EXIT, OrderPurpose.FLATTEN):
                return "exit authorization is not bound to an EXIT or FLATTEN order"
        elif authorization_kind is AuthorizationKind.PROTECTIVE:
            if order.purpose not in (OrderPurpose.STOP, OrderPurpose.TARGET):
                return "protective authorization is not bound to a STOP or TARGET order"
        else:
            return "entry authorization cannot use the exposure-reduction path"

        # Unexpected exposure in a different instrument remains an entry blocker and an
        # operational alarm, but it must not trap a targeted emergency close. Aggregate
        # venue rows only for this exact instrument, then require full opposing coverage.
        matching = tuple(
            position
            for position in snapshot.nonflat_positions
            if position.instrument == order.instrument
        )
        aggregate_quantity = sum(position.quantity for position in matching)
        if not matching or aggregate_quantity == 0:
            return (
                "risk-reducing order has no authoritative non-flat broker exposure "
                "in its instrument"
            )
        expected_side = Side.SELL if aggregate_quantity > 0 else Side.BUY
        if order.side is not expected_side:
            return "risk-reducing order side does not oppose broker exposure"
        if order.quantity != abs(aggregate_quantity):
            return "risk-reducing order quantity must exactly cover broker exposure"
        return None

    def reconcile_broker_snapshot(
        self,
        snapshot: AuthoritativeBrokerSnapshot | None,
        *,
        now: datetime | None = None,
        order: Order | None = None,
    ) -> Rejection | None:
        """Validate and bind one authoritative adapter read, then reconcile equity.

        This seam intentionally does not manufacture a local ``Position`` from a broker
        quantity; recovery needs full fill/stop provenance for that.  Entry verification
        consumes the position/order collections directly and refuses while either could
        represent exposure.
        """
        now = now or self.clock.now()
        if not isinstance(snapshot, AuthoritativeBrokerSnapshot):
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                "entry verification requires a fresh authoritative broker snapshot",
                order=order,
                stage="BROKER_SNAPSHOT",
            )
        if not _aware_datetime(now):
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                "broker snapshot verification time must be timezone-aware",
                order=order,
                stage="BROKER_SNAPSHOT",
            )
        if not _aware_datetime(snapshot.read_started_at) or not _aware_datetime(
            snapshot.captured_at
        ):
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                "broker snapshot timestamps must be timezone-aware",
                order=order,
                stage="BROKER_SNAPSHOT",
            )

        read_duration = snapshot.captured_at - snapshot.read_started_at
        age = now - snapshot.captured_at
        if read_duration < timedelta(0) or read_duration > _MAX_BROKER_SNAPSHOT_READ_DURATION:
            return self._reject(
                now,
                RejectReason.STALE_MARKET_DATA,
                "broker state read was incomplete or exceeded the two-second acquisition cap",
                order=order,
                stage="BROKER_SNAPSHOT",
                read_duration_seconds=read_duration.total_seconds(),
            )
        if age < timedelta(0) or age > _MAX_BROKER_SNAPSHOT_AGE:
            return self._reject(
                now,
                RejectReason.STALE_MARKET_DATA,
                "broker snapshot is future-dated or older than one second",
                order=order,
                stage="BROKER_SNAPSHOT",
                snapshot_age_seconds=age.total_seconds(),
            )

        structural_error = self._broker_snapshot_structural_error(snapshot)
        if structural_error is not None:
            return self._reject(
                now,
                RejectReason.BROKER_ERROR,
                structural_error,
                order=order,
                stage="BROKER_SNAPSHOT",
            )

        account_id = snapshot.account.account_id.strip()
        broker_name = snapshot.broker_name.strip()
        broker_route = snapshot.execution_route.strip().casefold()
        if self._broker_account_id is not None and account_id != self._broker_account_id:
            return self._reject(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "broker account does not match the configured or previously bound identity",
                order=order,
                stage="BROKER_SNAPSHOT",
            )
        if self._broker_name is not None and broker_name != self._broker_name:
            return self._reject(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "broker route does not match the configured or previously bound identity",
                order=order,
                stage="BROKER_SNAPSHOT",
            )
        if (
            self._broker_is_paper is not None
            and snapshot.broker_is_paper is not self._broker_is_paper
        ):
            return self._reject(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "broker paper/live mode does not match the configured or previously bound identity",
                order=order,
                stage="BROKER_SNAPSHOT",
            )
        if self._broker_route is not None and broker_route != self._broker_route:
            return self._reject(
                now,
                RejectReason.LIVE_TRADING_DISABLED,
                "broker execution route does not match the configured or previously bound route",
                order=order,
                stage="BROKER_SNAPSHOT",
            )

        self._broker_account_id = account_id
        self._broker_name = broker_name
        self._broker_is_paper = snapshot.broker_is_paper
        self._broker_route = broker_route
        self.state.equity = snapshot.account.equity
        self.state.peak_equity = max(self.state.peak_equity, snapshot.account.equity)
        recovery_can_clear = False
        if self._risk_state_store is not None:
            nonflat = snapshot.nonflat_positions
            working = snapshot.working_orders
            if self._risk_state_recovery_required:
                if nonflat or working:
                    if nonflat and self.state.open_position_id is None:
                        self.state.open_position_id = (
                            f"BROKER-RECOVERY:{self.instrument.symbol}"
                        )
                    self._risk_state_recovery_reason = (
                        "fresh broker state still contains exposure or working orders"
                    )
                elif self.state.open_position_id is not None or self.state.entry_ever_filled:
                    self._risk_state_recovery_reason = (
                        "persisted filled exposure has no reconciled terminal trade outcome"
                    )
                else:
                    if self.state.active_entry_order_id is not None:
                        self._clear_active_entry_state()
                    session_day = now.astimezone(self._session_timezone).date()
                    if self.state.session_date != session_day and not self.roll_session(
                        now,
                        broker_flat_confirmed=True,
                    ):
                        return self._risk_state_entry_rejection(now, order=order)
                    recovery_can_clear = True

            persisted = self._persist_risk_state(
                now,
                exposure_possible=bool(nonflat),
            )
            if persisted and recovery_can_clear:
                self._risk_state_recovery_required = False
                self._risk_state_recovery_reason = ""
            if not persisted and (
                order is None or order.purpose is OrderPurpose.ENTRY
            ):
                return self._risk_state_entry_rejection(now, order=order)
        return None

    @staticmethod
    def _broker_snapshot_structural_error(
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> str | None:
        if not isinstance(snapshot.broker_name, str) or not snapshot.broker_name.strip():
            return "broker snapshot source name is missing"
        if not isinstance(snapshot.broker_is_paper, bool):
            return "broker snapshot paper/live flag is invalid"
        if (
            not isinstance(snapshot.execution_route, str)
            or not snapshot.execution_route.strip()
        ):
            return "broker snapshot execution route is missing"
        account = snapshot.account
        if not isinstance(account, BrokerRiskAccount):
            return "broker snapshot account payload is invalid"
        if not isinstance(account.account_id, str) or not account.account_id.strip():
            return "broker account id is missing"
        if not isinstance(account.is_paper, bool) or account.is_paper is not snapshot.broker_is_paper:
            return "broker account paper/live identity does not match the adapter"
        if not isinstance(account.currency, str) or not account.currency.strip():
            return "broker account currency is missing"
        for name in ("equity", "cash", "realized_pnl", "unrealized_pnl"):
            value = getattr(account, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return f"broker account {name} must be finite"
        if not isinstance(snapshot.positions, tuple) or not all(
            isinstance(position, BrokerRiskPosition) for position in snapshot.positions
        ):
            return "broker positions payload must be an immutable typed tuple"
        for position in snapshot.positions:
            if not isinstance(position.instrument, str) or not position.instrument.strip():
                return "broker position instrument is missing"
            if (
                isinstance(position.quantity, bool)
                or not isinstance(position.quantity, int)
                or not isinstance(position.average_price, (int, float))
                or isinstance(position.average_price, bool)
                or not math.isfinite(position.average_price)
            ):
                return "broker position quantity or price is invalid"
        if not isinstance(snapshot.orders, tuple) or not all(
            isinstance(item, BrokerRiskOrder) for item in snapshot.orders
        ):
            return "broker orders payload must be an immutable typed tuple"
        seen: set[tuple[str, str]] = set()
        for item in snapshot.orders:
            if not isinstance(item.status, OrderStatus):
                return "broker order status is invalid"
            if not item.order_id and not item.broker_order_id:
                return "broker order is missing both local and venue identifiers"
            if (
                isinstance(item.filled_quantity, bool)
                or not isinstance(item.filled_quantity, int)
                or item.filled_quantity < 0
                or isinstance(item.average_fill_price, bool)
                or not isinstance(item.average_fill_price, (int, float))
                or not math.isfinite(item.average_fill_price)
            ):
                return "broker order fill quantity or average price is invalid"
            identity = (item.order_id, item.broker_order_id)
            if identity in seen:
                return "broker snapshot contains duplicate order identities"
            seen.add(identity)
        return None

    # ================================================================== lifecycle

    def on_entry_fill_observed(
        self,
        order_id: str,
        *,
        now: datetime,
    ) -> bool:
        """Durably consume the daily quota on the first broker-observed entry fill."""
        if self._risk_state_store is None:
            return True
        if order_id in self.state.consumed_entry_order_ids:
            return True
        if self.state.active_entry_order_id != order_id:
            self._risk_state_store_error = (
                "broker reported an entry fill that is not bound to durable approval state"
            )
            self.kill_switch.trip("unbound entry fill observed")
            return False
        self.state.entry_ever_filled = True
        self.state.trades_today += 1
        self.state.consumed_entry_order_ids = _append_bounded_id(
            self.state.consumed_entry_order_ids,
            order_id,
        )
        return self._persist_risk_state(now, exposure_possible=True)

    def on_entry_terminal_unfilled(self, order_id: str, *, now: datetime) -> bool:
        """Release a durable approval only after the broker reports no fill."""
        if self._risk_state_store is None:
            return True
        if self.state.active_entry_order_id != order_id:
            return True
        if self.state.entry_ever_filled or order_id in self.state.consumed_entry_order_ids:
            return False
        self._clear_active_entry_state()
        return self._persist_risk_state(now)

    def roll_session(
        self,
        now: datetime,
        *,
        broker_flat_confirmed: bool = False,
    ) -> bool:
        """Reset per-session counters when the date changes. Returns True if it rolled."""
        if not _aware_datetime(now):
            raise ValueError("session rollover time must be timezone-aware")
        day = now.astimezone(self._session_timezone).date()
        if self.state.session_date == day:
            return False
        if self._risk_state_store is not None and not broker_flat_confirmed:
            self._risk_state_recovery_required = True
            self._risk_state_recovery_reason = (
                "session rollover requires a fresh flat broker reconciliation"
            )
            return False
        if self._risk_state_store is not None and (
            self.state.open_position_id is not None or self.state.entry_ever_filled
        ):
            self._risk_state_recovery_required = True
            self._risk_state_recovery_reason = (
                "unresolved prior-session exposure blocks risk-state rollover"
            )
            return False
        previous_state = deepcopy(self.state)
        previous_pending_fingerprints = dict(self._pending_entry_fingerprints)
        self.state.session_date = day
        self.state.session_start_equity = self.state.equity
        self.state.daily_realized_pnl = 0.0
        self.state.daily_r = 0.0
        self.state.trades_today = 0
        self.state.consecutive_losses = 0
        self.state.cooldown_until = None
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.recent_fingerprints.clear()
        self.state.consumed_entry_order_ids = ()
        self._pending_entry_fingerprints.clear()
        if not self._persist_risk_state(now):
            # A rollover is permission-expanding: it restores quota and daily risk. Do not
            # publish it in memory unless the same transition is durable.
            self.state = previous_state
            self._pending_entry_fingerprints = previous_pending_fingerprints
            return False
        return True

    def on_position_opened(self, position: Position) -> None:
        same_position_update = self.state.open_position_id == position.position_id
        if self._risk_state_store is not None:
            lifecycle_error: str | None = None
            if position.instrument != self.instrument.symbol:
                lifecycle_error = "position-open event uses the wrong instrument"
            elif self.state.active_entry_order_id is None:
                lifecycle_error = "position opened without a durable active entry"
            elif position.quantity > self.state.active_approved_quantity:
                lifecycle_error = "opened position exceeds the durably approved quantity"
            elif (
                self.state.active_approved_quantity > 0
                and (
                    abs(position.entry_price - position.initial_stop)
                    * self.instrument.multiplier
                    + self._fixed_risk_reserve_per_contract_usd()
                )
                > (
                    self.state.active_entry_risk_usd
                    / self.state.active_approved_quantity
                )
                + 1e-9
            ):
                lifecycle_error = (
                    "actual entry fill exceeds the durably approved all-in "
                    "per-contract risk envelope"
                )
            elif (
                self.state.open_position_id is not None
                and self.state.open_position_id != position.position_id
            ):
                lifecycle_error = "a different position is already bound to personal-risk state"
            if lifecycle_error is not None:
                self._risk_state_store_error = lifecycle_error
                self.kill_switch.trip(lifecycle_error)
                return

        self.state.open_position_id = position.position_id
        if self._risk_state_store is None:
            # Partial entry fills update the same Position repeatedly. The first fill
            # consumes the quota; later cumulative exposure checks must not look like
            # additional trades merely because the risk envelope is revalidated.
            if not same_position_update:
                self.state.trades_today += 1
        elif not self.state.entry_ever_filled:
            active_order_id = self.state.active_entry_order_id
            if active_order_id is None:
                self._risk_state_store_error = (
                    "position opened without a durable active entry"
                )
                self.kill_switch.trip("position opened without durable entry state")
                return
            self.on_entry_fill_observed(active_order_id, now=position.entry_time)
        self._persist_risk_state(position.entry_time, exposure_possible=True)

    def on_trade_closed(self, trade: Trade) -> None:
        if (
            self._risk_state_store is not None
            and trade.trade_id in self.state.applied_trade_ids
        ):
            return
        approved_risk = self.state.active_entry_risk_usd
        approved_quantity = self.state.active_approved_quantity
        planned_loss_envelope = 0.0
        if self._risk_state_store is not None:
            lifecycle_error: str | None = None
            if trade.instrument != self.instrument.symbol:
                lifecycle_error = "trade-close event uses the wrong instrument"
            elif self.state.active_entry_order_id is None:
                lifecycle_error = "trade closed without a durable active entry"
            elif (
                not self.state.entry_ever_filled
                or self.state.active_entry_order_id
                not in self.state.consumed_entry_order_ids
            ):
                lifecycle_error = "trade closed before an entry fill was durably consumed"
            elif trade.quantity > self.state.active_approved_quantity:
                lifecycle_error = "closed trade exceeds the durably approved quantity"
            elif (
                self.state.open_position_id is not None
                and trade.position_id != self.state.open_position_id
            ):
                lifecycle_error = "trade close does not match the durably tracked position"
            elif (
                self.state.last_trade_closed_at is not None
                and trade.exit_time <= self.state.last_trade_closed_at
            ):
                lifecycle_error = "trade close is not newer than the durable trade watermark"
            if lifecycle_error is not None:
                self._risk_state_store_error = lifecycle_error
                self.kill_switch.trip(lifecycle_error)
                return

            # A partially filled entry must anchor the next post-loss ceiling to contracts
            # that actually became exposure, not the larger requested quantity.  Include
            # any adverse entry gap by retaining the greater of filled pro-rata approval
            # risk and actual entry-to-stop risk.
            pro_rata_approved_risk = (
                approved_risk * trade.quantity / approved_quantity
                if approved_risk > 0 and approved_quantity > 0
                else 0.0
            )
            planned_loss_envelope = pro_rata_approved_risk
            actual_initial_risk = (
                abs(trade.entry_price - trade.initial_stop)
                * trade.quantity
                * self.instrument.multiplier
            )
            approved_risk = max(1e-9, pro_rata_approved_risk, actual_initial_risk)
            approved_quantity = trade.quantity
        else:
            if approved_risk <= 0:
                approved_risk = max(
                    1e-9,
                    abs(trade.entry_price - trade.initial_stop)
                    * trade.quantity
                    * self.instrument.multiplier,
                )
            if approved_quantity <= 0:
                approved_quantity = trade.quantity
        self.state.open_position_id = None
        if self._risk_state_store is None:
            # Memory-only research uses trade deltas. Durable/paper operation instead
            # treats broker account equity as an absolute authority and refreshes it after
            # terminal fills; adding a delta to a marked snapshot double-counts P&L.
            self.state.equity += trade.net_pnl_usd
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.daily_realized_pnl += trade.net_pnl_usd
        self.state.daily_r += trade.r_multiple

        if (
            planned_loss_envelope > 0
            and -trade.net_pnl_usd > planned_loss_envelope + 1e-9
        ):
            # Stop-market protection is not a loss guarantee. If a real gap overwhelms
            # the configured reserve, preserve the economic result but latch the account
            # before any next entry and make the breach explicit in durable state.
            self.kill_switch.trip(
                "realized trade loss exceeded its approved all-in risk envelope"
            )
            self._halt("planned_risk_envelope_breach")

        if trade.net_pnl_usd <= 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.state.consecutive_losses >= self.config.streaks.max_consecutive_losses:
            self.state.consecutive_losses = 0
            self.state.cooldown_until = trade.exit_time + timedelta(
                minutes=self.config.streaks.cooldown_minutes
            )

        if self.state.daily_realized_pnl <= -abs(self.config.daily.max_daily_loss_usd):
            self._halt("daily_loss_cap_usd")
        elif self.state.daily_r <= -abs(self.config.daily.max_daily_loss_r):
            self._halt("daily_loss_cap_r")

        if self._floor_breached():
            if self.floor_breached_at is None:
                self.floor_breached_at = trade.exit_time
            if self.enforce_drawdown_floor:
                self._halt("trailing_drawdown_floor")

        self.state.applied_trade_ids = _append_bounded_id(
            self.state.applied_trade_ids,
            trade.trade_id,
        )
        self.state.last_trade_id = trade.trade_id
        self.state.last_trade_closed_at = trade.exit_time
        self.state.last_trade_net_pnl_usd = trade.net_pnl_usd
        self.state.last_trade_contracts = trade.quantity
        self.state.last_trade_was_loss = trade.net_pnl_usd <= 0
        self.state.last_trade_approved_risk_usd = approved_risk
        self.state.last_trade_quantity = approved_quantity
        self._clear_active_entry_state()
        if self._risk_state_store is not None:
            self._persist_risk_state(trade.exit_time)

    def _clear_active_entry_state(self) -> None:
        self.state.active_entry_order_id = None
        self.state.active_entry_intent_id = None
        self.state.active_entry_fingerprint = None
        self.state.active_entry_risk_usd = 0.0
        self.state.active_approved_quantity = 0
        self.state.entry_ever_filled = False

    def forced_exit_reason(
        self, position: Position, mark: float, now: datetime
    ) -> tuple[RejectReason, str] | None:
        """Whether an *open* position must be closed right now, for a risk reason.

        Marked equity is used rather than realised: a limit that only acts once a loss has
        been booked cannot prevent the loss it exists to prevent.
        """
        if self.kill_switch.is_active():
            return (
                RejectReason.KILL_SWITCH_ACTIVE,
                f"kill switch engaged: {self.kill_switch.state().reason}",
            )

        if self._risk_state_store is not None and (
            self._risk_state_store_error is not None
            or self._risk_state_recovery_required
        ):
            return (
                RejectReason.BROKER_ERROR,
                "durable personal-risk state is unavailable or unreconciled while "
                "exposure is open; fail-safe flatten required",
            )
        if self.session.must_flatten(now):
            return (
                RejectReason.OUTSIDE_TRADING_HOURS,
                f"flatten time {self.session.window.flatten_at} reached; no overnight positions",
            )

        unrealized = position.unrealized_usd(mark, self.instrument.multiplier)
        marked_daily = self.state.daily_realized_pnl + unrealized
        if marked_daily <= -abs(self.config.daily.max_daily_loss_usd):
            return (
                RejectReason.MAX_DAILY_LOSS,
                f"marked daily P&L ${marked_daily:,.2f} has reached the "
                f"${self.config.daily.max_daily_loss_usd:,.2f} cap",
            )

        marked_equity = self.state.equity + unrealized
        floor = self._floor_level()
        if self.enforce_drawdown_floor and marked_equity <= floor:
            return (
                RejectReason.TRAILING_DRAWDOWN,
                f"marked equity ${marked_equity:,.2f} is at the trailing floor ${floor:,.2f}",
            )
        return None

    # ================================================================== helpers

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self._persist_risk_state(self.clock.now())

    def _floor_level(self) -> float:
        return self.state.peak_equity * (
            1 - self.config.drawdown.trailing_drawdown_pct / 100.0
        )

    def _floor_breached(self) -> bool:
        return self.state.equity <= self._floor_level()

    def _open_position_count(self, position: Position | None = None) -> int:
        """Current v1 exposure count from either local execution or tracked risk state.

        The execution engine normally supplies its position, but the risk engine must not
        become bypassable merely because a caller omits that argument. V1 supports one
        position, so the two sources describe the same possible exposure rather than two
        independently countable positions.
        """
        return int(position is not None or self.state.open_position_id is not None)

    def cushion_fraction(self) -> float:
        """Fraction of the trailing-drawdown allowance still unused, in [0, 1]."""
        allowance = self.state.peak_equity * (
            self.config.drawdown.trailing_drawdown_pct / 100.0
        )
        if allowance <= 0:
            return 1.0
        used = self.state.peak_equity - self.state.equity
        return max(0.0, min(1.0, 1.0 - used / allowance))

    def _reject(
        self,
        now: datetime,
        reason: RejectReason,
        detail: str,
        *,
        intent: OrderIntent | None = None,
        order: Order | None = None,
        stage: str = "RISK",
        **context,
    ) -> Rejection:
        return Rejection(
            timestamp=now,
            reason=reason,
            detail=detail,
            stage=stage,
            instrument=self.instrument.symbol,
            strategy=(intent.strategy if intent else (order.strategy if order else "")),
            intent_id=intent.intent_id if intent else (order.intent_id if order else None),
            order_id=order.order_id if order else None,
            context={**context, **self.state.snapshot()},
        )

    def limits_snapshot(self) -> dict:
        """What the dashboard shows: each limit, its value, and how much is used."""
        cfg = self.config
        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "daily_pnl": self.state.daily_realized_pnl,
            "daily_loss_limit": cfg.daily.max_daily_loss_usd,
            "daily_loss_used_pct": min(
                100.0,
                100.0 * max(0.0, -self.state.daily_realized_pnl) / cfg.daily.max_daily_loss_usd,
            ),
            "daily_r": self.state.daily_r,
            "daily_r_limit": cfg.daily.max_daily_loss_r,
            "trades_today": self.state.trades_today,
            "max_trades_per_day": cfg.daily.max_trades_per_day,
            "max_open_positions": cfg.max_open_positions,
            "consecutive_losses": self.state.consecutive_losses,
            "max_consecutive_losses": cfg.streaks.max_consecutive_losses,
            "max_contracts": cfg.per_trade.max_contracts,
            "max_risk_per_trade_usd": cfg.per_trade.max_risk_per_trade_usd,
            "drawdown_floor": self._floor_level(),
            "cushion_pct": 100.0 * self.cushion_fraction(),
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "kill_switch": self.kill_switch.state().describe(),
            "cooldown_until": (
                self.state.cooldown_until.isoformat() if self.state.cooldown_until else None
            ),
            "session_window": self.session.describe(),
        }
