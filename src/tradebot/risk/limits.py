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

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ..config import RiskConfig, SessionConfig
from ..core.clock import Clock, SystemClock
from ..core.models import Order, OrderIntent, Position, Rejection, Trade, new_id
from ..core.types import OrderPurpose, OrderStatus, OrderType, RejectReason, Side
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
from .tokens import AuthorizationKind, RiskToken, TokenError, TokenMint


_MAX_BROKER_SNAPSHOT_AGE = timedelta(seconds=1)
_MAX_BROKER_SNAPSHOT_READ_DURATION = timedelta(seconds=2)


def _aware_datetime(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


@dataclass(slots=True)
class RiskState:
    """Mutable bookkeeping, kept separate from the engine so the runner can persist and
    restore it across a restart without re-deriving it from the trade log."""

    equity: float
    peak_equity: float
    session_date: date | None = None
    session_start_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_r: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    open_position_id: str | None = None
    recent_fingerprints: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "session_date": str(self.session_date) if self.session_date else None,
            "session_start_equity": self.session_start_equity,
            "daily_realized_pnl": self.daily_realized_pnl,
            "daily_r": self.daily_r,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "trades_today": self.trades_today,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }


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
    ) -> None:
        for label, value in (
            ("expected_broker_account_id", expected_broker_account_id),
            ("expected_broker_name", expected_broker_name),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{label} must be null or a non-empty string")
        if expected_broker_is_paper is not None and not isinstance(
            expected_broker_is_paper, bool
        ):
            raise ValueError("expected_broker_is_paper must be null or boolean")
        self.config = config
        self.instrument = instrument
        self.clock = clock or SystemClock()
        self.session = SessionGuard(session_config)
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
        self.state = RiskState(
            equity=equity, peak_equity=equity, session_start_equity=equity
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

    # ================================================================== entry

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

        stop_distance = abs(intent.reference_price - intent.stop_price)
        # With the floor disabled for research, the cushion throttle must be disabled too.
        # Peak equity never falls, so a fully-consumed cushion multiplies the risk budget
        # by zero and vetoes every subsequent trade -- a second, silent path to exactly the
        # deadlock that turning the floor off was meant to remove, and one that truncates a
        # whole-sample run to a handful of trades without saying why.
        cushion = self.cushion_fraction() if self.enforce_drawdown_floor else 1.0
        sizing = self.sizer.size(
            equity=self.state.equity,
            stop_distance_points=stop_distance,
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
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            stop_price=None if intent.order_type is OrderType.MARKET else intent.limit_price,
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
        self.state.recent_fingerprints[intent.fingerprint()] = now

        return RiskDecision(
            approval=Approval(
                order=order,
                token=token,
                risk_usd=sizing.risk_usd,
                contracts=sizing.contracts,
                detail=(
                    f"{sizing.contracts} contract(s), ${sizing.risk_usd:,.2f} at risk"
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

        seen = self.state.recent_fingerprints.get(intent.fingerprint())
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

        self._tokens.spend(token)
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

        self._broker_account_id = account_id
        self._broker_name = broker_name
        self._broker_is_paper = snapshot.broker_is_paper
        self.state.equity = snapshot.account.equity
        self.state.peak_equity = max(self.state.peak_equity, snapshot.account.equity)
        return None

    @staticmethod
    def _broker_snapshot_structural_error(
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> str | None:
        if not isinstance(snapshot.broker_name, str) or not snapshot.broker_name.strip():
            return "broker snapshot source name is missing"
        if not isinstance(snapshot.broker_is_paper, bool):
            return "broker snapshot paper/live flag is invalid"
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

    def roll_session(self, now: datetime) -> bool:
        """Reset per-session counters when the date changes. Returns True if it rolled."""
        day = now.date()
        if self.state.session_date == day:
            return False
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
        return True

    def on_position_opened(self, position: Position) -> None:
        self.state.open_position_id = position.position_id
        self.state.trades_today += 1

    def on_trade_closed(self, trade: Trade) -> None:
        self.state.open_position_id = None
        self.state.equity += trade.net_pnl_usd
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.daily_realized_pnl += trade.net_pnl_usd
        self.state.daily_r += trade.r_multiple

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
