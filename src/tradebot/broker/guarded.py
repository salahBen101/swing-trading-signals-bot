"""The guard between the execution engine and any broker.

This is layer 3 of the non-bypass property (PROJECT_SPEC §3.1), and the one the
requirement states most directly:

> *"The broker execution layer must reject any order violating these limits regardless of
> what the strategy requests."*

Three mechanisms, all structural rather than procedural:

1. **The signature.** `place_order(order, token)` takes a required `RiskToken`. There is no
   default and no overload without one, so an unapproved order cannot be expressed — it is
   a `TypeError` at the call site, not a policy violation discovered at runtime.

2. **Independent re-validation.** Holding a valid token is necessary but not sufficient.
   The guard calls back into the risk engine, which re-runs the live limits *now*. An
   approval minted before a limit was breached does not execute after it.

3. **Pending-entry reservation.** Entry submission is serialized and reserved before the
   adapter call. When a reservation store is configured, that state is atomically durable
   across restart and can only be deleted after a fresh broker snapshot proves the order
   terminal and the account flat. Persistence faults block entries without blocking exits
   or protective orders.

The wrapped adapter is stored privately and the guard exposes only read-only pass-throughs
plus narrowly typed order operations. Entry cancellation is allowed only for an entry the
same guard submitted. Protective retirement requires fresh proof that the account is flat
or that an accepted, working, no-weaker replacement covers the current exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from threading import RLock

from ..core.models import Fill, Order, Rejection
from ..core.types import OrderPurpose, OrderStatus, OrderType, RejectReason, Side
from ..deployment.stages import DeploymentStage
from ..risk.broker_state import (
    AuthoritativeBrokerSnapshot,
    BrokerRiskAccount,
    BrokerRiskOrder,
    BrokerRiskPosition,
)
from ..risk.limits import RiskEngine
from ..risk.reservations import (
    PendingEntryReservation,
    PendingEntryReservationStore,
    PendingEntryState,
    ReservationStoreError,
)
from ..risk.tokens import RiskToken
from .base import (
    BrokerAccount,
    BrokerAdapter,
    BrokerCapabilities,
    BrokerEvent,
    BrokerOrderState,
    BrokerPosition,
    EventKind,
    LiveTradingDisabled,
    OrderAck,
    OrderRejected,
)


class RiskViolation(OrderRejected):
    """The guard refused the order. Carries the machine-readable rejection."""

    def __init__(self, rejection: Rejection) -> None:
        super().__init__(f"{rejection.reason.value}: {rejection.detail}")
        self.rejection = rejection


@dataclass(frozen=True, slots=True)
class GuardResult:
    ack: OrderAck | None
    rejection: Rejection | None

    @property
    def accepted(self) -> bool:
        return self.ack is not None


@dataclass(frozen=True, slots=True)
class CancelResult:
    """Result of a guarded cancellation request.

    ``accepted`` means the guard allowed the request (or the known order was already
    terminal).  It never means the venue has cancelled the order; only a later terminal
    broker event establishes that fact.
    """

    request_sent: bool
    rejection: Rejection | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection is None


# States where a cancellation request would be pointless: the entry has already
# reached a terminal outcome, or a cancellation is already in flight.
_PENDING_ENTRY_CANCEL_COMPLETE = frozenset({
    PendingEntryState.CANCEL_REQUESTED,
    PendingEntryState.FILLED,
})

# Broker timestamps cross a network/process trust boundary.  The authoritative snapshot
# gate already permits at most one second of age/skew; acknowledgements and streaming
# events use the same narrow tolerance.  Unknown lifecycle events are additionally
# bounded to one complete trading day so a replayed ancient message cannot mutate the
# current order ledger.
_MAX_BROKER_MESSAGE_CLOCK_SKEW = timedelta(seconds=1)
_MAX_UNBOUND_BROKER_EVENT_AGE = timedelta(days=1)

_ORDER_EVENT_KINDS = frozenset({
    EventKind.ACCEPTED,
    EventKind.REJECTED,
    EventKind.PARTIAL_FILL,
    EventKind.FILL,
    EventKind.CANCEL_REQUESTED,
    EventKind.CANCELLED,
})
_FILL_EVENT_KINDS = frozenset({EventKind.PARTIAL_FILL, EventKind.FILL})
_TERMINAL_FILL_EVENT_KINDS = frozenset({EventKind.CANCELLED, EventKind.REJECTED})
_ACCEPTABLE_SUBMISSION_ACK_STATUSES = frozenset({
    OrderStatus.SUBMITTED,
    OrderStatus.ACCEPTED,
})
_TERMINAL_NO_FILL_ACK_STATUSES = frozenset({
    OrderStatus.REJECTED,
    OrderStatus.CANCELLED,
    OrderStatus.EXPIRED,
})
_FILL_PROGRESS_ACK_STATUSES = frozenset({
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.FILLED,
})


class GuardedBroker:
    """Wraps a `BrokerAdapter`. The only thing the execution engine is ever handed."""

    def __init__(
        self,
        adapter: BrokerAdapter,
        risk: RiskEngine,
        *,
        reservation_store: PendingEntryReservationStore | None = None,
        deployment_stage: DeploymentStage = DeploymentStage.BACKTEST,
    ) -> None:
        self._deployment_stage = DeploymentStage(deployment_stage)
        execution_route = getattr(adapter, "execution_route", None)
        if not isinstance(execution_route, str) or not execution_route.strip():
            raise LiveTradingDisabled(
                "broker adapter does not declare a canonical execution route"
            )
        capabilities = getattr(adapter, "capabilities", None)
        if not isinstance(capabilities, BrokerCapabilities):
            raise LiveTradingDisabled(
                "broker adapter does not declare its execution-safety capabilities"
            )
        if self._deployment_stage >= DeploymentStage.PAPER:
            if reservation_store is None or not bool(
                getattr(reservation_store, "is_durable", False)
            ):
                raise LiveTradingDisabled(
                    "Stage 2+ requires a certified durable pending-entry reservation store"
                )
            if not bool(getattr(risk, "durable_state_certified", False)):
                raise LiveTradingDisabled(
                    "Stage 2+ requires a certified durable personal-risk state store"
                )
            if bool(getattr(risk, "risk_state_bootstrapped_this_process", False)):
                raise LiveTradingDisabled(
                    "Stage 2+ cannot start in the same process that bootstrapped risk "
                    "history; bootstrap is a separate operator workflow"
                )
            if not bool(getattr(risk, "deployment_context_verified", False)):
                raise LiveTradingDisabled(
                    "Stage 2+ durable state is not bound to a canonical coordinator context"
                )
            if not bool(getattr(risk, "broker_identity_is_pinned", False)):
                raise LiveTradingDisabled(
                    "Stage 2+ requires exact broker, route, paper/live, and account pins"
                )
            if not capabilities.stage_2_recovery_safe:
                raise LiveTradingDisabled(
                    f"{adapter.name} is blocked for Stage {int(self._deployment_stage)}: "
                    "native OCO/reduce-only, exact terminal and session execution history, "
                    "authoritative cancellation, and account-owner fencing are not all "
                    "implemented"
                )
            # Necessary capability declarations are not a complete Stage-2 execution
            # path. The current engine submits an entry first and attaches its stop only
            # after observing a fill; a process death in that interval can leave naked
            # venue exposure. Until GuardedBroker submits and reconciles one atomic
            # entry-with-protection transaction, no adapter may opt itself into Stage 2+
            # merely by advertising flags.
            raise LiveTradingDisabled(
                "Stage 2+ order submission is blocked until guarded atomic "
                "entry-with-protection, authoritative session replay, and account-owner "
                "fencing are implemented and behaviorally verified"
            )
        if capabilities.external_execution:
            if self._deployment_stage < DeploymentStage.PAPER:
                raise LiveTradingDisabled(
                    "external broker connectivity is forbidden in Stage 0/1"
                )
            if not capabilities.stage_2_protection_safe:
                raise LiveTradingDisabled(
                    f"{adapter.name} is blocked for Stage {int(self._deployment_stage)}: "
                    "native OCO, reduce-only/close-position, and authoritative cancel "
                    "status are not all implemented"
                )
        self._adapter = adapter
        self._execution_route = execution_route.strip().casefold()
        self._risk = risk
        self._capabilities = capabilities
        self.rejections: list[Rejection] = []
        self.submitted_count = 0
        self.refused_count = 0
        # Entries are serialized across snapshot -> verification -> adapter submission.
        # Risk-reducing orders intentionally do not take this lock.
        self._entry_lock = RLock()
        self._order_lock = RLock()
        # Only orders that crossed this guard are eligible for a cancellation operation.
        # The economic fields on these immutable Orders are the canonical identities used
        # to distinguish entries from protective orders and validate replacements.
        self._submitted_orders: dict[str, Order] = {}
        self._broker_ids_by_order_id: dict[str, str] = {}
        self._submission_requested_at_by_order_id: dict[str, datetime] = {}
        self._observed_fills_by_id: dict[str, Fill] = {}
        self._pending_entry: PendingEntryReservation | None = None
        self._reservation_store = reservation_store
        self._reservation_store_error: str | None = None
        self._recovery_in_progress = False
        if reservation_store is not None:
            try:
                restored = reservation_store.load()
                if restored is not None:
                    if not isinstance(restored, PendingEntryReservation):
                        raise TypeError(
                            "reservation store returned a non-reservation value"
                        )
                    restored.validate()
                    risk_instrument = getattr(
                        getattr(self._risk, "instrument", None),
                        "symbol",
                        None,
                    )
                    if restored.instrument != risk_instrument:
                        raise ValueError(
                            f"restored pending-entry instrument {restored.instrument!r} "
                            f"does not match risk instrument {risk_instrument!r}"
                        )
                    envelope_error = self._pending_price_envelope_error(restored)
                    if envelope_error is not None:
                        raise ValueError(envelope_error)
                    if restored.broker_execution_route != self._execution_route:
                        raise ValueError(
                            "restored pending-entry route "
                            f"{restored.broker_execution_route!r} does not match active "
                            f"route {self._execution_route!r}"
                        )
                self._pending_entry = restored
            except Exception as exc:
                # Construction still succeeds so risk-reducing orders remain possible.
                # Every entry remains locked until the operator repairs the configured
                # state and constructs a new guard.
                self._reservation_store_error = (
                    f"could not restore pending-entry reservation: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def is_paper(self) -> bool:
        return self._adapter.is_paper

    @property
    def capabilities(self) -> BrokerCapabilities:
        return self._capabilities

    @property
    def execution_route(self) -> str:
        return self._execution_route

    # ------------------------------------------------------------------ the gate

    def place_order(
        self, order: Order, token: RiskToken, *, now: datetime | None = None
    ) -> GuardResult:
        """Submit an order. `token` is required — that is the whole point of this class.

        Returns a `GuardResult` rather than raising on refusal: a refusal is an expected,
        journalled outcome, not an exceptional one. Broker-side failures still raise, since
        those are genuinely exceptional and need the error taxonomy.
        """
        verification_time = now or self._risk.clock.now()
        if order.purpose is OrderPurpose.ENTRY:
            return self._place_entry(order, token, now=verification_time)

        # Every risk-reducing order needs fresh venue proof too. A locally fabricated
        # Position must never turn a SELL on a flat account into an apparent exit.
        snapshot, snapshot_failure = self._read_authoritative_snapshot(
            now=verification_time
        )
        if snapshot_failure is not None:
            return self._refuse(snapshot_failure)
        assert snapshot is not None
        verification_time = self._verification_time_after_snapshot(
            requested=verification_time,
            snapshot=snapshot,
        )
        rejection = self._risk.verify(
            order,
            token,
            now=verification_time,
            broker_snapshot=snapshot,
        )
        if rejection is not None:
            return self._refuse(rejection)
        return self._submit_verified(order, now=verification_time, is_entry=False)

    def _place_entry(
        self, order: Order, token: RiskToken, *, now: datetime
    ) -> GuardResult:
        with self._entry_lock:
            if self._reservation_store_error is not None:
                return self._refuse(self._reservation_error_rejection(order, now=now))
            snapshot, snapshot_failure = self._read_authoritative_snapshot(now=now)
            if snapshot_failure is not None:
                return self._refuse(snapshot_failure)
            assert snapshot is not None
            now = self._verification_time_after_snapshot(
                requested=now,
                snapshot=snapshot,
            )

            if self._pending_entry is not None:
                # Validate before using the read to release an old reservation.  A
                # malformed, stale, or account-switched snapshot can only preserve the
                # lock, never clear it. With no reservation, RiskEngine performs this
                # validation after authenticating the token, preserving token-error
                # precedence without weakening the broker-state gate.
                if not self._pending_identity_matches_snapshot(
                    self._pending_entry,
                    snapshot,
                ):
                    return self._refuse(
                        self._reservation_error_rejection(order, now=now)
                    )
                structural_error = self._pending_snapshot_structural_error(snapshot)
                if structural_error is not None:
                    self._reservation_store_error = structural_error
                    return self._refuse(
                        self._reservation_error_rejection(order, now=now)
                    )
                # Resolve the reservation before personal-risk reconciliation can roll
                # the session and discard the old entry id. If it remains unresolved,
                # do not let a second entry trigger that rollover at all.
                self._reconcile_pending_entry(snapshot, now=now)
            if self._reservation_store_error is not None:
                return self._refuse(self._reservation_error_rejection(order, now=now))
            if self._pending_entry is not None:
                pending = self._pending_entry
                if (
                    self._reservation_store is None
                    and pending.order_id == order.order_id
                ):
                    # Preserve the legacy exact-replay diagnosis for process-local
                    # backtest/replay guards. Durable recovery does not call verify here,
                    # because its personal state must not roll while a reservation lives.
                    replay_rejection = self._risk.verify(
                        order,
                        token,
                        now=now,
                        broker_snapshot=snapshot,
                    )
                    if replay_rejection is not None:
                        return self._refuse(replay_rejection)
                if pending.ever_fill_observed and not self._risk.on_entry_fill_observed(
                    pending.order_id,
                    now=now,
                ):
                    return self._refuse(
                        self._reservation_error_rejection(order, now=now)
                    )
                return self._refuse(
                    Rejection(
                        timestamp=now,
                        reason=RejectReason.POSITION_ALREADY_OPEN,
                        detail=(
                            f"entry {pending.order_id} is still reserved in "
                            f"{pending.state.value} state; no second entry is permitted"
                        ),
                        stage="GUARD_RESERVATION",
                        instrument=order.instrument,
                        strategy=order.strategy,
                        intent_id=order.intent_id,
                        order_id=order.order_id,
                        context={
                            "pending_order_id": pending.order_id,
                            "pending_broker_order_id": pending.broker_order_id,
                            "pending_quantity": pending.quantity,
                            "pending_risk_usd": pending.risk_usd,
                            "pending_filled_quantity": (
                                pending.cumulative_filled_quantity
                            ),
                            "pending_ever_fill_observed": (
                                pending.ever_fill_observed
                            ),
                            "pending_broker_account_id": pending.broker_account_id,
                            "pending_execution_route": (
                                pending.broker_execution_route
                            ),
                        },
                    )
                )

            rejection = self._risk.verify(
                order, token, now=now, broker_snapshot=snapshot
            )
            if rejection is not None:
                return self._refuse(rejection)

            # The personal token is now spent.  Reserve before touching the adapter so a
            # timeout or unknown exception cannot reopen the entry path.
            reservation = PendingEntryReservation(
                order_id=order.order_id,
                broker_order_id=None,
                state=PendingEntryState.SUBMITTING,
                reserved_at=now,
                updated_at=now,
                quantity=order.quantity,
                risk_usd=token.risk_usd,
                instrument=order.instrument,
                side=order.side,
                order_type=order.order_type,
                limit_price=order.limit_price,
                protective_stop_price=token.stop_price,
                broker_account_id=snapshot.account.account_id,
                broker_execution_route=snapshot.execution_route,
                cumulative_filled_quantity=0,
                ever_fill_observed=False,
                observed_fill_ids=(),
                revision=1,
            )
            if not self._set_pending_entry(reservation):
                return self._refuse(self._reservation_error_rejection(order, now=now))
            try:
                return self._submit_verified(order, now=now, is_entry=True)
            except Exception:
                # The adapter may already have accepted or filled the entry even though
                # acknowledgement handling failed. Resolve its exact local/broker id when
                # possible, cancel any working remainder, and close confirmed exposure
                # before preserving the original exception for the operator.
                self._cancel_pending_entry_and_flatten(now=now)
                raise

    def _verification_time_after_snapshot(
        self,
        *,
        requested: datetime,
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> datetime:
        """Use a clock reading that cannot predate the snapshot it verifies.

        A wall clock normally advances between the initial request timestamp and the
        adapter reads.  Comparing that earlier timestamp to ``captured_at`` would label
        every real snapshot future-dated even though the read just completed.
        """
        observed_after_capture = self._risk.clock.now()
        return max(requested, snapshot.captured_at, observed_after_capture)

    def _submit_verified(
        self, order: Order, *, now: datetime, is_entry: bool
    ) -> GuardResult:

        try:
            ack = self._adapter.place_order(order)
        except OrderRejected as exc:
            if is_entry:
                self._mark_pending_entry(
                    order.order_id,
                    state=PendingEntryState.TERMINAL_REPORTED,
                    now=now,
                )
                terminal_resolved = self._reconcile_terminal_pending_entry(now=now)
                if not terminal_resolved:
                    self._cancel_pending_entry_and_flatten(
                        now=self._risk.clock.now()
                    )
            rejection = Rejection(
                timestamp=now,
                reason=RejectReason.BROKER_ERROR,
                detail=str(exc),
                stage="BROKER",
                instrument=order.instrument,
                strategy=order.strategy,
                order_id=order.order_id,
            )
            self.rejections.append(rejection)
            return GuardResult(ack=None, rejection=rejection)
        except Exception:
            if is_entry and self._pending_entry is not None:
                self._mark_pending_entry(
                    order.order_id,
                    state=PendingEntryState.OUTCOME_UNKNOWN,
                    now=now,
                )
            raise

        observed_at = self._risk.clock.now()
        ack_error = self._ack_structural_error(
            order,
            ack,
            requested_at=now,
            observed_at=observed_at,
        )
        if ack_error is not None:
            safe_broker_id = self._safe_ack_broker_order_id(order, ack)
            if is_entry and self._pending_entry is not None:
                self._mark_pending_entry(
                    order.order_id,
                    broker_order_id=safe_broker_id,
                    state=PendingEntryState.OUTCOME_UNKNOWN,
                    now=max(now, observed_at),
                )
            self._latch_broker_input_error(
                "malformed broker acknowledgement; " + ack_error
            )
            self.submitted_count += 1
            rejection = self._broker_ack_rejection(
                order,
                now=observed_at,
                detail=(
                    "broker acknowledgement was structurally invalid and its outcome "
                    "is unknown; recovery is required"
                ),
                ack_status=(
                    ack.status
                    if isinstance(ack, OrderAck) and type(ack.status) is OrderStatus
                    else None
                ),
                structural_error=ack_error,
            )
            self._recover_after_untrusted_broker_input(now=observed_at)
            return self._refuse(rejection)

        # Every field used below has crossed the structural trust boundary.
        assert isinstance(ack, OrderAck)
        assert type(ack.status) is OrderStatus

        if is_entry and self._pending_entry is not None:
            if ack.status in {
                OrderStatus.REJECTED,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
            }:
                state = PendingEntryState.TERMINAL_REPORTED
            elif ack.status in {
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
            }:
                # OrderAck intentionally has no Fill payload, so neither price nor costs
                # can be authenticated against the signed reservation. Preserve the
                # conservative quantity evidence but keep cancellation expressible.
                state = PendingEntryState.OUTCOME_UNKNOWN
            else:
                state = PendingEntryState.ACKNOWLEDGED
            if ack.status is OrderStatus.FILLED:
                ack_filled_quantity = order.quantity
                ack_fill_observed = True
            elif ack.status is OrderStatus.PARTIALLY_FILLED:
                # OrderAck has no fill-quantity field. Preserve the strongest fact it
                # does carry: at least one integer futures contract filled. A later
                # event or authoritative order snapshot can only raise this lower bound.
                ack_filled_quantity = min(1, order.quantity)
                ack_fill_observed = True
            else:
                ack_filled_quantity = None
                ack_fill_observed = None
            persisted = self._mark_pending_entry(
                order.order_id,
                broker_order_id=ack.broker_order_id,
                state=state,
                now=now,
                cumulative_filled_quantity=ack_filled_quantity,
                ever_fill_observed=ack_fill_observed,
            )
            if not persisted:
                raise ReservationStoreError(
                    "broker acknowledged an entry but its durable reservation could "
                    "not be advanced; the acknowledgement is withheld pending "
                    "authoritative recovery"
                )
            if ack.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                if not self._risk.on_entry_fill_observed(order.order_id, now=now):
                    raise ReservationStoreError(
                        "broker acknowledged an entry fill but durable personal-risk "
                        "state could not absorb it"
                    )
            if ack.status in {
                OrderStatus.REJECTED,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
            }:
                terminal_resolved = self._reconcile_terminal_pending_entry(now=now)
                if not terminal_resolved:
                    # A terminal acknowledgement contradicted by working-order history,
                    # exposure, or unavailable exact history is not trusted. Best-effort
                    # recovery can only reduce risk and the durable lock remains.
                    self._cancel_pending_entry_and_flatten(
                        now=self._risk.clock.now()
                    )
                self.submitted_count += 1
                rejection = Rejection(
                    timestamp=now,
                    reason=RejectReason.BROKER_ERROR,
                    detail=(
                        "broker returned a terminal entry acknowledgement "
                        f"({ack.status.value}); no working entry was exposed locally"
                    ),
                    stage="BROKER_ACK",
                    instrument=order.instrument,
                    strategy=order.strategy,
                    intent_id=order.intent_id,
                    order_id=order.order_id,
                    context={"ack_status": ack.status.value},
                )
                self.rejections.append(rejection)
                return GuardResult(ack=None, rejection=rejection)
            if ack.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                self.submitted_count += 1
                self._latch_broker_input_error(
                    "broker entry acknowledgement claimed fill progress without a "
                    "complete Fill payload; entry events are withheld and recovery is "
                    "required"
                )
                rejection = Rejection(
                    timestamp=now,
                    reason=RejectReason.BROKER_ERROR,
                    detail=self._reservation_store_error,
                    stage="BROKER_ACK",
                    instrument=order.instrument,
                    strategy=order.strategy,
                    intent_id=order.intent_id,
                    order_id=order.order_id,
                    context={"ack_status": ack.status.value},
                )
                self.rejections.append(rejection)
                self._recover_after_untrusted_broker_input(
                    now=self._risk.clock.now(),
                )
                return GuardResult(ack=None, rejection=rejection)

        # A terminal or fill-progress acknowledgement is not a successful submission for
        # exits, stops, targets, or emergency flattens either.  In particular, a STOP
        # reported terminal here must fail back to ExecutionEngine so it can engage its
        # independent emergency-flatten path.  Fill progress without a Fill payload has
        # no authenticated price, quantity delta, commission, or slippage economics.
        if not is_entry and ack.status in _TERMINAL_NO_FILL_ACK_STATUSES:
            self.submitted_count += 1
            return self._refuse(self._broker_ack_rejection(
                order,
                now=observed_at,
                detail=(
                    "broker returned a terminal submission acknowledgement "
                    f"({ack.status.value}); it is not a working-order acceptance"
                ),
                ack_status=ack.status,
            ))
        if not is_entry and ack.status in _FILL_PROGRESS_ACK_STATUSES:
            self.submitted_count += 1
            self._latch_broker_input_error(
                "broker acknowledgement claimed fill progress without a complete Fill "
                "payload; broker events are withheld and recovery is required"
            )
            rejection = self._broker_ack_rejection(
                order,
                now=observed_at,
                detail=self._reservation_store_error or (
                    "broker acknowledgement claimed unauthenticated fill progress"
                ),
                ack_status=ack.status,
            )
            self._recover_after_untrusted_broker_input(now=observed_at)
            return self._refuse(rejection)

        assert ack.status in _ACCEPTABLE_SUBMISSION_ACK_STATUSES
        canonical = order.with_status(
            ack.status,
            broker_order_id=ack.broker_order_id,
        )
        with self._order_lock:
            self._submitted_orders[ack.broker_order_id] = canonical
            self._broker_ids_by_order_id[order.order_id] = ack.broker_order_id
            self._submission_requested_at_by_order_id[order.order_id] = now
        self.submitted_count += 1
        return GuardResult(ack=ack, rejection=None)

    def _ack_structural_error(
        self,
        order: Order,
        ack: object,
        *,
        requested_at: datetime,
        observed_at: datetime,
    ) -> str | None:
        """Return why an immediate venue acknowledgement cannot be trusted."""

        if not isinstance(ack, OrderAck):
            return "adapter returned a non-OrderAck value"
        if not isinstance(ack.order_id, str) or ack.order_id != order.order_id:
            return "acknowledgement local order id does not match the submitted order"
        if ack.order_id != ack.order_id.strip() or not ack.order_id:
            return "acknowledgement local order id must be stable non-empty stripped text"
        if (
            not isinstance(ack.broker_order_id, str)
            or not ack.broker_order_id
            or ack.broker_order_id != ack.broker_order_id.strip()
        ):
            return "acknowledgement venue order id must be stable non-empty stripped text"
        if type(ack.status) is not OrderStatus:
            return "acknowledgement status is not an OrderStatus enum member"
        if ack.status not in (
            _ACCEPTABLE_SUBMISSION_ACK_STATUSES
            | _TERMINAL_NO_FILL_ACK_STATUSES
            | _FILL_PROGRESS_ACK_STATUSES
        ):
            return f"acknowledgement status {ack.status.value} is invalid at submission"
        if (
            not isinstance(ack.accepted_at, datetime)
            or ack.accepted_at.tzinfo is None
            or ack.accepted_at.utcoffset() is None
        ):
            return "acknowledgement timestamp must be timezone-aware"
        if ack.accepted_at < requested_at - _MAX_BROKER_MESSAGE_CLOCK_SKEW:
            return "acknowledgement timestamp implausibly predates submission"
        if ack.accepted_at > observed_at + _MAX_BROKER_MESSAGE_CLOCK_SKEW:
            return "acknowledgement timestamp is implausibly future-dated"
        if not isinstance(ack.detail, str):
            return "acknowledgement detail must be text"
        with self._order_lock:
            existing_broker_id = self._broker_ids_by_order_id.get(order.order_id)
            if (
                existing_broker_id is not None
                and existing_broker_id != ack.broker_order_id
            ):
                return "local order id was already bound to a different venue order id"
            existing_order = self._submitted_orders.get(ack.broker_order_id)
            if existing_order is not None and existing_order.order_id != order.order_id:
                return "venue order id collides with another local order"
        return None

    def _safe_ack_broker_order_id(
        self,
        order: Order,
        ack: object,
    ) -> str | None:
        """Return an id safe enough to cancel, even if another ack field is malformed."""

        if not isinstance(ack, OrderAck) or ack.order_id != order.order_id:
            return None
        broker_order_id = ack.broker_order_id
        if (
            not isinstance(broker_order_id, str)
            or not broker_order_id
            or broker_order_id != broker_order_id.strip()
        ):
            return None
        with self._order_lock:
            prior_for_local = self._broker_ids_by_order_id.get(order.order_id)
            prior_for_broker = self._submitted_orders.get(broker_order_id)
            if prior_for_local not in (None, broker_order_id):
                return None
            if prior_for_broker is not None and prior_for_broker.order_id != order.order_id:
                return None
        return broker_order_id

    @staticmethod
    def _broker_ack_rejection(
        order: Order,
        *,
        now: datetime,
        detail: str,
        ack_status: OrderStatus | None,
        structural_error: str | None = None,
    ) -> Rejection:
        context: dict[str, str] = {}
        if ack_status is not None:
            context["ack_status"] = ack_status.value
        if structural_error is not None:
            context["structural_error"] = structural_error
        return Rejection(
            timestamp=now,
            reason=RejectReason.BROKER_ERROR,
            detail=detail,
            stage="BROKER_ACK",
            instrument=order.instrument,
            strategy=order.strategy,
            intent_id=order.intent_id,
            order_id=order.order_id,
            context=context,
        )

    def _latch_broker_input_error(self, detail: str) -> None:
        # Preserve the first trust-boundary failure. Later recovery attempts may produce
        # secondary errors, but they must never erase the fact that entry permission was
        # already lost.
        if self._reservation_store_error is None:
            self._reservation_store_error = detail
        self._risk.kill_switch.trip(
            "broker trust-boundary validation failed; operator review required",
            by="guarded-broker",
        )

    def _read_authoritative_snapshot(
        self, *, now: datetime
    ) -> tuple[AuthoritativeBrokerSnapshot | None, Rejection | None]:
        read_started_at = self._risk.clock.now()
        try:
            # Orders first, positions second: if a working order fills during the read it
            # is observed either as working in the first collection or as a position in
            # the later collection. Account is read last for the freshest equity mark.
            orders = tuple(self._adapter.get_orders())
            positions = tuple(self._adapter.get_positions())
            account = self._adapter.get_account()
            captured_at = self._risk.clock.now()
            snapshot = AuthoritativeBrokerSnapshot(
                read_started_at=read_started_at,
                captured_at=captured_at,
                broker_name=self._adapter.name,
                broker_is_paper=self._adapter.is_paper,
                execution_route=self._execution_route,
                account=BrokerRiskAccount(
                    account_id=account.account_id,
                    equity=account.equity,
                    cash=account.cash,
                    realized_pnl=account.realized_pnl,
                    unrealized_pnl=account.unrealized_pnl,
                    currency=account.currency,
                    is_paper=account.is_paper,
                ),
                positions=tuple(
                    BrokerRiskPosition(
                        instrument=position.instrument,
                        quantity=position.quantity,
                        average_price=position.average_price,
                    )
                    for position in positions
                ),
                orders=tuple(
                    BrokerRiskOrder(
                        order_id=item.order_id,
                        broker_order_id=item.broker_order_id,
                        status=item.status,
                        filled_quantity=item.filled_quantity,
                        average_fill_price=item.average_fill_price,
                    )
                    for item in orders
                ),
            )
            return snapshot, None
        except Exception:
            return None, Rejection(
                timestamp=now,
                reason=RejectReason.BROKER_ERROR,
                detail=(
                    "could not obtain authoritative broker account/positions/orders "
                    "snapshot; one or more adapter reads failed"
                ),
                stage="BROKER_SNAPSHOT",
            )

    def _reconcile_pending_entry(
        self, snapshot: AuthoritativeBrokerSnapshot, *, now: datetime
    ) -> None:
        pending = self._pending_entry
        if pending is None:
            return
        if not self._pending_identity_matches_snapshot(pending, snapshot):
            return
        structural_error = self._pending_snapshot_structural_error(snapshot)
        if structural_error is not None:
            self._reservation_store_error = structural_error
            return
        matches = self._matching_pending_orders(pending, snapshot)
        if matches is None:
            return
        coherence_error = self._matched_order_coherence_error(pending, matches)
        if coherence_error is not None:
            self._reservation_store_error = coherence_error
            return
        working = tuple(item for item in matches if item.is_working)
        reported_filled_quantity = min(
            pending.quantity,
            max(
                (item.filled_quantity for item in matches),
                default=0,
            ),
        )
        if working:
            state = self._monotonic_pending_state(
                pending,
                (
                    PendingEntryState.PARTIALLY_FILLED
                    if reported_filled_quantity > 0
                    else PendingEntryState.ACKNOWLEDGED
                ),
                cumulative_filled_quantity=max(
                    pending.cumulative_filled_quantity,
                    reported_filled_quantity,
                ),
            )
            broker_id = next(
                (item.broker_order_id for item in working if item.broker_order_id),
                pending.broker_order_id,
            )
            self._set_pending_entry(replace(
                pending,
                broker_order_id=broker_id,
                state=state,
                updated_at=max(pending.updated_at, now),
                cumulative_filled_quantity=max(
                    pending.cumulative_filled_quantity,
                    reported_filled_quantity,
                ),
                ever_fill_observed=(
                    pending.ever_fill_observed
                    or reported_filled_quantity > 0
                ),
            ))
            return
        has_exposure = any(position.quantity != 0 for position in snapshot.positions)
        if matches:
            # The complete authoritative order read says this order is terminal.  It is
            # still unsafe to release the reservation if the same snapshot shows any
            # position: the entry may have filled before becoming terminal.
            if has_exposure:
                inferred_from_position = min(
                    pending.quantity,
                    sum(
                        abs(position.quantity)
                        for position in snapshot.positions
                        if position.instrument == pending.instrument
                    ),
                )
                filled_quantity = max(
                    pending.cumulative_filled_quantity,
                    reported_filled_quantity,
                    inferred_from_position,
                    1,
                )
                state = self._monotonic_pending_state(
                    pending,
                    PendingEntryState.TERMINAL_REPORTED,
                    cumulative_filled_quantity=filled_quantity,
                )
                self._set_pending_entry(replace(
                    pending,
                    state=state,
                    updated_at=max(pending.updated_at, now),
                    cumulative_filled_quantity=filled_quantity,
                    ever_fill_observed=True,
                ))
            else:
                self._clear_pending_entry_after_snapshot(snapshot)
            return
        if has_exposure:
            inferred_from_position = min(
                pending.quantity,
                sum(
                    abs(position.quantity)
                    for position in snapshot.positions
                    if position.instrument == pending.instrument
                ),
            )
            filled_quantity = max(
                pending.cumulative_filled_quantity,
                inferred_from_position,
                1,
            )
            state = self._monotonic_pending_state(
                pending,
                (
                    PendingEntryState.FILLED
                    if filled_quantity >= pending.quantity
                    else PendingEntryState.PARTIALLY_FILLED
                ),
                cumulative_filled_quantity=filled_quantity,
            )
            self._set_pending_entry(replace(
                pending,
                state=state,
                updated_at=max(pending.updated_at, now),
                cumulative_filled_quantity=filled_quantity,
                ever_fill_observed=True,
            ))
            return
        # Absence from a collection endpoint is not exact terminal history. A venue may
        # paginate, prune, or lag completed orders, so every state remains locked until
        # this specific client/broker order id is returned with a terminal status.

    def _reconcile_terminal_pending_entry(self, *, now: datetime) -> bool:
        """Try to release a terminal entry using fresh exact history and flat proof.

        An acknowledgement, cancellation command, or event is not by itself enough to
        release the durable reservation.  This helper is deliberately best-effort: an
        unavailable, stale, mismatched, or pruned snapshot leaves both reservation layers
        locked; only the existing exact-order reconciliation path may clear them.
        """

        snapshot, failure = self._read_authoritative_snapshot(now=now)
        if failure is not None or snapshot is None:
            return False
        verification_time = self._verification_time_after_snapshot(
            requested=now,
            snapshot=snapshot,
        )
        with self._entry_lock:
            pending = self._pending_entry
            if pending is None:
                return True
            if not self._pending_identity_matches_snapshot(pending, snapshot):
                return False
            structural_error = self._pending_snapshot_structural_error(snapshot)
            if structural_error is not None:
                self._reservation_store_error = structural_error
                return False
            self._reconcile_pending_entry(snapshot, now=verification_time)
            return self._pending_entry is None

    def _pending_identity_matches_snapshot(
        self,
        pending: PendingEntryReservation,
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> bool:
        mismatch: str | None = None
        if pending.broker_execution_route != self._execution_route:
            mismatch = (
                f"reservation route {pending.broker_execution_route!r} does not match "
                f"active route {self._execution_route!r}"
            )
        elif snapshot.execution_route != pending.broker_execution_route:
            mismatch = (
                f"snapshot route {snapshot.execution_route!r} does not match reserved "
                f"route {pending.broker_execution_route!r}"
            )
        elif snapshot.account.account_id != pending.broker_account_id:
            mismatch = (
                f"snapshot account {snapshot.account.account_id!r} does not match "
                f"reserved account {pending.broker_account_id!r}"
            )
        if mismatch is None:
            return True
        self._reservation_store_error = (
            "pending-entry reservation identity mismatch; " + mismatch
        )
        return False

    def _matching_pending_orders(
        self,
        pending: PendingEntryReservation,
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> tuple[BrokerRiskOrder, ...] | None:
        matches: list[BrokerRiskOrder] = []
        for item in snapshot.orders:
            local_match = item.order_id == pending.order_id
            if pending.broker_order_id is None:
                if local_match:
                    matches.append(item)
                continue
            broker_match = item.broker_order_id == pending.broker_order_id
            if local_match != broker_match:
                self._reservation_store_error = (
                    "pending-entry order identity conflict: local and broker order ids "
                    "do not identify the same authoritative row"
                )
                return None
            if local_match and broker_match:
                matches.append(item)
        return tuple(matches)

    def _pending_event_matches(
        self,
        pending: PendingEntryReservation,
        event: BrokerEvent,
    ) -> bool | None:
        local_match = event.order_id == pending.order_id
        if pending.broker_order_id is None:
            return local_match
        broker_match = event.broker_order_id == pending.broker_order_id
        if local_match != broker_match:
            self._reservation_store_error = (
                "pending-entry event identity conflict: local and broker order ids "
                "do not identify the same reservation"
            )
            return None
        return local_match and broker_match

    def _pending_fill_evidence_error(
        self,
        pending: PendingEntryReservation,
        event: BrokerEvent,
    ) -> str | None:
        """Return why broker fill evidence cannot fit the signed reservation envelope.

        Broker events are untrusted venue input.  No part of a fill is allowed to mutate
        durable state until the complete economic envelope is coherent; in particular an
        apparent overfill is rejected instead of being silently capped to the reservation.
        """

        fill_event_kinds = {EventKind.PARTIAL_FILL, EventKind.FILL}
        terminal_fill_kinds = {EventKind.CANCELLED, EventKind.REJECTED}
        fill = event.fill
        if event.kind in fill_event_kinds and fill is None:
            return f"{event.kind.value} event does not contain fill details"
        if fill is None:
            return None
        if event.kind not in fill_event_kinds | terminal_fill_kinds:
            return f"{event.kind.value} event cannot carry fill details"

        if (
            not isinstance(event.order_id, str)
            or event.order_id != pending.order_id
        ):
            return "fill event local order id does not exactly match the reservation"
        if (
            not isinstance(event.broker_order_id, str)
            or not event.broker_order_id.strip()
            or event.broker_order_id != event.broker_order_id.strip()
        ):
            return "fill event broker order id must be stable non-empty text"
        if (
            pending.broker_order_id is not None
            and event.broker_order_id != pending.broker_order_id
        ):
            return "fill event broker order id does not exactly match the reservation"
        if (
            not isinstance(fill.order_id, str)
            or fill.order_id != pending.order_id
        ):
            return "fill local order id does not exactly match the reservation"
        if (
            not isinstance(fill.instrument, str)
            or fill.instrument != pending.instrument
        ):
            return "fill instrument does not exactly match the reservation"
        if type(fill.side) is not Side or fill.side is not pending.side:
            return "fill side does not exactly match the signed reservation side"
        if (
            not isinstance(fill.fill_id, str)
            or not fill.fill_id.strip()
            or fill.fill_id != fill.fill_id.strip()
        ):
            return "fill id must be stable non-empty stripped text"
        if fill.broker_fill_id is not None and (
            not isinstance(fill.broker_fill_id, str)
            or not fill.broker_fill_id.strip()
            or fill.broker_fill_id != fill.broker_fill_id.strip()
        ):
            return "broker fill id must be null or stable non-empty stripped text"

        for label, moment in (
            ("event timestamp", event.timestamp),
            ("fill timestamp", fill.timestamp),
        ):
            if (
                not isinstance(moment, datetime)
                or moment.tzinfo is None
                or moment.utcoffset() is None
            ):
                return f"{label} must be timezone-aware"
        if event.timestamp < pending.reserved_at:
            return "fill event timestamp precedes the reservation"
        if fill.timestamp < pending.reserved_at:
            return "fill timestamp precedes the reservation"
        if fill.timestamp > event.timestamp:
            return "fill timestamp cannot be later than its observation event"

        if (
            not isinstance(fill.quantity, int)
            or isinstance(fill.quantity, bool)
            or fill.quantity <= 0
        ):
            return "fill quantity must be a positive integer"
        if (
            isinstance(fill.price, bool)
            or not isinstance(fill.price, (int, float))
            or not math.isfinite(float(fill.price))
            or fill.price <= 0
        ):
            return "fill price must be finite and positive"
        price = float(fill.price)
        tick_size = self._risk.instrument.tick_size
        tolerance = max(1e-9, abs(tick_size) * 1e-9)
        if abs(self._risk.instrument.round_to_tick(price) - price) > tolerance:
            return (
                f"fill price {price} is not aligned to the {tick_size}-point tick"
            )
        if pending.limit_price is not None:
            limit = float(pending.limit_price)
            if pending.side is Side.BUY and price > limit + tolerance:
                return "BUY fill price exceeds the signed reservation limit"
            if pending.side is Side.SELL and price < limit - tolerance:
                return "SELL fill price is below the signed reservation limit"

        for label, value in (
            ("commission_usd", fill.commission_usd),
            ("slippage_points", fill.slippage_points),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                return f"fill {label} must be finite and non-negative"
        if type(fill.is_partial) is not bool:
            return "fill is_partial must be a boolean"
        if event.kind is EventKind.PARTIAL_FILL and not fill.is_partial:
            return "PARTIAL_FILL event must carry partial fill evidence"
        if event.kind is EventKind.FILL and fill.is_partial:
            return "FILL event cannot carry partial fill evidence"

        if fill.fill_id not in pending.observed_fill_ids:
            cumulative = pending.cumulative_filled_quantity + fill.quantity
            if cumulative > pending.quantity:
                return (
                    "fill cumulative quantity exceeds the reserved entry quantity"
                )
            if event.kind is EventKind.PARTIAL_FILL and cumulative >= pending.quantity:
                return "PARTIAL_FILL event completes the reserved entry quantity"
            if event.kind is EventKind.FILL and cumulative != pending.quantity:
                return "FILL event does not complete the reserved entry quantity"
        return None

    def _pending_price_envelope_error(
        self,
        pending: PendingEntryReservation,
    ) -> str | None:
        """Validate price increments that the broker-agnostic file schema cannot know."""

        tick_size = self._risk.instrument.tick_size
        tolerance = max(1e-9, abs(tick_size) * 1e-9)
        for label, price in (
            ("limit_price", pending.limit_price),
            ("protective_stop_price", pending.protective_stop_price),
        ):
            if price is None:
                continue
            if abs(self._risk.instrument.round_to_tick(price) - price) > tolerance:
                return (
                    f"pending entry {label} {price} is not aligned to the "
                    f"{tick_size}-point tick"
                )
        return None

    @staticmethod
    def _matched_order_coherence_error(
        pending: PendingEntryReservation,
        matches: tuple[BrokerRiskOrder, ...],
    ) -> str | None:
        for item in matches:
            filled = item.filled_quantity
            if filled > pending.quantity:
                return "pending-entry snapshot reports fills beyond reserved quantity"
            if item.status is OrderStatus.PARTIALLY_FILLED and not (
                0 < filled < pending.quantity
            ):
                return (
                    "pending-entry PARTIALLY_FILLED status has incoherent filled quantity"
                )
            if (
                item.status is OrderStatus.FILLED
                and filled != pending.quantity
            ):
                return "pending-entry FILLED status does not equal reserved quantity"
            if item.status in {
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED,
                OrderStatus.REJECTED,
            } and filled != 0:
                return (
                    f"pending-entry {item.status.value} status cannot report fills"
                )
        return None

    @staticmethod
    def _pending_snapshot_structural_error(
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> str | None:
        """Validate fields used before the broader personal-risk reconciliation."""
        if not isinstance(snapshot.positions, tuple) or not all(
            isinstance(position, BrokerRiskPosition)
            for position in snapshot.positions
        ):
            return "pending-entry snapshot positions are not an immutable typed tuple"
        for position in snapshot.positions:
            if (
                not isinstance(position.instrument, str)
                or not position.instrument.strip()
                or not isinstance(position.quantity, int)
                or isinstance(position.quantity, bool)
            ):
                return "pending-entry snapshot contains a malformed position"
        if not isinstance(snapshot.orders, tuple) or not all(
            isinstance(order, BrokerRiskOrder) for order in snapshot.orders
        ):
            return "pending-entry snapshot orders are not an immutable typed tuple"
        for order in snapshot.orders:
            if (
                not isinstance(order.order_id, str)
                or not isinstance(order.broker_order_id, str)
                or not isinstance(order.status, OrderStatus)
                or not isinstance(order.filled_quantity, int)
                or isinstance(order.filled_quantity, bool)
                or order.filled_quantity < 0
            ):
                return "pending-entry snapshot contains a malformed order"
            if (
                order.status
                in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
                and order.filled_quantity == 0
            ):
                return (
                    f"pending-entry snapshot {order.status.value} status requires a "
                    "positive filled quantity"
                )
        return None

    @staticmethod
    def _monotonic_pending_state(
        pending: PendingEntryReservation,
        candidate: PendingEntryState,
        *,
        cumulative_filled_quantity: int,
    ) -> PendingEntryState:
        """Keep venue progress and cancellation intent from moving backwards."""
        if pending.state is PendingEntryState.FILLED:
            return PendingEntryState.FILLED
        if pending.state is PendingEntryState.TERMINAL_REPORTED:
            return PendingEntryState.TERMINAL_REPORTED
        if (
            pending.state is PendingEntryState.CANCEL_REQUESTED
            and candidate
            in {
                PendingEntryState.ACKNOWLEDGED,
                PendingEntryState.PARTIALLY_FILLED,
            }
        ):
            return PendingEntryState.CANCEL_REQUESTED
        if (
            pending.state is PendingEntryState.PARTIALLY_FILLED
            and candidate is PendingEntryState.ACKNOWLEDGED
        ):
            return PendingEntryState.PARTIALLY_FILLED
        if (
            candidate is PendingEntryState.PARTIALLY_FILLED
            and cumulative_filled_quantity >= pending.quantity
        ):
            return PendingEntryState.FILLED
        return candidate

    def _set_pending_entry(self, reservation: PendingEntryReservation) -> bool:
        """Update memory and then the durable copy before any later unsafe action."""
        current = self._pending_entry
        if current is not None:
            reservation = replace(reservation, revision=current.revision + 1)
        try:
            reservation.validate()
            envelope_error = self._pending_price_envelope_error(reservation)
            if envelope_error is not None:
                raise ValueError(envelope_error)
        except Exception as exc:
            self._reservation_store_error = (
                "refused invalid pending-entry reservation transition: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        self._pending_entry = reservation
        if self._reservation_store is None:
            return True
        if self._reservation_store_error is not None:
            return False
        try:
            persisted = self._reservation_store.save(reservation)
            if not isinstance(persisted, PendingEntryReservation):
                raise TypeError(
                    "reservation store did not return the exact persisted state"
                )
            persisted.validate()
            self._pending_entry = persisted
        except Exception as exc:
            # A previous durable state is conservative and remains on disk when atomic
            # replace fails.  Latch the fault so this process cannot submit another entry.
            self._reservation_store_error = (
                f"could not persist pending-entry reservation: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        return True

    def _mark_pending_entry(
        self,
        order_id: str,
        *,
        state: PendingEntryState,
        now: datetime,
        broker_order_id: str | None = None,
        cumulative_filled_quantity: int | None = None,
        ever_fill_observed: bool | None = None,
    ) -> bool:
        pending = self._pending_entry
        if pending is None or pending.order_id != order_id:
            return False
        filled_quantity = max(
            pending.cumulative_filled_quantity,
            (
                cumulative_filled_quantity
                if cumulative_filled_quantity is not None
                else 0
            ),
        )
        state = self._monotonic_pending_state(
            pending,
            state,
            cumulative_filled_quantity=filled_quantity,
        )
        return self._set_pending_entry(replace(
            pending,
            broker_order_id=(
                broker_order_id
                if broker_order_id is not None
                else pending.broker_order_id
            ),
            state=state,
            updated_at=max(pending.updated_at, now),
            cumulative_filled_quantity=filled_quantity,
            ever_fill_observed=(
                pending.ever_fill_observed
                or bool(ever_fill_observed)
                or filled_quantity > 0
            ),
        ))

    def _clear_pending_entry_after_snapshot(
        self, snapshot: AuthoritativeBrokerSnapshot
    ) -> bool:
        """Clear only when this exact authoritative read proves terminal and flat."""
        pending = self._pending_entry
        if pending is None:
            return True
        if not self._pending_identity_matches_snapshot(pending, snapshot):
            return False
        if any(position.quantity != 0 for position in snapshot.positions):
            return False
        matches = self._matching_pending_orders(pending, snapshot)
        if matches is None:
            return False
        coherence_error = self._matched_order_coherence_error(pending, matches)
        if coherence_error is not None:
            self._reservation_store_error = coherence_error
            return False
        terminal_proven = bool(matches) and not any(
            item.is_working for item in matches
        )
        if not terminal_proven:
            return False
        reported_filled_quantity = min(
            pending.quantity,
            max((item.filled_quantity for item in matches), default=0),
        )
        filled_quantity = max(
            pending.cumulative_filled_quantity,
            reported_filled_quantity,
        )
        terminal_pending = replace(
            pending,
            state=self._monotonic_pending_state(
                pending,
                PendingEntryState.TERMINAL_REPORTED,
                cumulative_filled_quantity=filled_quantity,
            ),
            updated_at=max(pending.updated_at, snapshot.captured_at),
            cumulative_filled_quantity=filled_quantity,
            ever_fill_observed=(
                pending.ever_fill_observed or reported_filled_quantity > 0
            ),
        )
        # Persist terminal fill evidence before either the personal-risk state absorbs
        # it or the file is removed. A crash at any later point can only leave a
        # conservative reservation behind; it cannot resurrect an "unfilled" story.
        if not self._set_pending_entry(terminal_pending):
            return False
        pending = self._pending_entry
        assert pending is not None
        filled_evidence = (
            pending.ever_fill_observed
            or pending.cumulative_filled_quantity > 0
        )
        absorbed = (
            self._risk.on_entry_fill_observed(
                pending.order_id,
                now=snapshot.captured_at,
            )
            if filled_evidence
            else self._risk.on_entry_terminal_unfilled(
                pending.order_id,
                now=snapshot.captured_at,
            )
        )
        if not absorbed:
            return False
        if self._reservation_store is not None:
            if self._reservation_store_error is not None:
                return False
            try:
                self._reservation_store.clear(pending)
            except Exception as exc:
                self._reservation_store_error = (
                    f"could not clear pending-entry reservation: "
                    f"{type(exc).__name__}: {exc}"
                )
                return False
        self._pending_entry = None
        return True

    def _reservation_error_rejection(
        self, order: Order, *, now: datetime
    ) -> Rejection:
        return Rejection(
            timestamp=now,
            reason=RejectReason.BROKER_ERROR,
            detail=(
                "pending-entry reservation persistence is unavailable; entry remains "
                "locked until the configured state is repaired"
            ),
            stage="GUARD_RESERVATION_STORE",
            instrument=order.instrument,
            strategy=order.strategy,
            intent_id=order.intent_id,
            order_id=order.order_id,
        )

    def _refuse(self, rejection: Rejection) -> GuardResult:
        self.refused_count += 1
        self.rejections.append(rejection)
        return GuardResult(ack=None, rejection=rejection)

    @property
    def pending_entry(self) -> PendingEntryReservation | None:
        with self._entry_lock:
            return self._pending_entry

    @property
    def reservation_store_error(self) -> str | None:
        """Latched persistence fault; entries stay blocked while this is non-null."""
        with self._entry_lock:
            return self._reservation_store_error

    def cancel_entry(
        self, broker_order_id: str, *, now: datetime | None = None
    ) -> CancelResult:
        """Request cancellation of an entry submitted by this exact guard.

        No risk token is needed because the operation can only remove still-pending entry
        exposure.  The guard-owned ledger is the authority: an unknown id or a protective
        order can never reach the adapter through this method.
        """

        requested_at = now or self._risk.clock.now()
        order = self.submitted_order(broker_order_id)
        if order is None:
            # A restarted process has an empty in-memory order ledger, but the pending
            # entry itself was persisted precisely so the in-flight order can still be
            # resolved. Accepting an id that matches the restored reservation is what
            # makes crash recovery possible; anything else is still refused, so an
            # arbitrary id cannot reach the adapter through this path.
            recovered = self._cancel_recovered_entry(broker_order_id, requested_at)
            if recovered is not None:
                return recovered
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    requested_at,
                    "entry cancellation id was not submitted by this guard",
                )
            )
        if order.purpose is not OrderPurpose.ENTRY:
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    requested_at,
                    f"{order.purpose.value} orders cannot use the entry cancellation path",
                    order=order,
                )
            )
        result = self._request_cancel(order, broker_order_id, now=requested_at)
        if result.accepted:
            with self._entry_lock:
                if (
                    self._pending_entry is not None
                    and self._pending_entry.broker_order_id == broker_order_id
                ):
                    pending = self._pending_entry
                    self._set_pending_entry(replace(
                        pending,
                        state=self._monotonic_pending_state(
                            pending,
                            PendingEntryState.CANCEL_REQUESTED,
                            cumulative_filled_quantity=(
                                pending.cumulative_filled_quantity
                            ),
                        ),
                        updated_at=max(pending.updated_at, requested_at),
                    ))
        return result

    def retire_protective(
        self,
        broker_order_id: str,
        *,
        replacement_broker_order_id: str | None = None,
        now: datetime | None = None,
    ) -> CancelResult:
        """Request retirement of guard-submitted STOP/TARGET protection.

        Protection may be removed only after one fresh, identity-valid broker snapshot
        proves the covered instrument flat, or proves that an explicitly named replacement
        submitted by this guard is working and exactly covers current exposure.  A stop
        replacement must be equal or tighter than the stop it supersedes.
        """

        requested_at = now or self._risk.clock.now()
        old = self.submitted_order(broker_order_id)
        if old is None:
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    requested_at,
                    "protective cancellation id was not submitted by this guard",
                )
            )
        if old.purpose not in (OrderPurpose.STOP, OrderPurpose.TARGET):
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    requested_at,
                    f"{old.purpose.value} is not guard-submitted protection",
                    order=old,
                )
            )

        snapshot, snapshot_failure = self._read_authoritative_snapshot(now=requested_at)
        if snapshot_failure is not None:
            return self._refuse_cancel(snapshot_failure)
        assert snapshot is not None
        verified_at = self._verification_time_after_snapshot(
            requested=requested_at,
            snapshot=snapshot,
        )
        snapshot_rejection = self._risk.reconcile_broker_snapshot(
            snapshot,
            now=verified_at,
            order=old,
        )
        if snapshot_rejection is not None:
            return self._refuse_cancel(snapshot_rejection)

        matching_positions = tuple(
            position
            for position in snapshot.nonflat_positions
            if position.instrument == old.instrument
        )
        if not matching_positions:
            return self._request_cancel(old, broker_order_id, now=verified_at)
        if len(matching_positions) != 1:
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    verified_at,
                    "protective retirement requires one unambiguous position for its instrument",
                    order=old,
                )
            )

        position = matching_positions[0]
        replacement = (
            self.submitted_order(replacement_broker_order_id)
            if replacement_broker_order_id
            else None
        )
        replacement_error = self._protective_replacement_error(
            old,
            replacement,
            replacement_broker_order_id=replacement_broker_order_id,
            position=position,
            snapshot=snapshot,
        )
        if replacement_error is not None:
            return self._refuse_cancel(
                self._cancel_rejection(
                    broker_order_id,
                    verified_at,
                    replacement_error,
                    order=old,
                )
            )
        return self._request_cancel(old, broker_order_id, now=verified_at)

    def submitted_order(self, broker_order_id: str | None) -> Order | None:
        """Return the immutable canonical order recorded for one venue id."""

        if not broker_order_id:
            return None
        with self._order_lock:
            return self._submitted_orders.get(broker_order_id)

    @staticmethod
    def _protective_replacement_error(
        old: Order,
        replacement: Order | None,
        *,
        replacement_broker_order_id: str | None,
        position: BrokerRiskPosition,
        snapshot: AuthoritativeBrokerSnapshot,
    ) -> str | None:
        if replacement_broker_order_id is None:
            return "open exposure requires an explicit working protective replacement"
        if replacement is None:
            return "protective replacement id was not submitted by this guard"
        if replacement_broker_order_id == old.broker_order_id:
            return "protective replacement must be a distinct order"
        if replacement.purpose is not old.purpose:
            return "protective replacement purpose does not match the retired order"
        if replacement.instrument != old.instrument or replacement.side is not old.side:
            return "protective replacement instrument/side does not match"
        expected_side = Side.SELL if position.quantity > 0 else Side.BUY
        if replacement.side is not expected_side:
            return "protective replacement does not oppose current broker exposure"
        if replacement.quantity != abs(position.quantity):
            return "protective replacement quantity does not cover current broker exposure"
        if old.purpose is OrderPurpose.STOP:
            if old.stop_price is None or replacement.stop_price is None:
                return "stop retirement requires two explicit stop prices"
            widening = (
                old.side is Side.SELL
                and replacement.stop_price < old.stop_price - 1e-9
            ) or (
                old.side is Side.BUY
                and replacement.stop_price > old.stop_price + 1e-9
            )
            if widening:
                return "protective stop replacement is wider than the retired stop"

        working = any(
            item.is_working
            and (
                item.broker_order_id == replacement_broker_order_id
                or item.order_id == replacement.order_id
            )
            for item in snapshot.orders
        )
        if not working:
            return "protective replacement is not working in the fresh broker snapshot"
        return None

    def _cancel_recovered_entry(
        self, broker_order_id: str, now: datetime
    ) -> CancelResult | None:
        """Cancel an entry known only from a restored reservation.

        Returns None when this id is not the restored pending entry, so the caller falls
        through to its ordinary refusal.
        """
        with self._entry_lock:
            pending = self._pending_entry
            if pending is None or pending.broker_order_id != broker_order_id:
                return None
            if pending.state in _PENDING_ENTRY_CANCEL_COMPLETE:
                return CancelResult(request_sent=False)

        # A broker order id alone is not sufficient authority after restart. Prove that
        # the active adapter still addresses the exact account and route captured before
        # submission; otherwise an id collision or account switch could cancel the wrong
        # venue order while leaving the original exposure alive.
        snapshot, snapshot_failure = self._read_authoritative_snapshot(now=now)
        if snapshot_failure is not None:
            return self._refuse_cancel(snapshot_failure)
        assert snapshot is not None
        with self._entry_lock:
            pending = self._pending_entry
            if pending is None or pending.broker_order_id != broker_order_id:
                return CancelResult(request_sent=False)
            if not self._pending_identity_matches_snapshot(pending, snapshot):
                return self._refuse_cancel(
                    self._cancel_rejection(
                        broker_order_id,
                        now,
                        "recovered entry reservation account/route identity does not "
                        "match the active broker snapshot",
                    )
                )
            self._reconcile_pending_entry(snapshot, now=now)
            pending = self._pending_entry
            if pending is None or pending.broker_order_id != broker_order_id:
                return CancelResult(request_sent=False)
            if pending.state in _PENDING_ENTRY_CANCEL_COMPLETE:
                return CancelResult(request_sent=False)

        self._adapter.cancel_order(broker_order_id)
        with self._entry_lock:
            pending = self._pending_entry
            if pending is not None and pending.broker_order_id == broker_order_id:
                self._set_pending_entry(replace(
                    pending,
                    state=self._monotonic_pending_state(
                        pending,
                        PendingEntryState.CANCEL_REQUESTED,
                        cumulative_filled_quantity=(
                            pending.cumulative_filled_quantity
                        ),
                    ),
                    updated_at=max(pending.updated_at, now),
                ))
        return CancelResult(request_sent=True)

    def _request_cancel(
        self, order: Order, broker_order_id: str, *, now: datetime
    ) -> CancelResult:
        if order.status.is_terminal:
            return CancelResult(request_sent=False)
        if order.status is OrderStatus.CANCEL_REQUESTED:
            return CancelResult(request_sent=False)
        self._adapter.cancel_order(broker_order_id)
        with self._order_lock:
            current = self._submitted_orders.get(broker_order_id, order)
            if not current.status.is_terminal:
                self._submitted_orders[broker_order_id] = current.with_status(
                    OrderStatus.CANCEL_REQUESTED
                )
        return CancelResult(request_sent=True)

    def _cancel_rejection(
        self,
        broker_order_id: str,
        now: datetime,
        detail: str,
        *,
        order: Order | None = None,
    ) -> Rejection:
        return Rejection(
            timestamp=now,
            reason=RejectReason.INVALID_ORDER,
            detail=detail,
            stage="GUARD_CANCEL",
            instrument=order.instrument if order is not None else "",
            strategy=order.strategy if order is not None else "",
            intent_id=order.intent_id if order is not None else None,
            order_id=order.order_id if order is not None else broker_order_id,
            context={"broker_order_id": broker_order_id},
        )

    def _refuse_cancel(self, rejection: Rejection) -> CancelResult:
        self.refused_count += 1
        self.rejections.append(rejection)
        return CancelResult(request_sent=False, rejection=rejection)

    # ------------------------------------------------------------------ read-only

    def connect(self) -> None:
        self._adapter.connect()

    def disconnect(self) -> None:
        self._adapter.disconnect()

    def is_connected(self) -> bool:
        return self._adapter.is_connected()

    def get_orders(self) -> list[BrokerOrderState]:
        return self._adapter.get_orders()

    def get_positions(self) -> list[BrokerPosition]:
        return self._adapter.get_positions()

    def get_account(self) -> BrokerAccount:
        return self._adapter.get_account()

    def reconcile_risk_state(self, *, now: datetime | None = None) -> Rejection | None:
        """Reconcile a pending entry before personal state is allowed to roll."""
        requested = now or self._risk.clock.now()
        if self._reservation_store_error is not None:
            return Rejection(
                timestamp=requested,
                reason=RejectReason.BROKER_ERROR,
                detail=(
                    "pending-entry reservation is unavailable; startup remains locked"
                ),
                stage="GUARD_RESERVATION_STORE",
            )
        snapshot, failure = self._read_authoritative_snapshot(now=requested)
        if failure is not None:
            return failure
        assert snapshot is not None
        verification_time = self._verification_time_after_snapshot(
            requested=requested,
            snapshot=snapshot,
        )
        with self._entry_lock:
            pending = self._pending_entry
            if pending is not None:
                if not self._pending_identity_matches_snapshot(pending, snapshot):
                    return Rejection(
                        timestamp=verification_time,
                        reason=RejectReason.BROKER_ERROR,
                        detail=(
                            "pending-entry reservation identity does not match the "
                            "startup broker snapshot"
                        ),
                        stage="GUARD_RESERVATION_STORE",
                    )
                structural_error = self._pending_snapshot_structural_error(snapshot)
                if structural_error is not None:
                    self._reservation_store_error = structural_error
                    return Rejection(
                        timestamp=verification_time,
                        reason=RejectReason.BROKER_ERROR,
                        detail=structural_error,
                        stage="GUARD_RESERVATION_STORE",
                    )
                self._reconcile_pending_entry(snapshot, now=verification_time)
                if self._reservation_store_error is not None:
                    return Rejection(
                        timestamp=verification_time,
                        reason=RejectReason.BROKER_ERROR,
                        detail=self._reservation_store_error,
                        stage="GUARD_RESERVATION_STORE",
                    )
                pending = self._pending_entry
                if pending is not None:
                    if (
                        pending.ever_fill_observed
                        and not self._risk.on_entry_fill_observed(
                            pending.order_id,
                            now=verification_time,
                        )
                    ):
                        return Rejection(
                            timestamp=verification_time,
                            reason=RejectReason.BROKER_ERROR,
                            detail=(
                                "durable personal-risk state could not absorb pending "
                                "entry fill evidence"
                            ),
                            stage="GUARD_RESERVATION_STORE",
                        )
                    return Rejection(
                        timestamp=verification_time,
                        reason=RejectReason.POSITION_ALREADY_OPEN,
                        detail=(
                            f"pending entry {pending.order_id} remains unresolved in "
                            f"{pending.state.value} state"
                        ),
                        stage="GUARD_RESERVATION",
                    )
        return self._risk.reconcile_broker_snapshot(
            snapshot,
            now=verification_time,
        )

    def flatten_recovered_exposure(
        self,
        *,
        now: datetime | None = None,
    ) -> GuardResult:
        """Flatten exact venue exposure when restart left no local Position object."""

        requested = now or self._risk.clock.now()
        if not self._capabilities.reduce_only_or_close_position:
            return self._refuse(Rejection(
                timestamp=requested,
                reason=RejectReason.LIVE_TRADING_DISABLED,
                detail=(
                    "broker route has no venue-enforced reduce-only/close-position "
                    "semantics for emergency recovery"
                ),
                stage="GUARD_RECOVERY",
            ))
        snapshot, failure = self._read_authoritative_snapshot(now=requested)
        if failure is not None:
            return self._refuse(failure)
        assert snapshot is not None
        verification_time = self._verification_time_after_snapshot(
            requested=requested,
            snapshot=snapshot,
        )
        decision = self._risk.evaluate_snapshot_flatten(
            snapshot,
            now=verification_time,
        )
        if not decision.approved:
            assert decision.rejection is not None
            return self._refuse(decision.rejection)
        approval = decision.approval
        assert approval is not None
        # place_order deliberately obtains a second authoritative snapshot. If exposure
        # changed between authorization and submission, exact-quantity verification fails.
        return self.place_order(
            approval.order,
            approval.token,
            now=verification_time,
        )

    def _cancel_pending_entry_and_flatten(self, *, now: datetime) -> None:
        """Best-effort fail-safe after an entry outcome becomes locally unreliable."""

        with self._entry_lock:
            pending_broker_order_id = (
                self._pending_entry.broker_order_id
                if self._pending_entry is not None
                else None
            )

        if not pending_broker_order_id:
            # A timeout can hide the acknowledgement while the venue still exposes the
            # client order id. Reconcile only to discover that exact broker id; any state
            # write failure stays latched and cannot reopen entry permission.
            snapshot, failure = self._read_authoritative_snapshot(now=now)
            if failure is None and snapshot is not None:
                with self._entry_lock:
                    pending = self._pending_entry
                    if (
                        pending is not None
                        and self._pending_identity_matches_snapshot(pending, snapshot)
                        and self._pending_snapshot_structural_error(snapshot) is None
                    ):
                        self._reconcile_pending_entry(snapshot, now=now)
                    pending_broker_order_id = (
                        self._pending_entry.broker_order_id
                        if self._pending_entry is not None
                        else None
                    )

        if pending_broker_order_id:
            try:
                self.cancel_entry(pending_broker_order_id, now=now)
            except Exception:
                pass
        try:
            self.flatten_recovered_exposure(now=now)
        except Exception:
            pass

    def _recover_after_untrusted_broker_input(self, *, now: datetime) -> None:
        """Bounded recovery for an acknowledgement/event whose outcome is unknowable."""

        # A malformed acknowledgement on the emergency FLATTEN submitted below must not
        # recursively submit an unbounded chain of more flattens.  The first recovery
        # attempt owns this flag until its cancel/reconcile/flatten sequence completes.
        with self._entry_lock:
            if self._recovery_in_progress:
                return
            self._recovery_in_progress = True
        try:
            self._cancel_pending_entry_and_flatten(now=now)
        finally:
            with self._entry_lock:
                self._recovery_in_progress = False

    def _broker_events_structural_error(
        self,
        events: object,
        *,
        observed_at: datetime,
    ) -> str | None:
        """Validate an entire event batch before exposing or applying any event."""

        if not isinstance(events, list):
            return "adapter returned a non-list broker event batch"
        batch_local_to_broker: dict[str, str] = {}
        batch_broker_to_local: dict[str, str] = {}
        with self._order_lock:
            shadow_orders = dict(self._submitted_orders)
            shadow_fills = dict(self._observed_fills_by_id)
        for index, event in enumerate(events):
            error = self._broker_event_structural_error(
                event,
                observed_at=observed_at,
                batch_local_to_broker=batch_local_to_broker,
                batch_broker_to_local=batch_broker_to_local,
                shadow_orders=shadow_orders,
                shadow_fills=shadow_fills,
            )
            if error is not None:
                return f"event[{index}] {error}"
            assert isinstance(event, BrokerEvent)
            self._advance_event_validation_shadow(
                event,
                shadow_orders=shadow_orders,
                shadow_fills=shadow_fills,
            )
        return None

    def _broker_event_structural_error(
        self,
        event: object,
        *,
        observed_at: datetime,
        batch_local_to_broker: dict[str, str],
        batch_broker_to_local: dict[str, str],
        shadow_orders: dict[str, Order],
        shadow_fills: dict[str, Fill],
    ) -> str | None:
        if not isinstance(event, BrokerEvent):
            return "is not a typed BrokerEvent"
        if type(event.kind) is not EventKind:
            return "kind is not an EventKind enum member"
        if (
            not isinstance(event.timestamp, datetime)
            or event.timestamp.tzinfo is None
            or event.timestamp.utcoffset() is None
        ):
            return "timestamp must be timezone-aware"
        if event.timestamp > observed_at + _MAX_BROKER_MESSAGE_CLOCK_SKEW:
            return "timestamp is implausibly future-dated"
        if not isinstance(event.detail, str):
            return "detail must be text"

        for label, value in (
            ("local order id", event.order_id),
            ("venue order id", event.broker_order_id),
        ):
            if not isinstance(value, str) or value != value.strip():
                return f"{label} must be stable stripped text"

        local_id = event.order_id
        broker_id = event.broker_order_id
        if event.kind in _ORDER_EVENT_KINDS:
            if not local_id:
                return "order lifecycle event has no stable local order id"
            # A venue-side rejection may happen before a venue id is allocated. Every
            # other lifecycle state, including fill-less CANCELLED/CANCEL_REQUESTED, must
            # carry both stable identities. A rejection for an already-bound order also
            # has no exemption.
            with self._order_lock:
                known_broker_id = self._broker_ids_by_order_id.get(local_id)
            if not broker_id and (
                event.kind is not EventKind.REJECTED or known_broker_id is not None
            ):
                return "order lifecycle event has no stable venue order id"
        elif bool(local_id) != bool(broker_id):
            return "non-order event carries only one of its two order identities"

        if event.fill is not None and (not local_id or not broker_id):
            return "fill-bearing event requires stable local and venue order ids"

        submitted_order: Order | None = None
        if broker_id:
            with self._order_lock:
                known_broker_id = self._broker_ids_by_order_id.get(local_id)
            submitted_order = shadow_orders.get(broker_id)
            if known_broker_id is not None and known_broker_id != broker_id:
                return "local order id changed its venue order identity"
            if submitted_order is not None and submitted_order.order_id != local_id:
                return "venue order id collides with another local order"
            batch_broker = batch_local_to_broker.get(local_id)
            if batch_broker is not None and batch_broker != broker_id:
                return "local order id changes venue identity within one event batch"
            batch_local = batch_broker_to_local.get(broker_id)
            if batch_local is not None and batch_local != local_id:
                return "venue order id collides within one event batch"
            batch_local_to_broker[local_id] = broker_id
            batch_broker_to_local[broker_id] = local_id

        submitted_at: datetime | None = None
        if local_id:
            with self._order_lock:
                submitted_at = self._submission_requested_at_by_order_id.get(local_id)
            if submitted_at is None:
                with self._entry_lock:
                    pending = self._pending_entry
                    if pending is not None and pending.order_id == local_id:
                        submitted_at = pending.reserved_at
        if submitted_at is not None:
            if event.timestamp < submitted_at - _MAX_BROKER_MESSAGE_CLOCK_SKEW:
                return "timestamp implausibly predates the order submission"
        elif event.timestamp < observed_at - _MAX_UNBOUND_BROKER_EVENT_AGE:
            return "unbound event timestamp is older than one complete trading day"

        fill_error = self._submitted_fill_event_error(
            event,
            submitted_order=submitted_order,
            submitted_at=submitted_at,
            observed_at=observed_at,
            observed_fills=shadow_fills,
        )
        if fill_error is not None:
            return fill_error
        return None

    def _submitted_fill_event_error(
        self,
        event: BrokerEvent,
        *,
        submitted_order: Order | None,
        submitted_at: datetime | None,
        observed_at: datetime,
        observed_fills: dict[str, Fill],
    ) -> str | None:
        """Validate fill economics against any immutable guard-submitted order."""

        fill = event.fill
        if event.kind in _FILL_EVENT_KINDS and fill is None:
            return f"{event.kind.value} event has no Fill payload"
        if fill is None:
            return None
        if not isinstance(fill, Fill):
            return "fill payload is not a typed Fill"
        if event.kind not in _FILL_EVENT_KINDS | _TERMINAL_FILL_EVENT_KINDS:
            return f"{event.kind.value} event cannot carry fill economics"
        if submitted_order is None:
            return "fill references an order that this guard did not submit"
        if fill.order_id != event.order_id or fill.order_id != submitted_order.order_id:
            return "fill local order id does not match its event and submitted order"
        if (
            not isinstance(fill.fill_id, str)
            or not fill.fill_id
            or fill.fill_id != fill.fill_id.strip()
        ):
            return "fill id must be stable non-empty stripped text"
        prior_fill = observed_fills.get(fill.fill_id)
        if prior_fill is not None and prior_fill != fill:
            return "fill id was reused with different economic evidence"
        duplicate_fill = prior_fill is not None
        if fill.broker_fill_id is not None and (
            not isinstance(fill.broker_fill_id, str)
            or not fill.broker_fill_id
            or fill.broker_fill_id != fill.broker_fill_id.strip()
        ):
            return "broker fill id must be null or stable non-empty stripped text"
        if (
            not isinstance(fill.timestamp, datetime)
            or fill.timestamp.tzinfo is None
            or fill.timestamp.utcoffset() is None
        ):
            return "fill timestamp must be timezone-aware"
        if fill.timestamp > event.timestamp:
            return "fill timestamp cannot be later than its observation event"
        if fill.timestamp > observed_at + _MAX_BROKER_MESSAGE_CLOCK_SKEW:
            return "fill timestamp is implausibly future-dated"
        if (
            submitted_at is not None
            and fill.timestamp < submitted_at - _MAX_BROKER_MESSAGE_CLOCK_SKEW
        ):
            return "fill timestamp implausibly predates the order submission"
        if (
            not isinstance(fill.instrument, str)
            or not fill.instrument
            or fill.instrument != fill.instrument.strip()
            or fill.instrument != submitted_order.instrument
        ):
            return "fill instrument does not match the submitted order"
        if type(fill.side) is not Side or fill.side is not submitted_order.side:
            return "fill side does not match the submitted order"
        if (
            not isinstance(fill.quantity, int)
            or isinstance(fill.quantity, bool)
            or fill.quantity <= 0
        ):
            return "fill quantity must be a positive integer"
        remaining_quantity = submitted_order.remaining_quantity
        if not duplicate_fill and fill.quantity > remaining_quantity:
            return "fill cumulative quantity exceeds the submitted order quantity"
        if (
            isinstance(fill.price, bool)
            or not isinstance(fill.price, (int, float))
            or not math.isfinite(float(fill.price))
            or float(fill.price) <= 0
        ):
            return "fill price must be finite and positive"
        price = float(fill.price)
        tick_size = self._risk.instrument.tick_size
        tolerance = max(1e-9, abs(tick_size) * 1e-9)
        if abs(self._risk.instrument.round_to_tick(price) - price) > tolerance:
            return f"fill price is not aligned to the {tick_size}-point tick"
        if submitted_order.order_type is OrderType.LIMIT:
            limit = submitted_order.limit_price
            assert limit is not None
            if submitted_order.side is Side.BUY and price > limit + tolerance:
                return "BUY limit fill exceeds the submitted price envelope"
            if submitted_order.side is Side.SELL and price < limit - tolerance:
                return "SELL limit fill is below the submitted price envelope"
        for label, value in (
            ("commission", fill.commission_usd),
            ("slippage", fill.slippage_points),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                return f"fill {label} must be finite and non-negative"
        if type(fill.is_partial) is not bool:
            return "fill partial flag must be boolean"
        if event.kind is EventKind.PARTIAL_FILL:
            if not fill.is_partial:
                return "PARTIAL_FILL event must carry partial fill evidence"
            if not duplicate_fill and fill.quantity >= remaining_quantity:
                return "PARTIAL_FILL event completes the submitted order quantity"
        if event.kind is EventKind.FILL:
            if fill.is_partial:
                return "FILL event cannot carry partial fill evidence"
            if not duplicate_fill and fill.quantity != remaining_quantity:
                return "FILL event does not complete the submitted order quantity"
        return None

    @staticmethod
    def _advance_event_validation_shadow(
        event: BrokerEvent,
        *,
        shadow_orders: dict[str, Order],
        shadow_fills: dict[str, Fill],
    ) -> None:
        """Advance an isolated ledger so one batch cannot hide cumulative overfills."""

        broker_id = event.broker_order_id
        order = shadow_orders.get(broker_id)
        if order is None:
            return
        fill = event.fill
        if fill is not None and fill.fill_id not in shadow_fills:
            filled = order.filled_quantity + fill.quantity
            average = (
                (
                    order.average_fill_price * order.filled_quantity
                    + fill.price * fill.quantity
                )
                / filled
            )
            order = order.with_status(
                order.status,
                filled_quantity=filled,
                average_fill_price=average,
            )
            shadow_fills[fill.fill_id] = fill
        if event.kind is EventKind.FILL:
            order = order.with_status(OrderStatus.FILLED)
        elif event.kind is EventKind.PARTIAL_FILL:
            order = order.with_status(OrderStatus.PARTIALLY_FILLED)
        elif event.kind is EventKind.CANCEL_REQUESTED:
            order = order.with_status(OrderStatus.CANCEL_REQUESTED)
        elif event.kind is EventKind.CANCELLED:
            order = order.with_status(OrderStatus.CANCELLED)
        elif event.kind is EventKind.REJECTED:
            order = order.with_status(OrderStatus.REJECTED)
        shadow_orders[broker_id] = order

    def poll_events(self) -> list[BrokerEvent]:
        events = self._adapter.poll_events()
        self._observe_events_fail_safe(events)
        return events

    def on_bar(self, bar) -> list[BrokerEvent]:
        """Pass a closed bar to a simulating adapter. A no-op for a live one."""
        split_open = getattr(self._adapter, "on_bar_open", None)
        split_range = getattr(self._adapter, "on_bar_range", None)
        if split_open is not None and split_range is not None:
            events = [*split_open(bar), *split_range(bar)]
            self._observe_events_fail_safe(events)
            return events
        handler = getattr(self._adapter, "on_bar", None)
        events = handler(bar) if handler is not None else []
        self._observe_events_fail_safe(events)
        return events

    def on_bar_open(self, bar) -> list[BrokerEvent]:
        """Match orders executable at the bar open.

        The split phase is important in a backtest: an entry market order fills at the
        open, the execution engine attaches its protective bracket, and only then may the
        rest of this bar's range touch that bracket.
        """
        handler = getattr(self._adapter, "on_bar_open", None)
        events = handler(bar) if handler is not None else []
        self._observe_events_fail_safe(events)
        return events

    def on_bar_range(self, bar) -> list[BrokerEvent]:
        """Match resting price orders against the bar after open fills were applied."""
        handler = getattr(self._adapter, "on_bar_range", None)
        if handler is not None:
            events = handler(bar)
            self._observe_events_fail_safe(events)
            return events
        # Compatibility for adapters that only implement the original one-phase seam.
        fallback = getattr(self._adapter, "on_bar", None)
        events = fallback(bar) if fallback is not None else []
        self._observe_events_fail_safe(events)
        return events

    def _observe_events_fail_safe(self, events: object) -> None:
        try:
            observed_at = self._risk.clock.now()
            structural_error = self._broker_events_structural_error(
                events,
                observed_at=observed_at,
            )
            if structural_error is not None:
                # Persist OUTCOME_UNKNOWN before latching the store fault. The old durable
                # reservation would already block a second entry, but the stronger state
                # makes restart diagnosis explicit without trusting the malformed event's
                # own timestamp or ids.
                with self._entry_lock:
                    pending = self._pending_entry
                    if pending is not None:
                        self._mark_pending_entry(
                            pending.order_id,
                            state=PendingEntryState.OUTCOME_UNKNOWN,
                            now=observed_at,
                        )
                self._latch_broker_input_error(
                    "malformed broker event was withheld; " + structural_error
                )
                raise ReservationStoreError(self._reservation_store_error)
            assert isinstance(events, list)
            self._observe_events(events)
        except ReservationStoreError:
            self._latch_broker_input_error(
                self._reservation_store_error
                or "broker event could not cross the guarded trust boundary"
            )
            # A broker fill can be real even when its evidence cannot be written. The
            # normal execution path must not see the event first, but the account must not
            # remain exposed waiting for local Position reconstruction either. Cancel the
            # unfilled entry remainder before submitting the close: otherwise the entry
            # and emergency flatten can race on the next venue match and recreate
            # exposure immediately after the close.
            self._recover_after_untrusted_broker_input(now=self._risk.clock.now())
            raise

    def _observe_events(self, events: list[BrokerEvent]) -> None:
        with self._entry_lock:
            for event in events:
                pending = self._pending_entry
                if pending is None:
                    break
                matches = self._pending_event_matches(pending, event)
                if matches is None:
                    raise ReservationStoreError(
                        "broker event was withheld because its order identity conflicts "
                        "with the durable pending entry"
                    )
                if not matches:
                    if (
                        event.fill is not None
                        and event.fill.order_id == pending.order_id
                    ):
                        self._reservation_store_error = (
                            "pending-entry event identity conflict: embedded fill order "
                            "id matches the reservation while the event envelope does not"
                        )
                        raise ReservationStoreError(
                            self._reservation_store_error
                        )
                    continue
                if event.fill is not None or event.kind in {
                    EventKind.PARTIAL_FILL,
                    EventKind.FILL,
                }:
                    evidence_error = self._pending_fill_evidence_error(
                        pending,
                        event,
                    )
                    if evidence_error is not None:
                        self._reservation_store_error = (
                            "invalid pending-entry fill evidence; " + evidence_error
                        )
                        raise ReservationStoreError(
                            self._reservation_store_error
                        )
                if event.kind in {EventKind.REJECTED, EventKind.CANCELLED}:
                    filled_quantity = pending.cumulative_filled_quantity
                    observed_fill_ids = pending.observed_fill_ids
                    ever_fill_observed = pending.ever_fill_observed
                    if event.fill is not None:
                        fill_id = event.fill.fill_id
                        if fill_id not in observed_fill_ids:
                            filled_quantity += event.fill.quantity
                            observed_fill_ids = (*observed_fill_ids, fill_id)
                        ever_fill_observed = True
                    terminal = replace(
                        pending,
                        broker_order_id=(
                            pending.broker_order_id
                            or event.broker_order_id
                            or None
                        ),
                        state=self._monotonic_pending_state(
                            pending,
                            PendingEntryState.TERMINAL_REPORTED,
                            cumulative_filled_quantity=filled_quantity,
                        ),
                        updated_at=max(pending.updated_at, event.timestamp),
                        cumulative_filled_quantity=filled_quantity,
                        ever_fill_observed=ever_fill_observed,
                        observed_fill_ids=observed_fill_ids,
                    )
                    if not self._set_pending_entry(terminal):
                        raise ReservationStoreError(
                            "terminal entry event was withheld because its durable "
                            "reservation could not be advanced"
                        )
                    absorbed = True
                    if terminal.ever_fill_observed:
                        absorbed = self._risk.on_entry_fill_observed(
                            pending.order_id,
                            now=event.timestamp,
                        )
                    elif event.kind is EventKind.REJECTED:
                        # A venue rejection proves the order never became executable.
                        # Cancellation does not: it may follow one or more fills and must
                        # wait for exact order history plus a flat snapshot.
                        absorbed = self._risk.on_entry_terminal_unfilled(
                            pending.order_id,
                            now=event.timestamp,
                        )
                    if not absorbed:
                        raise ReservationStoreError(
                            "terminal entry event was withheld because durable "
                            "personal-risk state could not absorb it"
                        )
                    # Release only if a separate fresh collection read now returns this
                    # exact order terminal and the account flat.  Missing/pruned history
                    # preserves the reservation and personal active-entry lock.
                    self._reconcile_terminal_pending_entry(now=event.timestamp)
                    continue
                if event.kind is EventKind.CANCEL_REQUESTED:
                    pending = replace(
                        pending,
                        broker_order_id=(
                            pending.broker_order_id
                            or event.broker_order_id
                            or None
                        ),
                        state=self._monotonic_pending_state(
                            pending,
                            PendingEntryState.CANCEL_REQUESTED,
                            cumulative_filled_quantity=(
                                pending.cumulative_filled_quantity
                            ),
                        ),
                        updated_at=max(pending.updated_at, event.timestamp),
                    )
                    if not self._set_pending_entry(pending):
                        raise ReservationStoreError(
                            "entry cancellation event was withheld because its durable "
                            "reservation could not be advanced"
                        )
                    continue
                if event.kind not in {EventKind.PARTIAL_FILL, EventKind.FILL}:
                    continue

                # Fill-bearing lifecycle events cannot reach this branch without complete
                # evidence validated against the immutable signed envelope above.
                assert event.fill is not None
                fill_id = event.fill.fill_id
                fill_delta = (
                    event.fill.quantity
                    if fill_id not in pending.observed_fill_ids
                    else 0
                )

                observed_fill_ids = pending.observed_fill_ids
                if fill_id not in observed_fill_ids:
                    observed_fill_ids = (*observed_fill_ids, fill_id)
                filled_quantity = (
                    pending.cumulative_filled_quantity + fill_delta
                )
                candidate = (
                    PendingEntryState.FILLED
                    if event.kind is EventKind.FILL
                    or filled_quantity >= pending.quantity
                    else PendingEntryState.PARTIALLY_FILLED
                )
                advanced = replace(
                    pending,
                    broker_order_id=(
                        pending.broker_order_id
                        or event.broker_order_id
                        or None
                    ),
                    state=self._monotonic_pending_state(
                        pending,
                        candidate,
                        cumulative_filled_quantity=filled_quantity,
                    ),
                    updated_at=max(pending.updated_at, event.timestamp),
                    cumulative_filled_quantity=filled_quantity,
                    ever_fill_observed=True,
                    observed_fill_ids=observed_fill_ids,
                )
                if not self._set_pending_entry(advanced):
                    raise ReservationStoreError(
                        "entry fill was withheld because its durable reservation could "
                        "not record the fill evidence"
                    )
                if not self._risk.on_entry_fill_observed(
                    pending.order_id,
                    now=event.timestamp,
                ):
                    raise ReservationStoreError(
                        "entry fill was withheld because durable personal-risk state "
                        "could not absorb it"
                    )

        # The execution-facing in-memory order ledger advances only after pending-entry
        # fill evidence and personal quota consumption are safely recorded. The caller
        # cannot observe these events until this method returns.
        for event in events:
            self._observe_submitted_order(event)

    def _observe_submitted_order(self, event: BrokerEvent) -> None:
        with self._order_lock:
            broker_id = event.broker_order_id or self._broker_ids_by_order_id.get(
                event.order_id, ""
            )
            if not broker_id:
                return
            order = self._submitted_orders.get(broker_id)
            if order is None:
                return

            if event.fill is not None:
                prior_fill = self._observed_fills_by_id.get(event.fill.fill_id)
                if prior_fill is None:
                    filled = order.filled_quantity + event.fill.quantity
                    average = (
                        (
                            order.average_fill_price * order.filled_quantity
                            + event.fill.price * event.fill.quantity
                        )
                        / filled
                    )
                    order = order.with_status(
                        order.status,
                        filled_quantity=filled,
                        average_fill_price=average,
                    )
                    self._submitted_orders[broker_id] = order
                    self._observed_fills_by_id[event.fill.fill_id] = event.fill

            if event.kind is EventKind.ACCEPTED:
                status = OrderStatus.ACCEPTED
            elif event.kind is EventKind.CANCEL_REQUESTED:
                status = OrderStatus.CANCEL_REQUESTED
            elif event.kind is EventKind.CANCELLED:
                status = OrderStatus.CANCELLED
            elif event.kind is EventKind.REJECTED:
                status = OrderStatus.REJECTED
            elif event.kind in (EventKind.FILL, EventKind.PARTIAL_FILL) and event.fill:
                status = (
                    OrderStatus.FILLED
                    if event.kind is EventKind.FILL
                    or order.filled_quantity >= order.quantity
                    else OrderStatus.PARTIALLY_FILLED
                )
                self._submitted_orders[broker_id] = order.with_status(
                    status,
                )
                return
            else:
                return
            self._submitted_orders[broker_id] = order.with_status(status)
