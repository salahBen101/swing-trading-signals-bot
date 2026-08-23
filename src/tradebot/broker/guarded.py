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

from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock

from ..core.models import Order, Rejection
from ..core.types import OrderPurpose, OrderStatus, RejectReason, Side
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
    PendingEntryState.TERMINAL_REPORTED,
    PendingEntryState.FILLED,
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
        capabilities = getattr(adapter, "capabilities", None)
        if not isinstance(capabilities, BrokerCapabilities):
            raise LiveTradingDisabled(
                "broker adapter does not declare its execution-safety capabilities"
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
        self._pending_entry: PendingEntryReservation | None = None
        self._reservation_store = reservation_store
        self._reservation_store_error: str | None = None
        if reservation_store is not None:
            try:
                restored = reservation_store.load()
                if restored is not None:
                    if not isinstance(restored, PendingEntryReservation):
                        raise TypeError(
                            "reservation store returned a non-reservation value"
                        )
                    restored.validate()
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
                reconciliation = self._risk.reconcile_broker_snapshot(snapshot, now=now)
                if reconciliation is not None:
                    return self._refuse(reconciliation)
            self._reconcile_pending_entry(snapshot, now=now)
            if self._reservation_store_error is not None:
                return self._refuse(self._reservation_error_rejection(order, now=now))
            if self._pending_entry is not None:
                pending = self._pending_entry
                if pending.order_id == order.order_id:
                    # Preserve the stronger signed-token diagnosis for an exact replay.
                    # A different order cannot take this branch and remains blocked by
                    # the reservation without spending its still-valid approval.
                    replay_rejection = self._risk.verify(
                        order, token, now=now, broker_snapshot=snapshot
                    )
                    if replay_rejection is not None:
                        return self._refuse(replay_rejection)
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
            )
            if not self._set_pending_entry(reservation):
                return self._refuse(self._reservation_error_rejection(order, now=now))
            return self._submit_verified(order, now=now, is_entry=True)

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

        if is_entry and self._pending_entry is not None:
            if ack.status in {
                OrderStatus.REJECTED,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
            }:
                state = PendingEntryState.TERMINAL_REPORTED
            elif ack.status is OrderStatus.FILLED:
                state = PendingEntryState.FILLED
            elif ack.status is OrderStatus.PARTIALLY_FILLED:
                state = PendingEntryState.PARTIALLY_FILLED
            else:
                state = PendingEntryState.ACKNOWLEDGED
            self._mark_pending_entry(
                order.order_id,
                broker_order_id=ack.broker_order_id,
                state=state,
                now=now,
            )
        canonical = order.with_status(
            ack.status,
            broker_order_id=ack.broker_order_id,
        )
        with self._order_lock:
            self._submitted_orders[ack.broker_order_id] = canonical
            self._broker_ids_by_order_id[order.order_id] = ack.broker_order_id
        self.submitted_count += 1
        return GuardResult(ack=ack, rejection=None)

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
        matches = tuple(
            item
            for item in snapshot.orders
            if item.order_id == pending.order_id
            or (
                pending.broker_order_id is not None
                and item.broker_order_id == pending.broker_order_id
            )
        )
        working = tuple(item for item in matches if item.is_working)
        if working:
            state = (
                PendingEntryState.PARTIALLY_FILLED
                if any(item.filled_quantity > 0 for item in working)
                else PendingEntryState.ACKNOWLEDGED
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
            ))
            return
        has_exposure = any(position.quantity != 0 for position in snapshot.positions)
        if matches:
            # The complete authoritative order read says this order is terminal.  It is
            # still unsafe to release the reservation if the same snapshot shows any
            # position: the entry may have filled before becoming terminal.
            if has_exposure:
                state = (
                    PendingEntryState.FILLED
                    if any(item.filled_quantity > 0 for item in matches)
                    else PendingEntryState.TERMINAL_REPORTED
                )
                self._set_pending_entry(replace(
                    pending,
                    state=state,
                    updated_at=max(pending.updated_at, now),
                ))
            else:
                self._clear_pending_entry_after_snapshot(snapshot)
            return
        if has_exposure:
            self._set_pending_entry(replace(
                pending,
                state=PendingEntryState.FILLED,
                updated_at=max(pending.updated_at, now),
            ))
            return
        if pending.state not in {
            PendingEntryState.SUBMITTING,
            PendingEntryState.OUTCOME_UNKNOWN,
        }:
            # An acknowledged/cancel-requested order absent from a complete authoritative
            # order read is terminal and the same snapshot proved the account flat.
            # Unknown submissions remain locked for human review unless the venue later
            # returns an explicit terminal record.
            self._clear_pending_entry_after_snapshot(snapshot)

    def _set_pending_entry(self, reservation: PendingEntryReservation) -> bool:
        """Update memory and then the durable copy before any later unsafe action."""
        self._pending_entry = reservation
        if self._reservation_store is None:
            return True
        if self._reservation_store_error is not None:
            return False
        try:
            self._reservation_store.save(reservation)
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
    ) -> None:
        pending = self._pending_entry
        if pending is None or pending.order_id != order_id:
            return
        self._set_pending_entry(replace(
            pending,
            broker_order_id=(
                broker_order_id
                if broker_order_id is not None
                else pending.broker_order_id
            ),
            state=state,
            updated_at=max(pending.updated_at, now),
        ))

    def _clear_pending_entry_after_snapshot(
        self, snapshot: AuthoritativeBrokerSnapshot
    ) -> bool:
        """Clear only when this exact authoritative read proves terminal and flat."""
        pending = self._pending_entry
        if pending is None:
            return True
        if any(position.quantity != 0 for position in snapshot.positions):
            return False
        matches = tuple(
            item
            for item in snapshot.orders
            if item.order_id == pending.order_id
            or (
                pending.broker_order_id is not None
                and item.broker_order_id == pending.broker_order_id
            )
        )
        terminal_proven = (
            bool(matches) and not any(item.is_working for item in matches)
        ) or (
            not matches
            and pending.state
            not in {PendingEntryState.SUBMITTING, PendingEntryState.OUTCOME_UNKNOWN}
        )
        if not terminal_proven:
            return False
        if self._reservation_store is not None:
            if self._reservation_store_error is not None:
                return False
            try:
                self._reservation_store.clear()
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
                        state=PendingEntryState.CANCEL_REQUESTED,
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

        self._adapter.cancel_order(broker_order_id)
        with self._entry_lock:
            pending = self._pending_entry
            if pending is not None and pending.broker_order_id == broker_order_id:
                self._set_pending_entry(replace(
                    pending,
                    state=PendingEntryState.CANCEL_REQUESTED,
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

    def poll_events(self) -> list[BrokerEvent]:
        events = self._adapter.poll_events()
        self._observe_events(events)
        return events

    def on_bar(self, bar) -> list[BrokerEvent]:
        """Pass a closed bar to a simulating adapter. A no-op for a live one."""
        split_open = getattr(self._adapter, "on_bar_open", None)
        split_range = getattr(self._adapter, "on_bar_range", None)
        if split_open is not None and split_range is not None:
            events = [*split_open(bar), *split_range(bar)]
            self._observe_events(events)
            return events
        handler = getattr(self._adapter, "on_bar", None)
        events = handler(bar) if handler is not None else []
        self._observe_events(events)
        return events

    def on_bar_open(self, bar) -> list[BrokerEvent]:
        """Match orders executable at the bar open.

        The split phase is important in a backtest: an entry market order fills at the
        open, the execution engine attaches its protective bracket, and only then may the
        rest of this bar's range touch that bracket.
        """
        handler = getattr(self._adapter, "on_bar_open", None)
        events = handler(bar) if handler is not None else []
        self._observe_events(events)
        return events

    def on_bar_range(self, bar) -> list[BrokerEvent]:
        """Match resting price orders against the bar after open fills were applied."""
        handler = getattr(self._adapter, "on_bar_range", None)
        if handler is not None:
            events = handler(bar)
            self._observe_events(events)
            return events
        # Compatibility for adapters that only implement the original one-phase seam.
        fallback = getattr(self._adapter, "on_bar", None)
        events = fallback(bar) if fallback is not None else []
        self._observe_events(events)
        return events

    def _observe_events(self, events: list[BrokerEvent]) -> None:
        for event in events:
            self._observe_submitted_order(event)
        with self._entry_lock:
            pending = self._pending_entry
            if pending is None:
                return
            for event in events:
                matches = event.order_id == pending.order_id or (
                    pending.broker_order_id is not None
                    and event.broker_order_id == pending.broker_order_id
                )
                if not matches:
                    continue
                if event.kind in {EventKind.REJECTED, EventKind.CANCELLED}:
                    self._set_pending_entry(replace(
                        pending,
                        state=PendingEntryState.TERMINAL_REPORTED,
                        updated_at=max(pending.updated_at, event.timestamp),
                    ))
                    return
                if event.kind is EventKind.CANCEL_REQUESTED:
                    pending = replace(
                        pending,
                        state=PendingEntryState.CANCEL_REQUESTED,
                        updated_at=max(pending.updated_at, event.timestamp),
                    )
                    self._set_pending_entry(pending)
                if event.kind is EventKind.PARTIAL_FILL:
                    pending = replace(
                        pending,
                        state=PendingEntryState.PARTIALLY_FILLED,
                        updated_at=max(pending.updated_at, event.timestamp),
                    )
                    self._set_pending_entry(pending)
                elif event.kind is EventKind.FILL:
                    pending = replace(
                        pending,
                        state=PendingEntryState.FILLED,
                        updated_at=max(pending.updated_at, event.timestamp),
                    )
                    self._set_pending_entry(pending)

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

            if event.kind is EventKind.ACCEPTED:
                status = OrderStatus.ACCEPTED
            elif event.kind is EventKind.CANCEL_REQUESTED:
                status = OrderStatus.CANCEL_REQUESTED
            elif event.kind is EventKind.CANCELLED:
                status = OrderStatus.CANCELLED
            elif event.kind is EventKind.REJECTED:
                status = OrderStatus.REJECTED
            elif event.kind in (EventKind.FILL, EventKind.PARTIAL_FILL) and event.fill:
                filled = min(order.quantity, order.filled_quantity + event.fill.quantity)
                average = (
                    (
                        order.average_fill_price * order.filled_quantity
                        + event.fill.price * event.fill.quantity
                    )
                    / filled
                )
                status = (
                    OrderStatus.FILLED
                    if event.kind is EventKind.FILL or filled >= order.quantity
                    else OrderStatus.PARTIALLY_FILLED
                )
                self._submitted_orders[broker_id] = order.with_status(
                    status,
                    filled_quantity=filled,
                    average_fill_price=average,
                )
                return
            else:
                return
            self._submitted_orders[broker_id] = order.with_status(status)
