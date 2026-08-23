"""The execution engine.

One bar at a time, the same code path in a backtest and in a paper run. Only the broker and
the feed differ between them, which is this project's main structural defence against
backtest/live divergence.

**Per-bar order of operations**, which is where all the subtlety lives:

1. **Match.** Hand the closed bar to the broker so resting orders can fill against its
   range. Protective stops rest *at the venue*, so a bar that gaps through one fills at the
   gapped open, not at the level.
2. **Apply events.** Fills update the position; a completed exit closes the trade and
   informs the risk engine and the strategy.
3. **Force.** Ask the risk engine whether the open position must be closed now — kill
   switch, marked daily loss, trailing floor, or the flatten deadline. This runs before the
   strategy so a strategy can never talk the engine out of a mandatory exit.
4. **Manage.** Let the strategy adjust the stop or request a discretionary exit.
5. **Enter.** Only if flat and with no order in flight, ask the strategy for an intent, run
   it through the risk engine, and submit through the guard.
6. **Mark.** Record equity.

A signal produced on bar *i* is submitted after bar *i* closes and can only fill on bar
*i+1* or later, because the broker will not match an order whose `releasable_at` has not
been reached and never re-matches the bar that is already being processed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..broker.base import BrokerError, BrokerEvent, EventKind, NotConnected
from ..broker.costs import CostModel
from ..broker.guarded import GuardedBroker
from ..core.models import (
    AccountSnapshot,
    Bar,
    Fill,
    Order,
    OrderIntent,
    Position,
    Rejection,
    Trade,
    new_id,
)
from ..core.types import (
    BotState,
    ExitReason,
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
)
from ..instruments.registry import InstrumentSpec
from ..journal.db import Journal
from ..risk.limits import RiskEngine
from ..strategy.base import ExitDecision, Strategy, StrategyContext, StopUpdate


@dataclass(slots=True)
class PendingExit:
    """The reason an exit was requested, held until the closing fill arrives."""

    reason: ExitReason
    detail: str = ""


@dataclass(slots=True)
class ExecutionState:
    position: Position | None = None
    entry_order: Order | None = None
    stop_order: Order | None = None
    target_order: Order | None = None
    pending_exit: PendingExit | None = None
    bot_state: BotState = BotState.STARTING
    trades: list[Trade] = field(default_factory=list)
    equity: float = 0.0
    last_bar: Bar | None = None
    bars_seen: int = 0

    @property
    def is_flat(self) -> bool:
        return self.position is None

    @property
    def has_order_in_flight(self) -> bool:
        return self.entry_order is not None


class ExecutionEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk: RiskEngine,
        broker: GuardedBroker,
        instrument: InstrumentSpec,
        costs: CostModel,
        *,
        journal: Journal | None = None,
        starting_equity: float = 50_000.0,
        log_every_bar: bool = False,
    ) -> None:
        self.strategy = strategy
        self.risk = risk
        self.broker = broker
        self.instrument = instrument
        self.costs = costs
        self.journal = journal
        self.log_every_bar = log_every_bar

        self.state = ExecutionState(equity=starting_equity)
        self.rejections: list[Rejection] = []
        self._orders: dict[str, Order] = {}
        self._broker_ids: dict[str, str] = {}  # order_id -> broker_order_id
        # Intents are kept until their entry fills, because the stop and target levels live
        # on the intent and the protective orders are only placed once the fill price is
        # known. Cleared on fill so a long run does not accumulate them.
        self._intents: dict[str, OrderIntent] = {}

    # ================================================================== main loop

    def on_bar(
        self, bar: Bar, features: pd.Series, *, index: int, frame: pd.DataFrame,
        bars: pd.DataFrame, warmed_up: bool = True,
    ) -> None:
        self.state.bars_seen += 1
        self.state.last_bar = bar
        self.risk.roll_session(bar.timestamp)

        if self.log_every_bar and self.journal is not None:
            self.journal.record_bar(bar, self.instrument.symbol, features.to_dict())

        # 1-2. Fill market orders at the open, apply those fills (which attaches a new
        # protective bracket), then match stops/limits against the remainder of the same
        # bar. A one-phase matcher would miss an entry-bar stop and flatter the backtest.
        #
        # Both phases queue what they produce. Events are drained after each phase and in
        # exactly one place, because applying the same fill twice silently doubles the
        # position while every individual component still looks correct.
        self.broker.on_bar_open(bar)
        self._apply_events(self.broker.poll_events())
        self.broker.on_bar_range(bar)
        self._apply_events(self.broker.poll_events())

        ctx = self._context(bar, features, index, frame, bars)

        # 3. Mandatory exits, before the strategy gets a say.
        if self.state.position is not None:
            forced = self.risk.forced_exit_reason(self.state.position, bar.close, bar.timestamp)
            if forced is not None:
                reason_code, detail = forced
                self._flatten(bar, _exit_reason_for(reason_code), detail)
                return

        # 4. Strategy management of an open position.
        if self.state.position is not None:
            decision = self.strategy.manage(ctx, self.state.position)
            if isinstance(decision, StopUpdate):
                self._move_stop(bar, decision.new_stop, decision.detail)
            elif isinstance(decision, ExitDecision):
                self._flatten(bar, decision.reason, decision.detail)
            self._mark(bar)
            return

        # 5. A new entry, only when genuinely flat and idle.
        if warmed_up and not self.state.has_order_in_flight:
            self._try_entry(ctx, bar)

        self._mark(bar)

    def _context(
        self, bar: Bar, features: pd.Series, index: int, frame: pd.DataFrame,
        bars: pd.DataFrame,
    ) -> StrategyContext:
        return StrategyContext(
            i=index,
            timestamp=bar.timestamp,
            bar=bars.iloc[index],
            features=features,
            instrument=self.instrument,
            equity=self.state.equity,
            session_date=bar.timestamp.date(),
            trades_this_session=self.strategy.trades_this_session,
            _frame=frame,
            _bars=bars,
        )

    # ================================================================== entries

    def _try_entry(self, ctx: StrategyContext, bar: Bar) -> None:
        intent = self.strategy.on_bar(ctx)
        self._absorb(ctx.rejections)
        if intent is None:
            return

        if self.journal is not None:
            self.journal.record_signal(intent)

        decision = self.risk.evaluate_entry(
            intent, equity=self.state.equity, position=self.state.position,
            now=bar.timestamp,
        )
        # Coordinated risk decisions expose an intentionally token-free ``to_dict``
        # audit representation.  Persist it before acting on either branch so an
        # accepted entry and a fail-closed rejection are equally observable.  Legacy
        # personal-only RiskDecision objects do not expose this method and retain their
        # existing journal behavior.
        decision_payload = getattr(decision, "to_dict", None)
        if self.journal is not None and callable(decision_payload):
            payload = decision_payload()
            if not isinstance(payload, dict):
                raise TypeError("risk decision to_dict() must return a dictionary")
            self.journal.record_event(
                bar.timestamp,
                "RISK_DECISION",
                f"ENTRY_EVALUATE {'ALLOWED' if decision.approved else 'REFUSED'}",
                level="INFO" if decision.approved else "WARN",
                payload=payload,
            )
        if not decision.approved:
            self._absorb([decision.rejection])
            return

        approval = decision.approval
        self._register(approval.order)
        result = self.broker.place_order(approval.order, approval.token, now=bar.timestamp)
        if not result.accepted:
            self._absorb([result.rejection])
            self._update_order(approval.order, OrderStatus.REJECTED, bar.timestamp,
                               detail=result.rejection.detail)
            return

        self.state.entry_order = approval.order
        self._intents[intent.intent_id] = intent
        self._broker_ids[approval.order.order_id] = result.ack.broker_order_id
        self._update_order(approval.order, OrderStatus.ACCEPTED, bar.timestamp,
                           broker_order_id=result.ack.broker_order_id,
                           detail=approval.detail)
        self._event(bar.timestamp, "ENTRY_SUBMITTED",
                    f"{approval.order.side.name} {approval.order.quantity} "
                    f"{self.instrument.symbol}: {approval.detail}")

    # ================================================================== fills

    def _apply_events(self, events: list[BrokerEvent]) -> None:
        for event in events:
            if event.kind in (EventKind.FILL, EventKind.PARTIAL_FILL) and event.fill:
                self._on_fill(event.fill)
            elif event.kind is EventKind.REJECTED:
                self.state.entry_order = None
                self._event(event.timestamp, "ORDER_REJECTED", event.detail, level="WARN")
            elif event.kind is EventKind.CANCELLED:
                self._on_cancelled(event)
            elif event.kind is EventKind.DISCONNECTED:
                self._event(event.timestamp, "DISCONNECTED", event.detail, level="ERROR")

    def _on_fill(self, fill: Fill) -> None:
        if self.journal is not None:
            self.journal.record_fill(fill)

        order = self._orders.get(fill.order_id)
        if order is None:
            # A fill for an order this process did not place. That is a reconciliation
            # problem, not something to trade around, so it is recorded loudly and the
            # position is left to `reconcile()`.
            self._event(fill.timestamp, "ORPHAN_FILL",
                        f"fill {fill.fill_id} references unknown order {fill.order_id}",
                        level="ERROR")
            return

        prior_filled = order.filled_quantity
        filled = prior_filled + fill.quantity
        status = OrderStatus.FILLED if filled >= order.quantity else OrderStatus.PARTIALLY_FILLED
        average_fill_price = (
            (order.average_fill_price * prior_filled + fill.price * fill.quantity) / filled
        )
        updated = order.with_status(status, filled_quantity=filled,
                                    average_fill_price=average_fill_price)
        self._orders[order.order_id] = updated
        self._update_order(updated, status, fill.timestamp,
                           detail=f"{fill.quantity} @ {fill.price}")

        if order.purpose is OrderPurpose.ENTRY:
            self.state.entry_order = updated
            self._on_entry_fill(updated, fill)
        else:
            self._on_exit_fill(updated, fill)

    def _on_entry_fill(self, order: Order, fill: Fill) -> None:
        # Commission is cash leaving the account at the fill, not at some later close.
        # Debiting it now keeps marked equity, the broker account and completed Trade P&L
        # on the same basis throughout the position lifecycle.
        self.state.equity -= fill.commission_usd
        slippage_usd = (
            fill.slippage_points * fill.quantity * self.instrument.multiplier
        )
        position = self.state.position
        if position is None:
            position = Position(
                instrument=fill.instrument,
                side=fill.side,
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_time=fill.timestamp,
                strategy=order.strategy,
                initial_stop=0.0,
                stop_price=0.0,
                entry_commission_usd=fill.commission_usd,
                entry_slippage_usd=slippage_usd,
                entry_intent_id=order.intent_id,
            )
            self.state.position = position
            self._attach_protection(order, position, fill.timestamp)
            # A partial entry is already real exposure and already consumes the daily
            # trade quota. Waiting for the final child fill would leave it invisible to
            # the personal-risk layer.
            self.risk.on_position_opened(position)
            self.strategy.on_trade_opened(position)
            self.state.bot_state = BotState.IN_POSITION
        else:
            # Adding on a partial: re-average and widen the protective orders to match.
            total = position.quantity + fill.quantity
            position.entry_price = (
                position.entry_price * position.quantity + fill.price * fill.quantity
            ) / total
            position.quantity = total
            position.entered_quantity += fill.quantity
            position.entry_commission_usd += fill.commission_usd
            position.entry_slippage_usd += slippage_usd
            position.risk_per_contract_points = abs(
                position.entry_price - position.initial_stop
            )
            self._resize_protection(position, fill.timestamp)

        if order.status is OrderStatus.FILLED:
            self.state.entry_order = None
            if order.intent_id:
                self._intents.pop(order.intent_id, None)
            self._event(fill.timestamp, "POSITION_OPENED",
                        f"{position.side.name} {position.quantity} @ {position.entry_price} "
                        f"stop {position.stop_price} target {position.target_price}")

    def _on_exit_fill(self, order: Order, fill: Fill) -> None:
        position = self.state.position
        if position is None:
            self._event(fill.timestamp, "ORPHAN_EXIT",
                        f"exit fill with no open position: {fill.fill_id}", level="ERROR")
            return

        if fill.quantity > position.quantity:
            raise RuntimeError(
                f"exit fill {fill.fill_id} closes {fill.quantity} contracts but only "
                f"{position.quantity} are open"
            )

        gross = self.costs.pnl_usd(position.entry_price, fill.price, fill.quantity,
                                   position.side)
        position.closed_quantity += fill.quantity
        position.realized_gross_pnl_usd += gross
        position.exit_commission_usd += fill.commission_usd
        position.exit_slippage_usd += (
            fill.slippage_points * fill.quantity * self.instrument.multiplier
        )
        position.exit_notional += fill.price * fill.quantity
        self.state.equity += gross - fill.commission_usd

        # Any protective exit means the premise has already been invalidated. Cancel an
        # unfilled entry remainder immediately; allowing it to add contracts on the next
        # bar would be an accidental average-down/re-entry after the stop was touched.
        if self.state.entry_order is not None:
            pending_entry = self.state.entry_order
            self._cancel(pending_entry, fill.timestamp)
            self.state.entry_order = None
            if pending_entry.intent_id:
                self._intents.pop(pending_entry.intent_id, None)

        remaining = position.quantity - fill.quantity
        if remaining > 0:
            # Scaled out: keep the position open with the reduced size.
            position.quantity = remaining
            self._resize_protection(position, fill.timestamp)
            self._event(fill.timestamp, "SCALE_OUT",
                        f"{fill.quantity} @ {fill.price}, {remaining} remaining")
            return

        reason = self._exit_reason_for_order(order)
        commission = position.entry_commission_usd + position.exit_commission_usd
        gross_total = position.realized_gross_pnl_usd
        net = gross_total - commission
        risk_usd = (
            position.risk_per_contract_points
            * position.entered_quantity
            * self.instrument.multiplier
        )
        average_exit_price = position.exit_notional / position.closed_quantity

        trade = Trade(
            trade_id=new_id("trd"),
            instrument=position.instrument,
            strategy=position.strategy,
            side=position.side,
            quantity=position.entered_quantity,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=fill.timestamp,
            exit_price=average_exit_price,
            exit_reason=reason,
            gross_pnl_usd=gross_total,
            commission_usd=commission,
            net_pnl_usd=net,
            r_multiple=(net / risk_usd) if risk_usd > 0 else 0.0,
            bars_held=position.bars_held,
            initial_stop=position.initial_stop,
            target_price=position.target_price,
            max_favorable_excursion_points=abs(
                position.max_favorable_price - position.entry_price
            ),
            max_adverse_excursion_points=abs(
                position.max_adverse_price - position.entry_price
            ),
            entry_conditions=position.entry_conditions,
            entry_features=position.entry_features,
            position_id=position.position_id,
            slippage_usd=(
                position.entry_slippage_usd + position.exit_slippage_usd
            ),
        )

        self.state.trades.append(trade)
        self.state.position = None
        self.state.pending_exit = None
        self.state.bot_state = BotState.RUNNING

        self._cancel_protection(fill.timestamp)
        self.risk.on_trade_closed(trade)
        self.strategy.on_trade_closed(trade)

        if self.journal is not None:
            self.journal.record_trade(trade)
        self._event(fill.timestamp, "TRADE_CLOSED",
                    f"{reason.value} {trade.net_pnl_usd:+.2f} USD ({trade.r_multiple:+.2f}R)")

    def _on_cancelled(self, event: BrokerEvent) -> None:
        for slot in ("stop_order", "target_order"):
            order = getattr(self.state, slot)
            if order is not None and order.order_id == event.order_id:
                setattr(self.state, slot, None)
                self._update_order(order, OrderStatus.CANCELLED, event.timestamp,
                                   detail=event.detail)
        if self.state.entry_order is not None and self.state.entry_order.order_id == event.order_id:
            self.state.entry_order = None

    # ================================================================== protective orders

    def _attach_protection(self, entry: Order, position: Position, ts: datetime) -> None:
        """Place the resting stop and target as an OCO pair, immediately on entry.

        Doing this at the venue rather than in this process is what makes a crash survivable:
        a resting stop is still working when the bot is not.
        """
        intent = self._intent_for(entry)
        if intent is None:
            # No recorded intent (recovery, or a manual position). Fall back to a stop
            # derived from nothing is not acceptable, so the position is flattened instead.
            self._event(ts, "NO_PROTECTION",
                        "cannot derive a stop for this position; flattening", level="ERROR")
            self._flatten_unprotected(ts, "entry intent is missing; no stop can be derived")
            return

        position.initial_stop = intent.stop_price
        position.stop_price = intent.stop_price
        position.target_price = intent.target_price
        position.risk_per_contract_points = abs(position.entry_price - intent.stop_price)
        position.entry_conditions = intent.conditions
        position.entry_features = dict(intent.features)
        position.max_hold_bars = intent.max_hold_bars

        group = new_id("oco")
        if self._place_stop(position, group, ts) is None:
            self._flatten_unprotected(ts, "initial protective stop was not accepted")
            return
        if intent.target_price is not None:
            self._place_target(position, group, ts)

    def _place_stop(self, position: Position, group: str, ts: datetime) -> Order | None:
        order = self._protective_order(position, OrderType.STOP, position.stop_price,
                                       OrderPurpose.STOP, group, ts)
        if order is not None:
            self.state.stop_order = order
        return order

    def _place_target(self, position: Position, group: str, ts: datetime) -> Order | None:
        order = self._protective_order(position, OrderType.LIMIT, position.target_price,
                                       OrderPurpose.TARGET, group, ts)
        if order is not None:
            self.state.target_order = order
        return order

    def _protective_order(
        self,
        position: Position,
        order_type: OrderType,
        price: float,
        purpose: OrderPurpose,
        group: str, ts: datetime,
    ) -> Order | None:
        order = Order(
            order_id=new_id("ord"),
            timestamp=ts,
            instrument=position.instrument,
            side=position.side.opposite,
            quantity=position.quantity,
            order_type=order_type,
            limit_price=price if order_type is OrderType.LIMIT else None,
            stop_price=price if order_type is OrderType.STOP else None,
            strategy=position.strategy,
            purpose=purpose,
            oco_group=group,
        )
        decision = self.risk.evaluate_protective(position, order, now=ts)
        if not decision.approved:
            self._absorb([decision.rejection])
            self._event(
                ts,
                "PROTECTION_REFUSED",
                f"{purpose.value}: {decision.rejection.detail}",
                level="ERROR",
            )
            return None
        approval = decision.approval
        self._register(order)
        try:
            result = self.broker.place_order(order, approval.token, now=ts)
        except BrokerError as exc:
            self._event(
                ts, "PROTECTION_FAILED", f"{purpose.value}: {exc}", level="ERROR"
            )
            return None

        if not result.accepted:
            self._absorb([result.rejection])
            self._event(ts, "PROTECTION_REFUSED",
                        f"{purpose.value}: {result.rejection.detail}", level="ERROR")
            return None

        self._broker_ids[order.order_id] = result.ack.broker_order_id
        self._update_order(order, OrderStatus.ACCEPTED, ts,
                           broker_order_id=result.ack.broker_order_id, detail=purpose.value)
        return order

    def _move_stop(self, bar: Bar, new_stop: float, detail: str) -> None:
        position = self.state.position
        if position is None:
            return

        old_price = position.stop_price
        widening = (
            position.side is Side.BUY and new_stop < old_price - 1e-9
        ) or (
            position.side is Side.SELL and new_stop > old_price + 1e-9
        )
        if widening:
            self._event(
                bar.timestamp,
                "STOP_MOVE_REFUSED",
                f"refused widening stop {old_price} -> {new_stop} ({detail})",
                level="WARN",
            )
            return
        if abs(new_stop - old_price) <= 1e-9:
            return

        old = self.state.stop_order
        group = (
            old.oco_group
            if old is not None
            else (
                self.state.target_order.oco_group
                if self.state.target_order is not None
                else new_id("oco")
            )
        )
        # Submit and receive an acknowledgement before touching the old stop.  The broker
        # interface has no atomic replace operation, so brief overlap is safer than a gap.
        replacement = self._protective_order(
            position,
            OrderType.STOP,
            new_stop,
            OrderPurpose.STOP,
            group or new_id("oco"),
            bar.timestamp,
        )
        if replacement is None:
            self._event(
                bar.timestamp,
                "STOP_REPLACE_FAILED",
                f"retained stop {old_price}; replacement {new_stop} was not accepted",
                level="ERROR",
            )
            if old is None:
                self._flatten_unprotected(
                    bar.timestamp, "stop replacement failed and no prior stop exists"
                )
            return

        self.state.stop_order = replacement
        position.stop_price = new_stop
        if old is not None:
            self._cancel(old, bar.timestamp, replacement=replacement)
        self._event(bar.timestamp, "STOP_MOVED", f"-> {new_stop} ({detail})")

    def _resize_protection(self, position: Position, ts: datetime) -> None:
        old_stop = self.state.stop_order
        old_target = self.state.target_order
        group = (
            (old_stop.oco_group if old_stop is not None else None)
            or (old_target.oco_group if old_target is not None else None)
            or new_id("oco")
        )

        replacement_stop = self._protective_order(
            position,
            OrderType.STOP,
            position.stop_price,
            OrderPurpose.STOP,
            group,
            ts,
        )
        if replacement_stop is None:
            self._event(
                ts,
                "PROTECTION_RESIZE_FAILED",
                "retained the prior stop after replacement refusal",
                level="ERROR",
            )
            if old_stop is None:
                self._flatten_unprotected(ts, "no stop remains after protection resize")
            return
        self.state.stop_order = replacement_stop
        if old_stop is not None:
            self._cancel(old_stop, ts, replacement=replacement_stop)

        if position.target_price is not None:
            replacement_target = self._protective_order(
                position,
                OrderType.LIMIT,
                position.target_price,
                OrderPurpose.TARGET,
                group,
                ts,
            )
            if replacement_target is not None:
                self.state.target_order = replacement_target
                if old_target is not None:
                    self._cancel(old_target, ts, replacement=replacement_target)
        elif old_target is not None:
            self._cancel(old_target, ts)
            self.state.target_order = None

    def _flatten_unprotected(self, ts: datetime, detail: str) -> None:
        """Fail closed after discovering exposure without an accepted stop."""
        self.risk.kill_switch.trip(f"unprotected position: {detail}")

        # Do not allow an unfilled entry remainder to add more exposure while the emergency
        # market close waits for its acknowledgement/fill.
        pending_entry = self.state.entry_order
        if pending_entry is not None and pending_entry.status.is_working:
            if self._cancel(pending_entry, ts):
                self.state.entry_order = None
                if pending_entry.intent_id:
                    self._intents.pop(pending_entry.intent_id, None)

        if self.state.last_bar is None:
            self._event(
                ts,
                "FLATTEN_FAILED",
                f"cannot submit emergency flatten without a market bar: {detail}",
                level="ERROR",
            )
            return
        self._flatten(self.state.last_bar, ExitReason.RISK_HALT, detail)

    def _cancel_protection(self, ts: datetime) -> None:
        for slot in ("stop_order", "target_order"):
            order = getattr(self.state, slot)
            if order is not None:
                self._cancel(order, ts)
                setattr(self.state, slot, None)

    def _cancel(
        self, order: Order, ts: datetime, *, replacement: Order | None = None
    ) -> bool:
        """Best-effort cancellation, routed by the order's purpose.

        The guard exposes no generic `cancel_order`, and deliberately so: removing an
        entry and removing a live stop are different acts with different safety
        preconditions. Entries go through `cancel_entry`, which only ever reduces pending
        exposure. Protection goes through `retire_protective`, which refuses unless a
        fresh broker snapshot proves the instrument flat or a named replacement already
        covers the exposure — which is why callers that swap a stop place the replacement
        first and pass it here.

        A failure never aborts the caller. `_flatten` cancels the protective pair before
        sending its market exit, and a broker that has just gone away would otherwise
        leave the position both unprotected *and* un-flattened — strictly worse than
        either alone. The venue's own OCO still handles the sibling, and the failure is
        recorded.
        """
        broker_id = self._broker_ids.get(order.order_id)
        if not broker_id:
            # Never acknowledged, so there is nothing at the venue to cancel.
            self._update_order(order, OrderStatus.CANCELLED, ts, detail="cancelled by engine")
            return True

        try:
            if order.purpose is OrderPurpose.ENTRY:
                result = self.broker.cancel_entry(broker_id, now=ts)
            elif order.purpose in (OrderPurpose.STOP, OrderPurpose.TARGET):
                result = self.broker.retire_protective(
                    broker_id,
                    replacement_broker_order_id=(
                        self._broker_ids.get(replacement.order_id) if replacement else None
                    ),
                    now=ts,
                )
            else:
                self._event(
                    ts, "CANCEL_SKIPPED",
                    f"{order.purpose.value} orders have no cancellation path",
                    level="WARN",
                )
                return False
        except BrokerError as exc:
            self._event(ts, "CANCEL_FAILED",
                        f"could not cancel {order.purpose.value} {order.order_id}: {exc}",
                        level="WARN")
            return False

        if not result.accepted:
            self._absorb([result.rejection])
            self._event(ts, "CANCEL_REFUSED",
                        f"guard refused to cancel {order.purpose.value} "
                        f"{order.order_id}: {result.rejection.detail}",
                        level="WARN")
            return False

        # The guard accepted the request; the venue confirms it later with a terminal
        # event. Recording CANCEL_REQUESTED rather than CANCELLED keeps the journal from
        # claiming an outcome that has not happened yet.
        self._update_order(order, OrderStatus.CANCEL_REQUESTED, ts,
                           detail="cancellation requested by engine")
        return True

    # ================================================================== flatten

    def _flatten(self, bar: Bar, reason: ExitReason, detail: str) -> None:
        position = self.state.position
        if position is None:
            return

        # Keep the venue stop working until a closing fill is authoritative.  Cancelling
        # protection before a market flatten is acknowledged/filled creates a naked gap
        # during an outage -- precisely when protection matters most.  ``_on_exit_fill``
        # cancels the remaining stop/target only after the position is actually closed.
        self.state.pending_exit = PendingExit(reason, detail)

        decision = self.risk.evaluate_exit(position, reason="FLATTEN", now=bar.timestamp)
        approval = decision.approval
        self._register(approval.order)
        try:
            result = self.broker.place_order(approval.order, approval.token, now=bar.timestamp)
        except BrokerError as exc:
            self._event(bar.timestamp, "FLATTEN_FAILED", str(exc), level="ERROR")
            self.risk.kill_switch.record_error(f"flatten failed: {exc}")
            return

        if not result.accepted:
            self._absorb([result.rejection])
            kind = (
                "FLATTEN_FAILED"
                if result.rejection.reason is RejectReason.BROKER_ERROR
                else "FLATTEN_REFUSED"
            )
            self._event(bar.timestamp, kind, result.rejection.detail, level="ERROR")
            self.risk.kill_switch.record_error(
                f"flatten refused: {result.rejection.reason.value}"
            )
            return

        self._broker_ids[approval.order.order_id] = result.ack.broker_order_id
        self._update_order(approval.order, OrderStatus.ACCEPTED, bar.timestamp,
                           broker_order_id=result.ack.broker_order_id, detail=reason.value)
        self._event(bar.timestamp, "FLATTEN", f"{reason.value}: {detail}")

    def flatten_now(self, reason: ExitReason = ExitReason.MANUAL, detail: str = "") -> None:
        """External flatten request — the dashboard STOP button, or shutdown."""
        if self.state.last_bar is not None and self.state.position is not None:
            self._flatten(self.state.last_bar, reason, detail)

    # ================================================================== recovery

    def reconcile(self, now: datetime) -> dict:
        """Bring local state in line with the broker after a restart.

        **Broker truth wins.** The journal is the audit record, not the source of live
        position state; if they disagree the broker is right, and the difference is itself
        journalled so the divergence is visible rather than papered over.
        """
        report = {"broker_positions": [], "local_position": None, "action": "none"}
        try:
            positions = self.broker.get_positions()
        except NotConnected:
            self._event(now, "RECONCILE_FAILED", "broker not connected", level="ERROR")
            report["action"] = "failed"
            return report

        report["broker_positions"] = [
            {"instrument": p.instrument, "quantity": p.quantity, "price": p.average_price}
            for p in positions
        ]
        local = self.state.position
        report["local_position"] = (
            None if local is None
            else {"instrument": local.instrument, "quantity": local.signed_quantity}
        )

        broker_qty = sum(p.quantity for p in positions if p.instrument == self.instrument.symbol)
        local_qty = local.signed_quantity if local is not None else 0

        if broker_qty == local_qty:
            report["action"] = "in_sync"
            self._event(now, "RECONCILED", f"in sync at {broker_qty} contracts")
            return report

        self._event(
            now, "RECONCILE_DIVERGENCE",
            f"broker says {broker_qty}, local says {local_qty}; broker wins",
            level="WARN",
            payload=report,
        )

        if broker_qty == 0:
            self.state.position = None
            self.state.bot_state = BotState.RUNNING
            report["action"] = "cleared_local_position"
            return report

        # The broker holds a position this process does not know how to manage: it has no
        # recorded stop, and inventing one would be worse than closing. Flatten it.
        matching = next(p for p in positions if p.instrument == self.instrument.symbol)
        self.state.position = Position(
            instrument=matching.instrument,
            side=Side.BUY if broker_qty > 0 else Side.SELL,
            quantity=abs(broker_qty),
            entry_price=matching.average_price,
            entry_time=now,
            strategy=self.strategy.name,
            initial_stop=0.0,
            stop_price=0.0,
        )
        report["action"] = "adopted_and_will_flatten"
        self.state.bot_state = BotState.IN_POSITION
        if self.state.last_bar is not None:
            self._flatten(self.state.last_bar, ExitReason.MANUAL,
                          "adopted an unmanaged position on restart")
        return report

    # ================================================================== bookkeeping

    def _register(self, order: Order) -> None:
        self._orders[order.order_id] = order
        if self.journal is not None:
            self.journal.record_order(order)

    def _update_order(
        self, order: Order, status: OrderStatus, ts: datetime, *, detail: str = "",
        broker_order_id: str | None = None,
    ) -> None:
        self._orders[order.order_id] = order.with_status(
            status, broker_order_id=broker_order_id or order.broker_order_id
        )
        if self.journal is not None:
            self.journal.update_order(
                order.order_id, status, ts=ts, broker_order_id=broker_order_id,
                filled_quantity=order.filled_quantity,
                average_fill_price=order.average_fill_price, detail=detail,
            )

    def _intent_for(self, entry: Order) -> OrderIntent | None:
        return self._intents.get(entry.intent_id) if entry.intent_id else None

    def _absorb(self, rejections) -> None:
        for rejection in rejections:
            if rejection is None:
                continue
            self.rejections.append(rejection)
            if self.journal is not None:
                self.journal.record_rejection(rejection)

    def _event(self, ts: datetime, kind: str, detail: str, *, level: str = "INFO",
               payload: dict | None = None) -> None:
        if self.journal is not None:
            self.journal.record_event(ts, kind, detail, level=level, payload=payload)

    def _mark(self, bar: Bar) -> None:
        position = self.state.position
        if position is not None:
            position.observe(bar)
        unrealized = (
            position.unrealized_usd(bar.close, self.instrument.multiplier)
            if position is not None else 0.0
        )
        if self.journal is not None:
            self.journal.record_equity(
                AccountSnapshot(
                    timestamp=bar.timestamp,
                    equity=self.state.equity + unrealized,
                    realized_pnl_today=self.risk.state.daily_realized_pnl,
                    unrealized_pnl=unrealized,
                    open_positions=0 if position is None else 1,
                    trades_today=self.risk.state.trades_today,
                )
            )

    def _exit_reason_for_order(self, order: Order) -> ExitReason:
        if self.state.pending_exit is not None:
            return self.state.pending_exit.reason
        if order.purpose is OrderPurpose.TARGET:
            return ExitReason.TAKE_PROFIT
        if order.purpose is OrderPurpose.STOP:
            position = self.state.position
            trailing = (
                position is not None and position.stop_price != position.initial_stop
            )
            return ExitReason.TRAILING_STOP if trailing else ExitReason.STOP_LOSS
        return ExitReason.MANUAL

    # ================================================================== reporting

    def snapshot(self) -> dict:
        position = self.state.position
        mark = self.state.last_bar.close if self.state.last_bar else 0.0
        return {
            "state": self.state.bot_state.value,
            "paper": self.broker.is_paper,
            "broker": self.broker.name,
            "strategy": self.strategy.name,
            "instrument": self.instrument.symbol,
            "equity": self.state.equity,
            "bars_seen": self.state.bars_seen,
            "trades": len(self.state.trades),
            "position": None if position is None else {
                "side": position.side.name,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "stop": position.stop_price,
                "target": position.target_price,
                "bars_held": position.bars_held,
                "unrealized_usd": position.unrealized_usd(mark, self.instrument.multiplier),
                "r_multiple": position.r_multiple_at(mark, self.instrument.multiplier),
            },
            "risk": self.risk.limits_snapshot(),
        }


def _exit_reason_for(reason: RejectReason) -> ExitReason:
    return {
        RejectReason.KILL_SWITCH_ACTIVE: ExitReason.KILL_SWITCH,
        RejectReason.OUTSIDE_TRADING_HOURS: ExitReason.SESSION_CLOSE,
        RejectReason.MAX_DAILY_LOSS: ExitReason.RISK_HALT,
        RejectReason.TRAILING_DRAWDOWN: ExitReason.RISK_HALT,
        RejectReason.STALE_MARKET_DATA: ExitReason.STALE_DATA,
    }.get(reason, ExitReason.RISK_HALT)
