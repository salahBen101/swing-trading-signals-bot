"""A local, deterministic broker simulator.

This is the default adapter, and the one every test and every paper run uses. It has no
credentials and no network, so it can never touch a real venue — which is the strongest
possible form of the "no real-money order during development" requirement.

**Fill model.** A market order submitted while bar *i* is being processed fills at bar
*i+1*'s open, plus slippage. Stop and limit orders fill when a later bar's range reaches
their level. When one bar's range contains both a stop and a target, the **stop is assumed
first** — the intra-bar path is not recoverable from OHLCV, so the pessimistic branch is
taken rather than the flattering one.

**Failure injection.** Rejections, partial fills and disconnects are driven by a seeded
RNG, so a scenario that exposed a bug reproduces exactly. With the default configuration
all three probabilities are zero and the simulator is fully deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..config import SimulatedBrokerConfig
from ..core.clock import Clock, SystemClock
from ..core.models import Bar, Fill, Order, new_id
from ..core.types import OrderPurpose, OrderStatus, OrderType, Side, TimeInForce
from ..instruments.registry import InstrumentSpec
from .base import (
    BrokerAccount,
    BrokerCapabilities,
    BrokerEvent,
    BrokerOrderState,
    BrokerPosition,
    EventKind,
    NotConnected,
    OrderAck,
    OrderRejected,
)
from .costs import CostModel


_RISK_REDUCING_PURPOSES = frozenset(
    {
        OrderPurpose.EXIT,
        OrderPurpose.FLATTEN,
        OrderPurpose.STOP,
        OrderPurpose.TARGET,
    }
)


@dataclass(slots=True)
class _Working:
    order: Order
    broker_order_id: str
    submitted_at: datetime
    releasable_at: datetime
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    last_fill_at: datetime | None = None
    status: OrderStatus = OrderStatus.SUBMITTED
    detail: str = ""


@dataclass(slots=True)
class _NetPosition:
    instrument: str
    quantity: int = 0  # signed
    average_price: float = 0.0
    realized_pnl: float = 0.0


class SimulatedBroker:
    name = "simulated"
    is_paper = True
    execution_route = "simulated-local-paper"
    # The simulator is itself the venue boundary: OCO and terminal cancellation are
    # deterministic and synchronous there. Its fill boundary also caps every privileged
    # reducing order to the exact opposing position, so competing exits cannot reverse
    # through flat.
    capabilities = BrokerCapabilities(
        external_execution=False,
        server_side_oco=True,
        reduce_only_or_close_position=True,
        authoritative_cancel_status=True,
        exact_terminal_order_history=True,
        # The in-memory venue does not yet persist a session fill cursor or an
        # account-owner fencing generation across process restarts.
        authoritative_session_execution_history=False,
        account_owner_fencing=False,
    )

    def __init__(
        self,
        instrument: InstrumentSpec,
        cost_model: CostModel,
        *,
        config: SimulatedBrokerConfig | None = None,
        clock: Clock | None = None,
        starting_equity: float = 50_000.0,
        account_id: str = "SIM-1",
    ) -> None:
        self.instrument = instrument
        self.costs = cost_model
        self.config = config or SimulatedBrokerConfig()
        self.clock = clock or SystemClock()
        self.account_id = account_id
        self.starting_equity = starting_equity

        self._rng = random.Random(self.config.seed)
        self._connected = False
        self._working: dict[str, _Working] = {}
        self._history: dict[str, _Working] = {}
        self._events: list[BrokerEvent] = []
        self._position = _NetPosition(instrument.symbol)
        self._cash = starting_equity
        self._last_price = 0.0
        self.submit_count = 0

    # ------------------------------------------------------------------ connection

    def connect(self) -> None:
        self._connected = True
        self._emit(EventKind.RECONNECTED, detail="simulated broker connected")

    def disconnect(self) -> None:
        self._connected = False
        self._emit(EventKind.DISCONNECTED, detail="simulated broker disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def force_disconnect(self, detail: str = "injected disconnect") -> None:
        """Test/monitoring hook: drop the link without a clean shutdown.

        Working orders are deliberately *kept*. A dropped socket does not cancel orders at
        the exchange, and a simulator that quietly flattened them would teach the recovery
        path a lesson that is false in production.
        """
        self._connected = False
        self._emit(EventKind.DISCONNECTED, detail=detail)

    def _require_connection(self) -> None:
        if not self._connected:
            raise NotConnected("simulated broker is not connected")

    # ------------------------------------------------------------------ orders

    def place_order(self, order: Order) -> OrderAck:
        self._require_connection()
        self.submit_count += 1
        now = self.clock.now()

        if self._rng.random() < self.config.reject_probability:
            self._history[order.order_id] = _Working(
                order, "", now, now, status=OrderStatus.REJECTED, detail="injected rejection"
            )
            self._emit(EventKind.REJECTED, order_id=order.order_id, detail="injected rejection")
            raise OrderRejected(f"order {order.order_id} rejected by the simulated venue")

        if order.quantity <= 0:
            raise OrderRejected("quantity must be positive")

        broker_id = new_id("sim")
        working = _Working(
            order=order,
            broker_order_id=broker_id,
            submitted_at=now,
            releasable_at=now + timedelta(milliseconds=self.config.latency_ms),
            status=OrderStatus.ACCEPTED,
        )
        self._working[broker_id] = working
        self._emit(
            EventKind.ACCEPTED, order_id=order.order_id, broker_order_id=broker_id,
            detail=f"{order.order_type.value} {order.side.name} x{order.quantity}",
        )
        return OrderAck(
            order_id=order.order_id,
            broker_order_id=broker_id,
            status=OrderStatus.ACCEPTED,
            accepted_at=now,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self._require_connection()
        working = self._working.pop(broker_order_id, None)
        if working is None:
            return  # already terminal; cancelling twice is not an error
        working.status = OrderStatus.CANCELLED
        self._history[broker_order_id] = working
        self._emit(
            EventKind.CANCELLED, order_id=working.order.order_id,
            broker_order_id=broker_order_id, detail="cancelled",
        )

    # ------------------------------------------------------------------ matching

    def on_bar(self, bar: Bar) -> list[BrokerEvent]:
        """Compatibility wrapper for callers that do not need two-phase matching.

        The execution engine calls :meth:`on_bar_open`, applies entry fills (which places
        the protective OCO pair), and then calls :meth:`on_bar_range`. Keeping this wrapper
        preserves the broker-level API tests and simple replay clients.
        """
        return [*self.on_bar_open(bar), *self.on_bar_range(bar)]

    def on_bar_open(self, bar: Bar) -> list[BrokerEvent]:
        """Fill market orders and already-marketable resting orders at the open."""
        self._last_price = bar.open
        return self._match_orders(bar, market_phase=True)

    def on_bar_range(self, bar: Bar) -> list[BrokerEvent]:
        """Match resting stops and limits against the remainder of this bar.

        Newly attached protection is deliberately eligible in this phase even when the
        configured simulated latency is non-zero. OHLCV cannot locate the high/low within
        the minute, so assuming the adverse protective level can be reached after the
        bracket becomes active is the safe, pessimistic choice.
        """
        self._last_price = bar.close
        produced = self._match_orders(bar, market_phase=False)
        if self._connected and self._rng.random() < self.config.disconnect_probability:
            self.force_disconnect("injected random disconnect")
            produced.append(self._events[-1])
        return produced

    def _match_orders(self, bar: Bar, *, market_phase: bool) -> list[BrokerEvent]:
        produced: list[BrokerEvent] = []
        if not self._connected:
            return produced

        # Stops are matched before limits. When one bar's range contains both a protective
        # stop and a target, the intra-bar path is not recoverable from OHLCV, so the
        # pessimistic branch is taken rather than the flattering one. Encoding it in the
        # match order puts the rule at the venue, where both the backtest and the paper run
        # inherit it identically.
        touched_oco_groups: set[str] = set()
        for broker_id in sorted(
            self._working,
            key=lambda bid: 0 if self._working[bid].order.order_type
            in (OrderType.STOP, OrderType.STOP_LIMIT) else 1,
        ):
            working = self._working.get(broker_id)
            if working is None:
                continue  # cancelled by an OCO sibling earlier in this same bar
            is_market = working.order.order_type is OrderType.MARKET
            group = working.order.oco_group
            if group is not None and group in touched_oco_groups:
                # With an unknown intra-bar path, once the pessimistic stop side of an OCO
                # has received any fill it is invalid to also assume the target traded
                # later in the same bar. The engine resizes both siblings after the event.
                continue
            same_bar_protection = (
                not market_phase
                and working.order.purpose in ("STOP", "TARGET")
                and working.submitted_at == bar.timestamp
            )
            if bar.timestamp < working.releasable_at and not same_bar_protection:
                continue  # still in flight: latency has not elapsed

            if market_phase:
                if not is_market and not self._executable_at_open(working.order, bar):
                    if working.order.time_in_force is TimeInForce.IOC:
                        produced.append(
                            self._retire_ioc_order(
                                working,
                                detail="IOC entry was not executable at its first eligible open",
                            )
                        )
                    continue
            elif is_market or working.order.time_in_force is TimeInForce.IOC:
                # IOC orders get exactly one venue opportunity at an eligible bar open.
                # A later intrabar wick or future bar must never revive a stale setup.
                continue
            elif (
                working.order.purpose is OrderPurpose.ENTRY
                and working.last_fill_at == bar.timestamp
            ):
                # An entry remainder gets at most one venue match per bar. Otherwise a
                # limit partially filled at the open could add again after its newly
                # attached stop traded later in the same unknown OHLC path.
                continue

            quote = self._fill_price(working.order, bar)
            if quote is None:
                continue
            price, reference_price = quote

            reduce_only_capacity = self._reduce_only_capacity(working.order)
            if reduce_only_capacity == 0:
                produced.append(
                    self._retire_reduce_only_order(
                        working,
                        detail=(
                            "reduce-only: no opposing exposure for "
                            f"{working.order.side.name} {working.order.instrument}"
                        ),
                    )
                )
                if group is not None:
                    touched_oco_groups.add(group)
                produced.extend(
                    self._cancel_oco_siblings(working, reduce_only_terminal=True)
                )
                continue

            remaining = working.order.quantity - working.filled_quantity
            quantity = remaining
            partial = False
            if remaining > 1 and self._rng.random() < self.config.partial_fill_probability:
                quantity = max(1, remaining // 2)
                partial = True

            if reduce_only_capacity is not None:
                quantity = min(quantity, reduce_only_capacity)
                partial = quantity < remaining

            produced.append(
                self._book_fill(
                    working,
                    price,
                    reference_price,
                    quantity,
                    bar.timestamp,
                    partial,
                )
            )

            if (
                working.order.time_in_force is TimeInForce.IOC
                and working.status.is_working
            ):
                produced.append(
                    self._retire_ioc_order(
                        working,
                        detail="IOC unfilled remainder cancelled after its first match",
                    )
                )

            reduce_only_terminal = False
            if (
                reduce_only_capacity is not None
                and self._position.quantity == 0
                and working.status.is_working
            ):
                produced.append(
                    self._retire_reduce_only_order(
                        working,
                        detail="reduce-only: unfilled remainder cancelled at flat",
                    )
                )
                reduce_only_terminal = True
            if group is not None:
                touched_oco_groups.add(group)
            produced.extend(
                self._cancel_oco_siblings(
                    working,
                    reduce_only_terminal=reduce_only_terminal,
                )
            )
        return produced

    def _retire_ioc_order(self, working: _Working, *, detail: str) -> BrokerEvent:
        """Terminally cancel an IOC order or its unfilled remainder."""
        self._working.pop(working.broker_order_id, None)
        working.status = OrderStatus.CANCELLED
        working.detail = detail
        self._history[working.broker_order_id] = working
        return self._emit(
            EventKind.CANCELLED,
            order_id=working.order.order_id,
            broker_order_id=working.broker_order_id,
            detail=detail,
        )

    @staticmethod
    def _executable_at_open(order: Order, bar: Bar) -> bool:
        if order.order_type is OrderType.LIMIT:
            return (
                bar.open <= order.limit_price
                if order.side is Side.BUY
                else bar.open >= order.limit_price
            )
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            return (
                bar.open >= order.stop_price
                if order.side is Side.BUY
                else bar.open <= order.stop_price
            )
        return False

    def _reduce_only_capacity(self, order: Order) -> int | None:
        """Return the exact quantity this order can close, or ``None`` for an entry.

        Purpose, instrument, side, and the venue's current signed position all participate
        in the decision. A privileged risk-reducing order can never add exposure: zero
        means it has no opposing position to close and must terminate without a fill.
        """
        if order.purpose not in _RISK_REDUCING_PURPOSES:
            return None
        position = self._position
        if order.instrument != position.instrument or position.quantity == 0:
            return 0
        if position.quantity * int(order.side) >= 0:
            return 0
        return abs(position.quantity)

    def _retire_reduce_only_order(self, working: _Working, *, detail: str) -> BrokerEvent:
        """Cancel a reducing order whose remaining quantity cannot legally fill."""
        self._working.pop(working.broker_order_id, None)
        working.status = OrderStatus.CANCELLED
        working.detail = detail
        self._history[working.broker_order_id] = working
        return self._emit(
            EventKind.CANCELLED,
            order_id=working.order.order_id,
            broker_order_id=working.broker_order_id,
            detail=detail,
        )

    def _fill_price(self, order: Order, bar: Bar) -> tuple[float, float] | None:
        """Return ``(fill, benchmark)`` or ``None`` when the order does not fill.

        The benchmark is the price against which adverse execution slippage can be
        measured honestly: the next bar's open for market orders, the requested price for
        limits, and the trigger price for stops.  A stop gap is therefore visible as
        slippage instead of disappearing inside P&L, while favourable limit improvement
        is retained in the fill price but never reported as negative "adverse" slippage.
        """
        if order.order_type is OrderType.MARKET:
            return self.costs.apply_slippage(bar.open, order.side), bar.open

        if order.order_type is OrderType.LIMIT:
            limit = order.limit_price
            if order.side is Side.BUY and bar.low <= limit:
                # A gap through the limit fills at the open, which is better than the limit.
                return self.instrument.round_to_tick(min(limit, bar.open)), limit
            if order.side is Side.SELL and bar.high >= limit:
                return self.instrument.round_to_tick(max(limit, bar.open)), limit
            return None

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            stop = order.stop_price
            triggered = (
                bar.high >= stop if order.side is Side.BUY else bar.low <= stop
            )
            if not triggered:
                return None
            # A gap through a stop fills at the open, not at the stop level. Modelling the
            # gap honestly is the difference between a believable backtest and a fantasy.
            raw = max(stop, bar.open) if order.side is Side.BUY else min(stop, bar.open)
            return self.costs.apply_slippage(raw, order.side), stop

        return None

    def _book_fill(
        self,
        working: _Working,
        price: float,
        reference_price: float,
        quantity: int,
        ts: datetime,
        partial: bool,
    ) -> BrokerEvent:
        commission = self.costs.commission_per_fill(quantity)
        adverse_slippage = max(
            0.0,
            (price - reference_price) * int(working.order.side),
        )
        fill = Fill(
            fill_id=new_id("fil"),
            order_id=working.order.order_id,
            timestamp=ts,
            instrument=working.order.instrument,
            side=working.order.side,
            quantity=quantity,
            price=price,
            commission_usd=commission,
            slippage_points=adverse_slippage,
            broker_fill_id=working.broker_order_id,
            is_partial=partial,
        )

        total = working.filled_quantity + quantity
        working.average_fill_price = (
            (working.average_fill_price * working.filled_quantity + price * quantity) / total
        )
        working.filled_quantity = total
        working.last_fill_at = ts

        self._apply_to_position(fill)
        self._cash -= commission

        if working.filled_quantity >= working.order.quantity:
            working.status = OrderStatus.FILLED
            self._working.pop(working.broker_order_id, None)
            self._history[working.broker_order_id] = working
            kind = EventKind.FILL
        else:
            working.status = OrderStatus.PARTIALLY_FILLED
            kind = EventKind.PARTIAL_FILL

        return self._emit(
            kind, order_id=working.order.order_id,
            broker_order_id=working.broker_order_id, fill=fill,
            detail=f"{quantity} @ {price}",
        )

    def _cancel_oco_siblings(
        self,
        filled: _Working,
        *,
        reduce_only_terminal: bool = False,
    ) -> list[BrokerEvent]:
        """Cancel the other side of a one-cancels-other pair once one side is done.

        Only a *completed* order cancels its sibling: a partial fill on the stop leaves the
        target working against the remaining contracts, which is the correct behaviour and
        the one a real venue implements. A reduce-only order retired at flat is also
        complete for OCO purposes, even when its original quantity exceeded the position.
        """
        group = filled.order.oco_group
        if group is None or (
            filled.status is not OrderStatus.FILLED and not reduce_only_terminal
        ):
            return []

        events = []
        for broker_id in list(self._working):
            sibling = self._working[broker_id]
            if sibling.order.oco_group == group and broker_id != filled.broker_order_id:
                self._working.pop(broker_id)
                sibling.status = OrderStatus.CANCELLED
                sibling.detail = f"OCO: cancelled by {filled.order.order_id}"
                self._history[broker_id] = sibling
                events.append(
                    self._emit(
                        EventKind.CANCELLED, order_id=sibling.order.order_id,
                        broker_order_id=broker_id, detail=sibling.detail,
                    )
                )
        return events

    def _apply_to_position(self, fill: Fill) -> None:
        signed = fill.quantity * int(fill.side)
        pos = self._position
        current = pos.quantity
        new_quantity = current + signed

        if current == 0 or (current > 0) == (signed > 0):
            # Opening or adding: weighted-average the entry price.
            total_cost = pos.average_price * abs(current) + fill.price * fill.quantity
            pos.average_price = total_cost / abs(new_quantity) if new_quantity else 0.0
        else:
            # Reducing or reversing: realise P&L on the closed portion.
            closed = min(abs(current), fill.quantity)
            direction = 1 if current > 0 else -1
            pos.realized_pnl += (
                (fill.price - pos.average_price) * direction * closed * self.instrument.multiplier
            )
            if new_quantity == 0:
                pos.average_price = 0.0
            elif (new_quantity > 0) != (current > 0):
                pos.average_price = fill.price  # reversed through flat
        pos.quantity = new_quantity

    # ------------------------------------------------------------------ state

    def get_orders(self) -> list[BrokerOrderState]:
        self._require_connection()
        return [
            BrokerOrderState(
                broker_order_id=w.broker_order_id,
                order_id=w.order.order_id,
                status=w.status,
                filled_quantity=w.filled_quantity,
                average_fill_price=w.average_fill_price,
                detail=w.detail,
            )
            for w in list(self._working.values()) + list(self._history.values())
        ]

    def get_positions(self) -> list[BrokerPosition]:
        self._require_connection()
        if self._position.quantity == 0:
            return []
        return [
            BrokerPosition(
                instrument=self._position.instrument,
                quantity=self._position.quantity,
                average_price=self._position.average_price,
            )
        ]

    def get_account(self) -> BrokerAccount:
        self._require_connection()
        unrealized = 0.0
        if self._position.quantity and self._last_price:
            unrealized = (
                (self._last_price - self._position.average_price)
                * self._position.quantity
                * self.instrument.multiplier
            )
        equity = self._cash + self._position.realized_pnl + unrealized
        return BrokerAccount(
            account_id=self.account_id,
            equity=equity,
            cash=self._cash,
            realized_pnl=self._position.realized_pnl,
            unrealized_pnl=unrealized,
            is_paper=True,
        )

    def poll_events(self) -> list[BrokerEvent]:
        events, self._events = self._events, []
        return events

    # ------------------------------------------------------------------ helpers

    def _emit(self, kind: EventKind, **kwargs) -> BrokerEvent:
        event = BrokerEvent(kind=kind, timestamp=self.clock.now(), **kwargs)
        self._events.append(event)
        return event

    @property
    def working_order_count(self) -> int:
        return len(self._working)
