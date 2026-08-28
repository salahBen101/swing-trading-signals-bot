"""Adversarial tests for simulator-native reduce-only execution semantics."""

from __future__ import annotations

from datetime import datetime

import pytest

from tradebot.broker.base import EventKind
from tradebot.broker.costs import CostModel
from tradebot.broker.simulated import SimulatedBroker
from tradebot.config import CostConfig, SimulatedBrokerConfig
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.models import Bar, Order
from tradebot.core.types import OrderPurpose, OrderStatus, OrderType, Side
from tradebot.instruments.registry import get_instrument


MNQ = get_instrument("MNQ")


def _at(minute: int) -> datetime:
    return datetime(2024, 4, 1, 11, minute, tzinfo=MARKET_TZ)


def _bar(minute: int, *, price: float = 18_000.0) -> Bar:
    return Bar(
        timestamp=_at(minute),
        open=price,
        high=price + 20.0,
        low=price - 20.0,
        close=price,
        volume=100.0,
    )


def _broker() -> SimulatedBroker:
    broker = SimulatedBroker(
        MNQ,
        CostModel.from_config(
            CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0),
            MNQ,
        ),
        config=SimulatedBrokerConfig(latency_ms=0),
        clock=SimulatedClock(_at(0)),
    )
    broker.connect()
    broker.poll_events()
    return broker


def _seed_position(broker: SimulatedBroker, *, side: Side, quantity: int) -> None:
    broker.place_order(
        Order(
            order_id="entry",
            timestamp=_at(0),
            instrument="MNQ",
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            purpose=OrderPurpose.ENTRY,
        )
    )
    broker.on_bar(_bar(1))
    broker.poll_events()


def test_two_full_exits_accepted_before_matching_can_never_reverse_through_flat() -> None:
    broker = _broker()
    _seed_position(broker, side=Side.BUY, quantity=3)

    for order_id in ("exit-a", "exit-b"):
        broker.place_order(
            Order(
                order_id=order_id,
                timestamp=_at(2),
                instrument="MNQ",
                side=Side.SELL,
                quantity=3,
                order_type=OrderType.MARKET,
                purpose=OrderPurpose.EXIT,
            )
        )

    events = broker.on_bar(_bar(3, price=18_010.0))

    exit_fills = [event.fill for event in events if event.fill is not None]
    assert sum(fill.quantity for fill in exit_fills) == 3
    assert all(fill.side is Side.SELL for fill in exit_fills)
    assert broker.get_positions() == [], "the second exit must not open a short position"

    states = {state.order_id: state for state in broker.get_orders()}
    assert {states[order_id].status for order_id in ("exit-a", "exit-b")} == {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    }
    assert any(
        event.kind is EventKind.CANCELLED and "reduce-only" in event.detail
        for event in events
    )


@pytest.mark.parametrize(
    ("purpose", "order_type", "price_field"),
    [
        (OrderPurpose.EXIT, OrderType.MARKET, {}),
        (OrderPurpose.FLATTEN, OrderType.MARKET, {}),
        (OrderPurpose.STOP, OrderType.STOP, {"stop_price": 17_995.0}),
        (OrderPurpose.TARGET, OrderType.LIMIT, {"limit_price": 18_005.0}),
    ],
)
def test_every_risk_reducing_purpose_is_capped_to_current_opposing_exposure(
    purpose: OrderPurpose,
    order_type: OrderType,
    price_field: dict[str, float],
) -> None:
    broker = _broker()
    _seed_position(broker, side=Side.BUY, quantity=2)
    broker.place_order(
        Order(
            order_id=f"reduce-{purpose.value.lower()}",
            timestamp=_at(2),
            instrument="MNQ",
            side=Side.SELL,
            quantity=5,
            order_type=order_type,
            purpose=purpose,
            **price_field,
        )
    )

    events = broker.on_bar(_bar(3))

    fills = [event.fill for event in events if event.fill is not None]
    assert len(fills) == 1
    assert fills[0].quantity == 2
    assert fills[0].is_partial is True
    assert broker.get_positions() == []
    state = next(state for state in broker.get_orders() if state.order_id.startswith("reduce-"))
    assert state.status is OrderStatus.CANCELLED
    assert state.filled_quantity == 2
    assert "unfilled remainder" in state.detail


@pytest.mark.parametrize(
    "purpose",
    [OrderPurpose.EXIT, OrderPurpose.FLATTEN, OrderPurpose.STOP, OrderPurpose.TARGET],
)
def test_a_risk_reducing_order_on_the_same_side_terminates_without_adding(
    purpose: OrderPurpose,
) -> None:
    broker = _broker()
    _seed_position(broker, side=Side.SELL, quantity=2)
    broker.place_order(
        Order(
            order_id=f"same-side-{purpose.value.lower()}",
            timestamp=_at(2),
            instrument="MNQ",
            side=Side.SELL,
            quantity=2,
            order_type=OrderType.MARKET,
            purpose=purpose,
        )
    )

    events = broker.on_bar(_bar(3))

    assert all(event.fill is None for event in events)
    assert broker.get_positions()[0].quantity == -2
    assert any(
        event.kind is EventKind.CANCELLED and "no opposing" in event.detail
        for event in events
    )


def test_capped_protective_fill_retires_its_oco_sibling() -> None:
    broker = _broker()
    _seed_position(broker, side=Side.BUY, quantity=2)
    group = "protect-position"
    broker.place_order(
        Order(
            order_id="oversized-stop",
            timestamp=_at(2),
            instrument="MNQ",
            side=Side.SELL,
            quantity=5,
            order_type=OrderType.STOP,
            stop_price=17_995.0,
            purpose=OrderPurpose.STOP,
            oco_group=group,
        )
    )
    broker.place_order(
        Order(
            order_id="oversized-target",
            timestamp=_at(2),
            instrument="MNQ",
            side=Side.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=18_005.0,
            purpose=OrderPurpose.TARGET,
            oco_group=group,
        )
    )

    events = broker.on_bar(_bar(3))

    assert sum(event.fill.quantity for event in events if event.fill is not None) == 2
    assert broker.get_positions() == []
    assert broker.working_order_count == 0
    states = {state.order_id: state for state in broker.get_orders()}
    assert states["oversized-stop"].status is OrderStatus.CANCELLED
    assert states["oversized-stop"].filled_quantity == 2
    assert states["oversized-target"].status is OrderStatus.CANCELLED
    assert states["oversized-target"].filled_quantity == 0
