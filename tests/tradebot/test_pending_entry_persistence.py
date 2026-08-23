from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from tradebot.broker.base import BrokerTimeout
from tradebot.broker.costs import CostModel
from tradebot.broker.guarded import GuardedBroker
from tradebot.broker.simulated import SimulatedBroker
from tradebot.config import (
    CostConfig,
    DailyRisk,
    KillSwitchConfig,
    PerTradeRisk,
    RiskConfig,
    SessionConfig,
    SimulatedBrokerConfig,
)
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.models import Bar, Order, OrderIntent, Position
from tradebot.core.types import OrderPurpose, OrderType, RejectReason, Side
from tradebot.instruments.registry import get_instrument
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import RiskEngine
from tradebot.risk.reservations import (
    FilePendingEntryReservationStore,
    PendingEntryReservation,
    PendingEntryState,
    ReservationStoreError,
)
import tradebot.risk.reservations as reservation_module


MNQ = get_instrument("MNQ")


def at(hour: int = 11, minute: int = 0) -> datetime:
    return datetime(2024, 4, 1, hour, minute, tzinfo=MARKET_TZ)


def make_engine(tmp_path, clock: SimulatedClock, name: str) -> RiskEngine:
    config = RiskConfig(
        starting_equity_usd=50_000.0,
        per_trade=PerTradeRisk(
            risk_pct_of_equity=0.5,
            max_risk_per_trade_usd=200.0,
            max_contracts=3,
            min_contracts=1,
        ),
        daily=DailyRisk(
            max_daily_loss_usd=200.0,
            max_daily_loss_r=4.0,
            max_trades_per_day=1,
        ),
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / f"{name}.flag")),
    )
    return RiskEngine(
        config,
        MNQ,
        SessionConfig(),
        clock=clock,
        kill_switch=KillSwitch(tmp_path / f"{name}.flag", clock=clock),
    )


def make_sim(clock: SimulatedClock, *, partial: bool = False) -> SimulatedBroker:
    costs = CostModel.from_config(
        CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0),
        MNQ,
    )
    broker = SimulatedBroker(
        MNQ,
        costs,
        config=SimulatedBrokerConfig(
            latency_ms=0,
            partial_fill_probability=1.0 if partial else 0.0,
            seed=1,
        ),
        clock=clock,
    )
    broker.connect()
    return broker


def intent(*, ts: datetime | None = None, stop: float = 17_990.0) -> OrderIntent:
    return OrderIntent(
        timestamp=ts or at(),
        instrument="MNQ",
        side=Side.BUY,
        strategy="test",
        stop_price=stop,
        reference_price=18_000.0,
        conditions=("deterministic",),
    )


def market_bar(ts: datetime, price: float = 18_000.0) -> Bar:
    return Bar(
        timestamp=ts,
        open=price,
        high=price + 2,
        low=price - 2,
        close=price,
        volume=100,
    )


def position(quantity: int = 1) -> Position:
    return Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=quantity,
        entry_price=18_000.25,
        entry_time=at(11, 1),
        strategy="test",
        initial_stop=17_990.0,
        stop_price=17_990.0,
        target_price=18_020.0,
    )


def seed_long_broker_position(
    broker: SimulatedBroker, clock: SimulatedClock, *, quantity: int = 1
) -> None:
    broker.place_order(
        Order(
            order_id="authoritative-long",
            timestamp=at(),
            instrument="MNQ",
            side=Side.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
    )
    clock.set(at(11, 1))
    broker.on_bar_open(market_bar(at(11, 1)))
    assert broker.get_positions()[0].quantity == quantity


def test_restart_restores_submitting_crash_window_and_blocks_duplicate(
    tmp_path, monkeypatch
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    original_submit = broker.place_order

    def crash_after_venue_submission(order):
        original_submit(order)
        raise SystemExit("simulated process death before ack handling")

    monkeypatch.setattr(broker, "place_order", crash_after_venue_submission)
    with pytest.raises(SystemExit):
        first_guard.place_order(approval.order, approval.token)
    assert store.load().state is PendingEntryState.SUBMITTING
    monkeypatch.setattr(broker, "place_order", original_submit)

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    assert restarted.pending_entry.state is PendingEntryState.SUBMITTING
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval

    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert broker.submit_count == 1
    assert not restarted_risk._tokens.is_spent(second.token)


def test_restart_restores_outcome_unknown_after_timeout(tmp_path, monkeypatch) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    original_submit = broker.place_order

    def timeout_after_venue_submission(order):
        original_submit(order)
        raise BrokerTimeout("ack lost")

    monkeypatch.setattr(broker, "place_order", timeout_after_venue_submission)
    with pytest.raises(BrokerTimeout):
        first_guard.place_order(approval.order, approval.token)
    assert store.load().state is PendingEntryState.OUTCOME_UNKNOWN
    monkeypatch.setattr(broker, "place_order", original_submit)

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    assert restarted.pending_entry.state is PendingEntryState.OUTCOME_UNKNOWN
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval

    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert broker.submit_count == 1
    # The fresh snapshot found the working venue order and upgraded the durable state;
    # it remained reserved and no duplicate was sent.
    assert store.load().state is PendingEntryState.ACKNOWLEDGED


def test_ack_cancel_and_authoritative_terminal_flat_snapshot_allow_next_entry(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    first = first_risk.evaluate_entry(intent()).approval
    first_result = first_guard.place_order(first.order, first.token)
    assert store.load().state is PendingEntryState.ACKNOWLEDGED

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    assert restarted.pending_entry.state is PendingEntryState.ACKNOWLEDGED
    restarted.cancel_entry(first_result.ack.broker_order_id)
    assert store.load().state is PendingEntryState.CANCEL_REQUESTED

    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval
    result = restarted.place_order(second.order, second.token)

    assert result.accepted
    assert broker.submit_count == 2
    assert store.load().order_id == second.order.order_id
    assert store.load().state is PendingEntryState.ACKNOWLEDGED


def test_partial_fill_state_is_durable(tmp_path) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock, partial=True)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    events = guard.on_bar_open(market_bar(at(11, 1)))

    assert events[0].kind.value == "PARTIAL_FILL"
    assert guard.pending_entry.state is PendingEntryState.PARTIALLY_FILLED
    assert store.load().state is PendingEntryState.PARTIALLY_FILLED


def test_filled_entry_is_not_cleared_until_snapshot_is_flat_and_exit_still_works(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    first = first_risk.evaluate_entry(intent()).approval
    first_guard.place_order(first.order, first.token)
    clock.set(at(11, 1))
    first_guard.on_bar_open(market_bar(at(11, 1)))
    persisted = store.load()
    assert persisted.state is PendingEntryState.FILLED

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    blocked = restarted_risk.evaluate_entry(intent(ts=at(11, 1), stop=17_989.0)).approval
    assert not restarted.place_order(blocked.order, blocked.token).accepted
    assert store.load().order_id == first.order.order_id

    closing_position = position(quantity=first.order.quantity)
    exit_approval = restarted_risk.evaluate_exit(closing_position).approval
    assert restarted.place_order(exit_approval.order, exit_approval.token).accepted
    clock.set(at(11, 2))
    restarted.on_bar_open(market_bar(at(11, 2), price=18_001.0))

    next_entry = restarted_risk.evaluate_entry(
        intent(ts=at(11, 2), stop=17_989.0)
    ).approval
    result = restarted.place_order(next_entry.order, next_entry.token)

    assert result.accepted
    assert store.load().order_id == next_entry.order.order_id


@pytest.mark.parametrize("corrupt_kind", ["invalid_json", "unreadable_directory"])
def test_corrupt_or_unreadable_store_fails_closed_but_exit_remains_possible(
    tmp_path, corrupt_kind: str
) -> None:
    state_path = tmp_path / "pending.json"
    if corrupt_kind == "invalid_json":
        state_path.write_text("{not-json", encoding="utf-8")
    else:
        state_path.mkdir()
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(state_path)
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    assert guard.reservation_store_error is not None

    entry = risk.evaluate_entry(intent()).approval
    result = guard.place_order(entry.order, entry.token)
    assert not result.accepted
    assert result.rejection.reason is RejectReason.BROKER_ERROR
    assert result.rejection.stage == "GUARD_RESERVATION_STORE"
    assert not risk._tokens.is_spent(entry.token)

    seed_long_broker_position(broker, clock)
    exit_approval = risk.evaluate_exit(position()).approval
    assert guard.place_order(exit_approval.order, exit_approval.token).accepted


def test_protective_order_remains_possible_when_store_is_corrupt(tmp_path) -> None:
    state_path = tmp_path / "pending.json"
    state_path.write_text("[]", encoding="utf-8")
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(
        broker,
        risk,
        reservation_store=FilePendingEntryReservationStore(state_path),
    )
    seed_long_broker_position(broker, clock)
    open_position = position()
    stop = Order(
        order_id="protective-stop",
        timestamp=at(),
        instrument="MNQ",
        side=Side.SELL,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=17_990.0,
        strategy="test",
        purpose=OrderPurpose.STOP,
        oco_group="protection",
    )
    approval = risk.evaluate_protective(open_position, stop).approval

    assert guard.place_order(approval.order, approval.token).accepted


def test_file_store_atomic_replace_failure_preserves_previous_reservation(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "pending.json"
    store = FilePendingEntryReservationStore(path)
    original = PendingEntryReservation(
        order_id="first",
        broker_order_id=None,
        state=PendingEntryState.SUBMITTING,
        reserved_at=at(),
        updated_at=at(),
        quantity=1,
        risk_usd=100.0,
        instrument="MNQ",
    )
    store.save(original)
    replacement = replace(
        original,
        broker_order_id="broker-first",
        state=PendingEntryState.ACKNOWLEDGED,
    )

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(reservation_module.os, "replace", fail_replace)
    with pytest.raises(ReservationStoreError, match="could not persist"):
        store.save(replacement)

    assert store.load() == original
    assert list(tmp_path.glob(".pending.json.*.tmp")) == []


def test_a_restarted_guard_refuses_a_cancel_id_it_never_reserved(tmp_path) -> None:
    """Recovery widens what `cancel_entry` accepts; it must not widen it to anything.

    A restarted guard has an empty in-memory order ledger, so it accepts a cancellation id
    that matches the *restored reservation*. Any other id is still refused and never
    reaches the adapter.
    """
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    first = first_risk.evaluate_entry(intent()).approval
    first_guard.place_order(first.order, first.token)

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)

    result = restarted.cancel_entry("sim-not-a-real-order")
    assert not result.accepted
    assert not result.request_sent
    assert "not submitted by this guard" in result.rejection.detail
    assert store.load().state is PendingEntryState.ACKNOWLEDGED


def test_cancelling_a_recovered_entry_twice_sends_only_one_request(tmp_path) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk, reservation_store=store)
    first = first_risk.evaluate_entry(intent()).approval
    ack = first_guard.place_order(first.order, first.token).ack

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)

    assert restarted.cancel_entry(ack.broker_order_id).request_sent
    repeat = restarted.cancel_entry(ack.broker_order_id)
    assert repeat.accepted
    assert not repeat.request_sent, "a cancellation already in flight is not re-sent"
