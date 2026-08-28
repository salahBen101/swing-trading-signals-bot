from __future__ import annotations

import json
import multiprocessing
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tradebot.broker.base import (
    BrokerEvent,
    BrokerOrderState,
    BrokerTimeout,
    EventKind,
    OrderAck,
    OrderRejected,
)
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
from tradebot.core.models import Bar, Fill, Order, OrderIntent, Position, Trade
from tradebot.core.types import (
    ExitReason,
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
)
from tradebot.instruments.registry import get_instrument
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import RiskEngine
from tradebot.risk.reservations import (
    FilePendingEntryReservationStore,
    PendingEntryReservation,
    PendingEntryState,
    ReservationStoreError,
)
from tradebot.risk.state_store import FileRiskStateStore
import tradebot.risk.reservations as reservation_module


MNQ = get_instrument("MNQ")


def _concurrent_reservation_save(path, reservation, start, results) -> None:
    start.wait()
    try:
        FilePendingEntryReservationStore(path).save(reservation)
    except ReservationStoreError:
        results.put("stale")
    else:
        results.put("saved")


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


def test_timeout_after_venue_submission_cancels_before_restart(
    tmp_path, monkeypatch
) -> None:
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
    assert store.load().state is PendingEntryState.CANCEL_REQUESTED
    assert broker.working_order_count == 0
    monkeypatch.setattr(broker, "place_order", original_submit)

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    assert restarted.pending_entry.state is PendingEntryState.CANCEL_REQUESTED
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval

    result = restarted.place_order(second.order, second.token)

    assert result.accepted
    assert broker.submit_count == 2
    assert store.load().state is PendingEntryState.ACKNOWLEDGED


def test_ack_persistence_failure_cancels_the_accepted_entry(tmp_path) -> None:
    class FailOnAcknowledgementStore:
        def __init__(self, delegate):
            self.delegate = delegate

        def load(self):
            return self.delegate.load()

        def save(self, reservation):
            if reservation.broker_order_id is not None:
                raise ReservationStoreError("injected acknowledgement persistence failure")
            return self.delegate.save(reservation)

        def clear(self, expected):
            self.delegate.clear(expected)

    clock = SimulatedClock(at())
    broker = make_sim(clock)
    delegate = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(
        broker,
        risk,
        reservation_store=FailOnAcknowledgementStore(delegate),
    )
    approval = risk.evaluate_entry(intent()).approval

    with pytest.raises(ReservationStoreError, match="acknowledged an entry"):
        guard.place_order(approval.order, approval.token)

    assert guard.reservation_store_error is not None
    assert delegate.load().state is PendingEntryState.SUBMITTING
    assert broker.working_order_count == 0
    assert broker.get_positions() == []
    assert broker.submit_count == 1


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

    assert result.accepted, restarted.reservation_store_error
    assert broker.submit_count == 2
    assert store.load().order_id == second.order.order_id
    assert store.load().state is PendingEntryState.ACKNOWLEDGED


def test_partial_fill_then_ioc_remainder_cancel_is_durable(tmp_path) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock, partial=True)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    events = guard.on_bar_open(market_bar(at(11, 1)))

    assert [event.kind for event in events] == [
        EventKind.PARTIAL_FILL,
        EventKind.CANCELLED,
    ]
    assert guard.pending_entry.state is PendingEntryState.TERMINAL_REPORTED
    persisted = store.load()
    assert persisted.state is PendingEntryState.TERMINAL_REPORTED
    assert persisted.ever_fill_observed
    assert persisted.cumulative_filled_quantity == events[0].fill.quantity
    assert persisted.observed_fill_ids == (events[0].fill.fill_id,)
    assert persisted.broker_account_id == "SIM-1"
    assert persisted.broker_execution_route == "simulated-local-paper"

    # The simulator's explicit bar hook and later event poll can surface the same event.
    # Its stable fill id prevents replay from inflating durable cumulative quantity.
    guard.poll_events()
    replayed = store.load()
    assert replayed.cumulative_filled_quantity == persisted.cumulative_filled_quantity
    assert replayed.observed_fill_ids == persisted.observed_fill_ids


def test_ioc_terminal_event_cannot_erase_prior_partial_fill(tmp_path) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock, partial=True)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    events = guard.on_bar_open(market_bar(at(11, 1)))
    partial = events[0]
    assert [event.kind for event in events] == [
        EventKind.PARTIAL_FILL,
        EventKind.CANCELLED,
    ]
    terminal = store.load()
    assert terminal.state is PendingEntryState.TERMINAL_REPORTED
    assert terminal.ever_fill_observed
    assert terminal.cumulative_filled_quantity == partial.fill.quantity
    assert terminal.observed_fill_ids == (partial.fill.fill_id,)


def test_fill_event_is_not_exposed_when_fill_evidence_cannot_be_persisted(
    tmp_path,
) -> None:
    class FailOnFillStore:
        def __init__(self, delegate):
            self.delegate = delegate

        def load(self):
            return self.delegate.load()

        def save(self, reservation):
            if reservation.ever_fill_observed:
                raise ReservationStoreError("injected fill persistence failure")
            return self.delegate.save(reservation)

        def clear(self, expected):
            self.delegate.clear(expected)

    clock = SimulatedClock(at())
    broker = make_sim(clock, partial=True)
    delegate = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(
        broker,
        risk,
        reservation_store=FailOnFillStore(delegate),
    )

    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    with pytest.raises(ReservationStoreError, match="fill was withheld"):
        guard.on_bar_open(market_bar(at(11, 1)))

    assert guard.reservation_store_error is not None
    assert delegate.load().state is PendingEntryState.ACKNOWLEDGED
    # Persistence failed after real venue exposure appeared. The guard must submit an
    # exact broker-snapshot flatten without waiting for a fabricated local Position.
    assert broker.submit_count == 2
    assert broker.get_positions()[0].quantity != 0

    clock.set(at(11, 2))
    guard.on_bar_open(market_bar(at(11, 2)))
    assert broker.get_positions() == []


@pytest.mark.parametrize(
    "mutation",
    [
        "fill-order",
        "event-envelope",
        "instrument",
        "side",
        "fill-id",
        "broker-fill-id",
        "fill-before-reservation",
        "fill-after-event",
        "naive-event-time",
        "boolean-quantity",
        "overfill",
        "nonfinite-price",
        "off-tick-price",
        "limit-violation",
        "negative-commission",
        "nonfinite-slippage",
        "partial-flag",
        "missing-fill",
        "fill-on-accepted",
    ],
)
def test_invalid_fill_evidence_is_latched_before_persistence_and_cancels_entry(
    tmp_path,
    monkeypatch,
    mutation: str,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / f"pending-{mutation}.json")
    risk = make_engine(tmp_path, clock, f"risk-{mutation}")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    ack = guard.place_order(approval.order, approval.token).ack
    before = store.load()
    fill = Fill(
        fill_id="venue-fill-1",
        order_id=approval.order.order_id,
        timestamp=at(11, 1),
        instrument="MNQ",
        side=Side.BUY,
        quantity=approval.order.quantity,
        price=18_000.25,
        commission_usd=1.0,
        slippage_points=0.25,
        broker_fill_id="venue-fill-1",
        is_partial=False,
    )
    event = BrokerEvent(
        kind=EventKind.FILL,
        timestamp=at(11, 1),
        order_id=approval.order.order_id,
        broker_order_id=ack.broker_order_id,
        fill=fill,
    )

    if mutation == "fill-order":
        fill = replace(fill, order_id="different-order")
    elif mutation == "event-envelope":
        event = replace(
            event,
            order_id="different-order",
            broker_order_id="different-broker-order",
        )
    elif mutation == "instrument":
        fill = replace(fill, instrument="MES")
    elif mutation == "side":
        fill = replace(fill, side=Side.SELL)
    elif mutation == "fill-id":
        fill = replace(fill, fill_id=" ")
    elif mutation == "broker-fill-id":
        fill = replace(fill, broker_fill_id=" ")
    elif mutation == "fill-before-reservation":
        fill = replace(fill, timestamp=at(10, 59))
    elif mutation == "fill-after-event":
        fill = replace(fill, timestamp=at(11, 2))
    elif mutation == "naive-event-time":
        event = replace(event, timestamp=at(11, 1).replace(tzinfo=None))
    elif mutation == "boolean-quantity":
        fill = replace(fill, quantity=True)
    elif mutation == "overfill":
        fill = replace(fill, quantity=approval.order.quantity + 1)
    elif mutation == "nonfinite-price":
        fill = replace(fill, price=float("nan"))
    elif mutation == "off-tick-price":
        fill = replace(fill, price=18_000.10)
    elif mutation == "limit-violation":
        fill = replace(fill, price=approval.order.limit_price + MNQ.tick_size)
    elif mutation == "negative-commission":
        fill = replace(fill, commission_usd=-0.01)
    elif mutation == "nonfinite-slippage":
        fill = replace(fill, slippage_points=float("nan"))
    elif mutation == "partial-flag":
        fill = replace(fill, is_partial=True)
    elif mutation == "missing-fill":
        event = replace(event, fill=None)
    elif mutation == "fill-on-accepted":
        event = replace(event, kind=EventKind.ACCEPTED)
    if mutation not in {"missing-fill", "naive-event-time"}:
        event = replace(event, fill=fill)
    monkeypatch.setattr(broker, "poll_events", lambda: [event])
    clock.set(at(11, 1))

    with pytest.raises(ReservationStoreError):
        guard.poll_events()

    assert guard.reservation_store_error is not None
    after = store.load()
    assert after.state is PendingEntryState.OUTCOME_UNKNOWN
    assert after.order_id == before.order_id
    assert after.broker_order_id == before.broker_order_id
    assert after.cumulative_filled_quantity == before.cumulative_filled_quantity
    assert after.observed_fill_ids == before.observed_fill_ids
    assert broker.working_order_count == 0


def test_invalid_post_fill_evidence_engages_emergency_flatten(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)
    original_open = broker.on_bar_open

    def fill_at_venue_then_corrupt_evidence(bar):
        events = original_open(bar)
        assert events and events[0].fill is not None
        bad_fill = replace(events[0].fill, side=Side.SELL)
        return [replace(events[0], fill=bad_fill)]

    monkeypatch.setattr(broker, "on_bar_open", fill_at_venue_then_corrupt_evidence)
    clock.set(at(11, 1))

    with pytest.raises(ReservationStoreError):
        guard.on_bar_open(market_bar(at(11, 1)))

    assert guard.reservation_store_error is not None
    assert broker.submit_count == 2, "confirmed exposure must trigger an exact flatten"
    assert broker.get_positions()[0].quantity == approval.order.quantity

    monkeypatch.setattr(broker, "on_bar_open", original_open)
    clock.set(at(11, 2))
    guard.on_bar_open(market_bar(at(11, 2)))
    assert broker.get_positions() == []


def test_sell_fill_cannot_trade_below_its_signed_limit_envelope(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    sell_intent = OrderIntent(
        timestamp=at(),
        instrument="MNQ",
        side=Side.SELL,
        strategy="test",
        stop_price=18_010.0,
        reference_price=18_000.0,
        conditions=("deterministic",),
    )
    approval = risk.evaluate_entry(sell_intent).approval
    ack = guard.place_order(approval.order, approval.token).ack
    fill = Fill(
        fill_id="sell-below-limit",
        order_id=approval.order.order_id,
        timestamp=at(11, 1),
        instrument="MNQ",
        side=Side.SELL,
        quantity=approval.order.quantity,
        price=approval.order.limit_price - MNQ.tick_size,
        commission_usd=1.0,
        slippage_points=0.25,
        broker_fill_id="sell-below-limit",
        is_partial=False,
    )
    event = BrokerEvent(
        kind=EventKind.FILL,
        timestamp=at(11, 1),
        order_id=approval.order.order_id,
        broker_order_id=ack.broker_order_id,
        fill=fill,
    )
    monkeypatch.setattr(broker, "poll_events", lambda: [event])
    clock.set(at(11, 1))

    with pytest.raises(ReservationStoreError, match="SELL .*fill"):
        guard.poll_events()

    assert guard.reservation_store_error is not None
    assert broker.working_order_count == 0


def make_durable_engine(
    tmp_path,
    clock: SimulatedClock,
    store: FileRiskStateStore,
    *,
    bootstrap: bool,
) -> RiskEngine:
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
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "durable.flag")),
    )
    return RiskEngine(
        config,
        MNQ,
        SessionConfig(),
        clock=clock,
        kill_switch=KillSwitch(tmp_path / "durable.flag", clock=clock),
        expected_broker_account_id="SIM-1",
        expected_broker_name="simulated",
        expected_broker_is_paper=True,
        expected_broker_route="simulated-local-paper",
        risk_state_store=store,
        bootstrap_risk_state_store=bootstrap,
        risk_state_context_id="pending-entry-integration:stage2",
    )


def _reconciled_durable_guard(tmp_path, clock, broker, name):
    risk = make_durable_engine(
        tmp_path,
        clock,
        FileRiskStateStore(tmp_path / f"{name}-risk.json"),
        bootstrap=True,
    )
    reservation_store = FilePendingEntryReservationStore(
        tmp_path / f"{name}-pending.json"
    )
    guard = GuardedBroker(
        broker,
        risk,
        reservation_store=reservation_store,
    )
    assert guard.reconcile_risk_state() is None
    return risk, guard, reservation_store


def test_zero_fill_ioc_terminal_event_releases_only_after_exact_flat_history(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "ioc",
    )
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)
    assert risk.state.active_entry_order_id == approval.order.order_id

    clock.set(at(11, 1))
    events = guard.on_bar_open(market_bar(at(11, 1), price=18_010.0))

    assert [event.kind for event in events] == [EventKind.CANCELLED]
    assert reservation_store.load() is None
    assert guard.pending_entry is None
    assert risk.state.active_entry_order_id is None
    assert risk.state.trades_today == 0


def test_terminal_cancel_event_without_exact_history_preserves_both_locks(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "missing-history",
    )
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)
    monkeypatch.setattr(broker, "get_orders", lambda: [])

    clock.set(at(11, 1))
    events = guard.on_bar_open(market_bar(at(11, 1), price=18_010.0))

    assert [event.kind for event in events] == [EventKind.CANCELLED]
    assert reservation_store.load().state is PendingEntryState.TERMINAL_REPORTED
    assert guard.pending_entry is not None
    assert risk.state.active_entry_order_id == approval.order.order_id


@pytest.mark.parametrize("status", [OrderStatus.CANCELLED, OrderStatus.EXPIRED])
def test_terminal_ack_releases_after_independent_exact_flat_history(
    tmp_path,
    monkeypatch,
    status: OrderStatus,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        f"ack-{status.value}",
    )
    original_submit = broker.place_order

    def terminal_ack(order):
        ack = original_submit(order)
        broker.cancel_order(ack.broker_order_id)
        return replace(ack, status=status)

    monkeypatch.setattr(broker, "place_order", terminal_ack)
    approval = risk.evaluate_entry(intent()).approval

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert reservation_store.load() is None
    assert risk.state.active_entry_order_id is None


def test_partial_fill_ack_without_fill_economics_is_withheld_and_cancelled(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "partial-ack",
    )
    original_submit = broker.place_order

    def partial_ack(order):
        ack = original_submit(order)
        return replace(ack, status=OrderStatus.PARTIALLY_FILLED)

    monkeypatch.setattr(broker, "place_order", partial_ack)
    approval = risk.evaluate_entry(intent()).approval

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert guard.reservation_store_error is not None
    persisted = reservation_store.load()
    assert persisted.state is PendingEntryState.OUTCOME_UNKNOWN
    assert persisted.ever_fill_observed
    assert persisted.cumulative_filled_quantity == 1
    assert risk.state.trades_today == 1
    assert broker.working_order_count == 0


def test_filled_ack_without_fill_economics_is_withheld_and_exposure_flattened(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "filled-ack",
    )
    original_submit = broker.place_order

    def filled_ack(order):
        ack = original_submit(order)
        if order.purpose is OrderPurpose.ENTRY:
            clock.set(at(11, 1))
            events = broker.on_bar_open(market_bar(at(11, 1)))
            assert [event.kind for event in events] == [EventKind.FILL]
            return replace(ack, status=OrderStatus.FILLED)
        return ack

    monkeypatch.setattr(broker, "place_order", filled_ack)
    approval = risk.evaluate_entry(intent()).approval

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert guard.reservation_store_error is not None
    assert reservation_store.load().state is PendingEntryState.OUTCOME_UNKNOWN
    assert risk.state.trades_today == 1
    assert broker.submit_count == 2
    assert broker.get_positions()[0].quantity == approval.order.quantity

    clock.set(at(11, 2))
    guard.on_bar_open(market_bar(at(11, 2)))
    assert broker.get_positions() == []


def test_exception_order_rejection_reconciles_exact_terminal_flat_history(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "exception-reject",
    )
    original_submit = broker.place_order

    def terminal_then_raise(order):
        ack = original_submit(order)
        broker.cancel_order(ack.broker_order_id)
        raise OrderRejected("venue rejected entry")

    monkeypatch.setattr(broker, "place_order", terminal_then_raise)
    approval = risk.evaluate_entry(intent()).approval

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.BROKER_ERROR
    assert reservation_store.load() is None
    assert risk.state.active_entry_order_id is None


def test_exception_order_rejection_cancels_when_history_still_says_working(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    risk, guard, reservation_store = _reconciled_durable_guard(
        tmp_path,
        clock,
        broker,
        "exception-working",
    )
    original_submit = broker.place_order

    def working_then_raise(order):
        original_submit(order)
        raise OrderRejected("ambiguous rejection after venue acceptance")

    monkeypatch.setattr(broker, "place_order", working_then_raise)
    approval = risk.evaluate_entry(intent()).approval

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert broker.working_order_count == 0
    persisted = reservation_store.load()
    assert persisted.state is PendingEntryState.TERMINAL_REPORTED
    assert persisted.broker_order_id is not None
    assert risk.state.active_entry_order_id == approval.order.order_id


def test_schema_v3_binds_signed_entry_envelope_and_rejects_legacy_or_unbound(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    path = tmp_path / "pending.json"
    store = FilePendingEntryReservationStore(path)
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 3
    assert document["reservation"]["broker_account_id"] == "SIM-1"
    assert (
        document["reservation"]["broker_execution_route"]
        == "simulated-local-paper"
    )
    assert document["reservation"]["cumulative_filled_quantity"] == 0
    assert document["reservation"]["ever_fill_observed"] is False
    assert document["reservation"]["side"] == int(approval.order.side)
    assert document["reservation"]["order_type"] == approval.order.order_type.value
    assert document["reservation"]["limit_price"] == approval.order.limit_price
    assert (
        document["reservation"]["protective_stop_price"]
        == approval.token.stop_price
    )
    assert document["reservation"]["revision"] >= 2

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps({"schema_version": 2, "reservation": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ReservationStoreError, match="unsupported reservation schema"):
        FilePendingEntryReservationStore(legacy_path).load()

    document["reservation"].pop("broker_account_id")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReservationStoreError, match="missing=.*broker_account_id"):
        store.load()


def test_restarted_cancel_refuses_account_switch_and_preserves_original_order(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    ack = first.place_order(approval.order, approval.token).ack
    assert store.load().broker_account_id == "SIM-1"

    broker.account_id = "SIM-2"
    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)

    result = restarted.cancel_entry(ack.broker_order_id)

    assert not result.accepted
    assert not result.request_sent
    assert broker.working_order_count == 1
    assert restarted.reservation_store_error is not None
    assert "identity mismatch" in restarted.reservation_store_error
    assert store.load().broker_account_id == "SIM-1"


def test_restored_route_mismatch_locks_entries_without_rewriting_reservation(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    first.place_order(approval.order, approval.token)
    original = store.load()
    tampered_store = FilePendingEntryReservationStore(tmp_path / "tampered.json")
    tampered_store.save(
        replace(
            original,
            broker_execution_route="different-paper-route",
            revision=1,
        )
    )

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(
        broker,
        restarted_risk,
        reservation_store=tampered_store,
    )
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval
    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.stage == "GUARD_RESERVATION_STORE"
    assert broker.submit_count == 1
    assert tampered_store.load().broker_execution_route == "different-paper-route"


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

    assert result.accepted, restarted.reservation_store_error
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
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        limit_price=18_000.0,
        protective_stop_price=17_990.0,
        broker_account_id="SIM-1",
        broker_execution_route="simulated-local-paper",
        cumulative_filled_quantity=0,
        ever_fill_observed=False,
        observed_fill_ids=(),
        revision=1,
    )
    store.save(original)
    replacement = replace(
        original,
        broker_order_id="broker-first",
        state=PendingEntryState.ACKNOWLEDGED,
        revision=2,
    )

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(reservation_module.os, "replace", fail_replace)
    with pytest.raises(ReservationStoreError, match="could not persist"):
        store.save(replacement)

    assert store.load() == original
    assert list(tmp_path.glob(".pending.json.*.tmp")) == []


def test_stale_writer_and_stale_clear_cannot_erase_newer_fill_evidence(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    path = tmp_path / "pending.json"
    store = FilePendingEntryReservationStore(path)
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)
    stale = FilePendingEntryReservationStore(path).load()
    current = FilePendingEntryReservationStore(path).load()

    filled = replace(
        current,
        state=PendingEntryState.PARTIALLY_FILLED,
        cumulative_filled_quantity=1,
        ever_fill_observed=True,
        observed_fill_ids=("fill-won-cas",),
        revision=current.revision + 1,
    )
    FilePendingEntryReservationStore(path).save(filled)

    stale_terminal = replace(
        stale,
        state=PendingEntryState.TERMINAL_REPORTED,
        revision=stale.revision + 1,
    )
    with pytest.raises(ReservationStoreError, match="stale reservation revision"):
        FilePendingEntryReservationStore(path).save(stale_terminal)
    with pytest.raises(ReservationStoreError, match="changed before compare-and-clear"):
        FilePendingEntryReservationStore(path).clear(stale)

    persisted = store.load()
    assert persisted == filled
    assert persisted.ever_fill_observed
    assert persisted.cumulative_filled_quantity == 1


def test_concurrent_weak_terminal_cannot_discard_stronger_fill_evidence(
    tmp_path,
) -> None:
    path = tmp_path / "pending.json"
    store = FilePendingEntryReservationStore(path)
    initial = PendingEntryReservation(
        order_id="concurrent-entry",
        broker_order_id="concurrent-broker",
        state=PendingEntryState.ACKNOWLEDGED,
        reserved_at=at(),
        updated_at=at(),
        quantity=3,
        risk_usd=60.0,
        instrument="MNQ",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        limit_price=18_000.0,
        protective_stop_price=17_990.0,
        broker_account_id="SIM-1",
        broker_execution_route="simulated-local-paper",
        cumulative_filled_quantity=0,
        ever_fill_observed=False,
        observed_fill_ids=(),
        revision=1,
    )
    store.save(initial)
    proposals = (
        replace(
            initial,
            state=PendingEntryState.TERMINAL_REPORTED,
            updated_at=at(11, 1),
            revision=2,
        ),
        replace(
            initial,
            state=PendingEntryState.PARTIALLY_FILLED,
            updated_at=at(11, 1),
            cumulative_filled_quantity=1,
            ever_fill_observed=True,
            observed_fill_ids=("concurrent-fill",),
            revision=2,
        ),
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_reservation_save,
            args=(path, proposal, start, results),
        )
        for proposal in proposals
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive(), "reservation writer deadlocked on advisory lock"
        assert process.exitcode == 0

    outcomes = [results.get(timeout=2) for _ in processes]
    assert "saved" in outcomes
    assert set(outcomes) <= {"saved", "stale"}
    persisted = store.load()
    assert persisted.revision in {2, 3}
    assert persisted.ever_fill_observed
    assert persisted.cumulative_filled_quantity >= 1
    assert "concurrent-fill" in persisted.observed_fill_ids


@pytest.mark.parametrize(
    "mutation",
    [
        "account",
        "route",
        "risk",
        "side",
        "order-type",
        "limit",
        "protective-stop",
        "fill-regression",
        "state-regression",
    ],
)
def test_store_rejects_identity_risk_and_fill_regressions(
    tmp_path,
    mutation: str,
) -> None:
    path = tmp_path / f"pending-{mutation}.json"
    store = FilePendingEntryReservationStore(path)
    original = PendingEntryReservation(
        order_id="entry-one",
        broker_order_id="broker-one",
        state=PendingEntryState.PARTIALLY_FILLED,
        reserved_at=at(),
        updated_at=at(11, 1),
        quantity=3,
        risk_usd=60.0,
        instrument="MNQ",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        limit_price=18_000.0,
        protective_stop_price=17_990.0,
        broker_account_id="SIM-1",
        broker_execution_route="simulated-local-paper",
        cumulative_filled_quantity=1,
        ever_fill_observed=True,
        observed_fill_ids=("fill-one",),
        revision=1,
    )
    store.save(original)
    changes = {"revision": 2, "updated_at": at(11, 2)}
    if mutation == "account":
        changes["broker_account_id"] = "SIM-2"
    elif mutation == "route":
        changes["broker_execution_route"] = "different-paper-route"
    elif mutation == "risk":
        changes["risk_usd"] = 59.0
    elif mutation == "side":
        changes.update(side=Side.SELL, protective_stop_price=18_010.0)
    elif mutation == "order-type":
        changes.update(order_type=OrderType.MARKET, limit_price=None)
    elif mutation == "limit":
        changes["limit_price"] = 18_001.0
    elif mutation == "protective-stop":
        changes["protective_stop_price"] = 17_989.0
    elif mutation == "fill-regression":
        changes.update(
            state=PendingEntryState.TERMINAL_REPORTED,
            cumulative_filled_quantity=0,
            ever_fill_observed=False,
            observed_fill_ids=(),
        )
    else:
        changes["state"] = PendingEntryState.ACKNOWLEDGED

    with pytest.raises(ReservationStoreError):
        store.save(replace(original, **changes))

    assert store.load() == original


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


def test_terminal_ack_does_not_claim_cancelled_or_expired_entry_was_unfilled(
    tmp_path,
    monkeypatch,
) -> None:
    for status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
        clock = SimulatedClock(at())
        broker = make_sim(clock)
        store = FilePendingEntryReservationStore(
            tmp_path / f"pending-{status.value}.json"
        )
        risk = make_engine(tmp_path, clock, f"risk-{status.value}")
        guard = GuardedBroker(broker, risk, reservation_store=store)
        approval = risk.evaluate_entry(intent()).approval
        original_submit = broker.place_order

        def terminal_ack(order, *, terminal_status=status):
            ack = original_submit(order)
            return replace(ack, status=terminal_status)

        monkeypatch.setattr(broker, "place_order", terminal_ack)

        def must_not_release(*args, **kwargs):
            raise AssertionError("cancelled/expired acknowledgement is not zero-fill proof")

        monkeypatch.setattr(risk, "on_entry_terminal_unfilled", must_not_release)
        result = guard.place_order(approval.order, approval.token)

        assert not result.accepted
        assert result.rejection.stage == "BROKER_ACK"
        assert broker.working_order_count == 0
        persisted = store.load()
        assert persisted.state is PendingEntryState.TERMINAL_REPORTED
        assert not persisted.ever_fill_observed
        assert persisted.cumulative_filled_quantity == 0


def test_missing_terminal_order_history_never_releases_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    ack = first.place_order(approval.order, approval.token).ack
    assert first.cancel_entry(ack.broker_order_id).request_sent
    # Model an eventually consistent/pruned collection endpoint before the terminal
    # event is observed. The event alone must not release either durable lock.
    monkeypatch.setattr(broker, "get_orders", lambda: [])
    first.poll_events()
    assert store.load().state is PendingEntryState.TERMINAL_REPORTED

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval
    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert store.load().state is PendingEntryState.TERMINAL_REPORTED
    assert broker.submit_count == 1


@pytest.mark.parametrize(
    ("status", "filled_quantity"),
    [
        (OrderStatus.PARTIALLY_FILLED, 0),
        (OrderStatus.FILLED, 0),
        (OrderStatus.PARTIALLY_FILLED, 3),
        (OrderStatus.FILLED, 1),
    ],
)
def test_incoherent_matched_fill_status_latches_reservation(
    tmp_path,
    monkeypatch,
    status: OrderStatus,
    filled_quantity: int,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    ack = first.place_order(approval.order, approval.token).ack
    monkeypatch.setattr(
        broker,
        "get_orders",
        lambda: [
            BrokerOrderState(
                broker_order_id=ack.broker_order_id,
                order_id=approval.order.order_id,
                status=status,
                filled_quantity=filled_quantity,
                average_fill_price=18_000.25 if filled_quantity else 0.0,
            )
        ],
    )
    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval

    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.stage == "GUARD_RESERVATION_STORE"
    assert restarted.reservation_store_error is not None
    assert store.load().order_id == approval.order.order_id
    assert broker.submit_count == 1


@pytest.mark.parametrize("conflict", ["local", "broker"])
def test_one_of_two_order_ids_matching_is_a_hard_identity_conflict(
    tmp_path,
    monkeypatch,
    conflict: str,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    ack = first.place_order(approval.order, approval.token).ack
    monkeypatch.setattr(
        broker,
        "get_orders",
        lambda: [
            BrokerOrderState(
                broker_order_id=(
                    "different-broker-id"
                    if conflict == "local"
                    else ack.broker_order_id
                ),
                order_id=(
                    approval.order.order_id
                    if conflict == "local"
                    else "different-local-id"
                ),
                status=OrderStatus.CANCELLED,
                filled_quantity=0,
                average_fill_price=0.0,
            )
        ],
    )
    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    second = restarted_risk.evaluate_entry(intent(stop=17_989.0)).approval

    result = restarted.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.stage == "GUARD_RESERVATION_STORE"
    assert "identity conflict" in restarted.reservation_store_error
    assert store.load().order_id == approval.order.order_id


def test_terminal_reservation_still_cancels_when_snapshot_says_working(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    first_risk = make_engine(tmp_path, clock, "first")
    first = GuardedBroker(broker, first_risk, reservation_store=store)
    approval = first_risk.evaluate_entry(intent()).approval
    ack = first.place_order(approval.order, approval.token).ack
    acknowledged = store.load()
    store.save(
        replace(
            acknowledged,
            state=PendingEntryState.TERMINAL_REPORTED,
            revision=acknowledged.revision + 1,
        )
    )

    restarted_risk = make_engine(tmp_path, clock, "restart")
    restarted = GuardedBroker(broker, restarted_risk, reservation_store=store)
    result = restarted.cancel_entry(ack.broker_order_id)

    assert result.accepted
    assert result.request_sent
    assert broker.working_order_count == 0
    assert store.load().state is PendingEntryState.TERMINAL_REPORTED


def test_cancelled_event_merges_fill_before_marking_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk = make_engine(tmp_path, clock, "risk")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    ack = guard.place_order(approval.order, approval.token).ack
    fill = Fill(
        fill_id="late-partial-fill",
        order_id=approval.order.order_id,
        timestamp=at(11, 1),
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        price=18_000.25,
        is_partial=True,
    )
    cancelled = BrokerEvent(
        kind=EventKind.CANCELLED,
        timestamp=at(11, 1),
        order_id=approval.order.order_id,
        broker_order_id=ack.broker_order_id,
        fill=fill,
    )
    monkeypatch.setattr(broker, "poll_events", lambda: [cancelled])
    clock.set(at(11, 1))

    events = guard.poll_events()

    assert events == [cancelled]
    persisted = store.load()
    assert persisted.state is PendingEntryState.TERMINAL_REPORTED
    assert persisted.ever_fill_observed
    assert persisted.cumulative_filled_quantity == 1
    assert persisted.observed_fill_ids == ("late-partial-fill",)


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


def test_guard_flattens_restart_exposure_without_a_local_position(tmp_path) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    first_risk = make_engine(tmp_path, clock, "first")
    first_guard = GuardedBroker(broker, first_risk)
    approval = first_risk.evaluate_entry(intent()).approval
    first_guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    first_guard.on_bar_open(market_bar(at(11, 1)))
    assert broker.get_positions()[0].quantity == approval.order.quantity

    # Model a crash before execution state reconstructed a Position. Personal-risk
    # recovery is deliberately unavailable, but an exact broker-snapshot close must
    # remain expressible and independently revalidated by the guard.
    restarted_risk = make_durable_engine(
        tmp_path,
        clock,
        FileRiskStateStore(tmp_path / "missing-risk.json"),
        bootstrap=False,
    )
    assert restarted_risk.risk_state_store_error is not None
    restarted = GuardedBroker(broker, restarted_risk)

    result = restarted.flatten_recovered_exposure(now=at(11, 1))

    assert result.accepted
    assert restarted.submitted_order(result.ack.broker_order_id).purpose.value == "FLATTEN"
    clock.set(at(11, 2))
    restarted.on_bar_open(market_bar(at(11, 2)))
    assert broker.get_positions() == []


def test_pending_entry_clears_before_next_session_personal_risk_rollover(
    tmp_path,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    reservation_store = FilePendingEntryReservationStore(tmp_path / "pending.json")
    risk_path = tmp_path / "risk.json"
    risk = make_durable_engine(
        tmp_path,
        clock,
        FileRiskStateStore(risk_path),
        bootstrap=True,
    )
    guard = GuardedBroker(broker, risk, reservation_store=reservation_store)
    assert guard.reconcile_risk_state() is None
    approval = risk.evaluate_entry(intent()).approval
    guard.place_order(approval.order, approval.token)

    clock.set(at(11, 1))
    guard.on_bar_open(market_bar(at(11, 1)))
    opened = position(quantity=approval.order.quantity)
    risk.on_position_opened(opened)
    exit_approval = risk.evaluate_exit(opened, now=at(11, 1)).approval
    guard.place_order(exit_approval.order, exit_approval.token)
    clock.set(at(11, 2))
    guard.on_bar_open(market_bar(at(11, 2), price=18_001.0))
    assert broker.get_positions() == []
    risk.on_trade_closed(
        Trade(
            trade_id="closed-before-restart",
            instrument="MNQ",
            strategy="test",
            side=Side.BUY,
            quantity=approval.order.quantity,
            entry_time=at(11, 1),
            entry_price=18_000.25,
            exit_time=at(11, 2),
            exit_price=18_001.25,
            exit_reason=ExitReason.SESSION_CLOSE,
            gross_pnl_usd=6.0,
            commission_usd=0.0,
            net_pnl_usd=6.0,
            r_multiple=0.1,
            bars_held=1,
            initial_stop=approval.token.stop_price,
            position_id=opened.position_id,
        )
    )
    assert reservation_store.load().state is PendingEntryState.FILLED

    next_session = at() + timedelta(days=1)
    clock.set(next_session)
    restarted_risk = make_durable_engine(
        tmp_path,
        clock,
        FileRiskStateStore(risk_path),
        bootstrap=False,
    )
    restarted = GuardedBroker(
        broker,
        restarted_risk,
        reservation_store=FilePendingEntryReservationStore(
            tmp_path / "pending.json"
        ),
    )

    assert restarted.reconcile_risk_state(now=next_session) is None
    assert FilePendingEntryReservationStore(tmp_path / "pending.json").load() is None
    assert not restarted_risk.risk_state_recovery_required
    next_decision = restarted_risk.evaluate_entry(
        intent(ts=next_session, stop=17_989.0),
        now=next_session,
    )
    assert next_decision.approved, next_decision.rejection


@pytest.mark.parametrize(
    "mutation",
    [
        "non-ack",
        "wrong-local-id",
        "empty-venue-id",
        "whitespace-venue-id",
        "string-status",
        "local-pending-status",
        "naive-time",
        "stale-time",
        "future-time",
        "non-text-detail",
    ],
)
def test_malformed_order_ack_is_withheld_locked_and_recovered(
    tmp_path,
    monkeypatch,
    mutation: str,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(tmp_path / f"ack-{mutation}.json")
    risk = make_engine(tmp_path, clock, f"ack-{mutation}")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    original_submit = broker.place_order

    def malformed_ack(order):
        ack = original_submit(order)
        if mutation == "non-ack":
            return {"order_id": order.order_id}
        if mutation == "wrong-local-id":
            return replace(ack, order_id="different-local-order")
        if mutation == "empty-venue-id":
            return replace(ack, broker_order_id="")
        if mutation == "whitespace-venue-id":
            return replace(ack, broker_order_id=" venue-id ")
        if mutation == "string-status":
            return replace(ack, status="ACCEPTED")
        if mutation == "local-pending-status":
            return replace(ack, status=OrderStatus.PENDING)
        if mutation == "naive-time":
            return replace(ack, accepted_at=ack.accepted_at.replace(tzinfo=None))
        if mutation == "stale-time":
            return replace(ack, accepted_at=at(10, 59))
        if mutation == "future-time":
            return replace(ack, accepted_at=at(11, 2))
        if mutation == "non-text-detail":
            return replace(ack, detail=object())
        raise AssertionError(mutation)

    monkeypatch.setattr(broker, "place_order", malformed_ack)

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert "structural_error" in result.rejection.context
    assert guard.reservation_store_error is not None
    persisted = store.load()
    assert persisted is not None
    assert persisted.order_id == approval.order.order_id
    assert persisted.state is PendingEntryState.OUTCOME_UNKNOWN
    assert risk.kill_switch.is_active()
    assert broker.working_order_count == 0


def test_ack_venue_id_collision_with_another_local_order_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    seed_long_broker_position(broker, clock)
    risk = make_engine(tmp_path, clock, "ack-collision")
    guard = GuardedBroker(broker, risk)
    local_position = position(quantity=1)
    stop = Order(
        order_id="collision-stop",
        timestamp=clock.now(),
        instrument="MNQ",
        side=Side.SELL,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=17_990.0,
        purpose=OrderPurpose.STOP,
        strategy="test",
        oco_group="collision-oco",
    )
    stop_approval = risk.evaluate_protective(local_position, stop).approval
    first = guard.place_order(stop_approval.order, stop_approval.token)
    assert first.accepted

    target = Order(
        order_id="collision-target",
        timestamp=clock.now(),
        instrument="MNQ",
        side=Side.SELL,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=18_020.0,
        purpose=OrderPurpose.TARGET,
        strategy="test",
        oco_group="collision-oco",
    )
    target_approval = risk.evaluate_protective(local_position, target).approval
    original_submit = broker.place_order
    actual_target_ids: list[str] = []

    def colliding_ack(order):
        ack = original_submit(order)
        if order.order_id == target.order_id:
            actual_target_ids.append(ack.broker_order_id)
            return replace(ack, broker_order_id=first.ack.broker_order_id)
        return ack

    monkeypatch.setattr(broker, "place_order", colliding_ack)

    result = guard.place_order(target_approval.order, target_approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert "collides" in result.rejection.context["structural_error"]
    assert guard.submitted_order(first.ack.broker_order_id).order_id == stop.order_id
    assert guard.submitted_order(actual_target_ids[0]) is None
    assert risk.kill_switch.is_active()


@pytest.mark.parametrize("purpose", [OrderPurpose.STOP, OrderPurpose.FLATTEN])
@pytest.mark.parametrize(
    "status",
    [
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    ],
)
def test_terminal_or_fill_only_ack_is_never_accepted_for_risk_reducing_orders(
    tmp_path,
    monkeypatch,
    purpose: OrderPurpose,
    status: OrderStatus,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    seed_long_broker_position(broker, clock)
    risk = make_engine(tmp_path, clock, f"{purpose.value}-{status.value}")
    guard = GuardedBroker(broker, risk)
    local_position = position(quantity=1)
    if purpose is OrderPurpose.STOP:
        order = Order(
            order_id=f"terminal-{status.value}-stop",
            timestamp=clock.now(),
            instrument="MNQ",
            side=Side.SELL,
            quantity=1,
            order_type=OrderType.STOP,
            stop_price=17_990.0,
            purpose=OrderPurpose.STOP,
            strategy="test",
            oco_group=f"terminal-{status.value}-oco",
        )
        approval = risk.evaluate_protective(local_position, order).approval
    else:
        approval = risk.evaluate_exit(
            local_position,
            reason="FLATTEN",
            now=clock.now(),
        ).approval
    original_submit = broker.place_order
    venue_ids: list[str] = []

    def terminal_ack(order):
        ack = original_submit(order)
        if order.order_id == approval.order.order_id:
            venue_ids.append(ack.broker_order_id)
            return replace(ack, status=status)
        return ack

    monkeypatch.setattr(broker, "place_order", terminal_ack)

    result = guard.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.stage == "BROKER_ACK"
    assert result.rejection.context["ack_status"] == status.value
    assert guard.submitted_order(venue_ids[0]) is None
    if status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}:
        assert guard.reservation_store_error is not None
        assert risk.kill_switch.is_active()


@pytest.mark.parametrize("kind", [EventKind.CANCEL_REQUESTED, EventKind.CANCELLED])
@pytest.mark.parametrize(
    "mutation",
    [
        "non-event",
        "string-kind",
        "missing-venue-id",
        "whitespace-local-id",
        "naive-time",
        "pre-submission-time",
        "future-time",
    ],
)
def test_fill_less_lifecycle_event_is_validated_before_dispatch(
    tmp_path,
    monkeypatch,
    kind: EventKind,
    mutation: str,
) -> None:
    clock = SimulatedClock(at())
    broker = make_sim(clock)
    store = FilePendingEntryReservationStore(
        tmp_path / f"event-{kind.value}-{mutation}.json"
    )
    risk = make_engine(tmp_path, clock, f"event-{kind.value}-{mutation}")
    guard = GuardedBroker(broker, risk, reservation_store=store)
    approval = risk.evaluate_entry(intent()).approval
    ack = guard.place_order(approval.order, approval.token).ack
    event: object = BrokerEvent(
        kind=kind,
        timestamp=at(11, 1),
        order_id=approval.order.order_id,
        broker_order_id=ack.broker_order_id,
        detail="untrusted lifecycle event",
    )
    if mutation == "non-event":
        event = {"kind": kind.value}
    elif mutation == "string-kind":
        event = replace(event, kind=kind.value)
    elif mutation == "missing-venue-id":
        event = replace(event, broker_order_id="")
    elif mutation == "whitespace-local-id":
        event = replace(event, order_id=f" {approval.order.order_id} ")
    elif mutation == "naive-time":
        event = replace(event, timestamp=at(11, 1).replace(tzinfo=None))
    elif mutation == "pre-submission-time":
        event = replace(event, timestamp=at(10, 58))
    elif mutation == "future-time":
        event = replace(event, timestamp=at(11, 2))
    else:
        raise AssertionError(mutation)
    monkeypatch.setattr(broker, "poll_events", lambda: [event])
    clock.set(at(11, 1))

    with pytest.raises(ReservationStoreError):
        guard.poll_events()

    persisted = store.load()
    assert persisted is not None
    assert persisted.state is PendingEntryState.OUTCOME_UNKNOWN
    assert persisted.order_id == approval.order.order_id
    assert guard.reservation_store_error is not None
    assert risk.kill_switch.is_active()
    assert broker.working_order_count == 0
