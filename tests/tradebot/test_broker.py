"""M6: the broker layer.

The guard tests are the ones that matter. Everything else in this project is a policy that
could in principle be forgotten at a call site; `GuardedBroker.place_order` makes the
policy part of the type signature.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from threading import Barrier

import pytest

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
from tradebot.core.models import Bar, Order, OrderIntent
from tradebot.core.types import (
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)
from tradebot.instruments.registry import get_instrument
from tradebot.broker.base import (
    BrokerAdapter,
    BrokerConnectionError,
    BrokerRateLimitError,
    BrokerTimeout,
    EventKind,
    NotConnected,
    OrderRejected,
)
from tradebot.broker.costs import CostModel
from tradebot.broker.guarded import (
    GuardedBroker,
    PendingEntryState,
    RiskViolation,
)
from tradebot.broker.simulated import SimulatedBroker
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import RiskEngine

MNQ = get_instrument("MNQ")


class AdvancingClock:
    """Return a later instant on every read, like a real wall clock."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        current = self._now
        self._now += timedelta(milliseconds=1)
        return current


def at(hour=11, minute=0, second=0, day=1) -> datetime:
    return datetime(2024, 4, day, hour, minute, second, tzinfo=MARKET_TZ)


def bar(price=18000.0, ts=None, high=None, low=None, open_=None) -> Bar:
    o = open_ if open_ is not None else price
    return Bar(
        timestamp=ts or at(),
        open=o,
        high=high if high is not None else max(o, price) + 2,
        low=low if low is not None else min(o, price) - 2,
        close=price,
        volume=100.0,
    )


@pytest.fixture
def costs() -> CostModel:
    return CostModel.from_config(CostConfig(commission_round_trip_usd=1.24,
                                            slippage_ticks_per_side=1.0), MNQ)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(at())


@pytest.fixture
def sim(costs, clock) -> SimulatedBroker:
    broker = SimulatedBroker(MNQ, costs, config=SimulatedBrokerConfig(latency_ms=0), clock=clock)
    broker.connect()
    return broker


@pytest.fixture
def engine(tmp_path, clock) -> RiskEngine:
    cfg = RiskConfig(
        starting_equity_usd=50_000.0,
        per_trade=PerTradeRisk(risk_pct_of_equity=0.5, max_risk_per_trade_usd=250.0,
                               max_contracts=3, min_contracts=1),
        daily=DailyRisk(max_daily_loss_usd=1000.0, max_daily_loss_r=4.0, max_trades_per_day=6),
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "K.flag")),
    )
    return RiskEngine(cfg, MNQ, SessionConfig(), clock=clock,
                      kill_switch=KillSwitch(tmp_path / "K.flag", clock=clock))


def an_intent(stop=17990.0, entry=18000.0, side=Side.BUY, ts=None) -> OrderIntent:
    return OrderIntent(timestamp=ts or at(), instrument="MNQ", side=side, strategy="t",
                       stop_price=stop, reference_price=entry, conditions=("c",))


# ============================================================ cost model


def test_slippage_always_moves_the_price_against_the_order(costs):
    assert costs.apply_slippage(18000.0, Side.BUY) == 18000.25
    assert costs.apply_slippage(18000.0, Side.SELL) == 17999.75


def test_commission_is_half_a_round_turn_per_fill_per_contract(costs):
    assert costs.commission_per_fill(1) == pytest.approx(0.62)
    assert costs.commission_per_fill(3) == pytest.approx(1.86)


def test_round_trip_cost_counts_both_commissions_and_both_slippages(costs):
    # $1.24 commission + 2 sides x 0.25 pt x $2 = $1.00 slippage.
    assert costs.round_trip_cost_usd(1) == pytest.approx(2.24)
    assert "round trip" in costs.describe()


def test_stress_mode_widens_slippage_only(costs):
    stressed = CostModel.from_config(
        CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0,
                   slippage_stress_multiplier=2.0),
        MNQ, stress=True,
    )
    assert stressed.slippage_points == 2 * costs.slippage_points
    assert stressed.commission_round_trip_usd == costs.commission_round_trip_usd


# ============================================================ simulated broker


def test_the_simulator_is_a_broker_adapter(sim):
    assert isinstance(sim, BrokerAdapter)
    assert sim.is_paper is True


def test_orders_are_refused_while_disconnected(sim):
    sim.disconnect()
    with pytest.raises(NotConnected):
        sim.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                              quantity=1, order_type=OrderType.MARKET))


def test_a_market_order_fills_at_the_next_bars_open_plus_slippage(sim):
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.BUY,
                  quantity=1, order_type=OrderType.MARKET)
    ack = sim.place_order(order)
    assert ack.status is OrderStatus.ACCEPTED

    events = sim.on_bar(bar(open_=18010.0, price=18012.0, ts=at(11, 1)))
    fill = events[0].fill
    assert events[0].kind is EventKind.FILL
    assert fill.price == 18010.25  # open + 1 tick of slippage
    assert fill.quantity == 1
    assert fill.slippage_points == pytest.approx(0.25)


def test_a_limit_order_only_fills_when_price_reaches_it(sim):
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.BUY, quantity=1,
                  order_type=OrderType.LIMIT, limit_price=17990.0)
    sim.place_order(order)

    assert sim.on_bar(bar(open_=18000.0, price=18001.0, low=17995.0, ts=at(11, 1))) == []
    events = sim.on_bar(bar(open_=17995.0, price=17988.0, low=17985.0, ts=at(11, 2)))
    assert events[0].fill.price == 17990.0


def test_a_gap_through_a_limit_fills_at_the_better_open(sim):
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.BUY, quantity=1,
                  order_type=OrderType.LIMIT, limit_price=17990.0)
    sim.place_order(order)
    events = sim.on_bar(bar(open_=17980.0, price=17982.0, low=17978.0, ts=at(11, 1)))
    assert events[0].fill.price == 17980.0, "a gap below a buy limit fills better, not worse"
    assert events[0].fill.slippage_points == 0.0


def test_ioc_limit_expires_at_first_open_and_a_later_wick_cannot_fill_it(sim):
    order = Order(
        order_id="ioc-missed",
        timestamp=at(),
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=17_990.0,
        time_in_force=TimeInForce.IOC,
    )
    sim.place_order(order)

    # The open is outside the signed buy limit. Even though this same bar later trades
    # through the price, IOC is terminal at the first eligible open opportunity.
    events = sim.on_bar(
        bar(
            open_=18_000.0,
            price=17_985.0,
            high=18_001.0,
            low=17_980.0,
            ts=at(11, 1),
        )
    )

    assert [event.kind for event in events] == [EventKind.CANCELLED]
    assert sim.get_positions() == []
    assert sim.working_order_count == 0
    assert sim.on_bar(
        bar(open_=17_980.0, price=17_982.0, low=17_975.0, ts=at(11, 2))
    ) == []


def test_ioc_partial_fill_cancels_remainder_in_the_same_open_phase(costs, clock):
    broker = SimulatedBroker(
        MNQ,
        costs,
        config=SimulatedBrokerConfig(
            latency_ms=0,
            partial_fill_probability=1.0,
            seed=1,
        ),
        clock=clock,
    )
    broker.connect()
    broker.place_order(
        Order(
            order_id="ioc-partial",
            timestamp=at(),
            instrument="MNQ",
            side=Side.BUY,
            quantity=4,
            order_type=OrderType.LIMIT,
            limit_price=18_001.0,
            time_in_force=TimeInForce.IOC,
        )
    )

    events = broker.on_bar(bar(open_=18_000.0, ts=at(11, 1)))

    assert [event.kind for event in events] == [
        EventKind.PARTIAL_FILL,
        EventKind.CANCELLED,
    ]
    assert events[0].fill.quantity == 2
    assert broker.get_positions()[0].quantity == 2
    assert broker.working_order_count == 0
    assert broker.on_bar(bar(open_=17_999.0, ts=at(11, 2))) == []


def test_a_gap_through_a_stop_fills_at_the_open_not_at_the_stop(sim):
    """The single most common way a backtest flatters itself."""
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.SELL, quantity=1,
                  order_type=OrderType.STOP, stop_price=17990.0)
    sim.place_order(order)
    # Bar gaps straight down through the stop.
    events = sim.on_bar(bar(open_=17950.0, price=17945.0, high=17952.0, low=17940.0,
                            ts=at(11, 1)))
    assert events[0].fill.price == 17949.75, "filled at the gapped open minus slippage"
    assert events[0].fill.slippage_points == pytest.approx(40.25)


def test_a_stop_not_reached_does_not_fill(sim):
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.SELL, quantity=1,
                  order_type=OrderType.STOP, stop_price=17900.0)
    sim.place_order(order)
    assert sim.on_bar(bar(open_=18000.0, price=18001.0, low=17950.0, ts=at(11, 1))) == []


def test_latency_delays_the_fill_by_the_configured_amount(costs, clock):
    broker = SimulatedBroker(MNQ, costs, config=SimulatedBrokerConfig(latency_ms=60_000),
                             clock=clock)
    broker.connect()
    broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                             quantity=1, order_type=OrderType.MARKET))

    assert broker.on_bar(bar(ts=at(11, 0, 30))) == [], "still in flight"
    assert broker.on_bar(bar(ts=at(11, 1, 0))) != []


def test_a_partial_fill_accumulates_across_bars(costs, clock):
    broker = SimulatedBroker(
        MNQ, costs,
        config=SimulatedBrokerConfig(latency_ms=0, partial_fill_probability=1.0, seed=1),
        clock=clock,
    )
    broker.connect()
    broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                             quantity=4, order_type=OrderType.MARKET))

    first = broker.on_bar(bar(open_=18000.0, ts=at(11, 1)))
    assert first[0].kind is EventKind.PARTIAL_FILL
    assert first[0].fill.quantity == 2 and first[0].fill.is_partial

    second = broker.on_bar(bar(open_=18000.0, ts=at(11, 2)))
    assert second[0].fill.quantity == 1

    third = broker.on_bar(bar(open_=18000.0, ts=at(11, 3)))
    assert third[0].fill.quantity == 1
    assert third[0].kind is EventKind.FILL
    assert broker.get_positions()[0].quantity == 4


def test_a_single_contract_order_can_never_partially_fill(costs, clock):
    broker = SimulatedBroker(
        MNQ, costs,
        config=SimulatedBrokerConfig(latency_ms=0, partial_fill_probability=1.0),
        clock=clock,
    )
    broker.connect()
    broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                             quantity=1, order_type=OrderType.MARKET))
    events = broker.on_bar(bar(ts=at(11, 1)))
    assert events[0].kind is EventKind.FILL


def test_an_injected_rejection_raises_and_is_reported(costs, clock):
    broker = SimulatedBroker(
        MNQ, costs, config=SimulatedBrokerConfig(latency_ms=0, reject_probability=1.0),
        clock=clock,
    )
    broker.connect()
    with pytest.raises(OrderRejected):
        broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ",
                                 side=Side.BUY, quantity=1, order_type=OrderType.MARKET))
    assert any(e.kind is EventKind.REJECTED for e in broker.poll_events())


def test_a_disconnect_does_not_silently_cancel_working_orders(sim):
    sim.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                          quantity=1, order_type=OrderType.LIMIT, limit_price=17000.0))
    assert sim.working_order_count == 1

    sim.force_disconnect()
    assert not sim.is_connected()
    assert sim.working_order_count == 1, "a dropped socket does not cancel at the exchange"

    sim.connect()
    assert sim.working_order_count == 1


def test_no_matching_happens_while_disconnected(sim):
    sim.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                          quantity=1, order_type=OrderType.MARKET))
    sim.force_disconnect()
    assert sim.on_bar(bar(ts=at(11, 1))) == []


def test_cancelling_an_unknown_order_is_not_an_error(sim):
    sim.cancel_order("nope")  # idempotent by design


def test_cancelling_removes_the_working_order(sim):
    ack = sim.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                                quantity=1, order_type=OrderType.LIMIT, limit_price=17000.0))
    sim.cancel_order(ack.broker_order_id)
    assert sim.working_order_count == 0
    assert sim.on_bar(bar(open_=16000.0, price=16001.0, low=15999.0, ts=at(11, 1))) == []


def test_a_round_trip_realises_pnl_and_returns_to_flat(sim):
    sim.place_order(Order(order_id="in", timestamp=at(), instrument="MNQ", side=Side.BUY,
                          quantity=2, order_type=OrderType.MARKET))
    sim.on_bar(bar(open_=18000.0, price=18000.0, ts=at(11, 1)))
    assert sim.get_positions()[0].quantity == 2

    sim.place_order(Order(order_id="out", timestamp=at(11, 2), instrument="MNQ",
                          side=Side.SELL, quantity=2, order_type=OrderType.MARKET))
    sim.on_bar(bar(open_=18010.0, price=18010.0, ts=at(11, 3)))

    assert sim.get_positions() == []
    account = sim.get_account()
    # Bought at 18000.25, sold at 18009.75 -> 9.5 pts x 2 ct x $2 = $38 gross.
    assert account.realized_pnl == pytest.approx(38.0)
    assert account.equity < 50_000 + 38.0, "commissions were charged"


def test_the_simulator_is_deterministic_for_a_given_seed(costs, clock):
    def run():
        broker = SimulatedBroker(
            MNQ, costs,
            config=SimulatedBrokerConfig(latency_ms=0, partial_fill_probability=0.5, seed=42),
            clock=SimulatedClock(at()),
        )
        broker.connect()
        out = []
        for i in range(10):
            broker.place_order(Order(order_id=f"o{i}", timestamp=at(), instrument="MNQ",
                                     side=Side.BUY, quantity=4, order_type=OrderType.MARKET))
            for e in broker.on_bar(bar(open_=18000.0 + i, ts=at(11, i + 1))):
                out.append((e.kind, e.fill.quantity if e.fill else None))
        return out

    assert run() == run()


# ============================================================ the guard


def test_an_order_cannot_be_placed_without_a_token(sim, engine):
    guarded = GuardedBroker(sim, engine)
    order = Order(order_id="o", timestamp=at(), instrument="MNQ", side=Side.BUY,
                  quantity=1, order_type=OrderType.MARKET)
    with pytest.raises(TypeError):
        guarded.place_order(order)  # the signature itself refuses


def test_an_approved_order_reaches_the_broker(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval

    result = guarded.place_order(approval.order, approval.token)
    assert result.accepted
    assert result.ack.status is OrderStatus.ACCEPTED
    assert guarded.submitted_count == 1


def test_a_forged_token_never_reaches_the_broker(sim, engine, tmp_path):
    """A token minted by a different engine is worthless."""
    guarded = GuardedBroker(sim, engine)
    other = RiskEngine(engine.config, MNQ, SessionConfig(), clock=engine.clock,
                       kill_switch=KillSwitch(tmp_path / "other.flag"))
    stolen = other.evaluate_entry(an_intent()).approval

    result = guarded.place_order(stolen.order, stolen.token)
    assert not result.accepted
    assert result.rejection.reason is RejectReason.TOKEN_BINDING_MISMATCH
    assert sim.submit_count == 0


def test_an_order_edited_after_approval_never_reaches_the_broker(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval
    inflated = replace(approval.order, quantity=99)

    result = guarded.place_order(inflated, approval.token)
    assert not result.accepted
    assert result.rejection.reason is RejectReason.TOKEN_BINDING_MISMATCH
    assert sim.submit_count == 0


def test_a_replayed_approval_is_refused_as_a_duplicate(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval

    assert guarded.place_order(approval.order, approval.token).accepted
    second = guarded.place_order(approval.order, approval.token)
    assert not second.accepted
    assert second.rejection.reason is RejectReason.TOKEN_ALREADY_USED
    assert sim.submit_count == 1, "the broker saw the order exactly once"


def test_an_expired_approval_is_refused(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval
    result = guarded.place_order(approval.order, approval.token,
                                 now=at() + timedelta(seconds=120))
    assert result.rejection.reason is RejectReason.TOKEN_EXPIRED


def test_an_approval_that_predates_a_breach_is_refused_at_the_guard(sim, engine):
    """The scenario the requirement is really about: a valid approval, then a breach."""
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval

    engine.state.daily_realized_pnl = -2_000.0  # cap blown between approval and submission

    result = guarded.place_order(approval.order, approval.token)
    assert not result.accepted
    assert result.rejection.reason is RejectReason.MAX_DAILY_LOSS
    assert result.rejection.stage == "GUARD"
    assert sim.submit_count == 0


def test_the_kill_switch_stops_entries_at_the_guard(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval
    engine.kill_switch.trip("manual")

    result = guarded.place_order(approval.order, approval.token)
    assert result.rejection.reason is RejectReason.KILL_SWITCH_ACTIVE
    assert sim.submit_count == 0


def test_the_kill_switch_still_lets_a_flatten_through(sim, engine, position_factory):
    # The local Position is not sufficient authority.  Seed the same exposure at the
    # broker so the guard can prove this SELL closes, rather than opens, risk.
    sim.place_order(
        Order(
            order_id="external-long",
            timestamp=at(),
            instrument="MNQ",
            side=Side.BUY,
            quantity=2,
            order_type=OrderType.MARKET,
        )
    )
    sim.on_bar(bar(ts=at(11, 1)))
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_exit(position_factory(quantity=2)).approval
    engine.kill_switch.trip("manual")

    result = guarded.place_order(approval.order, approval.token)
    assert result.accepted, "the kill switch flattens; it must never trap a position"


def test_the_guard_rejects_an_exit_fabricated_from_a_local_position(
    sim, engine, position_factory
):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_exit(position_factory(quantity=1)).approval

    result = guarded.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.INVALID_ORDER
    assert result.rejection.stage == "BROKER_SNAPSHOT"
    assert sim.submit_count == 0
    assert not engine._tokens.is_spent(approval.token)


def test_a_broker_side_rejection_becomes_a_journalled_rejection(costs, clock, engine):
    broker = SimulatedBroker(
        MNQ, costs, config=SimulatedBrokerConfig(latency_ms=0, reject_probability=1.0),
        clock=clock,
    )
    broker.connect()
    guarded = GuardedBroker(broker, engine)
    approval = engine.evaluate_entry(an_intent()).approval

    result = guarded.place_order(approval.order, approval.token)
    assert not result.accepted
    assert result.rejection.reason is RejectReason.BROKER_ERROR
    assert result.rejection.stage == "BROKER"


def test_every_refusal_is_recorded_on_the_guard(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval
    guarded.place_order(approval.order, approval.token)
    guarded.place_order(approval.order, approval.token)  # replay

    assert guarded.refused_count == 1
    assert len(guarded.rejections) == 1
    assert guarded.rejections[0].reason is RejectReason.TOKEN_ALREADY_USED


def test_cancelling_an_entry_needs_no_token(sim, engine):
    """Cancelling can only reduce pending exposure, so it is not risk-gated.

    There is deliberately no generic `cancel_order` on the guard: removing an entry and
    removing a live stop have different safety preconditions, so they are different
    methods (`cancel_entry` and `retire_protective`).
    """
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(
        an_intent(entry=18000.0, stop=17990.0)
    ).approval
    result = guarded.place_order(approval.order, approval.token)

    assert not hasattr(guarded, "cancel_order")
    assert guarded.cancel_entry(result.ack.broker_order_id).accepted


def test_read_only_methods_pass_through(sim, engine):
    guarded = GuardedBroker(sim, engine)
    assert guarded.is_connected()
    assert guarded.get_account().is_paper
    assert guarded.get_positions() == []
    assert guarded.name == "simulated"
    assert guarded.is_paper


def test_the_guard_exposes_no_public_handle_to_the_raw_adapter(sim, engine):
    guarded = GuardedBroker(sim, engine)
    public = {n for n in dir(guarded) if not n.startswith("_")}
    assert "adapter" not in public
    assert "broker" not in public
    # place_order is the only way through, and it demands a token.
    import inspect

    params = inspect.signature(GuardedBroker.place_order).parameters
    assert "token" in params
    assert params["token"].default is inspect.Parameter.empty


def test_entry_fails_closed_when_any_authoritative_broker_read_fails(
    sim, engine, monkeypatch
):
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent()).approval

    def unavailable():
        raise NotConnected("positions feed unavailable")

    monkeypatch.setattr(sim, "get_positions", unavailable)
    result = guarded.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.BROKER_ERROR
    assert result.rejection.stage == "BROKER_SNAPSHOT"
    assert sim.submit_count == 0
    assert not engine._tokens.is_spent(approval.token)


def test_broker_visible_position_blocks_entry_even_when_local_risk_is_flat(sim, engine):
    raw = Order(
        order_id="external-entry",
        timestamp=at(),
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
    )
    sim.place_order(raw)
    sim.on_bar(bar(ts=at(11, 1)))
    assert engine.state.open_position_id is None

    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent(stop=17989.0)).approval
    result = guarded.place_order(approval.order, approval.token, now=at())

    assert not result.accepted
    assert result.rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert sim.submit_count == 1
    assert not engine._tokens.is_spent(approval.token)


def test_broker_visible_working_entry_blocks_a_second_approved_entry(sim, engine):
    sim.place_order(
        Order(
            order_id="external-working-entry",
            timestamp=at(),
            instrument="MNQ",
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )
    )
    guarded = GuardedBroker(sim, engine)
    approval = engine.evaluate_entry(an_intent(stop=17989.0)).approval

    result = guarded.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert "working order" in result.rejection.detail
    assert sim.submit_count == 1
    assert not engine._tokens.is_spent(approval.token)


def test_pending_entry_reservation_blocks_a_distinct_valid_approval(sim, engine):
    guarded = GuardedBroker(sim, engine)
    first = engine.evaluate_entry(an_intent(stop=17990.0)).approval
    second = engine.evaluate_entry(an_intent(stop=17989.0)).approval

    assert guarded.place_order(first.order, first.token).accepted
    result = guarded.place_order(second.order, second.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert result.rejection.stage == "GUARD_RESERVATION"
    assert guarded.pending_entry.state is PendingEntryState.ACKNOWLEDGED
    assert sim.submit_count == 1
    assert not engine._tokens.is_spent(second.token)


def test_simultaneous_approved_entries_are_serialized_to_one_submission(sim, engine):
    guarded = GuardedBroker(sim, engine)
    approvals = (
        engine.evaluate_entry(an_intent(stop=17990.0)).approval,
        engine.evaluate_entry(an_intent(stop=17989.0)).approval,
    )
    barrier = Barrier(3)

    def submit(approval):
        barrier.wait()
        return guarded.place_order(approval.order, approval.token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, approval) for approval in approvals]
        barrier.wait()
        results = [future.result(timeout=2) for future in futures]

    assert sum(result.accepted for result in results) == 1
    assert sim.submit_count == 1
    refused = next(result for result in results if not result.accepted)
    assert refused.rejection.reason is RejectReason.POSITION_ALREADY_OPEN


def test_cancelled_pending_entry_is_released_only_after_snapshot_reconciliation(
    sim, engine
):
    guarded = GuardedBroker(sim, engine)
    first = engine.evaluate_entry(an_intent(stop=17990.0)).approval
    first_result = guarded.place_order(first.order, first.token)

    guarded.cancel_entry(first_result.ack.broker_order_id)
    assert guarded.pending_entry.state is PendingEntryState.CANCEL_REQUESTED

    second = engine.evaluate_entry(an_intent(stop=17989.0)).approval
    second_result = guarded.place_order(second.order, second.token)

    assert second_result.accepted
    assert guarded.pending_entry.order_id == second.order.order_id
    assert sim.submit_count == 2


@pytest.mark.parametrize("purpose", ["EXIT", "STOP", "TARGET"])
def test_risk_reducing_orders_fail_closed_during_snapshot_outage(
    sim, engine, position_factory, monkeypatch, purpose
):
    guarded = GuardedBroker(sim, engine)
    position = position_factory(quantity=1, target=18020.0)
    if purpose == "EXIT":
        approval = engine.evaluate_exit(position).approval
    else:
        order = Order(
            order_id=f"outage-{purpose.lower()}",
            timestamp=at(),
            instrument="MNQ",
            side=Side.SELL,
            quantity=1,
            order_type=OrderType.STOP if purpose == "STOP" else OrderType.LIMIT,
            stop_price=17990.0 if purpose == "STOP" else None,
            limit_price=18020.0 if purpose == "TARGET" else None,
            strategy=position.strategy,
            purpose=OrderPurpose[purpose],
            oco_group="outage-oco",
        )
        approval = engine.evaluate_protective(position, order).approval

    secret_error = "private-adapter-secret-position-feed-token"

    def unavailable():
        raise RuntimeError(secret_error)

    monkeypatch.setattr(sim, "get_account", unavailable)
    monkeypatch.setattr(sim, "get_positions", unavailable)
    monkeypatch.setattr(sim, "get_orders", unavailable)

    result = guarded.place_order(approval.order, approval.token)

    assert not result.accepted
    assert result.rejection.reason is RejectReason.BROKER_ERROR
    assert result.rejection.stage == "BROKER_SNAPSHOT"
    assert secret_error not in result.rejection.detail
    assert sim.submit_count == 0
    assert not engine._tokens.is_spent(approval.token)


@pytest.mark.parametrize("purpose", ["ENTRY", "EXIT", "STOP", "TARGET"])
def test_guard_verifies_at_a_post_snapshot_time_with_an_advancing_clock(
    costs, engine, tmp_path, position_factory, purpose
):
    clock = AdvancingClock(at())
    broker = SimulatedBroker(
        MNQ,
        costs,
        config=SimulatedBrokerConfig(latency_ms=0),
        clock=clock,
    )
    broker.connect()
    risk = RiskEngine(
        engine.config,
        MNQ,
        SessionConfig(),
        clock=clock,
        kill_switch=KillSwitch(
            tmp_path / f"advancing-{purpose.casefold()}.flag", clock=clock
        ),
    )
    guard = GuardedBroker(broker, risk)

    if purpose == "ENTRY":
        approval = risk.evaluate_entry(an_intent()).approval
    else:
        broker.place_order(
            Order(
                order_id=f"authoritative-long-{purpose.casefold()}",
                timestamp=at(),
                instrument="MNQ",
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.MARKET,
            )
        )
        broker.on_bar_open(bar(ts=at(11, 1)))
        position = position_factory(quantity=1, target=18_020.0)
        if purpose == "EXIT":
            approval = risk.evaluate_exit(position).approval
        else:
            order = Order(
                order_id=f"advancing-{purpose.casefold()}",
                timestamp=at(),
                instrument="MNQ",
                side=Side.SELL,
                quantity=1,
                order_type=(
                    OrderType.STOP if purpose == "STOP" else OrderType.LIMIT
                ),
                stop_price=17_990.0 if purpose == "STOP" else None,
                limit_price=18_020.0 if purpose == "TARGET" else None,
                strategy=position.strategy,
                purpose=OrderPurpose[purpose],
                oco_group="advancing-clock-oco",
            )
            approval = risk.evaluate_protective(position, order).approval

    result = guard.place_order(approval.order, approval.token)

    assert result.accepted


# ============================================================ error taxonomy


@pytest.mark.parametrize(
    "error, retryable",
    [
        (BrokerConnectionError("socket closed"), True),
        (BrokerTimeout("no response"), True),
        (BrokerRateLimitError("429", retry_after_seconds=2.0), True),
        (OrderRejected("margin"), False),
    ],
)
def test_errors_declare_whether_a_retry_makes_sense(error, retryable):
    assert error.retryable is retryable


def test_a_rate_limit_error_carries_its_backoff():
    assert BrokerRateLimitError("429", retry_after_seconds=2.5).retry_after_seconds == 2.5


def test_a_risk_violation_carries_the_machine_readable_rejection(sim, engine):
    from tradebot.core.models import Rejection

    rejection = Rejection(timestamp=at(), reason=RejectReason.MAX_DAILY_LOSS,
                          detail="cap hit", stage="GUARD")
    err = RiskViolation(rejection)
    assert err.rejection.reason is RejectReason.MAX_DAILY_LOSS
    assert "MAX_DAILY_LOSS" in str(err)
