"""M7: the execution engine and the journal.

Covers the failure modes PROJECT_SPEC §11 names explicitly — API failures, disconnects,
duplicate orders, partial fills, stale data and restart recovery — plus the ordering
guarantees that make a backtest believable.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tradebot.broker.base import BrokerEvent, EventKind, NotConnected, OrderRejected
from tradebot.broker.costs import CostModel
from tradebot.broker.guarded import GuardResult, GuardedBroker
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
from tradebot.core.models import AccountSnapshot, Bar, Fill, OrderIntent, Rejection
from tradebot.core.types import (
    ExitReason,
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
)
from tradebot.execution.engine import ExecutionEngine
from tradebot.features.pipeline import build_features
from tradebot.instruments.registry import get_instrument
from tradebot.journal.db import Journal
from tradebot.journal.queries import JournalReader
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import RiskDecision, RiskEngine
from tradebot.risk.reservations import (
    FilePendingEntryReservationStore,
    PendingEntryState,
    ReservationStoreError,
)
from tradebot.risk.state_store import FileRiskStateStore
from tradebot.strategy.base import Strategy, StrategySpec, TradingHours

MNQ = get_instrument("MNQ")


def at(hour=11, minute=0, day=1) -> datetime:
    return datetime(2024, 4, day, hour, minute, tzinfo=MARKET_TZ)


# ---------------------------------------------------------------- a scriptable strategy


class ScriptedStrategy(Strategy):
    """Fires exactly when told to. Keeps engine tests about the engine."""

    def __init__(self, script: dict[int, tuple[Side, float, float | None]]) -> None:
        super().__init__(
            StrategySpec(
                name="scripted",
                description="test double",
                entry_conditions=("scripted",),
                invalidation_conditions=(),
                stop_loss="scripted",
                profit_target="scripted",
                filters=(),
                max_trades_per_session=99,
                trading_hours=TradingHours(datetime(2000, 1, 1, 9, 31).time(),
                                           datetime(2000, 1, 1, 15, 55).time(),
                                           datetime(2000, 1, 1, 15, 58).time()),
                params={},
            )
        )
        self.script = script
        self.managed: list[int] = []

    def _entry_signal(self, ctx):
        entry = self.script.get(ctx.i)
        if entry is None:
            return None
        side, stop, target = entry
        return self._make_intent(ctx, side, stop, target_price=target,
                                 conditions=("scripted",))

    def manage(self, ctx, position):
        self.managed.append(ctx.i)
        return super().manage(ctx, position)


# ---------------------------------------------------------------- fixtures


def make_bars(closes, *, day="2024-04-01", start="09:30", wick=3.0) -> pd.DataFrame:
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="1min", tz=MARKET_TZ,
                        name="timestamp")
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + wick for o, c in zip(opens, closes)],
            "low": [min(o, c) - wick for o, c in zip(opens, closes)],
            "close": list(closes),
            "volume": [100.0] * len(closes),
        },
        index=idx,
    )


@pytest.fixture
def rig(tmp_path):
    """A fully wired engine over a simulated broker, with a journal."""

    def build(bars: pd.DataFrame, script: dict, *, sim_config=None, equity=50_000.0,
              journal_path=None):
        clock = SimulatedClock(bars.index[0].to_pydatetime())
        costs = CostModel.from_config(
            CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0), MNQ
        )
        risk_cfg = RiskConfig(
            starting_equity_usd=equity,
            per_trade=PerTradeRisk(risk_pct_of_equity=1.0, max_risk_per_trade_usd=500.0,
                                   max_contracts=3, min_contracts=1),
            daily=DailyRisk(max_daily_loss_usd=2000.0, max_daily_loss_r=10.0,
                            max_trades_per_day=20),
            kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "K.flag")),
        )
        risk = RiskEngine(risk_cfg, MNQ, SessionConfig(entry_open_buffer_minutes=1,
                                                      entry_close_buffer_minutes=2),
                          clock=clock, kill_switch=KillSwitch(tmp_path / "K.flag", clock=clock),
                          starting_equity=equity)
        sim = SimulatedBroker(MNQ, costs,
                              config=sim_config or SimulatedBrokerConfig(latency_ms=0),
                              clock=clock, starting_equity=equity)
        sim.connect()
        guarded = GuardedBroker(sim, risk)
        journal = Journal(journal_path or (tmp_path / "j.sqlite3"), run_id="test-run")
        journal.start_run(mode="BACKTEST", instrument="MNQ", strategy="scripted",
                          broker="simulated", config={}, started_at=bars.index[0])

        strategy = ScriptedStrategy(script)
        engine = ExecutionEngine(strategy, risk, guarded, MNQ, costs, journal=journal,
                                 starting_equity=equity)
        return engine, sim, journal, clock

    return build


def run(engine, bars, clock, *, warmup=0, stop_after=None, start=0):
    features = build_features(bars).frame
    for i in range(start, len(bars)):
        if stop_after is not None and i > stop_after:
            break
        row = bars.iloc[i]
        bar = Bar(bars.index[i].to_pydatetime(), row.open, row.high, row.low, row.close,
                  row.volume)
        clock.set(bar.timestamp)
        engine.on_bar(bar, features.iloc[i], index=i, frame=features, bars=bars,
                      warmed_up=i >= warmup)
    return engine


def seed_broker_only_long(engine, sim, clock, bars, *, signal_index=2, fill_index=3):
    """Create venue exposure without giving ExecutionEngine fictional provenance."""

    signal_time = bars.index[signal_index].to_pydatetime()
    clock.set(signal_time)
    decision = engine.risk.evaluate_entry(
        OrderIntent(
            timestamp=signal_time,
            instrument="MNQ",
            side=Side.BUY,
            strategy="restart-seed",
            stop_price=17_980.0,
            target_price=18_100.0,
            reference_price=18_000.0,
            conditions=("broker-only seed",),
        ),
        equity=engine.state.equity,
        position=None,
        now=signal_time,
    )
    assert decision.approved
    result = engine.broker.place_order(
        decision.approval.order,
        decision.approval.token,
        now=signal_time,
    )
    assert result.accepted

    row = bars.iloc[fill_index]
    fill_bar = Bar(
        bars.index[fill_index].to_pydatetime(),
        row.open,
        row.high,
        row.low,
        row.close,
        row.volume,
    )
    clock.set(fill_bar.timestamp)
    engine.broker.on_bar_open(fill_bar)
    assert sim.get_positions()[0].quantity == decision.approval.order.quantity
    assert engine.state.position is None
    assert engine.state.last_bar is None
    return fill_bar


def durable_pending_restart_rig(tmp_path, bars):
    """Return a fresh engine whose guard restored an ACKNOWLEDGED IOC entry."""

    signal_time = bars.index[2].to_pydatetime()
    clock = SimulatedClock(signal_time)
    costs = CostModel.from_config(
        CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0),
        MNQ,
    )
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
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "durable-stop.flag")),
    )
    session = SessionConfig()
    sim = SimulatedBroker(
        MNQ,
        costs,
        config=SimulatedBrokerConfig(latency_ms=0),
        clock=clock,
    )
    sim.connect()
    risk_store = FileRiskStateStore(tmp_path / "durable-risk.json")
    pending_store = FilePendingEntryReservationStore(tmp_path / "durable-pending.json")

    def make_risk(*, bootstrap):
        return RiskEngine(
            config,
            MNQ,
            session,
            clock=clock,
            kill_switch=KillSwitch(tmp_path / "durable-stop.flag", clock=clock),
            expected_broker_account_id="SIM-1",
            expected_broker_name="simulated",
            expected_broker_is_paper=True,
            expected_broker_route="simulated-local-paper",
            risk_state_store=risk_store,
            bootstrap_risk_state_store=bootstrap,
            risk_state_context_id="execution-restart-recovery:stage0",
        )

    first_risk = make_risk(bootstrap=True)
    first_guard = GuardedBroker(
        sim,
        first_risk,
        reservation_store=pending_store,
    )
    assert first_guard.reconcile_risk_state(now=signal_time) is None
    decision = first_risk.evaluate_entry(
        OrderIntent(
            timestamp=signal_time,
            instrument="MNQ",
            side=Side.BUY,
            strategy="durable-ioc",
            stop_price=17_980.0,
            target_price=18_100.0,
            reference_price=18_000.0,
            conditions=("durable pending restart",),
        ),
        now=signal_time,
    )
    assert decision.approved
    submitted = first_guard.place_order(
        decision.approval.order,
        decision.approval.token,
        now=signal_time,
    )
    assert submitted.accepted
    assert pending_store.load().state is PendingEntryState.ACKNOWLEDGED

    restarted_risk = make_risk(bootstrap=False)
    restarted_guard = GuardedBroker(
        sim,
        restarted_risk,
        reservation_store=pending_store,
    )
    assert restarted_guard.pending_entry.state is PendingEntryState.ACKNOWLEDGED
    journal = Journal(tmp_path / "durable-restart.sqlite3", run_id="durable-restart")
    journal.start_run(
        mode="BACKTEST",
        instrument="MNQ",
        strategy="scripted",
        broker="simulated",
        config={},
        started_at=signal_time,
    )
    engine = ExecutionEngine(
        ScriptedStrategy({}),
        restarted_risk,
        restarted_guard,
        MNQ,
        costs,
        journal=journal,
    )
    return engine, sim, journal, clock, pending_store


# ============================================================ the happy path


def test_a_signal_never_fills_on_the_bar_that_produced_it(rig):
    bars = make_bars([18000 + i for i in range(20)])
    engine, sim, journal, clock = rig(bars, {5: (Side.BUY, 17990.0, 18050.0)})
    run(engine, bars, clock)

    trades = engine.state.trades
    position_or_trade = engine.state.position or (trades[0] if trades else None)
    assert position_or_trade is not None
    entry_time = position_or_trade.entry_time
    assert entry_time > bars.index[5], "the signal bar cannot also be the fill bar"
    assert entry_time == bars.index[6]


def test_bounded_entry_limit_fills_at_the_next_bars_open_without_adverse_slippage(rig):
    bars = make_bars([18000 + i for i in range(20)])
    engine, sim, journal, clock = rig(bars, {5: (Side.BUY, 17990.0, 18050.0)})
    run(engine, bars, clock)

    position = engine.state.position
    expected = MNQ.round_to_tick(bars["open"].iloc[6])
    assert position.entry_price == expected


def test_a_long_stop_can_fill_later_on_the_entry_bar(rig):
    bars = make_bars([18000.0] * 10)
    bars.iloc[5, bars.columns.get_loc("low")] = 17900.0
    engine, sim, journal, clock = rig(
        bars, {4: (Side.BUY, 17990.0, 18100.0)},
        sim_config=SimulatedBrokerConfig(latency_ms=120),
    )

    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.entry_time == bars.index[5]
    assert trade.exit_time == bars.index[5]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert engine.state.position is None


def test_a_short_stop_can_fill_later_on_the_entry_bar(rig):
    bars = make_bars([18000.0] * 10)
    bars.iloc[5, bars.columns.get_loc("high")] = 18100.0
    engine, sim, journal, clock = rig(bars, {4: (Side.SELL, 18010.0, 17900.0)})

    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.entry_time == bars.index[5]
    assert trade.exit_time == bars.index[5]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert engine.state.position is None


def test_entry_bar_spanning_stop_and_target_takes_the_stop_first(rig):
    bars = make_bars([18000.0] * 10)
    bars.iloc[5, bars.columns.get_loc("low")] = 17900.0
    bars.iloc[5, bars.columns.get_loc("high")] = 18100.0
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17990.0, 18010.0)})

    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    assert engine.state.trades[0].exit_reason is ExitReason.STOP_LOSS


def test_a_protective_stop_and_target_rest_at_the_venue_immediately(rig):
    bars = make_bars([18000 + i for i in range(20)])
    engine, sim, journal, clock = rig(bars, {5: (Side.BUY, 17990.0, 18050.0)})
    run(engine, bars, clock, stop_after=7)

    assert engine.state.stop_order is not None
    assert engine.state.target_order is not None
    assert engine.state.stop_order.order_type is OrderType.STOP
    assert engine.state.target_order.order_type is OrderType.LIMIT
    assert engine.state.stop_order.oco_group == engine.state.target_order.oco_group


def test_missing_entry_intent_immediately_submits_an_emergency_flatten(rig, monkeypatch):
    bars = make_bars([18000.0] * 12)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    monkeypatch.setattr(engine, "_intent_for", lambda entry: None)

    run(engine, bars, clock, stop_after=5)

    assert engine.state.position is not None
    assert engine.state.stop_order is None
    assert engine.state.pending_exit is not None
    assert engine.risk.kill_switch.is_active()
    assert any(
        order.purpose is OrderPurpose.FLATTEN and order.status.is_working
        for order in engine._orders.values()
    ), "the unprotected fill must immediately create a working flatten order"


def test_a_refused_initial_stop_immediately_submits_an_emergency_flatten(
    rig, monkeypatch
):
    bars = make_bars([18000.0] * 12)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    original_place = engine.broker.place_order

    def refuse_stop(order, token, *, now=None):
        if order.purpose is OrderPurpose.STOP:
            return GuardResult(
                ack=None,
                rejection=Rejection(
                    timestamp=now,
                    reason=RejectReason.BROKER_ERROR,
                    detail="venue refused protective stop",
                    stage="BROKER",
                    instrument=order.instrument,
                    strategy=order.strategy,
                    order_id=order.order_id,
                ),
            )
        return original_place(order, token, now=now)

    monkeypatch.setattr(engine.broker, "place_order", refuse_stop)
    run(engine, bars, clock, stop_after=5)

    assert engine.state.stop_order is None
    assert engine.state.pending_exit is not None
    assert any(
        order.purpose is OrderPurpose.FLATTEN and order.status.is_working
        for order in engine._orders.values()
    )
    events = JournalReader(journal.path, run_id="test-run").events(level="ERROR")
    assert {event["kind"] for event in events} >= {"PROTECTION_REFUSED"}


def test_terminal_stop_ack_fails_to_engine_and_engages_emergency_flatten(
    rig,
    monkeypatch,
):
    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
    )
    original_submit = sim.place_order

    def terminal_stop_ack(order):
        ack = original_submit(order)
        if order.purpose is OrderPurpose.STOP:
            return replace(ack, status=OrderStatus.CANCELLED)
        return ack

    monkeypatch.setattr(sim, "place_order", terminal_stop_ack)

    run(engine, bars, clock, stop_after=5)

    assert engine.state.position is not None
    assert engine.state.stop_order is None
    assert engine.state.pending_exit is not None
    assert engine.risk.kill_switch.is_active()
    assert any(
        order.purpose is OrderPurpose.FLATTEN and order.status.is_working
        for order in engine._orders.values()
    )
    errors = JournalReader(journal.path, run_id="test-run").events(level="ERROR")
    assert {event["kind"] for event in errors} >= {"PROTECTION_REFUSED"}


@pytest.mark.parametrize(
    ("protective_name", "wrong_field"),
    [("target_order", "side"), ("stop_order", "instrument")],
)
def test_malformed_protective_fill_is_withheld_before_local_trade_closes(
    rig,
    monkeypatch,
    protective_name: str,
    wrong_field: str,
):
    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
    )
    run(engine, bars, clock, stop_after=5)
    original_position = engine.state.position
    assert original_position is not None
    protective = getattr(engine.state, protective_name)
    assert protective is not None
    broker_order_id = engine._broker_ids[protective.order_id]
    expected_price = (
        protective.limit_price
        if protective.limit_price is not None
        else protective.stop_price
    )
    assert expected_price is not None
    fill = Fill(
        fill_id=f"malformed-{wrong_field}-protective-fill",
        order_id=protective.order_id,
        timestamp=clock.now(),
        instrument="MES" if wrong_field == "instrument" else protective.instrument,
        side=Side.BUY if wrong_field == "side" else protective.side,
        quantity=protective.quantity,
        price=expected_price,
        broker_fill_id=f"venue-malformed-{wrong_field}",
        is_partial=False,
    )
    event = BrokerEvent(
        kind=EventKind.FILL,
        timestamp=clock.now(),
        order_id=protective.order_id,
        broker_order_id=broker_order_id,
        fill=fill,
        detail="crafted protective fill",
    )
    monkeypatch.setattr(sim, "poll_events", lambda: [event])

    with pytest.raises(ReservationStoreError, match=wrong_field):
        engine.broker.poll_events()

    assert engine.state.position is original_position
    assert engine.state.trades == []
    assert engine.risk.kill_switch.is_active()


@pytest.mark.parametrize(
    ("side", "stop", "target", "wider"),
    [
        (Side.BUY, 17980.0, 18100.0, 17979.75),
        (Side.SELL, 18020.0, 17900.0, 18020.25),
    ],
)
def test_stop_moves_refuse_widening_for_long_and_short(
    rig, side, stop, target, wider
):
    bars = make_bars([18000.0] * 12)
    engine, sim, journal, clock = rig(bars, {4: (side, stop, target)})
    run(engine, bars, clock, stop_after=6)
    old_order = engine.state.stop_order
    old_price = engine.state.position.stop_price

    engine._move_stop(engine.state.last_bar, wider, "unsafe request")

    assert engine.state.position.stop_price == old_price
    assert engine.state.stop_order is old_order
    events = JournalReader(journal.path, run_id="test-run").events()
    assert any(event["kind"] == "STOP_MOVE_REFUSED" for event in events)


def test_failed_stop_replacement_retains_the_old_working_stop(rig, monkeypatch):
    bars = make_bars([18000.0] * 12)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=6)
    old_order = engine.state.stop_order
    original_evaluate = engine.risk.evaluate_protective

    def refuse_replacement(position, order, *, now=None):
        if order.purpose is OrderPurpose.STOP and order.stop_price == 17990.0:
            return RiskDecision(
                rejection=Rejection(
                    timestamp=now,
                    reason=RejectReason.INVALID_ORDER,
                    detail="test replacement refusal",
                    stage="RISK",
                    instrument=order.instrument,
                    strategy=order.strategy,
                    order_id=order.order_id,
                )
            )
        return original_evaluate(position, order, now=now)

    monkeypatch.setattr(engine.risk, "evaluate_protective", refuse_replacement)
    engine._move_stop(engine.state.last_bar, 17990.0, "tighten")

    assert engine.state.stop_order is old_order
    assert engine.state.position.stop_price == 17980.0
    old_broker_state = next(o for o in sim.get_orders() if o.order_id == old_order.order_id)
    assert old_broker_state.status.is_working


def test_stop_replacement_is_acknowledged_before_the_old_stop_is_cancelled(
    rig, monkeypatch
):
    bars = make_bars([18000.0] * 12)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=6)
    old_order = engine.state.stop_order
    original_place = engine.broker.place_order
    old_was_working_at_submit = []

    def observe_replacement(order, token, *, now=None):
        if order.purpose is OrderPurpose.STOP and order.stop_price == 17990.0:
            broker_old = next(o for o in sim.get_orders() if o.order_id == old_order.order_id)
            old_was_working_at_submit.append(broker_old.status.is_working)
        return original_place(order, token, now=now)

    monkeypatch.setattr(engine.broker, "place_order", observe_replacement)
    engine._move_stop(engine.state.last_bar, 17990.0, "tighten")

    assert old_was_working_at_submit == [True]
    assert engine.state.stop_order.order_id != old_order.order_id
    assert engine.state.position.stop_price == 17990.0
    broker_old = next(o for o in sim.get_orders() if o.order_id == old_order.order_id)
    assert broker_old.status is OrderStatus.CANCELLED


def test_hitting_the_target_closes_the_trade_and_cancels_the_stop(rig):
    closes = [18000] * 6 + [18010, 18030, 18060, 18060, 18060]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.exit_price == 18050.0
    assert engine.state.stop_order is None and engine.state.target_order is None
    assert engine.state.position is None


def test_hitting_the_stop_closes_the_trade_and_cancels_the_target(rig):
    closes = [18000] * 6 + [17990, 17970, 17950, 17950, 17950]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.net_pnl_usd < 0
    assert engine.state.stop_order is None and engine.state.target_order is None


def test_when_one_bar_spans_both_the_stop_is_assumed_first(rig):
    """The intra-bar path is unrecoverable from OHLCV; take the pessimistic branch."""
    bars = make_bars([18000] * 6 + [18000, 18000, 18000], wick=3.0)
    # Bar 7 sweeps from far below the stop to far above the target.
    bars.iloc[7, bars.columns.get_loc("low")] = 17900.0
    bars.iloc[7, bars.columns.get_loc("high")] = 18200.0

    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    assert len(engine.state.trades) == 1
    assert engine.state.trades[0].exit_reason is ExitReason.STOP_LOSS


def test_a_short_trade_makes_money_when_price_falls(rig):
    closes = [18000] * 6 + [17990, 17970, 17940, 17940]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.SELL, 18020.0, 17950.0)})
    run(engine, bars, clock)

    trade = engine.state.trades[0]
    assert trade.side is Side.SELL
    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.net_pnl_usd > 0


def test_the_r_multiple_is_computed_from_the_actual_risk_taken(rig):
    closes = [18000] * 6 + [18010, 18030, 18060, 18060]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    trade = engine.state.trades[0]
    risk_usd = abs(trade.entry_price - trade.initial_stop) * trade.quantity * MNQ.multiplier
    assert trade.r_multiple == pytest.approx(trade.net_pnl_usd / risk_usd)


def test_completed_trade_equity_reconciles_to_the_broker_and_net_pnl(rig):
    closes = [18000] * 6 + [18020] * 5
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18010.0)})

    run(engine, bars, clock)

    expected = 50_000.0 + sum(t.net_pnl_usd for t in engine.state.trades)
    assert engine.state.position is None
    assert engine.state.equity == pytest.approx(expected)
    assert sim.get_account().equity == pytest.approx(expected)


# ============================================================ forced exits


def test_a_position_is_flattened_before_the_close_and_never_held_overnight(rig):
    closes = [18000] * 400
    bars = make_bars(closes)  # 09:30 + 400 minutes runs past 16:00
    engine, sim, journal, clock = rig(bars, {10: (Side.BUY, 17900.0, 19000.0)})
    run(engine, bars, clock)

    assert engine.state.position is None
    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.exit_reason is ExitReason.SESSION_CLOSE
    assert trade.exit_time.time() <= datetime(2000, 1, 1, 16, 0).time()


def test_the_kill_switch_flattens_an_open_position(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17900.0, 19000.0)})
    features = build_features(bars).frame

    for i in range(10):
        row = bars.iloc[i]
        bar = Bar(bars.index[i].to_pydatetime(), row.open, row.high, row.low, row.close,
                  row.volume)
        clock.set(bar.timestamp)
        engine.on_bar(bar, features.iloc[i], index=i, frame=features, bars=bars)
        if i == 7:
            engine.risk.kill_switch.trip("operator pressed STOP")

    assert engine.state.position is None
    assert engine.state.trades[0].exit_reason is ExitReason.KILL_SWITCH


def test_a_forced_exit_beats_the_strategy(rig):
    """Step 3 runs before step 4: a strategy cannot talk the engine out of a mandatory exit."""
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17900.0, 19000.0)})
    features = build_features(bars).frame

    for i in range(12):
        row = bars.iloc[i]
        bar = Bar(bars.index[i].to_pydatetime(), row.open, row.high, row.low, row.close,
                  row.volume)
        clock.set(bar.timestamp)
        engine.on_bar(bar, features.iloc[i], index=i, frame=features, bars=bars)
        if i == 7:
            engine.risk.kill_switch.trip("halt")
    # The strategy's manage() was never consulted on the bar that forced the exit.
    assert 8 not in engine.strategy.managed


def test_stop_trading_cancels_an_unfilled_entry_before_the_next_match(rig):
    bars = make_bars([18_000.0] * 12, wick=20.0)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
    )
    run(engine, bars, clock, stop_after=4)
    assert engine.state.entry_order is not None
    assert sim.working_order_count == 1

    engine.flatten_now(ExitReason.MANUAL, "operator pressed STOP TRADING")

    assert engine.risk.kill_switch.is_active()
    assert engine.state.entry_order is None
    assert sim.working_order_count == 0
    run(engine, bars, clock, start=5, stop_after=6)
    assert engine.state.position is None
    assert sim.get_positions() == []
    assert engine.state.trades == []


def test_stop_without_a_bar_flattens_a_known_local_position(rig):
    bars = make_bars([18_000.0] * 14)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
    )
    run(engine, bars, clock, stop_after=7)
    assert engine.state.position is not None

    engine.state.last_bar = None
    engine.flatten_now(ExitReason.MANUAL, "STOP before next market bar")

    flatten_orders = [
        order
        for order in engine._orders.values()
        if order.purpose is OrderPurpose.FLATTEN
    ]
    assert len(flatten_orders) == 1
    assert engine.state.pending_exit is not None
    assert engine.risk.kill_switch.is_active()


def test_stop_with_flat_local_state_recovers_broker_long_before_first_bar(rig):
    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(bars, {})
    seed_broker_only_long(engine, sim, clock, bars)

    engine.flatten_now(ExitReason.MANUAL, "operator STOP after local-state loss")

    recovery_orders = [
        order
        for order in engine._orders.values()
        if order.strategy == "broker-recovery-flatten"
    ]
    assert len(recovery_orders) == 1
    assert recovery_orders[0].purpose is OrderPurpose.FLATTEN
    assert engine.state.position is None
    assert engine.state.last_bar is None
    assert engine.risk.kill_switch.is_active()

    run(engine, bars, clock, start=4, stop_after=4)

    assert sim.get_positions() == []
    assert engine.state.position is None
    assert engine.state.trades == []
    assert engine.state.bot_state.value == "KILLED"
    events = JournalReader(journal.path, run_id="test-run").events()
    kinds = {event["kind"] for event in events}
    assert "RECOVERY_FLATTEN_SUBMITTED" in kinds
    assert "RECOVERY_FLAT_CONFIRMED" in kinds


@pytest.mark.parametrize("entrypoint", ["stop", "reconcile"])
def test_fresh_engine_cancels_durable_acknowledged_ioc_before_declaring_flat(
    tmp_path, entrypoint
):
    bars = make_bars([18_000.0] * 8)
    engine, sim, journal, clock, pending_store = durable_pending_restart_rig(
        tmp_path,
        bars,
    )

    if entrypoint == "stop":
        engine.flatten_now(ExitReason.MANUAL, "durable pending IOC")
    else:
        report = engine.reconcile(clock.now())
        assert report["action"] == "in_sync"

    assert engine.broker.pending_entry is None
    assert pending_store.load() is None
    assert sim.working_order_count == 0
    assert engine.risk.kill_switch.is_active()

    run(engine, bars, clock, start=3, stop_after=3)
    assert sim.get_positions() == []
    assert engine.state.position is None
    events = JournalReader(journal.path, run_id="durable-restart").events()
    assert not any(event["kind"] == "UNTRACKED_BROKER_FILL" for event in events)


@pytest.mark.parametrize("entrypoint", ["stop", "reconcile"])
def test_durable_ioc_cancel_race_fill_is_recovered_without_fabricated_trade(
    tmp_path, monkeypatch, entrypoint
):
    bars = make_bars([18_000.0] * 8)
    engine, sim, journal, clock, pending_store = durable_pending_restart_rig(
        tmp_path,
        bars,
    )
    # Model an asynchronous venue: the cancel command returns, but the IOC remains
    # eligible and wins the race at the next open before terminal cancellation arrives.
    monkeypatch.setattr(sim, "cancel_order", lambda broker_order_id: None)

    if entrypoint == "stop":
        engine.flatten_now(ExitReason.MANUAL, "injected cancel race")
    else:
        report = engine.reconcile(clock.now())
        assert report["action"] == "pending_entry_unresolved"

    assert engine.broker.pending_entry.state is PendingEntryState.CANCEL_REQUESTED
    assert sim.working_order_count == 1
    assert engine.risk.kill_switch.is_active()

    # First open fills the unknown restored IOC and submits an exact recovery flatten;
    # second open fills that reduce-only close and completes authoritative confirmation.
    run(engine, bars, clock, start=3, stop_after=4)

    assert sim.get_positions() == []
    assert sim.working_order_count == 0
    assert engine.broker.pending_entry is None
    assert pending_store.load() is None
    assert engine.state.position is None
    assert engine.state.trades == []
    events = JournalReader(journal.path, run_id="durable-restart").events()
    kinds = {event["kind"] for event in events}
    assert "UNTRACKED_BROKER_FILL" in kinds
    assert "RECOVERY_FLATTEN_SUBMITTED" in kinds
    assert "RECOVERY_FLAT_CONFIRMED" in kinds


# ============================================================ partial fills


def test_a_partially_filled_entry_still_gets_full_protection(rig):
    bars = make_bars([18000] * 25)
    engine, sim, journal, clock = rig(
        bars, {4: (Side.BUY, 17980.0, 18100.0)},
        sim_config=SimulatedBrokerConfig(latency_ms=0, partial_fill_probability=1.0, seed=1),
    )
    run(engine, bars, clock, stop_after=12)

    position = engine.state.position
    assert position is not None
    assert engine.state.stop_order is not None
    assert engine.state.stop_order.quantity == position.quantity, (
        "the protective order must cover the whole position, not the first fill"
    )


def test_terminal_cancel_with_late_fill_applies_economics_once_and_protects(rig):
    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
        sim_config=SimulatedBrokerConfig(
            latency_ms=0,
            partial_fill_probability=1.0,
            seed=1,
        ),
    )
    run(engine, bars, clock, stop_after=4)

    row = bars.iloc[5]
    next_bar = Bar(
        bars.index[5].to_pydatetime(),
        row.open,
        row.high,
        row.low,
        row.close,
        row.volume,
    )
    clock.set(next_bar.timestamp)
    engine.broker.on_bar_open(next_bar)
    venue_events = engine.broker.poll_events()
    partial = next(
        event for event in venue_events if event.kind is EventKind.PARTIAL_FILL
    )
    terminal = BrokerEvent(
        kind=EventKind.CANCELLED,
        timestamp=partial.timestamp,
        order_id=partial.order_id,
        broker_order_id=partial.broker_order_id,
        fill=partial.fill,
        detail="cancel confirmed with late partial fill",
    )

    engine._apply_events([terminal, terminal])

    assert engine.state.position is not None
    assert engine.state.position.quantity == partial.fill.quantity == 1
    assert engine.state.entry_order is None
    assert engine.state.stop_order is not None
    assert engine.state.stop_order.quantity == 1
    assert sim.get_positions()[0].quantity == 1
    fills = JournalReader(journal.path, run_id="test-run").fills()
    assert [fill["fill_id"] for fill in fills] == [partial.fill.fill_id]


def test_a_partial_ioc_entry_is_revalidated_once_and_never_adds_later(
    rig, monkeypatch
):
    # Even when every later open remains inside the signed limit, IOC permits exactly one
    # eligible open match and terminally cancels the unfilled remainder.
    bars = make_bars([18000 + i * 0.25 for i in range(25)])
    engine, sim, journal, clock = rig(
        bars, {4: (Side.BUY, 17900.0, 19000.0)},
        sim_config=SimulatedBrokerConfig(latency_ms=0, partial_fill_probability=1.0, seed=1),
    )
    observed_quantities: list[int] = []
    original_position_opened = engine.risk.on_position_opened

    def observe_position(position):
        observed_quantities.append(position.quantity)
        return original_position_opened(position)

    monkeypatch.setattr(engine.risk, "on_position_opened", observe_position)
    run(engine, bars, clock, stop_after=14)

    position = engine.state.position
    assert position is not None and position.quantity == 1
    fills = JournalReader(journal.path, run_id="test-run").fills()
    entry_fills = [f for f in fills if f["side"] == 1]
    assert len(entry_fills) == 1
    weighted = sum(f["price"] * f["quantity"] for f in entry_fills) / sum(
        f["quantity"] for f in entry_fills
    )
    assert position.entry_price == pytest.approx(weighted)
    assert observed_quantities == [1]
    assert engine.state.entry_order is None
    assert engine.risk.state.trades_today == 1


def test_partial_exits_accumulate_and_terminal_fill_reconciles_broker_equity(
    rig, monkeypatch
):
    bars = make_bars([18000.0] * 8 + [18020.0] * 10)
    engine, sim, journal, clock = rig(
        bars, {4: (Side.BUY, 17980.0, 18010.0)},
        sim_config=SimulatedBrokerConfig(
            latency_ms=0, partial_fill_probability=0.0, seed=1
        ),
    )
    reconciled_at = []
    original_reconcile = engine.broker.reconcile_risk_state

    def observe_reconcile(*, now=None):
        reconciled_at.append(now)
        return original_reconcile(now=now)

    monkeypatch.setattr(engine.broker, "reconcile_risk_state", observe_reconcile)

    run(engine, bars, clock, stop_after=7)
    assert engine.state.position is not None
    assert engine.state.position.quantity == 3
    sim.config = replace(sim.config, partial_fill_probability=1.0)
    run(engine, bars, clock, start=8)

    assert len(engine.state.trades) == 1
    trade = engine.state.trades[0]
    assert trade.quantity == 3
    expected = 50_000.0 + trade.net_pnl_usd
    assert engine.state.equity == pytest.approx(expected)
    assert sim.get_account().equity == pytest.approx(expected)
    assert engine.risk.state.equity == pytest.approx(expected)
    assert engine.risk.state.daily_realized_pnl == pytest.approx(trade.net_pnl_usd)
    assert reconciled_at == [trade.exit_time]


def test_market_slippage_accumulates_across_partial_entry_and_exit_fills(rig):
    bars = make_bars([18000.0] * 20, wick=3.0)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17980.0, 18100.0)},
        sim_config=SimulatedBrokerConfig(
            latency_ms=0, partial_fill_probability=0.0, seed=1
        ),
    )

    run(engine, bars, clock, stop_after=8)
    assert engine.state.position is not None
    assert engine.state.position.quantity == 3
    sim.config = replace(sim.config, partial_fill_probability=1.0)
    engine.flatten_now(ExitReason.MANUAL, "slippage attribution test")

    features = build_features(bars).frame
    for i in range(9, len(bars)):
        row = bars.iloc[i]
        next_bar = Bar(
            bars.index[i].to_pydatetime(),
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
        )
        clock.set(next_bar.timestamp)
        engine.on_bar(
            next_bar,
            features.iloc[i],
            index=i,
            frame=features,
            bars=bars,
        )
        if engine.state.trades:
            break

    trade = engine.state.trades[0]
    # The bounded entry limit has no adverse slippage; only three market-exit contract
    # sides incur 0.25 point x the $2 MNQ multiplier.
    assert trade.slippage_usd == pytest.approx(1.5)
    assert trade.gross_pnl_usd == pytest.approx(-1.5)
    assert trade.net_pnl_usd == pytest.approx(
        trade.gross_pnl_usd - trade.commission_usd
    ), "slippage is already present in fill prices and must not be subtracted twice"

    fills = JournalReader(journal.path, run_id="test-run").fills()
    assert sum(row["slippage_points"] * row["quantity"] * MNQ.multiplier for row in fills) \
        == pytest.approx(trade.slippage_usd)
    assert any(row["is_partial"] for row in fills if row["side"] == int(Side.SELL))

    stored = JournalReader(journal.path, run_id="test-run").trades()
    assert stored[0].slippage_usd == pytest.approx(trade.slippage_usd)
    assert stored[0].net_pnl_usd == pytest.approx(trade.net_pnl_usd)


# ============================================================ broker failures


def test_a_rejected_entry_leaves_the_engine_flat_and_records_why(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(
        bars, {4: (Side.BUY, 17980.0, 18100.0)},
        sim_config=SimulatedBrokerConfig(latency_ms=0, reject_probability=1.0),
    )
    run(engine, bars, clock)

    assert engine.state.position is None
    reasons = {r.reason for r in engine.rejections}
    assert RejectReason.BROKER_ERROR in reasons


def test_a_disconnect_mid_session_does_not_lose_the_position(rig):
    bars = make_bars([18000] * 30)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    features = build_features(bars).frame

    for i in range(20):
        row = bars.iloc[i]
        bar = Bar(bars.index[i].to_pydatetime(), row.open, row.high, row.low, row.close,
                  row.volume)
        clock.set(bar.timestamp)
        if i == 10:
            sim.force_disconnect("network blip")
        if i == 14:
            sim.connect()
        engine.on_bar(bar, features.iloc[i], index=i, frame=features, bars=bars)

    assert engine.state.position is not None
    assert engine.state.stop_order is not None, "protection survived the disconnect"


def test_a_disconnect_is_journalled_at_error_level(rig):
    bars = make_bars([18000] * 15)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    features = build_features(bars).frame

    for i in range(10):
        row = bars.iloc[i]
        bar = Bar(bars.index[i].to_pydatetime(), row.open, row.high, row.low, row.close,
                  row.volume)
        clock.set(bar.timestamp)
        if i == 6:
            sim.force_disconnect("socket closed")
        engine.on_bar(bar, features.iloc[i], index=i, frame=features, bars=bars)

    events = JournalReader(journal.path, run_id="test-run").events(level="ERROR")
    assert any(e["kind"] == "DISCONNECTED" for e in events)


def test_a_flatten_that_fails_records_an_error_and_counts_toward_the_kill_switch(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)
    assert engine.state.position is not None

    sim.force_disconnect("gone")
    engine.flatten_now(ExitReason.MANUAL, "operator")

    events = JournalReader(journal.path, run_id="test-run").events(level="ERROR")
    assert any(e["kind"] == "FLATTEN_FAILED" for e in events)
    assert engine.risk.kill_switch.consecutive_errors == 1


# ============================================================ duplicate protection


def test_the_engine_never_has_two_entry_orders_in_flight(rig):
    bars = make_bars([18000] * 30)
    engine, sim, journal, clock = rig(
        bars,
        {i: (Side.BUY, 17980.0, 18100.0) for i in range(4, 12)},  # fires every bar
        sim_config=SimulatedBrokerConfig(latency_ms=180_000),  # 3 minutes in flight
    )
    run(engine, bars, clock)

    orders = JournalReader(journal.path, run_id="test-run").orders(limit=200)
    entries = [o for o in orders if o["purpose"] == "ENTRY"]
    assert len(entries) == 1, f"expected one entry order in flight, got {len(entries)}"


def test_an_identical_intent_on_the_same_bar_is_refused_by_risk(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {})
    features = build_features(bars).frame
    row = bars.iloc[5]
    bar = Bar(bars.index[5].to_pydatetime(), row.open, row.high, row.low, row.close, row.volume)

    intent = OrderIntent(timestamp=bar.timestamp, instrument="MNQ", side=Side.BUY,
                         strategy="t", stop_price=17980.0, reference_price=18000.0,
                         conditions=("c",))
    assert engine.risk.evaluate_entry(intent, now=bar.timestamp).approved
    again = engine.risk.evaluate_entry(
        OrderIntent(timestamp=bar.timestamp, instrument="MNQ", side=Side.BUY, strategy="t",
                    stop_price=17980.0, reference_price=18000.0, conditions=("c",)),
        now=bar.timestamp,
    )
    assert again.rejection.reason is RejectReason.DUPLICATE_ORDER


# ============================================================ restart recovery


def test_restart_reconciles_against_the_broker_when_both_agree(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)
    assert engine.state.position is not None

    clock.set(at(11, 0))
    report = engine.reconcile(at(11, 0))
    assert report["action"] == "in_sync"


def test_restart_clears_a_local_position_the_broker_does_not_have(rig, position_factory):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {})
    run(engine, bars, clock, stop_after=3)

    engine.state.position = position_factory(quantity=2)  # stale local belief
    clock.set(at(11, 0))
    report = engine.reconcile(at(11, 0))

    assert report["action"] == "cleared_local_position"
    assert engine.state.position is None
    events = JournalReader(journal.path, run_id="test-run").events()
    assert any(e["kind"] == "RECONCILE_DIVERGENCE" for e in events)


def test_restart_before_first_bar_flattens_broker_position_without_fabricating_trade(
    rig, monkeypatch
):
    """Broker quantity authorizes a close, never a made-up strategy Position."""

    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(bars, {})
    fill_bar = seed_broker_only_long(engine, sim, clock, bars)

    def forbid_fabricated_position(*args, **kwargs):
        raise AssertionError("restart recovery must not construct a Position")

    monkeypatch.setattr("tradebot.execution.engine.Position", forbid_fabricated_position)
    report = engine.reconcile(fill_bar.timestamp)

    assert report["action"] == "recovery_flatten_submitted"
    assert engine.state.position is None
    assert engine.state.last_bar is None
    assert engine.risk.kill_switch.is_active()

    run(engine, bars, clock, start=4, stop_after=4)

    assert sim.get_positions() == []
    assert engine.state.position is None
    assert engine.state.trades == []
    assert engine.state.bot_state.value == "KILLED"
    events = JournalReader(journal.path, run_id="test-run").events()
    assert any(event["kind"] == "RECOVERY_FLAT_CONFIRMED" for event in events)


def test_recovery_confirmation_account_mismatch_retains_local_state_and_protection(
    rig, monkeypatch
):
    bars = make_bars([18_000.0] * 12)
    engine, sim, journal, clock = rig(
        bars,
        {4: (Side.BUY, 17_980.0, 18_100.0)},
    )
    run(engine, bars, clock, stop_after=7)
    position = engine.state.position
    stop = engine.state.stop_order
    target = engine.state.target_order
    assert position is not None and stop is not None and target is not None

    mismatch = Rejection(
        timestamp=clock.now(),
        reason=RejectReason.LIVE_TRADING_DISABLED,
        detail="authoritative broker account identity changed",
        stage="BROKER_SNAPSHOT",
    )
    monkeypatch.setattr(
        engine.broker,
        "reconcile_risk_state",
        lambda *, now=None: mismatch,
    )
    # Reproduce the old unsafe ordering: a raw position read appears flat even though the
    # full guarded account/route reconciliation refuses the snapshot.
    monkeypatch.setattr(engine.broker, "get_positions", lambda: [])

    assert not engine._confirm_recovery_flat(clock.now())

    assert engine.state.position is position
    assert engine.state.stop_order is stop
    assert engine.state.target_order is target
    events = JournalReader(journal.path, run_id="test-run").events()
    assert not any(event["kind"] == "RECOVERY_FLAT_CONFIRMED" for event in events)


def test_reconcile_reports_failure_when_the_broker_is_unreachable(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {})
    sim.force_disconnect()
    report = engine.reconcile(at(11, 0))
    assert report["action"] == "failed"


# ============================================================ the journal


def test_every_decision_lands_a_row(rig):
    closes = [18000] * 6 + [18010, 18030, 18060, 18060]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    assert journal.count("signals") == 1
    assert journal.count("orders") >= 3     # entry + stop + target
    assert journal.count("order_events") >= 4
    assert journal.count("fills") >= 2      # entry + exit
    assert journal.count("trades") == 1
    assert journal.count("equity") == len(bars)
    assert journal.count("events") >= 2


def test_rejected_signals_are_journalled_with_their_reason(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {})
    # A signal outside the strategy's own hours, so the base class refuses it.
    features = build_features(bars).frame
    engine.strategy.script = {3: (Side.BUY, 17980.0, 18100.0)}
    engine.strategy.spec = engine.strategy.spec.__class__(
        **{**engine.strategy.spec.to_dict(),
           "trading_hours": TradingHours(datetime(2000, 1, 1, 14, 0).time(),
                                         datetime(2000, 1, 1, 15, 0).time(),
                                         datetime(2000, 1, 1, 15, 58).time()),
           "features": engine.strategy.spec.features}
    )
    run(engine, bars, clock)

    counts = JournalReader(journal.path, run_id="test-run").rejection_counts()
    assert counts, "a refusal must be recorded, not silently dropped"
    assert any(c["reason"] == RejectReason.OUTSIDE_TRADING_HOURS.value for c in counts)


def test_the_journal_reader_reconstructs_trades(rig):
    closes = [18000] * 6 + [18010, 18030, 18060, 18060]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    trades = JournalReader(journal.path, run_id="test-run").trades()
    assert len(trades) == 1
    assert trades[0].exit_reason is ExitReason.TAKE_PROFIT
    assert trades[0].entry_conditions == ("scripted",)
    assert trades[0].net_pnl_usd == pytest.approx(engine.state.trades[0].net_pnl_usd)


def test_the_journal_summary_matches_the_engine(rig):
    closes = [18000] * 6 + [18010, 18030, 18060, 18060]
    bars = make_bars(closes)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18050.0)})
    run(engine, bars, clock)

    summary = JournalReader(journal.path, run_id="test-run").summary()
    assert summary["trades"] == 1
    assert summary["net_pnl"] == pytest.approx(engine.state.trades[0].net_pnl_usd)


def test_an_order_history_is_recoverable(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)

    reader = JournalReader(journal.path, run_id="test-run")
    entry = next(o for o in reader.orders() if o["purpose"] == "ENTRY")
    history = reader.order_history(entry["order_id"])
    statuses = [h["status"] for h in history]
    assert statuses[0] == OrderStatus.PENDING.value
    assert OrderStatus.FILLED.value in statuses


def test_the_journal_reader_refuses_to_invent_an_empty_database(tmp_path):
    with pytest.raises(Exception):
        JournalReader(tmp_path / "does_not_exist.sqlite3")


def test_the_engine_snapshot_describes_the_whole_system(rig):
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)

    snap = engine.snapshot()
    assert snap["paper"] is True
    assert snap["strategy"] == "scripted"
    assert snap["instrument"] == "MNQ"
    assert snap["position"]["side"] == "BUY"
    assert "daily_loss_limit" in snap["risk"]


def test_a_fill_is_applied_exactly_once(rig):
    """Regression: `on_bar` both returns and queues its events.

    Draining in two places silently doubled every position while each component still
    looked correct on its own — the broker reported 3 contracts and the engine believed 6.
    """
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)

    position = engine.state.position
    broker_qty = sum(p.quantity for p in sim.get_positions())
    assert position.signed_quantity == broker_qty
    assert journal.count("fills") == 1


def test_failed_flatten_keeps_the_existing_stop_working(rig):
    """A broker outage must leave the open position protected, not naked."""
    bars = make_bars([18000] * 20)
    engine, sim, journal, clock = rig(bars, {4: (Side.BUY, 17980.0, 18100.0)})
    run(engine, bars, clock, stop_after=8)
    stop = engine.state.stop_order
    assert stop is not None

    sim.force_disconnect("gone mid-flatten")
    engine.flatten_now(ExitReason.MANUAL, "operator")  # must not raise

    events = JournalReader(journal.path, run_id="test-run").events()
    kinds = {e["kind"] for e in events}
    assert "FLATTEN_FAILED" in kinds
    assert engine.state.position is not None
    assert engine.state.stop_order == stop

    sim.connect()
    broker_stop = next(
        order for order in sim.get_orders() if order.order_id == stop.order_id
    )
    assert broker_stop.status.is_working
