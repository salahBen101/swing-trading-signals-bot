"""End-to-end invariants for restart-safe personal-risk bookkeeping."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tradebot.config import (
    DailyRisk,
    KillSwitchConfig,
    PerTradeRisk,
    RiskConfig,
    SessionConfig,
)
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.models import OrderIntent, Position, Trade
from tradebot.core.types import ExitReason, RejectReason, Side
from tradebot.instruments.registry import get_instrument
from tradebot.risk.broker_state import (
    AuthoritativeBrokerSnapshot,
    BrokerRiskAccount,
    BrokerRiskPosition,
)
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import Approval, RiskEngine
from tradebot.risk.state_store import FileRiskStateStore, RiskStateBinding, StoredRiskState


MNQ = get_instrument("MNQ")
NOW = datetime(2026, 8, 24, 11, 0, tzinfo=MARKET_TZ)
ACCOUNT_ID = "SIM-DURABLE-50K"
BROKER_NAME = "simulated"
BROKER_ROUTE = "simulated-local-paper"
CONTEXT_ID = "tradeify-growth-50k:stage2:paper"


def intent(*, now: datetime = NOW, stop: float = 17_990.0) -> OrderIntent:
    return OrderIntent(
        timestamp=now,
        instrument="MNQ",
        side=Side.BUY,
        strategy="durable-test",
        stop_price=stop,
        reference_price=18_000.0,
        conditions=("deterministic",),
    )


def snapshot(
    now: datetime,
    *,
    equity: float = 50_000.0,
    quantity: int = 0,
) -> AuthoritativeBrokerSnapshot:
    positions = (
        (BrokerRiskPosition("MNQ", quantity, 18_000.0),) if quantity else ()
    )
    return AuthoritativeBrokerSnapshot(
        read_started_at=now,
        captured_at=now,
        broker_name=BROKER_NAME,
        broker_is_paper=True,
        execution_route=BROKER_ROUTE,
        account=BrokerRiskAccount(
            account_id=ACCOUNT_ID,
            equity=equity,
            cash=equity,
            realized_pnl=equity - 50_000.0,
            unrealized_pnl=0.0,
            currency="USD",
            is_paper=True,
        ),
        positions=positions,
        orders=(),
    )


def make_engine(
    tmp_path,
    store,
    *,
    bootstrap: bool,
    now: datetime = NOW,
    max_daily_loss: float = 200.0,
    max_trades: int = 1,
    context_id: str = CONTEXT_ID,
    max_risk: float = 200.0,
    starting_equity: float | None = None,
    enforce_drawdown_floor: bool = True,
    kill_switch: KillSwitch | None = None,
) -> RiskEngine:
    config = RiskConfig(
        starting_equity_usd=50_000.0,
        per_trade=PerTradeRisk(
            risk_pct_of_equity=0.4,
            max_risk_per_trade_usd=max_risk,
            max_contracts=3,
            min_contracts=1,
        ),
        daily=DailyRisk(
            max_daily_loss_usd=max_daily_loss,
            max_daily_loss_r=4.0,
            max_trades_per_day=max_trades,
        ),
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "durable.kill")),
    )
    clock = SimulatedClock(now)
    return RiskEngine(
        config,
        MNQ,
        SessionConfig(),
        clock=clock,
        kill_switch=kill_switch or KillSwitch(tmp_path / "durable.kill", clock=clock),
        starting_equity=starting_equity,
        enforce_drawdown_floor=enforce_drawdown_floor,
        expected_broker_account_id=ACCOUNT_ID,
        expected_broker_name=BROKER_NAME,
        expected_broker_is_paper=True,
        expected_broker_route=BROKER_ROUTE,
        risk_state_store=store,
        bootstrap_risk_state_store=bootstrap,
        risk_state_context_id=context_id,
    )


def reconcile_flat(
    engine: RiskEngine,
    now: datetime = NOW,
    *,
    equity: float = 50_000.0,
) -> None:
    assert engine.reconcile_broker_snapshot(snapshot(now, equity=equity), now=now) is None
    assert not engine.risk_state_recovery_required


def approve_verified(
    engine: RiskEngine,
    requested: OrderIntent,
    *,
    now: datetime = NOW,
    equity: float = 50_000.0,
) -> Approval:
    decision = engine.evaluate_entry(requested, now=now)
    assert decision.approved, decision.rejection
    approval = decision.approval
    assert engine.verify(
        approval.order,
        approval.token,
        now=now,
        broker_snapshot=snapshot(now, equity=equity),
    ) is None
    return approval


def close_trade(
    engine: RiskEngine,
    approval: Approval,
    *,
    now: datetime,
    pnl: float,
    quantity: int | None = None,
    trade_id: str = "trade-1",
) -> Trade:
    quantity = approval.contracts if quantity is None else quantity
    assert engine.on_entry_fill_observed(approval.order.order_id, now=now)
    position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=quantity,
        entry_price=18_000.0,
        entry_time=now,
        strategy="durable-test",
        initial_stop=approval.token.stop_price,
        stop_price=approval.token.stop_price,
        position_id=f"position-{trade_id}",
    )
    engine.on_position_opened(position)
    trade = Trade(
        trade_id=trade_id,
        instrument="MNQ",
        strategy="durable-test",
        side=Side.BUY,
        quantity=quantity,
        entry_time=now,
        entry_price=18_000.0,
        exit_time=now + timedelta(minutes=1),
        exit_price=18_000.0 + pnl / (quantity * MNQ.multiplier),
        exit_reason=ExitReason.STOP_LOSS if pnl <= 0 else ExitReason.TAKE_PROFIT,
        gross_pnl_usd=pnl,
        commission_usd=0.0,
        net_pnl_usd=pnl,
        r_multiple=pnl / approval.risk_usd,
        bars_held=1,
        initial_stop=approval.token.stop_price,
        position_id=position.position_id,
    )
    engine.on_trade_closed(trade)
    return trade


def test_startup_stays_locked_until_a_fresh_flat_snapshot(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)

    blocked = engine.evaluate_entry(intent())
    assert blocked.rejection.reason is RejectReason.BROKER_ERROR
    assert "reconciliation" in blocked.rejection.detail

    reconcile_flat(engine)
    approval = engine.evaluate_entry(intent()).approval
    assert approval is not None
    persisted = store.load()
    assert persisted.state.active_entry_order_id is None
    assert engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=snapshot(NOW),
    ) is None
    persisted = store.load()
    assert persisted.state.active_entry_order_id == approval.order.order_id
    assert persisted.binding == RiskStateBinding(
        instrument="MNQ",
        risk_policy_sha256=persisted.binding.risk_policy_sha256,
        broker_account_id=ACCOUNT_ID,
        broker_name=BROKER_NAME,
        broker_is_paper=True,
        broker_execution_route=BROKER_ROUTE,
        deployment_context_id=CONTEXT_ID,
    )


def test_first_fill_consumes_the_daily_quota_exactly_once_and_survives_restart(
    tmp_path,
) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())

    fill_time = NOW + timedelta(minutes=1)
    assert engine.on_entry_fill_observed(approval.order.order_id, now=fill_time)
    first_revision = store.load().state.revision
    assert engine.on_entry_fill_observed(approval.order.order_id, now=fill_time)
    assert engine.state.trades_today == 1
    assert store.load().state.revision == first_revision

    close_trade(
        engine,
        approval,
        now=fill_time,
        pnl=50.0,
        trade_id="winner",
    )
    restarted = make_engine(
        tmp_path,
        FileRiskStateStore(tmp_path / "risk.json"),
        bootstrap=False,
        now=NOW + timedelta(minutes=5),
    )
    reconcile_flat(restarted, NOW + timedelta(minutes=5), equity=50_050.0)

    blocked = restarted.evaluate_entry(
        intent(now=NOW + timedelta(minutes=5), stop=17_989.0),
        now=NOW + timedelta(minutes=5),
    )
    assert blocked.rejection.reason is RejectReason.MAX_TRADES_PER_DAY


def test_partial_fill_close_persists_actual_size_without_reopening_quota(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    assert approval.contracts == 3

    close_trade(
        engine,
        approval,
        now=NOW + timedelta(minutes=1),
        pnl=-20.0,
        quantity=1,
        trade_id="partial-loss",
    )

    persisted = store.load().state
    assert persisted.trades_today == 1
    assert persisted.last_trade_contracts == 1
    assert persisted.last_trade_quantity == 1
    assert persisted.last_trade_approved_risk_usd == pytest.approx(28.24)
    assert engine.risk_state_store_error is None


def test_realized_stop_gap_beyond_reserved_envelope_latches_the_account(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    assert approval.risk_usd == pytest.approx(84.72)

    close_trade(
        engine,
        approval,
        now=NOW + timedelta(minutes=1),
        pnl=-100.0,
        trade_id="gap-beyond-reserve",
    )

    persisted = store.load().state
    assert engine.kill_switch.is_active()
    assert persisted.halted
    assert persisted.halt_reason == "planned_risk_envelope_breach"
    assert engine.evaluate_entry(
        intent(now=NOW + timedelta(minutes=3)),
        now=NOW + timedelta(minutes=3),
    ).rejection.reason is RejectReason.KILL_SWITCH_ACTIVE


def test_daily_loss_halt_and_trade_application_are_idempotent_across_restart(
    tmp_path,
) -> None:
    path = tmp_path / "risk.json"
    store = FileRiskStateStore(path)
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent(stop=17_950.0))
    trade = close_trade(
        engine,
        approval,
        now=NOW + timedelta(minutes=1),
        pnl=-200.0,
        trade_id="daily-cap-loss",
    )
    persisted = store.load().state
    assert persisted.halted
    assert persisted.daily_realized_pnl == -200.0

    restarted = make_engine(
        tmp_path,
        FileRiskStateStore(path),
        bootstrap=False,
        now=NOW + timedelta(minutes=5),
    )
    reconcile_flat(restarted, NOW + timedelta(minutes=5), equity=49_800.0)
    restarted.on_trade_closed(trade)
    assert restarted.state.daily_realized_pnl == -200.0
    assert FileRiskStateStore(path).load().state.daily_realized_pnl == -200.0
    assert restarted.evaluate_entry(
        intent(now=NOW + timedelta(minutes=6), stop=17_980.0),
        now=NOW + timedelta(minutes=6),
    ).rejection.reason is RejectReason.KILL_SWITCH_ACTIVE


def test_deleted_state_cannot_be_rebootstrapped_as_a_fresh_account(tmp_path) -> None:
    path = tmp_path / "risk.json"
    store = FileRiskStateStore(path)
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    close_trade(
        engine,
        approval,
        now=NOW + timedelta(minutes=1),
        pnl=25.0,
        trade_id="history-that-must-not-reset",
    )
    assert store.load().state.trades_today == 1
    assert store.initialization_marker_path.exists()

    path.unlink()
    restarted = make_engine(
        tmp_path,
        FileRiskStateStore(path),
        bootstrap=True,
        now=NOW + timedelta(minutes=5),
    )

    assert restarted.risk_state_store_error is not None
    assert "initialized before" in restarted.risk_state_store_error
    assert restarted.evaluate_entry(
        intent(now=NOW + timedelta(minutes=5), stop=17_989.0),
        now=NOW + timedelta(minutes=5),
    ).rejection.reason is RejectReason.BROKER_ERROR


@pytest.mark.parametrize("failure", ["missing", "corrupt", "wrong-context"])
def test_unrecoverable_state_blocks_entries_but_not_a_proven_flatten(
    tmp_path, failure
) -> None:
    path = tmp_path / "risk.json"
    store = FileRiskStateStore(path)
    if failure == "corrupt":
        path.write_text("not-json", encoding="utf-8")
    elif failure == "wrong-context":
        seeded = make_engine(tmp_path, store, bootstrap=True, context_id="other-context")
        assert seeded.risk_state_store_error is None

    engine = make_engine(tmp_path, FileRiskStateStore(path), bootstrap=False)
    assert engine.risk_state_store_error is not None
    assert engine.evaluate_entry(intent()).rejection.reason is RejectReason.BROKER_ERROR

    position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        entry_price=18_000.0,
        entry_time=NOW,
        strategy="durable-test",
        initial_stop=17_990.0,
        stop_price=17_990.0,
    )
    exit_approval = engine.evaluate_exit(position, reason="FLATTEN", now=NOW).approval
    nonflat = snapshot(NOW, quantity=1)
    assert engine.verify(
        exit_approval.order,
        exit_approval.token,
        now=NOW,
        broker_snapshot=nonflat,
    ) is None
    assert engine.kill_switch.is_active()
    forced = engine.forced_exit_reason(position, 18_000.0, NOW)
    assert forced is not None
    assert forced[0] is RejectReason.KILL_SWITCH_ACTIVE


def test_post_loss_size_and_risk_may_not_increase_after_session_rollover(tmp_path) -> None:
    path = tmp_path / "risk.json"
    engine = make_engine(tmp_path, FileRiskStateStore(path), bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent(stop=17_940.0))
    assert approval.contracts == 1
    assert approval.risk_usd == pytest.approx(128.24)
    close_trade(
        engine,
        approval,
        now=NOW + timedelta(minutes=1),
        pnl=-120.0,
        trade_id="size-anchor-loss",
    )

    next_day = NOW + timedelta(days=1)
    restarted = make_engine(
        tmp_path,
        FileRiskStateStore(path),
        bootstrap=False,
        now=next_day,
    )
    reconcile_flat(restarted, next_day, equity=49_880.0)

    bigger_size = restarted.evaluate_entry(intent(now=next_day, stop=17_990.0), now=next_day)
    assert bigger_size.rejection.reason is RejectReason.MAX_RISK_PER_TRADE
    assert "post-loss" in bigger_size.rejection.detail

    bigger_risk = restarted.evaluate_entry(intent(now=next_day, stop=17_930.0), now=next_day)
    assert bigger_risk.rejection.reason is RejectReason.MAX_RISK_PER_TRADE

    equal = restarted.evaluate_entry(intent(now=next_day, stop=17_940.0), now=next_day)
    assert equal.approved
    assert equal.approval.contracts == 1
    assert equal.approval.risk_usd == pytest.approx(128.24)


class ArmableFailingStore:
    def __init__(self, path) -> None:
        self.delegate = FileRiskStateStore(path)
        self.fail = False

    def load(self, *, expected_binding=None) -> StoredRiskState | None:
        return self.delegate.load(expected_binding=expected_binding)

    def initialize(self, stored_state: StoredRiskState) -> None:
        if self.fail:
            raise OSError("injected durable initialization failure")
        self.delegate.initialize(stored_state)

    def save(self, stored_state: StoredRiskState) -> None:
        if self.fail:
            raise OSError("injected durable write failure")
        self.delegate.save(stored_state)


class ExplodingKillSwitch(KillSwitch):
    def trip(self, reason: str, *, by: str = "system"):
        raise OSError("injected kill-switch write failure")


def test_failed_session_rollover_does_not_publish_reset_counters(tmp_path) -> None:
    store = ArmableFailingStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    engine.state.trades_today = 1
    engine.state.daily_realized_pnl = -75.0
    original_date = engine.state.session_date

    store.fail = True
    assert not engine.roll_session(
        NOW + timedelta(days=1),
        broker_flat_confirmed=True,
    )

    assert engine.state.session_date == original_date
    assert engine.state.trades_today == 1
    assert engine.state.daily_realized_pnl == -75.0
    assert engine.risk_state_store_error is not None


def test_durable_equity_uses_absolute_broker_snapshots_not_trade_deltas(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    entry_time = NOW + timedelta(minutes=1)
    assert engine.on_entry_fill_observed(approval.order.order_id, now=entry_time)
    position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=approval.contracts,
        entry_price=18_000.0,
        entry_time=entry_time,
        strategy="durable-test",
        initial_stop=approval.token.stop_price,
        stop_price=approval.token.stop_price,
        position_id="position-authoritative-equity",
    )
    engine.on_position_opened(position)
    marked_equity = 50_075.0
    assert engine.reconcile_broker_snapshot(
        snapshot(entry_time, equity=marked_equity, quantity=approval.contracts),
        now=entry_time,
    ) is None
    trade = Trade(
        trade_id="authoritative-equity-close",
        instrument="MNQ",
        strategy="durable-test",
        side=Side.BUY,
        quantity=approval.contracts,
        entry_time=entry_time,
        entry_price=18_000.0,
        exit_time=entry_time + timedelta(minutes=1),
        exit_price=18_050.0,
        exit_reason=ExitReason.TAKE_PROFIT,
        gross_pnl_usd=100.0,
        commission_usd=0.0,
        net_pnl_usd=100.0,
        r_multiple=100.0 / approval.risk_usd,
        bars_held=1,
        initial_stop=approval.token.stop_price,
        position_id=position.position_id,
    )

    engine.on_trade_closed(trade)
    assert engine.state.equity == marked_equity

    settled_equity = 50_098.0
    assert engine.reconcile_broker_snapshot(
        snapshot(trade.exit_time, equity=settled_equity),
        now=trade.exit_time,
    ) is None
    assert engine.state.equity == settled_equity
    assert engine.state.peak_equity == settled_equity


def test_fill_persistence_failure_trips_kill_switch_and_keeps_exits_available(
    tmp_path,
) -> None:
    store = ArmableFailingStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())

    store.fail = True
    assert not engine.on_entry_fill_observed(
        approval.order.order_id,
        now=NOW + timedelta(minutes=1),
    )
    assert engine.kill_switch.is_active()
    assert engine.risk_state_store_error is not None
    assert engine.evaluate_entry(
        intent(now=NOW + timedelta(minutes=2), stop=17_989.0),
        now=NOW + timedelta(minutes=2),
    ).rejection.reason is RejectReason.KILL_SWITCH_ACTIVE

    position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=approval.contracts,
        entry_price=18_000.0,
        entry_time=NOW + timedelta(minutes=1),
        strategy="durable-test",
        initial_stop=17_990.0,
        stop_price=17_990.0,
    )
    exit_approval = engine.evaluate_exit(
        position,
        reason="FLATTEN",
        now=NOW + timedelta(minutes=2),
    ).approval
    assert engine.verify(
        exit_approval.order,
        exit_approval.token,
        now=NOW + timedelta(minutes=2),
        broker_snapshot=snapshot(
            NOW + timedelta(minutes=2),
            quantity=approval.contracts,
        ),
    ) is None


def test_state_and_kill_file_failures_cannot_block_a_proven_exit(tmp_path) -> None:
    store = ArmableFailingStore(tmp_path / "risk.json")
    clock = SimulatedClock(NOW)
    engine = make_engine(
        tmp_path,
        store,
        bootstrap=True,
        kill_switch=ExplodingKillSwitch(tmp_path / "exploding.kill", clock=clock),
    )
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())

    store.fail = True
    assert not engine.on_entry_fill_observed(
        approval.order.order_id,
        now=NOW + timedelta(minutes=1),
    )
    assert engine.risk_state_store_error is not None

    position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=approval.contracts,
        entry_price=18_000.0,
        entry_time=NOW + timedelta(minutes=1),
        strategy="durable-test",
        initial_stop=17_990.0,
        stop_price=17_990.0,
    )
    exit_approval = engine.evaluate_exit(
        position,
        reason="FLATTEN",
        now=NOW + timedelta(minutes=2),
    ).approval
    assert exit_approval is not None
    assert engine.verify(
        exit_approval.order,
        exit_approval.token,
        now=NOW + timedelta(minutes=2),
        broker_snapshot=snapshot(
            NOW + timedelta(minutes=2),
            quantity=approval.contracts,
        ),
    ) is None


def test_restart_without_local_position_can_authorize_exact_snapshot_flatten(
    tmp_path,
) -> None:
    # No state file and no local Position object: this models a crash after the venue
    # accepted/fill an entry but before strategy execution state was reconstructed.
    engine = make_engine(
        tmp_path,
        FileRiskStateStore(tmp_path / "missing-risk.json"),
        bootstrap=False,
    )
    assert engine.risk_state_store_error is not None
    exposed = snapshot(NOW, quantity=3)

    decision = engine.evaluate_snapshot_flatten(exposed, now=NOW)

    assert decision.approved
    assert decision.approval.order.purpose.value == "FLATTEN"
    assert decision.approval.order.side is Side.SELL
    assert decision.approval.order.quantity == 3
    assert engine.verify(
        decision.approval.order,
        decision.approval.token,
        now=NOW,
        broker_snapshot=exposed,
    ) is None


def test_fill_outside_signed_entry_limit_latches_kill_and_blocks_entries(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    assert approval.order.limit_price == 18_001.0
    fill_time = NOW + timedelta(minutes=1)
    assert engine.on_entry_fill_observed(approval.order.order_id, now=fill_time)
    impossible_fill = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        entry_price=18_001.25,
        entry_time=fill_time,
        strategy="durable-test",
        initial_stop=approval.token.stop_price,
        stop_price=approval.token.stop_price,
        position_id="position-outside-limit",
    )

    engine.on_position_opened(impossible_fill)

    assert engine.risk_state_store_error is not None
    assert "all-in" in engine.risk_state_store_error
    assert engine.kill_switch.is_active()
    blocked = engine.evaluate_entry(
        intent(now=fill_time + timedelta(minutes=1), stop=17_989.0),
        now=fill_time + timedelta(minutes=1),
    )
    assert blocked.rejection.reason is RejectReason.KILL_SWITCH_ACTIVE


def test_later_partial_fill_revalidates_the_cumulative_all_in_envelope(tmp_path) -> None:
    store = FileRiskStateStore(tmp_path / "risk.json")
    engine = make_engine(tmp_path, store, bootstrap=True)
    reconcile_flat(engine)
    approval = approve_verified(engine, intent())
    fill_time = NOW + timedelta(minutes=1)
    assert engine.on_entry_fill_observed(approval.order.order_id, now=fill_time)

    first_partial = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=1,
        entry_price=18_000.25,
        entry_time=fill_time,
        strategy="durable-test",
        initial_stop=approval.token.stop_price,
        stop_price=approval.token.stop_price,
        position_id="cumulative-partial-position",
    )
    engine.on_position_opened(first_partial)
    assert engine.risk_state_store_error is None

    cumulative_position = Position(
        instrument="MNQ",
        side=Side.BUY,
        quantity=2,
        entry_price=18_001.25,
        entry_time=fill_time + timedelta(minutes=1),
        strategy="durable-test",
        initial_stop=approval.token.stop_price,
        stop_price=approval.token.stop_price,
        position_id=first_partial.position_id,
    )
    engine.on_position_opened(cumulative_position)

    assert engine.risk_state_store_error is not None
    assert "all-in" in engine.risk_state_store_error
    assert engine.kill_switch.is_active()


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ({"starting_equity": 51_000.0}, {"starting_equity": 50_000.0}),
        ({"enforce_drawdown_floor": True}, {"enforce_drawdown_floor": False}),
    ],
)
def test_effective_constructor_policy_is_bound_to_durable_state(
    tmp_path,
    original,
    changed,
) -> None:
    path = tmp_path / "risk.json"
    make_engine(
        tmp_path,
        FileRiskStateStore(path),
        bootstrap=True,
        **original,
    )

    reopened = make_engine(
        tmp_path,
        FileRiskStateStore(path),
        bootstrap=False,
        **changed,
    )

    assert reopened.risk_state_store_error is not None
    assert "different risk policy" in reopened.risk_state_store_error
