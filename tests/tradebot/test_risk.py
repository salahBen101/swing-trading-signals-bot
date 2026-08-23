"""M5: the risk engine.

Every hard limit gets driven to its boundary and one step past it. These are the tests that
matter most in this repository: a bug here costs money, and unlike a strategy bug it does
so silently, because the output still looks like trading.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tradebot.config import (
    DailyRisk,
    DrawdownRisk,
    KillSwitchConfig,
    PerTradeRisk,
    RiskConfig,
    SessionConfig,
    StreakRisk,
)
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.models import Order, OrderIntent, Trade
from tradebot.core.types import (
    ExitReason,
    OrderPurpose,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)
from tradebot.instruments.registry import get_instrument
from tradebot.risk.broker_state import (
    AuthoritativeBrokerSnapshot,
    BrokerRiskAccount,
    BrokerRiskPosition,
)
from tradebot.risk.killswitch import KillSwitch
from tradebot.risk.limits import RiskEngine
from tradebot.risk.session import SessionGuard
from tradebot.risk.sizing import VolatilityTargetSizer
from tradebot.risk.tokens import AuthorizationKind, TokenError, TokenMint

MNQ = get_instrument("MNQ")


def at(hour=11, minute=0, day=1) -> datetime:
    return datetime(2024, 4, day, hour, minute, tzinfo=MARKET_TZ)


def broker_snapshot(
    *,
    ts=None,
    equity=50_000.0,
    account_id="TEST-50K",
    broker_name="test-broker",
    is_paper=True,
    positions=(),
    orders=(),
):
    ts = ts or at()
    return AuthoritativeBrokerSnapshot(
        read_started_at=ts,
        captured_at=ts,
        broker_name=broker_name,
        broker_is_paper=is_paper,
        account=BrokerRiskAccount(
            account_id=account_id,
            equity=equity,
            cash=equity,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            currency="USD",
            is_paper=is_paper,
        ),
        positions=positions,
        orders=orders,
    )


def risk_config(**over) -> RiskConfig:
    base = dict(
        starting_equity_usd=50_000.0,
        per_trade=PerTradeRisk(risk_pct_of_equity=0.5, max_risk_per_trade_usd=250.0,
                               max_contracts=3, min_contracts=1),
        daily=DailyRisk(max_daily_loss_usd=1000.0, max_daily_loss_r=4.0, max_trades_per_day=6),
        streaks=StreakRisk(max_consecutive_losses=3, cooldown_minutes=20),
        drawdown=DrawdownRisk(trailing_drawdown_pct=5.0, size_reduction_cushion_pct=40.0),
        token_ttl_seconds=60.0,
    )
    base.update(over)
    return RiskConfig(**base)


@pytest.fixture
def engine(tmp_path):
    clock = SimulatedClock(at())
    cfg = risk_config(kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "KILL.flag")))
    return RiskEngine(cfg, MNQ, SessionConfig(), clock=clock,
                      kill_switch=KillSwitch(tmp_path / "KILL.flag", clock=clock))


def intent(*, side=Side.BUY, entry=18000.0, stop=17990.0, ts=None, strategy="t") -> OrderIntent:
    return OrderIntent(
        timestamp=ts or at(), instrument="MNQ", side=side, strategy=strategy,
        stop_price=stop, reference_price=entry, conditions=("c",),
    )


def losing_trade(pnl=-100.0, r=-1.0, ts=None) -> Trade:
    ts = ts or at()
    return Trade(
        trade_id="t", instrument="MNQ", strategy="t", side=Side.BUY, quantity=1,
        entry_time=ts, entry_price=18000.0, exit_time=ts, exit_price=17990.0,
        exit_reason=ExitReason.STOP_LOSS, gross_pnl_usd=pnl, commission_usd=0.0,
        net_pnl_usd=pnl, r_multiple=r, bars_held=5, initial_stop=17990.0,
    )


# ============================================================ position sizing


def test_size_is_the_risk_budget_divided_by_the_dollar_stop_distance():
    sizer = VolatilityTargetSizer(
        PerTradeRisk(risk_pct_of_equity=0.5, max_risk_per_trade_usd=250.0,
                     max_contracts=10, min_contracts=1),
        MNQ,
    )
    # Budget = min(50000 x 0.5%, 250) = $250. Risk per contract = 10 pts x $2 = $20,
    # so the budget buys 12 -- capped here at max_contracts = 10.
    result = sizer.size(equity=50_000.0, stop_distance_points=10.0)
    assert result.contracts == 10
    assert result.risk_per_contract_usd == 20.0
    assert result.budget_usd == 250.0


def test_size_scales_inversely_with_the_stop_distance():
    sizer = VolatilityTargetSizer(
        PerTradeRisk(risk_pct_of_equity=1.0, max_risk_per_trade_usd=1000.0,
                     max_contracts=100, min_contracts=1),
        MNQ,
    )
    tight = sizer.size(equity=50_000.0, stop_distance_points=10.0)   # $20/contract
    wide = sizer.size(equity=50_000.0, stop_distance_points=40.0)    # $80/contract
    assert tight.contracts == 25 and wide.contracts == 6
    assert tight.risk_usd <= 500.0 and wide.risk_usd <= 500.0


def test_the_hard_contract_cap_beats_the_budget():
    sizer = VolatilityTargetSizer(
        PerTradeRisk(risk_pct_of_equity=50.0, max_risk_per_trade_usd=100_000.0,
                     max_contracts=3, min_contracts=1),
        MNQ,
    )
    assert sizer.size(equity=1_000_000.0, stop_distance_points=1.0).contracts == 3


def test_a_trade_is_skipped_rather_than_silently_exceeding_the_risk_cap():
    sizer = VolatilityTargetSizer(
        PerTradeRisk(risk_pct_of_equity=0.01, max_risk_per_trade_usd=5.0,
                     max_contracts=3, min_contracts=1),
        MNQ,
    )
    # $5 budget cannot buy one contract at 200 pts x $2 = $400 of risk.
    result = sizer.size(equity=50_000.0, stop_distance_points=200.0)
    assert result.contracts == 0
    assert result.reason is RejectReason.SIZE_BELOW_MINIMUM
    assert "Skipping rather than" in result.detail


def test_a_zero_stop_distance_is_refused():
    sizer = VolatilityTargetSizer(PerTradeRisk(), MNQ)
    result = sizer.size(equity=50_000.0, stop_distance_points=0.0)
    assert result.contracts == 0 and result.reason is RejectReason.INVALID_ORDER


def test_size_throttles_as_the_drawdown_cushion_is_consumed():
    sizer = VolatilityTargetSizer(
        PerTradeRisk(risk_pct_of_equity=1.0, max_risk_per_trade_usd=1000.0,
                     max_contracts=100, min_contracts=1),
        MNQ,
    )
    full = sizer.size(equity=50_000.0, stop_distance_points=10.0, cushion_fraction=1.0)
    half = sizer.size(equity=50_000.0, stop_distance_points=10.0, cushion_fraction=0.2,
                      cushion_threshold=0.4)
    assert half.throttle == pytest.approx(0.5)
    assert half.contracts == full.contracts // 2


# ============================================================ entry limits


def test_a_clean_entry_is_approved_and_sized_by_the_engine_not_the_strategy(engine):
    decision = engine.evaluate_entry(intent())
    assert decision.approved
    approval = decision.approval
    # Budget min(50000 x 0.5%, 250) = 250; risk per contract 10 pts x $2 = $20 -> 12,
    # capped at max_contracts = 3.
    assert approval.contracts == 3
    assert approval.risk_usd == 60.0
    assert approval.order.quantity == 3
    assert approval.order.purpose == "ENTRY"
    assert approval.token is not None


def test_only_one_position_at_a_time(engine, position_factory):
    decision = engine.evaluate_entry(intent(), position=position_factory())
    assert not decision.approved
    assert decision.rejection.reason is RejectReason.POSITION_ALREADY_OPEN


def test_tracked_position_blocks_entry_even_when_caller_omits_position(engine):
    engine.state.open_position_id = "pos-already-open"
    decision = engine.evaluate_entry(intent(), position=None)
    assert not decision.approved
    assert decision.rejection.reason is RejectReason.POSITION_ALREADY_OPEN


def test_the_daily_trade_cap_blocks_the_next_entry(engine):
    engine.roll_session(at())
    for _ in range(6):
        engine.state.trades_today += 1
    decision = engine.evaluate_entry(intent())
    assert decision.rejection.reason is RejectReason.MAX_TRADES_PER_DAY


def test_the_daily_trade_cap_allows_exactly_the_configured_number(engine):
    engine.roll_session(at())
    engine.state.trades_today = 5
    assert engine.evaluate_entry(intent()).approved


def test_the_daily_dollar_loss_cap_halts_the_session(engine):
    engine.roll_session(at())
    engine.state.daily_realized_pnl = -999.99
    assert engine.evaluate_entry(intent()).approved, "just inside the cap must still trade"

    engine.state.daily_realized_pnl = -1000.0
    decision = engine.evaluate_entry(intent(stop=17989.0))
    assert decision.rejection.reason is RejectReason.MAX_DAILY_LOSS
    assert engine.state.halted


def test_the_daily_r_cap_halts_the_session(engine):
    engine.roll_session(at())
    engine.state.daily_r = -3.99
    assert engine.evaluate_entry(intent()).approved

    engine.state.daily_r = -4.0
    decision = engine.evaluate_entry(intent(stop=17989.0))
    assert decision.rejection.reason is RejectReason.MAX_DAILY_LOSS_R


def test_consecutive_losses_start_a_cooldown_that_blocks_entries(engine):
    engine.roll_session(at())
    for i in range(3):
        engine.on_trade_closed(losing_trade(pnl=-50.0, r=-0.5, ts=at(11, i)))

    assert engine.state.cooldown_until == at(11, 2) + timedelta(minutes=20)
    decision = engine.evaluate_entry(intent(ts=at(11, 10)), now=at(11, 10))
    assert decision.rejection.reason is RejectReason.CONSECUTIVE_LOSS_COOLDOWN

    # Once the cooldown expires, trading resumes.
    assert engine.evaluate_entry(intent(ts=at(11, 30)), now=at(11, 30)).approved


def test_a_winner_resets_the_consecutive_loss_counter(engine):
    engine.roll_session(at())
    engine.on_trade_closed(losing_trade(-50.0, -0.5))
    engine.on_trade_closed(losing_trade(-50.0, -0.5))
    winner = losing_trade(120.0, 1.2)
    engine.on_trade_closed(winner)
    assert engine.state.consecutive_losses == 0
    assert engine.state.cooldown_until is None


def test_the_trailing_drawdown_floor_is_terminal(tmp_path):
    # The cushion throttle is disabled here so the *floor* is what is being measured;
    # their interaction is pinned separately below.
    clock = SimulatedClock(at())
    cfg = risk_config(
        drawdown=DrawdownRisk(trailing_drawdown_pct=5.0, size_reduction_cushion_pct=0.0),
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "K.flag")),
    )
    engine = RiskEngine(cfg, MNQ, SessionConfig(), clock=clock,
                        kill_switch=KillSwitch(tmp_path / "K.flag", clock=clock))
    engine.roll_session(at())
    engine.state.peak_equity = 50_000.0

    assert engine.evaluate_entry(intent(), equity=47_501.0).approved, "one dollar above the floor"

    decision = engine.evaluate_entry(intent(stop=17989.0), equity=47_500.0)
    assert decision.rejection.reason is RejectReason.TRAILING_DRAWDOWN
    assert engine.floor_breached_at is not None
    assert engine.state.halted


def test_size_reaches_zero_before_the_floor_is_actually_hit(engine):
    """The throttle is meant to stop the account walking into the floor at full size.

    Approaching it, the risk budget shrinks to nothing and entries stop — which is the
    intended behaviour, and is why the floor test above disables the throttle to isolate
    the boundary.
    """
    engine.roll_session(at())
    engine.state.peak_equity = 50_000.0
    decision = engine.evaluate_entry(intent(), equity=47_501.0)
    assert not decision.approved
    assert decision.rejection.reason is RejectReason.SIZE_BELOW_MINIMUM


def test_the_drawdown_floor_can_be_disabled_for_measurement(tmp_path):
    clock = SimulatedClock(at())
    cfg = risk_config(kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "K.flag")))
    engine = RiskEngine(cfg, MNQ, SessionConfig(), clock=clock,
                        kill_switch=KillSwitch(tmp_path / "K.flag", clock=clock),
                        enforce_drawdown_floor=False)
    engine.roll_session(at())
    decision = engine.evaluate_entry(intent(), equity=40_000.0)
    assert decision.approved, "research mode must measure the whole sample"
    assert engine.floor_breached_at is not None, "but the breach is still recorded"


def test_an_identical_intent_is_rejected_as_a_duplicate(engine):
    first = intent()
    assert engine.evaluate_entry(first).approved
    again = engine.evaluate_entry(intent())  # same fingerprint, different intent_id
    assert again.rejection.reason is RejectReason.DUPLICATE_ORDER


def test_duplicate_memory_clears_on_a_new_session(engine):
    assert engine.evaluate_entry(intent()).approved
    assert engine.evaluate_entry(intent(ts=at(day=2)), now=at(day=2)).approved


# ============================================================ session windows


def test_entries_are_refused_inside_the_opening_buffer(engine):
    decision = engine.evaluate_entry(intent(ts=at(9, 33)), now=at(9, 33))
    assert decision.rejection.reason is RejectReason.OUTSIDE_TRADING_HOURS
    assert "opening buffer" in decision.rejection.detail


def test_entries_are_refused_inside_the_closing_buffer(engine):
    decision = engine.evaluate_entry(intent(ts=at(15, 50)), now=at(15, 50))
    assert decision.rejection.reason is RejectReason.OUTSIDE_TRADING_HOURS
    assert "closing buffer" in decision.rejection.detail


def test_entries_are_refused_outside_rth(engine):
    decision = engine.evaluate_entry(intent(ts=at(18, 0)), now=at(18, 0))
    assert decision.rejection.reason is RejectReason.OUTSIDE_TRADING_HOURS
    assert "outside RTH" in decision.rejection.detail


def test_the_session_guard_boundaries_are_exact():
    guard = SessionGuard(SessionConfig(entry_open_buffer_minutes=5,
                                       entry_close_buffer_minutes=15,
                                       flatten_before_close_minutes=2))
    assert not guard.may_enter(at(9, 34))[0]
    assert guard.may_enter(at(9, 35))[0]
    assert guard.may_enter(at(15, 44))[0]
    assert not guard.may_enter(at(15, 45))[0]
    assert not guard.must_flatten(at(15, 57))
    assert guard.must_flatten(at(15, 58))


def test_news_blackout_windows_block_entries():
    guard = SessionGuard(SessionConfig(news_blackout_windows=(
        {"start": "2024-04-01T14:00:00-04:00", "end": "2024-04-01T14:30:00-04:00"},
    )))
    assert not guard.may_enter(at(14, 15))[0]
    assert guard.may_enter(at(14, 15))[1] is RejectReason.NEWS_BLACKOUT
    assert guard.may_enter(at(13, 59))[0]
    assert guard.may_enter(at(14, 30))[0]


def test_a_malformed_blackout_window_is_a_load_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="tz-aware"):
        SessionGuard(SessionConfig(news_blackout_windows=({"start": "nonsense"},)))
    with pytest.raises(ValueError, match="ends before"):
        SessionGuard(SessionConfig(news_blackout_windows=(
            {"start": "2024-04-01T14:30:00-04:00", "end": "2024-04-01T14:00:00-04:00"},
        )))


# ============================================================ exits are never blocked


def test_an_exit_is_approved_even_when_every_entry_limit_is_breached(engine, position_factory):
    engine.roll_session(at())
    engine.state.daily_realized_pnl = -5000.0
    engine.state.halted = True
    engine.kill_switch.trip("test")

    decision = engine.evaluate_exit(position_factory(side=Side.BUY, quantity=2))
    assert decision.approved
    assert decision.approval.order.side is Side.SELL
    assert decision.approval.order.quantity == 2


def test_an_open_position_must_flatten_at_the_flatten_time(engine, position_factory):
    pos = position_factory()
    assert engine.forced_exit_reason(pos, 18000.0, at(15, 57)) is None
    reason, detail = engine.forced_exit_reason(pos, 18000.0, at(15, 58))
    assert reason is RejectReason.OUTSIDE_TRADING_HOURS
    assert "no overnight positions" in detail


def test_an_open_position_is_flattened_when_the_marked_daily_loss_hits_the_cap(
    engine, position_factory
):
    engine.roll_session(at())
    engine.state.daily_realized_pnl = -600.0
    pos = position_factory(side=Side.BUY, entry=18000.0, quantity=3)
    # 3 contracts x $2 x -70 points = -$420 unrealised; -600 - 420 = -1020, past the cap.
    result = engine.forced_exit_reason(pos, 17930.0, at(12, 0))
    assert result is not None and result[0] is RejectReason.MAX_DAILY_LOSS


def test_the_kill_switch_forces_an_open_position_flat(engine, position_factory):
    engine.kill_switch.trip("manual stop")
    reason, detail = engine.forced_exit_reason(position_factory(), 18000.0, at(12, 0))
    assert reason is RejectReason.KILL_SWITCH_ACTIVE
    assert "manual stop" in detail


# ============================================================ kill switch


def test_the_kill_switch_blocks_new_entries(engine):
    engine.kill_switch.trip("because")
    decision = engine.evaluate_entry(intent())
    assert decision.rejection.reason is RejectReason.KILL_SWITCH_ACTIVE


def test_the_kill_switch_survives_a_restart(tmp_path):
    flag = tmp_path / "K.flag"
    KillSwitch(flag).trip("overnight incident", by="human")
    assert KillSwitch(flag).is_active()
    assert "overnight incident" in KillSwitch(flag).state().reason


def test_tripping_twice_keeps_the_first_reason(tmp_path):
    ks = KillSwitch(tmp_path / "K.flag")
    ks.trip("the real trigger")
    ks.trip("a later symptom")
    assert ks.state().reason == "the real trigger"


def test_clearing_the_kill_switch_needs_an_explicit_call(tmp_path):
    ks = KillSwitch(tmp_path / "K.flag")
    ks.trip("stop")
    assert ks.is_active()
    ks.clear()
    assert not ks.is_active()


def test_an_unreadable_flag_file_still_counts_as_engaged(tmp_path):
    flag = tmp_path / "K.flag"
    flag.write_text("not json at all", encoding="utf-8")
    state = KillSwitch(flag).state()
    assert state.active, "failing open here would defeat the whole mechanism"


def test_consecutive_errors_trip_the_switch(tmp_path):
    ks = KillSwitch(tmp_path / "K.flag", max_consecutive_errors=3)
    assert not ks.record_error("broker timeout")
    assert not ks.record_error("broker timeout")
    assert ks.record_error("broker timeout")
    assert ks.is_active()
    assert "3 consecutive errors" in ks.state().reason


def test_a_success_resets_the_error_run(tmp_path):
    ks = KillSwitch(tmp_path / "K.flag", max_consecutive_errors=3)
    ks.record_error("a")
    ks.record_error("b")
    ks.record_success()
    assert ks.consecutive_errors == 0
    assert not ks.record_error("c")
    assert not ks.is_active()


# ============================================================ tokens


def _order(qty=3, side=Side.BUY, purpose="ENTRY") -> Order:
    return Order(order_id="o1", timestamp=at(), instrument="MNQ", side=side,
                 quantity=qty, order_type=OrderType.MARKET, purpose=purpose)


def test_a_token_authorises_exactly_the_order_it_was_issued_for():
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    mint.verify(order, token, now=at())  # does not raise


def test_changing_the_quantity_after_approval_invalidates_the_token():
    mint = TokenMint(ttl_seconds=60)
    token = mint.issue(_order(qty=1), now=at(), risk_usd=20.0, stop_price=17990.0)
    with pytest.raises(TokenError) as err:
        mint.verify(_order(qty=10), token, now=at())
    assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH


def test_changing_the_side_after_approval_invalidates_the_token():
    mint = TokenMint(ttl_seconds=60)
    token = mint.issue(_order(side=Side.BUY), now=at(), risk_usd=20.0, stop_price=17990.0)
    with pytest.raises(TokenError):
        mint.verify(_order(side=Side.SELL), token, now=at())


def test_editing_the_claimed_risk_invalidates_the_token():
    """The risk facts ride inside the signature, so they cannot be edited in transit."""
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    tampered = replace(token, risk_usd=99_999.0)

    with pytest.raises(TokenError) as err:
        mint.verify(order, tampered, now=at())
    assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH


def test_editing_the_stop_price_invalidates_the_token():
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    with pytest.raises(TokenError):
        mint.verify(order, replace(token, stop_price=1.0), now=at())


def test_extending_a_tokens_own_expiry_does_not_help():
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    stretched = replace(token, expires_at=at() + timedelta(days=365))
    with pytest.raises(TokenError) as err:
        mint.verify(order, stretched, now=at() + timedelta(seconds=120))
    assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH


@pytest.mark.parametrize("field", ["issued_at", "token_id", "authorization_kind"])
def test_editing_token_identity_or_authorization_invalidates_its_signature(field):
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    changes = {
        "issued_at": at() - timedelta(seconds=1),
        "token_id": "tok-substituted",
        "authorization_kind": AuthorizationKind.EXIT,
    }

    with pytest.raises(TokenError) as err:
        mint.verify(order, replace(token, **{field: changes[field]}), now=at())
    assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH


def test_every_immutable_order_identity_and_economic_field_is_signed():
    mint = TokenMint(ttl_seconds=60)
    order = Order(
        order_id="protect-1",
        timestamp=at(),
        instrument="MNQ",
        side=Side.SELL,
        quantity=2,
        order_type=OrderType.STOP_LIMIT,
        limit_price=17989.75,
        stop_price=17990.0,
        time_in_force=TimeInForce.GTC,
        intent_id="intent-1",
        strategy="deterministic",
        purpose=OrderPurpose.STOP,
        oco_group="oco-1",
    )
    token = mint.issue(
        order,
        now=at(),
        risk_usd=40.0,
        stop_price=17990.0,
        authorization_kind=AuthorizationKind.PROTECTIVE,
    )
    mutations = (
        {"order_id": "protect-2"},
        {"timestamp": at() + timedelta(microseconds=1)},
        {"instrument": "MES"},
        {"side": Side.BUY},
        {"quantity": 3},
        {"order_type": OrderType.STOP},
        {"limit_price": 17989.5},
        {"stop_price": 17989.75},
        {"time_in_force": TimeInForce.DAY},
        {"intent_id": "intent-2"},
        {"strategy": "other"},
        {"purpose": OrderPurpose.TARGET},
        {"oco_group": "oco-2"},
    )

    for mutation in mutations:
        with pytest.raises(TokenError) as err:
            mint.verify(replace(order, **mutation), token, now=at())
        assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH, mutation


def test_a_token_for_an_exit_cannot_authorise_an_entry():
    mint = TokenMint(ttl_seconds=60)
    exit_order = _order(purpose="EXIT")
    token = mint.issue(
        exit_order,
        now=at(),
        risk_usd=0.0,
        stop_price=17990.0,
        authorization_kind=AuthorizationKind.EXIT,
    )
    with pytest.raises(TokenError) as err:
        mint.verify(_order(purpose="ENTRY"), token, now=at())
    assert err.value.reason is RejectReason.TOKEN_BINDING_MISMATCH


def test_token_mint_refuses_an_authorization_kind_that_does_not_match_order_role():
    mint = TokenMint(ttl_seconds=60)
    with pytest.raises(ValueError, match="EXIT authorization cannot be issued for ENTRY"):
        mint.issue(
            _order(),
            now=at(),
            risk_usd=0.0,
            stop_price=17990.0,
            authorization_kind=AuthorizationKind.EXIT,
        )


def test_a_token_from_another_engine_is_rejected():
    order = _order()
    stolen = TokenMint().issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    with pytest.raises(TokenError):
        TokenMint().verify(order, stolen, now=at())


def test_an_expired_token_is_rejected():
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    with pytest.raises(TokenError) as err:
        mint.verify(order, token, now=at() + timedelta(seconds=61))
    assert err.value.reason is RejectReason.TOKEN_EXPIRED


def test_a_token_can_only_be_spent_once():
    mint = TokenMint(ttl_seconds=60)
    order = _order()
    token = mint.issue(order, now=at(), risk_usd=60.0, stop_price=17990.0)
    mint.verify(order, token, now=at())
    mint.spend(token)
    with pytest.raises(TokenError) as err:
        mint.verify(order, token, now=at())
    assert err.value.reason is RejectReason.TOKEN_ALREADY_USED
    assert "duplicate order" in err.value.detail


# ============================================================ broker-side re-validation


def test_verify_accepts_a_fresh_approval(engine):
    approval = engine.evaluate_entry(intent()).approval
    assert engine.verify(
        approval.order, approval.token, broker_snapshot=broker_snapshot()
    ) is None


def test_explicit_broker_identity_pins_accept_only_the_exact_snapshot(engine):
    pinned = RiskEngine(
        engine.config,
        MNQ,
        SessionConfig(),
        clock=engine.clock,
        kill_switch=engine.kill_switch,
        expected_broker_account_id="TEST-50K",
        expected_broker_name="test-broker",
        expected_broker_is_paper=True,
    )
    approval = pinned.evaluate_entry(intent()).approval

    assert pinned.verify(
        approval.order,
        approval.token,
        broker_snapshot=broker_snapshot(),
    ) is None


@pytest.mark.parametrize("mismatch", ["account", "broker", "paper"])
def test_explicit_broker_identity_pin_mismatch_rejects_without_spending_token(
    engine, mismatch: str
):
    pinned = RiskEngine(
        engine.config,
        MNQ,
        SessionConfig(),
        clock=engine.clock,
        kill_switch=engine.kill_switch,
        expected_broker_account_id="TEST-50K",
        expected_broker_name="test-broker",
        expected_broker_is_paper=True,
    )
    approval = pinned.evaluate_entry(intent()).approval
    changes = {
        "account": {"account_id": "WRONG-ACCOUNT"},
        "broker": {"broker_name": "wrong-broker"},
        "paper": {"is_paper": False},
    }[mismatch]
    snapshot = broker_snapshot(**changes)

    rejection = pinned.verify(
        approval.order,
        approval.token,
        broker_snapshot=snapshot,
    )

    assert rejection is not None
    assert rejection.reason is RejectReason.LIVE_TRADING_DISABLED
    assert not pinned._tokens.is_spent(approval.token)
    assert "WRONG-ACCOUNT" not in rejection.detail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_broker_account_id": ""},
        {"expected_broker_name": "  "},
        {"expected_broker_is_paper": "true"},
    ],
)
def test_invalid_explicit_broker_identity_pin_is_rejected(engine, kwargs):
    with pytest.raises(ValueError, match="expected_broker"):
        RiskEngine(
            engine.config,
            MNQ,
            SessionConfig(),
            clock=engine.clock,
            kill_switch=engine.kill_switch,
            **kwargs,
        )


def test_verify_refuses_entry_without_an_authoritative_broker_snapshot(engine):
    approval = engine.evaluate_entry(intent()).approval
    rejection = engine.verify(approval.order, approval.token)
    assert rejection.reason is RejectReason.BROKER_ERROR
    assert rejection.stage == "BROKER_SNAPSHOT"


def test_verify_refuses_an_approval_spent_twice(engine):
    approval = engine.evaluate_entry(intent()).approval
    snapshot = broker_snapshot()
    assert engine.verify(
        approval.order, approval.token, broker_snapshot=snapshot
    ) is None
    second = engine.verify(
        approval.order, approval.token, broker_snapshot=snapshot
    )
    assert second is not None and second.reason is RejectReason.TOKEN_ALREADY_USED


def test_verify_refuses_an_approval_that_predates_a_fresh_breach(engine):
    """The core of layer 3: approval is not a permanent licence."""
    approval = engine.evaluate_entry(intent()).approval
    engine.state.daily_realized_pnl = -1500.0  # blew the cap after approval

    rejection = engine.verify(
        approval.order, approval.token, broker_snapshot=broker_snapshot()
    )
    assert rejection is not None
    assert rejection.reason is RejectReason.MAX_DAILY_LOSS
    assert rejection.stage == "GUARD"


def test_verify_refuses_an_approval_that_predates_the_kill_switch(engine):
    approval = engine.evaluate_entry(intent()).approval
    engine.kill_switch.trip("stop everything")
    rejection = engine.verify(approval.order, approval.token)
    assert rejection.reason is RejectReason.KILL_SWITCH_ACTIVE


def test_verify_refuses_an_approval_after_a_position_opens(engine):
    approval = engine.evaluate_entry(intent()).approval
    engine.state.open_position_id = "pos-opened-after-approval"
    rejection = engine.verify(
        approval.order, approval.token, broker_snapshot=broker_snapshot()
    )
    assert rejection is not None
    assert rejection.reason is RejectReason.POSITION_ALREADY_OPEN
    assert rejection.stage == "GUARD"


def test_verify_still_lets_an_exit_through_after_the_kill_switch(engine, position_factory):
    position = position_factory()
    approval = engine.evaluate_exit(position).approval
    engine.kill_switch.trip("stop everything")
    snapshot = broker_snapshot(
        positions=(BrokerRiskPosition("MNQ", position.quantity, position.entry_price),)
    )
    assert engine.verify(
        approval.order, approval.token, broker_snapshot=snapshot
    ) is None


def test_verify_accepts_an_exact_short_position_exit(engine, position_factory):
    position = position_factory(side=Side.SELL, quantity=2)
    approval = engine.evaluate_exit(position).approval
    snapshot = broker_snapshot(
        positions=(BrokerRiskPosition("MNQ", -2, position.entry_price),)
    )

    assert engine.verify(
        approval.order, approval.token, broker_snapshot=snapshot
    ) is None


@pytest.mark.parametrize(
    "positions",
    [
        (),
        (BrokerRiskPosition("MES", 1, 5_000.0),),
        (BrokerRiskPosition("MNQ", 2, 18_000.0),),
        (BrokerRiskPosition("MNQ", -1, 18_000.0),),
    ],
    ids=("flat", "instrument", "quantity", "side"),
)
def test_verify_rejects_a_fabricated_or_mismatched_exit_without_spending_token(
    engine, position_factory, positions
):
    position = position_factory(quantity=1)
    approval = engine.evaluate_exit(position).approval

    rejection = engine.verify(
        approval.order,
        approval.token,
        broker_snapshot=broker_snapshot(positions=positions),
    )

    assert rejection is not None
    assert rejection.reason is RejectReason.INVALID_ORDER
    assert rejection.stage == "BROKER_SNAPSHOT"
    assert not engine._tokens.is_spent(approval.token)


def test_verify_allows_targeted_exit_when_another_instrument_has_exposure(
    engine, position_factory
):
    position = position_factory(quantity=1)
    approval = engine.evaluate_exit(position).approval
    snapshot = broker_snapshot(
        positions=(
            BrokerRiskPosition("MNQ", 1, position.entry_price),
            BrokerRiskPosition("MES", -2, 5_000.0),
        )
    )

    assert engine.verify(
        approval.order,
        approval.token,
        broker_snapshot=snapshot,
    ) is None


def test_verify_uses_aggregate_exposure_for_the_requested_instrument(
    engine, position_factory
):
    position = position_factory(quantity=3)
    approval = engine.evaluate_exit(position).approval
    snapshot = broker_snapshot(
        positions=(
            BrokerRiskPosition("MNQ", 1, position.entry_price),
            BrokerRiskPosition("MNQ", 2, position.entry_price + 1.0),
            BrokerRiskPosition("MES", 1, 5_000.0),
        )
    )

    assert engine.verify(
        approval.order,
        approval.token,
        broker_snapshot=snapshot,
    ) is None


def _protective_stop(position, **changes) -> Order:
    values = dict(
        order_id="stop-1",
        timestamp=at(),
        instrument=position.instrument,
        side=position.side.opposite,
        quantity=position.quantity,
        order_type=OrderType.STOP,
        stop_price=position.stop_price,
        strategy=position.strategy,
        purpose=OrderPurpose.STOP,
        oco_group="oco-1",
    )
    values.update(changes)
    return Order(**values)


def test_risk_engine_exposes_no_public_token_mint(engine):
    assert not hasattr(engine, "tokens")


def test_a_valid_protective_stop_gets_a_signed_protective_authorization(
    engine, position_factory
):
    position = position_factory(quantity=2)
    order = _protective_stop(position)
    approval = engine.evaluate_protective(position, order).approval

    assert approval is not None
    assert approval.token.authorization_kind is AuthorizationKind.PROTECTIVE
    snapshot = broker_snapshot(
        positions=(BrokerRiskPosition("MNQ", 2, position.entry_price),)
    )
    assert engine.verify(order, approval.token, broker_snapshot=snapshot) is None


def test_verify_rejects_a_protective_order_when_the_broker_is_flat(
    engine, position_factory
):
    position = position_factory(quantity=1)
    order = _protective_stop(position)
    approval = engine.evaluate_protective(position, order).approval

    rejection = engine.verify(
        order,
        approval.token,
        broker_snapshot=broker_snapshot(),
    )

    assert rejection is not None
    assert rejection.reason is RejectReason.INVALID_ORDER
    assert rejection.stage == "BROKER_SNAPSHOT"
    assert not engine._tokens.is_spent(approval.token)


@pytest.mark.parametrize(
    "changes",
    [
        {"instrument": "MES"},
        {"side": Side.BUY},
        {"quantity": 2},
        {
            "order_type": OrderType.LIMIT,
            "limit_price": 17990.0,
            "stop_price": None,
        },
        {"stop_price": 17989.75},
        {"stop_price": 17990.10},
        {"strategy": "some-other-position"},
        {"oco_group": None},
    ],
)
def test_protective_authorization_rejects_any_order_that_is_not_proven_reducing(
    engine, position_factory, changes
):
    position = position_factory(quantity=1)
    decision = engine.evaluate_protective(position, _protective_stop(position, **changes))
    assert not decision.approved
    assert decision.rejection.reason is RejectReason.INVALID_ORDER


def test_short_protective_stop_cannot_be_widened_higher(engine, position_factory):
    position = position_factory(side=Side.SELL, stop=18010.0)
    decision = engine.evaluate_protective(
        position, _protective_stop(position, stop_price=18010.25)
    )
    assert not decision.approved
    assert "widened" in decision.rejection.detail


def test_a_target_must_match_the_positions_approved_price(engine, position_factory):
    position = position_factory(quantity=2, target=18020.0)
    target = Order(
        order_id="target-1",
        timestamp=at(),
        instrument="MNQ",
        side=Side.SELL,
        quantity=2,
        order_type=OrderType.LIMIT,
        limit_price=18020.0,
        strategy=position.strategy,
        purpose=OrderPurpose.TARGET,
        oco_group="oco-1",
    )
    assert engine.evaluate_protective(position, target).approved
    refused = engine.evaluate_protective(
        position, replace(target, order_id="target-2", limit_price=18019.75)
    )
    assert not refused.approved


def test_verify_refuses_an_oversized_order_even_with_a_valid_token(engine):
    """A token proves the engine approved *something*; the size check is independent."""
    approval = engine.evaluate_entry(intent()).approval
    oversized = Order(
        order_id=approval.order.order_id, timestamp=approval.order.timestamp,
        instrument="MNQ", side=Side.BUY, quantity=99, order_type=OrderType.MARKET,
        purpose="ENTRY",
    )
    rejection = engine.verify(oversized, approval.token)
    # The binding check catches it first, which is the stronger guarantee.
    assert rejection is not None
    assert rejection.reason in (
        RejectReason.TOKEN_BINDING_MISMATCH, RejectReason.MAX_POSITION_SIZE
    )


def test_verify_refuses_an_entry_once_the_session_window_has_closed(engine):
    # Approved just inside the entry window, submitted forty seconds later, by which time
    # the closing buffer has begun. Still inside the token's 60-second TTL, so it is the
    # session check that refuses it rather than expiry.
    approved_at = at(15, 44) + timedelta(seconds=30)  # 15:44:30, entries close at 15:45
    approval = engine.evaluate_entry(intent(ts=approved_at), now=approved_at).approval
    rejection = engine.verify(
        approval.order,
        approval.token,
        now=approved_at + timedelta(seconds=40),
        broker_snapshot=broker_snapshot(ts=approved_at + timedelta(seconds=40)),
    )
    assert rejection is not None
    assert rejection.reason is RejectReason.OUTSIDE_TRADING_HOURS


# ============================================================ session lifecycle


def test_rolling_the_session_resets_the_daily_counters(engine):
    engine.roll_session(at())
    engine.state.daily_realized_pnl = -500.0
    engine.state.trades_today = 4
    engine.state.halted = True

    assert engine.roll_session(at(day=2))
    assert engine.state.daily_realized_pnl == 0.0
    assert engine.state.trades_today == 0
    assert not engine.state.halted


def test_rolling_within_the_same_day_changes_nothing(engine):
    engine.roll_session(at(10, 0))
    engine.state.trades_today = 2
    assert not engine.roll_session(at(14, 0))
    assert engine.state.trades_today == 2


def test_equity_and_peak_track_closed_trades(engine):
    engine.roll_session(at())
    engine.on_trade_closed(losing_trade(500.0, 2.0))
    assert engine.state.equity == 50_500.0
    assert engine.state.peak_equity == 50_500.0

    engine.on_trade_closed(losing_trade(-200.0, -1.0))
    assert engine.state.equity == 50_300.0
    assert engine.state.peak_equity == 50_500.0, "the peak never falls"


def test_the_cushion_fraction_reports_how_much_allowance_is_left(engine):
    engine.state.peak_equity = 50_000.0
    engine.state.equity = 50_000.0
    assert engine.cushion_fraction() == 1.0
    engine.state.equity = 48_750.0  # half of the $2,500 allowance used
    assert engine.cushion_fraction() == pytest.approx(0.5)
    engine.state.equity = 47_000.0
    assert engine.cushion_fraction() == 0.0


def test_the_limits_snapshot_covers_every_configured_limit(engine):
    snap = engine.limits_snapshot()
    for key in ("daily_loss_limit", "daily_r_limit", "max_trades_per_day",
                "max_open_positions",
                "max_consecutive_losses", "max_contracts", "max_risk_per_trade_usd",
                "drawdown_floor", "cushion_pct", "kill_switch", "session_window"):
        assert key in snap
