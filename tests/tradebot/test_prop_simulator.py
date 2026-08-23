from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tradebot.prop_firms import load_prop_profile
from tradebot.prop_simulator import (
    HistoricalPropSession,
    HistoricalPropTrade,
    JourneyScenario,
    JourneyStage,
    PropSimulationPolicy,
    SimulationEventKind,
    SimulationTermination,
    TradeOutcomeAdjustment,
    apply_trade_outcome_adjustment,
    compare_journey_scenarios,
    run_journey_session_bootstrap,
    run_session_bootstrap,
    simulate_evaluation_to_funded,
    simulate_prop_phase,
)
from tradebot.risk.prop import MarketDayStatus


ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
MONDAY = date(2026, 8, 24)


def _profile(name: str = "growth"):
    return load_prop_profile(ROOT / "config" / "prop_firms" / f"tradeify_{name}_50k.yaml")


def _weekdays(count: int, *, start: date = MONDAY) -> tuple[date, ...]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return tuple(result)


def _trade(
    session_date: date,
    pnl: float,
    *,
    adverse: float | None = None,
    hold_seconds: float = 11.0,
    minute: int = 0,
    commission: float = 0.0,
    fees: float = 0.0,
    slippage: float = 0.0,
    planned_risk: float = 200.0,
    micros: int = 1,
    trade_id: str = "",
) -> HistoricalPropTrade:
    opened = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        10,
        minute,
        tzinfo=ET,
    )
    return HistoricalPropTrade(
        opened_at=opened,
        closed_at=opened + timedelta(seconds=hold_seconds),
        net_pnl_usd=pnl,
        minimum_intratrade_pnl_usd=min(0.0, pnl) if adverse is None else adverse,
        planned_risk_usd=planned_risk,
        commission_usd=commission,
        fees_usd=fees,
        slippage_usd=slippage,
        micros=micros,
        trade_id=trade_id,
    )


def _session(
    session_date: date,
    *trades: HistoricalPropTrade,
    status: MarketDayStatus = MarketDayStatus.REGULAR,
) -> HistoricalPropSession:
    return HistoricalPropSession(
        session_date=session_date,
        market_day_status=status,
        trades=tuple(trades),
    )


def test_evaluation_starts_fresh_and_passes_only_after_eod_transition() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    trade = _trade(
        MONDAY,
        3_000,
        adverse=0,
        commission=12.0,
        fees=3.0,
        slippage=4.0,
        trade_id="winner",
    )

    result = simulate_prop_phase(profile, rules, (_session(MONDAY, trade),))

    assert result.initial_balance_usd == 50_000
    assert result.final_balance_usd == 53_000
    assert result.net_profit_usd == 3_000  # explicit costs were already in net P&L
    assert result.total_commission_usd == 12
    assert result.total_fees_usd == 3
    assert result.total_slippage_usd == 4
    assert result.evaluation_passed
    assert result.sessions_to_pass == 1
    assert result.trading_days == 1
    assert result.termination is SimulationTermination.EVALUATION_PASSED
    assert result.final_state.highest_end_of_day_balance_usd == 53_000
    assert result.final_state.drawdown_floor_usd == 51_000
    assert result.events[-1].kind is SimulationEventKind.EVALUATION_PASSED


def test_adverse_mark_hits_internal_threshold_before_better_historical_close() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    sessions = (
        _session(day1, _trade(day1, 1_000, adverse=0, trade_id="up")),
        _session(day2, _trade(day2, 200, adverse=-1_600, trade_id="reversal")),
    )

    result = simulate_prop_phase(profile, rules, sessions)

    assert result.internal_safety_locked
    assert not result.account_failed
    assert result.termination is SimulationTermination.INTERNAL_SAFETY_LOCK
    assert result.final_balance_usd == 49_400
    assert result.final_state.drawdown_floor_usd == 49_000
    assert result.final_state.internal_safety_threshold_usd == 49_400
    assert result.maximum_drawdown_usd == 1_600
    assert result.minimum_distance_to_failure_usd == 400
    assert any(event.kind is SimulationEventKind.INTERNAL_SAFETY_LOCK for event in result.events)


def test_exact_firm_floor_is_account_failure_and_takes_priority_over_zero_buffer() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    losing_trade = _trade(MONDAY, 100, adverse=-2_000, trade_id="floor-touch")

    result = simulate_prop_phase(
        profile,
        rules,
        (_session(MONDAY, losing_trade),),
        internal_safety_buffer_usd=0,
    )

    assert result.account_failed
    assert not result.internal_safety_locked
    assert result.termination is SimulationTermination.ACCOUNT_FAILED
    assert result.final_balance_usd == 48_000
    assert result.minimum_distance_to_failure_usd == 0
    assert result.final_state.hard_breached
    assert result.final_state.hard_breach_reasons == ("trailing_drawdown",)


def test_soft_daily_loss_lock_rejects_remaining_session_trades() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    first = _trade(MONDAY, -1_250, minute=0, trade_id="dll")
    second = _trade(MONDAY, 2_000, adverse=0, minute=2, trade_id="blocked")

    result = simulate_prop_phase(profile, rules, (_session(MONDAY, first, second),))

    assert not result.account_failed
    assert not result.internal_safety_locked
    assert result.final_balance_usd == 48_750
    assert result.executed_trades == 1
    assert result.rejected_trades == 1
    assert result.rejected_daily_loss_trades == 1
    assert result.final_state.soft_daily_loss_locked
    assert result.max_consecutive_losses == 1
    assert any(
        event.kind is SimulationEventKind.TRADE_REJECTED_DAILY_LOSS
        for event in result.events
    )


def test_dynamic_contract_cap_rejects_without_changing_account() -> None:
    profile = _profile("select")
    rules = profile.rules_for("sim_funded_flex")
    too_large = _trade(MONDAY, 500, adverse=0, minute=0, micros=21, trade_id="oversize")
    allowed = _trade(MONDAY, 100, adverse=0, minute=2, micros=20, trade_id="allowed")

    result = simulate_prop_phase(profile, rules, (_session(MONDAY, too_large, allowed),))

    assert result.executed_trades == 1
    assert result.rejected_contract_limit_trades == 1
    assert result.final_balance_usd == 50_100
    assert result.final_state.allowed_micros == 20


def test_personal_policy_is_immutable_and_cannot_weaken_pinned_limits() -> None:
    policy = PropSimulationPolicy()
    assert policy.max_planned_risk_per_trade_usd == 200
    assert policy.max_trades_per_session == 1
    assert policy.max_daily_strategy_loss_usd == 200
    assert policy.max_open_positions == 1
    with pytest.raises(FrozenInstanceError):
        policy.max_trades_per_session = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="no greater than 200"):
        replace(policy, max_planned_risk_per_trade_usd=201).validate()
    with pytest.raises(ValueError, match="exactly 1"):
        replace(policy, max_trades_per_session=2).validate()


def test_personal_risk_cap_and_trade_quota_rejections_do_not_change_equity() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    over_risk = _trade(
        day1,
        1_000,
        adverse=0,
        planned_risk=200.01,
        commission=50,
        fees=25,
        trade_id="over-risk",
    )
    first = _trade(day2, 100, adverse=0, minute=0, trade_id="first")
    quota = _trade(
        day2,
        2_000,
        adverse=0,
        minute=2,
        commission=50,
        fees=25,
        trade_id="quota",
    )

    result = simulate_prop_phase(
        profile,
        rules,
        (_session(day1, over_risk), _session(day2, first, quota)),
    )

    assert result.executed_trades == 1
    assert result.rejected_personal_risk_trades == 1
    assert result.rejected_trade_quota_trades == 1
    assert result.final_balance_usd == 50_100
    assert result.rejected_trades == 2
    assert result.total_commission_usd == 0
    assert result.total_fees_usd == 0


def test_daily_personal_loss_gate_uses_planned_post_stop_equity() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    policy = PropSimulationPolicy(max_daily_strategy_loss_usd=100)
    signal = _trade(MONDAY, 500, adverse=0, planned_risk=100.01)

    result = simulate_prop_phase(
        profile,
        rules,
        (_session(MONDAY, signal),),
        policy=policy,
    )

    assert result.executed_trades == 0
    assert result.rejected_daily_strategy_loss_trades == 1
    assert result.final_balance_usd == 50_000


def test_risk_cannot_increase_after_a_loss_even_on_the_next_session() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2, day3 = _weekdays(3)
    sessions = (
        _session(day1, _trade(day1, -100, planned_risk=100, trade_id="loss")),
        _session(
            day2,
            _trade(day2, 1_000, adverse=0, planned_risk=150, trade_id="increase"),
        ),
        _session(
            day3,
            _trade(day3, 200, adverse=0, planned_risk=100, trade_id="same-risk"),
        ),
    )

    result = simulate_prop_phase(profile, rules, sessions)

    assert result.executed_trades == 2
    assert result.rejected_post_loss_risk_increase_trades == 1
    assert result.final_balance_usd == 50_100
    assert result.max_consecutive_losses == 1


def test_planned_risk_must_fit_inside_the_internal_threshold_before_entry() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    sessions = (
        _session(day1, _trade(day1, -1_500, planned_risk=200, trade_id="gap-loss")),
        _session(
            day2,
            _trade(day2, 500, adverse=0, planned_risk=100, trade_id="too-close"),
        ),
    )

    result = simulate_prop_phase(profile, rules, sessions)

    assert not result.internal_safety_locked
    assert not result.account_failed
    assert result.executed_trades == 1
    assert result.rejected_pretrade_safety_trades == 1
    assert result.final_balance_usd == 48_500
    assert result.final_state.internal_safety_threshold_usd == 48_400


def test_closed_and_unknown_market_days_reject_every_signal() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    result = simulate_prop_phase(
        profile,
        rules,
        (
            _session(
                day1,
                _trade(day1, 500, adverse=0, trade_id="closed"),
                status=MarketDayStatus.CLOSED,
            ),
            _session(
                day2,
                _trade(day2, 500, adverse=0, trade_id="unknown"),
                status=MarketDayStatus.UNKNOWN,
            ),
        ),
    )

    assert result.executed_trades == 0
    assert result.rejected_market_day_trades == 2
    assert result.final_balance_usd == 50_000
    assert result.sessions_elapsed == 2


def test_early_close_uses_the_holiday_flatten_deadline() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    allowed = replace(
        _trade(MONDAY, 100, adverse=0),
        opened_at=datetime(2026, 8, 24, 12, 58, 50, tzinfo=ET),
        closed_at=datetime(2026, 8, 24, 12, 59, 0, tzinfo=ET),
    )
    result = simulate_prop_phase(
        profile,
        rules,
        (_session(MONDAY, allowed, status=MarketDayStatus.EARLY_CLOSE),),
    )
    assert result.executed_trades == 1

    late = replace(
        allowed,
        opened_at=datetime(2026, 8, 24, 12, 59, 0, tzinfo=ET),
        closed_at=datetime(2026, 8, 24, 12, 59, 1, tzinfo=ET),
    )
    with pytest.raises(ValueError, match="flat by"):
        simulate_prop_phase(
            profile,
            rules,
            (_session(MONDAY, late, status=MarketDayStatus.EARLY_CLOSE),),
        )


def test_gross_before_costs_drives_select_consistency_without_changing_balance() -> None:
    profile = _profile("select")
    rules = profile.rules_for("evaluation")
    day1, day2, day3 = _weekdays(3)
    sessions = (
        _session(day1, _trade(day1, 1_300, adverse=0, commission=0)),
        _session(day2, _trade(day2, 850, adverse=0, commission=300)),
        _session(day3, _trade(day3, 850, adverse=0, commission=300)),
    )

    result = simulate_prop_phase(profile, rules, sessions)

    assert result.final_balance_usd == 53_000
    assert result.total_commission_usd == 600
    assert result.final_state.cumulative_consistency_profit_usd == 3_600
    assert result.final_state.best_day_profit_usd == 1_300
    assert result.final_state.evaluation_consistency_ratio == pytest.approx(1_300 / 3_600)
    assert result.evaluation_passed


def test_zero_signal_sessions_are_elapsed_but_not_counted_as_trading_days() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2, day3 = _weekdays(3)

    result = simulate_prop_phase(
        profile,
        rules,
        (
            _session(day1),
            _session(day2),
            _session(day3, _trade(day3, 3_000, adverse=0)),
        ),
    )

    assert result.evaluation_passed
    assert result.sessions_to_pass == 3
    assert result.sessions_elapsed == 3
    assert result.trading_days == 1


def test_evaluation_to_funded_journey_starts_a_fresh_account_and_reaches_payout() -> None:
    profile = _profile()
    evaluation_rules = profile.rules_for("evaluation")
    funded_rules = profile.rules_for("sim_funded")
    days = _weekdays(7)
    sessions = [
        _session(days[0], _trade(days[0], 3_000, adverse=0, trade_id="eval-pass"))
    ]
    sessions.extend(
        _session(
            session_date,
            _trade(session_date, 500, adverse=0, hold_seconds=11, trade_id=f"funded-{index}"),
        )
        for index, session_date in enumerate(days[1:], start=1)
    )

    result = simulate_evaluation_to_funded(
        profile,
        evaluation_rules,
        funded_rules,
        tuple(sessions),
    )

    assert result.evaluation_passed
    assert result.funded_started
    assert result.payout_eligible
    assert result.terminal_stage is JourneyStage.FUNDED
    assert result.failure_stage is None
    assert result.termination is SimulationTermination.PAYOUT_ELIGIBLE
    assert result.sessions_to_evaluation_pass == 1
    assert result.sessions_to_payout == 7
    assert result.total_sessions_elapsed == 7
    assert result.account_lifetime_sessions == 7
    assert result.evaluation.final_balance_usd == 53_000
    assert result.funded is not None
    assert result.funded.initial_balance_usd == 50_000
    assert result.funded.final_balance_usd == 53_000
    assert result.combined_net_profit_usd == 6_000


def test_journey_identifies_a_funded_stage_account_failure() -> None:
    profile = _profile()
    days = _weekdays(2)
    sessions = (
        _session(days[0], _trade(days[0], 3_000, adverse=0, trade_id="eval-pass")),
        _session(days[1], _trade(days[1], 100, adverse=-2_000, trade_id="funded-fail")),
    )

    result = simulate_evaluation_to_funded(
        profile,
        profile.rules_for("evaluation"),
        profile.rules_for("sim_funded"),
        sessions,
        internal_safety_buffer_usd=0,
    )

    assert result.evaluation_passed
    assert result.account_failed
    assert result.failure_stage is JourneyStage.FUNDED
    assert result.terminal_stage is JourneyStage.FUNDED
    assert result.termination is SimulationTermination.ACCOUNT_FAILED
    assert result.total_sessions_elapsed == 2
    assert result.sessions_to_payout is None
    assert result.funded is not None
    assert result.funded.initial_balance_usd == 50_000
    assert result.funded.final_balance_usd == 48_000


def test_journey_never_starts_funded_when_evaluation_account_fails() -> None:
    profile = _profile()
    days = _weekdays(2)
    result = simulate_evaluation_to_funded(
        profile,
        profile.rules_for("evaluation"),
        profile.rules_for("sim_funded"),
        (
            _session(days[0], _trade(days[0], -2_000, trade_id="eval-fail")),
            _session(days[1], _trade(days[1], 3_000, adverse=0, trade_id="unused")),
        ),
        internal_safety_buffer_usd=0,
    )

    assert result.account_failed
    assert not result.evaluation_passed
    assert not result.funded_started
    assert result.funded is None
    assert result.terminal_stage is JourneyStage.EVALUATION
    assert result.failure_stage is JourneyStage.EVALUATION
    assert result.total_sessions_elapsed == 1


def test_funded_payout_requires_strictly_more_than_half_hold_ratios() -> None:
    profile = _profile()
    rules = profile.rules_for("sim_funded")
    days = _weekdays(7)
    sessions = []
    for index, session_date in enumerate(days):
        # After six sessions the account satisfies its monetary/winning-day requirements,
        # but exactly 3/6 trades and 3/6 positive profit are over ten seconds: not enough.
        duration = 11.0 if index in {0, 1, 2, 6} else 10.0
        sessions.append(
            _session(
                session_date,
                _trade(
                    session_date,
                    500,
                    adverse=0,
                    hold_seconds=duration,
                    trade_id=f"day-{index + 1}",
                ),
            )
        )

    first_six = simulate_prop_phase(profile, rules, tuple(sessions[:6]))
    assert first_six.final_state.payout_eligible
    assert first_six.hold_duration.trade_fraction == 0.5
    assert first_six.hold_duration.positive_profit_fraction == 0.5
    assert not first_six.hold_duration.compliant
    assert not first_six.payout_eligible

    result = simulate_prop_phase(profile, rules, tuple(sessions))
    assert result.payout_eligible
    assert result.sessions_to_payout == 7
    assert result.termination is SimulationTermination.PAYOUT_ELIGIBLE
    assert result.hold_duration.qualifying_trades == 4
    assert result.hold_duration.trade_fraction == pytest.approx(4 / 7)
    assert result.hold_duration.positive_profit_fraction == pytest.approx(4 / 7)
    assert result.hold_duration.compliant


@pytest.mark.parametrize(
    "sessions, error",
    [
        ((), "at least one historical session"),
        (
            (
                _session(
                    MONDAY,
                    replace(
                        _trade(MONDAY, 10),
                        opened_at=datetime(2026, 8, 24, 10, 0),
                    ),
                ),
            ),
            "opened_at must be timezone-aware",
        ),
        (
            (
                _session(
                    MONDAY,
                    replace(_trade(MONDAY, -100), minimum_intratrade_pnl_usd=-50),
                ),
            ),
            "no greater than the closing net P&L",
        ),
        (
            (_session(MONDAY, replace(_trade(MONDAY, 1), planned_risk_usd=0)),),
            "planned_risk_usd must be strictly positive",
        ),
    ],
)
def test_historical_input_fails_closed(sessions, error: str) -> None:
    profile = _profile()
    with pytest.raises(ValueError, match=error):
        simulate_prop_phase(profile, profile.rules_for("evaluation"), sessions)


def test_sessions_must_be_ordered_non_overlapping_and_inside_flatten_deadline() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_prop_phase(
            profile,
            rules,
            (
                _session(day2, _trade(day2, 1)),
                _session(day1, _trade(day1, 1)),
            ),
        )

    first = _trade(day1, 1, hold_seconds=120, minute=0)
    overlapping = _trade(day1, 1, hold_seconds=10, minute=1)
    with pytest.raises(ValueError, match="cannot overlap"):
        simulate_prop_phase(profile, rules, (_session(day1, first, overlapping),))

    late = replace(
        _trade(day1, 1),
        opened_at=datetime(2026, 8, 24, 16, 44, tzinfo=ET),
        closed_at=datetime(2026, 8, 24, 16, 46, tzinfo=ET),
    )
    with pytest.raises(ValueError, match="flat by"):
        simulate_prop_phase(profile, rules, (_session(day1, late),))


def test_safety_buffer_cannot_consume_the_entire_fresh_drawdown() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="leave positive fresh-account cushion"):
        simulate_prop_phase(
            profile,
            profile.rules_for("evaluation"),
            (_session(MONDAY, _trade(MONDAY, 1)),),
            internal_safety_buffer_usd=2_000,
        )


def test_bootstrap_is_seeded_and_resamples_whole_session_blocks() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    positive = _session(
        day1,
        _trade(day1, 100, adverse=0, minute=0, trade_id="p1"),
        _trade(day1, 100, adverse=0, minute=2, trade_id="p2"),
    )
    negative = _session(
        day2,
        _trade(day2, -100, minute=0, trade_id="n1"),
        _trade(day2, -100, minute=2, trade_id="n2"),
    )

    first = run_session_bootstrap(
        profile,
        rules,
        (positive, negative),
        simulations=40,
        seed=1234,
        horizon_sessions=2,
    )
    second = run_session_bootstrap(
        profile,
        rules,
        (positive, negative),
        simulations=40,
        seed=1234,
        horizon_sessions=2,
    )

    assert first == second
    # The personal quota admits the first signal and journals the second inside each
    # sampled block.  Both outcomes move together as one session-level unit.
    assert {run.net_profit_usd for run in first.runs} <= {-200.0, 0.0, 200.0}
    assert all(run.executed_trades == 2 for run in first.runs)
    assert all(run.rejected_trade_quota_trades == 2 for run in first.runs)


def test_bootstrap_preserves_zero_signal_session_blocks_deterministically() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    source = (
        _session(day1),
        _session(day2, _trade(day2, 3_000, adverse=0, trade_id="pass")),
    )

    first = run_session_bootstrap(
        profile,
        rules,
        source,
        simulations=60,
        seed=909,
        horizon_sessions=2,
    )
    second = run_session_bootstrap(
        profile,
        rules,
        source,
        simulations=60,
        seed=909,
        horizon_sessions=2,
    )

    assert first == second
    assert any(run.executed_trades == 0 for run in first.runs)
    assert any(run.sessions_to_pass == 2 for run in first.runs)
    assert all(run.trading_days <= run.sessions_elapsed for run in first.runs)


def test_bootstrap_aggregates_pass_failure_drawdown_and_lifetime_metrics() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    day1, day2 = _weekdays(2)
    pass_block = _session(day1, _trade(day1, 3_000, adverse=0, trade_id="pass"))
    fail_block = _session(day2, _trade(day2, -2_000, trade_id="fail"))

    report = run_session_bootstrap(
        profile,
        rules,
        (pass_block, fail_block),
        simulations=200,
        seed=42,
        horizon_sessions=1,
        internal_safety_buffer_usd=0,
    )

    assert 0 < report.pass_rate < 1
    assert 0 < report.account_failure_rate < 1
    assert report.pass_rate + report.account_failure_rate == pytest.approx(1)
    assert report.internal_safety_lock_rate == 0
    assert report.payout_probability == 0
    assert report.median_days_to_pass == 1
    assert report.expected_lifetime_sessions == 1
    assert report.maximum_drawdown_usd == 2_000
    assert report.average_drawdown_usd >= 0
    assert report.expected_profit_per_account_usd == pytest.approx(
        sum(run.net_profit_usd for run in report.runs) / report.simulations
    )


def test_bootstrap_reports_funded_payout_probability() -> None:
    profile = _profile()
    rules = profile.rules_for("sim_funded")
    source = _session(MONDAY, _trade(MONDAY, 500, adverse=0, hold_seconds=11))

    report = run_session_bootstrap(
        profile,
        rules,
        (source,),
        simulations=5,
        seed=8,
        horizon_sessions=7,
    )

    assert report.pass_rate == 0
    assert report.account_failure_rate == 0
    assert report.payout_probability == 1
    assert report.expected_lifetime_sessions == 6


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"simulations": 0, "seed": 1}, "simulations"),
        ({"simulations": 1, "seed": True}, "seed"),
        ({"simulations": 1, "seed": 1, "horizon_sessions": 0}, "horizon_sessions"),
        (
            {"simulations": 1, "seed": 1, "internal_safety_buffer_usd": float("nan")},
            "internal_safety_buffer_usd",
        ),
    ],
)
def test_bootstrap_parameters_fail_closed(kwargs, error: str) -> None:
    profile = _profile()
    source = _session(MONDAY, _trade(MONDAY, 1, adverse=0))
    with pytest.raises(ValueError, match=error):
        run_session_bootstrap(
            profile,
            profile.rules_for("evaluation"),
            (source,),
            **kwargs,
        )


def test_bootstrap_rejects_empty_or_all_zero_trade_sources() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    with pytest.raises(ValueError, match="at least one historical session"):
        run_session_bootstrap(profile, rules, (), simulations=1, seed=1)
    mixed = run_session_bootstrap(
        profile,
        rules,
        (_session(MONDAY), _session(_weekdays(2)[1], _trade(_weekdays(2)[1], 1))),
        simulations=10,
        seed=1,
        horizon_sessions=2,
    )
    assert mixed.simulations == 10

    with pytest.raises(ValueError, match="must contain at least one trade"):
        run_session_bootstrap(
            profile,
            rules,
            (_session(MONDAY), _session(_weekdays(2)[1])),
            simulations=1,
            seed=1,
        )


def test_journey_bootstrap_is_reproducible_preserves_zero_blocks_and_fresh_accounts() -> None:
    profile = _profile()
    evaluation_rules = profile.rules_for("evaluation")
    funded_rules = profile.rules_for("sim_funded")
    day1, day2 = _weekdays(2)
    source = (
        _session(day1),
        _session(
            day2,
            _trade(day2, 3_000, adverse=0, planned_risk=150, trade_id="winner"),
        ),
    )

    first = run_journey_session_bootstrap(
        profile,
        evaluation_rules,
        funded_rules,
        source,
        simulations=200,
        seed=5150,
        horizon_sessions=8,
    )
    second = run_journey_session_bootstrap(
        profile,
        evaluation_rules,
        funded_rules,
        source,
        simulations=200,
        seed=5150,
        horizon_sessions=8,
    )

    assert first == second
    assert len(first.sampled_source_indices) == 200
    assert all(len(path) == 8 for path in first.sampled_source_indices)
    assert all(set(path) <= {0, 1} for path in first.sampled_source_indices)
    assert any(0 in path for path in first.sampled_source_indices)
    assert 0 < first.evaluation_pass_rate < 1
    assert 0 < first.payout_probability < first.evaluation_pass_rate
    assert first.account_failure_rate == 0
    assert first.internal_safety_lock_rate == 0
    assert first.median_days_to_pass is not None
    assert first.median_days_to_payout is not None
    assert first.maximum_drawdown_usd >= first.average_drawdown_usd >= 0
    assert first.expected_lifetime_sessions > 0
    assert first.expected_profit_per_journey_usd >= 0
    assert first.limitations
    assert all(
        run.funded is None or run.funded.initial_balance_usd == 50_000
        for run in first.runs
    )


def test_journey_bootstrap_reports_failure_stage_rates_from_runs() -> None:
    profile = _profile()
    days = _weekdays(2)
    source = (
        _session(days[0], _trade(days[0], 3_000, adverse=0, trade_id="pass")),
        _session(days[1], _trade(days[1], -2_000, trade_id="failure")),
    )

    report = run_journey_session_bootstrap(
        profile,
        profile.rules_for("evaluation"),
        profile.rules_for("sim_funded"),
        source,
        simulations=120,
        seed=77,
        horizon_sessions=3,
        internal_safety_buffer_usd=0,
    )

    assert report.account_failure_rate == pytest.approx(
        sum(run.account_failed for run in report.runs) / report.simulations
    )
    assert report.evaluation_account_failure_rate == pytest.approx(
        sum(run.evaluation.account_failed for run in report.runs) / report.simulations
    )
    assert report.funded_account_failure_rate == pytest.approx(
        sum(run.funded is not None and run.funded.account_failed for run in report.runs)
        / report.simulations
    )
    assert report.account_failure_rate > 0


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"simulations": 0, "seed": 1}, "simulations"),
        ({"simulations": 1, "seed": True}, "seed"),
        ({"simulations": 1, "seed": 1, "horizon_sessions": 0}, "horizon_sessions"),
    ],
)
def test_journey_bootstrap_parameters_fail_closed(kwargs, error: str) -> None:
    profile = _profile()
    source = (_session(MONDAY, _trade(MONDAY, 1, adverse=0)),)
    with pytest.raises(ValueError, match=error):
        run_journey_session_bootstrap(
            profile,
            profile.rules_for("evaluation"),
            profile.rules_for("sim_funded"),
            source,
            **kwargs,
        )


def test_journey_bootstrap_rejects_an_all_zero_trade_source() -> None:
    profile = _profile()
    day1, day2 = _weekdays(2)
    with pytest.raises(ValueError, match="must contain at least one trade"):
        run_journey_session_bootstrap(
            profile,
            profile.rules_for("evaluation"),
            profile.rules_for("sim_funded"),
            (_session(day1), _session(day2)),
            simulations=1,
            seed=1,
        )


def test_explicit_trade_adjustment_is_immutable_and_fully_audited() -> None:
    day1, day2 = _weekdays(2)
    source_trade = _trade(
        day2,
        300,
        adverse=-50,
        planned_risk=150,
        commission=2,
        fees=1,
        slippage=3,
        trade_id="source",
    )
    source = (_session(day1), _session(day2, source_trade))
    adjustment = TradeOutcomeAdjustment(
        additional_commission_usd_per_trade=4,
        additional_fees_usd_per_trade=2,
        additional_slippage_usd_per_trade=5,
    )

    transformed, audit = apply_trade_outcome_adjustment(source, adjustment)
    changed = transformed[1].trades[0]

    assert source[1].trades[0] == source_trade
    assert transformed[0].trades == ()
    assert len(transformed) == len(source)
    assert changed.net_pnl_usd == 289
    assert changed.minimum_intratrade_pnl_usd == -61
    assert changed.planned_risk_usd == 161
    assert changed.commission_usd == 6
    assert changed.fees_usd == 3
    assert changed.slippage_usd == 8
    assert transformed[1].market_day_status is source[1].market_day_status
    assert len(audit) == 1
    assert audit[0].original_net_pnl_usd == 300
    assert audit[0].adjusted_net_pnl_usd == 289
    assert audit[0].added_slippage_usd == 5


def test_scenario_comparison_is_paired_named_and_never_profit_ranked() -> None:
    profile = _profile()
    source_trade = _trade(
        MONDAY,
        3_000,
        adverse=0,
        planned_risk=150,
        trade_id="evaluation-target",
    )
    source = (_session(MONDAY, source_trade),)
    stress = JourneyScenario(
        name="higher execution costs",
        description="Add explicit commission, fee, and slippage dollars per source signal.",
        outcome_adjustment=TradeOutcomeAdjustment(
            additional_commission_usd_per_trade=10,
            additional_fees_usd_per_trade=5,
            additional_slippage_usd_per_trade=10,
        ),
    )
    tighter_policy = JourneyScenario(
        name="tighter personal risk",
        description="Reduce maximum planned risk from $200 to $100.",
        policy=PropSimulationPolicy(max_planned_risk_per_trade_usd=100),
        internal_safety_buffer_usd=500,
    )

    comparison = compare_journey_scenarios(
        profile,
        profile.rules_for("evaluation"),
        profile.rules_for("sim_funded"),
        source,
        scenarios=(stress, tighter_policy),
        simulations=10,
        seed=123,
        horizon_sessions=1,
    )

    assert source[0].trades[0] == source_trade
    assert comparison.baseline.report.evaluation_pass_rate == 1
    assert comparison.baseline.report.expected_profit_per_journey_usd == 3_000
    assert [scenario.name for scenario in comparison.scenarios] == [
        "higher execution costs",
        "tighter personal risk",
    ]
    stressed, tightened = comparison.scenarios
    assert stressed.report.evaluation_pass_rate == 0
    assert stressed.report.expected_profit_per_journey_usd == 2_975
    assert len(stressed.trade_adjustments) == 1
    assert stressed.trade_adjustments[0].adjusted_planned_risk_usd == 175
    assert all(run.evaluation.total_commission_usd == 10 for run in stressed.report.runs)
    assert all(run.evaluation.total_fees_usd == 5 for run in stressed.report.runs)
    assert all(run.evaluation.total_slippage_usd == 10 for run in stressed.report.runs)
    assert tightened.report.evaluation_pass_rate == 0
    assert tightened.report.expected_profit_per_journey_usd == 0
    assert {change.parameter for change in tightened.parameter_changes} == {
        "policy.max_planned_risk_per_trade_usd",
        "internal_safety_buffer_usd",
    }
    baseline_paths = comparison.baseline.report.sampled_source_indices
    assert all(item.report.sampled_source_indices == baseline_paths for item in comparison.scenarios)
    assert comparison.limitations
    assert any("unsupported" in limitation for limitation in comparison.limitations)
    assert not hasattr(comparison, "ranking")


def test_scenario_inputs_fail_closed_on_negative_adjustment_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        TradeOutcomeAdjustment(additional_slippage_usd_per_trade=-0.01).validate()

    profile = _profile()
    source = (_session(MONDAY, _trade(MONDAY, 3_000, adverse=0, planned_risk=150)),)
    duplicate = JourneyScenario(name="stress", description="first")
    with pytest.raises(ValueError, match="duplicate scenario name"):
        compare_journey_scenarios(
            profile,
            profile.rules_for("evaluation"),
            profile.rules_for("sim_funded"),
            source,
            scenarios=(duplicate, replace(duplicate, name="STRESS")),
            simulations=1,
            seed=1,
        )
