from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tradebot.prop_firms import BreachKind, DrawdownMethod, load_prop_profile
from tradebot.risk.prop import (
    DeploymentStage,
    MarketDayStatus,
    PayoutBlockReason,
    PropGateReason,
    PropOrderAction,
    PropPreTradeRequest,
    close_prop_session,
    evaluate_prop_order,
    initial_prop_account_state,
    reconcile_prop_account,
    record_prop_payout,
    start_prop_session,
)


ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
CHECKED = date(2026, 8, 20)
NOW = datetime(2026, 8, 20, 10, 0, tzinfo=ET)


def _profile(name: str = "growth", *, ambiguity: tuple[str, ...] = ()):
    profile = load_prop_profile(ROOT / "config" / "prop_firms" / f"tradeify_{name}_50k.yaml")
    sources = tuple(replace(source, checked_on=CHECKED) for source in profile.sources)
    return replace(
        profile,
        verified_on=CHECKED,
        reverify_after_hours=72,
        sources=sources,
        ambiguity_notes=ambiguity,
    )


def _entry(
    *,
    now: datetime = NOW,
    micros: int = 1,
    minis: int = 0,
    loss: float = 200.0,
    status: MarketDayStatus | None = MarketDayStatus.REGULAR,
    stage: DeploymentStage = DeploymentStage.PAPER,
    authorized: bool = False,
    averaging_down: bool = False,
) -> PropPreTradeRequest:
    return PropPreTradeRequest(
        action=PropOrderAction.ENTRY,
        now=now,
        rule_verification_as_of=NOW,
        requested_minis=minis,
        requested_micros=micros,
        worst_case_loss_usd=loss,
        deployment_stage=stage,
        stage_authorized=authorized,
        market_day_status=status,
        averaging_down=averaging_down,
    )


def _decision(request, profile, rules, state):
    return evaluate_prop_order(request, profile=profile, rules=rules, state=state)


def _run_day(
    state,
    rules,
    profit: float,
    *,
    consistency_profit: float | None = None,
    restart: bool = True,
):
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=state.current_balance_usd + profit,
        daily_realized_pnl_usd=profit,
        daily_consistency_profit_usd=consistency_profit,
        trades_this_session=1,
    )
    state = close_prop_session(state, rules)
    return start_prop_session(state, rules) if restart else state


def test_fresh_state_exposes_firm_floor_and_internal_buffer() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")

    state = initial_prop_account_state(profile, rules, internal_safety_buffer_usd=400)

    assert state.starting_balance_usd == 50_000
    assert state.current_balance_usd == 50_000
    assert state.real_time_net_liquidation_usd == 50_000
    assert state.highest_end_of_day_balance_usd == 50_000
    assert state.drawdown_floor_usd == 48_000
    assert state.remaining_prop_drawdown_usd == 2_000
    assert state.internal_safety_threshold_usd == 48_400
    assert state.remaining_internal_cushion_usd == 1_600
    assert state.firm_daily_loss_limit_usd == 1_250
    assert state.remaining_firm_daily_risk_usd == 1_250
    assert not state.soft_daily_loss_locked
    assert not state.hard_breached

    payload = state.to_dict()
    assert payload["phase"] == "evaluation"
    assert payload["daily_pnl_usd"] == 0
    assert payload["distance_to_prop_failure_usd"] == 2_000
    assert payload["has_open_exposure"] is False


def test_end_of_day_trailing_floor_moves_only_at_eod_and_equality_is_breach() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=51_000,
        daily_realized_pnl_usd=1_000,
        trades_this_session=1,
    )
    assert state.drawdown_floor_usd == 48_000

    state = close_prop_session(state, rules)
    assert state.highest_end_of_day_balance_usd == 51_000
    assert state.drawdown_floor_usd == 49_000

    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=51_000,
        daily_realized_pnl_usd=1_000,
        daily_unrealized_pnl_usd=-2_000,
    )
    assert state.real_time_net_liquidation_usd == 49_000
    assert state.distance_to_prop_failure_usd == 0
    assert state.remaining_prop_drawdown_usd == 0
    assert state.hard_breached
    assert "trailing_drawdown" in state.hard_breach_reasons


def test_intraday_trailing_rule_moves_on_mark_to_market() -> None:
    profile = _profile()
    base = profile.rules_for("evaluation")
    rules = replace(
        base,
        drawdown=replace(
            base.drawdown,
            method=DrawdownMethod.INTRADAY_TRAILING,
            high_water_mark_basis="intraday_net_liquidation",
        ),
    )
    profile = replace(profile, phases=(rules, profile.rules_for("sim_funded")))
    state = initial_prop_account_state(profile, rules)

    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=50_000,
        daily_realized_pnl_usd=0,
        daily_unrealized_pnl_usd=1_000,
    )

    assert state.drawdown_high_water_mark_usd == 51_000
    assert state.drawdown_floor_usd == 49_000


def test_sim_funded_floor_locks_at_exact_trigger_and_dll_escalates_at_eod() -> None:
    profile = _profile()
    rules = profile.rules_for("sim_funded")
    state = initial_prop_account_state(profile, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=52_100,
        daily_realized_pnl_usd=2_100,
        trades_this_session=1,
    )
    state = close_prop_session(state, rules)

    assert state.drawdown_locked
    assert state.drawdown_floor_usd == 50_100
    assert state.internal_safety_threshold_usd == 50_500

    state = start_prop_session(state, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=53_000,
        daily_realized_pnl_usd=900,
        trades_this_session=1,
    )
    state = close_prop_session(state, rules)

    assert state.drawdown_floor_usd == 50_100
    assert state.highest_end_of_day_balance_usd == 53_000
    assert state.firm_daily_loss_limit_usd == 2_000


def test_soft_daily_loss_locks_at_equality_and_does_not_unlock_on_recovery() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=48_750,
        daily_realized_pnl_usd=-1_250,
    )

    assert state.soft_daily_loss_locked
    assert not state.hard_breached
    assert state.remaining_firm_daily_risk_usd == 0
    decision = _decision(_entry(), profile, rules, state)
    assert PropGateReason.SOFT_DAILY_LOSS_LOCK in decision.reason_codes
    assert PropGateReason.DAILY_LOSS_LIMIT in decision.reason_codes

    recovered = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=49_000,
        daily_realized_pnl_usd=-1_000,
    )
    assert recovered.soft_daily_loss_locked
    assert recovered.remaining_firm_daily_risk_usd == 250

    next_session = start_prop_session(recovered, rules)
    assert not next_session.soft_daily_loss_locked
    assert next_session.remaining_firm_daily_risk_usd == 1_250


def test_hard_daily_loss_is_latched_for_hard_rule() -> None:
    profile = _profile()
    base = profile.rules_for("evaluation")
    rules = replace(base, daily_loss_breach_kind=BreachKind.HARD)
    profile = replace(profile, phases=(rules, profile.rules_for("sim_funded")))
    state = initial_prop_account_state(profile, rules)

    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=48_750,
        daily_realized_pnl_usd=-1_250,
    )
    assert state.hard_breached
    assert "daily_loss_limit" in state.hard_breach_reasons
    assert start_prop_session(state, rules).hard_breached


def test_post_stop_equity_must_remain_strictly_above_internal_threshold() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    at_boundary = _decision(_entry(loss=1_600), profile, rules, state)
    one_cent_clear = _decision(_entry(loss=1_599.99), profile, rules, state)

    assert not at_boundary.allowed
    assert PropGateReason.INTERNAL_SAFETY_THRESHOLD in at_boundary.reason_codes
    assert one_cent_clear.allowed
    assert at_boundary.to_dict()["reason_codes"] == ["internal_safety_threshold"]


def test_contract_limit_accepts_boundary_and_rejects_one_micro_over() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    assert _decision(_entry(micros=40), profile, rules, state).allowed
    too_many = _decision(_entry(micros=41), profile, rules, state)
    mixed_over = _decision(_entry(minis=4, micros=1), profile, rules, state)

    assert PropGateReason.CONTRACT_LIMIT in too_many.reason_codes
    assert PropGateReason.CONTRACT_LIMIT in mixed_over.reason_codes


def test_select_contract_scaling_is_sticky_on_highest_eod_balance() -> None:
    profile = _profile("select")
    rules = profile.rules_for("sim_funded_flex")
    state = initial_prop_account_state(profile, rules)

    assert state.allowed_minis == 2
    assert state.allowed_micros == 20
    assert _decision(_entry(micros=20), profile, rules, state).allowed
    assert PropGateReason.CONTRACT_LIMIT in _decision(
        _entry(micros=21), profile, rules, state
    ).reason_codes

    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=51_500,
        daily_realized_pnl_usd=1_500,
        trades_this_session=1,
    )
    state = close_prop_session(state, rules)
    state = start_prop_session(state, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=50_900,
        daily_realized_pnl_usd=-600,
    )

    assert state.highest_end_of_day_balance_usd == 51_500
    assert state.allowed_minis == 3
    assert state.allowed_micros == 30


def test_existing_exposure_and_averaging_are_independent_rejections() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=50_000,
        daily_realized_pnl_usd=0,
        open_micros=1,
    )

    decision = _decision(_entry(averaging_down=True), profile, rules, state)

    assert not decision.allowed
    assert PropGateReason.EXISTING_EXPOSURE in decision.reason_codes
    assert PropGateReason.AVERAGING_DOWN in decision.reason_codes


def test_regular_and_early_close_deadlines_are_exclusive_boundaries() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    assert _decision(
        _entry(now=datetime(2026, 8, 20, 16, 44, 59, tzinfo=ET)),
        profile,
        rules,
        state,
    ).allowed
    regular_deadline = _decision(
        _entry(now=datetime(2026, 8, 20, 16, 45, tzinfo=ET)),
        profile,
        rules,
        state,
    )
    assert PropGateReason.OUTSIDE_PERMITTED_HOURS in regular_deadline.reason_codes

    assert _decision(
        _entry(
            now=datetime(2026, 8, 20, 12, 58, 59, tzinfo=ET),
            status=MarketDayStatus.EARLY_CLOSE,
        ),
        profile,
        rules,
        state,
    ).allowed
    holiday_deadline = _decision(
        _entry(
            now=datetime(2026, 8, 20, 12, 59, tzinfo=ET),
            status=MarketDayStatus.EARLY_CLOSE,
        ),
        profile,
        rules,
        state,
    )
    assert PropGateReason.OUTSIDE_PERMITTED_HOURS in holiday_deadline.reason_codes


def test_overnight_window_allows_evening_leg_but_not_daily_halt_or_weekend() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    evening = _decision(
        _entry(now=datetime(2026, 8, 20, 18, 0, tzinfo=ET)), profile, rules, state
    )
    halt = _decision(
        _entry(now=datetime(2026, 8, 20, 17, 30, tzinfo=ET)), profile, rules, state
    )
    weekend = _decision(
        _entry(now=datetime(2026, 8, 22, 10, 0, tzinfo=ET)), profile, rules, state
    )

    assert evening.allowed
    assert PropGateReason.OUTSIDE_PERMITTED_HOURS in halt.reason_codes
    assert PropGateReason.MARKET_CLOSED in weekend.reason_codes


def test_unknown_holiday_status_and_missing_holiday_schedule_fail_closed() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    unknown = _decision(_entry(status=None), profile, rules, state)
    assert PropGateReason.HOLIDAY_STATUS_UNKNOWN in unknown.reason_codes

    no_holiday_window = replace(
        profile,
        trading_window=replace(profile.trading_window, holiday_flatten_by=None),
    )
    missing_schedule = _decision(
        _entry(status=MarketDayStatus.EARLY_CLOSE), no_holiday_window, rules, state
    )
    assert PropGateReason.HOLIDAY_SCHEDULE_UNKNOWN in missing_schedule.reason_codes


def test_missing_and_stale_profiles_fail_closed() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    missing = _decision(_entry(), None, rules, state)
    stale_profile = replace(profile, reverify_after_hours=1)
    stale = _decision(_entry(), stale_profile, rules, state)

    assert PropGateReason.PROFILE_MISSING in missing.reason_codes
    assert PropGateReason.PROFILE_STALE in stale.reason_codes


def test_stage_three_needs_approval_and_rejects_any_unresolved_ambiguity() -> None:
    clean = _profile()
    rules = clean.rules_for("evaluation")
    state = initial_prop_account_state(clean, rules)

    unapproved = _decision(
        _entry(stage=DeploymentStage.PROP_EVALUATION), clean, rules, state
    )
    assert PropGateReason.STAGE_APPROVAL_REQUIRED in unapproved.reason_codes

    ambiguous = replace(clean, ambiguity_notes=("DLL basis awaits written confirmation",))
    ambiguous_state = replace(state, profile_id=ambiguous.profile_id)
    rejected = _decision(
        _entry(stage=DeploymentStage.PROP_EVALUATION, authorized=True),
        ambiguous,
        rules,
        ambiguous_state,
    )
    assert PropGateReason.AMBIGUOUS_RULES in rejected.reason_codes

    # Ambiguous rules may still be modelled in paper research; they cannot be promoted.
    assert _decision(_entry(), ambiguous, rules, ambiguous_state).allowed


def test_stage_and_account_phase_must_match() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    decision = _decision(
        _entry(stage=DeploymentStage.FUNDED, authorized=True), profile, rules, state
    )

    assert PropGateReason.STAGE_PHASE_MISMATCH in decision.reason_codes


def test_exits_are_always_allowed_even_with_no_profile_state_or_valid_timestamp() -> None:
    exit_request = PropPreTradeRequest(
        action=PropOrderAction.EXIT,
        now=datetime(2026, 8, 20, 12, 0),
        requested_micros=-99,
        worst_case_loss_usd=float("nan"),
    )

    decision = evaluate_prop_order(exit_request, profile=None, rules=None, state=None)

    assert decision.allowed
    assert decision.reason_codes == ()


def test_evaluation_pass_requires_target_days_and_consistency_at_boundary() -> None:
    profile = _profile("select")
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = _run_day(state, rules, 1_200)
    state = _run_day(state, rules, 900)
    state = _run_day(state, rules, 900, restart=False)

    assert state.current_balance_usd == 53_000
    assert state.trading_days == 3
    assert state.evaluation_consistency_ratio == pytest.approx(0.40)
    assert state.evaluation_consistency_met
    assert state.profit_target_remaining_usd == 0
    assert state.evaluation_passed


def test_consistency_can_use_gross_profit_when_commissions_are_excluded() -> None:
    profile = _profile("select")
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = _run_day(state, rules, 1_195, consistency_profit=1_200)
    state = _run_day(state, rules, 902.50, consistency_profit=907.50)
    state = _run_day(
        state,
        rules,
        902.50,
        consistency_profit=907.50,
        restart=False,
    )

    assert state.current_balance_usd == 53_000
    assert state.cumulative_consistency_profit_usd == 3_015
    assert state.evaluation_consistency_ratio == pytest.approx(1_200 / 3_015)
    assert state.evaluation_passed


def test_growth_payout_fields_enforce_strict_winning_day_and_exact_boundaries() -> None:
    profile = _profile()
    rules = profile.rules_for("sim_funded")
    state = initial_prop_account_state(profile, rules)

    boundary = _run_day(state, rules, 150)
    assert boundary.payout_winning_days == 0  # Growth says strictly greater than $150.

    state = initial_prop_account_state(profile, rules)
    for index in range(5):
        state = _run_day(state, rules, 600, restart=index < 4)

    assert state.current_balance_usd == 53_000
    assert state.payout_winning_days == 5
    assert state.payout_consistency_ratio == pytest.approx(0.20)
    assert state.payout_consistency_met
    assert state.payout_cap_usd == 1_500
    assert state.payout_available_usd == 1_500
    assert state.payout_block_reasons == ()
    assert state.payout_eligible


def test_lightning_payout_consistency_accepts_exact_twenty_percent() -> None:
    profile = _profile("lightning")
    rules = profile.rules_for("sim_funded")
    state = initial_prop_account_state(profile, rules)
    for index in range(5):
        state = _run_day(state, rules, 600, restart=index < 4)

    assert state.payout_cycle_best_day_profit_usd == 600
    assert state.payout_consistency_ratio == pytest.approx(0.20)
    assert state.payout_consistency_met
    assert state.payout_eligible


def test_record_payout_checks_amount_and_resets_cycle_without_resetting_hard_floor() -> None:
    profile = _profile()
    rules = profile.rules_for("sim_funded")
    state = initial_prop_account_state(profile, rules)
    for index in range(5):
        state = _run_day(state, rules, 600, restart=index < 4)

    with pytest.raises(ValueError, match="below the configured minimum"):
        record_prop_payout(state, rules, amount_usd=499)

    paid = record_prop_payout(state, rules, amount_usd=1_000)
    assert paid.current_balance_usd == 52_000
    assert paid.payout_cycle_number == 2
    assert paid.payout_cycle_start_balance_usd == 52_000
    assert paid.payout_winning_days == 0
    assert paid.drawdown_locked
    assert paid.drawdown_floor_usd == 50_100
    assert not paid.payout_eligible
    assert PayoutBlockReason.WINNING_DAYS in paid.payout_block_reasons


def test_select_flex_payout_fraction_uses_total_profit_not_only_new_cycle_profit() -> None:
    profile = _profile("select")
    rules = profile.rules_for("sim_funded_flex")
    state = initial_prop_account_state(profile, rules)
    for index in range(5):
        state = _run_day(state, rules, 600, restart=index < 4)
    state = record_prop_payout(state, rules, amount_usd=1_500)
    state = start_prop_session(state, rules)
    for index in range(5):
        state = _run_day(state, rules, 150, restart=index < 4)

    assert state.current_balance_usd == 52_250
    assert state.current_balance_usd - state.payout_cycle_start_balance_usd == 750
    assert state.payout_available_usd == pytest.approx(1_125)
    assert state.payout_eligible


def test_select_daily_payout_must_leave_the_required_buffer_in_account() -> None:
    profile = _profile("select")
    rules = profile.rules_for("sim_funded_daily")
    state = initial_prop_account_state(profile, rules)
    state = _run_day(state, rules, 2_200, restart=False)

    assert state.payout_available_usd == 100
    assert not state.payout_eligible
    assert PayoutBlockReason.MINIMUM_PAYOUT in state.payout_block_reasons

    state = start_prop_session(state, rules)
    state = _run_day(state, rules, 200, restart=False)
    assert state.payout_available_usd == 300
    assert state.payout_eligible
    with pytest.raises(ValueError, match="currently available"):
        record_prop_payout(state, rules, amount_usd=300.01)

    paid = record_prop_payout(state, rules, amount_usd=300)
    assert paid.current_balance_usd == 52_100


def test_no_daily_limit_is_explicitly_unbounded_not_zero() -> None:
    profile = _profile("select")
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)

    assert state.firm_daily_loss_limit_usd is None
    assert state.remaining_firm_daily_risk_usd is None


def test_session_roll_refuses_overnight_exposure() -> None:
    profile = _profile()
    rules = profile.rules_for("evaluation")
    state = initial_prop_account_state(profile, rules)
    state = reconcile_prop_account(
        state,
        rules,
        current_balance_usd=50_000,
        daily_realized_pnl_usd=0,
        open_micros=1,
    )

    with pytest.raises(ValueError, match="open exposure"):
        close_prop_session(state, rules)
    with pytest.raises(ValueError, match="open exposure"):
        start_prop_session(state, rules)
