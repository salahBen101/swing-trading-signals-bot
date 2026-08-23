"""Deterministic, session-aware historical prop-account simulation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
import math
from typing import Sequence
from zoneinfo import ZoneInfo

from tradebot.prop_firms import AccountPhase, AccountRuleSet, PropFirmProfile
from tradebot.risk.prop import (
    MarketDayStatus,
    PropAccountState,
    close_prop_session,
    initial_prop_account_state,
    reconcile_prop_account,
    start_prop_session,
)

from .models import (
    HistoricalPropSession,
    HistoricalPropTrade,
    HoldDurationCompliance,
    JourneyStage,
    PropJourneyResult,
    PropSimulationResult,
    PropSimulationPolicy,
    SimulationEvent,
    SimulationEventKind,
    SimulationTermination,
)


_EPSILON = 1e-9


def _as_finite_non_negative(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _session_date_for(moment: datetime, profile: PropFirmProfile) -> date:
    """Return the ending calendar date of the profile session containing ``moment``."""

    zone = ZoneInfo(profile.trading_window.timezone)
    local = moment.astimezone(zone)
    local_clock = local.time().replace(tzinfo=None)
    start = time.fromisoformat(profile.trading_window.session_start)
    end = time.fromisoformat(profile.trading_window.session_end)
    if start > end:
        if local_clock >= start:
            return local.date() + timedelta(days=1)
        if local_clock < end:
            return local.date()
        raise ValueError(f"trade timestamp {moment.isoformat()} falls in the daily session halt")
    if start <= local_clock < end:
        return local.date()
    raise ValueError(f"trade timestamp {moment.isoformat()} is outside the configured session")


def _validate_sessions(
    sessions: Sequence[HistoricalPropSession], profile: PropFirmProfile
) -> tuple[HistoricalPropSession, ...]:
    if isinstance(sessions, (str, bytes)):
        raise ValueError("sessions must be a sequence of HistoricalPropSession objects")
    materialized = tuple(sessions)
    if not materialized:
        raise ValueError("at least one historical session is required")

    previous_date: date | None = None
    seen_trade_ids: set[str] = set()
    zone = ZoneInfo(profile.trading_window.timezone)
    for session in materialized:
        if not isinstance(session, HistoricalPropSession):
            raise ValueError("every input item must be a HistoricalPropSession")
        session.validate()
        if session.session_date.weekday() >= 5:
            raise ValueError("session_date must be a Monday-through-Friday trading date")
        if previous_date is not None and session.session_date <= previous_date:
            raise ValueError("historical sessions must have unique, strictly increasing dates")
        previous_date = session.session_date
        if session.market_day_status is MarketDayStatus.EARLY_CLOSE:
            holiday_deadline = profile.trading_window.holiday_flatten_by
            if holiday_deadline is None:
                raise ValueError("early-close session has no configured holiday flatten deadline")
            deadline_clock = time.fromisoformat(holiday_deadline)
        else:
            deadline_clock = time.fromisoformat(profile.trading_window.flatten_by)
        deadline = datetime.combine(session.session_date, deadline_clock, tzinfo=zone)
        previous_close: datetime | None = None
        for trade in session.trades:
            opened_session = _session_date_for(trade.opened_at, profile)
            closed_session = _session_date_for(trade.closed_at, profile)
            if opened_session != session.session_date or closed_session != session.session_date:
                raise ValueError("a trade cannot cross or disagree with its Tradeify session date")
            opened_local = trade.opened_at.astimezone(zone)
            closed_local = trade.closed_at.astimezone(zone)
            if opened_local >= deadline or closed_local > deadline:
                raise ValueError("historical trades must be flat by the configured deadline")
            if previous_close is not None and trade.opened_at < previous_close:
                raise ValueError("trades within a session cannot overlap or be out of order")
            previous_close = trade.closed_at
            if trade.trade_id:
                if trade.trade_id in seen_trade_ids:
                    raise ValueError(f"duplicate trade_id {trade.trade_id!r}")
                seen_trade_ids.add(trade.trade_id)
    return materialized


def _contract_allowed(
    trade: HistoricalPropTrade,
    state: PropAccountState,
    rules: AccountRuleSet,
) -> bool:
    if trade.minis > state.allowed_minis or trade.micros > state.allowed_micros:
        return False
    limits = rules.contracts
    if trade.minis and trade.micros and not limits.mixed_sizes_allowed:
        return False
    if limits.aggregate_gross_open_position:
        return (
            trade.minis * state.micro_per_mini + trade.micros
            <= state.allowed_micros
        )
    return True


def _hold_duration_compliance(
    trades: Sequence[HistoricalPropTrade], profile: PropFirmProfile
) -> HoldDurationCompliance:
    compliance = profile.compliance
    threshold = compliance.minimum_hold_seconds_for_payout
    trade_requirement = compliance.min_fraction_trades_over_hold
    profit_requirement = compliance.min_fraction_profit_over_hold

    if threshold is None:
        qualifying: tuple[HistoricalPropTrade, ...] = ()
    else:
        # The official rule says *longer than* the threshold; equality does not count.
        qualifying = tuple(
            trade for trade in trades if trade.hold_seconds > threshold + _EPSILON
        )
    total = len(trades)
    trade_fraction = len(qualifying) / total if total else None
    total_positive = sum(max(0.0, trade.net_pnl_usd) for trade in trades)
    qualifying_positive = sum(max(0.0, trade.net_pnl_usd) for trade in qualifying)
    profit_fraction = (
        qualifying_positive / total_positive if total_positive > _EPSILON else None
    )

    requirements_exist = trade_requirement is not None or profit_requirement is not None
    if requirements_exist and threshold is None:
        is_compliant = False
    else:
        trade_met = (
            True
            if trade_requirement is None
            else trade_fraction is not None
            and trade_fraction > trade_requirement + _EPSILON
        )
        profit_met = (
            True
            if profit_requirement is None
            else profit_fraction is not None
            and profit_fraction > profit_requirement + _EPSILON
        )
        is_compliant = trade_met and profit_met

    return HoldDurationCompliance(
        threshold_seconds=threshold,
        total_trades=total,
        qualifying_trades=len(qualifying),
        trade_fraction=trade_fraction,
        total_positive_profit_usd=total_positive,
        qualifying_positive_profit_usd=qualifying_positive,
        positive_profit_fraction=profit_fraction,
        trade_fraction_required=trade_requirement,
        positive_profit_fraction_required=profit_requirement,
        compliant=is_compliant,
    )


def simulate_prop_phase(
    profile: PropFirmProfile,
    rules: AccountRuleSet,
    sessions: Sequence[HistoricalPropSession],
    *,
    internal_safety_buffer_usd: float = 400.0,
    policy: PropSimulationPolicy = PropSimulationPolicy(),
) -> PropSimulationResult:
    """Run one profile phase from its fresh starting balance.

    The first terminal condition wins.  Evaluation runs stop when the target, minimum-day,
    and consistency requirements pass at EOD.  Funded runs stop at the first EOD where
    both the profile payout state and the independent hold-duration test are eligible.
    Hard firm breach always takes precedence over the internal safety lock.
    """

    if not isinstance(profile, PropFirmProfile):
        raise ValueError("profile must be a PropFirmProfile")
    if not isinstance(rules, AccountRuleSet):
        raise ValueError("rules must be an AccountRuleSet")
    if not isinstance(policy, PropSimulationPolicy):
        raise ValueError("policy must be a PropSimulationPolicy")
    policy.validate()
    profile.validate()
    if rules not in profile.phases:
        raise ValueError("rules do not belong to the supplied profile")
    buffer = _as_finite_non_negative(internal_safety_buffer_usd, "internal safety buffer")
    validated_sessions = _validate_sessions(sessions, profile)

    state = initial_prop_account_state(
        profile, rules, internal_safety_buffer_usd=buffer
    )
    if state.remaining_internal_cushion_usd <= _EPSILON:
        raise ValueError("internal safety buffer must leave positive fresh-account cushion")

    initial_balance = state.current_balance_usd
    peak_equity = state.real_time_net_liquidation_usd
    maximum_drawdown = 0.0
    minimum_distance = state.distance_to_prop_failure_usd
    sessions_elapsed = 0
    executed = 0
    rejected_daily = 0
    rejected_contract = 0
    rejected_personal_risk = 0
    rejected_quota = 0
    rejected_personal_daily = 0
    rejected_post_loss_increase = 0
    rejected_pretrade_safety = 0
    rejected_market_day = 0
    commission = 0.0
    fees = 0.0
    slippage = 0.0
    current_losing_streak = 0
    maximum_losing_streak = 0
    previous_executed_was_loss = False
    previous_executed_planned_risk: float | None = None
    executed_trade_records: list[HistoricalPropTrade] = []
    events: list[SimulationEvent] = []
    termination = SimulationTermination.HORIZON_EXHAUSTED
    account_failed = False
    internal_locked = False
    evaluation_passed = False
    sessions_to_pass: int | None = None
    payout_eligible = False
    sessions_to_payout: int | None = None

    def observe_equity(observed_state: PropAccountState) -> None:
        nonlocal peak_equity, maximum_drawdown, minimum_distance
        equity = observed_state.real_time_net_liquidation_usd
        peak_equity = max(peak_equity, equity)
        maximum_drawdown = max(maximum_drawdown, peak_equity - equity)
        minimum_distance = min(
            minimum_distance, equity - observed_state.drawdown_floor_usd
        )

    def event(
        kind: SimulationEventKind,
        session_date: date,
        detail: str,
        trade_id: str | None = None,
    ) -> None:
        events.append(
            SimulationEvent(
                kind=kind,
                session_date=session_date,
                trade_id=trade_id,
                balance_usd=state.current_balance_usd,
                net_liquidation_usd=state.real_time_net_liquidation_usd,
                detail=detail,
            )
        )

    for session in validated_sessions:
        state = start_prop_session(state, rules)
        sessions_elapsed += 1
        event(SimulationEventKind.SESSION_STARTED, session.session_date, "session state reset")
        day_realized = 0.0
        day_consistency_profit = 0.0
        terminal_this_session = False

        for ordinal, trade in enumerate(session.trades, start=1):
            trade_id = trade.trade_id or f"{session.session_date.isoformat()}:{ordinal}"
            if session.market_day_status in {
                MarketDayStatus.CLOSED,
                MarketDayStatus.UNKNOWN,
            }:
                rejected_market_day += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_MARKET_DAY,
                    session.session_date,
                    f"{session.market_day_status.value} market-day status rejects every entry",
                    trade_id,
                )
                continue
            if state.soft_daily_loss_locked:
                rejected_daily += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_DAILY_LOSS,
                    session.session_date,
                    "firm daily-loss soft lock rejected the remaining session signal",
                    trade_id,
                )
                continue
            if trade.planned_risk_usd > (
                policy.max_planned_risk_per_trade_usd + _EPSILON
            ):
                rejected_personal_risk += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_PERSONAL_RISK,
                    session.session_date,
                    f"planned risk {trade.planned_risk_usd:.2f} exceeds personal cap "
                    f"{policy.max_planned_risk_per_trade_usd:.2f}",
                    trade_id,
                )
                continue
            if state.trades_this_session >= policy.max_trades_per_session:
                rejected_quota += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_TRADE_QUOTA,
                    session.session_date,
                    f"personal trade quota {policy.max_trades_per_session} already reached",
                    trade_id,
                )
                continue
            if (
                day_realized <= -policy.max_daily_strategy_loss_usd + _EPSILON
                or day_realized - trade.planned_risk_usd
                < -policy.max_daily_strategy_loss_usd - _EPSILON
            ):
                rejected_personal_daily += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_DAILY_STRATEGY_LOSS,
                    session.session_date,
                    f"post-stop session P&L would exceed personal daily loss "
                    f"{policy.max_daily_strategy_loss_usd:.2f}",
                    trade_id,
                )
                continue
            if (
                previous_executed_was_loss
                and previous_executed_planned_risk is not None
                and trade.planned_risk_usd
                > previous_executed_planned_risk + _EPSILON
            ):
                rejected_post_loss_increase += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_POST_LOSS_RISK_INCREASE,
                    session.session_date,
                    f"risk {trade.planned_risk_usd:.2f} exceeds prior losing trade risk "
                    f"{previous_executed_planned_risk:.2f}",
                    trade_id,
                )
                continue
            if state.real_time_net_liquidation_usd - trade.planned_risk_usd <= (
                state.internal_safety_threshold_usd + _EPSILON
            ):
                rejected_pretrade_safety += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_PRETRADE_SAFETY,
                    session.session_date,
                    f"planned stopped equity would touch/breach internal threshold "
                    f"{state.internal_safety_threshold_usd:.2f}",
                    trade_id,
                )
                continue
            if not _contract_allowed(trade, state, rules):
                rejected_contract += 1
                event(
                    SimulationEventKind.TRADE_REJECTED_CONTRACT_LIMIT,
                    session.session_date,
                    f"requested {trade.minis} mini/{trade.micros} micro exceeds "
                    f"{state.allowed_minis} mini/{state.allowed_micros} micro",
                    trade_id,
                )
                continue

            executed += 1
            commission += trade.commission_usd
            fees += trade.fees_usd
            slippage += trade.slippage_usd
            trades_this_session = state.trades_this_session + 1
            balance_before = state.current_balance_usd

            # First expose the account to the historical worst open mark.  This is the
            # critical real-time check that an EOD-only balance simulation misses.
            adverse_net_liq = balance_before + trade.minimum_intratrade_pnl_usd
            state = reconcile_prop_account(
                state,
                rules,
                current_balance_usd=balance_before,
                daily_realized_pnl_usd=day_realized,
                daily_unrealized_pnl_usd=trade.minimum_intratrade_pnl_usd,
                daily_consistency_profit_usd=day_consistency_profit,
                real_time_net_liquidation_usd=adverse_net_liq,
                open_minis=trade.minis,
                open_micros=trade.micros,
                trades_this_session=trades_this_session,
            )
            observe_equity(state)

            terminal_kind: SimulationEventKind | None = None
            if state.hard_breached:
                account_failed = True
                termination = SimulationTermination.ACCOUNT_FAILED
                terminal_kind = SimulationEventKind.HARD_ACCOUNT_BREACH
            elif state.real_time_net_liquidation_usd <= (
                state.internal_safety_threshold_usd + _EPSILON
            ):
                internal_locked = True
                termination = SimulationTermination.INTERNAL_SAFETY_LOCK
                terminal_kind = SimulationEventKind.INTERNAL_SAFETY_LOCK

            if terminal_kind is not None:
                # Conservatively flatten at the adverse mark, not at a potentially better
                # historical closing price after the safety/failure boundary was touched.
                forced_pnl = max(-balance_before, trade.minimum_intratrade_pnl_usd)
                forced_consistency_pnl = (
                    forced_pnl + trade.commission_usd + trade.fees_usd
                    if rules.consistency.profit_excludes_commissions
                    else forced_pnl
                )
                executed_trade_records.append(
                    replace(
                        trade,
                        net_pnl_usd=forced_pnl,
                        minimum_intratrade_pnl_usd=forced_pnl,
                    )
                )
                day_realized += forced_pnl
                day_consistency_profit += forced_consistency_pnl
                state = reconcile_prop_account(
                    state,
                    rules,
                    current_balance_usd=balance_before + forced_pnl,
                    daily_realized_pnl_usd=day_realized,
                    daily_unrealized_pnl_usd=0.0,
                    daily_consistency_profit_usd=day_consistency_profit,
                    real_time_net_liquidation_usd=balance_before + forced_pnl,
                    open_minis=0,
                    open_micros=0,
                    trades_this_session=trades_this_session,
                )
                observe_equity(state)
                current_losing_streak = current_losing_streak + 1 if forced_pnl < 0 else 0
                maximum_losing_streak = max(maximum_losing_streak, current_losing_streak)
                previous_executed_was_loss = forced_pnl < -_EPSILON
                previous_executed_planned_risk = trade.planned_risk_usd
                event(
                    SimulationEventKind.TRADE_EXECUTED,
                    session.session_date,
                    f"forced flat at adverse marked P&L {forced_pnl:.2f}",
                    trade_id,
                )
                event(
                    terminal_kind,
                    session.session_date,
                    "hard account failure" if account_failed else "internal safety threshold reached",
                    trade_id,
                )
                terminal_this_session = True
                break

            executed_trade_records.append(trade)
            day_realized += trade.net_pnl_usd
            day_consistency_profit += (
                trade.gross_pnl_before_explicit_costs_usd
                if rules.consistency.profit_excludes_commissions
                else trade.net_pnl_usd
            )
            state = reconcile_prop_account(
                state,
                rules,
                current_balance_usd=balance_before + trade.net_pnl_usd,
                daily_realized_pnl_usd=day_realized,
                daily_unrealized_pnl_usd=0.0,
                daily_consistency_profit_usd=day_consistency_profit,
                real_time_net_liquidation_usd=balance_before + trade.net_pnl_usd,
                open_minis=0,
                open_micros=0,
                trades_this_session=trades_this_session,
            )
            observe_equity(state)
            if trade.net_pnl_usd < -_EPSILON:
                current_losing_streak += 1
                maximum_losing_streak = max(maximum_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0
            previous_executed_was_loss = trade.net_pnl_usd < -_EPSILON
            previous_executed_planned_risk = trade.planned_risk_usd
            event(
                SimulationEventKind.TRADE_EXECUTED,
                session.session_date,
                f"net P&L {trade.net_pnl_usd:.2f}; commission {trade.commission_usd:.2f}; "
                f"fees {trade.fees_usd:.2f}; slippage {trade.slippage_usd:.2f}",
                trade_id,
            )

            if state.hard_breached:
                account_failed = True
                termination = SimulationTermination.ACCOUNT_FAILED
                event(
                    SimulationEventKind.HARD_ACCOUNT_BREACH,
                    session.session_date,
                    "closing balance breached the firm failure floor",
                    trade_id,
                )
                terminal_this_session = True
                break
            if state.real_time_net_liquidation_usd <= (
                state.internal_safety_threshold_usd + _EPSILON
            ):
                internal_locked = True
                termination = SimulationTermination.INTERNAL_SAFETY_LOCK
                event(
                    SimulationEventKind.INTERNAL_SAFETY_LOCK,
                    session.session_date,
                    "closing balance reached the internal safety threshold",
                    trade_id,
                )
                terminal_this_session = True
                break

        state = close_prop_session(state, rules)
        observe_equity(state)
        event(
            SimulationEventKind.SESSION_CLOSED,
            session.session_date,
            f"EOD balance {state.current_balance_usd:.2f}; floor {state.drawdown_floor_usd:.2f}",
        )

        if terminal_this_session:
            break

        if rules.phase is AccountPhase.EVALUATION and state.evaluation_passed:
            evaluation_passed = True
            sessions_to_pass = sessions_elapsed
            termination = SimulationTermination.EVALUATION_PASSED
            event(
                SimulationEventKind.EVALUATION_PASSED,
                session.session_date,
                "profit target, trading-day, and consistency rules passed at EOD",
            )
            break

        hold_state = _hold_duration_compliance(executed_trade_records, profile)
        if (
            rules.phase in {AccountPhase.SIM_FUNDED, AccountPhase.LIVE}
            and state.payout_eligible
            and hold_state.compliant
        ):
            payout_eligible = True
            sessions_to_payout = sessions_elapsed
            termination = SimulationTermination.PAYOUT_ELIGIBLE
            event(
                SimulationEventKind.PAYOUT_ELIGIBLE,
                session.session_date,
                "profile payout state and hold-duration ratios are eligible",
            )
            break

    hold_state = _hold_duration_compliance(executed_trade_records, profile)
    if rules.phase is AccountPhase.EVALUATION:
        evaluation_passed = evaluation_passed or (
            state.evaluation_passed and not account_failed and not internal_locked
        )
        if evaluation_passed and sessions_to_pass is None:
            sessions_to_pass = sessions_elapsed
    else:
        payout_eligible = payout_eligible or (
            state.payout_eligible
            and hold_state.compliant
            and not account_failed
            and not internal_locked
        )
        if payout_eligible and sessions_to_payout is None:
            sessions_to_payout = sessions_elapsed

    return PropSimulationResult(
        profile_id=profile.profile_id,
        rule_set_name=rules.name,
        phase=rules.phase,
        termination=termination,
        initial_balance_usd=initial_balance,
        final_balance_usd=state.current_balance_usd,
        net_profit_usd=state.current_balance_usd - initial_balance,
        maximum_drawdown_usd=maximum_drawdown,
        minimum_distance_to_failure_usd=minimum_distance,
        sessions_elapsed=sessions_elapsed,
        account_lifetime_sessions=sessions_elapsed,
        trading_days=state.trading_days,
        executed_trades=executed,
        rejected_trades=(
            rejected_daily
            + rejected_contract
            + rejected_personal_risk
            + rejected_quota
            + rejected_personal_daily
            + rejected_post_loss_increase
            + rejected_pretrade_safety
            + rejected_market_day
        ),
        rejected_daily_loss_trades=rejected_daily,
        rejected_contract_limit_trades=rejected_contract,
        rejected_personal_risk_trades=rejected_personal_risk,
        rejected_trade_quota_trades=rejected_quota,
        rejected_daily_strategy_loss_trades=rejected_personal_daily,
        rejected_post_loss_risk_increase_trades=rejected_post_loss_increase,
        rejected_pretrade_safety_trades=rejected_pretrade_safety,
        rejected_market_day_trades=rejected_market_day,
        total_commission_usd=commission,
        total_fees_usd=fees,
        total_slippage_usd=slippage,
        evaluation_passed=evaluation_passed,
        sessions_to_pass=sessions_to_pass,
        account_failed=account_failed,
        internal_safety_locked=internal_locked,
        payout_eligible=payout_eligible,
        sessions_to_payout=sessions_to_payout,
        hold_duration=hold_state,
        max_consecutive_losses=maximum_losing_streak,
        final_state=state,
        events=tuple(events),
    )


def simulate_evaluation_to_funded(
    profile: PropFirmProfile,
    evaluation_rules: AccountRuleSet,
    funded_rules: AccountRuleSet,
    sessions: Sequence[HistoricalPropSession],
    *,
    internal_safety_buffer_usd: float = 400.0,
    policy: PropSimulationPolicy = PropSimulationPolicy(),
) -> PropJourneyResult:
    """Run a fresh evaluation, then a separate fresh funded account on remaining blocks."""

    materialized = _validate_sessions(sessions, profile)
    if evaluation_rules not in profile.phases or funded_rules not in profile.phases:
        raise ValueError("journey rule sets must both belong to the supplied profile")
    if evaluation_rules.phase is not AccountPhase.EVALUATION:
        raise ValueError("evaluation_rules must use the evaluation phase")
    if funded_rules.phase not in {AccountPhase.SIM_FUNDED, AccountPhase.LIVE}:
        raise ValueError("funded_rules must use a funded phase")

    evaluation = simulate_prop_phase(
        profile,
        evaluation_rules,
        materialized,
        internal_safety_buffer_usd=internal_safety_buffer_usd,
        policy=policy,
    )
    funded: PropSimulationResult | None = None
    terminal_stage = JourneyStage.EVALUATION
    failure_stage: JourneyStage | None = (
        JourneyStage.EVALUATION
        if evaluation.account_failed
        else None
    )
    termination = evaluation.termination
    if evaluation.evaluation_passed:
        terminal_stage = JourneyStage.FUNDED
        termination = SimulationTermination.HORIZON_EXHAUSTED
        remaining = materialized[evaluation.sessions_elapsed :]
        if remaining:
            funded = simulate_prop_phase(
                profile,
                funded_rules,
                remaining,
                internal_safety_buffer_usd=internal_safety_buffer_usd,
                policy=policy,
            )
            termination = funded.termination
            if funded.account_failed:
                failure_stage = JourneyStage.FUNDED

    total_sessions = evaluation.sessions_elapsed + (
        funded.sessions_elapsed if funded is not None else 0
    )
    funded_payout_sessions = (
        funded.sessions_to_payout
        if funded is not None and funded.sessions_to_payout is not None
        else None
    )
    return PropJourneyResult(
        profile_id=profile.profile_id,
        evaluation_rule_set_name=evaluation_rules.name,
        funded_rule_set_name=funded_rules.name,
        terminal_stage=terminal_stage,
        failure_stage=failure_stage,
        termination=termination,
        total_sessions_elapsed=total_sessions,
        account_lifetime_sessions=total_sessions,
        sessions_to_evaluation_pass=evaluation.sessions_to_pass,
        sessions_to_payout=(
            evaluation.sessions_elapsed + funded_payout_sessions
            if funded_payout_sessions is not None
            else None
        ),
        evaluation_passed=evaluation.evaluation_passed,
        funded_started=funded is not None,
        payout_eligible=funded.payout_eligible if funded is not None else False,
        account_failed=(
            evaluation.account_failed
            or (funded.account_failed if funded is not None else False)
        ),
        internal_safety_locked=(
            evaluation.internal_safety_locked
            or (funded.internal_safety_locked if funded is not None else False)
        ),
        combined_net_profit_usd=(
            evaluation.net_profit_usd
            + (funded.net_profit_usd if funded is not None else 0.0)
        ),
        evaluation=evaluation,
        funded=funded,
    )


__all__ = ["simulate_evaluation_to_funded", "simulate_prop_phase"]
