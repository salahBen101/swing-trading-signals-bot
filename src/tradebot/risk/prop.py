"""Pure prop-account state transitions and the independent prop-risk gate.

This module deliberately has no broker, execution, strategy, clock, filesystem, or
network dependency.  Callers reconcile authoritative account values into an immutable
``PropAccountState`` and pass an inert proposal to :func:`evaluate_prop_order`.

The prop firm's published maximum is a failure boundary, not a risk budget.  The internal
threshold therefore sits *above* the firm floor by a configurable safety buffer.  Entries
whose worst-case stopped equity would touch either threshold are refused; exits bypass
every entry gate so a safety rule can never trap exposure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from ..deployment.stages import DeploymentStage
from ..prop_firms.models import (
    AccountPhase,
    AccountRuleSet,
    BreachKind,
    DrawdownMethod,
    PropFirmProfile,
)


_EPSILON = 1e-9


class MarketDayStatus(str, Enum):
    """Result supplied by an external exchange-calendar component.

    The gate intentionally does not guess whether a date is a holiday.  ``EARLY_CLOSE``
    selects the profile's holiday flatten deadline; ``CLOSED`` and ``UNKNOWN`` reject all
    entries.  The status refers to the active futures trade date (the date on which the
    overnight session settles), not necessarily the wall-clock date at the 18:00 open.
    """

    REGULAR = "regular"
    EARLY_CLOSE = "early_close"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class PropOrderAction(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class PropGateReason(str, Enum):
    PROFILE_MISSING = "profile_missing"
    PROFILE_INVALID = "profile_invalid"
    PROFILE_STALE = "profile_stale"
    RULE_VERIFICATION_TIME_INVALID = "rule_verification_time_invalid"
    RULE_SET_MISSING = "rule_set_missing"
    RULE_SET_MISMATCH = "rule_set_mismatch"
    STATE_MISSING = "state_missing"
    STATE_MISMATCH = "state_mismatch"
    STAGE_INVALID = "stage_invalid"
    STAGE_APPROVAL_REQUIRED = "stage_approval_required"
    STAGE_PHASE_MISMATCH = "stage_phase_mismatch"
    AMBIGUOUS_RULES = "ambiguous_rules"
    INVALID_TIMESTAMP = "invalid_timestamp"
    HOLIDAY_STATUS_UNKNOWN = "holiday_status_unknown"
    HOLIDAY_SCHEDULE_UNKNOWN = "holiday_schedule_unknown"
    MARKET_CLOSED = "market_closed"
    OUTSIDE_PERMITTED_HOURS = "outside_permitted_hours"
    AUTOMATION_PROHIBITED = "automation_prohibited"
    INVALID_ORDER = "invalid_order"
    CONTRACT_LIMIT = "contract_limit"
    EXISTING_EXPOSURE = "existing_exposure"
    AVERAGING_DOWN = "averaging_down"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SOFT_DAILY_LOSS_LOCK = "soft_daily_loss_lock"
    INTERNAL_SAFETY_THRESHOLD = "internal_safety_threshold"
    HARD_ACCOUNT_BREACH = "hard_account_breach"


class PayoutBlockReason(str, Enum):
    NOT_SUPPORTED = "payout_not_supported"
    HARD_ACCOUNT_BREACH = "hard_account_breach"
    MINIMUM_BALANCE = "minimum_balance_not_reached"
    WINNING_DAYS = "winning_days_not_reached"
    PROFIT_GOAL = "profit_goal_not_reached"
    POSITIVE_CYCLE = "positive_cycle_profit_required"
    CONSISTENCY = "payout_consistency_not_met"
    NO_AVAILABLE_PROFIT = "no_available_profit"
    MINIMUM_PAYOUT = "minimum_payout_not_available"
    CYCLE_MULTIPLIER = "cycle_profit_multiplier_not_met"


@dataclass(frozen=True, slots=True)
class PropAccountState:
    """Immutable account state used by both simulation and future guarded execution."""

    profile_id: str
    rule_set_name: str
    phase: AccountPhase
    starting_balance_usd: float
    current_balance_usd: float
    real_time_net_liquidation_usd: float
    highest_end_of_day_balance_usd: float
    drawdown_high_water_mark_usd: float
    drawdown_floor_usd: float
    drawdown_locked: bool
    remaining_prop_drawdown_usd: float
    distance_to_prop_failure_usd: float
    internal_safety_buffer_usd: float
    internal_safety_threshold_usd: float
    remaining_internal_cushion_usd: float
    daily_realized_pnl_usd: float
    daily_unrealized_pnl_usd: float
    firm_daily_loss_limit_usd: float | None
    remaining_firm_daily_risk_usd: float | None
    open_minis: int
    open_micros: int
    micro_per_mini: int
    allowed_minis: int
    allowed_micros: int
    trades_this_session: int
    trading_days: int
    payout_winning_days: int
    best_day_profit_usd: float
    cumulative_consistency_profit_usd: float
    daily_consistency_profit_usd: float
    payout_cycle_number: int
    payout_cycle_start_balance_usd: float
    payout_cycle_best_day_profit_usd: float
    payout_cycle_consistency_profit_usd: float
    evaluation_consistency_ratio: float | None
    evaluation_consistency_met: bool
    evaluation_passed: bool
    profit_target_remaining_usd: float | None
    payout_consistency_ratio: float | None
    payout_consistency_met: bool
    payout_eligible: bool
    payout_block_reasons: tuple[PayoutBlockReason, ...]
    payout_cap_usd: float | None
    payout_available_usd: float
    soft_daily_loss_locked: bool
    hard_breached: bool
    hard_breach_reasons: tuple[str, ...]

    @property
    def daily_pnl_usd(self) -> float:
        return self.daily_realized_pnl_usd + self.daily_unrealized_pnl_usd

    @property
    def gross_open_micro_equivalents(self) -> int:
        return self.open_minis * self.micro_per_mini + self.open_micros

    @property
    def has_open_exposure(self) -> bool:
        return self.open_minis > 0 or self.open_micros > 0

    def to_dict(self) -> dict:
        """JSON-safe complete dashboard/journal projection."""
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["payout_block_reasons"] = [item.value for item in self.payout_block_reasons]
        payload["daily_pnl_usd"] = self.daily_pnl_usd
        payload["gross_open_micro_equivalents"] = self.gross_open_micro_equivalents
        payload["has_open_exposure"] = self.has_open_exposure
        payload["consistency_status"] = (
            "MET"
            if self.evaluation_consistency_met and self.payout_consistency_met
            else "NOT MET"
        )
        payload["payout_status"] = (
            "ELIGIBLE"
            if self.payout_eligible
            else ", ".join(item.value for item in self.payout_block_reasons)
        )
        return payload


@dataclass(frozen=True, slots=True)
class PropPreTradeRequest:
    action: PropOrderAction
    now: datetime
    # Profile freshness is an operational fact, not a market-simulation fact.  A
    # historical bar timestamp must never make an old rule review appear current.
    rule_verification_as_of: datetime | None = None
    requested_minis: int = 0
    requested_micros: int = 0
    worst_case_loss_usd: float = 0.0
    deployment_stage: DeploymentStage = DeploymentStage.PAPER
    stage_authorized: bool = False
    market_day_status: MarketDayStatus | None = None
    increases_exposure: bool = True
    averaging_down: bool = False
    automated: bool = True


@dataclass(frozen=True, slots=True)
class PropGateDecision:
    allowed: bool
    reason_codes: tuple[PropGateReason, ...] = ()
    details: tuple[str, ...] = ()

    @property
    def primary_reason(self) -> PropGateReason | None:
        return self.reason_codes[0] if self.reason_codes else None

    def to_dict(self) -> dict:
        return {
            "layer": "PROP_FIRM",
            "allowed": self.allowed,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "details": list(self.details),
        }


def _finite_non_negative(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    return values if value in values else (*values, value)


def _allowed_contracts(rules: AccountRuleSet, highest_eod_balance: float) -> tuple[int, int]:
    limits = rules.contracts
    if not limits.scale_tiers:
        return limits.max_minis, limits.max_micros
    allowed_minis = limits.scale_tiers[0].max_minis
    allowed_micros = limits.scale_tiers[0].max_micros
    for tier in limits.scale_tiers:
        if highest_eod_balance + _EPSILON < tier.min_end_of_day_balance_usd:
            break
        allowed_minis = tier.max_minis
        allowed_micros = tier.max_micros
    return min(allowed_minis, limits.max_minis), min(allowed_micros, limits.max_micros)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= _EPSILON:
        return None
    return max(0.0, numerator) / denominator


def _consistency_met(ratio: float | None, limit: float | None) -> bool:
    if limit is None:
        return True
    return ratio is not None and ratio <= limit + _EPSILON


def _derive_status(state: PropAccountState, rules: AccountRuleSet) -> PropAccountState:
    """Refresh every value derived from authoritative state, latching breaches."""

    floor = rules.drawdown.floor(
        state.starting_balance_usd,
        state.drawdown_high_water_mark_usd,
        locked=state.drawdown_locked,
    )
    distance = state.real_time_net_liquidation_usd - floor
    internal_threshold = floor + state.internal_safety_buffer_usd
    daily_limit = rules.daily_loss_limit_for(state.highest_end_of_day_balance_usd)
    daily_pnl = state.daily_pnl_usd
    remaining_daily = (
        None
        if daily_limit is None
        else max(0.0, daily_limit - max(0.0, -daily_pnl))
    )

    hard_reasons = state.hard_breach_reasons
    hard_breached = state.hard_breached
    breach_value = (
        state.real_time_net_liquidation_usd
        if rules.drawdown.breach_basis == "real_time_net_liquidation"
        else state.current_balance_usd
    )
    if rules.drawdown.is_breached(breach_value, floor):
        if rules.drawdown.breach_kind is BreachKind.HARD:
            hard_breached = True
            hard_reasons = _append_unique(hard_reasons, "trailing_drawdown")

    soft_daily_locked = state.soft_daily_loss_locked
    if daily_limit is not None and daily_pnl <= -daily_limit + _EPSILON:
        if rules.daily_loss_breach_kind is BreachKind.SOFT:
            soft_daily_locked = True
        elif rules.daily_loss_breach_kind is BreachKind.HARD:
            hard_breached = True
            hard_reasons = _append_unique(hard_reasons, "daily_loss_limit")

    allowed_minis, allowed_micros = _allowed_contracts(
        rules, state.highest_end_of_day_balance_usd
    )

    total_profit = state.current_balance_usd - state.starting_balance_usd
    eval_ratio = _ratio(
        state.best_day_profit_usd,
        state.cumulative_consistency_profit_usd,
    )
    eval_limit = (
        rules.consistency.fraction_for_cycle(1)
        if "evaluation_pass" in rules.consistency.applies_to
        else None
    )
    eval_consistency_met = _consistency_met(eval_ratio, eval_limit)
    if rules.profit_target_usd is None:
        target_remaining = None
        evaluation_passed = False
    else:
        target_remaining = max(0.0, rules.profit_target_usd - total_profit)
        evaluation_passed = (
            rules.phase is AccountPhase.EVALUATION
            and target_remaining <= _EPSILON
            and state.trading_days >= rules.minimum_trading_days
            and eval_consistency_met
            and not hard_breached
        )

    cycle_profit = state.current_balance_usd - state.payout_cycle_start_balance_usd
    payout_ratio = _ratio(
        state.payout_cycle_best_day_profit_usd,
        state.payout_cycle_consistency_profit_usd,
    )
    payout_limit = (
        rules.consistency.fraction_for_cycle(state.payout_cycle_number)
        if "payout" in rules.consistency.applies_to
        else None
    )
    payout_consistency_met = _consistency_met(payout_ratio, payout_limit)
    payout = rules.payout
    payout_cap = payout.payout_cap(state.payout_cycle_number) if payout.eligible else None
    available = max(0.0, cycle_profit)
    if payout.max_fraction_total_profit is not None:
        total_available_profit = max(0.0, state.current_balance_usd - state.starting_balance_usd)
        # Select Flex permits up to a fraction of total remaining account profit.  A
        # positive new cycle is a separate eligibility condition, not the amount cap.
        available = total_available_profit * payout.max_fraction_total_profit
    if (
        payout.minimum_balance_must_remain_after_payout
        and payout.minimum_balance_usd is not None
    ):
        available = min(
            available,
            max(0.0, state.current_balance_usd - payout.minimum_balance_usd),
        )
    if payout_cap is not None:
        available = min(available, payout_cap)

    payout_blocks: list[PayoutBlockReason] = []
    if not payout.eligible:
        payout_blocks.append(PayoutBlockReason.NOT_SUPPORTED)
    else:
        if hard_breached:
            payout_blocks.append(PayoutBlockReason.HARD_ACCOUNT_BREACH)
        if (
            payout.minimum_balance_usd is not None
            and state.current_balance_usd + _EPSILON < payout.minimum_balance_usd
        ):
            payout_blocks.append(PayoutBlockReason.MINIMUM_BALANCE)
        if state.payout_winning_days < payout.winning_days_required:
            payout_blocks.append(PayoutBlockReason.WINNING_DAYS)
        profit_goal = (
            payout.first_profit_goal_usd
            if state.payout_cycle_number == 1
            else payout.subsequent_profit_goal_usd
        )
        if profit_goal is not None and cycle_profit + _EPSILON < profit_goal:
            payout_blocks.append(PayoutBlockReason.PROFIT_GOAL)
        if (
            state.payout_cycle_number > 1
            and payout.require_positive_cycle_after_first
            and cycle_profit <= _EPSILON
        ):
            payout_blocks.append(PayoutBlockReason.POSITIVE_CYCLE)
        if not payout_consistency_met:
            payout_blocks.append(PayoutBlockReason.CONSISTENCY)
        if available <= _EPSILON:
            payout_blocks.append(PayoutBlockReason.NO_AVAILABLE_PROFIT)
        if payout.minimum_payout_usd is not None and available + _EPSILON < payout.minimum_payout_usd:
            payout_blocks.append(PayoutBlockReason.MINIMUM_PAYOUT)
        # The profile cannot express a requested amount in account state.  For the generic
        # "could I request at least the minimum?" status, apply the multiplier to the
        # minimum payout on cycles after the first.  record_payout() checks the exact amount.
        if (
            state.payout_cycle_number > 1
            and payout.cycle_profit_multiplier is not None
            and payout.minimum_payout_usd is not None
            and cycle_profit + _EPSILON
            < payout.minimum_payout_usd * payout.cycle_profit_multiplier
        ):
            payout_blocks.append(PayoutBlockReason.CYCLE_MULTIPLIER)

    return replace(
        state,
        drawdown_floor_usd=floor,
        distance_to_prop_failure_usd=distance,
        remaining_prop_drawdown_usd=max(0.0, distance),
        internal_safety_threshold_usd=internal_threshold,
        remaining_internal_cushion_usd=max(
            0.0, state.real_time_net_liquidation_usd - internal_threshold
        ),
        firm_daily_loss_limit_usd=daily_limit,
        remaining_firm_daily_risk_usd=remaining_daily,
        allowed_minis=allowed_minis,
        allowed_micros=allowed_micros,
        evaluation_consistency_ratio=eval_ratio,
        evaluation_consistency_met=eval_consistency_met,
        evaluation_passed=evaluation_passed,
        profit_target_remaining_usd=target_remaining,
        payout_consistency_ratio=payout_ratio,
        payout_consistency_met=payout_consistency_met,
        payout_eligible=payout.eligible and not payout_blocks,
        payout_block_reasons=tuple(payout_blocks),
        payout_cap_usd=payout_cap,
        payout_available_usd=available,
        soft_daily_loss_locked=soft_daily_locked,
        hard_breached=hard_breached,
        hard_breach_reasons=hard_reasons,
    )


def initial_prop_account_state(
    profile: PropFirmProfile,
    rules: AccountRuleSet,
    *,
    internal_safety_buffer_usd: float = 400.0,
) -> PropAccountState:
    """Create a fresh account bound to one validated profile phase."""

    profile.validate()
    if rules not in profile.phases:
        raise ValueError(f"rule set {rules.name!r} does not belong to profile {profile.profile_id!r}")
    buffer = _finite_non_negative(internal_safety_buffer_usd, "internal safety buffer")
    start = float(rules.starting_balance_usd)
    floor = rules.drawdown.floor(start, start)
    state = PropAccountState(
        profile_id=profile.profile_id,
        rule_set_name=rules.name,
        phase=rules.phase,
        starting_balance_usd=start,
        current_balance_usd=start,
        real_time_net_liquidation_usd=start,
        highest_end_of_day_balance_usd=start,
        drawdown_high_water_mark_usd=start,
        drawdown_floor_usd=floor,
        drawdown_locked=False,
        remaining_prop_drawdown_usd=max(0.0, start - floor),
        distance_to_prop_failure_usd=start - floor,
        internal_safety_buffer_usd=buffer,
        internal_safety_threshold_usd=floor + buffer,
        remaining_internal_cushion_usd=max(0.0, start - floor - buffer),
        daily_realized_pnl_usd=0.0,
        daily_unrealized_pnl_usd=0.0,
        firm_daily_loss_limit_usd=rules.daily_loss_limit_for(start),
        remaining_firm_daily_risk_usd=rules.daily_loss_limit_for(start),
        open_minis=0,
        open_micros=0,
        micro_per_mini=rules.contracts.micro_per_mini,
        allowed_minis=rules.contracts.max_minis,
        allowed_micros=rules.contracts.max_micros,
        trades_this_session=0,
        trading_days=0,
        payout_winning_days=0,
        best_day_profit_usd=0.0,
        cumulative_consistency_profit_usd=0.0,
        daily_consistency_profit_usd=0.0,
        payout_cycle_number=1,
        payout_cycle_start_balance_usd=start,
        payout_cycle_best_day_profit_usd=0.0,
        payout_cycle_consistency_profit_usd=0.0,
        evaluation_consistency_ratio=None,
        evaluation_consistency_met=True,
        evaluation_passed=False,
        profit_target_remaining_usd=rules.profit_target_usd,
        payout_consistency_ratio=None,
        payout_consistency_met=True,
        payout_eligible=False,
        payout_block_reasons=(),
        payout_cap_usd=None,
        payout_available_usd=0.0,
        soft_daily_loss_locked=False,
        hard_breached=False,
        hard_breach_reasons=(),
    )
    return _derive_status(state, rules)


def reconcile_prop_account(
    state: PropAccountState,
    rules: AccountRuleSet,
    *,
    current_balance_usd: float,
    daily_realized_pnl_usd: float,
    daily_unrealized_pnl_usd: float = 0.0,
    daily_consistency_profit_usd: float | None = None,
    real_time_net_liquidation_usd: float | None = None,
    open_minis: int = 0,
    open_micros: int = 0,
    trades_this_session: int | None = None,
) -> PropAccountState:
    """Reconcile an authoritative broker/dashboard snapshot without side effects.

    Daily values are absolute session-to-date values rather than deltas, making repeated
    reconciliation idempotent.  When net liquidation is omitted it is derived as realized
    balance plus unrealized P&L.
    """

    if state.rule_set_name != rules.name or state.phase is not rules.phase:
        raise ValueError("state and rule set do not match")
    balance = _finite_non_negative(current_balance_usd, "current balance")
    realized = float(daily_realized_pnl_usd)
    unrealized = float(daily_unrealized_pnl_usd)
    if not math.isfinite(realized) or not math.isfinite(unrealized):
        raise ValueError("daily P&L values must be finite")
    consistency_profit = realized if daily_consistency_profit_usd is None else float(
        daily_consistency_profit_usd
    )
    if not math.isfinite(consistency_profit):
        raise ValueError("daily consistency profit must be finite")
    net_liq = balance + unrealized if real_time_net_liquidation_usd is None else float(
        real_time_net_liquidation_usd
    )
    if not math.isfinite(net_liq):
        raise ValueError("real-time net liquidation must be finite")
    if open_minis < 0 or open_micros < 0:
        raise ValueError("open contract counts must be non-negative")
    trades = state.trades_this_session if trades_this_session is None else trades_this_session
    if trades < 0:
        raise ValueError("trades_this_session must be non-negative")

    high_water = state.drawdown_high_water_mark_usd
    if rules.drawdown.method is DrawdownMethod.INTRADAY_TRAILING:
        high_water = max(high_water, net_liq)
    elif rules.drawdown.method is DrawdownMethod.STATIC:
        high_water = state.starting_balance_usd

    updated = replace(
        state,
        current_balance_usd=balance,
        real_time_net_liquidation_usd=net_liq,
        daily_realized_pnl_usd=realized,
        daily_unrealized_pnl_usd=unrealized,
        daily_consistency_profit_usd=consistency_profit,
        drawdown_high_water_mark_usd=high_water,
        open_minis=int(open_minis),
        open_micros=int(open_micros),
        trades_this_session=int(trades),
    )
    return _derive_status(updated, rules)


def close_prop_session(
    state: PropAccountState,
    rules: AccountRuleSet,
    *,
    end_of_day_balance_usd: float | None = None,
) -> PropAccountState:
    """Record a flat account's session result and advance EOD-based sticky rules."""

    if state.has_open_exposure:
        raise ValueError("cannot close a prop session with open exposure")
    if abs(state.daily_unrealized_pnl_usd) > _EPSILON:
        raise ValueError("cannot close a prop session with unrealized P&L")
    eod_balance = state.current_balance_usd if end_of_day_balance_usd is None else (
        _finite_non_negative(end_of_day_balance_usd, "end-of-day balance")
    )
    highest_eod = max(state.highest_end_of_day_balance_usd, eod_balance)
    high_water = state.drawdown_high_water_mark_usd
    if rules.drawdown.method is DrawdownMethod.END_OF_DAY_TRAILING:
        high_water = max(high_water, highest_eod)
    elif rules.drawdown.method is DrawdownMethod.STATIC:
        high_water = state.starting_balance_usd

    locked = state.drawdown_locked
    if (
        rules.drawdown.locks
        and rules.drawdown.lock_trigger_balance_usd is not None
        and highest_eod + _EPSILON >= rules.drawdown.lock_trigger_balance_usd
    ):
        locked = True

    traded = state.trades_this_session > 0
    winning_day = False
    if traded and rules.payout.winning_days_required > 0:
        threshold = rules.payout.winning_day_min_profit_usd
        winning_day = (
            state.daily_realized_pnl_usd > threshold + _EPSILON
            if rules.payout.winning_day_strictly_greater
            else state.daily_realized_pnl_usd + _EPSILON >= threshold
        )
    day_profit = state.daily_consistency_profit_usd if traded else 0.0
    updated = replace(
        state,
        current_balance_usd=eod_balance,
        real_time_net_liquidation_usd=eod_balance,
        highest_end_of_day_balance_usd=highest_eod,
        drawdown_high_water_mark_usd=high_water,
        drawdown_locked=locked,
        trading_days=state.trading_days + int(traded),
        payout_winning_days=state.payout_winning_days + int(winning_day),
        best_day_profit_usd=max(state.best_day_profit_usd, day_profit),
        cumulative_consistency_profit_usd=(
            state.cumulative_consistency_profit_usd + day_profit
        ),
        payout_cycle_best_day_profit_usd=max(
            state.payout_cycle_best_day_profit_usd, day_profit
        ),
        payout_cycle_consistency_profit_usd=(
            state.payout_cycle_consistency_profit_usd + day_profit
        ),
    )
    return _derive_status(updated, rules)


def start_prop_session(state: PropAccountState, rules: AccountRuleSet) -> PropAccountState:
    """Reset soft, session-scoped fields; hard breaches can never be reset here."""

    if state.has_open_exposure:
        raise ValueError("cannot start a new prop session with open exposure")
    updated = replace(
        state,
        real_time_net_liquidation_usd=state.current_balance_usd,
        daily_realized_pnl_usd=0.0,
        daily_unrealized_pnl_usd=0.0,
        daily_consistency_profit_usd=0.0,
        trades_this_session=0,
        soft_daily_loss_locked=False,
    )
    return _derive_status(updated, rules)


def record_prop_payout(
    state: PropAccountState,
    rules: AccountRuleSet,
    *,
    amount_usd: float,
) -> PropAccountState:
    """Apply an already-approved payout and begin the next payout cycle."""

    if state.has_open_exposure:
        raise ValueError("cannot record a payout with open exposure")
    amount = _finite_non_negative(amount_usd, "payout amount")
    if amount <= _EPSILON:
        raise ValueError("payout amount must be positive")
    if not state.payout_eligible:
        raise ValueError(f"payout is not eligible: {[item.value for item in state.payout_block_reasons]}")
    payout = rules.payout
    if payout.minimum_payout_usd is not None and amount + _EPSILON < payout.minimum_payout_usd:
        raise ValueError("payout amount is below the configured minimum")
    if amount > state.payout_available_usd + _EPSILON:
        raise ValueError("payout amount exceeds the currently available amount")
    if (
        payout.minimum_balance_must_remain_after_payout
        and payout.minimum_balance_usd is not None
        and state.current_balance_usd - amount < payout.minimum_balance_usd - _EPSILON
    ):
        raise ValueError("payout would reduce the account below the required balance")
    cycle_profit = state.current_balance_usd - state.payout_cycle_start_balance_usd
    if (
        state.payout_cycle_number > 1
        and payout.cycle_profit_multiplier is not None
        and cycle_profit + _EPSILON < amount * payout.cycle_profit_multiplier
    ):
        raise ValueError("cycle profit is below the requested payout multiplier")

    balance = state.current_balance_usd - amount
    net_liq = state.real_time_net_liquidation_usd - amount
    locked = state.drawdown_locked or payout.locks_drawdown_on_payout
    updated = replace(
        state,
        current_balance_usd=balance,
        real_time_net_liquidation_usd=net_liq,
        drawdown_locked=locked,
        payout_cycle_number=state.payout_cycle_number + 1,
        payout_cycle_start_balance_usd=balance,
        payout_cycle_best_day_profit_usd=(
            0.0 if rules.consistency.resets_after_payout else state.payout_cycle_best_day_profit_usd
        ),
        payout_cycle_consistency_profit_usd=(
            0.0
            if rules.consistency.resets_after_payout
            else state.payout_cycle_consistency_profit_usd
        ),
        payout_winning_days=0,
    )
    return _derive_status(updated, rules)


def _add_reason(
    reasons: list[PropGateReason],
    details: list[str],
    reason: PropGateReason,
    detail: str,
) -> None:
    if reason not in reasons:
        reasons.append(reason)
        details.append(detail)


def _permitted_time_reason(
    profile: PropFirmProfile,
    moment: datetime,
    market_day_status: MarketDayStatus,
) -> tuple[PropGateReason, str] | None:
    if market_day_status is MarketDayStatus.CLOSED:
        return PropGateReason.MARKET_CLOSED, "exchange calendar marks the trade date closed"
    if market_day_status is MarketDayStatus.UNKNOWN:
        return (
            PropGateReason.HOLIDAY_STATUS_UNKNOWN,
            "exchange holiday/session status is unknown",
        )

    window = profile.trading_window
    local = moment.astimezone(ZoneInfo(window.timezone))
    local_time = local.time().replace(tzinfo=None)
    start = time.fromisoformat(window.session_start)
    end = time.fromisoformat(window.session_end)

    # Determine the session anchor so Sunday morning and Friday evening do not look open
    # merely because their clock times fall inside an overnight 18:00-17:00 window.
    if start > end:
        if local_time >= start:
            anchor = local.date()
            evening_leg = True
        elif local_time < end:
            anchor = local.date() - timedelta(days=1)
            evening_leg = False
        else:
            return (
                PropGateReason.OUTSIDE_PERMITTED_HOURS,
                f"{local_time} is in the daily session halt {end}-{start}",
            )
        # CME index futures sessions start Sunday through Thursday.
        if anchor.weekday() not in {6, 0, 1, 2, 3}:
            return PropGateReason.MARKET_CLOSED, "no futures session is open on this weekend leg"
    else:
        evening_leg = False
        if not start <= local_time < end:
            return (
                PropGateReason.OUTSIDE_PERMITTED_HOURS,
                f"{local_time} is outside the permitted session {start}-{end}",
            )
        if local.weekday() >= 5:
            return PropGateReason.MARKET_CLOSED, "exchange session is closed on weekends"

    deadline_text = window.flatten_by
    if market_day_status is MarketDayStatus.EARLY_CLOSE:
        if window.holiday_flatten_by is None:
            return (
                PropGateReason.HOLIDAY_SCHEDULE_UNKNOWN,
                "profile has no holiday flatten deadline for an early-close session",
            )
        deadline_text = window.holiday_flatten_by
    deadline = time.fromisoformat(deadline_text)
    # On an overnight session the 18:00 evening leg is *after the prior deadline* but
    # before the following day's deadline, so only apply flatten_by on the daytime leg.
    if not evening_leg and local_time >= deadline:
        return (
            PropGateReason.OUTSIDE_PERMITTED_HOURS,
            f"{local_time} is at/after the required {deadline} flatten deadline",
        )
    return None


def evaluate_prop_order(
    request: PropPreTradeRequest,
    *,
    profile: PropFirmProfile | None,
    rules: AccountRuleSet | None,
    state: PropAccountState | None,
) -> PropGateDecision:
    """Evaluate an entry against prop rules and immutable account state.

    Every uncertainty is a rejection.  The sole exception is deliberate: an ``EXIT`` is
    always accepted before profile, time, breach, and even request-shape checks.
    """

    if request.action is PropOrderAction.EXIT:
        return PropGateDecision(
            allowed=True,
            details=("risk-reducing exits bypass every entry-only prop gate",),
        )

    reasons: list[PropGateReason] = []
    details: list[str] = []
    moment_is_aware = request.now.tzinfo is not None and request.now.utcoffset() is not None
    if not moment_is_aware:
        _add_reason(
            reasons,
            details,
            PropGateReason.INVALID_TIMESTAMP,
            "pre-trade timestamp must be timezone-aware",
        )

    verification_as_of = request.rule_verification_as_of
    verification_time_is_aware = (
        isinstance(verification_as_of, datetime)
        and verification_as_of.tzinfo is not None
        and verification_as_of.utcoffset() is not None
    )
    if not verification_time_is_aware:
        _add_reason(
            reasons,
            details,
            PropGateReason.RULE_VERIFICATION_TIME_INVALID,
            "rule-verification as-of time must be supplied and timezone-aware",
        )

    valid_profile = False
    if profile is None:
        _add_reason(reasons, details, PropGateReason.PROFILE_MISSING, "no prop profile supplied")
    else:
        try:
            profile.validate()
            valid_profile = True
        except (TypeError, ValueError) as exc:
            _add_reason(reasons, details, PropGateReason.PROFILE_INVALID, str(exc))
        if valid_profile and verification_time_is_aware:
            assert verification_as_of is not None
            utc_date = verification_as_of.astimezone(ZoneInfo("UTC")).date()
            if profile.verified_on > utc_date:
                _add_reason(
                    reasons,
                    details,
                    PropGateReason.PROFILE_INVALID,
                    "profile verification date is in the future",
                )
            elif not profile.rules_are_fresh(verification_as_of):
                _add_reason(
                    reasons,
                    details,
                    PropGateReason.PROFILE_STALE,
                    f"profile age exceeds {profile.reverify_after_hours} hours",
                )

    if rules is None:
        _add_reason(reasons, details, PropGateReason.RULE_SET_MISSING, "no account rule set supplied")
    elif profile is not None and valid_profile and rules not in profile.phases:
        _add_reason(
            reasons,
            details,
            PropGateReason.RULE_SET_MISMATCH,
            f"rule set {rules.name!r} does not belong to profile {profile.profile_id!r}",
        )

    if state is None:
        _add_reason(reasons, details, PropGateReason.STATE_MISSING, "no prop account state supplied")
    elif profile is not None and rules is not None and (
        state.profile_id != profile.profile_id
        or state.rule_set_name != rules.name
        or state.phase is not rules.phase
    ):
        _add_reason(
            reasons,
            details,
            PropGateReason.STATE_MISMATCH,
            "account state is not bound to the selected profile phase",
        )

    try:
        stage = DeploymentStage(request.deployment_stage)
    except (TypeError, ValueError):
        stage = None
        _add_reason(
            reasons,
            details,
            PropGateReason.STAGE_INVALID,
            f"unsupported deployment stage {request.deployment_stage!r}",
        )
    if stage is not None and stage.requires_human_approval:
        if not request.stage_authorized:
            _add_reason(
                reasons,
                details,
                PropGateReason.STAGE_APPROVAL_REQUIRED,
                f"Stage {int(stage)} requires explicit human approval",
            )
        if profile is not None and profile.ambiguity_notes:
            _add_reason(
                reasons,
                details,
                PropGateReason.AMBIGUOUS_RULES,
                "profile has unresolved ambiguity notes and cannot be used at Stage 3/4",
            )
        if rules is not None:
            expected = (
                rules.phase is AccountPhase.EVALUATION
                if stage is DeploymentStage.PROP_EVALUATION
                else rules.phase in {AccountPhase.SIM_FUNDED, AccountPhase.LIVE}
            )
            if not expected:
                _add_reason(
                    reasons,
                    details,
                    PropGateReason.STAGE_PHASE_MISMATCH,
                    f"{rules.phase.value} rules do not match Stage {int(stage)}",
                )

    if profile is not None and valid_profile:
        if request.automated and not profile.compliance.automation_allowed:
            _add_reason(
                reasons,
                details,
                PropGateReason.AUTOMATION_PROHIBITED,
                "selected profile prohibits automated trading",
            )
        status = request.market_day_status
        if status is None:
            _add_reason(
                reasons,
                details,
                PropGateReason.HOLIDAY_STATUS_UNKNOWN,
                "no exchange calendar status was supplied",
            )
        else:
            try:
                status = MarketDayStatus(status)
            except (TypeError, ValueError):
                status = MarketDayStatus.UNKNOWN
            if not moment_is_aware:
                pass
            else:
                time_rejection = _permitted_time_reason(profile, request.now, status)
                if time_rejection is not None:
                    _add_reason(reasons, details, *time_rejection)

    quantities_valid = (
        isinstance(request.requested_minis, int)
        and not isinstance(request.requested_minis, bool)
        and isinstance(request.requested_micros, int)
        and not isinstance(request.requested_micros, bool)
        and request.requested_minis >= 0
        and request.requested_micros >= 0
        and request.requested_minis + request.requested_micros > 0
    )
    loss_valid = (
        isinstance(request.worst_case_loss_usd, (int, float))
        and not isinstance(request.worst_case_loss_usd, bool)
        and math.isfinite(float(request.worst_case_loss_usd))
        and request.worst_case_loss_usd > 0
    )
    if not quantities_valid or not loss_valid:
        _add_reason(
            reasons,
            details,
            PropGateReason.INVALID_ORDER,
            "entry needs positive integer contract quantity and finite positive worst-case loss",
        )

    if request.averaging_down:
        _add_reason(
            reasons,
            details,
            PropGateReason.AVERAGING_DOWN,
            "averaging down is prohibited even when the firm permits it",
        )

    if state is not None:
        if state.hard_breached:
            _add_reason(
                reasons,
                details,
                PropGateReason.HARD_ACCOUNT_BREACH,
                f"account has a latched hard breach: {state.hard_breach_reasons}",
            )
        if state.soft_daily_loss_locked:
            _add_reason(
                reasons,
                details,
                PropGateReason.SOFT_DAILY_LOSS_LOCK,
                "firm daily-loss limit locked entries for this session",
            )
        if (
            state.firm_daily_loss_limit_usd is not None
            and state.daily_pnl_usd <= -state.firm_daily_loss_limit_usd + _EPSILON
        ):
            _add_reason(
                reasons,
                details,
                PropGateReason.DAILY_LOSS_LIMIT,
                "marked daily P&L is at or beyond the firm daily-loss boundary",
            )
        if state.has_open_exposure and request.increases_exposure:
            _add_reason(
                reasons,
                details,
                PropGateReason.EXISTING_EXPOSURE,
                "an entry cannot add to an existing position",
            )
        if loss_valid:
            stopped_equity = state.real_time_net_liquidation_usd - float(
                request.worst_case_loss_usd
            )
            if stopped_equity <= state.internal_safety_threshold_usd + _EPSILON:
                _add_reason(
                    reasons,
                    details,
                    PropGateReason.INTERNAL_SAFETY_THRESHOLD,
                    f"post-stop equity {stopped_equity:.2f} would touch/breach internal "
                    f"threshold {state.internal_safety_threshold_usd:.2f}",
                )

    if quantities_valid and rules is not None and state is not None:
        total_minis = state.open_minis + request.requested_minis
        total_micros = state.open_micros + request.requested_micros
        limits = rules.contracts
        allowed_minis, allowed_micros = _allowed_contracts(
            rules, state.highest_end_of_day_balance_usd
        )
        exceeds = total_minis > allowed_minis or total_micros > allowed_micros
        if total_minis and total_micros and not limits.mixed_sizes_allowed:
            exceeds = True
        if limits.aggregate_gross_open_position:
            equivalent_micros = total_minis * limits.micro_per_mini + total_micros
            exceeds = exceeds or equivalent_micros > allowed_micros
        if exceeds:
            _add_reason(
                reasons,
                details,
                PropGateReason.CONTRACT_LIMIT,
                f"post-order gross exposure {total_minis} mini/{total_micros} micro exceeds "
                f"current cap {allowed_minis} mini/{allowed_micros} micro",
            )

    return PropGateDecision(
        allowed=not reasons,
        reason_codes=tuple(reasons),
        details=tuple(details),
    )


__all__ = [
    "DeploymentStage",
    "MarketDayStatus",
    "PayoutBlockReason",
    "PropAccountState",
    "PropGateDecision",
    "PropGateReason",
    "PropOrderAction",
    "PropPreTradeRequest",
    "close_prop_session",
    "evaluate_prop_order",
    "initial_prop_account_state",
    "reconcile_prop_account",
    "record_prop_payout",
    "start_prop_session",
]
