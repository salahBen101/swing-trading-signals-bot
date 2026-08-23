"""Deterministic whole-session bootstrap for prop-account survival outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
import math
import random
from statistics import fmean, median
from typing import Sequence
from zoneinfo import ZoneInfo

from tradebot.prop_firms import AccountRuleSet, PropFirmProfile

from .engine import _validate_sessions, simulate_prop_phase
from .models import HistoricalPropSession, PropSimulationPolicy, PropSimulationResult


@dataclass(frozen=True, slots=True)
class MonteCarloReport:
    """Aggregate survival metrics plus immutable per-run evidence."""

    simulations: int
    seed: int
    horizon_sessions: int
    pass_rate: float
    account_failure_rate: float
    internal_safety_lock_rate: float
    payout_probability: float
    median_days_to_pass: float | None
    average_drawdown_usd: float
    maximum_drawdown_usd: float
    expected_lifetime_sessions: float
    expected_profit_per_account_usd: float
    average_max_consecutive_losses: float
    consecutive_losses_survived: int
    runs: tuple[PropSimulationResult, ...]


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _synthetic_dates(count: int) -> tuple[date, ...]:
    first = date(2040, 1, 2)
    while first.weekday() >= 5:
        first += timedelta(days=1)
    values = [first]
    for _ in range(1, count):
        values.append(_next_weekday(values[-1]))
    return tuple(values)


def _rebase_session(
    source: HistoricalPropSession,
    target_date: date,
    *,
    profile: PropFirmProfile,
    run_number: int,
    draw_number: int,
) -> HistoricalPropSession:
    """Move a complete block to a synthetic date without changing its within-day path."""

    zone = ZoneInfo(profile.trading_window.timezone)
    source_anchor = datetime.combine(source.session_date, time.min, tzinfo=zone)
    target_anchor = datetime.combine(target_date, time.min, tzinfo=zone)
    rebased = []
    for ordinal, trade in enumerate(source.trades, start=1):
        opened_delta = trade.opened_at.astimezone(zone) - source_anchor
        closed_delta = trade.closed_at.astimezone(zone) - source_anchor
        original_id = trade.trade_id or str(ordinal)
        rebased.append(
            replace(
                trade,
                opened_at=target_anchor + opened_delta,
                closed_at=target_anchor + closed_delta,
                trade_id=f"mc{run_number}:block{draw_number}:{original_id}",
            )
        )
    return HistoricalPropSession(
        session_date=target_date,
        market_day_status=source.market_day_status,
        trades=tuple(rebased),
    )


def run_session_bootstrap(
    profile: PropFirmProfile,
    rules: AccountRuleSet,
    sessions: Sequence[HistoricalPropSession],
    *,
    simulations: int,
    seed: int,
    horizon_sessions: int | None = None,
    internal_safety_buffer_usd: float = 400.0,
    policy: PropSimulationPolicy = PropSimulationPolicy(),
) -> MonteCarloReport:
    """Bootstrap complete historical session blocks with replacement.

    No trade is sampled independently.  This preserves within-session order, correlation,
    transaction costs, adverse excursions, and DLL interactions.  A local PRNG makes the
    result independent of process-global random state and reproducible from ``seed``.
    """

    if not isinstance(profile, PropFirmProfile):
        raise ValueError("profile must be a PropFirmProfile")
    if not isinstance(rules, AccountRuleSet):
        raise ValueError("rules must be an AccountRuleSet")
    if not isinstance(policy, PropSimulationPolicy):
        raise ValueError("policy must be a PropSimulationPolicy")
    policy.validate()
    profile.validate()
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    source = _validate_sessions(sessions, profile)
    if not any(session.trades for session in source):
        raise ValueError("Monte Carlo source must contain at least one trade")
    if rules not in profile.phases:
        raise ValueError("rules do not belong to the supplied profile")
    horizon = len(source) if horizon_sessions is None else horizon_sessions
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon_sessions must be a positive integer")
    if (
        isinstance(internal_safety_buffer_usd, bool)
        or not isinstance(internal_safety_buffer_usd, (int, float))
        or not math.isfinite(float(internal_safety_buffer_usd))
        or internal_safety_buffer_usd < 0
    ):
        raise ValueError("internal_safety_buffer_usd must be finite and non-negative")

    generator = random.Random(seed)
    target_dates = _synthetic_dates(horizon)
    runs: list[PropSimulationResult] = []
    for run_number in range(1, simulations + 1):
        sampled = tuple(
            _rebase_session(
                source[generator.randrange(len(source))],
                target_date,
                profile=profile,
                run_number=run_number,
                draw_number=draw_number,
            )
            for draw_number, target_date in enumerate(target_dates, start=1)
        )
        runs.append(
            simulate_prop_phase(
                profile,
                rules,
                sampled,
                internal_safety_buffer_usd=float(internal_safety_buffer_usd),
                policy=policy,
            )
        )

    immutable_runs = tuple(runs)
    pass_days = [run.sessions_to_pass for run in runs if run.sessions_to_pass is not None]
    surviving_runs = [
        run
        for run in runs
        if not run.account_failed and not run.internal_safety_locked
    ]
    consecutive_losses_survived = max(
        (run.max_consecutive_losses for run in surviving_runs), default=0
    )
    return MonteCarloReport(
        simulations=simulations,
        seed=seed,
        horizon_sessions=horizon,
        pass_rate=sum(run.evaluation_passed for run in runs) / simulations,
        account_failure_rate=sum(run.account_failed for run in runs) / simulations,
        internal_safety_lock_rate=sum(run.internal_safety_locked for run in runs) / simulations,
        payout_probability=sum(run.payout_eligible for run in runs) / simulations,
        median_days_to_pass=float(median(pass_days)) if pass_days else None,
        average_drawdown_usd=fmean(run.maximum_drawdown_usd for run in runs),
        maximum_drawdown_usd=max(run.maximum_drawdown_usd for run in runs),
        expected_lifetime_sessions=fmean(run.account_lifetime_sessions for run in runs),
        expected_profit_per_account_usd=fmean(run.net_profit_usd for run in runs),
        average_max_consecutive_losses=fmean(run.max_consecutive_losses for run in runs),
        consecutive_losses_survived=consecutive_losses_survived,
        runs=immutable_runs,
    )


__all__ = ["MonteCarloReport", "run_session_bootstrap"]
