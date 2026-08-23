"""Whole-session Monte Carlo for fresh evaluation-to-funded journeys."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean, median
from typing import Sequence

from tradebot.prop_firms import AccountPhase, AccountRuleSet, PropFirmProfile

from .engine import _validate_sessions, simulate_evaluation_to_funded
from .models import HistoricalPropSession, PropJourneyResult, PropSimulationPolicy
from .monte_carlo import _rebase_session, _synthetic_dates


JOURNEY_MONTE_CARLO_LIMITATIONS = (
    "Historical session blocks are treated as exchangeable; regime persistence outside a "
    "sampled session is not modeled.",
    "Intratrade risk is limited to each source trade's supplied adverse excursion; no tick "
    "path, partial-fill, liquidity, or market-impact path is invented.",
    "Runs that neither pass nor fail before the configured horizon are right-censored at "
    "that horizon.",
    "Journey drawdown is the larger phase-account drawdown, not an artificial equity curve "
    "that carries evaluation profits into the fresh funded account.",
    "Expected journey profit is the arithmetic sum of two separate simulated account "
    "outcomes; evaluation profit is not transferable cash or a payout.",
)


@dataclass(frozen=True, slots=True)
class JourneyMonteCarloReport:
    """Survival-first metrics and exact zero-based input-session sampling evidence."""

    simulations: int
    seed: int
    horizon_sessions: int
    evaluation_pass_rate: float
    account_failure_rate: float
    evaluation_account_failure_rate: float
    funded_account_failure_rate: float
    internal_safety_lock_rate: float
    payout_probability: float
    median_days_to_pass: float | None
    median_days_to_payout: float | None
    average_drawdown_usd: float
    maximum_drawdown_usd: float
    expected_lifetime_sessions: float
    expected_profit_per_journey_usd: float
    sampled_source_indices: tuple[tuple[int, ...], ...]
    limitations: tuple[str, ...]
    runs: tuple[PropJourneyResult, ...]


def _journey_drawdown(result: PropJourneyResult) -> float:
    funded_drawdown = (
        result.funded.maximum_drawdown_usd if result.funded is not None else 0.0
    )
    return max(result.evaluation.maximum_drawdown_usd, funded_drawdown)


def run_journey_session_bootstrap(
    profile: PropFirmProfile,
    evaluation_rules: AccountRuleSet,
    funded_rules: AccountRuleSet,
    sessions: Sequence[HistoricalPropSession],
    *,
    simulations: int,
    seed: int,
    horizon_sessions: int | None = None,
    internal_safety_buffer_usd: float = 400.0,
    policy: PropSimulationPolicy = PropSimulationPolicy(),
) -> JourneyMonteCarloReport:
    """Bootstrap complete sessions and run a fresh two-account journey for each path."""

    if not isinstance(profile, PropFirmProfile):
        raise ValueError("profile must be a PropFirmProfile")
    if not isinstance(evaluation_rules, AccountRuleSet):
        raise ValueError("evaluation_rules must be an AccountRuleSet")
    if not isinstance(funded_rules, AccountRuleSet):
        raise ValueError("funded_rules must be an AccountRuleSet")
    if not isinstance(policy, PropSimulationPolicy):
        raise ValueError("policy must be a PropSimulationPolicy")
    profile.validate()
    policy.validate()
    if evaluation_rules not in profile.phases or funded_rules not in profile.phases:
        raise ValueError("journey rule sets must both belong to the supplied profile")
    if evaluation_rules.phase is not AccountPhase.EVALUATION:
        raise ValueError("evaluation_rules must use the evaluation phase")
    if funded_rules.phase not in {AccountPhase.SIM_FUNDED, AccountPhase.LIVE}:
        raise ValueError("funded_rules must use a funded phase")
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if (
        isinstance(internal_safety_buffer_usd, bool)
        or not isinstance(internal_safety_buffer_usd, (int, float))
        or not math.isfinite(float(internal_safety_buffer_usd))
        or internal_safety_buffer_usd < 0
    ):
        raise ValueError("internal_safety_buffer_usd must be finite and non-negative")

    source = _validate_sessions(sessions, profile)
    if not any(session.trades for session in source):
        raise ValueError("Monte Carlo source must contain at least one trade")
    horizon = len(source) if horizon_sessions is None else horizon_sessions
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon_sessions must be a positive integer")

    generator = random.Random(seed)
    target_dates = _synthetic_dates(horizon)
    paths: list[tuple[int, ...]] = []
    runs: list[PropJourneyResult] = []
    for run_number in range(1, simulations + 1):
        indices = tuple(generator.randrange(len(source)) for _ in range(horizon))
        paths.append(indices)
        sampled = tuple(
            _rebase_session(
                source[source_index],
                target_date,
                profile=profile,
                run_number=run_number,
                draw_number=draw_number,
            )
            for draw_number, (source_index, target_date) in enumerate(
                zip(indices, target_dates, strict=True), start=1
            )
        )
        runs.append(
            simulate_evaluation_to_funded(
                profile,
                evaluation_rules,
                funded_rules,
                sampled,
                internal_safety_buffer_usd=float(internal_safety_buffer_usd),
                policy=policy,
            )
        )

    pass_days = [
        run.sessions_to_evaluation_pass
        for run in runs
        if run.sessions_to_evaluation_pass is not None
    ]
    payout_days = [run.sessions_to_payout for run in runs if run.sessions_to_payout is not None]
    drawdowns = [_journey_drawdown(run) for run in runs]
    return JourneyMonteCarloReport(
        simulations=simulations,
        seed=seed,
        horizon_sessions=horizon,
        evaluation_pass_rate=sum(run.evaluation_passed for run in runs) / simulations,
        account_failure_rate=sum(run.account_failed for run in runs) / simulations,
        evaluation_account_failure_rate=sum(run.evaluation.account_failed for run in runs)
        / simulations,
        funded_account_failure_rate=sum(
            run.funded is not None and run.funded.account_failed for run in runs
        )
        / simulations,
        internal_safety_lock_rate=sum(run.internal_safety_locked for run in runs)
        / simulations,
        payout_probability=sum(run.payout_eligible for run in runs) / simulations,
        median_days_to_pass=float(median(pass_days)) if pass_days else None,
        median_days_to_payout=float(median(payout_days)) if payout_days else None,
        average_drawdown_usd=fmean(drawdowns),
        maximum_drawdown_usd=max(drawdowns),
        expected_lifetime_sessions=fmean(run.account_lifetime_sessions for run in runs),
        expected_profit_per_journey_usd=fmean(run.combined_net_profit_usd for run in runs),
        sampled_source_indices=tuple(paths),
        limitations=JOURNEY_MONTE_CARLO_LIMITATIONS,
        runs=tuple(runs),
    )


__all__ = [
    "JOURNEY_MONTE_CARLO_LIMITATIONS",
    "JourneyMonteCarloReport",
    "run_journey_session_bootstrap",
]
