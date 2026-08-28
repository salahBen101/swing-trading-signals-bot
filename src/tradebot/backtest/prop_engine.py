"""Historical Stage-0/1 runs through the complete three-layer prop-risk path.

This module does not replace the normal backtest engine.  It constructs the exact
StrategyGate -> personal RiskEngine -> selected PropFirmRiskGate coordinator, wraps the
simulated venue in GuardedBroker, and then hands those components to the shared
``ExecutionEngine`` event loop.

Market time and rule-review time are intentionally separate.  Historical bar timestamps
drive strategy/session checks; ``rule_verification_as_of`` drives profile freshness.  An
explicit exchange-calendar provider (or complete date map) is mandatory, and missing or
invalid dates resolve to UNKNOWN rather than being guessed to be regular sessions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..broker.costs import CostModel
from ..broker.guarded import GuardedBroker
from ..broker.simulated import SimulatedBroker
from ..config import PROJECT_ROOT, Config
from ..core.clock import SimulatedClock
from ..core.models import Bar, Rejection, Trade
from ..core.types import TradingMode
from ..data.splits import Split, slice_split
from ..deployment.stages import DeploymentStage
from ..execution.engine import ExecutionEngine
from ..features.pipeline import FeatureFrame, build_features
from ..instruments.registry import InstrumentSpec, get_instrument
from ..journal.db import Journal
from ..prop_firms import load_prop_profile
from ..prop_firms.models import AccountRuleSet, PropFirmProfile
from ..risk.coordinator import (
    ContractKind,
    RuntimeRiskContext,
    ThreeLayerDecision,
    ThreeLayerRiskEngine,
)
from ..risk.killswitch import KillSwitch
from ..risk.limits import RiskEngine
from ..risk.prop import (
    MarketDayStatus,
    PropAccountState,
    close_prop_session,
    initial_prop_account_state,
    reconcile_prop_account,
    start_prop_session,
)
from ..risk.strategy_gate import StrategyGate
from ..strategy.base import Strategy
from .engine import (
    BacktestResult,
    _align_precomputed_features,
    _drive_prepared_backtest,
    _scratch_flag,
)


MarketDayStatusProvider = Callable[[datetime], MarketDayStatus]
MarketDayStatusSource = MarketDayStatusProvider | Mapping[date, MarketDayStatus]


@dataclass(frozen=True, slots=True)
class PropStatePoint:
    timestamp: datetime
    kind: str
    state: PropAccountState


@dataclass(frozen=True, slots=True)
class PropBacktestResult:
    """Normal backtest output plus prop-account state and token-free gate traces."""

    backtest: BacktestResult
    deployment_stage: DeploymentStage
    profile_id: str
    rule_set_name: str
    profile_path: str
    contract_kind: ContractKind
    rule_verification_as_of: datetime
    final_prop_state: PropAccountState
    prop_state_points: tuple[PropStatePoint, ...]
    decision_traces: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]

    @property
    def trades(self) -> list[Trade]:
        return self.backtest.trades

    @property
    def rejections(self) -> list[Rejection]:
        return self.backtest.rejections

    @property
    def equity_curve(self) -> pd.Series:
        return self.backtest.equity_curve

    @property
    def split(self) -> str:
        return self.backtest.split

    @property
    def final_equity(self) -> float:
        return self.backtest.final_equity


class _TracingThreeLayerRiskEngine(ThreeLayerRiskEngine):
    """Capture audit-safe decisions while retaining the coordinator's exact behavior."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trace_payloads: list[dict[str, Any]] = []

    def evaluate_entry(self, *args, **kwargs):
        decision = super().evaluate_entry(*args, **kwargs)
        self._capture(decision, kwargs.get("now"))
        return decision

    def verify(self, *args, **kwargs):
        rejection = super().verify(*args, **kwargs)
        decision = self.last_verify_decision
        if decision is not None:
            self._capture(decision, kwargs.get("now"))
        return rejection

    def _capture(
        self, decision: ThreeLayerDecision, captured_at: datetime | None
    ) -> None:
        payload = decision.to_dict()
        payload["captured_at"] = (captured_at or self.clock.now()).isoformat()
        self.trace_payloads.append(deepcopy(payload))


class _PropBacktestRuntime:
    """Reconcile one immutable prop state from the simulator's broker truth."""

    def __init__(
        self,
        *,
        profile: PropFirmProfile,
        rules: AccountRuleSet,
        broker: SimulatedBroker,
        personal_risk: RiskEngine,
        strategy: Strategy,
        contract_kind: ContractKind,
        market_days: MarketDayStatusSource,
        rule_verification_as_of: datetime,
        internal_safety_buffer_usd: float,
    ) -> None:
        self.profile = profile
        self.rules = rules
        self.broker = broker
        self.personal_risk = personal_risk
        self.strategy = strategy
        self.contract_kind = contract_kind
        self.market_days = market_days
        self.rule_verification_as_of = rule_verification_as_of
        self.state = initial_prop_account_state(
            profile,
            rules,
            internal_safety_buffer_usd=internal_safety_buffer_usd,
        )
        self.points: list[PropStatePoint] = []
        self._market_data_timestamp: datetime | None = None
        self._session_date: date | None = None
        self._session_start_balance_usd = rules.starting_balance_usd
        self._session_start_gross_realized_usd = 0.0

    def before_bar(self, bar: Bar) -> None:
        if self._session_date is None:
            self._begin_session(bar.timestamp.date())
        elif self._session_date != bar.timestamp.date():
            raise RuntimeError("prop session boundary hook did not run before the next bar")
        self._market_data_timestamp = bar.timestamp

    def after_bar(self, bar: Bar) -> None:
        self._reconcile()
        self.points.append(PropStatePoint(bar.timestamp, "MARK", self.state))

    def end_session(self, at: datetime, final: bool) -> None:
        self._reconcile()
        self.state = close_prop_session(self.state, self.rules)
        self.points.append(PropStatePoint(at, "EOD", self.state))
        if not final:
            self.state = start_prop_session(self.state, self.rules)
            self._session_date = None

    def context(self, now: datetime) -> RuntimeRiskContext:
        # This read is the Layer-3 account-state input. GuardedBroker independently takes
        # another fresh account/orders/positions snapshot immediately before submission.
        self._reconcile()
        session_permitted, _, _ = self.personal_risk.session.may_enter(now)
        return RuntimeRiskContext(
            prop_state=self.state,
            market_data_timestamp=self._market_data_timestamp,
            strategy_permitted=self.strategy.spec.trading_hours.may_enter(now),
            session_permitted=session_permitted,
            market_day_status=self._market_day_status(now),
            rule_verification_as_of=self.rule_verification_as_of,
            stage_authorization=None,
        )

    def _begin_session(self, session_date: date) -> None:
        account = self.broker.get_account()
        self._session_date = session_date
        self._session_start_balance_usd = account.cash + account.realized_pnl
        self._session_start_gross_realized_usd = account.realized_pnl

    def _reconcile(self) -> None:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        balance = account.cash + account.realized_pnl
        daily_net_realized = balance - self._session_start_balance_usd
        daily_gross_realized = (
            account.realized_pnl - self._session_start_gross_realized_usd
        )
        open_contracts = sum(abs(position.quantity) for position in positions)
        self.state = reconcile_prop_account(
            self.state,
            self.rules,
            current_balance_usd=balance,
            real_time_net_liquidation_usd=account.equity,
            daily_realized_pnl_usd=daily_net_realized,
            daily_unrealized_pnl_usd=account.unrealized_pnl,
            daily_consistency_profit_usd=(
                daily_gross_realized
                if self.rules.consistency.profit_excludes_commissions
                else daily_net_realized
            ),
            open_minis=(
                open_contracts if self.contract_kind is ContractKind.MINI else 0
            ),
            open_micros=(
                open_contracts if self.contract_kind is ContractKind.MICRO else 0
            ),
            trades_this_session=self.personal_risk.state.trades_today,
        )

    def _market_day_status(self, now: datetime) -> MarketDayStatus:
        try:
            if isinstance(self.market_days, Mapping):
                value = self.market_days.get(now.date(), MarketDayStatus.UNKNOWN)
            else:
                value = self.market_days(now)
            return MarketDayStatus(value)
        except Exception:
            return MarketDayStatus.UNKNOWN


def _run_guarded_prop_history(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: Config,
    *,
    deployment_stage: DeploymentStage,
    rule_verification_as_of: datetime,
    market_day_status_provider: MarketDayStatusSource,
    minimum_expected_rr: float,
    split: Split | str = Split.DEV,
    max_signal_age: timedelta = timedelta(0),
    max_data_age: timedelta = timedelta(0),
    journal: Journal | None = None,
    stress_costs: bool = False,
    features: FeatureFrame | None = None,
    dataset_hash: str = "",
    progress_every: int = 0,
) -> PropBacktestResult:
    """Run approved historical stages through the authoritative guarded stack.

    ``market_day_status_provider`` has no permissive default. A mapping entry missing for
    a signal date resolves to ``UNKNOWN`` and is refused by the prop gate. HOLDOUT/ALL use
    the same loud split warnings as :func:`run_backtest`.
    """

    if (
        rule_verification_as_of.tzinfo is None
        or rule_verification_as_of.utcoffset() is None
    ):
        raise ValueError("rule_verification_as_of must be timezone-aware")
    if not callable(market_day_status_provider) and not isinstance(
        market_day_status_provider, Mapping
    ):
        raise TypeError("market_day_status_provider must be a callable or date mapping")

    if deployment_stage not in {
        DeploymentStage.BACKTEST,
        DeploymentStage.MARKET_REPLAY,
    }:
        raise ValueError("historical prop runner permits only Stage 0 or Stage 1")
    config.validate()
    expected_mode = (
        TradingMode.BACKTEST
        if deployment_stage is DeploymentStage.BACKTEST
        else TradingMode.PAPER
    )
    if (
        config.mode is not expected_mode
        or config.deployment.stage != int(deployment_stage)
    ):
        raise ValueError(
            f"runner requires {expected_mode.value} mode and Stage "
            f"{int(deployment_stage)} configuration"
        )

    profile_path = Path(config.prop_firm.profile_path)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    profile_path = profile_path.resolve()
    profile = load_prop_profile(profile_path)
    rules = profile.rules_for(config.prop_firm.phase)
    _validate_personal_policy(config, rules)

    instrument = get_instrument(config.instrument)
    contract_kind = _contract_kind(instrument)
    selected_split = split if isinstance(split, Split) else Split(str(split).strip().lower())
    sliced = slice_split(bars, selected_split)
    if sliced.empty:
        raise ValueError(f"split {selected_split.value!r} contains no bars")
    selected_features = (
        _align_precomputed_features(features, sliced, strategy)
        if features is not None
        else build_features(sliced, strategy.features)
    )

    clock = SimulatedClock(sliced.index[0].to_pydatetime())
    costs = CostModel.from_config(config.costs, instrument, stress=stress_costs)
    kill_switch = KillSwitch(
        _scratch_flag(config),
        max_consecutive_errors=config.risk.kill_switch.max_consecutive_errors,
        clock=clock,
    )
    kill_switch.clear()
    personal = RiskEngine(
        config.risk,
        instrument,
        config.session,
        clock=clock,
        kill_switch=kill_switch,
        starting_equity=rules.starting_balance_usd,
        # The versioned prop state below owns the actual firm floor. The generic percent
        # research circuit breaker is deliberately not allowed to impersonate it.
        enforce_drawdown_floor=False,
        cost_config=config.costs,
    )
    broker = SimulatedBroker(
        instrument,
        costs,
        config=config.broker.simulated,
        clock=clock,
        starting_equity=rules.starting_balance_usd,
        account_id=f"STAGE{int(deployment_stage)}:{profile.profile_id}:{rules.name}",
    )
    broker.connect()
    runtime = _PropBacktestRuntime(
        profile=profile,
        rules=rules,
        broker=broker,
        personal_risk=personal,
        strategy=strategy,
        contract_kind=contract_kind,
        market_days=market_day_status_provider,
        rule_verification_as_of=rule_verification_as_of,
        internal_safety_buffer_usd=config.prop_firm.internal_safety_buffer_usd,
    )
    coordinator = _TracingThreeLayerRiskEngine(
        strategy_gate=StrategyGate(
            instrument,
            minimum_expected_rr=minimum_expected_rr,
            max_signal_age=max_signal_age,
            max_data_age=max_data_age,
        ),
        personal_risk=personal,
        profile=profile,
        rules=rules,
        deployment_stage=deployment_stage,
        contract_kind=contract_kind,
        context_provider=runtime.context,
    )
    guarded = GuardedBroker(broker, coordinator)
    engine = ExecutionEngine(
        strategy,
        coordinator,
        guarded,
        instrument,
        costs,
        journal=journal,
        starting_equity=rules.starting_balance_usd,
        log_every_bar=config.journal.log_every_bar,
    )

    limitations = (
        f"Stage {int(deployment_stage)} simulated history only; this result cannot "
        "authorize Stage 2, Stage 3, or Stage 4.",
        "Open-position personal and prop thresholds are fail-safe flatten triggers, but "
        "this local simulated route is not an external deployment route.",
    )
    backtest = _drive_prepared_backtest(
        sliced=sliced,
        features=selected_features,
        strategy=strategy,
        config=config,
        selected_split=selected_split,
        instrument=instrument,
        costs=costs,
        clock=clock,
        risk=coordinator,
        guarded=guarded,
        engine=engine,
        stress_costs=stress_costs,
        enforce_drawdown_floor=False,
        dataset_hash=dataset_hash,
        progress_every=progress_every,
        before_bar=runtime.before_bar,
        after_bar=runtime.after_bar,
        on_session_end=runtime.end_session,
        additional_notes=(
            f"three-layer Stage-{int(deployment_stage)} prop path: "
            f"{profile.profile_id}/{rules.name}",
            f"versioned prop floor plus ${config.prop_firm.internal_safety_buffer_usd:,.2f} "
            "internal buffer enforced for every entry; legacy percent floor disabled",
            *limitations,
        ),
        describe_legacy_floor=False,
    )

    return PropBacktestResult(
        backtest=backtest,
        deployment_stage=deployment_stage,
        profile_id=profile.profile_id,
        rule_set_name=rules.name,
        profile_path=str(profile_path),
        contract_kind=contract_kind,
        rule_verification_as_of=rule_verification_as_of,
        final_prop_state=runtime.state,
        prop_state_points=tuple(runtime.points),
        decision_traces=tuple(deepcopy(coordinator.trace_payloads)),
        limitations=limitations,
    )


def run_prop_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: Config,
    *,
    rule_verification_as_of: datetime,
    market_day_status_provider: MarketDayStatusSource,
    minimum_expected_rr: float,
    split: Split | str = Split.DEV,
    max_signal_age: timedelta = timedelta(0),
    max_data_age: timedelta = timedelta(0),
    journal: Journal | None = None,
    stress_costs: bool = False,
    features: FeatureFrame | None = None,
    dataset_hash: str = "",
    progress_every: int = 0,
) -> PropBacktestResult:
    """Run the DEV-default Stage-0 research path."""

    return _run_guarded_prop_history(
        bars,
        strategy,
        config,
        deployment_stage=DeploymentStage.BACKTEST,
        rule_verification_as_of=rule_verification_as_of,
        market_day_status_provider=market_day_status_provider,
        minimum_expected_rr=minimum_expected_rr,
        split=split,
        max_signal_age=max_signal_age,
        max_data_age=max_data_age,
        journal=journal,
        stress_costs=stress_costs,
        features=features,
        dataset_hash=dataset_hash,
        progress_every=progress_every,
    )


def run_prop_market_replay(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: Config,
    *,
    rule_verification_as_of: datetime,
    market_day_statuses: Mapping[date, MarketDayStatus],
    minimum_expected_rr: float,
    max_signal_age: timedelta = timedelta(0),
    max_data_age: timedelta = timedelta(0),
    journal: Journal | None = None,
    stress_costs: bool = False,
    features: FeatureFrame | None = None,
    dataset_hash: str = "",
    progress_every: int = 0,
) -> PropBacktestResult:
    """Run Stage 1 against DEV only; no split selector can expose HOLDOUT."""

    return _run_guarded_prop_history(
        bars,
        strategy,
        config,
        deployment_stage=DeploymentStage.MARKET_REPLAY,
        rule_verification_as_of=rule_verification_as_of,
        market_day_status_provider=market_day_statuses,
        minimum_expected_rr=minimum_expected_rr,
        split=Split.DEV,
        max_signal_age=max_signal_age,
        max_data_age=max_data_age,
        journal=journal,
        stress_costs=stress_costs,
        features=features,
        dataset_hash=dataset_hash,
        progress_every=progress_every,
    )


def _contract_kind(instrument: InstrumentSpec) -> ContractKind:
    if instrument.symbol in {"MNQ", "MES"}:
        return ContractKind.MICRO
    if instrument.symbol in {"NQ", "ES"}:
        return ContractKind.MINI
    raise ValueError(
        f"instrument {instrument.symbol!r} has no explicit prop contract-kind mapping"
    )


def _validate_personal_policy(config: Config, rules: AccountRuleSet) -> None:
    risk = config.risk
    violations: list[str] = []
    if risk.per_trade.max_risk_per_trade_usd > 200.0:
        violations.append("max_risk_per_trade_usd must be <= 200")
    if risk.daily.max_trades_per_day > 1:
        violations.append("max_trades_per_day must be <= 1")
    if risk.daily.max_daily_loss_usd > 200.0:
        violations.append("max_daily_loss_usd must be <= 200")
    if risk.max_open_positions > 1:
        violations.append("max_open_positions must be <= 1")
    if abs(risk.starting_equity_usd - rules.starting_balance_usd) > 1e-9:
        violations.append(
            "risk.starting_equity_usd must equal the selected prop phase starting balance"
        )
    if violations:
        raise ValueError("unsafe guarded personal policy: " + "; ".join(violations))


__all__ = [
    "MarketDayStatusProvider",
    "MarketDayStatusSource",
    "PropBacktestResult",
    "PropStatePoint",
    "run_prop_backtest",
    "run_prop_market_replay",
]
