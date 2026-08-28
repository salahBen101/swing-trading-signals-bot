"""M13: structural composition and broker-side ordering of all three risk layers."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import tradebot.risk.coordinator as coordinator_module
import tradebot.deployment.stages as stages_module
from tradebot.config import KillSwitchConfig, RiskConfig, SessionConfig
from tradebot.core.clock import SimulatedClock
from tradebot.core.models import Order, OrderIntent, Position
from tradebot.core.types import OrderPurpose, OrderType, RejectReason, Side
from tradebot.deployment.stages import (
    ApprovalManifest,
    DeploymentStage,
    StageAuthorization,
    artifact_sha256,
    authorize_stage,
)
from tradebot.instruments.registry import get_instrument
from tradebot.prop_firms import (
    RuleVerificationResult,
    SourceCheck,
    VerificationStatus,
    load_prop_profile,
)
from tradebot.risk.coordinator import (
    ContractKind,
    RuntimeRiskContext,
    ThreeLayerRiskEngine,
    canonical_risk_state_context_id,
)
from tradebot.risk.broker_state import (
    AuthoritativeBrokerSnapshot,
    BrokerRiskAccount,
    BrokerRiskPosition,
)
from tradebot.risk.limits import RiskEngine
from tradebot.risk.prop import (
    MarketDayStatus,
    PropGateReason,
    initial_prop_account_state,
)
from tradebot.risk.strategy_gate import StrategyGate
from tradebot.risk.state_store import FileRiskStateStore


ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
CHECKED = date(2026, 8, 20)
NOW = datetime(2026, 8, 20, 10, 0, tzinfo=ET)
MNQ = get_instrument("MNQ")
PROFILE_PATH = ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
SNAPSHOT_PATH = ROOT / "config" / "prop_firms" / "source_snapshots.yaml"
CONFIG_PATH = ROOT / "config" / "tradebot.yaml"
STRATEGY_PATH = ROOT / "src" / "tradebot" / "strategy" / "orb_breakout.py"
TEST_REVISION = "0" * 40
TEST_RUNTIME_CONFIG_SHA256 = "1" * 64


def broker_snapshot(ts=NOW, *, positions=()):
    return AuthoritativeBrokerSnapshot(
        read_started_at=ts,
        captured_at=ts,
        broker_name="coordinator-test-broker",
        broker_is_paper=True,
        execution_route="coordinator-test-broker paper",
        account=BrokerRiskAccount(
            account_id="COORDINATOR-50K",
            equity=50_000.0,
            cash=50_000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            currency="USD",
            is_paper=True,
        ),
        positions=positions,
        orders=(),
    )


def fresh_profile():
    profile = load_prop_profile(
        ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
    )
    return replace(
        profile,
        verified_on=CHECKED,
        reverify_after_hours=72,
        sources=tuple(replace(source, checked_on=CHECKED) for source in profile.sources),
        ambiguity_notes=(),
    )


def signed_stage_authorization(
    *,
    stage: DeploymentStage = DeploymentStage.PROP_EVALUATION,
    account_id: str = "COORDINATOR-50K",
    phase: str = "evaluation",
) -> StageAuthorization:
    """Mint the same authenticated capability the real startup gate must supply."""
    profile = fresh_profile()
    digest = "a" * 64
    verification = RuleVerificationResult(
        profile_id=profile.profile_id,
        checked_at=NOW,
        status=VerificationStatus.UNCHANGED,
        checks=tuple(
            SourceCheck(source.url, digest, digest, changed=False)
            for source in profile.sources
        ),
    )
    approval = ApprovalManifest(
        schema_version=1,
        stage=stage,
        approved=True,
        approved_by="coordinator-test-human",
        approved_at=NOW,
        account_id=account_id,
        prop_profile_id=profile.profile_id,
        prop_phase=phase,
        prop_profile_sha256=artifact_sha256(PROFILE_PATH),
        source_snapshot_sha256=artifact_sha256(SNAPSHOT_PATH),
        runtime_config_sha256=artifact_sha256(CONFIG_PATH),
        strategy_artifact_sha256=artifact_sha256(STRATEGY_PATH),
        code_revision=TEST_REVISION,
        execution_route="coordinator-test-broker paper",
        execution_route_verified=True,
        sole_owner_attested=True,
        firm_exclusive_use_attested=True,
        production_frozen=True,
    )
    original = stages_module._read_authorization_environment
    original_profile_loader = stages_module._load_authorization_profile
    original_verifier = stages_module._run_authorization_rule_verification
    stages_module._read_authorization_environment = lambda: (
        stages_module._AuthorizationEnvironment(NOW, TEST_REVISION, True)
    )
    stages_module._load_authorization_profile = lambda path: profile
    stages_module._run_authorization_rule_verification = (
        lambda loaded, path, *, checked_at: verification
    )
    try:
        authorization = authorize_stage(
            stage,
            as_of=NOW,
            profile_path=PROFILE_PATH,
            source_snapshot_path=SNAPSHOT_PATH,
            runtime_config_path=CONFIG_PATH,
            strategy_artifact_path=STRATEGY_PATH,
            manifest=approval,
        )
    finally:
        stages_module._read_authorization_environment = original
        stages_module._load_authorization_profile = original_profile_loader
        stages_module._run_authorization_rule_verification = original_verifier
    assert authorization.allowed, authorization.reason
    return authorization


def valid_intent(**changes) -> OrderIntent:
    values = {
        "timestamp": NOW,
        "instrument": "MNQ",
        "side": Side.BUY,
        "strategy": "deterministic_pullback",
        "reference_price": 18_000.0,
        "stop_price": 17_990.0,
        # The signed entry can be four ticks adverse (18,001). This target preserves
        # exactly 2R at that executable bound: (18,023-18,001)/(18,001-17,990).
        "target_price": 18_023.0,
        "conditions": ("trend_up", "pullback_complete"),
    }
    values.update(changes)
    return OrderIntent(**values)


class ContextSource:
    def __init__(self, context):
        self.context = context
        self.calls: list[datetime] = []
        self.error: Exception | None = None

    def __call__(self, now: datetime):
        self.calls.append(now)
        if self.error is not None:
            raise self.error
        return self.context


class RecordingStrategyGate(StrategyGate):
    def __init__(self, events: list[str] | None = None):
        super().__init__(
            MNQ,
            minimum_expected_rr=2.0,
            max_signal_age=timedelta(seconds=60),
            max_data_age=timedelta(seconds=10),
        )
        self.events = events if events is not None else []
        self.calls = 0

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        self.events.append("strategy")
        return super().evaluate(*args, **kwargs)


class RecordingRiskEngine(RiskEngine):
    def __init__(self, *args, events: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events if events is not None else []
        self.entry_calls = 0
        self.verify_calls = 0

    def evaluate_entry(self, *args, **kwargs):
        self.entry_calls += 1
        self.events.append("personal_evaluate")
        return super().evaluate_entry(*args, **kwargs)

    def verify(self, *args, **kwargs):
        self.verify_calls += 1
        self.events.append("personal_verify")
        return super().verify(*args, **kwargs)


@pytest.fixture
def system_factory(tmp_path):
    created = []

    def make(
        *,
        context=None,
        provider=None,
        stage=DeploymentStage.PAPER,
        contract_kind=ContractKind.MICRO,
        rules_name="evaluation",
        events=None,
        risk_state_store=None,
        bootstrap_risk_state_store=False,
    ):
        profile = fresh_profile()
        rules = profile.rules_for(rules_name)
        state = initial_prop_account_state(profile, rules)
        authorization = (
            signed_stage_authorization(stage=stage, phase=rules.name)
            if stage.requires_human_approval
            else StageAuthorization(stage, True, ())
        )
        if context is None:
            context = RuntimeRiskContext(
                prop_state=state,
                market_data_timestamp=NOW,
                strategy_permitted=True,
                session_permitted=True,
                market_day_status=MarketDayStatus.REGULAR,
                rule_verification_as_of=NOW,
                stage_authorization=authorization,
            )
        source = provider or ContextSource(context)
        clock = SimulatedClock(NOW)
        strategy = RecordingStrategyGate(events)
        risk_context_id = (
            canonical_risk_state_context_id(
                strategy_gate=strategy,
                profile=profile,
                rules=rules,
                deployment_stage=stage,
                contract_kind=contract_kind,
                runtime_config_sha256=TEST_RUNTIME_CONFIG_SHA256,
            )
            if risk_state_store is not None and stage >= DeploymentStage.PAPER
            else None
        )
        personal = RecordingRiskEngine(
            RiskConfig(
                kill_switch=KillSwitchConfig(
                    flag_file=str(tmp_path / f"kill-{len(created)}.flag")
                )
            ),
            MNQ,
            SessionConfig(),
            clock=clock,
            events=events,
            risk_state_store=risk_state_store,
            bootstrap_risk_state_store=bootstrap_risk_state_store,
            risk_state_context_id=risk_context_id,
        )
        engine = ThreeLayerRiskEngine(
            strategy_gate=strategy,
            personal_risk=personal,
            profile=profile,
            rules=rules,
            deployment_stage=stage,
            contract_kind=contract_kind,
            context_provider=source,
            runtime_config_sha256=TEST_RUNTIME_CONFIG_SHA256,
        )
        bundle = (engine, personal, strategy, source, profile, rules, state)
        created.append(bundle)
        return bundle

    return make


def test_clean_entry_passes_all_layers_and_trace_omits_token(system_factory):
    engine, personal, strategy, source, *_ = system_factory()

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert decision.approved
    assert decision.approval is not None
    assert decision.strategy_trace.allowed
    assert decision.personal_trace.allowed
    assert decision.prop_trace.allowed
    assert personal.entry_calls == 1
    assert strategy.calls == 1
    assert source.calls == [NOW]
    payload = decision.to_dict()
    encoded = json.dumps(payload)
    assert "signature" not in encoded.casefold()
    assert "token" not in encoded.casefold()
    assert payload["approval"]["risk_usd"] == pytest.approx(84.72)
    prop_state = payload["context"]["facts"]["prop_account_state"]
    assert prop_state["current_balance_usd"] == 50_000.0
    assert prop_state["drawdown_high_water_mark_usd"] == 50_000.0
    assert prop_state["drawdown_floor_usd"] == 48_000.0
    assert prop_state["remaining_prop_drawdown_usd"] == 2_000.0
    assert prop_state["daily_pnl_usd"] == 0.0
    assert "payout_status" in prop_state


def test_signal_level_two_r_is_rejected_after_adverse_entry_repricing(system_factory):
    engine, personal, *_ = system_factory()

    decision = engine.evaluate_entry(
        valid_intent(target_price=18_020.0),
        now=NOW,
    )

    assert not decision.allowed
    assert decision.rejection.stage == "STRATEGY"
    assert decision.rejection.reason is RejectReason.INVALID_ORDER
    assert decision.personal_trace.allowed
    assert not decision.prop_trace.evaluated
    assert personal.entry_calls == 1
    trace = decision.strategy_trace.decision
    assert trace.expected_reward_risk == pytest.approx(19.0 / 11.0)
    assert trace.check("entry_price").observed == 18_001.0
    assert trace.failure_codes == ("minimum_expected_rr",)


def test_rejected_decision_retains_complete_prop_account_trace_without_secrets(
    system_factory,
):
    engine, _, _, _, _, _, state = system_factory()

    decision = engine.evaluate_entry(
        valid_intent(timestamp=NOW - timedelta(minutes=5)), now=NOW
    )
    payload = decision.to_dict()
    prop_state = payload["context"]["facts"]["prop_account_state"]

    assert not decision.allowed
    expected = state.to_dict()
    assert set(prop_state) == set(expected)
    assert prop_state["current_balance_usd"] == expected["current_balance_usd"]
    assert prop_state["drawdown_high_water_mark_usd"] == expected[
        "drawdown_high_water_mark_usd"
    ]
    assert prop_state["drawdown_floor_usd"] == expected["drawdown_floor_usd"]
    assert prop_state["remaining_prop_drawdown_usd"] == expected[
        "remaining_prop_drawdown_usd"
    ]
    assert prop_state["daily_pnl_usd"] == expected["daily_pnl_usd"]
    assert prop_state["payout_status"] == expected["payout_status"]
    assert prop_state["hard_breach_reasons"] == list(expected["hard_breach_reasons"])
    encoded = json.dumps(payload).casefold()
    assert "signature" not in encoded
    assert "token" not in encoded


def test_rejected_nonfinite_strategy_fact_remains_strict_json(system_factory):
    engine, *_ = system_factory()
    intent = valid_intent()
    object.__setattr__(intent, "target_price", float("nan"))

    decision = engine.evaluate_entry(intent, now=NOW)

    assert not decision.allowed
    json.dumps(decision.to_dict(), allow_nan=False)


def test_layer_one_blocks_before_personal_or_prop(
    system_factory, monkeypatch
):
    events: list[str] = []
    engine, personal, _, _, *_ = system_factory(events=events)
    prop_calls = 0
    real_prop = coordinator_module.evaluate_prop_order

    def record_prop(*args, **kwargs):
        nonlocal prop_calls
        prop_calls += 1
        return real_prop(*args, **kwargs)

    monkeypatch.setattr(coordinator_module, "evaluate_prop_order", record_prop)

    decision = engine.evaluate_entry(valid_intent(instrument="MES"), now=NOW)

    assert not decision.allowed
    assert decision.rejection.stage == "STRATEGY"
    assert decision.strategy_trace.decision.failure_codes == ("instrument_match",)
    assert not decision.personal_trace.evaluated
    assert not decision.prop_trace.evaluated
    assert personal.entry_calls == 0
    assert prop_calls == 0
    assert events == ["strategy"]


def test_stale_market_data_is_an_independent_layer_one_refusal(system_factory):
    _, _, _, _, _, _, state = system_factory()
    context = RuntimeRiskContext(
        state,
        NOW - timedelta(seconds=10, microseconds=1),
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        StageAuthorization(DeploymentStage.PAPER, True, ()),
    )
    engine, personal, *_ = system_factory(context=context)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert decision.rejection.reason is RejectReason.STALE_MARKET_DATA
    assert "data_fresh" in decision.strategy_trace.decision.failure_codes
    assert personal.entry_calls == 0


def test_layer_two_blocks_before_prop(system_factory, monkeypatch):
    engine, personal, *_ = system_factory()
    personal.roll_session(NOW)
    personal.state.halted = True
    personal.state.halt_reason = "test personal lock"
    prop_calls = 0

    def should_not_run(*args, **kwargs):
        nonlocal prop_calls
        prop_calls += 1
        raise AssertionError("prop gate must be short-circuited")

    monkeypatch.setattr(coordinator_module, "evaluate_prop_order", should_not_run)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert decision.rejection.reason is RejectReason.HALTED_FOR_SESSION
    assert decision.strategy_trace.allowed
    assert decision.personal_trace.evaluated
    assert not decision.personal_trace.allowed
    assert not decision.prop_trace.evaluated
    assert prop_calls == 0


def test_layer_three_hides_personal_approval_when_prop_refuses(system_factory):
    _, _, _, _, _, _, state = system_factory()
    breached = replace(
        state,
        hard_breached=True,
        hard_breach_reasons=("test account breach",),
    )
    context = RuntimeRiskContext(
        breached,
        NOW,
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        StageAuthorization(DeploymentStage.PAPER, True, ()),
    )
    engine, personal, *_ = system_factory(context=context)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert decision.approval is None
    assert decision.personal_trace.allowed
    assert not hasattr(decision.personal_trace, "approval")
    assert PropGateReason.HARD_ACCOUNT_BREACH in decision.prop_trace.decision.reason_codes
    assert decision.rejection.stage == "PROP_FIRM"
    assert personal.entry_calls == 1
    assert "token" not in json.dumps(decision.to_dict()).casefold()


def test_prop_refusal_never_creates_a_durable_personal_entry_reservation(
    system_factory, tmp_path
):
    profile = fresh_profile()
    rules = profile.rules_for("evaluation")
    breached = replace(
        initial_prop_account_state(profile, rules),
        hard_breached=True,
        hard_breach_reasons=("test account breach",),
    )
    context = RuntimeRiskContext(
        breached,
        NOW,
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        StageAuthorization(DeploymentStage.PAPER, True, ()),
    )
    store = FileRiskStateStore(tmp_path / "coordinated-risk.json")
    engine, personal, *_ = system_factory(
        context=context,
        risk_state_store=store,
        bootstrap_risk_state_store=True,
    )
    assert personal.reconcile_broker_snapshot(broker_snapshot(), now=NOW) is None

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert decision.rejection.stage == "PROP_FIRM"
    assert store.load().state.active_entry_order_id is None


def test_durable_entry_is_reserved_only_at_successful_final_three_layer_verify(
    system_factory, tmp_path
):
    store = FileRiskStateStore(tmp_path / "coordinated-risk.json")
    engine, personal, *_ = system_factory(
        risk_state_store=store,
        bootstrap_risk_state_store=True,
    )
    assert personal.reconcile_broker_snapshot(broker_snapshot(), now=NOW) is None
    decision = engine.evaluate_entry(valid_intent(), now=NOW)
    assert decision.approved
    assert store.load().state.active_entry_order_id is None

    rejection = engine.verify(
        decision.approval.order,
        decision.approval.token,
        now=NOW,
        broker_snapshot=broker_snapshot(),
    )

    assert rejection is None
    assert store.load().state.active_entry_order_id == decision.approval.order.order_id


def test_canonical_durable_context_binds_material_runtime_configuration() -> None:
    profile = fresh_profile()
    rules = profile.rules_for("evaluation")
    strategy = RecordingStrategyGate()

    first = canonical_risk_state_context_id(
        strategy_gate=strategy,
        profile=profile,
        rules=rules,
        deployment_stage=DeploymentStage.PAPER,
        contract_kind=ContractKind.MICRO,
        runtime_config_sha256="1" * 64,
    )
    changed = canonical_risk_state_context_id(
        strategy_gate=strategy,
        profile=profile,
        rules=rules,
        deployment_stage=DeploymentStage.PAPER,
        contract_kind=ContractKind.MICRO,
        runtime_config_sha256="2" * 64,
    )

    assert first.startswith("risk-context-v1:")
    assert first != changed


def test_durable_stage2_refuses_free_form_deployment_context(tmp_path) -> None:
    profile = fresh_profile()
    rules = profile.rules_for("evaluation")
    strategy = RecordingStrategyGate()
    clock = SimulatedClock(NOW)
    personal = RiskEngine(
        RiskConfig(
            kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "context.kill"))
        ),
        MNQ,
        SessionConfig(),
        clock=clock,
        risk_state_store=FileRiskStateStore(tmp_path / "context-risk.json"),
        bootstrap_risk_state_store=True,
        risk_state_context_id="caller-chosen-text",
    )
    context = RuntimeRiskContext(
        prop_state=initial_prop_account_state(profile, rules),
        market_data_timestamp=NOW,
        strategy_permitted=True,
        session_permitted=True,
        market_day_status=MarketDayStatus.REGULAR,
        rule_verification_as_of=NOW,
        stage_authorization=StageAuthorization(DeploymentStage.PAPER, True, ()),
    )

    with pytest.raises(ValueError, match="canonical deployment context"):
        ThreeLayerRiskEngine(
            strategy_gate=strategy,
            personal_risk=personal,
            profile=profile,
            rules=rules,
            deployment_stage=DeploymentStage.PAPER,
            contract_kind=ContractKind.MICRO,
            context_provider=ContextSource(context),
            runtime_config_sha256=TEST_RUNTIME_CONFIG_SHA256,
        )


@pytest.mark.parametrize("failure", ["none", "wrong_type", "provider_exception"])
def test_context_provider_failures_are_closed_before_every_layer(
    system_factory, failure
):
    if failure == "provider_exception":
        source = ContextSource(None)
        source.error = RuntimeError("secret connection detail")
    elif failure == "wrong_type":
        source = ContextSource({"prop_state": "not immutable"})
    else:
        source = ContextSource(None)
    engine, personal, strategy, *_ = system_factory(provider=source)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert not decision.context_trace.allowed
    assert strategy.calls == 0
    assert personal.entry_calls == 0
    assert not decision.prop_trace.evaluated
    assert "secret connection detail" not in decision.rejection.detail


def test_mismatched_prop_state_is_a_context_failure(system_factory):
    _, _, _, _, _, _, state = system_factory()
    context = RuntimeRiskContext(
        replace(state, profile_id="some-other-profile"),
        NOW,
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        StageAuthorization(DeploymentStage.PAPER, True, ()),
    )
    engine, personal, strategy, *_ = system_factory(context=context)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert "prop_profile_mismatch" in decision.context_trace.reason_codes
    assert strategy.calls == 0
    assert personal.entry_calls == 0


@pytest.mark.parametrize(
    ("stage", "authorization", "allowed", "reason"),
    [
        (DeploymentStage.PAPER, None, True, None),
        (
            DeploymentStage.PAPER,
            StageAuthorization(DeploymentStage.BACKTEST, True, ()),
            False,
            "stage_authorization_mismatch",
        ),
        (
            DeploymentStage.PAPER,
            StageAuthorization(DeploymentStage.PAPER, False, ("denied",)),
            False,
            "stage_authorization_denied",
        ),
        (DeploymentStage.PROP_EVALUATION, None, False, "stage_authorization_missing"),
        (
            DeploymentStage.PROP_EVALUATION,
            StageAuthorization(DeploymentStage.PAPER, True, ()),
            False,
            "stage_authorization_mismatch",
        ),
        (
            DeploymentStage.PROP_EVALUATION,
            StageAuthorization(DeploymentStage.PROP_EVALUATION, False, ("denied",)),
            False,
            "stage_authorization_denied",
        ),
        (
            DeploymentStage.PROP_EVALUATION,
            StageAuthorization(
                DeploymentStage.PROP_EVALUATION,
                True,
                (),
                account_id="COORDINATOR-50K",
            ),
            False,
            "stage_authorization_unauthenticated",
        ),
        (
            DeploymentStage.PROP_EVALUATION,
            signed_stage_authorization(),
            True,
            None,
        ),
    ],
)
def test_stage_authorization_is_typed_bound_and_mandatory_for_live_stages(
    system_factory, stage, authorization, allowed, reason
):
    _, _, _, _, profile, rules, state = system_factory()
    context = RuntimeRiskContext(
        state,
        NOW,
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        authorization,
    )
    # Rebind state to the separately loaded fixed objects used by this coordinator.
    state = initial_prop_account_state(profile, rules)
    context = replace(context, prop_state=state)
    engine, *_ = system_factory(context=context, stage=stage)

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert decision.allowed is allowed
    if reason is not None:
        assert reason in decision.context_trace.reason_codes


def test_bare_boolean_cannot_impersonate_stage_authorization(system_factory):
    _, _, _, _, _, _, state = system_factory()
    context = RuntimeRiskContext(
        state,
        NOW,
        True,
        True,
        MarketDayStatus.REGULAR,
        NOW,
        True,  # type: ignore[arg-type] -- adversarial runtime value
    )
    engine, *_ = system_factory(
        context=context, stage=DeploymentStage.PROP_EVALUATION
    )

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert "stage_authorization_invalid" in decision.context_trace.reason_codes


def test_live_stage_authorization_without_account_identity_fails_before_sizing(
    system_factory,
):
    engine, personal, strategy, source, *_ = system_factory(
        stage=DeploymentStage.PROP_EVALUATION
    )
    source.context = replace(
        source.context,
        stage_authorization=StageAuthorization(
            DeploymentStage.PROP_EVALUATION, True, ()
        ),
    )

    decision = engine.evaluate_entry(valid_intent(), now=NOW)

    assert not decision.allowed
    assert "stage_authorized_account_missing" in decision.context_trace.reason_codes
    assert decision.rejection.reason is RejectReason.LIVE_TRADING_DISABLED
    assert strategy.calls == 0
    assert personal.entry_calls == 0


def test_live_stage_broker_account_mismatch_rejects_before_token_verification(
    system_factory,
):
    engine, personal, *_ = system_factory(stage=DeploymentStage.PROP_EVALUATION)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0
    wrong_snapshot = broker_snapshot()
    wrong_snapshot = replace(
        wrong_snapshot,
        account=replace(wrong_snapshot.account, account_id="SOME-OTHER-ACCOUNT"),
    )

    rejection = engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=wrong_snapshot,
    )

    assert rejection is not None
    assert rejection.reason is RejectReason.LIVE_TRADING_DISABLED
    assert rejection.stage == "BROKER_IDENTITY"
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)
    assert "SOME-OTHER-ACCOUNT" not in rejection.detail
    assert "COORDINATOR-50K" not in rejection.detail


def test_live_stage_missing_snapshot_rejects_before_token_verification(system_factory):
    engine, personal, *_ = system_factory(stage=DeploymentStage.PROP_EVALUATION)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0

    rejection = engine.verify(approval.order, approval.token, now=NOW)

    assert rejection is not None
    assert rejection.reason is RejectReason.BROKER_ERROR
    assert rejection.stage == "BROKER_IDENTITY"
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)


def test_live_stage_authorized_account_cannot_change_after_approval(system_factory):
    engine, personal, _, source, *_ = system_factory(
        stage=DeploymentStage.PROP_EVALUATION
    )
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0
    source.context = replace(
        source.context,
        stage_authorization=signed_stage_authorization(
            account_id="CHANGED-AUTHORIZED-ACCOUNT"
        ),
    )
    changed_snapshot = broker_snapshot()
    changed_snapshot = replace(
        changed_snapshot,
        account=replace(
            changed_snapshot.account,
            account_id="CHANGED-AUTHORIZED-ACCOUNT",
        ),
    )

    rejection = engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=changed_snapshot,
    )

    assert rejection is not None
    assert rejection.stage == "BROKER_IDENTITY"
    assert "account identity changed after entry approval" in rejection.detail
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)


def test_live_stage_capability_cannot_rotate_after_entry_approval(system_factory):
    engine, personal, _, source, *_ = system_factory(
        stage=DeploymentStage.PROP_EVALUATION
    )
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0
    # Same approved account/profile/route, but a separately minted startup capability.
    source.context = replace(
        source.context,
        stage_authorization=signed_stage_authorization(),
    )

    rejection = engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=broker_snapshot(),
    )

    assert rejection is not None
    assert rejection.stage == "BROKER_IDENTITY"
    assert "capability changed after entry approval" in rejection.detail
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)


def test_live_stage_authorized_route_must_match_broker_snapshot(system_factory):
    engine, personal, *_ = system_factory(stage=DeploymentStage.PROP_EVALUATION)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0
    wrong_route = replace(broker_snapshot(), execution_route="different-approved-route")

    rejection = engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=wrong_route,
    )

    assert rejection is not None
    assert rejection.stage == "BROKER_IDENTITY"
    assert "broker route" in rejection.detail
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)


def test_live_stage_exact_authorized_broker_account_reaches_personal_verification(
    system_factory,
):
    engine, personal, *_ = system_factory(stage=DeploymentStage.PROP_EVALUATION)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0

    rejection = engine.verify(
        approval.order,
        approval.token,
        now=NOW,
        broker_snapshot=broker_snapshot(),
    )

    assert rejection is None
    assert personal.verify_calls == 1
    assert personal._tokens.is_spent(approval.token)


def test_verify_refreshes_context_and_prop_change_blocks_before_token_spend(
    system_factory
):
    engine, personal, _, source, _, _, state = system_factory()
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    assert approval is not None
    personal.verify_calls = 0
    source.context = replace(
        source.context,
        prop_state=replace(
            state,
            hard_breached=True,
            hard_breach_reasons=("breached after approval",),
        ),
    )

    rejection = engine.verify(
        approval.order, approval.token, now=NOW, broker_snapshot=broker_snapshot()
    )

    assert rejection is not None
    assert rejection.reason is RejectReason.TRAILING_DRAWDOWN
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(approval.token)
    assert source.calls == [NOW, NOW]
    assert not engine.last_verify_decision.prop_trace.allowed


def test_verify_order_is_strategy_then_prop_then_personal(
    system_factory, monkeypatch
):
    events: list[str] = []
    engine, personal, _, _, *_ = system_factory(events=events)
    real_prop = coordinator_module.evaluate_prop_order

    def record_prop(*args, **kwargs):
        events.append("prop")
        return real_prop(*args, **kwargs)

    monkeypatch.setattr(coordinator_module, "evaluate_prop_order", record_prop)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval
    assert events == ["strategy", "personal_evaluate", "prop"]
    events.clear()

    rejection = engine.verify(
        approval.order, approval.token, now=NOW, broker_snapshot=broker_snapshot()
    )

    assert rejection is None
    assert events == ["strategy", "prop", "personal_verify"]
    assert personal._tokens.is_spent(approval.token)


def test_verify_uses_stored_deep_copy_of_exact_intent(system_factory):
    engine, _, *_ = system_factory()
    intent = valid_intent()
    approval = engine.evaluate_entry(intent, now=NOW).approval
    object.__setattr__(intent, "timestamp", NOW - timedelta(days=1))
    object.__setattr__(intent, "instrument", "MES")

    assert engine.verify(
        approval.order, approval.token, now=NOW, broker_snapshot=broker_snapshot()
    ) is None
    assert engine.last_verify_decision.strategy_trace.allowed


def test_unknown_entry_is_rejected_without_personal_verification(system_factory):
    engine, personal, *_ = system_factory()
    direct = personal.evaluate_entry(valid_intent(), now=NOW).approval
    personal.verify_calls = 0

    rejection = engine.verify(direct.order, direct.token, now=NOW)

    assert rejection.reason is RejectReason.INVALID_TOKEN
    assert rejection.stage == "COORDINATOR"
    assert personal.verify_calls == 0
    assert not personal._tokens.is_spent(direct.token)


def test_risk_reducing_exit_bypasses_broken_entry_context(
    system_factory, position_factory
):
    source = ContextSource(None)
    source.error = RuntimeError("account feed unavailable")
    engine, personal, *_ = system_factory(provider=source)
    position = position_factory(quantity=2)
    exit_approval = engine.evaluate_exit(position, now=NOW).approval
    personal.kill_switch.trip("force flatten")

    rejection = engine.verify(
        exit_approval.order,
        exit_approval.token,
        now=NOW,
        broker_snapshot=broker_snapshot(
            positions=(
                BrokerRiskPosition(
                    position.instrument,
                    position.quantity,
                    position.entry_price,
                ),
            )
        ),
    )

    assert rejection is None
    assert source.calls == []
    assert personal.verify_calls == 1
    assert engine.last_verify_decision.operation == "RISK_REDUCING_VERIFY"
    assert not engine.last_verify_decision.context_trace.evaluated


def test_protective_orders_delegate_without_entry_prop_gate(
    system_factory, position_factory
):
    source = ContextSource(None)
    source.error = RuntimeError("account feed unavailable")
    engine, personal, *_ = system_factory(provider=source)
    position = position_factory(target=18_020.0)
    order = Order(
        order_id="protective-stop",
        timestamp=NOW,
        instrument="MNQ",
        side=Side.SELL,
        quantity=1,
        order_type=OrderType.STOP,
        stop_price=17_990.0,
        strategy=position.strategy,
        purpose=OrderPurpose.STOP,
        oco_group="oco-test",
    )
    approval = engine.evaluate_protective(position, order, now=NOW).approval

    assert engine.verify(
        order,
        approval.token,
        now=NOW,
        broker_snapshot=broker_snapshot(
            positions=(
                BrokerRiskPosition(
                    position.instrument,
                    position.quantity,
                    position.entry_price,
                ),
            )
        ),
    ) is None
    assert personal.verify_calls == 1
    assert source.calls == []


@pytest.mark.parametrize(
    ("kind", "expected_minis", "expected_micros"),
    [(ContractKind.MINI, 3, 0), (ContractKind.MICRO, 0, 3)],
)
def test_prop_receives_exact_sized_quantity_risk_and_contract_kind(
    system_factory, monkeypatch, kind, expected_minis, expected_micros
):
    requests = []
    real_prop = coordinator_module.evaluate_prop_order

    def capture(request, **kwargs):
        requests.append(request)
        return real_prop(request, **kwargs)

    monkeypatch.setattr(coordinator_module, "evaluate_prop_order", capture)
    engine, *_ = system_factory(contract_kind=kind)
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval

    assert approval is not None
    assert requests[-1].requested_minis == expected_minis
    assert requests[-1].requested_micros == expected_micros
    assert requests[-1].worst_case_loss_usd == approval.risk_usd
    assert approval.risk_usd == pytest.approx(84.72)

    assert engine.verify(
        approval.order, approval.token, now=NOW, broker_snapshot=broker_snapshot()
    ) is None
    assert requests[-1].requested_minis == expected_minis
    assert requests[-1].requested_micros == expected_micros
    assert requests[-1].worst_case_loss_usd == approval.token.risk_usd


def test_roll_session_clears_unspent_entry_identity(system_factory):
    engine, *_ = system_factory()
    approval = engine.evaluate_entry(valid_intent(), now=NOW).approval

    assert engine.roll_session(NOW + timedelta(days=1))
    rejection = engine.verify(
        approval.order, approval.token, now=NOW + timedelta(days=1)
    )

    assert rejection.reason is RejectReason.INVALID_TOKEN
    assert engine.last_verify_decision.context_trace.reason_codes == ("not_evaluated",)


def test_open_position_is_flattened_at_the_internal_prop_threshold(
    system_factory, position_factory
):
    engine, _, _, source, _, _, state = system_factory()
    source.context = replace(
        source.context,
        prop_state=replace(
            state,
            real_time_net_liquidation_usd=state.internal_safety_threshold_usd,
            remaining_prop_drawdown_usd=(
                state.internal_safety_threshold_usd - state.drawdown_floor_usd
            ),
            distance_to_prop_failure_usd=(
                state.internal_safety_threshold_usd - state.drawdown_floor_usd
            ),
            remaining_internal_cushion_usd=0.0,
        ),
    )

    reason, detail = engine.forced_exit_reason(
        position_factory(), 18_000.0, NOW
    )

    assert reason is RejectReason.TRAILING_DRAWDOWN
    assert "internal prop safety threshold" in detail


def test_open_position_fails_safe_when_prop_context_is_unavailable(
    system_factory, position_factory
):
    engine, _, _, source, *_ = system_factory()
    source.error = RuntimeError("credential-shaped transport detail")

    reason, detail = engine.forced_exit_reason(
        position_factory(), 18_000.0, NOW
    )

    assert reason is RejectReason.BROKER_ERROR
    assert "fail-safe flatten" in detail
    assert "credential-shaped" not in detail


def test_open_position_obeys_prop_early_close_deadline(
    system_factory, position_factory
):
    engine, _, _, source, *_ = system_factory()
    early_close = NOW.replace(hour=13, minute=0)
    source.context = replace(
        source.context,
        market_data_timestamp=early_close,
        market_day_status=MarketDayStatus.EARLY_CLOSE,
    )

    reason, detail = engine.forced_exit_reason(
        position_factory(), 18_000.0, early_close
    )

    assert reason is RejectReason.OUTSIDE_TRADING_HOURS
    assert "12:59" in detail


def test_open_position_obeys_the_firm_daily_loss_lock(
    system_factory, position_factory
):
    engine, _, _, source, _, _, state = system_factory()
    source.context = replace(
        source.context,
        prop_state=replace(
            state,
            current_balance_usd=48_750.0,
            real_time_net_liquidation_usd=48_750.0,
            remaining_prop_drawdown_usd=750.0,
            distance_to_prop_failure_usd=750.0,
            remaining_internal_cushion_usd=350.0,
            daily_realized_pnl_usd=-1_250.0,
            firm_daily_loss_limit_usd=1_250.0,
            remaining_firm_daily_risk_usd=0.0,
            soft_daily_loss_locked=True,
        ),
    )

    reason, detail = engine.forced_exit_reason(
        position_factory(), 18_000.0, NOW
    )

    assert reason is RejectReason.MAX_DAILY_LOSS
    assert "firm limit $1,250.00" in detail
