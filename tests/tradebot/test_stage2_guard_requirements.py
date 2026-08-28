"""Constructor invariants for Stage 2+ broker crash-recovery readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tradebot.broker.base import BrokerCapabilities, LiveTradingDisabled
from tradebot.broker.costs import CostModel
from tradebot.broker.guarded import GuardedBroker
from tradebot.broker.simulated import SimulatedBroker
from tradebot.config import CostConfig, KillSwitchConfig, RiskConfig, SessionConfig
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.deployment.stages import DeploymentStage
from tradebot.instruments.registry import get_instrument
from tradebot.prop_firms import load_prop_profile
from tradebot.risk.coordinator import (
    ContractKind,
    ThreeLayerRiskEngine,
    canonical_risk_state_context_id,
)
from tradebot.risk.limits import RiskEngine
from tradebot.risk.reservations import FilePendingEntryReservationStore
from tradebot.risk.state_store import FileRiskStateStore
from tradebot.risk.strategy_gate import StrategyGate


ROOT = Path(__file__).resolve().parents[2]
MNQ = get_instrument("MNQ")
NOW = datetime(2026, 8, 24, 11, 0, tzinfo=MARKET_TZ)
RUNTIME_CONFIG_SHA256 = "9" * 64


class FullyRecoverySafeSimulatedBroker(SimulatedBroker):
    """Test double proving the positive constructor path without external execution."""

    capabilities = BrokerCapabilities(
        external_execution=False,
        server_side_oco=True,
        reduce_only_or_close_position=True,
        authoritative_cancel_status=True,
        exact_terminal_order_history=True,
        authoritative_session_execution_history=True,
        account_owner_fencing=True,
    )


@dataclass
class RiskReadiness:
    """Only the readiness facts consumed by ``GuardedBroker.__init__``."""

    durable_state_certified: bool = True
    risk_state_bootstrapped_this_process: bool = False
    deployment_context_verified: bool = True
    broker_identity_is_pinned: bool = True
    instrument = MNQ


class NonDurableReservationStore:
    is_durable = False


def make_broker(
    clock: SimulatedClock, *, recovery_safe: bool = True
) -> SimulatedBroker:
    broker_type = FullyRecoverySafeSimulatedBroker if recovery_safe else SimulatedBroker
    return broker_type(
        MNQ,
        CostModel.from_config(CostConfig(), MNQ),
        clock=clock,
    )


@pytest.mark.parametrize(
    "stage",
    [
        DeploymentStage.PAPER,
        DeploymentStage.PROP_EVALUATION,
        DeploymentStage.FUNDED,
    ],
)
def test_every_stage_from_paper_up_requires_a_durable_reservation_store(
    tmp_path, stage
) -> None:
    broker = make_broker(SimulatedClock(NOW))

    with pytest.raises(
        LiveTradingDisabled,
        match="certified durable pending-entry reservation store",
    ):
        GuardedBroker(broker, RiskReadiness(), deployment_stage=stage)

    with pytest.raises(
        LiveTradingDisabled,
        match="certified durable pending-entry reservation store",
    ):
        GuardedBroker(
            broker,
            RiskReadiness(),
            reservation_store=NonDurableReservationStore(),
            deployment_stage=stage,
        )


@pytest.mark.parametrize(
    ("risk", "message"),
    [
        (
            RiskReadiness(durable_state_certified=False),
            "certified durable personal-risk state store",
        ),
        (
            RiskReadiness(risk_state_bootstrapped_this_process=True),
            "cannot start in the same process that bootstrapped risk history",
        ),
        (
            RiskReadiness(deployment_context_verified=False),
            "not bound to a canonical coordinator context",
        ),
        (
            RiskReadiness(broker_identity_is_pinned=False),
            "exact broker, route, paper/live, and account pins",
        ),
    ],
    ids=[
        "durable-personal-risk",
        "separate-bootstrap-workflow",
        "canonical-context",
        "broker-identity-pins",
    ],
)
def test_stage2_requires_each_personal_risk_readiness_fact(
    tmp_path, risk, message
) -> None:
    broker = make_broker(SimulatedClock(NOW))
    reservation_store = FilePendingEntryReservationStore(tmp_path / "pending.json")

    with pytest.raises(LiveTradingDisabled, match=message):
        GuardedBroker(
            broker,
            risk,
            reservation_store=reservation_store,
            deployment_stage=DeploymentStage.PAPER,
        )


def test_actual_simulator_remains_blocked_for_stage2_recovery(tmp_path) -> None:
    broker = make_broker(SimulatedClock(NOW), recovery_safe=False)
    capabilities = broker.capabilities
    assert capabilities.exact_terminal_order_history
    assert not capabilities.authoritative_session_execution_history
    assert not capabilities.account_owner_fencing
    assert not capabilities.stage_2_recovery_safe

    with pytest.raises(
        LiveTradingDisabled,
        match="exact terminal and session execution history.*account-owner fencing",
    ):
        GuardedBroker(
            broker,
            RiskReadiness(),
            reservation_store=FilePendingEntryReservationStore(
                tmp_path / "pending.json"
            ),
            deployment_stage=DeploymentStage.PAPER,
        )


def test_market_replay_does_not_claim_stage2_durability_requirements() -> None:
    broker = make_broker(SimulatedClock(NOW), recovery_safe=False)

    guard = GuardedBroker(
        broker,
        RiskReadiness(
            durable_state_certified=False,
            risk_state_bootstrapped_this_process=True,
            deployment_context_verified=False,
            broker_identity_is_pinned=False,
        ),
        deployment_stage=DeploymentStage.MARKET_REPLAY,
    )

    assert guard.execution_route == broker.execution_route


def test_capability_flags_cannot_enable_stage2_before_atomic_entry_protection(
    tmp_path,
) -> None:
    clock = SimulatedClock(NOW)
    broker = make_broker(clock)
    profile = load_prop_profile(
        ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
    )
    rules = profile.rules_for("evaluation")
    strategy_gate = StrategyGate(
        MNQ,
        minimum_expected_rr=2.0,
        max_signal_age=timedelta(seconds=60),
        max_data_age=timedelta(seconds=10),
    )
    context_id = canonical_risk_state_context_id(
        strategy_gate=strategy_gate,
        profile=profile,
        rules=rules,
        deployment_stage=DeploymentStage.PAPER,
        contract_kind=ContractKind.MICRO,
        runtime_config_sha256=RUNTIME_CONFIG_SHA256,
    )
    config = RiskConfig(
        kill_switch=KillSwitchConfig(flag_file=str(tmp_path / "stage2.kill"))
    )
    state_store = FileRiskStateStore(tmp_path / "personal-risk.json")
    risk_arguments = {
        "clock": clock,
        "expected_broker_account_id": "SIM-1",
        "expected_broker_name": broker.name,
        "expected_broker_is_paper": broker.is_paper,
        "expected_broker_route": broker.execution_route,
        "risk_state_store": state_store,
        "risk_state_context_id": context_id,
    }

    bootstrapped = RiskEngine(
        config,
        MNQ,
        SessionConfig(),
        bootstrap_risk_state_store=True,
        **risk_arguments,
    )
    assert bootstrapped.risk_state_store_error is None
    assert bootstrapped.risk_state_bootstrapped_this_process

    restored = RiskEngine(
        config,
        MNQ,
        SessionConfig(),
        bootstrap_risk_state_store=False,
        **risk_arguments,
    )
    assert restored.risk_state_store_error is None
    assert restored.durable_state_certified
    assert not restored.risk_state_bootstrapped_this_process
    coordinated = ThreeLayerRiskEngine(
        strategy_gate=strategy_gate,
        personal_risk=restored,
        profile=profile,
        rules=rules,
        deployment_stage=DeploymentStage.PAPER,
        contract_kind=ContractKind.MICRO,
        context_provider=lambda _now: None,
        runtime_config_sha256=RUNTIME_CONFIG_SHA256,
    )
    assert coordinated.deployment_context_verified
    assert coordinated.broker_identity_is_pinned

    assert broker.capabilities.stage_2_recovery_safe
    with pytest.raises(
        LiveTradingDisabled,
        match="guarded atomic entry-with-protection",
    ):
        GuardedBroker(
            broker,
            coordinated,
            reservation_store=FilePendingEntryReservationStore(
                tmp_path / "pending.json"
            ),
            deployment_stage=DeploymentStage.PAPER,
        )
