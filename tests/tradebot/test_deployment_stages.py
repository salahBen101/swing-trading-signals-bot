from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradebot.deployment import (
    ApprovalManifest,
    DeploymentStage,
    StageAuthorization,
    StageManifestError,
    artifact_sha256,
    authorize_stage,
    load_approval_manifest,
)
from tradebot.prop_firms import load_prop_profile
from tradebot.prop_firms import RuleVerificationResult, SourceCheck, VerificationStatus


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
CONFIG_PATH = ROOT / "config" / "tradebot.yaml"
STRATEGY_PATH = ROOT / "src" / "tradebot" / "strategy" / "orb_breakout.py"
SNAPSHOT_PATH = ROOT / "config" / "prop_firms" / "source_snapshots.yaml"
# The stage gate requires a full 40-character git commit, so these are real-shaped
# SHAs rather than placeholders. OTHER_REVISION exists to test a mismatch.
HEAD_REVISION = "0" * 40
OTHER_REVISION = "1" * 40

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def manifest(stage: DeploymentStage = DeploymentStage.PROP_EVALUATION, phase: str = "evaluation") -> ApprovalManifest:
    return ApprovalManifest(
        schema_version=1,
        stage=stage,
        approved=True,
        approved_by="human-operator",
        approved_at=NOW,
        account_id="ACCOUNT-EXAMPLE",
        prop_profile_id="tradeify_growth_50k_current",
        prop_phase=phase,
        prop_profile_sha256=artifact_sha256(PROFILE_PATH),
        source_snapshot_sha256=artifact_sha256(SNAPSHOT_PATH),
        runtime_config_sha256=artifact_sha256(CONFIG_PATH),
        strategy_artifact_sha256=artifact_sha256(STRATEGY_PATH),
        code_revision=HEAD_REVISION,
        execution_route="verified_platform_native_local",
        execution_route_verified=True,
        sole_owner_attested=True,
        firm_exclusive_use_attested=True,
        production_frozen=True,
    )


def authorize(m: ApprovalManifest, **overrides):
    profile = replace(load_prop_profile(PROFILE_PATH), ambiguity_notes=())
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
    args = dict(
        stage=m.stage,
        as_of=NOW,
        profile=profile,
        profile_path=PROFILE_PATH,
        source_snapshot_path=SNAPSHOT_PATH,
        runtime_config_path=CONFIG_PATH,
        strategy_artifact_path=STRATEGY_PATH,
        manifest=m,
        rule_verification=verification,
        code_revision=HEAD_REVISION,
        working_tree_clean=True,
    )
    args.update(overrides)
    return authorize_stage(**args)


@pytest.mark.parametrize("stage", [
    DeploymentStage.BACKTEST,
    DeploymentStage.MARKET_REPLAY,
    DeploymentStage.PAPER,
])
def test_stage_zero_to_two_need_no_approval(stage: DeploymentStage) -> None:
    decision = authorize_stage(stage, as_of=NOW)
    assert decision.allowed
    assert decision.reasons == ()
    assert decision.account_id is None
    assert decision.execution_route is None


def test_three_argument_stage_authorization_construction_remains_compatible() -> None:
    authorization = StageAuthorization(DeploymentStage.PAPER, True, ())

    assert authorization.allowed
    assert authorization.account_id is None
    assert authorization.execution_route is None


@pytest.mark.parametrize("stage", [DeploymentStage.PROP_EVALUATION, DeploymentStage.FUNDED])
def test_stage_three_and_four_never_auto_enable(stage: DeploymentStage) -> None:
    decision = authorize_stage(stage, as_of=NOW)
    assert not decision.allowed
    assert "human approval manifest" in decision.reason


def test_exact_clean_human_approved_stage_three_manifest_can_pass_the_generic_gate() -> None:
    approved_manifest = manifest()
    decision = authorize(approved_manifest)
    assert decision.allowed, decision.reason
    assert decision.account_id == approved_manifest.account_id
    assert decision.execution_route == approved_manifest.execution_route


def test_invalid_manifest_cannot_supply_authorized_identity() -> None:
    invalid = replace(manifest(), account_id="")

    decision = authorize(invalid)

    assert not decision.allowed
    assert decision.account_id is None
    assert decision.execution_route is None


def test_checked_in_tradeify_conflicts_keep_stage_three_blocked() -> None:
    m = manifest()
    decision = authorize(
        m,
        profile=load_prop_profile(PROFILE_PATH),
    )
    assert not decision.allowed
    assert "unresolved rule conflicts" in decision.reason


def test_stage_three_requires_complete_unchanged_current_source_verification() -> None:
    m = manifest()
    missing = authorize(m, rule_verification=None)
    assert not missing.allowed
    assert "no current official-source verification" in missing.reason

    changed = RuleVerificationResult(
        profile_id=m.prop_profile_id,
        checked_at=NOW,
        status=VerificationStatus.CHANGED,
        checks=(),
    )
    rejected = authorize(m, rule_verification=changed)
    assert not rejected.allowed
    assert "not unchanged" in rejected.reason
    assert "does not cover every profile source" in rejected.reason


def test_tradeify_tradovate_api_route_is_rejected_for_evaluation() -> None:
    m = replace(manifest(), execution_route="Tradovate API")
    decision = authorize(m)
    assert not decision.allowed
    assert "does not permit Tradovate API" in decision.reason


def test_stale_rules_dirty_tree_and_revision_mismatch_each_fail_closed() -> None:
    m = manifest()
    decision = authorize(
        m,
        as_of=datetime(2026, 8, 23, 0, 0, 1, tzinfo=timezone.utc),
        working_tree_clean=False,
        code_revision=OTHER_REVISION,
    )
    assert not decision.allowed
    assert "stale" in decision.reason
    assert "not clean" in decision.reason
    assert "revision" in decision.reason


def test_a_stage_three_manifest_cannot_enable_stage_four() -> None:
    m = manifest()
    decision = authorize(m, stage=DeploymentStage.FUNDED)
    assert not decision.allowed
    assert "not Stage 4" in decision.reason


def test_stage_four_requires_a_funded_phase() -> None:
    evaluation_manifest = manifest(DeploymentStage.FUNDED, "evaluation")
    decision = authorize(evaluation_manifest)
    assert not decision.allowed
    assert "funded-phase" in decision.reason


def test_artifact_hash_mismatch_blocks_startup(tmp_path: Path) -> None:
    m = manifest()
    changed = tmp_path / "strategy.py"
    changed.write_text("# different\n", encoding="utf-8")
    decision = authorize(m, strategy_artifact_path=changed)
    assert not decision.allowed
    assert "strategy artifact hash differs" in decision.reason


def test_manifest_loader_rejects_template_and_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(StageManifestError, match="must not be empty|SHA-256"):
        load_approval_manifest(ROOT / "config" / "deployment.example.yaml")

    bad = tmp_path / "approval.yaml"
    bad.write_text("schema_version: 1\nstage: 3\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(StageManifestError, match="unknown approval manifest"):
        load_approval_manifest(bad)


def test_manifest_boolean_strings_are_not_accepted(tmp_path: Path) -> None:
    m = manifest()
    lines = [
        "schema_version: 1",
        "stage: 3",
        'approved: "true"',
        f'approved_by: "{m.approved_by}"',
        f'approved_at: "{m.approved_at.isoformat()}"',
        f'account_id: "{m.account_id}"',
        f'prop_profile_id: "{m.prop_profile_id}"',
        f'prop_phase: "{m.prop_phase}"',
        f'prop_profile_sha256: "{m.prop_profile_sha256}"',
        f'source_snapshot_sha256: "{m.source_snapshot_sha256}"',
        f'runtime_config_sha256: "{m.runtime_config_sha256}"',
        f'strategy_artifact_sha256: "{m.strategy_artifact_sha256}"',
        f'code_revision: "{m.code_revision}"',
        f'execution_route: "{m.execution_route}"',
        "execution_route_verified: true",
        "sole_owner_attested: true",
        "firm_exclusive_use_attested: true",
        "production_frozen: true",
    ]
    path = tmp_path / "approval.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(StageManifestError, match="approved must be true or false"):
        load_approval_manifest(path)
