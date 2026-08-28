from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import tradebot.deployment.stages as stages_module

from tradebot.deployment import (
    ApprovalManifest,
    DeploymentStage,
    StageAuthorization,
    StageManifestError,
    artifact_sha256,
    authorize_stage,
    load_approval_manifest,
    stage_authorization_failure_codes,
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
    profile = overrides.pop(
        "authorization_profile",
        replace(load_prop_profile(PROFILE_PATH), ambiguity_notes=()),
    )
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
    verification = overrides.pop("authorization_verification", verification)
    args = dict(
        stage=m.stage,
        as_of=NOW,
        profile_path=PROFILE_PATH,
        source_snapshot_path=SNAPSHOT_PATH,
        runtime_config_path=CONFIG_PATH,
        strategy_artifact_path=STRATEGY_PATH,
        manifest=m,
    )
    args.update(overrides)
    environment = stages_module._AuthorizationEnvironment(
        now=args["as_of"],
        code_revision=(
            OTHER_REVISION
            if overrides.get("repository_revision_mismatch", False)
            else HEAD_REVISION
        ),
        working_tree_clean=not overrides.get("repository_dirty", False),
    )
    args.pop("repository_revision_mismatch", None)
    args.pop("repository_dirty", None)
    original = stages_module._read_authorization_environment
    original_profile_loader = stages_module._load_authorization_profile
    original_verifier = stages_module._run_authorization_rule_verification
    stages_module._read_authorization_environment = lambda: environment
    stages_module._load_authorization_profile = lambda path: profile
    stages_module._run_authorization_rule_verification = (
        lambda loaded, path, *, checked_at: verification
    )
    try:
        return authorize_stage(**args)
    finally:
        stages_module._read_authorization_environment = original
        stages_module._load_authorization_profile = original_profile_loader
        stages_module._run_authorization_rule_verification = original_verifier


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
    assert stage_authorization_failure_codes(
        decision,
        stage=DeploymentStage.PROP_EVALUATION,
        as_of=NOW,
    ) == ()


def test_constructed_or_modified_live_authorization_is_not_a_capability() -> None:
    constructed = StageAuthorization(
        DeploymentStage.PROP_EVALUATION,
        True,
        (),
        account_id="ACCOUNT-EXAMPLE",
    )
    constructed_failures = stage_authorization_failure_codes(
        constructed,
        stage=DeploymentStage.PROP_EVALUATION,
        as_of=NOW,
    )
    assert "stage_authorization_unauthenticated" in constructed_failures

    approved = authorize(manifest())
    modified = replace(approved, account_id="DIFFERENT-ACCOUNT")
    modified_failures = stage_authorization_failure_codes(
        modified,
        stage=DeploymentStage.PROP_EVALUATION,
        as_of=NOW,
    )
    assert modified_failures == ("stage_authorization_unauthenticated",)


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": object()},
        {"reasons": (object(),)},
        {"_signature": object()},
    ],
)
def test_corrupt_live_authorization_fails_closed_without_raising(changes) -> None:
    corrupt = replace(authorize(manifest()), **changes)

    failures = stage_authorization_failure_codes(
        corrupt,
        stage=DeploymentStage.PROP_EVALUATION,
        as_of=NOW,
    )

    assert "stage_authorization_malformed" in failures
    assert "stage_authorization_unauthenticated" in failures


def test_live_authorization_expires_after_its_bounded_window() -> None:
    approved = authorize(manifest())

    failures = stage_authorization_failure_codes(
        approved,
        stage=DeploymentStage.PROP_EVALUATION,
        as_of=NOW + timedelta(minutes=5),
    )

    assert failures == ("stage_authorization_expired",)


def test_invalid_manifest_cannot_supply_authorized_identity() -> None:
    invalid = replace(manifest(), account_id="")

    decision = authorize(invalid)

    assert not decision.allowed
    assert decision.account_id is None
    assert decision.execution_route is None


def test_naive_manifest_timestamp_fails_closed_without_raising() -> None:
    decision = authorize(replace(manifest(), approved_at=NOW.replace(tzinfo=None)))

    assert not decision.allowed
    assert "approved_at must be timezone-aware" in decision.reason


def test_direct_manifest_construction_cannot_use_truthy_string_flags() -> None:
    string_flags = replace(
        manifest(),
        approved="false",
        execution_route_verified="false",
        sole_owner_attested="false",
        firm_exclusive_use_attested="false",
        production_frozen="false",
    )

    decision = authorize(string_flags)

    assert not decision.allowed
    assert "must be true or false" in decision.reason


def test_checked_in_tradeify_conflicts_keep_stage_three_blocked() -> None:
    m = manifest()
    decision = authorize(
        m,
        authorization_profile=load_prop_profile(PROFILE_PATH),
    )
    assert not decision.allowed
    assert "unresolved rule conflicts" in decision.reason


def test_stage_three_requires_complete_unchanged_current_source_verification() -> None:
    m = manifest()
    missing = authorize(m, authorization_verification=None)
    assert not missing.allowed
    assert "verification result is missing or invalid" in missing.reason

    changed = RuleVerificationResult(
        profile_id=m.prop_profile_id,
        checked_at=NOW,
        status=VerificationStatus.CHANGED,
        checks=(),
    )
    rejected = authorize(m, authorization_verification=changed)
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
        repository_dirty=True,
        repository_revision_mismatch=True,
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


def test_exact_clean_human_approved_stage_four_manifest_mints_a_valid_capability() -> None:
    decision = authorize(manifest(DeploymentStage.FUNDED, "sim_funded"))

    assert decision.allowed, decision.reason
    assert stage_authorization_failure_codes(
        decision,
        stage=DeploymentStage.FUNDED,
        as_of=NOW,
    ) == ()


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
