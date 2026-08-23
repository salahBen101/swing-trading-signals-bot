"""Fail-closed stage control.

Stage 0-2 are research/paper activities.  Stage 3 and 4 are never inferred from a broker
name or a successful test: they require a separate, human-created manifest that pins every
material artifact.  This module authorizes nothing by side effect; it only returns a
decision trace for the runner to enforce and journal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import subprocess
import unicodedata
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml

from ..prop_firms.models import AccountPhase, PropFirmProfile
from ..prop_firms.verification import RuleVerificationResult, VerificationStatus


class DeploymentStage(IntEnum):
    BACKTEST = 0
    MARKET_REPLAY = 1
    PAPER = 2
    PROP_EVALUATION = 3
    FUNDED = 4

    @property
    def requires_human_approval(self) -> bool:
        return self >= DeploymentStage.PROP_EVALUATION


class StageManifestError(ValueError):
    pass


_AUTHORIZATION_TTL = timedelta(minutes=5)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class ApprovalManifest:
    schema_version: int
    stage: DeploymentStage
    approved: bool
    approved_by: str
    approved_at: datetime
    account_id: str
    prop_profile_id: str
    prop_phase: str
    prop_profile_sha256: str
    source_snapshot_sha256: str
    runtime_config_sha256: str
    strategy_artifact_sha256: str
    code_revision: str
    execution_route: str
    execution_route_verified: bool
    sole_owner_attested: bool
    firm_exclusive_use_attested: bool
    production_frozen: bool
    notes: str = ""

    def validate(self) -> None:
        if self.schema_version != 1:
            raise StageManifestError(f"unsupported approval manifest version {self.schema_version}")
        if self.stage < DeploymentStage.PROP_EVALUATION:
            raise StageManifestError("approval manifests are only valid for Stage 3 or Stage 4")
        if self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None:
            raise StageManifestError("approved_at must be timezone-aware")
        for name in (
            "approved_by", "account_id", "prop_profile_id", "prop_phase",
            "prop_profile_sha256", "source_snapshot_sha256", "runtime_config_sha256",
            "strategy_artifact_sha256",
            "code_revision", "execution_route",
        ):
            if not str(getattr(self, name)).strip():
                raise StageManifestError(f"{name} must not be empty")
        for name in (
            "prop_profile_sha256", "source_snapshot_sha256", "runtime_config_sha256",
            "strategy_artifact_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise StageManifestError(f"{name} must be a SHA-256 hex digest")
        if not _GIT_REVISION.fullmatch(self.code_revision.casefold()):
            raise StageManifestError("code_revision must be a full 40-character git commit")


@dataclass(frozen=True, slots=True)
class StageAuthorization:
    """A public decision envelope; Stage 3/4 permission is an authenticated capability.

    Three-argument construction remains useful for inert Stage 0-2 contexts.  For Stage
    3/4, ``allowed=True`` is never trusted by itself: the coordinator authenticates every
    signed claim and its short expiry using this process's private authority.
    """

    stage: DeploymentStage
    allowed: bool
    reasons: tuple[str, ...]
    account_id: str | None = None
    execution_route: str | None = None
    profile_id: str | None = None
    prop_phase: str | None = None
    manifest_sha256: str | None = None
    prop_profile_sha256: str | None = None
    source_snapshot_sha256: str | None = None
    runtime_config_sha256: str | None = None
    strategy_artifact_sha256: str | None = None
    code_revision: str | None = None
    rule_verification_sha256: str | None = None
    rule_verification_checked_at: datetime | None = None
    issued_at: datetime | None = None
    expires_at: datetime | None = None
    authorization_id: str | None = None
    _signature: str | None = dataclass_field(default=None, repr=False, compare=False)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "authorized"


@dataclass(frozen=True, slots=True)
class _AuthorizationEnvironment:
    """Private deterministic seam. Production always derives these facts internally."""

    now: datetime
    code_revision: str
    working_tree_clean: bool


def normalize_execution_route(value: str) -> str:
    """Return one stable route identity for manifest and adapter comparisons."""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().casefold().split())


def _aware(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def approval_manifest_sha256(manifest: ApprovalManifest) -> str:
    payload = {
        "schema_version": manifest.schema_version,
        "stage": int(manifest.stage),
        "approved": manifest.approved,
        "approved_by": manifest.approved_by,
        "approved_at": manifest.approved_at.isoformat(),
        "account_id": manifest.account_id,
        "prop_profile_id": manifest.prop_profile_id,
        "prop_phase": manifest.prop_phase,
        "prop_profile_sha256": manifest.prop_profile_sha256,
        "source_snapshot_sha256": manifest.source_snapshot_sha256,
        "runtime_config_sha256": manifest.runtime_config_sha256,
        "strategy_artifact_sha256": manifest.strategy_artifact_sha256,
        "code_revision": manifest.code_revision,
        "execution_route": normalize_execution_route(manifest.execution_route),
        "execution_route_verified": manifest.execution_route_verified,
        "sole_owner_attested": manifest.sole_owner_attested,
        "firm_exclusive_use_attested": manifest.firm_exclusive_use_attested,
        "production_frozen": manifest.production_frozen,
        "notes": manifest.notes,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def rule_verification_sha256(result: RuleVerificationResult) -> str:
    return hashlib.sha256(_canonical_json(result.to_dict())).hexdigest()


def _authorization_payload(authorization: StageAuthorization) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": int(authorization.stage),
        "allowed": authorization.allowed,
        "reasons": list(authorization.reasons),
        "account_id": authorization.account_id,
        "execution_route": authorization.execution_route,
        "profile_id": authorization.profile_id,
        "prop_phase": authorization.prop_phase,
        "manifest_sha256": authorization.manifest_sha256,
        "prop_profile_sha256": authorization.prop_profile_sha256,
        "source_snapshot_sha256": authorization.source_snapshot_sha256,
        "runtime_config_sha256": authorization.runtime_config_sha256,
        "strategy_artifact_sha256": authorization.strategy_artifact_sha256,
        "code_revision": authorization.code_revision,
        "rule_verification_sha256": authorization.rule_verification_sha256,
        "rule_verification_checked_at": authorization.rule_verification_checked_at.isoformat(),
        "issued_at": authorization.issued_at.isoformat(),
        "expires_at": authorization.expires_at.isoformat(),
        "authorization_id": authorization.authorization_id,
    }


class _StageAuthorizationAuthority:
    """Process-local signer; capabilities cannot survive or bypass a fresh startup."""

    def __init__(self) -> None:
        self.__secret = secrets.token_bytes(32)

    def mint(self, authorization: StageAuthorization) -> StageAuthorization:
        signature = hmac.new(
            self.__secret,
            _canonical_json(_authorization_payload(authorization)),
            hashlib.sha256,
        ).hexdigest()
        return replace(authorization, _signature=signature)

    def authentic(self, authorization: StageAuthorization) -> bool:
        expected = hmac.new(
            self.__secret,
            _canonical_json(_authorization_payload(authorization)),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, authorization._signature or "")


_AUTHORITY = _StageAuthorizationAuthority()


_MANIFEST_KEYS = {
    "schema_version", "stage", "approved", "approved_by", "approved_at", "account_id",
    "prop_profile_id", "prop_phase", "prop_profile_sha256", "source_snapshot_sha256",
    "runtime_config_sha256", "strategy_artifact_sha256", "code_revision", "execution_route",
    "execution_route_verified", "sole_owner_attested", "firm_exclusive_use_attested",
    "production_frozen", "notes",
}


def _strict_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise StageManifestError(f"{path} must be true or false")
    return value


def load_approval_manifest(path: str | Path) -> ApprovalManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise StageManifestError(f"approval manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise StageManifestError("approval manifest root must be a mapping")
    unknown = set(raw) - _MANIFEST_KEYS
    missing = (_MANIFEST_KEYS - {"notes"}) - set(raw)
    if unknown:
        raise StageManifestError(f"unknown approval manifest key(s): {sorted(unknown)}")
    if missing:
        raise StageManifestError(f"missing approval manifest key(s): {sorted(missing)}")
    try:
        approved_at = datetime.fromisoformat(str(raw["approved_at"]))
        stage = DeploymentStage(int(raw["stage"]))
    except (TypeError, ValueError) as exc:
        raise StageManifestError("stage must be 3/4 and approved_at must be ISO-8601") from exc
    manifest = ApprovalManifest(
        schema_version=int(raw["schema_version"]),
        stage=stage,
        approved=_strict_bool(raw["approved"], "approved"),
        approved_by=str(raw["approved_by"]),
        approved_at=approved_at,
        account_id=str(raw["account_id"]),
        prop_profile_id=str(raw["prop_profile_id"]),
        prop_phase=str(raw["prop_phase"]),
        prop_profile_sha256=str(raw["prop_profile_sha256"]).lower(),
        source_snapshot_sha256=str(raw["source_snapshot_sha256"]).lower(),
        runtime_config_sha256=str(raw["runtime_config_sha256"]).lower(),
        strategy_artifact_sha256=str(raw["strategy_artifact_sha256"]).lower(),
        code_revision=str(raw["code_revision"]),
        execution_route=str(raw["execution_route"]),
        execution_route_verified=_strict_bool(raw["execution_route_verified"], "execution_route_verified"),
        sole_owner_attested=_strict_bool(raw["sole_owner_attested"], "sole_owner_attested"),
        firm_exclusive_use_attested=_strict_bool(raw["firm_exclusive_use_attested"], "firm_exclusive_use_attested"),
        production_frozen=_strict_bool(raw["production_frozen"], "production_frozen"),
        notes=str(raw.get("notes", "")),
    )
    manifest.validate()
    return manifest


def artifact_sha256(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise StageManifestError(f"pinned artifact not found: {artifact}")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def authorize_stage(
    stage: DeploymentStage,
    *,
    as_of: datetime,
    profile: PropFirmProfile | None = None,
    profile_path: str | Path | None = None,
    source_snapshot_path: str | Path | None = None,
    runtime_config_path: str | Path | None = None,
    strategy_artifact_path: str | Path | None = None,
    manifest: ApprovalManifest | None = None,
    rule_verification: RuleVerificationResult | None = None,
    code_revision: str = "",
    working_tree_clean: bool = False,
) -> StageAuthorization:
    """Evaluate stage permission without mutating external state.

    The runner must call this again on every Stage 3/4 startup.  A prior success is not a
    durable permission because official-rule freshness and repository state can change.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        return StageAuthorization(stage, False, ("authorization time is not timezone-aware",))
    if stage <= DeploymentStage.PAPER:
        return StageAuthorization(stage, True, ())

    reasons: list[str] = []
    if manifest is None:
        return StageAuthorization(stage, False, ("Stage 3/4 requires a human approval manifest",))

    manifest_identity_valid = False
    try:
        manifest.validate()
        manifest_identity_valid = True
    except StageManifestError as exc:
        reasons.append(str(exc))
    if manifest.stage is not stage:
        reasons.append(f"manifest authorizes Stage {int(manifest.stage)}, not Stage {int(stage)}")
    if not manifest.approved:
        reasons.append("human approval flag is false")
    if not manifest.execution_route_verified:
        reasons.append("execution route has not been independently verified")
    if not manifest.sole_owner_attested:
        reasons.append("sole bot ownership has not been attested")
    if not manifest.firm_exclusive_use_attested:
        reasons.append("firm-exclusive deployment has not been attested")
    if not manifest.production_frozen:
        reasons.append("production artifact is not marked immutable")
    if not working_tree_clean:
        reasons.append("working tree is not clean")
    if not code_revision or manifest.code_revision != code_revision:
        reasons.append("code revision does not match the approved manifest")

    if profile is None:
        reasons.append("no prop-firm profile selected")
    else:
        if manifest.prop_profile_id != profile.profile_id:
            reasons.append("selected prop profile id differs from the approved manifest")
        if not profile.rules_are_fresh(as_of):
            reasons.append("official prop rules are stale and require reverification")
        if profile.ambiguity_notes:
            reasons.append("prop profile contains unresolved rule conflicts")
        try:
            rules = profile.rules_for(manifest.prop_phase)
        except KeyError:
            reasons.append(f"approved prop phase {manifest.prop_phase!r} is not in the profile")
        else:
            expected = AccountPhase.EVALUATION if stage is DeploymentStage.PROP_EVALUATION else None
            if expected is not None and rules.phase is not expected:
                reasons.append("Stage 3 requires an evaluation-phase rule set")
            if stage is DeploymentStage.FUNDED and rules.phase not in (AccountPhase.SIM_FUNDED, AccountPhase.LIVE):
                reasons.append("Stage 4 requires a funded-phase rule set")
            if (
                profile.firm.casefold() == "tradeify"
                and rules.phase in (AccountPhase.EVALUATION, AccountPhase.SIM_FUNDED)
                and "tradovate" in manifest.execution_route.casefold()
                and "api" in manifest.execution_route.casefold()
            ):
                reasons.append("Tradeify does not permit Tradovate API access for Evaluation/Sim Funded")

        if rule_verification is None:
            reasons.append("no current official-source verification result was supplied")
        else:
            if rule_verification.profile_id != profile.profile_id:
                reasons.append("official-source verification belongs to a different profile")
            if rule_verification.status is not VerificationStatus.UNCHANGED:
                reasons.append(
                    f"official-source verification is {rule_verification.status.value}, not unchanged"
                )
            if (
                rule_verification.checked_at.tzinfo is None
                or rule_verification.checked_at.utcoffset() is None
            ):
                reasons.append("official-source verification timestamp is not timezone-aware")
            else:
                if rule_verification.checked_at > as_of:
                    reasons.append("official-source verification timestamp is in the future")
                elif as_of - rule_verification.checked_at > timedelta(
                    hours=profile.reverify_after_hours
                ):
                    reasons.append("official-source verification is stale")
            expected_urls = {source.url for source in profile.sources}
            checked_urls = {check.url for check in rule_verification.checks}
            if checked_urls != expected_urls:
                reasons.append("official-source verification does not cover every profile source")
            if any(
                check.changed
                or bool(check.error)
                or check.expected_sha256 is None
                or check.observed_sha256 is None
                or check.expected_sha256 != check.observed_sha256
                for check in rule_verification.checks
            ):
                reasons.append("official-source verification contains changed or incomplete checks")

    pinned = (
        ("prop profile", profile_path, manifest.prop_profile_sha256),
        ("source snapshot", source_snapshot_path, manifest.source_snapshot_sha256),
        ("runtime config", runtime_config_path, manifest.runtime_config_sha256),
        ("strategy artifact", strategy_artifact_path, manifest.strategy_artifact_sha256),
    )
    for label, path, expected_hash in pinned:
        if path is None:
            reasons.append(f"{label} path is missing")
            continue
        try:
            actual = artifact_sha256(path)
        except StageManifestError as exc:
            reasons.append(str(exc))
            continue
        if actual != expected_hash:
            reasons.append(f"{label} hash differs from the approved manifest")

    return StageAuthorization(
        stage,
        not reasons,
        tuple(reasons),
        account_id=(manifest.account_id if manifest_identity_valid else None),
        execution_route=(manifest.execution_route if manifest_identity_valid else None),
    )
