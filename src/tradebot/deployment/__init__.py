"""Explicit deployment stages and human approval manifests."""

from .stages import (
    ApprovalManifest,
    DeploymentStage,
    StageAuthorization,
    StageManifestError,
    artifact_sha256,
    authorize_stage,
    load_approval_manifest,
    stage_authorization_failure_codes,
)

__all__ = [
    "ApprovalManifest",
    "DeploymentStage",
    "StageAuthorization",
    "StageManifestError",
    "artifact_sha256",
    "authorize_stage",
    "load_approval_manifest",
    "stage_authorization_failure_codes",
]
