"""Explicit deployment stages and human approval manifests."""

from .stages import (
    ApprovalManifest,
    DeploymentStage,
    StageAuthorization,
    StageManifestError,
    artifact_sha256,
    authorize_stage,
    load_approval_manifest,
)

__all__ = [
    "ApprovalManifest",
    "DeploymentStage",
    "StageAuthorization",
    "StageManifestError",
    "artifact_sha256",
    "authorize_stage",
    "load_approval_manifest",
]

