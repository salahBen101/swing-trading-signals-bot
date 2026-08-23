"""Official-source change detection for prop profiles.

The detector deliberately cannot edit a rule profile.  It compares normalized official
page text with a reviewed baseline and emits a result that an operator can record or turn
into a review proposal.  A changed or unavailable page is never interpreted as permission
to keep trading.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml

from .models import PropFirmProfile


NORMALIZATION_VERSION = 1


class VerificationStatus(str, Enum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    INCOMPLETE_BASELINE = "incomplete_baseline"
    FETCH_ERROR = "fetch_error"


@dataclass(frozen=True, slots=True)
class SourceBaseline:
    url: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SourceCheck:
    url: str
    expected_sha256: str | None
    observed_sha256: str | None
    changed: bool
    error: str = ""


@dataclass(frozen=True, slots=True)
class RuleVerificationResult:
    profile_id: str
    checked_at: datetime
    status: VerificationStatus
    checks: tuple[SourceCheck, ...]

    @property
    def safe_to_reuse_reviewed_rules(self) -> bool:
        return self.status is VerificationStatus.UNCHANGED

    @property
    def changed_urls(self) -> tuple[str, ...]:
        return tuple(check.url for check in self.checks if check.changed)

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "checked_at": self.checked_at.isoformat(),
            "status": self.status.value,
            "safe_to_reuse_reviewed_rules": self.safe_to_reuse_reviewed_rules,
            "checks": [
                {
                    "url": check.url,
                    "expected_sha256": check.expected_sha256,
                    "observed_sha256": check.observed_sha256,
                    "changed": check.changed,
                    "error": check.error,
                }
                for check in self.checks
            ],
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def normalize_official_page(content: bytes | str) -> str:
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    parser = _TextExtractor()
    parser.feed(text)
    visible = " ".join(parser.parts)
    return re.sub(r"\s+", " ", visible).strip()


def content_sha256(content: bytes | str) -> str:
    normalized = normalize_official_page(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _download_page(url: str, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"User-Agent": "tradebot-rule-verifier/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - candidates are constrained below
        return response.read()


def _tradeify_intercom_mirror(url: str) -> str | None:
    """Return Tradeify's official provider mirror for its challenged custom domain.

    The live help center identifies itself as Intercom help-center ``tradeify``.  Keeping
    this mapping exact prevents an arbitrary profile URL from turning the verifier into a
    general redirecting fetcher.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "help.tradeify.co":
        return None
    path = parts.path if parts.path.startswith("/") else f"/{parts.path}"
    return urlunsplit(("https", "intercom.help", f"/tradeify{path}", parts.query, ""))


def _is_browser_challenge(content: bytes) -> bool:
    prefix = content[:16_384].lower()
    return b"just a moment" in prefix and (
        b"cloudflare" in prefix or b"challenge-platform" in prefix
    )


def fetch_official_page(url: str, *, timeout_seconds: float = 20.0) -> bytes:
    """Fetch a reviewed official URL, using only Tradeify's exact Intercom mirror.

    Tradeify's custom domain currently returns Cloudflare 403 to unattended clients while
    the provider-owned mirror serves the identical help-center article. Both candidates
    preserve the same article path. If neither works, the caller receives an exception and
    verification remains fail-closed.
    """
    if urlsplit(url).scheme != "https":
        raise ValueError("official rule source must use HTTPS")
    mirror = _tradeify_intercom_mirror(url)
    try:
        content = _download_page(url, timeout_seconds)
        if not _is_browser_challenge(content):
            return content
        if mirror is None:
            raise OSError("official source returned a browser challenge")
    except Exception as primary_error:
        if mirror is None:
            raise
        try:
            return _download_page(mirror, timeout_seconds)
        except Exception as mirror_error:
            raise OSError(
                f"official source and its reviewed Intercom mirror were unavailable: "
                f"{primary_error}; {mirror_error}"
            ) from mirror_error
    return _download_page(mirror, timeout_seconds)


def load_source_baselines(path: str | Path) -> dict[str, SourceBaseline]:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return {}
    raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "normalization_version", "captured_on", "sources"}:
        raise ValueError("source snapshot must contain schema_version, normalization_version, captured_on, sources")
    if int(raw["schema_version"]) != 1 or int(raw["normalization_version"]) != NORMALIZATION_VERSION:
        raise ValueError("unsupported prop source snapshot version")
    sources = raw["sources"]
    if not isinstance(sources, dict):
        raise ValueError("source snapshot sources must be a mapping")
    result: dict[str, SourceBaseline] = {}
    for url, value in sources.items():
        if not isinstance(value, dict) or set(value) != {"content_sha256"}:
            raise ValueError(f"invalid source snapshot entry for {url}")
        digest = str(value["content_sha256"]).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 for {url}")
        result[str(url)] = SourceBaseline(str(url), digest)
    return result


def verify_profile_sources(
    profile: PropFirmProfile,
    baselines: dict[str, SourceBaseline],
    *,
    checked_at: datetime,
    fetcher: Callable[[str], bytes | str] = fetch_official_page,
) -> RuleVerificationResult:
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("rule verification timestamp must be timezone-aware")
    checks: list[SourceCheck] = []
    missing = False
    had_error = False
    for source in profile.sources:
        baseline = baselines.get(source.url)
        if baseline is None:
            missing = True
            checks.append(SourceCheck(source.url, None, None, True, "source has no reviewed baseline"))
            continue
        try:
            observed = content_sha256(fetcher(source.url))
        except Exception as exc:  # network and parser failures all fail closed
            had_error = True
            checks.append(SourceCheck(source.url, baseline.content_sha256, None, True, str(exc)))
            continue
        checks.append(
            SourceCheck(
                source.url,
                baseline.content_sha256,
                observed,
                changed=observed != baseline.content_sha256,
            )
        )

    if had_error:
        status = VerificationStatus.FETCH_ERROR
    elif missing:
        status = VerificationStatus.INCOMPLETE_BASELINE
    elif any(check.changed for check in checks):
        status = VerificationStatus.CHANGED
    else:
        status = VerificationStatus.UNCHANGED
    return RuleVerificationResult(profile.profile_id, checked_at, status, tuple(checks))


def append_verification_record(result: RuleVerificationResult, path: str | Path) -> None:
    """Append an audit record. This does not update or approve a rule profile."""
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def write_change_proposal(result: RuleVerificationResult, directory: str | Path) -> tuple[Path, Path]:
    """Emit an alert and a review stub without changing production configuration."""
    if result.safe_to_reuse_reviewed_rules:
        raise ValueError("unchanged rules do not need a change proposal")
    proposal_dir = Path(directory)
    proposal_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.checked_at.strftime("%Y%m%dT%H%M%SZ")
    alert_path = proposal_dir / f"PROP_RULE_ALERT_{result.profile_id}_{stamp}.json"
    proposal_path = proposal_dir / f"PROP_RULE_PROPOSAL_{result.profile_id}_{stamp}.yaml"
    alert_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proposal = {
        "profile_id": result.profile_id,
        "status": "human_review_required",
        "detected_at": result.checked_at.isoformat(),
        "changed_sources": list(result.changed_urls),
        "proposed_profile_edits": [],
        "approved": False,
        "note": "Research official changes, edit a new versioned profile, test it, then obtain human approval.",
    }
    proposal_path.write_text(yaml.safe_dump(proposal, sort_keys=False), encoding="utf-8")
    return alert_path, proposal_path
