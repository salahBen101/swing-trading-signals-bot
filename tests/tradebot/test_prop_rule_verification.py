from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tradebot.prop_firms import (
    SourceBaseline,
    VerificationStatus,
    append_verification_record,
    content_sha256,
    fetch_official_page,
    load_prop_profile,
    load_source_baselines,
    verify_profile_sources,
    write_change_proposal,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_prop_profile(ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml")
NOW = datetime(2026, 8, 22, 16, 0, tzinfo=timezone.utc)


def pages() -> dict[str, bytes]:
    return {source.url: f"<html><body><h1>{source.title}</h1><p>reviewed text</p></body></html>".encode() for source in PROFILE.sources}


def baselines(current: dict[str, bytes]) -> dict[str, SourceBaseline]:
    return {url: SourceBaseline(url, content_sha256(body)) for url, body in current.items()}


def test_normalization_ignores_scripts_styles_and_whitespace() -> None:
    a = "<html><style>x{}</style><body> Rule   text <script>volatile()</script></body></html>"
    b = "<body>Rule text</body>"
    assert content_sha256(a) == content_sha256(b)


def test_unchanged_sources_are_safe_to_reuse_but_do_not_edit_profile() -> None:
    current = pages()
    result = verify_profile_sources(PROFILE, baselines(current), checked_at=NOW, fetcher=current.__getitem__)
    assert result.status is VerificationStatus.UNCHANGED
    assert result.safe_to_reuse_reviewed_rules
    assert result.changed_urls == ()


def test_one_changed_source_generates_alert_and_unapproved_proposal(tmp_path: Path) -> None:
    current = pages()
    reviewed = baselines(current)
    changed_url = PROFILE.sources[0].url
    current[changed_url] = b"<html><body>materially changed rule</body></html>"
    result = verify_profile_sources(PROFILE, reviewed, checked_at=NOW, fetcher=current.__getitem__)
    assert result.status is VerificationStatus.CHANGED
    assert not result.safe_to_reuse_reviewed_rules
    assert result.changed_urls == (changed_url,)

    alert, proposal = write_change_proposal(result, tmp_path)
    assert json.loads(alert.read_text(encoding="utf-8"))["status"] == "changed"
    text = proposal.read_text(encoding="utf-8")
    assert "human_review_required" in text
    assert "approved: false" in text
    assert changed_url in text


def test_missing_baseline_and_fetch_error_both_fail_closed() -> None:
    current = pages()
    reviewed = baselines(current)
    reviewed.pop(PROFILE.sources[0].url)
    missing = verify_profile_sources(PROFILE, reviewed, checked_at=NOW, fetcher=current.__getitem__)
    assert missing.status is VerificationStatus.INCOMPLETE_BASELINE
    assert not missing.safe_to_reuse_reviewed_rules

    def broken(url: str):
        raise OSError("offline")

    errored = verify_profile_sources(PROFILE, baselines(current), checked_at=NOW, fetcher=broken)
    assert errored.status is VerificationStatus.FETCH_ERROR
    assert all(check.changed for check in errored.checks)


def test_verification_record_is_append_only_json_lines(tmp_path: Path) -> None:
    current = pages()
    result = verify_profile_sources(PROFILE, baselines(current), checked_at=NOW, fetcher=current.__getitem__)
    path = tmp_path / "verification.jsonl"
    append_verification_record(result, path)
    append_verification_record(result, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["profile_id"] == PROFILE.profile_id for row in rows)


def test_unchanged_result_refuses_spurious_change_proposal(tmp_path: Path) -> None:
    current = pages()
    result = verify_profile_sources(PROFILE, baselines(current), checked_at=NOW, fetcher=current.__getitem__)
    with pytest.raises(ValueError, match="do not need"):
        write_change_proposal(result, tmp_path)


def test_snapshot_loader_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.yaml"
    digest = "a" * 64
    path.write_text(
        "schema_version: 1\nnormalization_version: 1\ncaptured_on: 2026-08-22\n"
        f"sources:\n  https://example.test/rules:\n    content_sha256: {digest}\n",
        encoding="utf-8",
    )
    loaded = load_source_baselines(path)
    assert loaded["https://example.test/rules"].content_sha256 == digest

    path.write_text(path.read_text(encoding="utf-8") + "surprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_source_baselines(path)


def test_reviewed_snapshot_covers_every_checked_in_tradeify_profile_source() -> None:
    snapshot = load_source_baselines(ROOT / "config" / "prop_firms" / "source_snapshots.yaml")
    profiles = (
        load_prop_profile(ROOT / "config" / "prop_firms" / name)
        for name in (
            "tradeify_growth_50k.yaml",
            "tradeify_select_50k.yaml",
            "tradeify_lightning_50k.yaml",
        )
    )
    missing = {
        source.url
        for profile in profiles
        for source in profile.sources
        if source.url not in snapshot
    }
    assert missing == set()


def test_tradeify_fetch_falls_back_only_to_the_exact_official_intercom_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"<html><body>official article</body></html>"

    def fake_urlopen(request, *, timeout):
        requested.append(request.full_url)
        if len(requested) == 1:
            raise HTTPError(request.full_url, 403, "challenged", {}, None)
        return Response()

    monkeypatch.setattr("tradebot.prop_firms.verification.urlopen", fake_urlopen)
    body = fetch_official_page(
        "https://help.tradeify.co/en/articles/10495915-growth-evaluation-accounts"
    )

    assert body == b"<html><body>official article</body></html>"
    assert requested == [
        "https://help.tradeify.co/en/articles/10495915-growth-evaluation-accounts",
        "https://intercom.help/tradeify/en/articles/10495915-growth-evaluation-accounts",
    ]


def test_non_tradeify_source_never_gets_a_guessed_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def broken(request, *, timeout):
        requested.append(request.full_url)
        raise OSError("offline")

    monkeypatch.setattr("tradebot.prop_firms.verification.urlopen", broken)
    with pytest.raises(OSError, match="offline"):
        fetch_official_page("https://example.test/rules")
    assert requested == ["https://example.test/rules"]
