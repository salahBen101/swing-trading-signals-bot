from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from tradebot.app import cli
from tradebot.prop_firms import content_sha256, load_prop_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
CONFIG = ROOT / "config" / "tradebot.yaml"
NOW = "2026-08-22T16:00:00Z"


def invoke(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = cli.main(list(args), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_help_states_the_exact_simulated_only_execution_boundary() -> None:
    help_text = cli.build_parser().format_help()

    assert "local Stage-1 simulated replay" in help_text
    assert "cannot contact an external broker" in help_text
    assert "authorize Stage 2-4" in help_text
    assert "cannot place orders" not in help_text


def fake_pages() -> dict[str, bytes]:
    profile = load_prop_profile(PROFILE)
    return {
        source.url: f"<html><body>{source.title} reviewed text</body></html>".encode()
        for source in profile.sources
    }


def write_snapshots(path: Path, pages: dict[str, bytes], *, omit: str | None = None) -> None:
    sources = {
        url: {"content_sha256": content_sha256(body)}
        for url, body in pages.items()
        if url != omit
    }
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "normalization_version": 1,
                "captured_on": "2026-08-22",
                "sources": sources,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_config_check_is_sanitized_research_only_and_stage_zero() -> None:
    code, stdout, stderr = invoke(
        "config-check", "--config", str(CONFIG), "--at", NOW, "--format", "json"
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["status"] == "valid"
    assert payload["deployment_stage"] == 0
    assert payload["cli_can_execute_orders"] is False
    assert payload["stage_3_4_authorized"] is False
    assert "account_id" not in stdout


def test_config_check_blocks_weakened_personal_risk(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["risk"]["per_trade"]["max_risk_per_trade_usd"] = 201
    path = tmp_path / "weakened.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    code, stdout, stderr = invoke(
        "config-check", "--config", str(path), "--at", NOW, "--format", "json"
    )

    assert code == cli.EXIT_BLOCKED
    assert stderr == ""
    assert "exceeds the $200 personal ceiling" in stdout


def test_config_check_refuses_stage_three_ordinary_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["deployment"]["stage"] = 3
    path = tmp_path / "stage3.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    code, stdout, stderr = invoke(
        "config-check", "--config", str(path), "--at", NOW
    )

    assert code == cli.EXIT_ERROR
    assert stdout == ""
    assert "Stage 3/4 requires" in stderr


def test_profile_check_reports_blockers_but_accepts_fresh_research_profile() -> None:
    code, stdout, stderr = invoke(
        "prop-profile-check",
        "--profile",
        str(PROFILE),
        "--at",
        NOW,
        "--format",
        "json",
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_OK
    assert stderr == ""
    row = payload["profiles"][0]
    assert row["fresh"] is True
    assert row["stage_3_4_blocked"] is True
    assert any("ambiguity" in reason for reason in row["stage_3_4_blockers"])
    assert payload["cli_can_authorize_stage_3_4"] is False


def test_profile_check_returns_nonzero_when_profile_metadata_is_stale() -> None:
    code, stdout, stderr = invoke(
        "prop-profile-check",
        "--profile",
        str(PROFILE),
        "--at",
        "2026-08-23T00:00:01Z",
        "--format",
        "json",
    )

    assert code == cli.EXIT_BLOCKED
    assert stderr == ""
    assert json.loads(stdout)["status"] == "stale"


def test_profile_check_rejects_naive_check_time() -> None:
    code, stdout, stderr = invoke(
        "prop-profile-check", "--profile", str(PROFILE), "--at", "2026-08-22T12:00:00"
    )

    assert code == cli.EXIT_ERROR
    assert stdout == ""
    assert "timezone offset" in stderr


def test_verify_rules_records_unchanged_result_without_mutating_profile(
    tmp_path: Path, monkeypatch,
) -> None:
    pages = fake_pages()
    snapshot = tmp_path / "snapshots.yaml"
    record = tmp_path / "verification.jsonl"
    proposals = tmp_path / "alerts"
    write_snapshots(snapshot, pages)
    before = PROFILE.read_bytes()
    monkeypatch.setattr(
        cli,
        "fetch_official_page",
        lambda url, *, timeout_seconds: pages[url],
    )

    code, stdout, stderr = invoke(
        "verify-rules",
        "--profile",
        str(PROFILE),
        "--snapshot",
        str(snapshot),
        "--record",
        str(record),
        "--proposal-dir",
        str(proposals),
        "--at",
        NOW,
        "--format",
        "json",
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["status"] == "unchanged"
    assert payload["stage_3_4_authorized"] is False
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "unchanged"
    assert not proposals.exists()
    assert PROFILE.read_bytes() == before


def test_verify_rules_changed_source_fails_closed_and_writes_unapproved_proposal(
    tmp_path: Path, monkeypatch,
) -> None:
    pages = fake_pages()
    snapshot = tmp_path / "snapshots.yaml"
    record = tmp_path / "verification.jsonl"
    proposals = tmp_path / "alerts"
    write_snapshots(snapshot, pages)
    changed_url = next(iter(pages))
    pages[changed_url] = b"<html><body>changed official terms</body></html>"
    monkeypatch.setattr(
        cli,
        "fetch_official_page",
        lambda url, *, timeout_seconds: pages[url],
    )

    code, stdout, stderr = invoke(
        "verify-rules",
        "--profile",
        str(PROFILE),
        "--snapshot",
        str(snapshot),
        "--record",
        str(record),
        "--proposal-dir",
        str(proposals),
        "--at",
        NOW,
        "--format",
        "json",
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_BLOCKED
    assert stderr == ""
    assert payload["status"] == "blocked"
    row = payload["profiles"][0]
    assert row["status"] == "changed"
    assert row["profile_files_mutated"] is False
    artifacts = list(proposals.iterdir())
    assert len(artifacts) == 2
    proposal = next(path for path in artifacts if path.suffix == ".yaml")
    assert "approved: false" in proposal.read_text(encoding="utf-8")


def test_verify_rules_incomplete_baseline_returns_nonzero(
    tmp_path: Path, monkeypatch,
) -> None:
    pages = fake_pages()
    missing_url = next(iter(pages))
    snapshot = tmp_path / "snapshots.yaml"
    write_snapshots(snapshot, pages, omit=missing_url)
    monkeypatch.setattr(
        cli,
        "fetch_official_page",
        lambda url, *, timeout_seconds: pages[url],
    )

    code, stdout, stderr = invoke(
        "verify-rules",
        "--profile",
        str(PROFILE),
        "--snapshot",
        str(snapshot),
        "--no-record",
        "--no-proposals",
        "--at",
        NOW,
        "--format",
        "json",
    )

    assert code == cli.EXIT_BLOCKED
    assert stderr == ""
    assert json.loads(stdout)["profiles"][0]["status"] == "incomplete_baseline"


def test_verify_rules_fetch_failure_returns_nonzero_and_never_claims_safe(
    tmp_path: Path, monkeypatch,
) -> None:
    pages = fake_pages()
    snapshot = tmp_path / "snapshots.yaml"
    write_snapshots(snapshot, pages)

    def offline(_url: str, *, timeout_seconds: float) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(cli, "fetch_official_page", offline)
    code, stdout, stderr = invoke(
        "verify-rules",
        "--profile",
        str(PROFILE),
        "--snapshot",
        str(snapshot),
        "--no-record",
        "--no-proposals",
        "--at",
        NOW,
        "--format",
        "json",
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_BLOCKED
    assert stderr == ""
    assert payload["profiles"][0]["status"] == "fetch_error"
    assert payload["profiles"][0]["safe_to_reuse_reviewed_rules"] is False


def test_verify_rules_fetches_shared_official_pages_once_per_run(
    tmp_path: Path, monkeypatch,
) -> None:
    select = ROOT / "config" / "prop_firms" / "tradeify_select_50k.yaml"
    profiles = (load_prop_profile(PROFILE), load_prop_profile(select))
    pages = {
        source.url: f"<html><body>{source.title} reviewed text</body></html>".encode()
        for profile in profiles
        for source in profile.sources
    }
    snapshot = tmp_path / "snapshots.yaml"
    write_snapshots(snapshot, pages)
    calls: dict[str, int] = {}

    def fetch(url: str, *, timeout_seconds: float) -> bytes:
        calls[url] = calls.get(url, 0) + 1
        return pages[url]

    monkeypatch.setattr(cli, "fetch_official_page", fetch)
    code, _, stderr = invoke(
        "verify-rules",
        "--profile",
        str(PROFILE),
        "--profile",
        str(select),
        "--snapshot",
        str(snapshot),
        "--no-record",
        "--no-proposals",
        "--at",
        NOW,
    )

    assert code == cli.EXIT_OK
    assert stderr == ""
    assert set(calls) == set(pages)
    assert set(calls.values()) == {1}


def test_market_replay_command_is_dev_simulated_persistent_and_flat(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["mode"] = "PAPER"
    raw["deployment"]["stage"] = 1
    raw["broker"]["adapter"] = "simulated"
    raw["broker"]["environment"] = "demo"
    raw["broker"]["simulated"]["latency_ms"] = 0
    config_path = tmp_path / "stage1.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    index = pd.date_range(
        "2022-04-01 09:30", periods=390, freq="1min", tz="America/New_York"
    )
    bars = pd.DataFrame(
        {
            "open": 18_000.0,
            "high": 18_001.0,
            "low": 17_999.0,
            "close": 18_000.0,
            "volume": 100.0,
        },
        index=index,
    )
    bars.index.name = "timestamp"
    data_path = tmp_path / "dev.csv"
    bars.to_csv(data_path)
    calendar_path = tmp_path / "calendar.yaml"
    calendar_path.write_text("2022-04-01: regular\n", encoding="utf-8")
    journal_path = tmp_path / "replay.sqlite3"
    report_path = tmp_path / "reports"

    code, stdout, stderr = invoke(
        "market-replay",
        "--config",
        str(config_path),
        "--data",
        str(data_path),
        "--calendar",
        str(calendar_path),
        "--rules-at",
        NOW,
        "--minimum-rr",
        "2.0",
        "--journal",
        str(journal_path),
        "--report-dir",
        str(report_path),
        "--format",
        "json",
    )
    payload = json.loads(stdout)

    assert code == cli.EXIT_OK
    assert stderr == ""
    assert payload["status"] == "complete"
    assert payload["stage"] == 1
    assert payload["mode"] == "PAPER"
    assert payload["broker"] == "simulated"
    assert payload["external_connectivity"] is False
    assert payload["split"] == "dev"
    assert payload["holdout_access"] is False
    assert payload["ended_flat"] is True
    assert payload["working_orders_at_end"] == 0
    assert payload["stage_2_3_4_authorized"] is False
    assert journal_path.exists()
    assert len(payload["daily_report_paths"]) == 1
    assert Path(payload["daily_report_paths"][0]["json"]).exists()


def test_cli_source_has_no_direct_execution_or_broker_imports() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "import ..broker" not in source
    assert "import ..execution" not in source
    assert "from ..broker" not in source
    assert "from ..execution" not in source
