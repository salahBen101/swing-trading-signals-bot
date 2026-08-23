"""Fail-closed operator CLI for research checks and local Stage-1 replay.

The only execution command is a historical DEV replay through ``SimulatedBroker``.  This
module has no external broker command, no credential path, and no ability to authorize
Stage 2, Stage 3, Stage 4, live, or Tradovate execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .. import __version__
from ..config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ConfigError, load_config
from ..prop_firms import (
    PropProfileError,
    VerificationStatus,
    append_verification_record,
    fetch_official_page,
    load_prop_profile,
    load_source_baselines,
    verify_profile_sources,
    write_change_proposal,
)


EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_ERROR = 2

_PROFILES_DIR = PROJECT_ROOT / "config" / "prop_firms"
_DEFAULT_SNAPSHOT = _PROFILES_DIR / "source_snapshots.yaml"
_DEFAULT_RECORD = PROJECT_ROOT / "logs" / "prop_rule_verification.jsonl"
_DEFAULT_PROPOSAL_DIR = PROJECT_ROOT / "logs" / "prop_rule_alerts"
_DEFAULT_REPLAY_JOURNAL = PROJECT_ROOT / "logs" / "market_replay.sqlite3"
_DEFAULT_REPLAY_REPORTS = PROJECT_ROOT / "logs" / "market_replay_reports"


class CliError(ValueError):
    """Operator-facing input error that should not produce a traceback."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradebot",
        description=(
            "Safety checks and local Stage-1 simulated replay for the MNQ prop-account "
            "system. This CLI cannot contact an external broker or authorize Stage 2-4."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tradebot {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "config-check",
        help="load and validate typed runtime configuration without starting a runner",
    )
    config.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config.add_argument(
        "--at",
        help="timezone-aware ISO-8601 check time (default: current UTC time)",
    )
    _add_output_format(config)
    config.set_defaults(handler=_run_config_check)

    profiles = commands.add_parser(
        "prop-profile-check",
        help="validate versioned prop profiles and report research/deployment blockers",
    )
    profiles.add_argument(
        "--profile",
        type=Path,
        action="append",
        help="profile YAML; repeat for more than one (default: all checked-in 50K profiles)",
    )
    profiles.add_argument(
        "--at",
        help="timezone-aware ISO-8601 check time (default: current UTC time)",
    )
    _add_output_format(profiles)
    profiles.set_defaults(handler=_run_profile_check)

    verify = commands.add_parser(
        "verify-rules",
        help="compare current official pages with reviewed fingerprints and record the result",
    )
    verify.add_argument(
        "--profile",
        type=Path,
        action="append",
        help="profile YAML; repeat for more than one (default: all checked-in 50K profiles)",
    )
    verify.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    verify.add_argument(
        "--record",
        type=Path,
        default=_DEFAULT_RECORD,
        help="append-only JSONL verification record",
    )
    verify.add_argument(
        "--no-record",
        action="store_true",
        help="diagnostic mode: do not append the verification record",
    )
    verify.add_argument(
        "--proposal-dir",
        type=Path,
        default=_DEFAULT_PROPOSAL_DIR,
        help="directory for alerts and unapproved review proposals",
    )
    verify.add_argument(
        "--no-proposals",
        action="store_true",
        help="diagnostic mode: do not write alert/proposal files",
    )
    verify.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="per-source network timeout (must be positive)",
    )
    verify.add_argument(
        "--at",
        help="timezone-aware ISO-8601 check time (default: current UTC time)",
    )
    _add_output_format(verify)
    verify.set_defaults(handler=_run_verify_rules)

    replay = commands.add_parser(
        "market-replay",
        help="run a persistent Stage-1 DEV replay through the simulated broker only",
        description=(
            "Run historical DEV data through the Stage-1 three-layer risk path and "
            "SimulatedBroker. Stage 2/3/4, live, Tradovate, and HOLDOUT are refused."
        ),
    )
    replay.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    replay.add_argument(
        "--data",
        type=Path,
        required=True,
        help="DEV-only local CSV or Parquet OHLCV artifact",
    )
    replay.add_argument(
        "--calendar",
        type=Path,
        required=True,
        help="explicit ISO-date to regular/early_close/closed map covering every date",
    )
    replay.add_argument(
        "--rules-at",
        required=True,
        help="current timezone-aware ISO-8601 rule-verification timestamp",
    )
    replay.add_argument(
        "--minimum-rr",
        type=float,
        required=True,
        help="minimum StrategyGate reward:risk ratio; must be positive",
    )
    replay.add_argument(
        "--journal",
        type=Path,
        default=_DEFAULT_REPLAY_JOURNAL,
        help="persistent SQLite audit journal (in-memory journals are refused)",
    )
    replay.add_argument(
        "--report-dir",
        type=Path,
        default=_DEFAULT_REPLAY_REPORTS,
        help="root directory for deterministic per-session JSON/text reports",
    )
    replay.add_argument(
        "--stress-costs",
        action="store_true",
        help="apply the configured stress cost multiplier",
    )
    replay.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="emit engine progress every N bars; zero disables progress output",
    )
    _add_output_format(replay)
    replay.set_defaults(handler=_run_market_replay)
    return parser


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _checked_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return _parse_timestamp(value, option="--at")


def _parse_timestamp(value: str, *, option: str) -> datetime:
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CliError(f"{option} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CliError(f"{option} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _profile_paths(values: list[Path] | None) -> tuple[Path, ...]:
    if values:
        return tuple(values)
    discovered = tuple(sorted(_PROFILES_DIR.glob("tradeify_*_50k.yaml")))
    if not discovered:
        raise CliError(f"no checked-in 50K prop profiles found under {_PROFILES_DIR}")
    return discovered


def _emit(payload: object, lines: list[str], *, output_format: str, stdout: TextIO) -> None:
    if output_format == "json":
        stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        stdout.write("\n".join(lines) + "\n")


def _run_config_check(args: argparse.Namespace, *, stdout: TextIO) -> int:
    config = load_config(args.config)
    checked_at = _checked_at(args.at)
    profile_path = Path(config.prop_firm.profile_path)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    profile = load_prop_profile(profile_path)
    safety_blockers: list[str] = []
    personal = config.risk
    if personal.per_trade.max_risk_per_trade_usd > 200.0:
        safety_blockers.append("max risk per trade exceeds the $200 personal ceiling")
    if personal.daily.max_trades_per_day > 1:
        safety_blockers.append("daily trade quota exceeds the one-trade personal ceiling")
    if personal.daily.max_daily_loss_usd > 200.0:
        safety_blockers.append("daily loss limit exceeds the $200 personal ceiling")
    if personal.max_open_positions > 1:
        safety_blockers.append("open-position limit exceeds the one-position personal ceiling")
    if abs(personal.starting_equity_usd - profile.account_size_usd) > 1e-9:
        safety_blockers.append("starting equity differs from the selected prop profile")
    if not profile.rules_are_fresh(checked_at):
        safety_blockers.append("selected official prop profile is stale")
    if config.broker.environment == "live":
        safety_blockers.append("live broker environment is forbidden")
    payload = {
        "status": "blocked" if safety_blockers else "valid",
        "config_path": str(Path(args.config).resolve()),
        "checked_at": checked_at.isoformat(),
        "mode": config.mode.value,
        "deployment_stage": config.deployment.stage,
        "instrument": config.instrument,
        "strategy": config.strategy.name,
        "prop_profile_path": config.prop_firm.profile_path,
        "prop_phase": config.prop_firm.phase,
        "personal_limits": {
            "max_risk_per_trade_usd": config.risk.per_trade.max_risk_per_trade_usd,
            "max_trades_per_day": config.risk.daily.max_trades_per_day,
            "max_daily_loss_usd": config.risk.daily.max_daily_loss_usd,
            "max_open_positions": config.risk.max_open_positions,
        },
        "cli_can_execute_orders": False,
        "stage_3_4_authorized": False,
        "safety_blockers": safety_blockers,
    }
    lines = [
        f"{'BLOCKED' if safety_blockers else 'VALID'}  {payload['config_path']}",
        f"  mode/stage       {payload['mode']} / {payload['deployment_stage']}",
        f"  instrument       {payload['instrument']}",
        f"  strategy         {payload['strategy']}",
        f"  prop phase       {payload['prop_phase']}",
        "  execution        not started by config-check",
        "  Stage 3/4        NOT AUTHORIZED",
    ]
    for blocker in safety_blockers:
        lines.append(f"  blocker          {blocker}")
    _emit(payload, lines, output_format=args.format, stdout=stdout)
    return EXIT_BLOCKED if safety_blockers else EXIT_OK


def _profile_summary(path: Path, checked_at: datetime) -> tuple[dict, list[str]]:
    profile = load_prop_profile(path)
    fresh = profile.rules_are_fresh(checked_at)
    blockers: list[str] = []
    if not fresh:
        blockers.append(
            f"profile verification age exceeds {profile.reverify_after_hours} hours"
        )
    if profile.ambiguity_notes:
        blockers.append("profile contains unresolved ambiguity notes")
    if not profile.compliance.automation_allowed:
        blockers.append("firm profile does not permit automation")
    if not profile.compliance.tradovate_api_allowed:
        blockers.append("Tradovate API is not an approved Evaluation/Sim Funded route")
    row = {
        "status": "valid" if fresh else "stale",
        "path": str(path.resolve()),
        "profile_id": profile.profile_id,
        "firm": profile.firm,
        "program": profile.program,
        "verified_on": profile.verified_on.isoformat(),
        "checked_at": checked_at.isoformat(),
        "reverify_after_hours": profile.reverify_after_hours,
        "fresh": fresh,
        "source_count": len(profile.sources),
        "phases": [rules.name for rules in profile.phases],
        "ambiguity_notes": list(profile.ambiguity_notes),
        "stage_3_4_blocked": bool(blockers),
        "stage_3_4_blockers": blockers,
    }
    lines = [
        f"{'VALID' if fresh else 'STALE'}  {profile.profile_id}",
        f"  profile          {path.resolve()}",
        f"  reviewed         {profile.verified_on.isoformat()} ({profile.reverify_after_hours}h maximum age)",
        f"  sources/phases   {len(profile.sources)} / {', '.join(row['phases'])}",
        f"  Stage 3/4        {'BLOCKED' if blockers else 'not authorized by this command'}",
    ]
    for blocker in blockers:
        lines.append(f"  blocker          {blocker}")
    return row, lines


def _run_profile_check(args: argparse.Namespace, *, stdout: TextIO) -> int:
    checked_at = _checked_at(args.at)
    rows: list[dict] = []
    lines: list[str] = []
    for path in _profile_paths(args.profile):
        row, row_lines = _profile_summary(path, checked_at)
        rows.append(row)
        if lines:
            lines.append("")
        lines.extend(row_lines)
    payload = {
        "status": "stale" if any(not row["fresh"] for row in rows) else "valid",
        "checked_at": checked_at.isoformat(),
        "profiles": rows,
        "cli_can_authorize_stage_3_4": False,
    }
    _emit(payload, lines, output_format=args.format, stdout=stdout)
    return EXIT_BLOCKED if any(not row["fresh"] for row in rows) else EXIT_OK


def _run_verify_rules(args: argparse.Namespace, *, stdout: TextIO) -> int:
    if args.timeout_seconds <= 0:
        raise CliError("--timeout-seconds must be positive")
    checked_at = _checked_at(args.at)
    baselines = load_source_baselines(args.snapshot)
    rows: list[dict] = []
    lines: list[str] = []
    operational_error = False
    fetched: dict[str, bytes | str | Exception] = {}

    def cached_fetch(url: str) -> bytes | str:
        if url not in fetched:
            try:
                fetched[url] = fetch_official_page(
                    url, timeout_seconds=args.timeout_seconds
                )
            except Exception as exc:  # verifier records the error and remains fail-closed
                fetched[url] = exc
        value = fetched[url]
        if isinstance(value, Exception):
            raise value
        return value

    for path in _profile_paths(args.profile):
        profile = load_prop_profile(path)
        result = verify_profile_sources(
            profile,
            baselines,
            checked_at=checked_at,
            fetcher=cached_fetch,
        )
        profile_fresh = profile.rules_are_fresh(checked_at)
        record_path: str | None = None
        proposal_paths: list[str] = []
        try:
            if not args.no_record:
                append_verification_record(result, args.record)
                record_path = str(Path(args.record).resolve())
            if not result.safe_to_reuse_reviewed_rules and not args.no_proposals:
                alert, proposal = write_change_proposal(result, args.proposal_dir)
                proposal_paths = [str(alert.resolve()), str(proposal.resolve())]
        except OSError as exc:
            operational_error = True
            proposal_paths.append(f"write failed: {exc}")

        row = result.to_dict()
        row.update(
            {
                "profile_path": str(path.resolve()),
                "profile_metadata_fresh": profile_fresh,
                "record_path": record_path,
                "alert_or_proposal_paths": proposal_paths,
                "profile_files_mutated": False,
            }
        )
        rows.append(row)
        if lines:
            lines.append("")
        label = result.status.value.upper()
        if not profile_fresh:
            label = f"{label} / STALE PROFILE"
        lines.extend(
            [
                f"{label}  {profile.profile_id}",
                f"  checked          {checked_at.isoformat()}",
                f"  official sources {len(result.checks)}",
                f"  reviewed profile {'fresh' if profile_fresh else 'STALE'}",
                f"  audit record     {record_path or 'not written (diagnostic mode)'}",
                "  profile edits    none; human review is mandatory for any change",
            ]
        )
        for check in result.checks:
            if check.changed:
                detail = check.error or "content fingerprint differs"
                lines.append(f"  BLOCKED source   {check.url} ({detail})")
        for proposal_path in proposal_paths:
            lines.append(f"  review artifact  {proposal_path}")

    blocked = any(
        row["status"] != VerificationStatus.UNCHANGED.value
        or not row["profile_metadata_fresh"]
        for row in rows
    )
    payload = {
        "status": "error" if operational_error else ("blocked" if blocked else "unchanged"),
        "checked_at": checked_at.isoformat(),
        "snapshot_path": str(Path(args.snapshot).resolve()),
        "profiles": rows,
        "stage_3_4_authorized": False,
    }
    _emit(payload, lines, output_format=args.format, stdout=stdout)
    if operational_error:
        return EXIT_ERROR
    return EXIT_BLOCKED if blocked else EXIT_OK


def _run_market_replay(args: argparse.Namespace, *, stdout: TextIO) -> int:
    """Run the bounded local replay without exposing a split or broker selector."""

    if args.minimum_rr <= 0:
        raise CliError("--minimum-rr must be positive")
    if args.progress_every < 0:
        raise CliError("--progress-every must be non-negative")
    rules_at = _parse_timestamp(args.rules_at, option="--rules-at")
    config = load_config(args.config)

    # Imports remain local to the one bounded command.  There is no CLI route that can
    # construct an external adapter or authorize a later deployment stage.
    from .replay import (
        load_market_day_statuses,
        load_replay_bars,
        run_market_replay as execute_market_replay,
    )
    from ..strategy.registry import UnknownStrategy, build_strategy

    try:
        strategy = build_strategy(config.strategy.name, config.strategy.params)
    except UnknownStrategy as exc:
        raise CliError(str(exc)) from exc

    bars = load_replay_bars(args.data, timeframe_seconds=config.bar_seconds)
    market_days = load_market_day_statuses(args.calendar)
    result = execute_market_replay(
        bars,
        strategy,
        config,
        rule_verification_as_of=rules_at,
        market_day_statuses=market_days,
        minimum_expected_rr=args.minimum_rr,
        journal_path=args.journal,
        report_directory=args.report_dir,
        dataset_hash=_sha256_file(args.data),
        stress_costs=args.stress_costs,
        progress_every=args.progress_every,
    )
    prop = result.prop_result
    payload = {
        "status": "complete",
        "run_id": result.run_id,
        "stage": int(prop.deployment_stage),
        "mode": config.mode.value,
        "broker": "simulated",
        "external_connectivity": False,
        "split": prop.split,
        "holdout_access": False,
        "profile_id": prop.profile_id,
        "phase": prop.rule_set_name,
        "rule_verification_as_of": prop.rule_verification_as_of.isoformat(),
        "journal_path": str(result.journal_path),
        "daily_report_paths": [
            {
                "date": artifact.session_date.isoformat(),
                "json": str(artifact.json_path.resolve()),
                "text": str(artifact.text_path.resolve()),
            }
            for artifact in result.report_artifacts
        ],
        "bars": prop.backtest.bars,
        "sessions": prop.backtest.sessions,
        "trades": len(prop.trades),
        "net_pnl_usd": prop.final_equity - prop.backtest.starting_equity,
        "ended_flat": result.ended_flat,
        "working_orders_at_end": result.working_orders_at_end,
        "stage_2_3_4_authorized": False,
        "limitations": list(prop.limitations),
    }
    lines = [
        f"COMPLETE  Stage 1 market replay {result.run_id}",
        "  route            simulated broker / local historical DEV only",
        f"  profile/phase    {prop.profile_id} / {prop.rule_set_name}",
        f"  bars/sessions    {prop.backtest.bars} / {prop.backtest.sessions}",
        f"  trades/net P&L   {len(prop.trades)} / ${payload['net_pnl_usd']:,.2f}",
        f"  final safety     flat={result.ended_flat}, working orders={result.working_orders_at_end}",
        f"  audit journal    {result.journal_path}",
        f"  daily reports    {len(result.report_artifacts)}",
        "  Stage 2/3/4      NOT AUTHORIZED; no external connectivity",
    ]
    _emit(payload, lines, output_format=args.format, stdout=stdout)
    return EXIT_OK


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run a CLI command and return a process exit code.

    Expected configuration, profile, snapshot, network, and filesystem failures are
    concise operator errors.  They never fall through into a permissive result.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args, stdout=out))
    except (CliError, ConfigError, PropProfileError, OSError, ValueError) as exc:
        err.write(f"ERROR {args.command}: {exc}\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point
    raise SystemExit(main())
