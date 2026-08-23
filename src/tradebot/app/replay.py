"""Safe, local-only Stage-1 market replay runner.

The runner is deliberately narrower than paper trading: it accepts historical DEV bars,
uses only ``SimulatedBroker``, persists the complete execution/audit trail, produces daily
reports, and exits flat.  It has no Stage-2 connectivity and no path to Stage 3/4.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..backtest.prop_engine import PropBacktestResult, run_prop_market_replay
from ..config import Config
from ..core.models import Trade, new_id
from ..core.types import TradingMode
from ..data.schema import validate_bars
from ..data.splits import VALIDATION_START, Split, slice_split
from ..data.store import normalize_bars
from ..deployment.stages import DeploymentStage
from ..features.pipeline import FeatureFrame
from ..journal.db import Journal
from ..reporting import (
    DailyReport,
    DailyReportInput,
    DailyTradeRecord,
    generate_daily_report,
)
from ..risk.prop import MarketDayStatus, PropAccountState
from ..strategy.base import Strategy


class MarketReplaySafetyError(ValueError):
    """The requested replay is outside the fail-closed Stage-1 boundary."""


@dataclass(frozen=True, slots=True)
class DailyReportArtifact:
    session_date: date
    json_path: Path
    text_path: Path


@dataclass(frozen=True, slots=True)
class MarketReplayResult:
    run_id: str
    journal_path: Path
    prop_result: PropBacktestResult
    daily_reports: tuple[DailyReport, ...]
    report_artifacts: tuple[DailyReportArtifact, ...]

    @property
    def ended_flat(self) -> bool:
        return self.prop_result.backtest.ended_flat

    @property
    def working_orders_at_end(self) -> int:
        return self.prop_result.backtest.working_orders_at_end


def load_market_day_statuses(path: str | Path) -> dict[date, MarketDayStatus]:
    """Load a strict ISO-date -> status YAML/JSON map; never synthesize a holiday."""

    calendar_path = Path(path)
    if not calendar_path.exists():
        raise MarketReplaySafetyError(f"market-day map not found: {calendar_path}")
    raw = yaml.safe_load(calendar_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise MarketReplaySafetyError(
            "market-day map must be a non-empty mapping of ISO dates to statuses"
        )
    statuses: dict[date, MarketDayStatus] = {}
    for raw_day, raw_status in raw.items():
        try:
            trade_date = date.fromisoformat(str(raw_day))
        except ValueError as exc:
            raise MarketReplaySafetyError(
                f"invalid market-day date {raw_day!r}; expected YYYY-MM-DD"
            ) from exc
        try:
            status = MarketDayStatus(str(raw_status).strip().lower())
        except ValueError as exc:
            raise MarketReplaySafetyError(
                f"invalid market-day status for {trade_date}: {raw_status!r}"
            ) from exc
        statuses[trade_date] = status
    return statuses


def load_replay_bars(path: str | Path, *, timeframe_seconds: float) -> pd.DataFrame:
    """Read one local CSV/Parquet replay artifact and validate its complete OHLCV shape."""

    data_path = Path(path)
    if not data_path.exists():
        raise MarketReplaySafetyError(f"replay data not found: {data_path}")
    suffix = data_path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(data_path)
    elif suffix in {".csv", ".txt"}:
        raw = pd.read_csv(data_path)
    else:
        raise MarketReplaySafetyError("replay data must be a .parquet, .pq, or .csv file")
    bars = normalize_bars(raw)
    validate_bars(bars, timeframe_seconds=timeframe_seconds)
    return bars


def run_market_replay(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: Config,
    *,
    rule_verification_as_of: datetime,
    market_day_statuses: dict[date, MarketDayStatus],
    minimum_expected_rr: float,
    journal_path: str | Path,
    report_directory: str | Path,
    features: FeatureFrame | None = None,
    dataset_hash: str = "",
    stress_costs: bool = False,
    progress_every: int = 0,
) -> MarketReplayResult:
    """Run a complete Stage-1 DEV replay and persist every safety-relevant artifact."""

    _validate_replay_boundary(
        bars,
        config,
        rule_verification_as_of=rule_verification_as_of,
        market_day_statuses=market_day_statuses,
    )
    dev_bars = slice_split(bars, Split.DEV)
    run_id = new_id("replay")
    db_path = Path(journal_path)
    if str(db_path) == ":memory:":
        raise MarketReplaySafetyError("Stage-1 replay requires a persistent SQLite path")

    first_ts = dev_bars.index[0].to_pydatetime()
    last_ts = dev_bars.index[-1].to_pydatetime()
    journal = Journal(db_path, run_id=run_id)
    journal.start_run(
        mode="MARKET_REPLAY_STAGE_1",
        instrument=config.instrument,
        strategy=strategy.name,
        broker="simulated",
        config={
            "runtime": asdict(config),
            "deployment_stage": int(DeploymentStage.MARKET_REPLAY),
            "split": Split.DEV.value,
            "rule_verification_as_of": rule_verification_as_of.isoformat(),
            "market_day_statuses": {
                day.isoformat(): status.value
                for day, status in sorted(market_day_statuses.items())
            },
            "dataset_hash": dataset_hash,
        },
        started_at=first_ts,
    )
    journal.record_event(
        first_ts,
        "MARKET_REPLAY_SAFETY_CONTEXT",
        "Stage 1 local replay started; simulated broker and DEV split only",
        payload={
            "stage": 1,
            "broker": "simulated",
            "external_connectivity": False,
            "split": "dev",
            "holdout_access": False,
            "rule_verification_as_of": rule_verification_as_of.isoformat(),
            "profile_path": config.prop_firm.profile_path,
            "phase": config.prop_firm.phase,
        },
    )

    try:
        prop_result = run_prop_market_replay(
            bars,
            strategy,
            config,
            rule_verification_as_of=rule_verification_as_of,
            market_day_statuses=market_day_statuses,
            minimum_expected_rr=minimum_expected_rr,
            journal=journal,
            stress_costs=stress_costs,
            features=features,
            dataset_hash=dataset_hash,
            progress_every=progress_every,
        )
        _record_prop_audit(journal, prop_result)
        reports = _daily_reports(prop_result, config)
        artifacts = _write_daily_reports(
            reports, Path(report_directory) / run_id
        )
        for report in reports:
            journal.record_event(
                datetime.combine(
                    report.session_date,
                    datetime.min.time(),
                    tzinfo=first_ts.tzinfo,
                ),
                "DAILY_REPORT",
                f"deterministic daily report for {report.session_date.isoformat()}",
                payload=report.to_dict(),
            )
        journal.record_event(
            last_ts,
            "MARKET_REPLAY_COMPLETE",
            "Stage 1 replay completed flat with no working orders",
            payload={
                "ended_flat": prop_result.backtest.ended_flat,
                "working_orders_at_end": prop_result.backtest.working_orders_at_end,
                "trades": len(prop_result.trades),
                "daily_reports": len(reports),
            },
        )
        journal.end_run(last_ts)
    except Exception as exc:
        journal.record_event(
            last_ts,
            "MARKET_REPLAY_FAILED",
            f"Stage 1 replay failed closed: {type(exc).__name__}: {exc}",
            level="ERROR",
        )
        journal.end_run(last_ts)
        raise
    finally:
        journal.close()

    return MarketReplayResult(
        run_id=run_id,
        journal_path=db_path.resolve(),
        prop_result=prop_result,
        daily_reports=reports,
        report_artifacts=artifacts,
    )


def _validate_replay_boundary(
    bars: pd.DataFrame,
    config: Config,
    *,
    rule_verification_as_of: datetime,
    market_day_statuses: dict[date, MarketDayStatus],
) -> None:
    stage = config.deployment.stage
    if stage != int(DeploymentStage.MARKET_REPLAY):
        if stage in (int(DeploymentStage.PROP_EVALUATION), int(DeploymentStage.FUNDED)):
            raise MarketReplaySafetyError(
                "Stage 1 replay categorically refuses Stage 3/4 authorization"
            )
        raise MarketReplaySafetyError(
            f"market replay requires deployment Stage 1, not Stage {stage}"
        )
    if config.mode is TradingMode.LIVE or config.broker.environment == "live":
        raise MarketReplaySafetyError(
            "Stage 1 categorically refuses live mode or a live broker environment"
        )
    if config.broker.adapter != "simulated":
        raise MarketReplaySafetyError(
            "Stage 1 categorically refuses Tradovate/external adapters; use simulated"
        )
    if config.mode is not TradingMode.PAPER:
        raise MarketReplaySafetyError(
            "market replay requires PAPER mode and deployment Stage 1"
        )
    config.validate()
    if (
        rule_verification_as_of.tzinfo is None
        or rule_verification_as_of.utcoffset() is None
    ):
        raise MarketReplaySafetyError("rule_verification_as_of must be timezone-aware")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise MarketReplaySafetyError("replay bars require a timezone-aware DatetimeIndex")
    if (bars.index >= VALIDATION_START).any():
        raise MarketReplaySafetyError(
            "Stage 1 accepts only a DEV extract and refuses VALIDATION/HOLDOUT rows"
        )
    dev_bars = slice_split(bars, Split.DEV)
    if dev_bars.empty:
        raise MarketReplaySafetyError("DEV split contains no replay bars")
    validate_bars(dev_bars, timeframe_seconds=config.bar_seconds)

    replay_dates = {timestamp.date() for timestamp in dev_bars.index}
    missing = sorted(replay_dates - set(market_day_statuses))
    invalid = sorted(
        day
        for day in replay_dates & set(market_day_statuses)
        if not isinstance(market_day_statuses[day], MarketDayStatus)
        or market_day_statuses[day] is MarketDayStatus.UNKNOWN
    )
    if missing or invalid:
        pieces = []
        if missing:
            pieces.append("missing=" + ",".join(day.isoformat() for day in missing))
        if invalid:
            pieces.append("unknown/invalid=" + ",".join(day.isoformat() for day in invalid))
        raise MarketReplaySafetyError(
            "market-day map must explicitly cover every DEV replay date ("
            + "; ".join(pieces)
            + ")"
        )


def _record_prop_audit(journal: Journal, result: PropBacktestResult) -> None:
    for trace in result.decision_traces:
        # ExecutionEngine writes ENTRY_EVALUATE synchronously before it can act on the
        # decision.  Do not duplicate that record here; broker-side re-verification is a
        # distinct ThreeLayerDecision and is persisted after the driver completes.
        if trace["operation"] == "ENTRY_EVALUATE":
            continue
        captured = datetime.fromisoformat(trace["captured_at"])
        journal.record_event(
            captured,
            "RISK_DECISION",
            f"{trace['operation']} {'ALLOWED' if trace['allowed'] else 'REFUSED'}",
            level="INFO" if trace["allowed"] else "WARN",
            payload=trace,
        )
    for point in result.prop_state_points:
        journal.record_event(
            point.timestamp,
            "PROP_ACCOUNT_STATE",
            f"{point.kind} prop-account state",
            payload={"kind": point.kind, "state": point.state.to_dict()},
        )


def _daily_reports(
    result: PropBacktestResult, config: Config
) -> tuple[DailyReport, ...]:
    trades_by_day: dict[date, list[Trade]] = defaultdict(list)
    for trade in result.trades:
        trades_by_day[trade.exit_time.date()].append(trade)

    decisions_by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for trace in result.decision_traces:
        if trace["operation"] == "ENTRY_EVALUATE":
            decisions_by_day[datetime.fromisoformat(trace["captured_at"]).date()].append(trace)

    strategy_rejections_by_day: dict[date, list[Any]] = defaultdict(list)
    for rejection in result.rejections:
        if rejection.stage.upper() == "STRATEGY":
            strategy_rejections_by_day[rejection.timestamp.date()].append(rejection)

    eod_states: dict[date, PropAccountState] = {
        point.timestamp.date(): point.state
        for point in result.prop_state_points
        if point.kind == "EOD"
    }
    reports: list[DailyReport] = []
    for session_date in sorted(eod_states):
        state = eod_states[session_date]
        decisions = decisions_by_day.get(session_date, [])
        strategy_rejections = strategy_rejections_by_day.get(session_date, [])
        accepted = sum(bool(trace["allowed"]) for trace in decisions)
        rejection_counts = Counter(
            _decision_rejection_label(trace)
            for trace in decisions
            if not trace["allowed"]
        )
        rejection_counts.update(
            f"STRATEGY:{rejection.reason.value}" for rejection in strategy_rejections
        )
        daily_trades = tuple(
            DailyTradeRecord(
                trade_id=trade.trade_id,
                instrument=trade.instrument,
                strategy=trade.strategy,
                setup=" & ".join(trade.entry_conditions) or trade.strategy,
                execution=trade.exit_reason.value,
                pnl_usd=trade.net_pnl_usd,
                r_multiple=trade.r_multiple,
                slippage_usd=trade.slippage_usd,
                fees_usd=trade.commission_usd,
            )
            for trade in sorted(trades_by_day.get(session_date, []), key=lambda item: item.exit_time)
        )
        reports.append(
            generate_daily_report(
                DailyReportInput(
                    session_date=session_date,
                    account_balance_usd=state.current_balance_usd,
                    prop_failure_level_usd=state.drawdown_floor_usd,
                    trades=daily_trades,
                    signals_generated=len(decisions) + len(strategy_rejections),
                    signals_accepted=accepted,
                    rejection_counts=rejection_counts,
                    rule_violations=state.hard_breach_reasons,
                    max_trades_per_day=config.risk.daily.max_trades_per_day,
                    max_daily_loss_usd=config.risk.daily.max_daily_loss_usd,
                )
            )
        )
    return tuple(reports)


def _decision_rejection_label(trace: dict[str, Any]) -> str:
    context = trace["context"]
    if context["reason_codes"]:
        return f"CONTEXT:{context['reason_codes'][0]}"
    strategy = trace["strategy"]
    if strategy.get("failure_codes"):
        return f"STRATEGY:{strategy['failure_codes'][0]}"
    personal = trace["personal"]
    if personal.get("reason_code"):
        return f"PERSONAL:{personal['reason_code']}"
    prop = trace["prop_firm"]
    if prop.get("reason_codes"):
        return f"PROP:{prop['reason_codes'][0]}"
    return "UNKNOWN_FAIL_CLOSED"


def _write_daily_reports(
    reports: tuple[DailyReport, ...], directory: Path
) -> tuple[DailyReportArtifact, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[DailyReportArtifact] = []
    for report in reports:
        stem = report.session_date.isoformat()
        json_path = directory / f"{stem}.json"
        text_path = directory / f"{stem}.txt"
        json_path.write_text(report.to_json() + "\n", encoding="utf-8")
        text_path.write_text(report.to_text() + "\n", encoding="utf-8")
        artifacts.append(DailyReportArtifact(report.session_date, json_path, text_path))
    return tuple(artifacts)


__all__ = [
    "DailyReportArtifact",
    "MarketReplayResult",
    "MarketReplaySafetyError",
    "load_market_day_statuses",
    "load_replay_bars",
    "run_market_replay",
]
