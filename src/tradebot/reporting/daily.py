"""Pure daily-session reporting with conservative consistency checks.

The future paper runner can build these inputs from its journal and prop-account state.
Keeping the calculation pure makes the report reproducible and prevents reporting code
from becoming another broker/control path.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class DailyTradeRecord:
    trade_id: str
    instrument: str
    strategy: str
    setup: str
    execution: str
    pnl_usd: float
    r_multiple: float
    slippage_usd: float
    fees_usd: float

    def __post_init__(self) -> None:
        for name in ("trade_id", "instrument", "strategy", "setup", "execution"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"daily trade {name} must not be empty")
        for name in ("pnl_usd", "r_multiple", "slippage_usd", "fees_usd"):
            _finite(getattr(self, name), f"daily trade {name}")
        if self.slippage_usd < 0 or self.fees_usd < 0:
            raise ValueError("daily trade slippage and fees must be non-negative")


@dataclass(frozen=True, slots=True)
class HistoricalSessionBaseline:
    sample_sessions: int
    mean_pnl_usd: float
    pnl_std_usd: float
    mean_signals: float
    signals_std: float
    mean_slippage_usd: float
    slippage_std_usd: float

    def __post_init__(self) -> None:
        if self.sample_sessions < 1:
            raise ValueError("historical baseline requires at least one session")
        for name in (
            "mean_pnl_usd",
            "pnl_std_usd",
            "mean_signals",
            "signals_std",
            "mean_slippage_usd",
            "slippage_std_usd",
        ):
            _finite(getattr(self, name), f"historical {name}")
        if self.pnl_std_usd < 0 or self.signals_std < 0 or self.slippage_std_usd < 0:
            raise ValueError("historical standard deviations must be non-negative")


@dataclass(frozen=True, slots=True)
class DailyReportInput:
    session_date: date
    account_balance_usd: float
    prop_failure_level_usd: float
    trades: tuple[DailyTradeRecord, ...] = ()
    signals_generated: int = 0
    signals_accepted: int = 0
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    rule_violations: tuple[str, ...] = ()
    payout_progress_usd: float | None = None
    payout_goal_usd: float | None = None
    max_trades_per_day: int = 1
    max_daily_loss_usd: float = 200.0
    abnormal_z_threshold: float = 3.0
    historical: HistoricalSessionBaseline | None = None

    def __post_init__(self) -> None:
        _finite(self.account_balance_usd, "account balance")
        _finite(self.prop_failure_level_usd, "prop failure level")
        if self.max_trades_per_day < 1:
            raise ValueError("max trades per day must be at least one")
        if _finite(self.max_daily_loss_usd, "maximum daily loss") <= 0:
            raise ValueError("maximum daily loss must be positive")
        if _finite(self.abnormal_z_threshold, "abnormal z threshold") <= 0:
            raise ValueError("abnormal z threshold must be positive")
        if self.signals_generated < 0 or self.signals_accepted < 0:
            raise ValueError("signal counts must be non-negative")
        if self.signals_accepted > self.signals_generated:
            raise ValueError("accepted signals cannot exceed generated signals")
        normalized: dict[str, int] = {}
        for reason, count in self.rejection_counts.items():
            label = str(reason).strip()
            if not label:
                raise ValueError("rejection reason must not be empty")
            if isinstance(count, bool) or int(count) != count or count < 0:
                raise ValueError(f"rejection count for {label!r} must be a non-negative integer")
            normalized[label] = int(count)
        rejected = sum(normalized.values())
        if self.signals_accepted + rejected != self.signals_generated:
            raise ValueError(
                "accepted plus rejected signal counts must equal generated signals"
            )
        if len(self.trades) > self.signals_accepted:
            raise ValueError("completed trades cannot exceed accepted signals")
        if (self.payout_progress_usd is None) != (self.payout_goal_usd is None):
            raise ValueError("payout progress and goal must either both be set or both be null")
        if self.payout_progress_usd is not None:
            if _finite(self.payout_progress_usd, "payout progress") < 0:
                raise ValueError("payout progress must be non-negative")
            if _finite(self.payout_goal_usd, "payout goal") <= 0:
                raise ValueError("payout goal must be positive")
        violations = tuple(str(item).strip() for item in self.rule_violations)
        if any(not item for item in violations):
            raise ValueError("rule violation labels must not be empty")
        object.__setattr__(self, "rejection_counts", MappingProxyType(normalized))
        object.__setattr__(self, "rule_violations", violations)


@dataclass(frozen=True, slots=True)
class DailyReport:
    session_date: date
    trade_count: int
    pnl_usd: float
    total_r: float
    slippage_usd: float
    fees_usd: float
    setups: tuple[tuple[str, int], ...]
    executions: tuple[tuple[str, int], ...]
    account_balance_usd: float
    prop_failure_level_usd: float
    distance_to_failure_usd: float
    payout_progress_usd: float | None
    payout_goal_usd: float | None
    payout_progress_fraction: float | None
    signals_generated: int
    signals_accepted: int
    signals_rejected: int
    rejection_counts: tuple[tuple[str, int], ...]
    rule_violations: tuple[str, ...]
    abnormality_flags: tuple[str, ...]
    historical_comparison: Mapping[str, float | int | None]

    def to_dict(self) -> dict:
        return {
            "session_date": self.session_date.isoformat(),
            "trades": self.trade_count,
            "pnl_usd": self.pnl_usd,
            "total_r": self.total_r,
            "slippage_usd": self.slippage_usd,
            "fees_usd": self.fees_usd,
            "setups": dict(self.setups),
            "executions": dict(self.executions),
            "account": {
                "balance_usd": self.account_balance_usd,
                "prop_failure_level_usd": self.prop_failure_level_usd,
                "distance_to_failure_usd": self.distance_to_failure_usd,
            },
            "payout": {
                "progress_usd": self.payout_progress_usd,
                "goal_usd": self.payout_goal_usd,
                "progress_fraction": self.payout_progress_fraction,
            },
            "signals": {
                "generated": self.signals_generated,
                "accepted": self.signals_accepted,
                "rejected": self.signals_rejected,
                "rejection_reasons": dict(self.rejection_counts),
            },
            "rule_violations": list(self.rule_violations),
            "abnormality_flags": list(self.abnormality_flags),
            "historical_comparison": dict(self.historical_comparison),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_text(self) -> str:
        violations = ", ".join(self.rule_violations) if self.rule_violations else "NONE"
        flags = ", ".join(self.abnormality_flags) if self.abnormality_flags else "NONE"
        rejection_text = (
            ", ".join(f"{reason}={count}" for reason, count in self.rejection_counts)
            if self.rejection_counts
            else "NONE"
        )
        setups = (
            ", ".join(f"{name}={count}" for name, count in self.setups)
            if self.setups
            else "NONE"
        )
        executions = (
            ", ".join(f"{name}={count}" for name, count in self.executions)
            if self.executions
            else "NONE"
        )
        payout = "not applicable"
        if self.payout_progress_usd is not None:
            payout = (
                f"${self.payout_progress_usd:,.2f} / ${self.payout_goal_usd:,.2f} "
                f"({self.payout_progress_fraction:.1%})"
            )
        return "\n".join(
            (
                f"DAILY REPORT  {self.session_date.isoformat()}",
                f"Trades: {self.trade_count}",
                f"P&L: ${self.pnl_usd:+,.2f}",
                f"R: {self.total_r:+.2f}",
                f"Setup: {setups}",
                f"Execution: {executions}",
                f"Slippage: ${self.slippage_usd:,.2f}",
                f"Fees: ${self.fees_usd:,.2f}",
                "",
                f"Account balance: ${self.account_balance_usd:,.2f}",
                f"Prop failure level: ${self.prop_failure_level_usd:,.2f}",
                f"Distance to drawdown: ${self.distance_to_failure_usd:,.2f}",
                f"Payout progress: {payout}",
                "",
                f"Signals generated: {self.signals_generated}",
                f"Signals accepted: {self.signals_accepted}",
                f"Signals rejected: {self.signals_rejected}",
                f"Rejected reasons: {rejection_text}",
                f"Rule violations: {violations}",
                f"Abnormalities: {flags}",
            )
        )


def _z_score(value: float, mean: float, standard_deviation: float) -> float | None:
    if standard_deviation <= 0:
        return None
    return (value - mean) / standard_deviation


def generate_daily_report(source: DailyReportInput) -> DailyReport:
    pnl = sum(trade.pnl_usd for trade in source.trades)
    total_r = sum(trade.r_multiple for trade in source.trades)
    slippage = sum(trade.slippage_usd for trade in source.trades)
    fees = sum(trade.fees_usd for trade in source.trades)
    distance = source.account_balance_usd - source.prop_failure_level_usd
    payout_fraction = (
        None
        if source.payout_progress_usd is None
        else min(1.0, source.payout_progress_usd / source.payout_goal_usd)
    )
    setups = tuple(sorted(Counter(trade.setup for trade in source.trades).items()))
    executions = tuple(sorted(Counter(trade.execution for trade in source.trades).items()))
    rejections = tuple(sorted(source.rejection_counts.items()))

    flags: list[str] = []
    if len(source.trades) > source.max_trades_per_day:
        flags.append("TRADE_QUOTA_EXCEEDED")
    if pnl <= -source.max_daily_loss_usd:
        flags.append("DAILY_LOSS_LIMIT_REACHED")
    if distance <= 0:
        flags.append("PROP_FAILURE_LEVEL_REACHED")
    if source.rule_violations:
        flags.append("RULE_VIOLATION_RECORDED")

    comparison: dict[str, float | int | None] = {
        "sample_sessions": None,
        "pnl_z_score": None,
        "signals_z_score": None,
        "slippage_z_score": None,
    }
    baseline = source.historical
    if baseline is not None:
        pnl_z = _z_score(pnl, baseline.mean_pnl_usd, baseline.pnl_std_usd)
        signals_z = _z_score(
            float(source.signals_generated), baseline.mean_signals, baseline.signals_std
        )
        slippage_z = _z_score(
            slippage, baseline.mean_slippage_usd, baseline.slippage_std_usd
        )
        comparison.update(
            {
                "sample_sessions": baseline.sample_sessions,
                "pnl_z_score": pnl_z,
                "signals_z_score": signals_z,
                "slippage_z_score": slippage_z,
            }
        )
        if pnl_z is not None and abs(pnl_z) >= source.abnormal_z_threshold:
            flags.append("PNL_OUTLIER")
        if signals_z is not None and signals_z >= source.abnormal_z_threshold:
            flags.append("SIGNAL_COUNT_HIGH")
        if slippage_z is not None and slippage_z >= source.abnormal_z_threshold:
            flags.append("SLIPPAGE_HIGH")

    return DailyReport(
        session_date=source.session_date,
        trade_count=len(source.trades),
        pnl_usd=pnl,
        total_r=total_r,
        slippage_usd=slippage,
        fees_usd=fees,
        setups=setups,
        executions=executions,
        account_balance_usd=source.account_balance_usd,
        prop_failure_level_usd=source.prop_failure_level_usd,
        distance_to_failure_usd=distance,
        payout_progress_usd=source.payout_progress_usd,
        payout_goal_usd=source.payout_goal_usd,
        payout_progress_fraction=payout_fraction,
        signals_generated=source.signals_generated,
        signals_accepted=source.signals_accepted,
        signals_rejected=sum(source.rejection_counts.values()),
        rejection_counts=rejections,
        rule_violations=source.rule_violations,
        abnormality_flags=tuple(flags),
        historical_comparison=MappingProxyType(comparison),
    )
