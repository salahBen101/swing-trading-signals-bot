"""Experiment identity, frozen configuration and provenance for the RSI options paper trial.

A result without its provenance is not a result. Every formal run records the git commit, the
configuration hash, the schema version and the universe it ran against, so that any number in the
dashboard can be traced back to the exact code and settings that produced it.

THE STATUS MACHINE

    DEVELOPMENT           default. Nothing recorded here counts. The clock is not running.
    READY_FOR_PAPER       acceptance criteria pass; the trial may be started by a human.
    PAPER_TRIAL_ACTIVE    the official clock is running. Configuration is frozen.
    PAPER_TRIAL_COMPLETE  the trial ended. The record is closed.

The transition into PAPER_TRIAL_ACTIVE is deliberately not automatic and cannot be reached by a
scheduled job. It requires an explicit human command, and it stamps `official_start_timestamp`
exactly once - the code refuses to rewrite it afterwards, because a start date that can move is
not a start date.

WHY THE CONFIG IS HASHED

Once the trial is active, changing a field that affects results (equity, risk limits, universe,
strategy thresholds, execution timing) does not silently continue the same experiment. The hash
changes, which the loader detects, and the correct response is a NEW experiment version rather
than a quietly altered old one. Otherwise a year of data ends up describing a strategy that no
longer exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STRATEGY_VERSION = "rsi2-long-only-v1"
OPTIONS_SELECTION_VERSION = "delta-target-liquidity-v1"
EXECUTION_MODEL_VERSION = "next-session-open-v1"


class ExperimentStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    READY_FOR_PAPER = "READY_FOR_PAPER"
    PAPER_TRIAL_ACTIVE = "PAPER_TRIAL_ACTIVE"
    PAPER_TRIAL_COMPLETE = "PAPER_TRIAL_COMPLETE"


class TradingMode(str, Enum):
    """Only PAPER is supported. LIVE exists so that an attempt to use it fails loudly."""

    PAPER = "PAPER"
    LIVE = "LIVE"


class ConfigFrozenError(RuntimeError):
    """Raised when a frozen experiment's configuration is modified in place."""


class UnsupportedTradingMode(RuntimeError):
    """Raised at startup when anything other than PAPER is requested."""


# --------------------------------------------------------------------------- risk policy
@dataclass(frozen=True, slots=True)
class RiskLimits:
    """The formal trial's risk envelope. Values are policy, not tuning parameters.

    MAX_PREMIUM_RISK_PER_TRADE is the whole debit paid, because a long option's maximum loss is
    its premium. If one contract costs more than this the opportunity is REJECTED and recorded -
    the limit is never raised to fit the trade.
    """

    starting_equity: float = 50_000.0
    max_premium_risk_per_trade: float = 200.0
    max_open_positions: int = 1
    max_new_trades_per_session: int = 1
    max_daily_strategy_loss: float = 200.0
    max_positions_per_ticker: int = 1
    max_positions_per_sector: int = 1
    max_aggregate_premium: float = 200.0


@dataclass(frozen=True, slots=True)
class OptionsPolicy:
    """Contract-selection gates. Anything unavailable means the trade cannot be validated."""

    min_dte: int = 21
    max_dte: int = 60
    min_delta: float = 0.55
    max_delta: float = 0.85
    max_spread_pct: float = 10.0
    min_open_interest: int = 100
    max_quote_age_seconds: int = 900
    contract_multiplier: int = 100
    reject_adjusted_contracts: bool = True
    earnings_blackout_days: int = 0        # 0 = gate disabled; equities only, ETFs unaffected


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """When a signal may be filled. This is where the pilot's lookahead bug is prevented."""

    entry_session_offset: int = 1          # signal at close of T -> eligible session T+1
    exit_session_offset: int = 1
    entry_window: str = "OPEN"             # the next session's opening window
    max_signal_age_sessions: int = 1       # a stale unfilled signal expires rather than lingering
    order_type: str = "LIMIT"
    limit_slippage_pct: float = 2.0        # limit placed this far through the ask


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    name: str
    hypothesis: str
    universe_name: str
    universe_members: tuple[str, ...]
    risk: RiskLimits = field(default_factory=RiskLimits)
    options: OptionsPolicy = field(default_factory=OptionsPolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    trading_mode: TradingMode = TradingMode.PAPER
    ml_enabled_for_execution: bool = False   # shadow only; see §23 of the protocol
    strategy_version: str = STRATEGY_VERSION
    options_selection_version: str = OPTIONS_SELECTION_VERSION
    execution_model_version: str = EXECUTION_MODEL_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["universe_members"] = list(self.universe_members)
        d["trading_mode"] = self.trading_mode.value
        return d

    def config_hash(self) -> str:
        """Stable hash over everything that can change a result."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Startup validation. Returns problems; an empty list means the config is usable."""
        problems: list[str] = []
        if self.trading_mode is not TradingMode.PAPER:
            problems.append(f"trading_mode must be PAPER, got {self.trading_mode.value}")
        if self.ml_enabled_for_execution:
            problems.append("ml_enabled_for_execution must be False during the formal trial")
        if self.risk.starting_equity <= 0:
            problems.append("starting_equity must be positive")
        if self.risk.max_premium_risk_per_trade <= 0:
            problems.append("max_premium_risk_per_trade must be positive")
        if self.risk.max_open_positions < 1:
            problems.append("max_open_positions must be at least 1")
        if not self.universe_members:
            problems.append("universe is empty")
        if self.execution.entry_session_offset < 1:
            problems.append("entry_session_offset must be >= 1 or execution is same-session")
        if self.execution.exit_session_offset < 1:
            problems.append("exit_session_offset must be >= 1 or exits use lookahead prices")
        if self.options.min_dte <= 0 or self.options.max_dte <= self.options.min_dte:
            problems.append("option DTE window is invalid")
        if not 0 < self.options.min_delta < self.options.max_delta <= 1:
            problems.append("delta window is invalid")
        return problems


# --------------------------------------------------------------------------- provenance
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parents[3])
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """A dirty tree means the recorded commit does not describe the code that ran."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parents[3])
        return bool(out.stdout.strip())
    except Exception:
        return True


@dataclass(frozen=True, slots=True)
class Provenance:
    git_commit: str
    git_dirty: bool
    config_hash: str
    schema_version: int
    strategy_version: str
    options_selection_version: str
    execution_model_version: str
    universe_name: str
    broker: str
    recorded_at: str

    @classmethod
    def capture(cls, cfg: ExperimentConfig, broker: str) -> Provenance:
        return cls(
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            config_hash=cfg.config_hash(),
            schema_version=cfg.schema_version,
            strategy_version=cfg.strategy_version,
            options_selection_version=cfg.options_selection_version,
            execution_model_version=cfg.execution_model_version,
            universe_name=cfg.universe_name,
            broker=broker,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )


# --------------------------------------------------------------------------- state file
@dataclass
class ExperimentState:
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DEVELOPMENT
    official_start_timestamp: str | None = None
    official_end_timestamp: str | None = None
    frozen_config_hash: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "official_start_timestamp": self.official_start_timestamp,
            "official_end_timestamp": self.official_end_timestamp,
            "frozen_config_hash": self.frozen_config_hash,
            "notes": self.notes,
        }

    @property
    def is_active(self) -> bool:
        return self.status is ExperimentStatus.PAPER_TRIAL_ACTIVE

    def check_not_tampered(self) -> None:
        """Once frozen, the config that runs must be the config that was frozen."""
        if self.frozen_config_hash is None:
            return
        current = self.config.config_hash()
        if current != self.frozen_config_hash:
            raise ConfigFrozenError(
                f"configuration changed after the trial was frozen "
                f"(frozen {self.frozen_config_hash}, current {current}). "
                "Start a NEW experiment version instead of editing this one."
            )

    def activate(self, when: datetime | None = None) -> None:
        """Start the official clock. Explicit, once, never rewritten."""
        if self.status is ExperimentStatus.PAPER_TRIAL_ACTIVE:
            raise RuntimeError("the trial is already active")
        if self.status is ExperimentStatus.PAPER_TRIAL_COMPLETE:
            raise RuntimeError("this experiment is complete; create a new one")
        if self.status is not ExperimentStatus.READY_FOR_PAPER:
            raise RuntimeError(
                f"cannot activate from {self.status.value}. Acceptance criteria must pass and "
                "the status must be READY_FOR_PAPER first."
            )
        if self.official_start_timestamp is not None:
            raise RuntimeError("official_start_timestamp is already set and cannot be rewritten")
        self.official_start_timestamp = (when or datetime.now(timezone.utc)).isoformat()
        self.frozen_config_hash = self.config.config_hash()
        self.status = ExperimentStatus.PAPER_TRIAL_ACTIVE


def require_paper_mode() -> None:
    """Fail startup if the environment asks for anything but paper trading."""
    mode = os.environ.get("TRADING_MODE", "PAPER").upper()
    if mode != "PAPER":
        raise UnsupportedTradingMode(
            f"TRADING_MODE={mode} is not supported by this project. Only PAPER is implemented; "
            "live execution requires a separate, explicit engineering decision."
        )


def default_config(universe_members: tuple[str, ...]) -> ExperimentConfig:
    """The pre-registered formal experiment.

    The universe is the five-ETF core the repository's own scanner/README.md specifies as the
    starting universe, chosen because it is defined by liquidity rules rather than by which
    names happen to have done well. The 120-name 'strong'/'all' sets are retrospective and
    survivorship-biased; README line 169 says so explicitly.
    """
    return ExperimentConfig(
        experiment_id="RSI2-OPT-2026-01",
        name="RSI(2) long-call paper trial, five-ETF core",
        hypothesis=(
            "The RSI(2) mean-reversion signal has a positive expectancy on the underlying, and "
            "the long-call implementation preserves enough of it after spread, theta and "
            "contract selection to be worth trading."
        ),
        universe_name="core_etf_v1",
        universe_members=universe_members,
    )
