"""Typed configuration.

YAML in, frozen dataclasses out, validated on load. Two rules earn their keep here:

* **Unknown keys are an error.** A typo like `max_daily_loss_used` would otherwise leave
  the real limit at its default while the operator believes they changed it. For a risk
  configuration that is the worst possible failure mode, so the loader refuses it.
* **Every error names the offending key path.** `risk.daily.max_daily_loss_usd must be
  positive` is actionable; `ValueError: invalid value` is not.

Environment variables may override any scalar via `TRADEBOT__SECTION__KEY` (double
underscores separate levels), which is how research is parameterised in CI without
editing files.  A future production runner must load with ``production_pinned=True``;
that mode permits only explicitly safer risk/cost/session changes and refuses every
other environment mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .core.timeframes import UnknownTimeframe, timeframe_seconds
from .core.types import TradingMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "tradebot.yaml"
ENV_PREFIX = "TRADEBOT__"


class ConfigError(ValueError):
    """Raised with a key path so the operator knows exactly what to fix."""


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionConfig:
    timezone: str = "US/Eastern"
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    # No new entries in the first N minutes: the opening auction's price discovery makes
    # level-based rules unreliable, and slippage is at its worst.
    entry_open_buffer_minutes: int = 5
    # No new entries in the last N minutes: a trade opened here cannot reach its target
    # before the forced flatten, so it is a coin flip with costs attached.
    entry_close_buffer_minutes: int = 15
    flatten_before_close_minutes: int = 2
    news_blackout_windows: tuple[dict[str, str], ...] = ()

    @property
    def start_time(self) -> time:
        return _parse_time(self.rth_start, "session.rth_start")

    @property
    def end_time(self) -> time:
        return _parse_time(self.rth_end, "session.rth_end")

    def validate(self, path: str) -> None:
        if self.start_time >= self.end_time:
            raise ConfigError(f"{path}.rth_start must be before rth_end")
        for key in ("entry_open_buffer_minutes", "entry_close_buffer_minutes",
                    "flatten_before_close_minutes"):
            if getattr(self, key) < 0:
                raise ConfigError(f"{path}.{key} must be >= 0")


@dataclass(frozen=True, slots=True)
class PerTradeRisk:
    risk_pct_of_equity: float = 0.375
    max_risk_per_trade_usd: float = 200.0
    max_contracts: int = 3
    min_contracts: int = 1

    def validate(self, path: str) -> None:
        if not 0 < self.risk_pct_of_equity <= 100:
            raise ConfigError(f"{path}.risk_pct_of_equity must be in (0, 100]")
        if self.max_risk_per_trade_usd <= 0:
            raise ConfigError(f"{path}.max_risk_per_trade_usd must be positive")
        if self.min_contracts < 1:
            raise ConfigError(f"{path}.min_contracts must be >= 1")
        if self.max_contracts < self.min_contracts:
            raise ConfigError(f"{path}.max_contracts must be >= min_contracts")


@dataclass(frozen=True, slots=True)
class DailyRisk:
    max_daily_loss_usd: float = 200.0
    max_daily_loss_r: float = 4.0
    max_trades_per_day: int = 1

    def validate(self, path: str) -> None:
        if self.max_daily_loss_usd <= 0:
            raise ConfigError(f"{path}.max_daily_loss_usd must be positive (it is a magnitude)")
        if self.max_daily_loss_r <= 0:
            raise ConfigError(f"{path}.max_daily_loss_r must be positive (it is a magnitude)")
        if self.max_trades_per_day < 1:
            raise ConfigError(f"{path}.max_trades_per_day must be >= 1")


@dataclass(frozen=True, slots=True)
class StreakRisk:
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 20

    def validate(self, path: str) -> None:
        if self.max_consecutive_losses < 1:
            raise ConfigError(f"{path}.max_consecutive_losses must be >= 1")
        if self.cooldown_minutes < 0:
            raise ConfigError(f"{path}.cooldown_minutes must be >= 0")


@dataclass(frozen=True, slots=True)
class DrawdownRisk:
    # Legacy research circuit breaker only. This is not a prop-firm rule and cannot be
    # used to authorize an order; the selected versioned profile and PropFirmRiskGate own
    # the actual EOD/intraday floor methodology and internal dollar buffer.
    trailing_drawdown_pct: float = 5.0
    size_reduction_cushion_pct: float = 40.0

    def validate(self, path: str) -> None:
        if not 0 < self.trailing_drawdown_pct <= 100:
            raise ConfigError(f"{path}.trailing_drawdown_pct must be in (0, 100]")
        if not 0 <= self.size_reduction_cushion_pct <= 100:
            raise ConfigError(f"{path}.size_reduction_cushion_pct must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class KillSwitchConfig:
    flag_file: str = "logs/KILL_SWITCH.flag"
    max_consecutive_errors: int = 5

    def validate(self, path: str) -> None:
        if self.max_consecutive_errors < 1:
            raise ConfigError(f"{path}.max_consecutive_errors must be >= 1")


@dataclass(frozen=True, slots=True)
class RiskConfig:
    starting_equity_usd: float = 50000.0
    max_open_positions: int = 1
    per_trade: PerTradeRisk = field(default_factory=PerTradeRisk)
    daily: DailyRisk = field(default_factory=DailyRisk)
    streaks: StreakRisk = field(default_factory=StreakRisk)
    drawdown: DrawdownRisk = field(default_factory=DrawdownRisk)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    # How long an approval stays spendable. Short on purpose: an approval is a statement
    # about the state of the account *now*, and the market moves.
    token_ttl_seconds: float = 60.0

    def validate(self, path: str) -> None:
        if self.starting_equity_usd <= 0:
            raise ConfigError(f"{path}.starting_equity_usd must be positive")
        if self.max_open_positions < 1:
            raise ConfigError(f"{path}.max_open_positions must be >= 1")
        if self.token_ttl_seconds <= 0:
            raise ConfigError(f"{path}.token_ttl_seconds must be positive")
        self.per_trade.validate(f"{path}.per_trade")
        self.daily.validate(f"{path}.daily")
        self.streaks.validate(f"{path}.streaks")
        self.drawdown.validate(f"{path}.drawdown")
        self.kill_switch.validate(f"{path}.kill_switch")


@dataclass(frozen=True, slots=True)
class CostConfig:
    commission_round_trip_usd: float = 1.24
    slippage_ticks_per_side: float = 1.0
    slippage_stress_multiplier: float = 2.0

    def validate(self, path: str) -> None:
        if self.commission_round_trip_usd < 0:
            raise ConfigError(f"{path}.commission_round_trip_usd must be >= 0")
        if self.slippage_ticks_per_side < 0:
            raise ConfigError(f"{path}.slippage_ticks_per_side must be >= 0")
        if self.slippage_stress_multiplier < 1:
            raise ConfigError(f"{path}.slippage_stress_multiplier must be >= 1")


@dataclass(frozen=True, slots=True)
class DataConfig:
    store_dir: str = "data_cache"
    # Multiples of the bar interval. Beyond `stale_bar_multiple` we stop opening new
    # positions; beyond `hard_stale_bar_multiple` an open position is flattened, because
    # by then we are managing a stop against a price we cannot see.
    stale_bar_multiple: float = 3.0
    hard_stale_bar_multiple: float = 10.0

    def validate(self, path: str) -> None:
        if self.stale_bar_multiple <= 1:
            raise ConfigError(f"{path}.stale_bar_multiple must be > 1")
        if self.hard_stale_bar_multiple < self.stale_bar_multiple:
            raise ConfigError(f"{path}.hard_stale_bar_multiple must be >= stale_bar_multiple")


@dataclass(frozen=True, slots=True)
class SimulatedBrokerConfig:
    latency_ms: float = 120.0
    reject_probability: float = 0.0
    partial_fill_probability: float = 0.0
    disconnect_probability: float = 0.0
    seed: int = 7

    def validate(self, path: str) -> None:
        for key in ("reject_probability", "partial_fill_probability", "disconnect_probability"):
            value = getattr(self, key)
            if not 0 <= value <= 1:
                raise ConfigError(f"{path}.{key} must be in [0, 1]")
        if self.latency_ms < 0:
            raise ConfigError(f"{path}.latency_ms must be >= 0")


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    adapter: str = "simulated"  # simulated | tradovate
    environment: str = "demo"  # demo | live -- live is refused in v1
    account_id: str = ""
    symbol_override: str = ""
    simulated: SimulatedBrokerConfig = field(default_factory=SimulatedBrokerConfig)
    request_timeout_seconds: float = 15.0
    max_retries: int = 3

    def validate(self, path: str) -> None:
        if self.adapter not in ("simulated", "tradovate"):
            raise ConfigError(f"{path}.adapter must be 'simulated' or 'tradovate'")
        if self.environment not in ("demo", "live"):
            raise ConfigError(f"{path}.environment must be 'demo' or 'live'")
        if self.max_retries < 0:
            raise ConfigError(f"{path}.max_retries must be >= 0")
        self.simulated.validate(f"{path}.simulated")


@dataclass(frozen=True, slots=True)
class JournalConfig:
    path: str = "logs/tradebot.sqlite3"
    log_every_bar: bool = False  # full per-bar market state; verbose, off by default


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8787

    def validate(self, path: str) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"{path}.port must be a valid TCP port")
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            # Binding a dashboard with a STOP button to a routable interface puts control
            # of a trading process on the network with no authentication in front of it.
            raise ConfigError(
                f"{path}.host must be a loopback address; the dashboard has no authentication"
            )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    name: str = "orb_breakout"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PropFirmConfig:
    profile_path: str = "config/prop_firms/tradeify_growth_50k.yaml"
    phase: str = "evaluation"
    internal_safety_buffer_usd: float = 400.0
    source_snapshot_path: str = "config/prop_firms/source_snapshots.yaml"

    def validate(self, path: str) -> None:
        if not self.profile_path.strip():
            raise ConfigError(f"{path}.profile_path must not be empty")
        if not self.phase.strip():
            raise ConfigError(f"{path}.phase must not be empty")
        if self.internal_safety_buffer_usd <= 0:
            raise ConfigError(f"{path}.internal_safety_buffer_usd must be positive")

        # Import lazily so the generic typed-config module stays usable even when a caller
        # is inspecting a malformed prop profile. Errors retain the profile key path.
        from .prop_firms import PropProfileError, load_prop_profile

        profile_file = Path(self.profile_path)
        if not profile_file.is_absolute():
            profile_file = PROJECT_ROOT / profile_file
        try:
            profile = load_prop_profile(profile_file)
            profile.rules_for(self.phase)
        except (PropProfileError, KeyError) as exc:
            raise ConfigError(f"{path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    # Stage 0 backtest, 1 replay, 2 paper, 3 evaluation, 4 funded. The current runtime
    # config deliberately refuses 3/4; those require the separate hash-bound manifest gate.
    stage: int = 0
    approval_manifest: str = ""

    def validate(self, path: str) -> None:
        if self.stage not in range(5):
            raise ConfigError(f"{path}.stage must be one of 0, 1, 2, 3, 4")
        if self.stage >= 3:
            raise ConfigError(
                f"{path}.stage {self.stage} is disabled in ordinary config; Stage 3/4 "
                "requires a separately verified human approval manifest and runner gate"
            )


@dataclass(frozen=True, slots=True)
class Config:
    mode: TradingMode = TradingMode.BACKTEST
    instrument: str = "MNQ"
    timeframe: str = "1min"
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    data: DataConfig = field(default_factory=DataConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    prop_firm: PropFirmConfig = field(default_factory=PropFirmConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def validate(self) -> None:
        self.session.validate("session")
        self.risk.validate("risk")
        self.costs.validate("costs")
        self.data.validate("data")
        self.broker.validate("broker")
        self.dashboard.validate("dashboard")
        self.prop_firm.validate("prop_firm")
        self.deployment.validate("deployment")
        if self.mode is TradingMode.LIVE:
            raise ConfigError(
                "mode: LIVE is not supported. Live trading is not implemented in v1 "
                "(see PROJECT_SPEC section 8)."
            )
        if self.deployment.stage == 0 and self.mode is not TradingMode.BACKTEST:
            raise ConfigError("mode must be BACKTEST while deployment.stage is 0")
        if self.deployment.stage in (1, 2) and self.mode is not TradingMode.PAPER:
            raise ConfigError("mode must be PAPER while deployment.stage is 1 or 2")
        _parse_timeframe(self.timeframe)

    @property
    def bar_seconds(self) -> float:
        return _parse_timeframe(self.timeframe)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _parse_time(value: str | time, path: str) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    try:
        hh, mm = str(value).split(":")[:2]
        return time(int(hh), int(mm))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path} must look like 'HH:MM', got {value!r}") from exc


def _parse_timeframe(value: str) -> float:
    try:
        return timeframe_seconds(value)
    except UnknownTimeframe as exc:
        raise ConfigError(f"timeframe: {exc}") from exc


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved field annotations.

    `from __future__ import annotations` makes every annotation a string, so
    `dataclasses.fields(cls)[i].type` is `"RiskConfig"` rather than the class. Resolving
    through `get_type_hints` keeps the coercion honest without a hand-maintained name map
    that silently rots whenever a field is added.
    """
    return get_type_hints(cls, globalns=globals())


def _coerce(raw: Any, target: Any, path: str) -> Any:
    """Recursively build `target` from plain YAML data, rejecting unknown keys."""
    if is_dataclass(target):
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must be a mapping, got {type(raw).__name__}")
        known = {f.name for f in fields(target)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"unknown key(s) under {path or 'root'}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        hints = _hints(target)
        kwargs = {}
        for name in known:
            if name not in raw:
                continue
            child = f"{path}.{name}" if path else name
            kwargs[name] = _coerce(raw[name], hints[name], child)
        return target(**kwargs)

    origin = get_origin(target)
    if origin is tuple:
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise ConfigError(f"{path} must be a list")
        args = get_args(target)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0], f"{path}[]") for item in raw)
        return tuple(raw)
    if origin is dict or target is dict:
        return dict(raw or {})

    if isinstance(target, type) and issubclass(target, TradingMode):
        try:
            return TradingMode(str(raw).upper())
        except ValueError as exc:
            raise ConfigError(
                f"{path} must be one of {[m.value for m in TradingMode]}, got {raw!r}"
            ) from exc
    if target is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if target is int:
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} must be an integer, got {raw!r}") from exc
    if target is float:
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} must be a number, got {raw!r}") from exc
    if target is str:
        return str(raw)
    return raw



def _apply_env_overrides(data: dict) -> tuple[dict, tuple[tuple[str, ...], ...]]:
    """`TRADEBOT__RISK__DAILY__MAX_DAILY_LOSS_USD=500` -> data['risk']['daily'][...]."""
    applied: list[tuple[str, ...]] = []
    for env_key, value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in env_key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        cursor = data
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"env override {env_key} conflicts with a scalar value")
        cursor[parts[-1]] = yaml.safe_load(value)
        applied.append(tuple(parts))
    return data, tuple(applied)


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict | None = None,
    production_pinned: bool = False,
) -> Config:
    """Load, override, coerce, validate. Any of those four steps may raise `ConfigError`."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        if path is not None:
            raise ConfigError(f"config file not found: {cfg_path}")
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a top-level mapping")

    pinned_cfg: Config | None = None
    if production_pinned:
        if overrides:
            raise ConfigError(
                "production-pinned config does not accept programmatic overrides; "
                "approve a new immutable artifact instead"
            )
        pinned_cfg = _coerce(raw, Config, "")
        pinned_cfg.validate()

    raw, env_paths = _apply_env_overrides(raw)
    if overrides:
        raw = _deep_merge(raw, overrides)

    if pinned_cfg is not None:
        _validate_production_env_paths(env_paths)

    cfg = _coerce(raw, Config, "")
    cfg.validate()
    if pinned_cfg is not None:
        _validate_production_env_overrides(pinned_cfg, cfg, env_paths)
    return cfg


# Direction of a demonstrably safer runtime override relative to the reviewed file.
# Anything absent from these maps is immutable in production-pinned mode.
_PRODUCTION_UPPER_BOUNDS = {
    ("risk", "max_open_positions"),
    ("risk", "token_ttl_seconds"),
    ("risk", "per_trade", "risk_pct_of_equity"),
    ("risk", "per_trade", "max_risk_per_trade_usd"),
    ("risk", "per_trade", "max_contracts"),
    ("risk", "daily", "max_daily_loss_usd"),
    ("risk", "daily", "max_daily_loss_r"),
    ("risk", "daily", "max_trades_per_day"),
    ("risk", "streaks", "max_consecutive_losses"),
    ("risk", "drawdown", "trailing_drawdown_pct"),
    ("risk", "kill_switch", "max_consecutive_errors"),
    ("data", "stale_bar_multiple"),
    ("data", "hard_stale_bar_multiple"),
}
_PRODUCTION_LOWER_BOUNDS = {
    ("risk", "per_trade", "min_contracts"),
    ("risk", "streaks", "cooldown_minutes"),
    ("risk", "drawdown", "size_reduction_cushion_pct"),
    ("prop_firm", "internal_safety_buffer_usd"),
    ("session", "entry_open_buffer_minutes"),
    ("session", "entry_close_buffer_minutes"),
    ("session", "flatten_before_close_minutes"),
    ("costs", "commission_round_trip_usd"),
    ("costs", "slippage_ticks_per_side"),
    ("costs", "slippage_stress_multiplier"),
}


def _config_value(config: Config, path: tuple[str, ...]) -> Any:
    value: Any = config
    for part in path:
        value = getattr(value, part)
    return value


def _validate_production_env_overrides(
    pinned: Config,
    candidate: Config,
    paths: tuple[tuple[str, ...], ...],
) -> None:
    """Allow environment changes only when their safety direction is unambiguous.

    Hashing the YAML file is not sufficient if process environment variables can change
    its effective values after approval.  This check keeps identity, stage, profile,
    strategy, broker route, kill-switch path, and every ambiguous setting immutable while
    still allowing an operator to reduce risk or use more conservative research costs.
    """
    for path in paths:
        label = ".".join(path)
        before = _config_value(pinned, path)
        after = _config_value(candidate, path)
        if path in _PRODUCTION_UPPER_BOUNDS and after > before:
            raise ConfigError(
                f"production env override {label} weakens the pinned limit "
                f"({before!r} -> {after!r})"
            )
        if path in _PRODUCTION_LOWER_BOUNDS and after < before:
            raise ConfigError(
                f"production env override {label} weakens the pinned safeguard "
                f"({before!r} -> {after!r})"
            )


def _validate_production_env_paths(paths: tuple[tuple[str, ...], ...]) -> None:
    allowed = _PRODUCTION_UPPER_BOUNDS | _PRODUCTION_LOWER_BOUNDS
    for path in paths:
        if path not in allowed:
            raise ConfigError(
                f"production env override {'.'.join(path)} is not an approved "
                "risk-tightening key"
            )


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
