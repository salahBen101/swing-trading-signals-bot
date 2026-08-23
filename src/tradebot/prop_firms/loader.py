"""Strict YAML loader for versioned prop-firm profiles."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import (
    AccountPhase,
    AccountRuleSet,
    BreachKind,
    ComplianceRules,
    ConsistencyRule,
    ContractLimits,
    ContractScaleTier,
    DrawdownMethod,
    DrawdownRule,
    PayoutRules,
    PropFirmProfile,
    RuleSource,
    TradingWindow,
)


class PropProfileError(ValueError):
    """A profile is incomplete, invalid, or contains a misspelled key."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PropProfileError(f"{path} must be a mapping")
    return value


def _keys(raw: dict[str, Any], path: str, allowed: set[str], required: set[str] = frozenset()) -> None:
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise PropProfileError(f"unknown key(s) under {path}: {sorted(unknown)}")
    if missing:
        raise PropProfileError(f"missing key(s) under {path}: {sorted(missing)}")


def _enum(enum_type: type, value: Any, path: str):
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise PropProfileError(f"{path} has unsupported value {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _tuple(raw: Any, mapper: Callable[[Any, int], Any], path: str) -> tuple:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PropProfileError(f"{path} must be a list")
    return tuple(mapper(value, index) for index, value in enumerate(raw))


def _source(value: Any, index: int) -> RuleSource:
    path = f"sources[{index}]"
    raw = _mapping(value, path)
    allowed = {"title", "url", "checked_on", "official", "note"}
    _keys(raw, path, allowed, {"title", "url", "checked_on"})
    try:
        checked = date.fromisoformat(str(raw["checked_on"]))
    except ValueError as exc:
        raise PropProfileError(f"{path}.checked_on must be YYYY-MM-DD") from exc
    return RuleSource(
        title=str(raw["title"]),
        url=str(raw["url"]),
        checked_on=checked,
        official=bool(raw.get("official", True)),
        note=str(raw.get("note", "")),
    )


def _trading_window(value: Any) -> TradingWindow:
    path = "trading_window"
    raw = _mapping(value, path)
    allowed = {"timezone", "session_start", "session_end", "flatten_by", "holiday_flatten_by"}
    _keys(raw, path, allowed, {"timezone", "session_start", "session_end", "flatten_by"})
    return TradingWindow(
        timezone=str(raw["timezone"]),
        session_start=str(raw["session_start"]),
        session_end=str(raw["session_end"]),
        flatten_by=str(raw["flatten_by"]),
        holiday_flatten_by=(None if raw.get("holiday_flatten_by") is None else str(raw["holiday_flatten_by"])),
    )


def _compliance(value: Any) -> ComplianceRules:
    path = "compliance"
    raw = _mapping(value, path)
    allowed = {
        "automation_allowed", "bot_owner_must_prove_sole_ownership", "bot_exclusive_to_firm",
        "high_frequency_bots_allowed", "hedging_allowed", "news_trading_allowed",
        "averaging_down_firm_allowed", "tradovate_api_allowed",
        "bot_cross_firm_deployment_allowed", "proof_video_may_be_required",
        "platform_error_exploitation_allowed", "cross_account_hedging_allowed",
        "minimum_hold_seconds_for_payout",
        "min_fraction_trades_over_hold", "min_fraction_profit_over_hold",
        "minimum_trades_per_week",
    }
    required = {
        "automation_allowed", "bot_owner_must_prove_sole_ownership", "bot_exclusive_to_firm",
        "high_frequency_bots_allowed", "hedging_allowed", "news_trading_allowed",
        "averaging_down_firm_allowed",
    }
    _keys(raw, path, allowed, required)
    return ComplianceRules(
        automation_allowed=bool(raw["automation_allowed"]),
        bot_owner_must_prove_sole_ownership=bool(raw["bot_owner_must_prove_sole_ownership"]),
        bot_exclusive_to_firm=bool(raw["bot_exclusive_to_firm"]),
        high_frequency_bots_allowed=bool(raw["high_frequency_bots_allowed"]),
        hedging_allowed=bool(raw["hedging_allowed"]),
        news_trading_allowed=bool(raw["news_trading_allowed"]),
        averaging_down_firm_allowed=bool(raw["averaging_down_firm_allowed"]),
        tradovate_api_allowed=bool(raw.get("tradovate_api_allowed", False)),
        bot_cross_firm_deployment_allowed=bool(raw.get("bot_cross_firm_deployment_allowed", False)),
        proof_video_may_be_required=bool(raw.get("proof_video_may_be_required", False)),
        platform_error_exploitation_allowed=bool(raw.get("platform_error_exploitation_allowed", False)),
        cross_account_hedging_allowed=bool(raw.get("cross_account_hedging_allowed", False)),
        minimum_hold_seconds_for_payout=_optional_float(raw.get("minimum_hold_seconds_for_payout")),
        min_fraction_trades_over_hold=_optional_float(raw.get("min_fraction_trades_over_hold")),
        min_fraction_profit_over_hold=_optional_float(raw.get("min_fraction_profit_over_hold")),
        minimum_trades_per_week=int(raw.get("minimum_trades_per_week", 1)),
    )


def _tier(value: Any, index: int, phase_path: str) -> ContractScaleTier:
    path = f"{phase_path}.contracts.scale_tiers[{index}]"
    raw = _mapping(value, path)
    allowed = {"min_end_of_day_balance_usd", "max_minis", "max_micros"}
    _keys(raw, path, allowed, allowed)
    return ContractScaleTier(float(raw["min_end_of_day_balance_usd"]), int(raw["max_minis"]), int(raw["max_micros"]))


def _contracts(value: Any, phase_path: str) -> ContractLimits:
    path = f"{phase_path}.contracts"
    raw = _mapping(value, path)
    allowed = {
        "max_minis", "max_micros", "micro_per_mini", "mixed_sizes_allowed",
        "aggregate_gross_open_position", "scale_tiers",
    }
    _keys(raw, path, allowed, {"max_minis", "max_micros"})
    tiers = _tuple(raw.get("scale_tiers", []), lambda value, index: _tier(value, index, phase_path), f"{path}.scale_tiers")
    return ContractLimits(
        max_minis=int(raw["max_minis"]),
        max_micros=int(raw["max_micros"]),
        micro_per_mini=int(raw.get("micro_per_mini", 10)),
        mixed_sizes_allowed=bool(raw.get("mixed_sizes_allowed", True)),
        aggregate_gross_open_position=bool(raw.get("aggregate_gross_open_position", True)),
        scale_tiers=tiers,
    )


def _drawdown(value: Any, phase_path: str) -> DrawdownRule:
    path = f"{phase_path}.drawdown"
    raw = _mapping(value, path)
    allowed = {
        "amount_usd", "method", "breach_kind", "enforced_intraday", "breach_at_or_below",
        "high_water_mark_basis", "breach_basis", "locks", "lock_trigger_balance_usd",
        "locked_floor_balance_usd",
    }
    _keys(raw, path, allowed, {"amount_usd", "method"})
    return DrawdownRule(
        amount_usd=float(raw["amount_usd"]),
        method=_enum(DrawdownMethod, raw["method"], f"{path}.method"),
        breach_kind=_enum(BreachKind, raw.get("breach_kind", "hard"), f"{path}.breach_kind"),
        enforced_intraday=bool(raw.get("enforced_intraday", True)),
        breach_at_or_below=bool(raw.get("breach_at_or_below", True)),
        high_water_mark_basis=str(raw.get("high_water_mark_basis", "highest_end_of_day_balance")),
        breach_basis=str(raw.get("breach_basis", "real_time_net_liquidation")),
        locks=bool(raw.get("locks", False)),
        lock_trigger_balance_usd=_optional_float(raw.get("lock_trigger_balance_usd")),
        locked_floor_balance_usd=_optional_float(raw.get("locked_floor_balance_usd")),
    )


def _consistency(value: Any, phase_path: str) -> ConsistencyRule:
    path = f"{phase_path}.consistency"
    raw = _mapping(value or {}, path)
    allowed = {
        "max_best_day_fraction", "max_best_day_fraction_by_cycle", "applies_to",
        "resets_after_payout", "profit_excludes_commissions",
    }
    _keys(raw, path, allowed)
    applies = raw.get("applies_to", [])
    if not isinstance(applies, list):
        raise PropProfileError(f"{path}.applies_to must be a list")
    by_cycle = raw.get("max_best_day_fraction_by_cycle", [])
    if not isinstance(by_cycle, list):
        raise PropProfileError(f"{path}.max_best_day_fraction_by_cycle must be a list")
    return ConsistencyRule(
        max_best_day_fraction=_optional_float(raw.get("max_best_day_fraction")),
        max_best_day_fraction_by_cycle=tuple(float(item) for item in by_cycle),
        applies_to=tuple(str(item) for item in applies),
        resets_after_payout=bool(raw.get("resets_after_payout", False)),
        profit_excludes_commissions=bool(raw.get("profit_excludes_commissions", False)),
    )


def _payout(value: Any, phase_path: str) -> PayoutRules:
    path = f"{phase_path}.payout"
    raw = _mapping(value or {}, path)
    allowed = {
        "eligible", "profit_split_trader", "minimum_balance_usd",
        "minimum_balance_must_remain_after_payout", "minimum_payout_usd",
        "winning_days_required", "winning_day_min_profit_usd", "winning_day_strictly_greater",
        "first_profit_goal_usd",
        "subsequent_profit_goal_usd", "max_payout_by_request_usd",
        "max_fraction_total_profit", "cycle_profit_multiplier",
        "require_positive_cycle_after_first", "locks_drawdown_on_payout",
    }
    _keys(raw, path, allowed)
    caps = raw.get("max_payout_by_request_usd", [])
    if not isinstance(caps, list):
        raise PropProfileError(f"{path}.max_payout_by_request_usd must be a list")
    return PayoutRules(
        eligible=bool(raw.get("eligible", False)),
        profit_split_trader=float(raw.get("profit_split_trader", 0.0)),
        minimum_balance_usd=_optional_float(raw.get("minimum_balance_usd")),
        minimum_balance_must_remain_after_payout=bool(
            raw.get("minimum_balance_must_remain_after_payout", False)
        ),
        minimum_payout_usd=_optional_float(raw.get("minimum_payout_usd")),
        winning_days_required=int(raw.get("winning_days_required", 0)),
        winning_day_min_profit_usd=float(raw.get("winning_day_min_profit_usd", 0.0)),
        winning_day_strictly_greater=bool(raw.get("winning_day_strictly_greater", False)),
        first_profit_goal_usd=_optional_float(raw.get("first_profit_goal_usd")),
        subsequent_profit_goal_usd=_optional_float(raw.get("subsequent_profit_goal_usd")),
        max_payout_by_request_usd=tuple(float(item) for item in caps),
        max_fraction_total_profit=_optional_float(raw.get("max_fraction_total_profit")),
        cycle_profit_multiplier=_optional_float(raw.get("cycle_profit_multiplier")),
        require_positive_cycle_after_first=bool(raw.get("require_positive_cycle_after_first", False)),
        locks_drawdown_on_payout=bool(raw.get("locks_drawdown_on_payout", False)),
    )


def _phase(name: str, value: Any) -> AccountRuleSet:
    path = f"phases.{name}"
    raw = _mapping(value, path)
    allowed = {
        "phase", "starting_balance_usd", "profit_target_usd", "minimum_trading_days",
        "daily_loss_limit_usd", "daily_loss_breach_kind", "daily_loss_escalation_balance_usd",
        "daily_loss_escalated_limit_usd", "drawdown", "contracts",
        "consistency", "payout",
    }
    required = {"phase", "starting_balance_usd", "drawdown", "contracts"}
    _keys(raw, path, allowed, required)
    daily_limit = _optional_float(raw.get("daily_loss_limit_usd"))
    daily_kind = raw.get("daily_loss_breach_kind")
    return AccountRuleSet(
        name=name,
        phase=_enum(AccountPhase, raw["phase"], f"{path}.phase"),
        starting_balance_usd=float(raw["starting_balance_usd"]),
        profit_target_usd=_optional_float(raw.get("profit_target_usd")),
        minimum_trading_days=int(raw.get("minimum_trading_days", 0)),
        daily_loss_limit_usd=daily_limit,
        daily_loss_breach_kind=(None if daily_kind is None else _enum(BreachKind, daily_kind, f"{path}.daily_loss_breach_kind")),
        daily_loss_escalation_balance_usd=_optional_float(raw.get("daily_loss_escalation_balance_usd")),
        daily_loss_escalated_limit_usd=_optional_float(raw.get("daily_loss_escalated_limit_usd")),
        drawdown=_drawdown(raw["drawdown"], path),
        contracts=_contracts(raw["contracts"], path),
        consistency=_consistency(raw.get("consistency", {}), path),
        payout=_payout(raw.get("payout", {}), path),
    )


def load_prop_profile(path: str | Path) -> PropFirmProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        raise PropProfileError(f"prop profile not found: {profile_path}")
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    raw = _mapping(raw, "root")
    allowed = {
        "schema_version", "profile_id", "firm", "program", "account_size_usd",
        "purchase_cohort", "verified_on", "reverify_after_hours", "sources",
        "trading_window", "compliance", "phases", "ambiguity_notes",
    }
    required = allowed - {"ambiguity_notes"}
    _keys(raw, "root", allowed, required)
    try:
        verified_on = date.fromisoformat(str(raw["verified_on"]))
    except ValueError as exc:
        raise PropProfileError("verified_on must be YYYY-MM-DD") from exc
    phases_raw = _mapping(raw["phases"], "phases")
    profile = PropFirmProfile(
        schema_version=int(raw["schema_version"]),
        profile_id=str(raw["profile_id"]),
        firm=str(raw["firm"]),
        program=str(raw["program"]),
        account_size_usd=float(raw["account_size_usd"]),
        purchase_cohort=str(raw["purchase_cohort"]),
        verified_on=verified_on,
        reverify_after_hours=int(raw["reverify_after_hours"]),
        sources=_tuple(raw["sources"], _source, "sources"),
        trading_window=_trading_window(raw["trading_window"]),
        compliance=_compliance(raw["compliance"]),
        phases=tuple(_phase(str(name), value) for name, value in phases_raw.items()),
        ambiguity_notes=tuple(str(item) for item in raw.get("ambiguity_notes", [])),
    )
    try:
        profile.validate()
    except (TypeError, ValueError) as exc:
        raise PropProfileError(f"invalid prop profile {profile_path}: {exc}") from exc
    return profile
