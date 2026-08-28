"""Fail-closed Layer-1 validation for strategy entry intents.

The gate is deliberately pure. It receives inert strategy output plus explicit runtime
facts and returns a complete decision trace; it has no broker or execution dependency and
cannot place, cancel, or modify an order.

Freshness and permission are inputs rather than assumptions. A caller must explicitly
provide the signal time, latest market-data time, strategy permission, and session
permission. Missing, malformed, stale, future-dated, or indeterminate facts are refusals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Any

from ..core.models import OrderIntent
from ..core.types import Side
from ..instruments.registry import InstrumentSpec


@dataclass(frozen=True, slots=True)
class StrategyGateCheck:
    """One independently inspectable fact in a Layer-1 decision."""

    code: str
    passed: bool
    detail: str
    observed: Any = None
    required: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "detail": self.detail,
            "observed": self.observed,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class StrategyGateDecision:
    """Complete Layer-1 result, including successes as well as failures."""

    checks: tuple[StrategyGateCheck, ...]
    expected_reward_risk: float | None

    @property
    def layer(self) -> str:
        return "STRATEGY"

    @property
    def approved(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[StrategyGateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.failures)

    def check(self, code: str) -> StrategyGateCheck:
        """Return a named check; raise only for a programmer typo in the check name."""
        for item in self.checks:
            if item.code == code:
                return item
        raise KeyError(code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "approved": self.approved,
            "expected_reward_risk": self.expected_reward_risk,
            "failure_codes": list(self.failure_codes),
            "checks": [check.to_dict() for check in self.checks],
        }


class StrategyGate:
    """Validate every fact a strategy must establish before personal-risk sizing.

    Policy values are mandatory. Freshness and acceptable reward:risk vary with timeframe
    and strategy, so silently choosing them here would turn an omitted production setting
    into permission to trade.
    """

    def __init__(
        self,
        instrument: InstrumentSpec,
        *,
        minimum_expected_rr: float,
        max_signal_age: timedelta,
        max_data_age: timedelta,
    ) -> None:
        if not _positive_finite(minimum_expected_rr):
            raise ValueError("minimum_expected_rr must be finite and positive")
        if max_signal_age < timedelta(0):
            raise ValueError("max_signal_age must be non-negative")
        if max_data_age < timedelta(0):
            raise ValueError("max_data_age must be non-negative")
        if not _positive_finite(instrument.tick_size):
            raise ValueError("instrument tick_size must be finite and positive")

        self.instrument = instrument
        self.minimum_expected_rr = float(minimum_expected_rr)
        self.max_signal_age = max_signal_age
        self.max_data_age = max_data_age

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        now: datetime,
        market_data_timestamp: datetime | None,
        strategy_permitted: bool | None,
        session_permitted: bool | None,
    ) -> StrategyGateDecision:
        """Return a full trace. Any unknown or invalid fact makes approval false."""
        checks: list[StrategyGateCheck] = []

        strategy_name = getattr(intent, "strategy", None)
        conditions = getattr(intent, "conditions", None)
        evidence_valid = (
            isinstance(strategy_name, str)
            and bool(strategy_name.strip())
            and isinstance(conditions, tuple)
            and bool(conditions)
            and all(isinstance(item, str) and bool(item.strip()) for item in conditions)
        )
        checks.append(
            StrategyGateCheck(
                "setup_evidence",
                evidence_valid,
                "strategy name and named setup conditions are present"
                if evidence_valid
                else "strategy name and every setup condition must be nonempty strings",
                observed={
                    "strategy": strategy_name,
                    "conditions": list(conditions) if isinstance(conditions, tuple) else None,
                },
                required="nonempty strategy and condition tuple",
            )
        )

        requested_instrument = getattr(intent, "instrument", None)
        instrument_matches = requested_instrument == self.instrument.symbol
        checks.append(
            StrategyGateCheck(
                "instrument_match",
                instrument_matches,
                "intent instrument matches the configured instrument"
                if instrument_matches
                else "intent instrument does not exactly match the configured instrument",
                observed=requested_instrument,
                required=self.instrument.symbol,
            )
        )

        signal_timestamp = getattr(intent, "timestamp", None)
        signal_age = _age_seconds(now, signal_timestamp)
        max_signal_seconds = self.max_signal_age.total_seconds()
        signal_current = signal_age is not None and 0.0 <= signal_age <= max_signal_seconds
        checks.append(
            StrategyGateCheck(
                "signal_current",
                signal_current,
                _freshness_detail(
                    "signal", signal_timestamp, now, signal_age, max_signal_seconds
                ),
                observed={
                    "timestamp": _iso(signal_timestamp),
                    "age_seconds": signal_age,
                },
                required={"timezone_aware": True, "max_age_seconds": max_signal_seconds},
            )
        )

        data_age = _age_seconds(now, market_data_timestamp)
        max_data_seconds = self.max_data_age.total_seconds()
        data_fresh = data_age is not None and 0.0 <= data_age <= max_data_seconds
        checks.append(
            StrategyGateCheck(
                "data_fresh",
                data_fresh,
                _freshness_detail(
                    "market data", market_data_timestamp, now, data_age, max_data_seconds
                ),
                observed={
                    "timestamp": _iso(market_data_timestamp),
                    "age_seconds": data_age,
                },
                required={"timezone_aware": True, "max_age_seconds": max_data_seconds},
            )
        )

        strategy_allowed = strategy_permitted is True
        checks.append(
            StrategyGateCheck(
                "strategy_permission",
                strategy_allowed,
                "strategy permits this setup"
                if strategy_allowed
                else "strategy permission is false or unknown",
                observed=strategy_permitted,
                required=True,
            )
        )

        session_allowed = session_permitted is True
        checks.append(
            StrategyGateCheck(
                "session_permission",
                session_allowed,
                "session permits entry"
                if session_allowed
                else "session permission is false or unknown",
                observed=session_permitted,
                required=True,
            )
        )

        geometry_checks, expected_rr = self._geometry_checks(
            intent,
            getattr(intent, "reference_price", None),
        )
        checks.extend(geometry_checks)

        return StrategyGateDecision(tuple(checks), expected_rr)

    def reprice_for_execution(
        self,
        decision: StrategyGateDecision,
        intent: OrderIntent,
        *,
        entry_price: float,
    ) -> StrategyGateDecision:
        """Recheck the complete Layer-1 trace at the signed executable entry bound.

        Strategy setup logic describes reward:risk from a signal reference. Personal risk
        may permit an adverse entry gap, so that reference is not the worst price the
        venue can execute. Replacing the geometry checks after sizing keeps the trace
        honest and prevents a nominal 2R setup from becoming sub-2R at the signed limit.
        """
        if not isinstance(decision, StrategyGateDecision):
            raise TypeError("decision must be a StrategyGateDecision")
        replacements, expected_rr = self._geometry_checks(intent, entry_price)
        by_code = {check.code: check for check in replacements}
        replaced_codes: set[str] = set()
        checks: list[StrategyGateCheck] = []
        for check in decision.checks:
            replacement = by_code.get(check.code)
            if replacement is None:
                checks.append(check)
            else:
                checks.append(replacement)
                replaced_codes.add(check.code)
        if replaced_codes != set(by_code):
            raise ValueError("base strategy decision has an incomplete geometry trace")
        return StrategyGateDecision(tuple(checks), expected_rr)

    def _geometry_checks(
        self,
        intent: OrderIntent,
        entry: Any,
    ) -> tuple[list[StrategyGateCheck], float | None]:
        """Build the price/stop/target checks for one candidate execution price."""
        stop = getattr(intent, "stop_price", None)
        target = getattr(intent, "target_price", None)

        entry_valid = self._valid_price(entry)
        stop_valid = self._valid_price(stop)
        target_present = target is not None
        target_valid = self._valid_price(target)

        checks = [
            self._price_check("entry_price", entry, entry_valid),
            self._price_check("stop_price", stop, stop_valid),
            StrategyGateCheck(
                "target_present",
                target_present,
                "target is present" if target_present else "target omission is not permitted",
                observed=target,
                required="explicit target price",
            ),
            self._price_check("target_price", target, target_valid),
        ]

        stop_distance: float | None = None
        distance_valid = False
        if entry_valid and stop_valid:
            stop_distance = abs(float(entry) - float(stop))
            distance_valid = stop_distance > _price_tolerance(self.instrument.tick_size)
        checks.append(
            StrategyGateCheck(
                "nonzero_stop_distance",
                distance_valid,
                "stop distance is positive"
                if distance_valid
                else "entry and stop must be distinct valid prices",
                observed=stop_distance,
                required=f"> {_price_tolerance(self.instrument.tick_size):g}",
            )
        )

        side = getattr(intent, "side", None)
        protective_sides = False
        if entry_valid and stop_valid and target_valid:
            if side is Side.BUY:
                protective_sides = float(stop) < float(entry) < float(target)
            elif side is Side.SELL:
                protective_sides = float(target) < float(entry) < float(stop)
        checks.append(
            StrategyGateCheck(
                "protective_sides",
                protective_sides,
                "stop and target are on their protective sides"
                if protective_sides
                else "long requires stop < entry < target; short requires target < entry < stop",
                observed={
                    "side": getattr(side, "name", str(side)),
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                },
                required="side-correct stop and target",
            )
        )

        expected_rr: float | None = None
        rr_valid = False
        if distance_valid and entry_valid and target_valid:
            expected_rr = abs(float(target) - float(entry)) / float(stop_distance)
            rr_valid = (
                math.isfinite(expected_rr)
                and expected_rr + 1e-12 >= self.minimum_expected_rr
            )
        checks.append(
            StrategyGateCheck(
                "minimum_expected_rr",
                rr_valid,
                f"expected reward:risk {expected_rr:.4f} meets the minimum"
                if rr_valid and expected_rr is not None
                else "expected reward:risk is unavailable or below the configured minimum",
                observed=expected_rr,
                required=self.minimum_expected_rr,
            )
        )
        return checks, expected_rr

    def _valid_price(self, value: Any) -> bool:
        if not _positive_finite(value):
            return False
        price = float(value)
        return abs(price - self.instrument.round_to_tick(price)) <= _price_tolerance(
            self.instrument.tick_size
        )

    def _price_check(self, code: str, value: Any, passed: bool) -> StrategyGateCheck:
        return StrategyGateCheck(
            code,
            passed,
            "price is finite, positive, and tick-aligned"
            if passed
            else "price must be finite, positive, and aligned to the instrument tick",
            observed=value,
            required={"positive": True, "finite": True, "tick_size": self.instrument.tick_size},
        )


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _aware(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _age_seconds(now: Any, then: Any) -> float | None:
    if not _aware(now) or not _aware(then):
        return None
    return (
        now.astimezone(timezone.utc) - then.astimezone(timezone.utc)
    ).total_seconds()


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _freshness_detail(
    label: str,
    timestamp: Any,
    now: Any,
    age_seconds: float | None,
    maximum_seconds: float,
) -> str:
    if not _aware(now):
        return "current time must be timezone-aware"
    if timestamp is None:
        return f"{label} timestamp is missing"
    if not _aware(timestamp):
        return f"{label} timestamp must be timezone-aware"
    if age_seconds is not None and age_seconds < 0:
        return f"{label} timestamp is in the future"
    if age_seconds is None or age_seconds > maximum_seconds:
        return f"{label} is stale"
    return f"{label} age is within the configured maximum"


def _price_tolerance(tick_size: float) -> float:
    return max(1e-9, abs(tick_size) * 1e-9)
