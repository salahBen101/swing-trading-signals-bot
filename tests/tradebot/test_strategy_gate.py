"""Layer-1 StrategyGate boundaries and fail-closed behaviour."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.core.models import OrderIntent
from tradebot.core.types import Side
from tradebot.instruments.registry import get_instrument
from tradebot.risk.strategy_gate import StrategyGate


MNQ = get_instrument("MNQ")
NOW = datetime(2024, 4, 1, 11, 0, tzinfo=MARKET_TZ)


@pytest.fixture
def gate() -> StrategyGate:
    return StrategyGate(
        MNQ,
        minimum_expected_rr=2.0,
        max_signal_age=timedelta(seconds=60),
        max_data_age=timedelta(seconds=10),
    )


def an_intent(
    *,
    timestamp: datetime | None = None,
    instrument: str = "MNQ",
    side: Side = Side.BUY,
    entry: float = 18_000.0,
    stop: float = 17_999.0,
    target: float | None = 18_002.0,
    strategy: str = "deterministic_test",
    conditions: tuple[str, ...] = ("trend_up", "pullback_complete"),
) -> OrderIntent:
    return OrderIntent(
        timestamp=timestamp or NOW - timedelta(seconds=30),
        instrument=instrument,
        side=side,
        strategy=strategy,
        reference_price=entry,
        stop_price=stop,
        target_price=target,
        conditions=conditions,
    )


def evaluate(gate: StrategyGate, intent: OrderIntent, **changes):
    inputs = {
        "now": NOW,
        "market_data_timestamp": NOW - timedelta(seconds=5),
        "strategy_permitted": True,
        "session_permitted": True,
    }
    inputs.update(changes)
    return gate.evaluate(intent, **inputs)


def corrupt(intent: OrderIntent, **changes) -> OrderIntent:
    """Bypass OrderIntent's own constructor guard to test this independent boundary."""
    for field, value in changes.items():
        object.__setattr__(intent, field, value)
    return intent


def test_a_valid_setup_passes_every_check_and_returns_a_serialisable_trace(gate):
    decision = evaluate(gate, an_intent())

    assert decision.approved
    assert decision.layer == "STRATEGY"
    assert decision.failures == ()
    assert decision.expected_reward_risk == pytest.approx(2.0)
    assert {check.code for check in decision.checks} == {
        "setup_evidence",
        "instrument_match",
        "signal_current",
        "data_fresh",
        "strategy_permission",
        "session_permission",
        "entry_price",
        "stop_price",
        "target_present",
        "target_price",
        "nonzero_stop_distance",
        "protective_sides",
        "minimum_expected_rr",
    }
    assert all(check.passed for check in decision.checks)
    json.dumps(decision.to_dict())


@pytest.mark.parametrize(
    ("strategy", "conditions"),
    [
        ("", ("valid",)),
        ("   ", ("valid",)),
        ("strategy", ()),
        ("strategy", ("",)),
        ("strategy", ("valid", "  ")),
    ],
)
def test_setup_evidence_must_be_nonempty_and_named(gate, strategy, conditions):
    decision = evaluate(gate, an_intent(strategy=strategy, conditions=conditions))
    assert not decision.approved
    assert not decision.check("setup_evidence").passed


@pytest.mark.parametrize("instrument", ["MES", "mnq", "MNQ ", ""])
def test_instrument_match_is_exact_and_fails_closed(gate, instrument):
    decision = evaluate(gate, an_intent(instrument=instrument))
    assert not decision.approved
    assert decision.failure_codes == ("instrument_match",)


def test_signal_age_at_the_boundary_is_current(gate):
    decision = evaluate(gate, an_intent(timestamp=NOW - timedelta(seconds=60)))
    assert decision.check("signal_current").passed
    assert decision.approved


def test_signal_one_microsecond_past_the_boundary_is_stale(gate):
    intent = an_intent(timestamp=NOW - timedelta(seconds=60, microseconds=1))
    decision = evaluate(gate, intent)
    assert not decision.check("signal_current").passed
    assert "signal_current" in decision.failure_codes


def test_future_signal_is_not_treated_as_fresh(gate):
    decision = evaluate(gate, an_intent(timestamp=NOW + timedelta(microseconds=1)))
    assert not decision.check("signal_current").passed
    assert "future" in decision.check("signal_current").detail


def test_naive_signal_or_clock_fails_closed_instead_of_being_coerced(gate):
    naive = datetime(2024, 4, 1, 11, 0)
    intent = corrupt(an_intent(), timestamp=naive)
    assert not evaluate(gate, intent).check("signal_current").passed

    decision = evaluate(gate, an_intent(), now=naive)
    assert not decision.check("signal_current").passed
    assert not decision.check("data_fresh").passed


def test_market_data_age_at_the_boundary_is_fresh(gate):
    decision = evaluate(gate, an_intent(), market_data_timestamp=NOW - timedelta(seconds=10))
    assert decision.check("data_fresh").passed
    assert decision.approved


def test_market_data_one_microsecond_past_the_boundary_is_stale(gate):
    decision = evaluate(
        gate,
        an_intent(),
        market_data_timestamp=NOW - timedelta(seconds=10, microseconds=1),
    )
    assert not decision.check("data_fresh").passed
    assert "data_fresh" in decision.failure_codes


@pytest.mark.parametrize("timestamp", [None, datetime(2024, 4, 1, 11, 0)])
def test_missing_or_naive_market_data_time_fails_closed(gate, timestamp):
    decision = evaluate(gate, an_intent(), market_data_timestamp=timestamp)
    assert not decision.check("data_fresh").passed


def test_future_market_data_timestamp_fails_closed(gate):
    decision = evaluate(
        gate, an_intent(), market_data_timestamp=NOW + timedelta(microseconds=1)
    )
    assert not decision.check("data_fresh").passed
    assert "future" in decision.check("data_fresh").detail


@pytest.mark.parametrize(
    ("field", "value", "failed_code"),
    [
        ("strategy_permitted", False, "strategy_permission"),
        ("strategy_permitted", None, "strategy_permission"),
        ("strategy_permitted", 1, "strategy_permission"),
        ("session_permitted", False, "session_permission"),
        ("session_permitted", None, "session_permission"),
        ("session_permitted", 1, "session_permission"),
    ],
)
def test_strategy_and_session_permission_must_be_explicitly_true(
    gate, field, value, failed_code
):
    decision = evaluate(gate, an_intent(), **{field: value})
    assert not decision.approved
    assert not decision.check(failed_code).passed


@pytest.mark.parametrize(
    ("field", "value", "failed_code"),
    [
        ("reference_price", 0.0, "entry_price"),
        ("reference_price", float("nan"), "entry_price"),
        ("reference_price", float("inf"), "entry_price"),
        ("stop_price", 0.0, "stop_price"),
        ("stop_price", float("nan"), "stop_price"),
        ("target_price", 0.0, "target_price"),
        ("target_price", float("inf"), "target_price"),
    ],
)
def test_prices_must_be_finite_and_positive(gate, field, value, failed_code):
    intent = corrupt(an_intent(), **{field: value})
    decision = evaluate(gate, intent)
    assert not decision.approved
    assert not decision.check(failed_code).passed


@pytest.mark.parametrize(
    ("field", "value", "failed_code"),
    [
        ("reference_price", 18_000.01, "entry_price"),
        ("stop_price", 17_999.01, "stop_price"),
        ("target_price", 18_002.01, "target_price"),
    ],
)
def test_every_price_must_be_tick_aligned(gate, field, value, failed_code):
    intent = corrupt(an_intent(), **{field: value})
    decision = evaluate(gate, intent)
    assert not decision.approved
    assert not decision.check(failed_code).passed


def test_target_omission_is_an_explicit_failure(gate):
    decision = evaluate(gate, an_intent(target=None))
    assert not decision.approved
    assert not decision.check("target_present").passed
    assert not decision.check("target_price").passed


def test_zero_stop_distance_is_rejected(gate):
    intent = corrupt(an_intent(), stop_price=18_000.0)
    decision = evaluate(gate, intent)
    assert not decision.check("nonzero_stop_distance").passed
    assert not decision.approved


def test_one_tick_stop_distance_is_nonzero(gate):
    intent = an_intent(stop=17_999.75, target=18_000.5)
    decision = evaluate(gate, intent)
    assert decision.check("nonzero_stop_distance").passed
    assert decision.expected_reward_risk == pytest.approx(2.0)
    assert decision.approved


@pytest.mark.parametrize(
    "intent",
    [
        corrupt(an_intent(), stop_price=18_001.0),
        an_intent(target=17_998.0),
        corrupt(
            an_intent(side=Side.SELL, stop=18_001.0, target=17_998.0),
            stop_price=17_999.0,
        ),
        an_intent(side=Side.SELL, stop=18_001.0, target=18_002.0),
    ],
)
def test_stop_and_target_must_be_on_the_protective_side(gate, intent):
    decision = evaluate(gate, intent)
    assert not decision.check("protective_sides").passed
    assert not decision.approved


def test_short_stop_target_and_reward_risk_are_supported(gate):
    intent = an_intent(side=Side.SELL, stop=18_001.0, target=17_998.0)
    decision = evaluate(gate, intent)
    assert decision.check("protective_sides").passed
    assert decision.expected_reward_risk == pytest.approx(2.0)
    assert decision.approved


def test_minimum_reward_risk_boundary_passes(gate):
    decision = evaluate(gate, an_intent(stop=17_999.0, target=18_002.0))
    assert decision.expected_reward_risk == pytest.approx(2.0)
    assert decision.check("minimum_expected_rr").passed


def test_one_tick_below_minimum_reward_risk_fails(gate):
    decision = evaluate(gate, an_intent(stop=17_999.0, target=18_001.75))
    assert decision.expected_reward_risk == pytest.approx(1.75)
    assert not decision.check("minimum_expected_rr").passed
    assert not decision.approved


@pytest.mark.parametrize("minimum", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_reward_risk_policy_is_rejected_at_startup(minimum):
    with pytest.raises(ValueError, match="minimum_expected_rr"):
        StrategyGate(
            MNQ,
            minimum_expected_rr=minimum,
            max_signal_age=timedelta(seconds=60),
            max_data_age=timedelta(seconds=10),
        )


@pytest.mark.parametrize("field", ["max_signal_age", "max_data_age"])
def test_negative_freshness_policy_is_rejected_at_startup(field):
    kwargs = {
        "minimum_expected_rr": 2.0,
        "max_signal_age": timedelta(seconds=60),
        "max_data_age": timedelta(seconds=10),
    }
    kwargs[field] = timedelta(microseconds=-1)
    with pytest.raises(ValueError, match=field):
        StrategyGate(MNQ, **kwargs)


def test_strategy_gate_has_no_broker_or_execution_dependency():
    import tradebot.risk.strategy_gate as module

    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split("."))

    assert "broker" not in names
    assert "execution" not in names
    public = {name for name in dir(module.StrategyGate) if not name.startswith("_")}
    assert public == {"evaluate"}
