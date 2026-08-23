from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tradebot.reporting import (
    DailyReportInput,
    HypothesisPolicy,
    PropSurvivalBaseline,
    RejectedSignalRecord,
    WeeklyReportInput,
    WeeklyTradeRecord,
    generate_daily_report,
    generate_weekly_report,
)


ET = ZoneInfo("America/New_York")


def survival_baseline(**overrides) -> PropSurvivalBaseline:
    values = {
        "simulations": 100,
        "evaluation_passes": 40,
        "hard_failures": 20,
        "internal_locks": 10,
        "first_payouts": 15,
        "median_days_to_pass": 22.0,
        "average_drawdown_usd": 600.0,
        "tail_drawdown_usd": 1_600.0,
        "expected_lifetime_sessions": 45.0,
        "expected_profit_per_account_usd": -50.0,
        "consecutive_losses_survived": 3,
    }
    values.update(overrides)
    return PropSurvivalBaseline(**values)


def accepted_trade(
    trade_id: str,
    *,
    minute: int,
    pnl: float,
    r: float,
    setup: str,
    direction: str,
    stop: float,
) -> WeeklyTradeRecord:
    hour, minute_in_hour = divmod(9 * 60 + 30 + minute, 60)
    return WeeklyTradeRecord(
        trade_id=trade_id,
        entry_time=datetime(2026, 8, 17, hour, minute_in_hour, tzinfo=ET),
        instrument="MNQ",
        strategy="vwap_continuation",
        setup=setup,
        direction=direction,
        pnl_usd=pnl,
        r_multiple=r,
        planned_risk_usd=100.0,
        initial_stop_points=stop,
        slippage_usd=4.0,
        fees_usd=2.0,
    )


def rejected(signal_id: str, reason: str, *, minute: int = 0) -> RejectedSignalRecord:
    return RejectedSignalRecord(
        signal_id=signal_id,
        timestamp=datetime(2026, 8, 18, 10, minute, tzinfo=ET),
        instrument="MNQ",
        strategy="vwap_continuation",
        setup="vwap touch",
        direction="LONG",
        reason=reason,
    )


def evidence_policy() -> HypothesisPolicy:
    return HypothesisPolicy(
        minimum_trades=6,
        minimum_slice_trades=2,
        weak_slice_average_r_at_most=-0.5,
        weak_slice_loss_rate_at_least=0.75,
        account_stop_rate_at_least=0.25,
        average_cost_to_risk_at_least=0.05,
        losing_stop_ratio_at_least=1.5,
        minimum_outcome_group_trades=2,
        minimum_rejections=4,
        dominant_rejection_share_at_least=0.75,
        max_hypotheses=2,
    )


def report_records() -> tuple[WeeklyTradeRecord, ...]:
    return (
        accepted_trade(
            "t1", minute=0, pnl=-100, r=-1.0, setup="weak", direction="LONG", stop=10
        ),
        accepted_trade(
            "t2", minute=15, pnl=-50, r=-0.5, setup="weak", direction="SHORT", stop=12
        ),
        accepted_trade(
            "t3", minute=30, pnl=100, r=1.0, setup="strong", direction="LONG", stop=4
        ),
        accepted_trade(
            "t4", minute=45, pnl=-25, r=-0.25, setup="strong", direction="SHORT", stop=8
        ),
        accepted_trade(
            "t5", minute=60, pnl=80, r=0.75, setup="strong", direction="LONG", stop=4
        ),
        accepted_trade(
            "t6", minute=75, pnl=0, r=0.0, setup="strong", direction="SHORT", stop=5
        ),
    )


def full_source(*, trades: tuple[WeeklyTradeRecord, ...] | None = None) -> WeeklyReportInput:
    rejection_records = (
        rejected("r1", "DAILY_TRADE_LIMIT", minute=0),
        rejected("r2", "DAILY_TRADE_LIMIT", minute=1),
        rejected("r3", "DAILY_TRADE_LIMIT", minute=2),
        rejected("r4", "STALE_DATA", minute=3),
    )
    accepted = report_records() if trades is None else trades
    return WeeklyReportInput(
        period_start=date(2026, 8, 17),
        period_end=date(2026, 8, 23),
        signals_generated=len(accepted) + len(rejection_records),
        accepted_trades=accepted,
        rejected_signals=rejection_records,
        survival_baseline=survival_baseline(),
        hypothesis_policy=evidence_policy(),
    )


def test_weekly_report_covers_performance_rejections_costs_stops_and_survival() -> None:
    report = generate_weekly_report(full_source())
    payload = json.loads(report.to_json())

    assert payload["signals"] == {"generated": 10, "accepted": 6, "rejected": 4}
    assert report.overall.expectancy_usd == pytest.approx(5 / 6)
    assert report.overall.profit_factor == pytest.approx(180 / 175)
    assert report.overall.average_r == pytest.approx(0.0)
    assert {item.label for item in report.by_setup} == {"strong", "weak"}
    assert {item.label for item in report.by_time_of_day} == {"09:00 ET", "10:00 ET"}
    assert {item.label for item in report.by_direction} == {"LONG", "SHORT"}

    assert report.rejections_by_reason[0].label == "DAILY_TRADE_LIMIT"
    assert report.rejections_by_reason[0].count == 3
    assert report.rejections_by_reason[0].share == pytest.approx(0.75)
    assert report.stop_distances.median_points == pytest.approx(6.5)
    assert report.stop_distances.winning_median_points == pytest.approx(4.0)
    assert report.stop_distances.losing_median_points == pytest.approx(10.0)
    assert report.costs.slippage_usd == pytest.approx(24.0)
    assert report.costs.fees_usd == pytest.approx(12.0)
    assert report.costs.average_cost_to_planned_risk == pytest.approx(0.06)
    assert report.losing_sequences.sequence_lengths == (2, 1)
    assert report.losing_sequences.survival_margin_losses == 1

    assert payload["prop_survival_baseline"]["rates"]["pass_rate"] == pytest.approx(0.4)
    assert payload["prop_survival_baseline"]["rates"]["account_stop_rate"] == pytest.approx(
        0.3
    )
    assert [item.code for item in report.research_hypotheses] == [
        "WEAK_SLICE_SURVIVAL_EXPERIMENT",
        "EXECUTION_COST_STRESS_EXPERIMENT",
    ]
    assert all(not item.production_change_permitted for item in report.research_hypotheses)
    assert payload["production_changes"] == []
    assert "PRODUCTION CHANGES: NONE" in report.to_text()
    assert "Research hypotheses (maximum two)" in report.to_text()


def test_hypothesis_choice_is_deterministic_and_not_ranked_by_dollar_profit() -> None:
    original = generate_weekly_report(full_source())
    scaled_trades = tuple(replace(item, pnl_usd=item.pnl_usd * 100) for item in report_records())
    scaled = generate_weekly_report(full_source(trades=scaled_trades))

    assert scaled.overall.net_pnl_usd != original.overall.net_pnl_usd
    assert [item.code for item in scaled.research_hypotheses] == [
        item.code for item in original.research_hypotheses
    ]
    assert [item.evidence for item in scaled.research_hypotheses] == [
        item.evidence for item in original.research_hypotheses
    ]


def test_rejected_only_week_can_propose_a_control_audit_without_accepting_rejections() -> None:
    rejections = tuple(
        rejected(f"r{index}", "STALE_DATA" if index < 10 else "DAILY_TRADE_LIMIT", minute=index)
        for index in range(12)
    )
    source = WeeklyReportInput(
        period_start=date(2026, 8, 17),
        period_end=date(2026, 8, 23),
        signals_generated=12,
        rejected_signals=rejections,
        survival_baseline=survival_baseline(),
    )

    report = generate_weekly_report(source)
    payload = json.loads(report.to_json())

    assert report.overall.trades == 0
    assert report.overall.profit_factor is None
    assert payload["performance"]["overall"]["profit_factor"] is None
    assert [item.code for item in report.research_hypotheses] == ["REJECTION_CAUSE_AUDIT"]
    assert "never weaken a risk gate" in report.research_hypotheses[0].rationale


def test_no_threshold_no_hypothesis_and_daily_exports_remain_available() -> None:
    trade = accepted_trade(
        "only", minute=0, pnl=100, r=1, setup="strong", direction="LONG", stop=5
    )
    report = generate_weekly_report(
        WeeklyReportInput(
            period_start=date(2026, 8, 17),
            period_end=date(2026, 8, 23),
            signals_generated=1,
            accepted_trades=(trade,),
            survival_baseline=survival_baseline(),
        )
    )

    assert report.research_hypotheses == ()
    assert "NONE — no predeclared evidence threshold was met" in report.to_text()
    daily = generate_daily_report(
        DailyReportInput(
            session_date=date(2026, 8, 17),
            account_balance_usd=50_000,
            prop_failure_level_usd=48_000,
        )
    )
    assert daily.trade_count == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pnl_usd", float("nan"), "must be finite"),
        ("slippage_usd", -1.0, "must be non-negative"),
        ("initial_stop_points", 0.0, "must be positive"),
        ("direction", "FLAT", "LONG or SHORT"),
    ],
)
def test_weekly_trade_rejects_invalid_values(field: str, value, message: str) -> None:
    values = {
        "trade_id": "invalid",
        "entry_time": datetime(2026, 8, 17, 9, 30, tzinfo=ET),
        "instrument": "MNQ",
        "strategy": "test",
        "setup": "test setup",
        "direction": "LONG",
        "pnl_usd": 0.0,
        "r_multiple": 0.0,
        "planned_risk_usd": 100.0,
        "initial_stop_points": 5.0,
        "slippage_usd": 0.0,
        "fees_usd": 0.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        WeeklyTradeRecord(**values)


def test_weekly_records_require_aware_in_period_unique_and_reconciled_counts() -> None:
    valid = accepted_trade(
        "same", minute=0, pnl=1, r=0.01, setup="test", direction="LONG", stop=5
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(valid, entry_time=datetime(2026, 8, 17, 9, 30))

    base = {
        "period_start": date(2026, 8, 17),
        "period_end": date(2026, 8, 23),
        "signals_generated": 2,
        "accepted_trades": (valid, valid),
        "survival_baseline": survival_baseline(),
    }
    with pytest.raises(ValueError, match="ids must be unique"):
        WeeklyReportInput(**base)
    with pytest.raises(ValueError, match="must equal"):
        WeeklyReportInput(**{**base, "signals_generated": 3})
    with pytest.raises(ValueError, match="seven calendar days"):
        WeeklyReportInput(**{**base, "period_end": date(2026, 8, 24)})


def test_survival_baseline_validates_counts_and_finite_values() -> None:
    with pytest.raises(ValueError, match="account-stop counts"):
        survival_baseline(hard_failures=70, internal_locks=40)
    with pytest.raises(ValueError, match="must be finite"):
        survival_baseline(expected_profit_per_account_usd=float("inf"))
    with pytest.raises(ValueError, match="required when simulations pass"):
        survival_baseline(median_days_to_pass=None)
    with pytest.raises(ValueError, match="tail drawdown"):
        survival_baseline(average_drawdown_usd=1_000, tail_drawdown_usd=900)
