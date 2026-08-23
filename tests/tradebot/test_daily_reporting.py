from __future__ import annotations

import json
from datetime import date

import pytest

from tradebot.reporting import (
    DailyReportInput,
    DailyTradeRecord,
    HistoricalSessionBaseline,
    generate_daily_report,
)


def trade(
    trade_id: str,
    *,
    pnl: float,
    r: float,
    slippage: float,
    fees: float,
    setup: str = "vwap continuation",
    execution: str = "target",
) -> DailyTradeRecord:
    return DailyTradeRecord(
        trade_id=trade_id,
        instrument="MNQ",
        strategy="vwap_continuation",
        setup=setup,
        execution=execution,
        pnl_usd=pnl,
        r_multiple=r,
        slippage_usd=slippage,
        fees_usd=fees,
    )


def test_daily_report_totals_every_required_field_and_serializes() -> None:
    source = DailyReportInput(
        session_date=date(2026, 8, 22),
        account_balance_usd=51_240,
        prop_failure_level_usd=49_600,
        trades=(trade("one", pnl=85, r=0.5, slippage=1.0, fees=1.24),),
        signals_generated=3,
        signals_accepted=1,
        rejection_counts={"DAILY_TRADE_LIMIT": 1, "STALE_DATA": 1},
        payout_progress_usd=1_240,
        payout_goal_usd=3_000,
    )

    report = generate_daily_report(source)
    payload = json.loads(report.to_json())

    assert report.pnl_usd == 85
    assert report.total_r == pytest.approx(0.5)
    assert report.slippage_usd == pytest.approx(1.0)
    assert report.fees_usd == pytest.approx(1.24)
    assert report.distance_to_failure_usd == 1_640
    assert report.payout_progress_fraction == pytest.approx(1_240 / 3_000)
    assert payload["signals"] == {
        "accepted": 1,
        "generated": 3,
        "rejected": 2,
        "rejection_reasons": {"DAILY_TRADE_LIMIT": 1, "STALE_DATA": 1},
    }
    assert report.abnormality_flags == ()
    assert "Rule violations: NONE" in report.to_text()
    assert "Setup: vwap continuation=1" in report.to_text()
    assert "Execution: target=1" in report.to_text()


def test_daily_report_flags_policy_rule_drawdown_and_historical_anomalies() -> None:
    baseline = HistoricalSessionBaseline(
        sample_sessions=100,
        mean_pnl_usd=20,
        pnl_std_usd=20,
        mean_signals=1,
        signals_std=1,
        mean_slippage_usd=1,
        slippage_std_usd=1,
    )
    source = DailyReportInput(
        session_date=date(2026, 8, 22),
        account_balance_usd=48_000,
        prop_failure_level_usd=48_000,
        trades=(
            trade("one", pnl=-110, r=-1, slippage=3, fees=1.24),
            trade("two", pnl=-110, r=-1, slippage=3, fees=1.24),
        ),
        signals_generated=5,
        signals_accepted=2,
        rejection_counts={"RISK": 3},
        rule_violations=("unexpected second trade",),
        max_trades_per_day=1,
        max_daily_loss_usd=200,
        historical=baseline,
    )

    report = generate_daily_report(source)

    assert report.abnormality_flags == (
        "TRADE_QUOTA_EXCEEDED",
        "DAILY_LOSS_LIMIT_REACHED",
        "PROP_FAILURE_LEVEL_REACHED",
        "RULE_VIOLATION_RECORDED",
        "PNL_OUTLIER",
        "SIGNAL_COUNT_HIGH",
        "SLIPPAGE_HIGH",
    )
    assert report.historical_comparison["sample_sessions"] == 100
    assert report.historical_comparison["pnl_z_score"] == pytest.approx(-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"signals_generated": 1, "signals_accepted": 1, "rejection_counts": {"RISK": 1}},
            "accepted plus rejected",
        ),
        (
            {"signals_generated": 0, "signals_accepted": 1},
            "accepted signals cannot exceed",
        ),
        (
            {"payout_progress_usd": 100},
            "progress and goal",
        ),
    ],
)
def test_daily_report_rejects_internally_inconsistent_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DailyReportInput(
            session_date=date(2026, 8, 22),
            account_balance_usd=50_000,
            prop_failure_level_usd=48_000,
            **kwargs,
        )


def test_completed_trades_cannot_exceed_accepted_signals() -> None:
    with pytest.raises(ValueError, match="completed trades"):
        DailyReportInput(
            session_date=date(2026, 8, 22),
            account_balance_usd=50_000,
            prop_failure_level_usd=48_000,
            trades=(trade("one", pnl=1, r=0.1, slippage=0, fees=0),),
        )
