"""Deterministic operator reports built from explicit, already-journalled inputs."""

from .daily import (
    DailyReport,
    DailyReportInput,
    DailyTradeRecord,
    HistoricalSessionBaseline,
    generate_daily_report,
)
from .weekly import (
    CostSummary,
    CountSlice,
    HypothesisPolicy,
    LosingSequenceSummary,
    PerformanceSlice,
    PropSurvivalBaseline,
    RejectedSignalRecord,
    ResearchHypothesis,
    StopDistanceSummary,
    WeeklyReportInput,
    WeeklyResearchReport,
    WeeklyTradeRecord,
    generate_weekly_report,
)

__all__ = [
    "DailyReport",
    "DailyReportInput",
    "DailyTradeRecord",
    "HistoricalSessionBaseline",
    "generate_daily_report",
    "CostSummary",
    "CountSlice",
    "HypothesisPolicy",
    "LosingSequenceSummary",
    "PerformanceSlice",
    "PropSurvivalBaseline",
    "RejectedSignalRecord",
    "ResearchHypothesis",
    "StopDistanceSummary",
    "WeeklyReportInput",
    "WeeklyResearchReport",
    "WeeklyTradeRecord",
    "generate_weekly_report",
]
