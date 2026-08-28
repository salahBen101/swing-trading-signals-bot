"""Temporal correctness: a signal must never be filled at a price that predates it.

This is the defect that invalidated the exploratory pilot. The pilot read a completed daily close,
then bought options using quotes from that same session - an execution price observed before the
signal existed. These tests exist so that regression is caught mechanically rather than by
someone noticing the P&L looks generous.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tradebot.rsi_options.calendar import (
    HALF_DAYS,
    MARKET_TZ,
    CalendarExpired,
    TradingCalendar,
)
from tradebot.rsi_options.experiment import ExecutionPolicy


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar()


class TestNextSessionIsNeverSameSession:
    def test_offset_zero_is_refused(self, cal):
        """Offset 0 IS the lookahead bug. It must be impossible to ask for it."""
        with pytest.raises(ValueError, match="same-session"):
            cal.next_session(date(2026, 8, 19), offset=0)

    def test_execution_policy_forbids_same_session(self):
        from tradebot.rsi_options.experiment import default_config

        cfg = default_config(("SPY",))
        assert cfg.execution.entry_session_offset >= 1
        assert cfg.execution.exit_session_offset >= 1
        assert cfg.validate() == []

    def test_config_rejects_same_session_entry(self):
        from dataclasses import replace

        from tradebot.rsi_options.experiment import default_config

        cfg = default_config(("SPY",))
        bad = replace(cfg, execution=ExecutionPolicy(entry_session_offset=0))
        problems = cfg.__class__.validate(bad)
        assert any("entry_session_offset" in p for p in problems)


class TestWeekendRollover:
    def test_friday_signal_executes_monday(self, cal):
        friday = date(2026, 8, 21)
        assert friday.weekday() == 4
        nxt = cal.next_session(friday)
        assert nxt == date(2026, 8, 24)
        assert nxt.weekday() == 0

    def test_naive_timedelta_would_have_been_saturday(self, cal):
        """The bug this replaces: date + 1 day lands on a Saturday."""
        from datetime import timedelta

        friday = date(2026, 8, 21)
        assert (friday + timedelta(days=1)).weekday() == 5      # Saturday - not a session
        assert cal.next_session(friday).weekday() == 0          # calendar gets Monday

    def test_saturday_is_not_a_session(self, cal):
        assert not cal.is_session(date(2026, 8, 22))

    def test_sunday_is_not_a_session(self, cal):
        assert not cal.is_session(date(2026, 8, 23))


class TestHolidays:
    def test_christmas_is_not_a_session(self, cal):
        assert not cal.is_session(date(2026, 12, 25))

    def test_july_fourth_observed_2026(self, cal):
        assert not cal.is_session(date(2026, 7, 3))

    def test_signal_before_thanksgiving_skips_the_holiday(self, cal):
        """Wednesday 2026-11-25 -> Thursday is Thanksgiving -> Friday 11-27 (a half day)."""
        wednesday = date(2026, 11, 25)
        assert cal.is_session(wednesday)
        assert not cal.is_session(date(2026, 11, 26))
        assert cal.next_session(wednesday) == date(2026, 11, 27)

    def test_half_day_is_still_a_tradeable_session(self, cal):
        d = date(2026, 11, 27)
        assert d in HALF_DAYS
        assert cal.is_session(d)
        sess = cal.session(d)
        assert sess.is_half_day
        assert sess.close_dt.hour == 13

    def test_signal_before_christmas_2026(self, cal):
        thursday = date(2026, 12, 24)
        assert cal.is_session(thursday)
        assert cal.next_session(thursday) == date(2026, 12, 28)


class TestSessionCounting:
    def test_sessions_between_excludes_weekend(self, cal):
        assert cal.sessions_between(date(2026, 8, 19), date(2026, 8, 26)) == 5

    def test_same_day_is_zero(self, cal):
        assert cal.sessions_between(date(2026, 8, 19), date(2026, 8, 19)) == 0

    def test_counting_is_independent_of_how_often_the_job_runs(self, cal):
        """The pilot counted script executions. Skipping runs made the hold-period stop
        fire late or never. Session counting cannot drift that way."""
        opened, later = date(2026, 8, 19), date(2026, 9, 16)
        assert cal.sessions_between(opened, later) == 19


class TestAfterClose:
    def test_partial_session_is_not_complete(self, cal):
        midday = datetime(2026, 8, 19, 11, 0, tzinfo=MARKET_TZ)
        assert not cal.is_after_close(midday)

    def test_after_four_pm_is_complete(self, cal):
        after = datetime(2026, 8, 19, 16, 1, tzinfo=MARKET_TZ)
        assert cal.is_after_close(after)

    def test_half_day_closes_at_one(self, cal):
        d = date(2026, 11, 27)
        assert cal.is_after_close(datetime(2026, 11, 27, 13, 5, tzinfo=MARKET_TZ), d)
        assert not cal.is_after_close(datetime(2026, 11, 27, 12, 0, tzinfo=MARKET_TZ), d)


class TestFailClosed:
    def test_date_past_the_table_raises(self, cal):
        with pytest.raises(CalendarExpired):
            cal.is_session(date(2099, 6, 1))

    def test_observed_calendar_needs_no_table(self):
        cal = TradingCalendar.from_observed([date(2099, 6, 1), date(2099, 6, 2)])
        assert cal.is_session(date(2099, 6, 1))
        assert not cal.is_session(date(2099, 6, 3))


class TestFullEntryTiming:
    def test_close_signal_becomes_next_session_order(self, cal):
        """The end-to-end property: signal known after T's close, order eligible in T+1."""
        signal_session = date(2026, 8, 19)
        signal_known_at = datetime(2026, 8, 19, 16, 30, tzinfo=MARKET_TZ)

        assert cal.is_after_close(signal_known_at, signal_session)
        execution_session = cal.next_session(signal_session, offset=1)

        assert execution_session > signal_session
        assert cal.is_session(execution_session)
        exec_open = cal.session(execution_session).open_dt
        assert exec_open > signal_known_at        # the fill cannot precede the signal
