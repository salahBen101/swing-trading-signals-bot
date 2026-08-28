"""Stage-1 market replay is local, fully audited, and terminally flat."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradebot.app.replay import MarketReplaySafetyError, run_market_replay
from tradebot.config import (
    BrokerConfig,
    Config,
    CostConfig,
    DailyRisk,
    DeploymentConfig,
    PerTradeRisk,
    PropFirmConfig,
    RiskConfig,
    SessionConfig,
    SimulatedBrokerConfig,
)
from tradebot.core.types import RejectReason, Side, TradingMode
from tradebot.data.splits import HOLDOUT_START, VALIDATION_START
from tradebot.features.pipeline import FeatureFrame
from tradebot.journal.queries import JournalReader
from tradebot.risk.prop import MarketDayStatus
from tradebot.strategy.base import Strategy, StrategySpec, TradingHours


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
ET = ZoneInfo("America/New_York")
RULES_CURRENT = datetime(2026, 8, 22, 10, 0, tzinfo=ET)


class FullSessionReplayStrategy(Strategy):
    """Hold one deterministic trade to the flatten time, then try one forbidden entry."""

    def __init__(self) -> None:
        super().__init__(
            StrategySpec(
                name="full_session_replay_test",
                description="exercise forced flat and the one-trade personal quota",
                entry_conditions=("scheduled_test_signal",),
                invalidation_conditions=(),
                stop_loss="fixed 90 points",
                profit_target="fixed 900 points",
                filters=(),
                max_trades_per_session=2,
                trading_hours=TradingHours(
                    earliest_entry=time(9, 30),
                    latest_entry=time(16, 0),
                    force_flat_at=time(15, 58),
                ),
            )
        )

    def _entry_signal(self, ctx):
        if ctx.timestamp.time() == time(9, 31):
            ctx.reject(RejectReason.FILTER_REGIME, "deterministic test filter")
            return None
        if ctx.timestamp.time() not in {time(10, 0), time(15, 59)}:
            return None
        close = float(ctx.bar["close"])
        return self._make_intent(
            ctx,
            Side.BUY,
            close - 90.0,
            target_price=close + 900.0,
            conditions=("scheduled_test_signal",),
        )


def replay_bars() -> pd.DataFrame:
    index = pd.date_range("2022-04-01 09:30", periods=390, freq="1min", tz=ET)
    bars = pd.DataFrame(
        {
            "open": 18_000.0,
            "high": 18_001.0,
            "low": 17_999.0,
            "close": 18_000.0,
            "volume": 100.0,
        },
        index=index,
    )
    bars.index.name = "timestamp"
    return bars


def replay_config() -> Config:
    return Config(
        mode=TradingMode.PAPER,
        instrument="MNQ",
        timeframe="1min",
        session=SessionConfig(
            entry_open_buffer_minutes=0,
            entry_close_buffer_minutes=0,
            flatten_before_close_minutes=2,
        ),
        risk=RiskConfig(
            starting_equity_usd=50_000.0,
            max_open_positions=1,
            per_trade=PerTradeRisk(
                # Permit one contract after the production stop-gap, fees, and
                # stressed-exit reserves are included in the all-in envelope.
                risk_pct_of_equity=0.4,
                max_risk_per_trade_usd=200.0,
                max_contracts=3,
                min_contracts=1,
            ),
            daily=DailyRisk(
                max_daily_loss_usd=200.0,
                max_daily_loss_r=4.0,
                max_trades_per_day=1,
            ),
        ),
        costs=CostConfig(
            commission_round_trip_usd=1.24,
            slippage_ticks_per_side=1.0,
        ),
        broker=BrokerConfig(
            adapter="simulated",
            environment="demo",
            simulated=SimulatedBrokerConfig(latency_ms=0),
        ),
        prop_firm=PropFirmConfig(
            profile_path=str(PROFILE),
            phase="evaluation",
            internal_safety_buffer_usd=400.0,
        ),
        deployment=DeploymentConfig(stage=1),
    )


def replay_features(bars: pd.DataFrame, strategy: Strategy) -> FeatureFrame:
    return FeatureFrame(
        frame=pd.DataFrame(index=bars.index),
        spec=strategy.features,
        warmup_bars=0,
    )


def test_full_stage_one_session_is_audited_reported_and_flat(tmp_path: Path) -> None:
    bars = replay_bars()
    strategy = FullSessionReplayStrategy()
    journal_path = tmp_path / "replay.sqlite3"
    result = run_market_replay(
        bars,
        strategy,
        replay_config(),
        rule_verification_as_of=RULES_CURRENT,
        market_day_statuses={bars.index[0].date(): MarketDayStatus.REGULAR},
        minimum_expected_rr=2.0,
        journal_path=journal_path,
        report_directory=tmp_path / "reports",
        features=replay_features(bars, strategy),
        dataset_hash="fixture-dev-sha256",
    )

    assert result.prop_result.deployment_stage.value == 1
    assert result.prop_result.split == "dev"
    assert result.ended_flat
    assert result.working_orders_at_end == 0
    assert result.prop_result.final_prop_state.open_micros == 0
    assert len(result.prop_result.trades) == 1
    trade = result.prop_result.trades[0]
    assert trade.entry_time.date() == trade.exit_time.date() == bars.index[0].date()
    assert trade.exit_time.time() == time(15, 59)

    assert len(result.daily_reports) == 1
    report = result.daily_reports[0]
    assert report.trade_count == 1
    assert report.signals_generated == 3
    assert report.signals_accepted == 1
    assert report.signals_rejected == 2
    assert dict(report.rejection_counts) == {
        "PERSONAL:MAX_TRADES_PER_DAY": 1,
        "STRATEGY:FILTER_REGIME": 1,
    }
    artifact = result.report_artifacts[0]
    assert artifact.json_path.read_text(encoding="utf-8") == report.to_json() + "\n"
    assert artifact.text_path.read_text(encoding="utf-8") == report.to_text() + "\n"

    with JournalReader(journal_path, run_id=result.run_id) as reader:
        assert len(reader.signals()) == 2
        assert len(reader.rejections()) == 2
        assert len(reader.orders()) == 4  # entry, stop, target, forced market exit
        assert len(reader.fills()) == 2
        assert len(reader.trades()) == 1
        events = reader.events(limit=2_000)
        run = reader.runs()[0]

    assert run["broker"] == "simulated"
    assert run["mode"] == "MARKET_REPLAY_STAGE_1"
    kinds = [event["kind"] for event in events]
    assert kinds.count("RISK_DECISION") == 6
    assert "PROP_ACCOUNT_STATE" in kinds
    assert "DAILY_REPORT" in kinds
    assert "MARKET_REPLAY_COMPLETE" in kinds

    decisions = [
        json.loads(event["payload"])
        for event in events
        if event["kind"] == "RISK_DECISION"
    ]
    operations = [(item["operation"], item["allowed"]) for item in decisions]
    assert operations.count(("ENTRY_EVALUATE", True)) == 1
    assert operations.count(("ENTRY_EVALUATE", False)) == 1
    assert operations.count(("ENTRY_VERIFY", True)) == 1
    assert operations.count(("RISK_REDUCING_VERIFY", True)) == 3
    refused = next(item for item in decisions if not item["allowed"])
    state = refused["context"]["facts"]["prop_account_state"]
    assert state["profile_id"] == "tradeify_growth_50k_current"
    assert state["rule_set_name"] == "evaluation"
    assert state["trades_this_session"] == 1
    assert state["has_open_exposure"] is False
    assert state["internal_safety_buffer_usd"] == 400.0
    assert state["daily_pnl_usd"] == pytest.approx(trade.net_pnl_usd)
    assert not (_all_mapping_keys(decisions) & {"token", "token_id", "signature"})


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            replace(
                replay_config(),
                mode=TradingMode.BACKTEST,
                deployment=DeploymentConfig(stage=0),
            ),
            "Stage 1",
        ),
        (replace(replay_config(), deployment=DeploymentConfig(stage=2)), "Stage 1"),
        (replace(replay_config(), deployment=DeploymentConfig(stage=3)), "Stage 3/4"),
        (
            replace(
                replay_config(),
                broker=replace(replay_config().broker, adapter="tradovate"),
            ),
            "Tradovate/external",
        ),
        (
            replace(
                replay_config(),
                broker=replace(replay_config().broker, environment="live"),
            ),
            "live",
        ),
    ],
)
def test_stage_one_runner_refuses_other_stages_and_external_routes(
    tmp_path: Path, config: Config, match: str
) -> None:
    bars = replay_bars()
    strategy = FullSessionReplayStrategy()
    with pytest.raises(MarketReplaySafetyError, match=match):
        run_market_replay(
            bars,
            strategy,
            config,
            rule_verification_as_of=RULES_CURRENT,
            market_day_statuses={bars.index[0].date(): MarketDayStatus.REGULAR},
            minimum_expected_rr=2.0,
            journal_path=tmp_path / "must-not-exist.sqlite3",
            report_directory=tmp_path / "must-not-exist",
            features=replay_features(bars, strategy),
        )
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


@pytest.mark.parametrize("status", [None, MarketDayStatus.UNKNOWN])
def test_calendar_coverage_is_required_before_a_journal_is_created(
    tmp_path: Path, status: MarketDayStatus | None
) -> None:
    bars = replay_bars()
    strategy = FullSessionReplayStrategy()
    statuses = {} if status is None else {bars.index[0].date(): status}
    journal_path = tmp_path / "must-not-exist.sqlite3"
    with pytest.raises(MarketReplaySafetyError, match="market-day map"):
        run_market_replay(
            bars,
            strategy,
            replay_config(),
            rule_verification_as_of=RULES_CURRENT,
            market_day_statuses=statuses,
            minimum_expected_rr=2.0,
            journal_path=journal_path,
            report_directory=tmp_path / "must-not-exist",
            features=replay_features(bars, strategy),
        )
    assert not journal_path.exists()


@pytest.mark.parametrize("boundary", [VALIDATION_START, HOLDOUT_START])
def test_stage_one_refuses_non_dev_rows_without_creating_a_journal(
    tmp_path: Path, boundary: pd.Timestamp
) -> None:
    bars = replay_bars().copy()
    bars.index = pd.date_range(boundary, periods=len(bars), freq="1min")
    strategy = FullSessionReplayStrategy()
    journal_path = tmp_path / "must-not-exist.sqlite3"
    with pytest.raises(MarketReplaySafetyError, match="VALIDATION/HOLDOUT"):
        run_market_replay(
            bars,
            strategy,
            replay_config(),
            rule_verification_as_of=RULES_CURRENT,
            market_day_statuses={bars.index[0].date(): MarketDayStatus.REGULAR},
            minimum_expected_rr=2.0,
            journal_path=journal_path,
            report_directory=tmp_path / "must-not-exist",
            features=replay_features(bars, strategy),
        )
    assert not journal_path.exists()


def _all_mapping_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).casefold())
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys
