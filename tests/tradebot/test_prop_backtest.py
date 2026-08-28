"""Stage-0 integration tests for the complete three-layer prop backtest path."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradebot.backtest.prop_engine import run_prop_backtest
from tradebot.config import (
    BrokerConfig,
    Config,
    CostConfig,
    DailyRisk,
    PerTradeRisk,
    PropFirmConfig,
    RiskConfig,
    SessionConfig,
    SimulatedBrokerConfig,
)
from tradebot.core.types import Side
from tradebot.features.pipeline import FeatureFrame
from tradebot.risk.prop import MarketDayStatus
from tradebot.strategy.base import Strategy, StrategySpec


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "prop_firms" / "tradeify_growth_50k.yaml"
ET = ZoneInfo("America/New_York")
RULES_CURRENT = datetime(2026, 8, 22, 10, 0, tzinfo=ET)


class DailySignalStrategy(Strategy):
    """Emit one simple, fully deterministic 2R setup each test session."""

    def __init__(self, *, stop_points: float = 1.0) -> None:
        super().__init__(
            StrategySpec(
                name="daily_test_signal",
                description="one deterministic integration-test setup per session",
                entry_conditions=("daily_test_signal",),
                invalidation_conditions=(),
                stop_loss=f"{stop_points} points",
                profit_target="2R",
                filters=(),
                max_trades_per_session=1,
            )
        )
        self.stop_points = stop_points

    def _entry_signal(self, ctx):
        if ctx.timestamp.time() != time(10, 0):
            return None
        close = float(ctx.bar["close"])
        # The production entry envelope permits four MNQ ticks (one point) of
        # adverse movement.  Keep this fixture at exactly 2R at that executable
        # bound, rather than only at the signal bar's close.
        executable_entry = close + 1.0
        return self._make_intent(
            ctx,
            Side.BUY,
            close - self.stop_points,
            target_price=executable_entry
            + 2 * (executable_entry - (close - self.stop_points)),
            conditions=("daily_test_signal",),
        )


def _bars(*, days: int = 1) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in pd.bdate_range("2022-04-01", periods=days):
        index = pd.date_range(
            day + pd.Timedelta(hours=10), periods=3, freq="1min", tz=ET
        )
        frames.append(
            pd.DataFrame(
                {
                    "open": [100.0, 100.0, 102.0],
                    "high": [100.25, 103.0, 105.25],
                    "low": [99.75, 99.5, 101.75],
                    "close": [100.0, 102.0, 105.0],
                    "volume": [100.0, 100.0, 100.0],
                },
                index=index,
            )
        )
    result = pd.concat(frames)
    result.index.name = "timestamp"
    return result


def _features(bars: pd.DataFrame, strategy: Strategy) -> FeatureFrame:
    return FeatureFrame(
        frame=pd.DataFrame(index=bars.index),
        spec=strategy.features,
        warmup_bars=0,
    )


def _config(
    *, internal_buffer: float = 400.0, personal_max_contracts: int = 3
) -> Config:
    return Config(
        instrument="MNQ",
        timeframe="1min",
        session=SessionConfig(
            entry_open_buffer_minutes=0,
            entry_close_buffer_minutes=0,
        ),
        risk=RiskConfig(
            starting_equity_usd=50_000.0,
            max_open_positions=1,
            per_trade=PerTradeRisk(
                risk_pct_of_equity=0.375,
                max_risk_per_trade_usd=200.0,
                max_contracts=personal_max_contracts,
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
        broker=BrokerConfig(simulated=SimulatedBrokerConfig(latency_ms=0)),
        prop_firm=PropFirmConfig(
            profile_path=str(PROFILE),
            phase="evaluation",
            internal_safety_buffer_usd=internal_buffer,
        ),
    )


def _run(
    *,
    config: Config | None = None,
    bars: pd.DataFrame | None = None,
    rules_as_of: datetime = RULES_CURRENT,
    market_days=None,
):
    bars = _bars() if bars is None else bars
    strategy = DailySignalStrategy()
    if market_days is None:
        market_days = {timestamp.date(): MarketDayStatus.REGULAR for timestamp in bars.index}
    return run_prop_backtest(
        bars,
        strategy,
        config or _config(),
        rule_verification_as_of=rules_as_of,
        market_day_status_provider=market_days,
        minimum_expected_rr=2.0,
        features=_features(bars, strategy),
    )


def _entry_evaluations(result):
    return [
        trace
        for trace in result.decision_traces
        if trace["operation"] == "ENTRY_EVALUATE"
    ]


def test_clean_stage_zero_entry_passes_all_layers_and_fresh_broker_guard() -> None:
    result = _run()

    assert result.split == "dev"
    assert len(result.trades) == 1
    evaluated = _entry_evaluations(result)
    assert len(evaluated) == 1
    assert evaluated[0]["strategy"]["allowed"]
    assert evaluated[0]["personal"]["allowed"]
    assert evaluated[0]["prop_firm"]["allowed"]
    verified = next(
        trace for trace in result.decision_traces if trace["operation"] == "ENTRY_VERIFY"
    )
    assert verified["strategy"]["allowed"]
    assert verified["personal"]["allowed"]
    assert verified["prop_firm"]["allowed"]
    assert verified["personal"]["detail"] == "personal token and current limits verified"
    assert result.final_prop_state.current_balance_usd == pytest.approx(
        result.final_equity
    )
    assert result.final_prop_state.internal_safety_buffer_usd == 400.0
    assert result.final_prop_state.open_micros == 0
    assert result.final_prop_state.trading_days == 1
    assert result.prop_state_points[-1].kind == "EOD"


def test_personal_valid_entry_is_refused_by_internal_prop_floor_buffer() -> None:
    result = _run(config=_config(internal_buffer=1_995.0))

    assert result.trades == []
    decision = _entry_evaluations(result)[0]
    assert decision["strategy"]["allowed"]
    assert decision["personal"]["allowed"]
    assert not decision["prop_firm"]["allowed"]
    assert "internal_safety_threshold" in decision["prop_firm"]["reason_codes"]
    assert result.final_prop_state.internal_safety_threshold_usd == 49_995.0


def test_personal_valid_entry_is_refused_by_selected_profile_contract_cap() -> None:
    config = _config(personal_max_contracts=50)
    # Eliminate the optional entry- and stop-gap reserves only in this targeted
    # synthetic fixture so the all-in personal size is above Growth's 40-micro cap
    # while remaining at or below the system's $200 maximum risk. Production keeps
    # both reserves; fees and stressed exit slippage still count here.
    config = replace(
        config,
        risk=replace(
            config.risk,
            per_trade=replace(
                config.risk.per_trade,
                risk_pct_of_equity=0.5,
                max_entry_gap_ticks=0.0,
                max_stop_gap_ticks=0.0,
            ),
        ),
    )
    result = _run(config=config)

    assert result.trades == []
    decision = _entry_evaluations(result)[0]
    assert decision["personal"]["allowed"]
    assert 40 < decision["personal"]["contracts"] <= 50
    assert decision["personal"]["risk_usd"] <= 200.0
    assert "contract_limit" in decision["prop_firm"]["reason_codes"]


def test_historical_bar_time_cannot_make_stale_current_rules_look_fresh() -> None:
    stale_as_of = datetime(2026, 8, 24, 10, 0, tzinfo=ET)

    result = _run(rules_as_of=stale_as_of)

    assert result.trades == []
    decision = _entry_evaluations(result)[0]
    assert decision["context"]["facts"]["market_data_timestamp"].startswith("2022-")
    assert decision["context"]["facts"]["rule_verification_as_of"].startswith("2026-")
    assert decision["personal"]["allowed"]
    assert "profile_stale" in decision["prop_firm"]["reason_codes"]


def test_unknown_or_missing_market_day_is_fail_closed() -> None:
    result = _run(market_days={})

    assert result.trades == []
    decision = _entry_evaluations(result)[0]
    assert decision["personal"]["allowed"]
    assert "holiday_status_unknown" in decision["prop_firm"]["reason_codes"]


def test_prop_state_uses_broker_truth_across_eod_transitions() -> None:
    bars = _bars(days=2)

    result = _run(bars=bars)

    assert len(result.trades) == 2
    assert result.final_prop_state.trading_days == 2
    assert result.final_prop_state.current_balance_usd == pytest.approx(
        result.final_equity
    )
    assert result.final_prop_state.highest_end_of_day_balance_usd == pytest.approx(
        result.final_prop_state.current_balance_usd
    )
    assert result.final_prop_state.drawdown_floor_usd == pytest.approx(
        result.final_prop_state.highest_end_of_day_balance_usd - 2_000.0
    )
    assert sum(point.kind == "EOD" for point in result.prop_state_points) == 2


def test_market_day_source_is_required_and_personal_limits_cannot_be_weakened() -> None:
    bars = _bars()
    strategy = DailySignalStrategy()
    with pytest.raises(TypeError, match="market_day_status_provider"):
        run_prop_backtest(
            bars,
            strategy,
            _config(),
            rule_verification_as_of=RULES_CURRENT,
            market_day_status_provider=None,  # type: ignore[arg-type]
            minimum_expected_rr=2.0,
            features=_features(bars, strategy),
        )

    base = _config()
    weak = replace(
        base,
        risk=replace(
            base.risk,
            daily=replace(base.risk.daily, max_trades_per_day=2),
        ),
    )
    weak_strategy = DailySignalStrategy()
    with pytest.raises(ValueError, match="max_trades_per_day"):
        run_prop_backtest(
            bars,
            weak_strategy,
            weak,
            rule_verification_as_of=RULES_CURRENT,
            market_day_status_provider={bars.index[0].date(): MarketDayStatus.REGULAR},
            minimum_expected_rr=2.0,
            features=_features(bars, weak_strategy),
        )
