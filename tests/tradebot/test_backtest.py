"""M8: the backtest engine and the statistics module.

The look-ahead guard is the load-bearing test. Everything else in a backtester can be
slightly wrong and still produce a useful ranking; look-ahead produces confident nonsense.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tradebot.analytics.metrics import (
    Metrics,
    by_year,
    compute_metrics,
    selection_bar,
)
from tradebot.analytics.report import json_report, text_report
from tradebot.backtest.engine import BacktestTerminalStateError, run_backtest
from tradebot.config import (
    Config,
    CostConfig,
    DailyRisk,
    PerTradeRisk,
    RiskConfig,
    SessionConfig,
    SimulatedBrokerConfig,
    BrokerConfig,
)
from tradebot.core.clock import MARKET_TZ
from tradebot.core.models import Trade
from tradebot.core.types import ExitReason, Side
from tradebot.data.splits import Split
from tradebot.features.pipeline import FeatureFrame, FeatureSpec, build_features
from tradebot.strategy.base import Strategy, StrategySpec
from tradebot.strategy.registry import build_strategy, known_strategies


def a_config(**over) -> Config:
    base = dict(
        instrument="MNQ",
        timeframe="1min",
        session=SessionConfig(entry_open_buffer_minutes=1, entry_close_buffer_minutes=5),
        risk=RiskConfig(
            starting_equity_usd=50_000.0,
            per_trade=PerTradeRisk(risk_pct_of_equity=1.0, max_risk_per_trade_usd=500.0,
                                   max_contracts=3, min_contracts=1),
            daily=DailyRisk(max_daily_loss_usd=5_000.0, max_daily_loss_r=50.0,
                            max_trades_per_day=50),
        ),
        costs=CostConfig(commission_round_trip_usd=1.24, slippage_ticks_per_side=1.0),
        broker=BrokerConfig(simulated=SimulatedBrokerConfig(latency_ms=0)),
    )
    base.update(over)
    return Config(**base)


class SpyStrategy(Strategy):
    """Records bar/feature timestamp pairing without placing orders."""

    def __init__(self) -> None:
        super().__init__(StrategySpec(
            name="spy", description="test double", entry_conditions=("spy",),
            invalidation_conditions=(), stop_loss="fixed", profit_target="fixed",
            filters=(), max_trades_per_session=1,
        ))
        self.seen: list[tuple[datetime, pd.Timestamp]] = []

    def _entry_signal(self, ctx):
        self.seen.append((ctx.timestamp, ctx.features.name))
        return None


class OneShotStrategy(Strategy):
    """Opens one deliberately long-lived trade for terminal-state tests."""

    def __init__(self, signal_index: int = 5) -> None:
        super().__init__(StrategySpec(
            name="one_shot", description="test double", entry_conditions=("one_shot",),
            invalidation_conditions=(), stop_loss="100 points", profit_target="100 points",
            filters=(), max_trades_per_session=1,
        ))
        self.signal_index = signal_index

    def _entry_signal(self, ctx):
        if ctx.i != self.signal_index:
            return None
        close = float(ctx.bar["close"])
        return self._make_intent(
            ctx, Side.BUY, close - 100.0, target_price=close + 100.0,
            conditions=("one_shot",),
        )


def trade(pnl: float, *, r: float = 1.0, entry=None, exit_=None, side=Side.BUY,
          reason=ExitReason.TAKE_PROFIT, bars=10) -> Trade:
    entry = entry or datetime(2024, 4, 1, 11, 0, tzinfo=MARKET_TZ)
    exit_ = exit_ or (entry + timedelta(minutes=bars))
    return Trade(
        trade_id=f"t{pnl}{entry.isoformat()}", instrument="MNQ", strategy="s", side=side,
        quantity=1, entry_time=entry, entry_price=18000.0, exit_time=exit_,
        exit_price=18000.0 + pnl, exit_reason=reason, gross_pnl_usd=pnl + 1.24,
        commission_usd=1.24, net_pnl_usd=pnl, r_multiple=r, bars_held=bars,
        initial_stop=17990.0,
    )


# ============================================================ metrics


def test_metrics_on_an_empty_trade_list_are_all_zero():
    m = compute_metrics([])
    assert m.trades == 0 and m.net_pnl == 0.0
    assert m.equity_curve == (50_000.0,)
    assert m.t_statistic is None


def test_the_headline_numbers_match_a_hand_computed_set():
    trades = [trade(100), trade(-50), trade(200), trade(-50), trade(-100)]
    m = compute_metrics(trades, starting_equity=10_000.0)

    assert m.trades == 5
    assert m.wins == 2 and m.losses == 3
    assert m.win_rate == pytest.approx(0.4)
    assert m.net_pnl == pytest.approx(100.0)
    assert m.expectancy == pytest.approx(20.0)
    assert m.avg_winner == pytest.approx(150.0)
    assert m.avg_loser == pytest.approx(-200 / 3)
    assert m.largest_winner == 200 and m.largest_loser == -100
    assert m.profit_factor == pytest.approx(300 / 200)
    assert m.final_equity == pytest.approx(10_100.0)


def test_max_drawdown_is_peak_to_trough_not_start_to_end():
    # Up 500, down 800, up 100. The drawdown is 800, not the 200 net.
    trades = [trade(500), trade(-800), trade(100)]
    m = compute_metrics(trades, starting_equity=10_000.0)
    assert m.max_drawdown == pytest.approx(800.0)
    assert m.max_drawdown_pct == pytest.approx(800 / 10_500)


def test_consecutive_streaks_are_the_longest_runs():
    trades = [trade(1), trade(1), trade(-1), trade(1), trade(1), trade(1),
              trade(-1), trade(-1)]
    m = compute_metrics(trades)
    assert m.max_consecutive_wins == 3
    assert m.max_consecutive_losses == 2


def test_a_scratch_trade_counts_as_neither_a_win_nor_a_loss():
    m = compute_metrics([trade(100), trade(0), trade(-100)])
    assert (m.wins, m.losses, m.scratches) == (1, 1, 1)


def test_a_scratch_breaks_both_win_and_loss_streaks():
    m = compute_metrics([trade(-10), trade(0), trade(-10), trade(10), trade(0), trade(10)])
    assert m.max_consecutive_losses == 1
    assert m.max_consecutive_wins == 1


def test_largest_winner_and_loser_do_not_borrow_from_the_other_side():
    winners = compute_metrics([trade(10), trade(20)])
    losers = compute_metrics([trade(-10), trade(-20)])
    assert winners.largest_winner == 20 and winners.largest_loser == 0
    assert losers.largest_winner == 0 and losers.largest_loser == -20


def test_profit_factor_is_infinite_with_no_losers():
    assert compute_metrics([trade(10), trade(20)]).profit_factor == float("inf")


def test_sharpe_is_withheld_below_thirty_trades():
    assert compute_metrics([trade(10)] * 29).sharpe is None
    assert compute_metrics([trade(10 + i, r=0.1 + i / 100) for i in range(40)]).sharpe is not None


def test_sharpe_uses_r_multiples_not_trade_dollars():
    same_r = [trade(10 + i, r=1.0) for i in range(40)]
    assert compute_metrics(same_r).sharpe is None


def test_cost_share_of_gross_is_reported():
    m = compute_metrics([trade(100), trade(-50)])
    # gross = (100+1.24) + (-50+1.24) = 52.48; commission = 2.48
    assert m.cost_share_of_gross == pytest.approx(2.48 / 52.48)


def test_performance_is_broken_out_by_hour_weekday_side_and_exit_reason():
    base = datetime(2024, 4, 1, 10, 0, tzinfo=MARKET_TZ)  # a Monday
    trades = [
        trade(100, entry=base, side=Side.BUY, reason=ExitReason.TAKE_PROFIT),
        trade(-50, entry=base.replace(hour=14), side=Side.SELL,
              reason=ExitReason.STOP_LOSS),
        trade(30, entry=base + timedelta(days=1), side=Side.BUY,
              reason=ExitReason.SESSION_CLOSE),
    ]
    m = compute_metrics(trades)

    assert {b.label for b in m.by_hour} == {"10:00", "14:00"}
    assert {b.label for b in m.by_weekday} == {"Mon", "Tue"}
    assert {b.label for b in m.by_side} == {"BUY", "SELL"}
    assert {b.label for b in m.by_exit_reason} == {
        "TAKE_PROFIT", "STOP_LOSS", "SESSION_CLOSE"
    }
    buy = next(b for b in m.by_side if b.label == "BUY")
    assert buy.trades == 2 and buy.net_pnl == pytest.approx(130.0)


def test_the_year_breakdown_exposes_a_single_good_year():
    """The test that killed this project's one candidate strategy."""
    trades = (
        [trade(-10, entry=datetime(2019, 6, 1, 11, tzinfo=MARKET_TZ)) for _ in range(20)]
        + [trade(500, entry=datetime(2022, 6, 1, 11, tzinfo=MARKET_TZ)) for _ in range(20)]
    )
    years = by_year(trades)
    assert [b.label for b in years] == ["2019", "2022"]
    total = sum(b.net_pnl for b in years)
    assert years[1].net_pnl / total > 1.0, "2022 alone exceeds the total"


def test_the_selection_bar_rises_with_the_number_of_configurations_tried():
    assert selection_bar(1) == pytest.approx(1.96)
    assert selection_bar(9) > selection_bar(1)
    assert selection_bar(100) > selection_bar(9)


def test_a_result_is_judged_against_its_own_search_count():
    """The same numbers, two verdicts, depending only on how hard you searched.

    Alternating -80/+120 over 200 trades gives mean 20 and sd ~100, so t is about 2.8 —
    comfortably significant if this was the only idea tried, and not significant at all if
    it is the best of 500.
    """
    trades = [trade(-80 if i % 2 == 0 else 120, r=0.1) for i in range(200)]
    lenient = compute_metrics(trades, configurations_tried=1)
    strict = compute_metrics(trades, configurations_tried=500)

    assert lenient.t_statistic == pytest.approx(strict.t_statistic)
    assert 1.96 < lenient.t_statistic < selection_bar(500)
    assert lenient.clears_selection_bar
    assert not strict.clears_selection_bar


def test_a_statistically_significant_loser_never_clears_the_positive_edge_bar():
    trades = [trade(-120 if i % 2 == 0 else 80, r=-0.1) for i in range(200)]
    result = compute_metrics(trades)
    assert result.t_statistic < -selection_bar(1)
    assert not result.clears_selection_bar


def test_marked_equity_exposes_intratrade_drawdown_hidden_by_closed_trades():
    result = compute_metrics(
        [trade(100)],
        starting_equity=50_000,
        marked_equity_curve=[50_000, 49_000, 50_100],
    )
    assert result.max_drawdown == 1_000
    assert result.max_drawdown_pct == pytest.approx(0.02)


def test_metrics_serialise_to_json_friendly_types():
    m = compute_metrics([trade(100), trade(-50)])
    payload = m.to_dict()
    assert isinstance(payload["by_hour"], list)
    assert isinstance(payload["equity_times"][0], str)
    assert "selection_bar" in payload


# ============================================================ the backtest engine


@pytest.fixture(scope="module")
def sample_bars() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    frames = []
    price = 18000.0
    for d in range(12):
        day = pd.Timestamp("2022-04-01") + pd.Timedelta(days=d)
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=390,
                            freq="1min", tz=MARKET_TZ)
        closes = price + np.cumsum(rng.normal(0.02 if d % 2 else -0.02, 2.2, 390))
        opens = np.concatenate([[closes[0]], closes[:-1]])
        highs = np.maximum(opens, closes) + rng.uniform(0.5, 5.0, 390)
        lows = np.minimum(opens, closes) - rng.uniform(0.5, 5.0, 390)
        frames.append(pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": rng.integers(60, 500, 390).astype(float)},
            index=idx,
        ))
        price = float(closes[-1])
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out


def test_a_backtest_runs_and_reports_its_dataset(sample_bars):
    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    assert result.bars == len(sample_bars)
    assert result.sessions == 12
    assert result.warmup_bars > 0
    assert result.strategy_spec["name"] == "orb_breakout"
    assert len(result.equity_curve) == len(sample_bars)
    assert result.split == "dev"
    assert result.ended_flat and result.working_orders_at_end == 0


@pytest.mark.parametrize("name", known_strategies())
def test_every_strategy_backtests_without_error(name, sample_bars):
    result = run_backtest(sample_bars, build_strategy(name), a_config())
    for t in result.trades:
        assert t.entry_time <= t.exit_time
        assert t.quantity >= 1


def test_no_position_is_ever_held_overnight(sample_bars):
    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    for t in result.trades:
        assert t.entry_time.date() == t.exit_time.date(), (
            f"trade {t.trade_id} spans sessions"
        )


def test_only_one_position_is_ever_open_at_a_time(sample_bars):
    result = run_backtest(sample_bars, build_strategy("vwap_reversion"), a_config())
    ordered = sorted(result.trades, key=lambda t: t.entry_time)
    for earlier, later in zip(ordered, ordered[1:]):
        assert later.entry_time >= earlier.exit_time, "trades overlapped"


def test_costs_are_charged_on_every_trade(sample_bars):
    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    for t in result.trades:
        assert t.commission_usd > 0
        assert t.net_pnl_usd == pytest.approx(t.gross_pnl_usd - t.commission_usd)


def test_stress_costs_make_results_strictly_worse(sample_bars):
    strategy = build_strategy("vwap_reversion")
    normal = run_backtest(sample_bars, strategy, a_config())
    stressed = run_backtest(sample_bars, build_strategy("vwap_reversion"), a_config(),
                            stress_costs=True)
    assert "stress" in " ".join(stressed.notes).lower()
    if normal.trades and stressed.trades:
        normal_net = sum(t.net_pnl_usd for t in normal.trades) / len(normal.trades)
        stressed_net = sum(t.net_pnl_usd for t in stressed.trades) / len(stressed.trades)
        assert stressed_net <= normal_net + 1e-9


def test_splits_partition_the_backtest(sample_bars):
    # This sample sits entirely inside DEV, which is also the safe default.
    full = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config(),
                        split=Split.ALL)
    dev = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    assert dev.bars == full.bars
    assert any("ALL DATA" in note for note in full.notes)
    with pytest.raises(ValueError, match="no bars"):
        run_backtest(sample_bars, build_strategy("orb_breakout"), a_config(),
                     split=Split.VALIDATION)


def test_holdout_access_is_loudly_marked(sample_bars):
    holdout = sample_bars.iloc[:390].copy()
    holdout.index = holdout.index + pd.DateOffset(years=3)
    result = run_backtest(
        holdout, build_strategy("orb_breakout", {"min_volume_ratio": 1e9}),
        a_config(), split=Split.HOLDOUT,
    )
    assert any("LOCKED HOLDOUT DATA WAS ACCESSED" in note for note in result.notes)


def test_full_precomputed_features_are_aligned_to_the_selected_split(sample_bars):
    dev = sample_bars.iloc[:10].copy()
    validation = dev.copy()
    validation.index = validation.index + pd.DateOffset(years=2)
    combined = pd.concat([dev, validation])
    strategy = SpyStrategy()
    computed = build_features(combined, strategy.features)
    supplied = FeatureFrame(computed.frame, computed.spec, warmup_bars=0)

    run_backtest(
        combined, strategy, a_config(), split=Split.VALIDATION, features=supplied,
    )

    assert strategy.seen
    assert all(pd.Timestamp(bar_ts) == feature_ts for bar_ts, feature_ts in strategy.seen)
    assert all(bar_ts.year == 2024 for bar_ts, _ in strategy.seen)


def test_precomputed_features_with_the_wrong_spec_are_rejected(sample_bars):
    bars = sample_bars.iloc[:10]
    strategy = SpyStrategy()
    computed = build_features(bars, strategy.features)
    wrong = FeatureFrame(
        computed.frame, FeatureSpec(atr_period=99), warmup_bars=0,
    )
    with pytest.raises(ValueError, match="feature spec"):
        run_backtest(bars, strategy, a_config(), features=wrong)


def test_precomputed_features_missing_a_bar_are_rejected(sample_bars):
    bars = sample_bars.iloc[:10]
    strategy = SpyStrategy()
    computed = build_features(bars, strategy.features)
    missing = FeatureFrame(
        computed.frame.drop(index=bars.index[5]), computed.spec, warmup_bars=0,
    )
    with pytest.raises(ValueError, match="missing 1 bar timestamp"):
        run_backtest(bars, strategy, a_config(), features=missing)


def test_an_incomplete_sample_with_open_exposure_fails_instead_of_reporting(sample_bars):
    bars = sample_bars.iloc[:10]
    strategy = OneShotStrategy()
    computed = build_features(bars, strategy.features)
    supplied = FeatureFrame(computed.frame, computed.spec, warmup_bars=0)

    with pytest.raises(BacktestTerminalStateError, match="end of data.*unresolved exposure"):
        run_backtest(bars, strategy, a_config(), features=supplied)


def test_the_drawdown_floor_is_off_by_default_and_the_fact_is_stated(sample_bars):
    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    assert any("NOT enforced" in n for n in result.notes)


def test_a_backtest_does_not_touch_the_live_kill_switch_flag(tmp_path, sample_bars):
    from tradebot.config import KillSwitchConfig

    live_flag = tmp_path / "LIVE_KILL.flag"
    live_flag.write_text("{}", encoding="utf-8")
    cfg = a_config(risk=RiskConfig(kill_switch=KillSwitchConfig(flag_file=str(live_flag))))

    run_backtest(sample_bars.iloc[:800], build_strategy("orb_breakout"), cfg)
    assert live_flag.exists(), "a backtest must never clear the live kill switch"


# ============================================================ look-ahead guards


def test_poisoning_every_future_bar_does_not_change_a_single_trade(sample_bars):
    """The definitive look-ahead test.

    Run the backtest, then re-run it with every bar after the halfway point replaced with
    garbage. The trades taken in the first half must be byte-identical. If any of them
    moved, something read a bar that had not happened yet.
    """
    cutoff = len(sample_bars) // 2
    strategy_a = build_strategy("orb_breakout")
    clean = run_backtest(sample_bars, strategy_a, a_config())

    poisoned = sample_bars.copy()
    tail = poisoned.index[cutoff:]
    poisoned.loc[tail, ["open", "high", "low", "close"]] = 99_999.0
    poisoned.loc[tail, "volume"] = 1.0

    strategy_b = build_strategy("orb_breakout")
    dirty = run_backtest(poisoned, strategy_b, a_config())

    boundary = sample_bars.index[cutoff]
    clean_early = [t for t in clean.trades if t.exit_time < boundary]
    dirty_early = [t for t in dirty.trades if t.exit_time < boundary]

    assert len(clean_early) == len(dirty_early)
    for a, b in zip(clean_early, dirty_early):
        assert a.entry_time == b.entry_time
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_time == b.exit_time
        assert a.exit_price == pytest.approx(b.exit_price)
        assert a.net_pnl_usd == pytest.approx(b.net_pnl_usd)


def test_the_poison_guard_would_catch_a_real_leak(sample_bars):
    """A guard that cannot fail proves nothing: confirm the poison actually differs."""
    cutoff = len(sample_bars) // 2
    poisoned = sample_bars.copy()
    tail = poisoned.index[cutoff:]
    poisoned.loc[tail, ["open", "high", "low", "close"]] = 99_999.0

    from tradebot.features.pipeline import build_features

    clean_tail = build_features(sample_bars).frame.iloc[cutoff + 50]
    dirty_tail = build_features(poisoned).frame.iloc[cutoff + 50]
    assert clean_tail["close"] != dirty_tail["close"]


def test_a_backtest_is_reproducible(sample_bars):
    a = run_backtest(sample_bars, build_strategy("vwap_reversion"), a_config())
    b = run_backtest(sample_bars, build_strategy("vwap_reversion"), a_config())
    assert len(a.trades) == len(b.trades)
    assert [t.entry_time for t in a.trades] == [t.entry_time for t in b.trades]
    assert sum(t.net_pnl_usd for t in a.trades) == pytest.approx(
        sum(t.net_pnl_usd for t in b.trades)
    )


# ============================================================ reports


def test_the_text_report_states_everything_needed_to_judge_the_result(sample_bars):
    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    report = text_report(result, configurations_tried=14)

    for required in ("DATASET", "RULES", "COSTS", "RESULTS", "REJECTED SIGNALS",
                     "EVIDENCE", "selection bar", "configurations"):
        assert required in report, f"the report omits {required!r}"
    assert "thirteen intraday NQ strategy families" in report
    assert result.strategy_name in report


def test_the_report_names_the_exact_rules_that_produced_the_trades(sample_bars):
    result = run_backtest(sample_bars, build_strategy("vwap_reversion"), a_config())
    report = text_report(result)
    assert "stretched" in report.lower() or "vwap" in report.lower()
    assert "stop" in report.lower() and "target" in report.lower()


def test_the_json_report_round_trips(sample_bars):
    import json

    result = run_backtest(sample_bars, build_strategy("orb_breakout"), a_config())
    payload = json.loads(json_report(result, configurations_tried=6))
    assert payload["strategy"] == "orb_breakout"
    assert payload["metrics"]["selection_bar"] > 1.96
    assert "period" in payload and payload["period"]["bars"] == len(sample_bars)


def test_a_zero_trade_report_is_still_a_valid_report(sample_bars):
    # A strategy configured so nothing can fire.
    strategy = build_strategy("orb_breakout", {"min_volume_ratio": 1e9})
    result = run_backtest(sample_bars, strategy, a_config())
    assert result.trades == []
    report = text_report(result)
    assert "no trades were taken" in report
