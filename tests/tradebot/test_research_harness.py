"""Guards on the research harness.

A screening harness that leaks the future produces confident nonsense faster than the
production engine can, because it is built for sweeps. These tests are the price of
trusting anything it reports.

The two that matter:

* `test_no_hypothesis_reads_the_future` poisons every bar after a cutoff and asserts the
  trades before that cutoff are byte-identical.
* `test_the_harness_matches_the_production_engines_fill_rules` pins the three fill
  conventions the harness shares with `ExecutionEngine`, since the whole justification for
  a second simulator is that those conventions agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.research.dataset import build_research_frame, split_frame
from tradebot.research.harness import (
    ROUND_TRIP_POINTS,
    ExitRule,
    Hypothesis,
    screen,
    simulate,
    summarize,
)
from tradebot.research.hypotheses import PRE_REGISTERED
from tradebot.research.sessions import build_session_table, overnight_slice, rth_slice


# --------------------------------------------------------------------------- fixtures


def make_eth(sessions: int = 40, seed: int = 3) -> pd.DataFrame:
    """Synthetic full-Globex bars: an overnight block then an RTH block per session."""
    rng = np.random.default_rng(seed)
    frames = []
    price = 18_000.0
    for d in range(sessions):
        day = pd.Timestamp("2021-03-01", tz=MARKET_TZ) + pd.Timedelta(days=d)
        # Overnight: 18:00 the evening before through 09:29.
        on_index = pd.date_range(
            day - pd.Timedelta(days=1) + pd.Timedelta(hours=18), periods=200,
            freq="5min", tz=MARKET_TZ,
        )
        rth_index = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min",
            tz=MARKET_TZ,
        )
        for index, is_rth in ((on_index, False), (rth_index, True)):
            closes = price + np.cumsum(rng.normal(0, 4.0, len(index)))
            opens = np.concatenate([[closes[0]], closes[:-1]])
            highs = np.maximum(opens, closes) + rng.uniform(1.0, 8.0, len(index))
            lows = np.minimum(opens, closes) - rng.uniform(1.0, 8.0, len(index))
            frames.append(pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes,
                 "volume": rng.integers(100, 900, len(index)).astype(float),
                 "session": day, "is_rth": is_rth},
                index=index,
            ))
            price = float(closes[-1])
    out = pd.concat(frames).sort_index()
    out.index.name = "timestamp"
    return out


@pytest.fixture(scope="module")
def eth_bars() -> pd.DataFrame:
    return make_eth()


@pytest.fixture(scope="module")
def frame(eth_bars):
    return build_research_frame(eth_bars, timeframe="5min")


# --------------------------------------------------------------------------- session layer


def test_the_session_boundary_is_the_globex_day_not_midnight(eth_bars):
    """An 18:00 bar belongs to the next day's session, or 'the overnight move' is nonsense."""
    evening = eth_bars[eth_bars.index.hour == 18].iloc[0]
    assert pd.Timestamp(evening["session"]).date() > eth_bars.index[0].date()


def test_overnight_and_rth_slices_are_disjoint(eth_bars):
    """They must not overlap. They need not cover everything: the 16:00-17:00 post-close
    tail belongs to neither tonight's inventory build nor today's regular session, and is
    deliberately in neither slice."""
    rth = rth_slice(eth_bars)
    overnight = overnight_slice(eth_bars)
    assert not rth.index.intersection(overnight.index).size
    # Every RTH bar is a regular-hours bar; every overnight bar is not.
    assert rth["is_rth"].all()
    assert not overnight["is_rth"].any()


def test_session_columns_describe_only_the_past(eth_bars):
    table = build_session_table(eth_bars)
    sessions = sorted(table.index)

    # Yesterday's close is exactly that, and the first session has none.
    assert pd.isna(table.loc[sessions[0], "prior_close"])
    assert table.loc[sessions[1], "prior_close"] == table.loc[sessions[0], "rth_close"]

    # The gap is measured from the prior settlement to today's open.
    row = table.loc[sessions[5]]
    assert row["gap"] == pytest.approx(row["rth_open"] - row["prior_close"])


def test_the_daily_atr_excludes_the_session_it_describes(eth_bars):
    table = build_session_table(eth_bars)
    # Rebuilding on a truncated history must not change earlier values.
    truncated = build_session_table(
        eth_bars[eth_bars["session"] <= sorted(table.index)[25]]
    )
    common = truncated.index.intersection(table.index)
    pd.testing.assert_series_equal(
        truncated.loc[common, "daily_atr"], table.loc[common, "daily_atr"],
        check_names=False,
    )


# --------------------------------------------------------------------------- causality


@pytest.mark.parametrize("hypothesis", PRE_REGISTERED, ids=lambda h: h.id)
def test_no_hypothesis_reads_the_future(eth_bars, hypothesis):
    """Poison every bar after a cutoff; trades before it must be identical.

    This is the definitive test. If a rule's earlier trades move when only later data
    changes, that rule has read the future and every number it produces is fiction.
    """
    clean_frame = build_research_frame(eth_bars, timeframe="5min")

    cutoff_session = sorted(eth_bars["session"].unique())[28]
    poisoned = eth_bars.copy()
    tail = poisoned["session"] > cutoff_session
    poisoned.loc[tail, ["open", "high", "low", "close"]] = 99_999.0
    poisoned.loc[tail, "volume"] = 1.0
    dirty_frame = build_research_frame(poisoned, timeframe="5min")

    clean = screen(hypothesis, clean_frame.bars, clean_frame.atr, bar_minutes=5,
                   keep_trades=True)
    dirty = screen(hypothesis, dirty_frame.bars, dirty_frame.atr, bar_minutes=5,
                   keep_trades=True)

    boundary = pd.Timestamp(cutoff_session)
    clean_early = [t for t in clean.trade_list if pd.Timestamp(t.session) <= boundary]
    dirty_early = [t for t in dirty.trade_list if pd.Timestamp(t.session) <= boundary]

    assert len(clean_early) == len(dirty_early), f"{hypothesis.id} changed its trade count"
    for a, b in zip(clean_early, dirty_early, strict=True):
        assert a.entry_index == b.entry_index
        assert a.direction == b.direction
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_price == pytest.approx(b.exit_price)


def test_the_poison_guard_can_actually_fail(eth_bars):
    """A guard that cannot fail proves nothing."""
    frame = build_research_frame(eth_bars, timeframe="5min")
    bars = frame.bars

    def peeking(b: pd.DataFrame) -> np.ndarray:
        # Deliberately illegal: reads the next bar's close.
        future = b["close"].shift(-1).to_numpy()
        return np.where(future > b["close"].to_numpy(), 1, -1).astype("int8")

    leaky = Hypothesis(id="leak", family="test", rationale="", rules="", signal=peeking)
    clean = screen(leaky, bars, frame.atr, bar_minutes=5, keep_trades=True)
    assert clean.gross_points_per_trade > 0, "a future-peeking rule should look wonderful"


# --------------------------------------------------------------------------- fill rules


def _one_bar_frame(rows: list[dict]) -> pd.DataFrame:
    index = pd.date_range("2021-03-01 09:30", periods=len(rows), freq="5min",
                          tz=MARKET_TZ)
    frame = pd.DataFrame(rows, index=index)
    frame["session"] = pd.Timestamp("2021-03-01", tz=MARKET_TZ)
    return frame


def test_the_harness_matches_the_production_engines_fill_rules():
    """Three conventions, shared with ExecutionEngine, that decide whether a result is real."""
    bars = _one_bar_frame([
        {"open": 100, "high": 101, "low": 99, "close": 100},    # 0 signal bar
        {"open": 100, "high": 106, "low": 94, "close": 100},    # 1 spans stop and target
        {"open": 100, "high": 101, "low": 99, "close": 100},
        {"open": 100, "high": 101, "low": 99, "close": 100},
    ])
    atr = np.full(len(bars), 5.0)
    direction = np.array([1, 0, 0, 0], dtype="int8")

    trades = simulate(bars, direction, atr, ExitRule(stop_atr=1.0, target_atr=1.0,
                                                    max_bars=3, no_entry_within_bars=0))
    assert len(trades) == 1
    trade = trades[0]

    # 1. Filled at the NEXT bar's open, never on the signal bar.
    assert trade.entry_index == 1 and trade.entry_price == 100

    # 2. One bar containing both stop and target resolves as the stop.
    assert trade.exit_kind == "stop"
    assert trade.exit_price == 95


def test_a_gap_through_the_stop_fills_at_the_open_not_the_level():
    bars = _one_bar_frame([
        {"open": 100, "high": 101, "low": 99, "close": 100},
        {"open": 100, "high": 101, "low": 99.5, "close": 100},
        {"open": 90, "high": 91, "low": 89, "close": 90},       # gapped far below the stop
    ])
    atr = np.full(len(bars), 5.0)
    direction = np.array([1, 0, 0], dtype="int8")

    trades = simulate(bars, direction, atr, ExitRule(1.0, 3.0, 3, no_entry_within_bars=0))
    assert trades[0].exit_price == 90, "the gap is modelled honestly, not filled at 95"


def test_a_position_never_crosses_a_session_boundary():
    index = pd.date_range("2021-03-01 15:50", periods=4, freq="5min", tz=MARKET_TZ)
    bars = pd.DataFrame(
        {"open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4},
        index=index,
    )
    bars["session"] = [pd.Timestamp("2021-03-01", tz=MARKET_TZ)] * 2 + [
        pd.Timestamp("2021-03-02", tz=MARKET_TZ)
    ] * 2
    atr = np.full(4, 5.0)

    trades = simulate(bars, np.array([1, 0, 0, 0], dtype="int8"), atr,
                      ExitRule(5.0, 5.0, 10, no_entry_within_bars=0))
    assert trades[0].exit_index == 1, "the trade was flattened at its own session's end"
    assert trades[0].exit_kind == "timeout"


def test_only_one_position_is_open_at_a_time():
    bars = _one_bar_frame([{"open": 100, "high": 101, "low": 99, "close": 100}] * 10)
    atr = np.full(len(bars), 5.0)
    always = np.ones(len(bars), dtype="int8")

    trades = simulate(bars, always, atr, ExitRule(5.0, 5.0, 3, no_entry_within_bars=0))
    for earlier, later in zip(trades, trades[1:], strict=False):
        assert later.entry_index > earlier.exit_index


def test_entries_are_suppressed_near_the_session_close(frame):
    """Otherwise the statistics measure the closing bell rather than the hypothesis."""
    rule = ExitRule(1.0, 2.0, 24, no_entry_within_bars=6)
    always = Hypothesis(
        id="always", family="test", rationale="", rules="",
        signal=lambda b: np.ones(len(b), dtype="int8"), exit_rule=rule,
    )
    result = screen(always, frame.bars, frame.atr, bar_minutes=5, keep_trades=True)
    order = np.arange(len(frame.bars))
    from_end = (
        pd.DataFrame({"s": frame.bars["session"].to_numpy(), "o": order})
        .groupby("s")["o"].transform("max").to_numpy() - order
    )
    for trade in result.trade_list:
        assert from_end[trade.entry_index - 1] >= rule.no_entry_within_bars


# --------------------------------------------------------------------------- accounting


def test_costs_are_the_documented_mnq_round_trip():
    # $1.24 commission / $2 per point + 2 x 1 tick x 0.25 = 1.12 index points.
    assert ROUND_TRIP_POINTS == pytest.approx(1.12)


def test_net_is_gross_less_the_full_round_trip(frame):
    always = Hypothesis(
        id="always", family="test", rationale="", rules="",
        signal=lambda b: np.ones(len(b), dtype="int8"),
    )
    result = screen(always, frame.bars, frame.atr, bar_minutes=5)
    assert result.net_points_per_trade == pytest.approx(
        result.gross_points_per_trade - ROUND_TRIP_POINTS
    )
    assert result.commission_usd == pytest.approx(1.24 * result.trades)
    assert result.slippage_usd == pytest.approx(1.0 * result.trades)


def test_stress_slippage_makes_results_strictly_worse(frame):
    always = Hypothesis(
        id="always", family="test", rationale="", rules="",
        signal=lambda b: np.ones(len(b), dtype="int8"),
    )
    normal = screen(always, frame.bars, frame.atr, bar_minutes=5)
    stressed = screen(always, frame.bars, frame.atr, bar_minutes=5, slippage_ticks=2.0)
    assert stressed.net_points_per_trade < normal.net_points_per_trade
    assert stressed.gross_points_per_trade == pytest.approx(normal.gross_points_per_trade)


def test_an_empty_result_is_reported_rather_than_crashing(frame):
    silent = Hypothesis(
        id="silent", family="test", rationale="", rules="",
        signal=lambda b: np.zeros(len(b), dtype="int8"),
    )
    result = screen(silent, frame.bars, frame.atr, bar_minutes=5)
    assert result.trades == 0 and result.t_stat is None


def test_a_repeated_event_counts_once_per_session(frame):
    """Signal de-duplication: an event true for twenty bars is one event."""
    from tradebot.research.hypotheses import _first_per_session

    condition = np.ones(len(frame.bars), dtype=bool)
    first = _first_per_session(frame.bars, condition)
    assert first.sum() == frame.bars["session"].nunique()


# --------------------------------------------------------------------------- splits


def test_splits_are_disjoint_and_locked(frame):
    dev = split_frame(frame, "dev")
    validation = split_frame(frame, "validation")
    holdout = split_frame(frame, "holdout")
    assert len(dev) + len(validation) + len(holdout) == len(frame)
    if len(dev):
        assert pd.Timestamp(dev.bars["session"].max()) < pd.Timestamp(
            "2023-01-01", tz=MARKET_TZ
        )


def test_asking_for_an_unknown_split_is_an_error(frame):
    with pytest.raises(ValueError, match="unknown split"):
        split_frame(frame, "the_good_bit")


# --------------------------------------------------------------------------- registry


def test_every_pre_registered_hypothesis_states_its_economics():
    assert len(PRE_REGISTERED) == 16
    for hypothesis in PRE_REGISTERED:
        assert hypothesis.rationale.strip(), f"{hypothesis.id} has no economic rationale"
        assert hypothesis.rules.strip(), f"{hypothesis.id} has no stated rules"
        assert hypothesis.family.strip()
        # Rationale must describe market behaviour, not indicator arithmetic.
        assert len(hypothesis.rationale.split()) >= 8


def test_hypothesis_ids_are_unique():
    ids = [h.id for h in PRE_REGISTERED]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("hypothesis", PRE_REGISTERED, ids=lambda h: h.id)
def test_every_hypothesis_produces_a_valid_signal_array(frame, hypothesis):
    signal = hypothesis.signal(frame.bars)
    assert len(signal) == len(frame.bars)
    assert set(np.unique(signal)) <= {-1, 0, 1}
