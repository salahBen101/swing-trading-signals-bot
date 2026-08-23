"""M2: bar integrity, the parquet store, locked splits, and the closed-bars-only feed.

The feed tests are the load-bearing ones. If a partial bar can reach a strategy, every
backtest number in the project is fiction, and no amount of care elsewhere recovers it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.timeframes import timeframe_seconds
from tradebot.core.models import Bar
from tradebot.data.feed import BarAggregator, LiveFeed, MarketDataFeed, ReplayFeed
from tradebot.data.schema import DataIntegrityError, range_pct_threshold, validate_bars
from tradebot.data.splits import (
    HOLDOUT_START,
    VALIDATION_START,
    Split,
    dev,
    holdout,
    slice_split,
    summarize,
    validation,
)
from tradebot.data.store import BarStore, normalize_bars, resample


REAL_1M = Path("data_cache/nq_1m.parquet")


def make_frame(
    n: int = 300,
    start: str = "2024-01-02 09:30",
    freq: str = "1min",
    *,
    vol: float = 6.0,
) -> pd.DataFrame:
    """A synthetic NQ-like frame.

    `vol` is set so the median bar range lands near the real archive's ~13 points for a
    1-minute NQ bar. Fixtures that are an order of magnitude quieter than the real thing
    make the wick-ratio guard fire on ranges that would be unremarkable in production,
    which turns a genuine test into a false alarm.
    """
    idx = pd.date_range(start, periods=n, freq=freq, tz=MARKET_TZ, name="timestamp")
    rng = np.random.default_rng(0)
    close = 18000 + np.cumsum(rng.normal(0, vol / 4, n))
    high = close + rng.uniform(vol / 4, vol, n)
    low = close - rng.uniform(vol / 4, vol, n)
    open_ = np.clip(close + rng.normal(0, vol / 3, n), low, high)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(50, 500, n).astype(float)},
        index=idx,
    )


# ---------------------------------------------------------------------------- integrity


def test_a_clean_frame_validates():
    validate_bars(make_frame())


def test_inconsistent_high_low_is_rejected():
    df = make_frame()
    df.iloc[50, df.columns.get_loc("high")] = df["low"].iloc[50] - 5
    with pytest.raises(DataIntegrityError, match="inconsistent OHLC"):
        validate_bars(df)


def test_duplicate_timestamps_are_rejected():
    df = make_frame()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    with pytest.raises(DataIntegrityError, match="duplicate timestamps"):
        validate_bars(df)


def test_unsorted_timestamps_are_rejected():
    df = make_frame()
    with pytest.raises(DataIntegrityError, match="monotonically increasing"):
        validate_bars(df.iloc[::-1])


def test_naive_index_is_rejected_rather_than_assumed_to_be_eastern():
    df = make_frame()
    df.index = df.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="timezone-naive"):
        validate_bars(df)


def test_a_stray_tick_an_order_of_magnitude_off_is_caught_at_every_timeframe():
    # 25000 printed against an 18000 market is a 28% range. It must be caught whether the
    # bar covers 15 seconds or 15 minutes.
    for secs in (15, 60, 300, 900):
        df = make_frame()
        df.iloc[100, df.columns.get_loc("high")] = 25000.0
        with pytest.raises(DataIntegrityError, match="almost certainly bad ticks"):
            validate_bars(df, timeframe_seconds=secs)


def test_the_bad_tick_threshold_scales_with_the_square_root_of_bar_duration():
    # A bar's range grows roughly as sqrt(time), so a flat percentage is right for one
    # timeframe and wrong for all the others.
    assert range_pct_threshold(15) == pytest.approx(0.02)
    assert range_pct_threshold(60) == pytest.approx(0.04)
    assert range_pct_threshold(300) == pytest.approx(0.0894, abs=1e-4)
    assert range_pct_threshold(900) == pytest.approx(0.1549, abs=1e-4)


@pytest.mark.parametrize(
    "secs, range_pct",
    [
        (60, 0.0335),   # 2025-04-09 tariff-pause rally, largest 1m range in 9 years
        (60, 0.0308),   # 2020-03-16 COVID limit-down
        (300, 0.0484),  # 2025-04-07, largest 5m range in 9 years
        (900, 0.0742),  # 2025-04-09, largest 15m range in 9 years
    ],
)
def test_the_largest_real_macro_bars_in_the_archive_are_not_flagged(secs, range_pct):
    # Baseline volatility scales with sqrt(time), as it does in the real archive, so the
    # wick-ratio guard sees a realistic ratio rather than an artefact of a quiet fixture.
    df = make_frame(vol=6.0 * (secs / 60) ** 0.5)
    i, cols = 150, df.columns
    low = 18000.0
    high = low * (1 + range_pct)
    df.iloc[i, cols.get_loc("low")] = low
    df.iloc[i, cols.get_loc("high")] = high
    df.iloc[i, cols.get_loc("open")] = low + 1
    df.iloc[i, cols.get_loc("close")] = high - 1
    validate_bars(df, timeframe_seconds=secs)


@pytest.mark.skipif(not REAL_1M.exists(), reason="9-year NQ archive not present")
@pytest.mark.parametrize("timeframe, rule", [("1min", None), ("5min", "5min"), ("15min", "15min")])
def test_the_real_nine_year_archive_passes_integrity_at_every_timeframe(timeframe, rule):
    """The guard has to accept real data, including COVID and the 2025 tariff sessions.

    Skipped when the archive is absent (it is gitignored and large), so a clean checkout
    still runs a green suite.
    """
    df = normalize_bars(pd.read_parquet(REAL_1M))
    if rule:
        df = resample(df, rule)
    validate_bars(df, timeframe_seconds=timeframe_seconds(timeframe))


def test_negative_volume_and_missing_values_are_rejected():
    df = make_frame()
    df.iloc[5, df.columns.get_loc("volume")] = -1
    with pytest.raises(DataIntegrityError, match="negative volume"):
        validate_bars(df)

    df = make_frame()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    with pytest.raises(DataIntegrityError, match="missing values"):
        validate_bars(df)


def test_a_real_fomc_sized_bar_is_not_flagged():
    # 2024-09-18: 166 points in 15 seconds, ~0.84% of price. Legitimate, must survive.
    df = make_frame()
    row = df.columns
    i = 150
    df.iloc[i, row.get_loc("low")] = 18000.0
    df.iloc[i, row.get_loc("high")] = 18166.0
    df.iloc[i, row.get_loc("open")] = 18010.0
    df.iloc[i, row.get_loc("close")] = 18150.0
    validate_bars(df, timeframe_seconds=15)


# ---------------------------------------------------------------------------- store


def test_import_then_load_round_trips(tmp_path):
    store = BarStore(tmp_path)
    df = make_frame(120)
    info = store.import_bars("MNQ", "1min", df)

    assert info.rows == 120
    assert info.instrument == "MNQ"
    loaded = store.load("MNQ", "1min")
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_importing_the_same_data_twice_changes_nothing(tmp_path):
    store = BarStore(tmp_path)
    df = make_frame(120)
    first = store.import_bars("MNQ", "1min", df)
    second = store.import_bars("MNQ", "1min", df)
    assert first.rows == second.rows
    assert first.content_hash == second.content_hash


def test_overlapping_import_merges_and_incoming_wins(tmp_path):
    store = BarStore(tmp_path)
    df = make_frame(100)
    store.import_bars("MNQ", "1min", df.iloc[:60])

    corrected = df.iloc[40:].copy()
    corrected.iloc[0, corrected.columns.get_loc("close")] = corrected["high"].iloc[0]
    info = store.import_bars("MNQ", "1min", corrected)

    assert info.rows == 100
    loaded = store.load("MNQ", "1min")
    assert loaded["close"].iloc[40] == corrected["close"].iloc[0]


def test_the_content_hash_moves_when_a_price_moves(tmp_path):
    store = BarStore(tmp_path)
    df = make_frame(80)
    a = store.import_bars("MNQ", "1min", df)
    df2 = df.copy()
    df2.iloc[10, df2.columns.get_loc("close")] = df2["high"].iloc[10]
    b = store.import_bars("MNQ", "1min", df2, replace=True)
    assert a.content_hash != b.content_hash


def test_the_manifest_records_what_is_in_the_store(tmp_path):
    store = BarStore(tmp_path)
    store.import_bars("MNQ", "1min", make_frame(50))
    store.import_bars("MES", "5min", make_frame(50, freq="5min"))
    names = {d.instrument for d in store.datasets()}
    assert names == {"MNQ", "MES"}
    assert store.info("MNQ", "1min").rows == 50
    assert "MNQ" in store.info("MNQ", "1min").summary()


def test_loading_a_missing_dataset_says_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="import-data"):
        BarStore(tmp_path).load("MNQ", "1min")


def test_normalize_accepts_a_timestamp_column_and_odd_capitalisation():
    df = make_frame(20).reset_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    out = normalize_bars(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is not None


def test_resample_drops_empty_buckets_rather_than_inventing_flat_bars():
    df = make_frame(60)
    gapped = pd.concat([df.iloc[:10], df.iloc[40:]])
    out = resample(gapped, "5min")
    assert len(out) < 12
    assert (out["volume"] > 0).all()


def test_resample_aggregates_ohlc_correctly():
    df = make_frame(10)
    out = resample(df, "5min")
    assert len(out) == 2
    assert out["open"].iloc[0] == df["open"].iloc[0]
    assert out["close"].iloc[0] == df["close"].iloc[4]
    assert out["high"].iloc[0] == df["high"].iloc[:5].max()
    assert out["volume"].iloc[0] == df["volume"].iloc[:5].sum()


# ---------------------------------------------------------------------------- splits


def test_splits_are_disjoint_and_cover_the_sample():
    idx = pd.date_range("2018-01-01", "2026-01-01", freq="7D", tz=MARKET_TZ)
    df = pd.DataFrame({"close": 1.0}, index=idx)
    parts = [dev(df), validation(df), holdout(df)]
    assert sum(len(p) for p in parts) == len(df)
    combined = pd.concat(parts).index
    assert combined.is_unique
    assert set(combined) == set(df.index)


def test_split_boundaries_are_the_locked_constants():
    idx = pd.DatetimeIndex(
        [VALIDATION_START - pd.Timedelta(seconds=1), VALIDATION_START,
         HOLDOUT_START - pd.Timedelta(seconds=1), HOLDOUT_START]
    )
    df = pd.DataFrame({"close": 1.0}, index=idx)
    assert len(dev(df)) == 1
    assert len(validation(df)) == 2
    assert len(holdout(df)) == 1


def test_train_is_dev_plus_validation_and_excludes_holdout():
    idx = pd.date_range("2018-01-01", "2026-01-01", freq="30D", tz=MARKET_TZ)
    df = pd.DataFrame({"close": 1.0}, index=idx)
    assert len(slice_split(df, Split.TRAIN)) == len(dev(df)) + len(validation(df))
    assert slice_split(df, Split.TRAIN).index.max() < HOLDOUT_START


def test_summarize_reports_each_split():
    idx = pd.date_range("2018-01-01", "2026-01-01", freq="30D", tz=MARKET_TZ)
    df = pd.DataFrame({"close": 1.0}, index=idx)
    rows = summarize(df)
    assert [r.name for r in rows] == ["DEV", "VALIDATION", "HOLDOUT"]
    assert all(r.rows > 0 for r in rows)


# ---------------------------------------------------------------------------- feeds


def test_replay_feed_yields_every_bar_in_order():
    df = make_frame(50)
    feed = ReplayFeed(df, "MNQ", 60)
    bars = list(feed)
    assert len(bars) == 50
    assert [b.timestamp for b in bars] == list(df.index)
    assert isinstance(feed, MarketDataFeed)


def test_replay_feed_pins_a_simulated_clock_to_each_bar_close():
    df = make_frame(3)
    clock = SimulatedClock(df.index[0].to_pydatetime())
    feed = ReplayFeed(df, "MNQ", 60, clock=clock)
    seen = []
    for bar in feed:
        seen.append((bar.timestamp, clock.now()))
    for bar_ts, clock_now in seen:
        # The bar's timestamp is its open; it is only knowable one interval later.
        assert clock_now == bar_ts + timedelta(seconds=60)


def test_an_aggregator_never_returns_a_bar_that_is_still_open():
    agg = BarAggregator(60)
    base = datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ)
    for sec in (0, 15, 30, 45, 59):
        assert agg.update(base + timedelta(seconds=sec), 18000 + sec) is None
    # The partial exists, but only through the diagnostic accessor.
    partial = agg.current_partial()
    assert partial is not None and partial.close == 18059


def test_the_bar_appears_only_once_the_next_interval_starts():
    agg = BarAggregator(60)
    base = datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ)
    agg.update(base, 100.0, 5)
    agg.update(base + timedelta(seconds=30), 104.0, 5)
    agg.update(base + timedelta(seconds=45), 98.0, 5)

    closed = agg.update(base + timedelta(seconds=60), 101.0, 1)
    assert closed is not None
    assert closed.timestamp == base
    assert (closed.open, closed.high, closed.low, closed.close) == (100.0, 104.0, 98.0, 98.0)
    assert closed.volume == 15


def test_flush_closes_a_quiet_bar_only_after_its_interval_elapses():
    agg = BarAggregator(60)
    base = datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ)
    agg.update(base, 100.0, 1)
    assert agg.flush(base + timedelta(seconds=59)) is None
    bar = agg.flush(base + timedelta(seconds=60))
    assert bar is not None and bar.timestamp == base
    assert agg.flush(base + timedelta(seconds=120)) is None  # nothing left to close


def test_live_feed_emits_only_completed_bars():
    base = datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ)
    updates = [(base + timedelta(seconds=s), 18000 + s, 1.0) for s in range(0, 150, 10)]
    clock = SimulatedClock(base + timedelta(seconds=200))
    feed = LiveFeed(updates, "MNQ", 60, clock=clock)
    bars = list(feed)
    assert [b.timestamp for b in bars] == [
        base, base + timedelta(seconds=60), base + timedelta(seconds=120)
    ]


def test_feed_health_reports_not_started_before_the_first_bar():
    feed = ReplayFeed(make_frame(3), "MNQ", 60, clock=SimulatedClock(
        datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ)))
    h = feed.health()
    assert not h.is_stale and not h.is_hard_stale
    assert h.last_bar_time is None


def test_feed_health_trips_at_the_configured_multiples():
    # The replay feed pins the simulated clock to the bar's close, so "arrival" is
    # 09:31 and the gaps below are measured from there.
    df = make_frame(1, start="2026-03-10 09:30")
    close = datetime(2026, 3, 10, 9, 31, tzinfo=MARKET_TZ)
    clock = SimulatedClock(close)
    feed = ReplayFeed(df, "MNQ", 60, clock=clock,
                      stale_multiple=3.0, hard_stale_multiple=10.0)
    list(feed)

    assert not feed.health(close + timedelta(seconds=119)).is_stale
    assert feed.health(close + timedelta(seconds=181)).is_stale
    assert not feed.health(close + timedelta(seconds=181)).is_hard_stale
    assert feed.health(close + timedelta(seconds=601)).is_hard_stale


def test_bar_objects_reaching_the_feed_are_validated():
    with pytest.raises(ValueError):
        Bar(datetime(2026, 3, 10, 9, 30, tzinfo=MARKET_TZ),
            open=100, high=99, low=101, close=100, volume=1)
