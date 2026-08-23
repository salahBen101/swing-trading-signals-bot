"""M3: the causal indicator library.

The centrepiece is `test_every_indicator_satisfies_prefix_equality`. If computing an
indicator on the first k bars does not match the first k values of computing it on the
whole series, that indicator has read the future, and every backtest number downstream of
it is fiction. It is cheap to assert and it catches the entire class of mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.features import indicators as ind
from tradebot.features import levels as lv
from tradebot.features.pipeline import FeatureSpec, build_features


def make_bars(n: int = 800, sessions: int = 3) -> pd.DataFrame:
    """Multi-session RTH-shaped bars, so session-anchored features are exercised."""
    per = n // sessions
    frames = []
    rng = np.random.default_rng(11)
    price = 18000.0
    for s in range(sessions):
        day = pd.Timestamp("2024-01-02", tz=MARKET_TZ) + pd.Timedelta(days=s)
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=per,
                            freq="1min", tz=MARKET_TZ)
        close = price + np.cumsum(rng.normal(0, 1.5, per))
        high = close + rng.uniform(0.5, 6.0, per)
        low = close - rng.uniform(0.5, 6.0, per)
        open_ = np.clip(close + rng.normal(0, 2.0, per), low, high)
        frames.append(pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": rng.integers(100, 900, per).astype(float)},
            index=idx,
        ))
        price = float(close[-1])
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out


BARS = make_bars()
O, H, L, C, V = (BARS[k] for k in ("open", "high", "low", "close", "volume"))


# --------------------------------------------------------------------- prefix equality

# (name, callable taking a bars frame and returning a Series or DataFrame)
CAUSAL_CASES = {
    "sma": lambda b: ind.sma(b["close"], 20),
    "ema": lambda b: ind.ema(b["close"], 20),
    "wilder_ema": lambda b: ind.wilder_ema(b["close"], 14),
    "true_range": lambda b: ind.true_range(b["high"], b["low"], b["close"]),
    "atr": lambda b: ind.atr(b["high"], b["low"], b["close"], 14),
    "rsi": lambda b: ind.rsi(b["close"], 14),
    "bollinger": lambda b: pd.concat(ind.bollinger(b["close"], 20, 2.0), axis=1),
    "donchian": lambda b: pd.concat(ind.donchian(b["high"], b["low"], 20), axis=1),
    "adx": lambda b: pd.concat(ind.adx(b["high"], b["low"], b["close"], 14), axis=1),
    "efficiency_ratio": lambda b: ind.efficiency_ratio(b["close"], 40),
    "realized_volatility": lambda b: ind.realized_volatility(b["close"], 50),
    "rolling_percentile": lambda b: ind.rolling_percentile(b["close"], 100, 20),
    "autocorrelation": lambda b: ind.autocorrelation(b["close"], 20, 1),
    "volume_ratio": lambda b: ind.volume_ratio(b["volume"], 50),
    "session_vwap": lambda b: ind.session_vwap(b["high"], b["low"], b["close"], b["volume"]),
    "session_vwap_bands": lambda b: pd.concat(
        ind.session_vwap_bands(b["high"], b["low"], b["close"], b["volume"], 1.0), axis=1),
    "session_high": lambda b: ind.session_cumulative(b["high"], "max"),
    "session_low": lambda b: ind.session_cumulative(b["low"], "min"),
    "bars_since_session_start": lambda b: ind.bars_since_session_start(b.index),
    "opening_range": lambda b: pd.concat(ind.opening_range(b["high"], b["low"], 30), axis=1),
    "pivot_high": lambda b: lv.pivot_high(b["high"], 3, 3),
    "pivot_low": lambda b: lv.pivot_low(b["low"], 3, 3),
    "last_pivot_high": lambda b: lv.last_pivot_high(b["high"], 3, 3),
    "recent_pivot_levels": lambda b: lv.recent_pivot_levels(b["high"], 3, 3, "high", 4),
    "bars_since": lambda b: lv.bars_since(b["close"] > b["open"]),
}


@pytest.mark.parametrize("name", sorted(CAUSAL_CASES))
@pytest.mark.parametrize("k", [120, 401, 799])
def test_every_indicator_satisfies_prefix_equality(name, k):
    """Computing on bars[:k] must equal the first k values of computing on everything."""
    fn = CAUSAL_CASES[name]
    full = fn(BARS)
    prefix = fn(BARS.iloc[:k])

    full_head = full.iloc[:k]
    if isinstance(full, pd.DataFrame):
        pd.testing.assert_frame_equal(prefix, full_head, check_names=False, rtol=1e-12)
    else:
        pd.testing.assert_series_equal(prefix, full_head, check_names=False, rtol=1e-12)


def test_the_prefix_equality_harness_actually_catches_a_leak():
    """A guard that cannot fail proves nothing. This is a deliberately leaky indicator."""
    def leaky(b):
        return b["close"].rolling(5, center=True).mean()

    full = leaky(BARS)
    prefix = leaky(BARS.iloc[:120])
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(prefix, full.iloc[:120], check_names=False)


def test_no_indicator_source_uses_a_negative_shift_or_a_centred_window():
    """A static check on the library itself, to catch a future edit rather than a bug.

    Parses the AST rather than grepping text, so the prose in a docstring that *names*
    these hazards does not trip the check that forbids them.
    """
    import ast
    import inspect

    for module in (ind, lv):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "center" and getattr(kw.value, "value", False) is True:
                    raise AssertionError(f"{module.__name__} uses a centred window")
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "shift":
                for arg in node.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        raise AssertionError(f"{module.__name__} shifts backwards in time")
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                        raise AssertionError(f"{module.__name__} shifts backwards in time")


# --------------------------------------------------------------------- correctness

def test_sma_and_ema_match_hand_computed_values():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    # EMA with adjust=False, span=3 -> alpha 0.5, seeded on the third value.
    assert ind.ema(s, 3).iloc[2] == pytest.approx(2.25)


def test_wilder_smoothing_is_not_an_ordinary_ema():
    s = pd.Series(np.arange(1, 60, dtype=float))
    assert not np.isclose(ind.wilder_ema(s, 14).iloc[-1], ind.ema(s, 14).iloc[-1])


def test_rsi_saturates_at_100_on_an_unbroken_advance():
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(s, 14).iloc[-1] == 100.0


def test_rsi_is_zero_on_an_unbroken_decline():
    s = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ind.rsi(s, 14).iloc[-1] == pytest.approx(0.0)


def test_true_range_accounts_for_gaps():
    high = pd.Series([10.0, 20.0])
    low = pd.Series([9.0, 19.0])
    close = pd.Series([9.5, 19.5])
    # Second bar gapped up: TR is high - prev_close = 20 - 9.5, not the 1-point bar range.
    assert ind.true_range(high, low, close).iloc[1] == pytest.approx(10.5)


def test_donchian_excludes_the_current_bar():
    high = pd.Series([1.0, 2, 3, 4, 100])
    low = pd.Series([1.0, 2, 3, 4, 100])
    _, upper = ind.donchian(high, low, 4)
    # The last bar's own 100 must not be inside the channel it is supposed to break.
    assert upper.iloc[4] == 4.0


def test_efficiency_ratio_is_one_on_a_straight_line_and_low_on_chop():
    straight = pd.Series(np.arange(100, dtype=float))
    assert ind.efficiency_ratio(straight, 20).iloc[-1] == pytest.approx(1.0)

    chop = pd.Series(np.tile([100.0, 101.0], 50))
    assert ind.efficiency_ratio(chop, 20).iloc[-1] < 0.15


def test_volume_ratio_excludes_the_surging_bar_from_its_own_baseline():
    v = pd.Series([100.0] * 50 + [1000.0])
    assert ind.volume_ratio(v, 50).iloc[-1] == pytest.approx(10.0)


# --------------------------------------------------------------------- session anchoring

def test_vwap_resets_at_every_session_boundary():
    bars = make_bars(240, sessions=3)
    vwap = ind.session_vwap(bars["high"], bars["low"], bars["close"], bars["volume"])
    session = bars.index.normalize()
    for _, group in vwap.groupby(session):
        typical_first = (bars.loc[group.index[0], ["high", "low", "close"]]).mean()
        # The session's first VWAP value is just that bar's typical price.
        assert group.iloc[0] == pytest.approx(typical_first)


def test_session_high_is_a_running_maximum_not_the_whole_session():
    bars = make_bars(240, sessions=2)
    sh = ind.session_cumulative(bars["high"], "max")
    assert (sh >= bars["high"]).all()
    assert sh.iloc[0] == bars["high"].iloc[0]
    assert sh.is_monotonic_increasing is False  # it resets between sessions


def test_prior_session_values_never_include_todays_bars():
    bars = make_bars(240, sessions=3)
    prior = ind.prior_session_value(bars["close"], "last")
    sessions = sorted(set(bars.index.normalize()))
    first_day = bars.index.normalize() == sessions[0]
    assert prior[first_day].isna().all()  # nothing before the first session

    second_day = bars.index.normalize() == sessions[1]
    expected = bars.loc[bars.index.normalize() == sessions[0], "close"].iloc[-1]
    assert (prior[second_day] == expected).all()


def test_the_opening_range_is_unreadable_until_its_window_closes():
    bars = make_bars(390, sessions=1)
    or_low, or_high, complete = ind.opening_range(bars["high"], bars["low"], 30)

    assert or_high.iloc[:30].isna().all(), "the range must not be readable while forming"
    assert not complete.iloc[:30].any()
    assert complete.iloc[30:].all()

    # Once frozen it is the first 30 minutes' extremes, and it never moves again.
    assert or_high.iloc[30] == pytest.approx(bars["high"].iloc[:30].max())
    assert or_low.iloc[30] == pytest.approx(bars["low"].iloc[:30].min())
    assert or_high.iloc[30:].nunique() == 1


def test_bars_into_session_restarts_each_day():
    bars = make_bars(240, sessions=3)
    n = ind.bars_since_session_start(bars.index)
    assert n.iloc[0] == 0
    assert n.groupby(bars.index.normalize()).min().eq(0).all()


# --------------------------------------------------------------------- pivots and levels

def test_a_pivot_is_reported_only_once_the_bars_after_it_exist():
    # A clean peak at index 5, with 3 bars either side.
    high = pd.Series([1.0, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1])
    piv = lv.pivot_high(high, left=3, right=3)

    assert piv.iloc[:8].isna().all(), "the peak must not be visible before its right side"
    assert piv.iloc[8] == 10.0, "confirmed exactly `right` bars after the peak"


def test_last_pivot_carries_the_confirmed_level_forward():
    high = pd.Series([1.0, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1])
    last = lv.last_pivot_high(high, 3, 3)
    assert last.iloc[8:].eq(10.0).all()


def test_pivot_lows_mirror_pivot_highs():
    low = pd.Series([10.0, 9, 8, 7, 6, 1, 6, 7, 8, 9, 10])
    piv = lv.pivot_low(low, 3, 3)
    assert piv.iloc[:8].isna().all()
    assert piv.iloc[8] == 1.0


def test_recent_levels_are_ordered_newest_first():
    # Three peaks in time order: 9, then 8, then 7. Newest-first means level_0 is 7.
    high = pd.Series([1, 2, 3, 9, 3, 2, 1, 2, 3, 8, 3, 2, 1, 2, 3, 7, 3, 2, 1.0])
    got = lv.recent_pivot_levels(high, 2, 2, "high", count=3)
    assert got.iloc[-1].tolist() == [7.0, 8.0, 9.0]


def test_recent_levels_reveal_an_older_pivot_only_once_it_is_superseded():
    high = pd.Series([1, 2, 3, 9, 3, 2, 1, 2, 3, 8, 3, 2, 1.0])
    got = lv.recent_pivot_levels(high, 2, 2, "high", count=2)
    # Before the second peak is confirmed there is only one known level.
    assert got["level_1"].iloc[:11].isna().all()
    assert got["level_0"].iloc[-1] == 8.0
    assert got["level_1"].iloc[-1] == 9.0


def test_nearest_levels_pick_the_closest_on_each_side():
    price = pd.Series([100.0, 100.0])
    levels = pd.DataFrame({"level_0": [90.0, 110.0], "level_1": [95.0, 105.0],
                           "level_2": [120.0, 80.0]})
    assert lv.nearest_level_above(price, levels).tolist() == [120.0, 105.0]
    assert lv.nearest_level_below(price, levels).tolist() == [95.0, 80.0]


def test_a_break_needs_a_close_through_the_level_not_a_wick():
    close = pd.Series([99.0, 101.0, 100.5])
    level = pd.Series([100.0, 100.0, 100.0])
    assert lv.broke_above(close, level).tolist() == [False, True, False]
    assert lv.broke_below(pd.Series([101.0, 99.0]), pd.Series([100.0, 100.0])).tolist() == [
        False, True
    ]


def test_bars_since_counts_back_to_the_last_occurrence():
    cond = pd.Series([False, True, False, False, True, False])
    assert lv.bars_since(cond).tolist()[1:] == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert np.isnan(lv.bars_since(cond).iloc[0])


# --------------------------------------------------------------------- pipeline

def test_the_pipeline_builds_every_declared_feature():
    ff = build_features(BARS)
    for col in ("atr", "rsi", "vwap", "adx", "or_high", "support", "resistance",
                "volume_ratio", "efficiency_ratio", "donchian_upper"):
        assert col in ff.frame.columns
    assert len(ff.frame) == len(BARS)


def test_dense_features_are_all_defined_from_the_warmup_boundary_onward():
    ff = build_features(BARS)
    dense = [c for c in ff.frame.columns
             if c not in {"support", "resistance", "support_distance_atr",
                          "resistance_distance_atr", "swing_high", "swing_low",
                          "or_low", "or_high", "or_width", "or_width_percentile",
                          "prior_close", "prior_high", "prior_low"}]
    tail = ff.frame[dense].iloc[ff.warmup_bars:]
    assert tail.notna().all().all()
    assert 0 < ff.warmup_bars < len(BARS)


def test_the_pipeline_is_causal_end_to_end():
    """The whole assembled frame, not just its parts."""
    k = 500
    full = build_features(BARS).frame
    prefix = build_features(BARS.iloc[:k]).frame
    pd.testing.assert_frame_equal(prefix, full.iloc[:k], rtol=1e-12)


def test_feature_spec_parameters_actually_change_the_output():
    a = build_features(BARS, FeatureSpec(atr_period=14)).frame["atr"]
    b = build_features(BARS, FeatureSpec(atr_period=50)).frame["atr"]
    assert not a.equals(b)
