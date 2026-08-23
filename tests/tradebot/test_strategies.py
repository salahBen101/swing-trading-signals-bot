"""M4: the six strategy families.

Two layers:

* **Rule tests** drive `_entry_signal` with a hand-built feature row, so each documented
  condition is asserted in isolation — it fires on the setup and stays silent on a
  near-miss that differs in exactly one condition.
* **An integration sweep** runs every registered strategy over multi-session bars and
  asserts the invariants that must hold no matter what the rules say: stops on the
  protective side, no entry outside declared hours, and the per-session cap respected.
"""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.core.types import RejectReason, Side
from tradebot.features.pipeline import build_features
from tradebot.instruments.registry import get_instrument
from tradebot.strategy.base import Strategy, StrategyContext, StrategySpec, TradingHours
from tradebot.strategy.registry import (
    UnknownStrategy,
    build_strategy,
    known_strategies,
    register_strategy,
)

from .conftest import make_context

MNQ = get_instrument("MNQ")


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 4, 1, hour, minute, tzinfo=MARKET_TZ)


# ============================================================ registry


def test_all_six_families_are_registered():
    assert known_strategies() == [
        "orb_breakout", "sr_breakout_retest", "sr_rejection",
        "trend_pullback", "vwap_continuation", "vwap_reversion",
    ]


def test_building_an_unknown_strategy_lists_the_known_ones():
    with pytest.raises(UnknownStrategy, match="orb_breakout"):
        build_strategy("moon_phase")


def test_registering_a_duplicate_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("orb_breakout", lambda **_: None)


@pytest.mark.parametrize("name", known_strategies())
def test_every_strategy_declares_a_complete_rule_set(name):
    """PROJECT_SPEC section 5: the spec is the contract, and it must be filled in."""
    spec = build_strategy(name).spec
    assert spec.entry_conditions, f"{name} declares no entry conditions"
    assert spec.stop_loss, f"{name} declares no stop rule"
    assert spec.profit_target, f"{name} declares no target rule"
    assert spec.max_trades_per_session == 1
    assert spec.trading_hours.earliest_entry < spec.trading_hours.latest_entry
    assert spec.trading_hours.latest_entry <= spec.trading_hours.force_flat_at
    assert spec.to_dict()["name"] == name
    assert name in spec.summary()


@pytest.mark.parametrize("name", known_strategies())
def test_no_strategy_uses_discretionary_language(name):
    """PROJECT_SPEC section 5 bans terms like 'looks bullish' from the rule set."""
    spec = build_strategy(name).spec
    text = " ".join(
        [*spec.entry_conditions, *spec.invalidation_conditions, *spec.filters,
         spec.stop_loss, spec.profit_target]
    ).lower()
    for word in ("looks", "seems", "appears", "feels", "probably", "strong-ish", "maybe"):
        assert word not in text, f"{name} uses discretionary language: {word!r}"


def test_parameters_reach_the_spec_and_change_behaviour():
    a = build_strategy("orb_breakout", {"or_minutes": 15, "target_r": 3.0})
    assert a.spec.params["or_minutes"] == 15
    assert a.spec.features.opening_range_minutes == 15
    assert "3.0R" in a.spec.profit_target
    assert a.spec.trading_hours.earliest_entry == time(9, 45)


# ============================================================ shared base behaviour


def test_a_strategy_holds_no_broker_handle():
    strategy = build_strategy("orb_breakout")
    forbidden = {"broker", "client", "session", "execution", "send", "submit", "place_order"}
    assert forbidden.isdisjoint(vars(strategy))
    assert forbidden.isdisjoint(dir(strategy))


def test_entries_outside_declared_hours_are_refused_and_recorded():
    strategy = build_strategy("orb_breakout")
    ctx = make_context({"atr": 5.0}, timestamp=at(9, 45))  # before the 10:00 earliest entry
    assert strategy.on_bar(ctx) is None
    assert ctx.rejections[0].reason is RejectReason.OUTSIDE_TRADING_HOURS


def test_the_per_session_trade_cap_is_enforced_by_the_base_class():
    strategy = build_strategy("orb_breakout", {"max_trades_per_session": 1})
    ctx = make_context({"atr": 5.0}, timestamp=at(11, 0))
    strategy.on_session_start(ctx.session_date)
    strategy.on_trade_opened(None)  # the session's one allowed trade

    assert strategy.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.MAX_TRADES_PER_SESSION


def test_the_trade_counter_resets_on_a_new_session():
    strategy = build_strategy("orb_breakout", {"max_trades_per_session": 1})
    strategy.on_session_start(at(11, 0).date())
    strategy.on_trade_opened(None)
    assert strategy.trades_this_session == 1

    strategy.on_session_start(datetime(2024, 4, 2, 11, 0, tzinfo=MARKET_TZ).date())
    assert strategy.trades_this_session == 0


def test_a_stop_that_lands_on_the_wrong_side_is_refused_not_shipped():
    strategy = build_strategy("orb_breakout")
    ctx = make_context({"atr": 5.0, "close": 18000.0}, timestamp=at(11, 0))
    intent = strategy._make_intent(ctx, Side.BUY, stop_price=18010.0, conditions=("x",))
    assert intent is None
    assert ctx.rejections[-1].reason is RejectReason.INVALID_ORDER


def test_protective_levels_are_rounded_away_from_the_entry_never_toward_it():
    strategy = build_strategy("orb_breakout")
    ctx = make_context({"atr": 5.0, "close": 18000.0}, timestamp=at(11, 0))
    intent = strategy._make_intent(ctx, Side.BUY, stop_price=17994.1873, conditions=("x",))
    assert intent.stop_price == 17994.0  # rounded down: further from entry

    ctx2 = make_context({"atr": 5.0, "close": 18000.0}, timestamp=at(11, 0))
    short = strategy._make_intent(ctx2, Side.SELL, stop_price=18005.1873, conditions=("x",))
    assert short.stop_price == 18005.25  # rounded up: further from entry


# ============================================================ opening range breakout


def _orb_features(**over):
    base = {
        "atr": 5.0, "close": 18020.0, "or_high": 18010.0, "or_low": 17990.0,
        "or_width_percentile": 0.5, "volume_ratio": 2.0, "rsi": 60.0, "adx": 20.0,
    }
    return {**base, **over}


def test_orb_fires_on_a_fresh_break_above_the_range():
    s = build_strategy("orb_breakout")
    ctx = make_context(_orb_features(), previous_bar={"close": 18005.0}, timestamp=at(11, 0))
    intent = s.on_bar(ctx)

    assert intent is not None and intent.side is Side.BUY
    assert intent.stop_price == 18015.0  # close 18020 - 1.0 x ATR 5
    assert intent.target_price == 18030.0  # 2R
    assert "close_above_or_high" in intent.conditions


def test_orb_fires_short_on_a_break_below_the_range():
    s = build_strategy("orb_breakout")
    ctx = make_context(_orb_features(close=17980.0), previous_bar={"close": 17995.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.SELL
    assert intent.stop_price == 17985.0
    assert intent.target_price == 17970.0


def test_orb_ignores_a_break_that_was_already_outside_last_bar():
    s = build_strategy("orb_breakout")
    ctx = make_context(_orb_features(), previous_bar={"close": 18015.0}, timestamp=at(11, 0))
    assert s.on_bar(ctx) is None, "the break must be fresh, not a continuation"


def test_orb_waits_for_the_opening_range_to_close():
    s = build_strategy("orb_breakout")
    ctx = make_context(_orb_features(or_high=float("nan"), or_low=float("nan")),
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.WARMUP_INCOMPLETE


def test_orb_requires_confirming_volume():
    s = build_strategy("orb_breakout")
    ctx = make_context(_orb_features(volume_ratio=0.8), previous_bar={"close": 18005.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_VOLUME


def test_orb_narrow_open_filter_rejects_a_wide_range():
    s = build_strategy("orb_breakout", {"max_or_width_percentile": 0.4})
    ctx = make_context(_orb_features(or_width_percentile=0.9), previous_bar={"close": 18005.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_VOLATILITY


def test_orb_range_stop_basis_uses_the_opposite_side_of_the_range():
    s = build_strategy("orb_breakout", {"stop_basis": "range"})
    ctx = make_context(_orb_features(), previous_bar={"close": 18005.0}, timestamp=at(11, 0))
    assert s.on_bar(ctx).stop_price == 17990.0


def test_orb_is_invalidated_by_a_close_through_the_far_side(position_factory):
    s = build_strategy("orb_breakout")
    pos = position_factory(side=Side.BUY, entry=18020.0, stop=18015.0)
    ctx = make_context(_orb_features(close=17985.0), timestamp=at(11, 30))
    decision = s.manage(ctx, pos)
    assert decision is not None and "opposite side" in decision.detail


# ============================================================ support / resistance rejection


def _rej_features(**over):
    base = {
        "atr": 4.0, "close": 17996.0, "resistance": 18000.0, "support": 17900.0,
        "adx": 18.0, "rsi": 62.0,
    }
    return {**base, **over}


def test_rejection_fades_a_failed_test_of_resistance():
    s = build_strategy("sr_rejection")
    ctx = make_context(_rej_features(), bar={"open": 18001.0, "high": 18002.0, "low": 17995.0,
                                             "close": 17996.0}, timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.SELL
    assert intent.stop_price == 18004.0  # max(high, level) + 0.5 x ATR
    assert "tested_resistance" in intent.conditions


def test_rejection_does_not_fade_a_level_that_actually_broke():
    s = build_strategy("sr_rejection")
    ctx = make_context(_rej_features(close=18006.0),
                       bar={"open": 17999.0, "high": 18008.0, "low": 17998.0, "close": 18006.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_REGIME


def test_rejection_needs_a_bar_that_closes_against_the_test():
    s = build_strategy("sr_rejection")
    # Touched resistance, closed below it, but the bar itself is bullish.
    ctx = make_context(_rej_features(),
                       bar={"open": 17990.0, "high": 18002.0, "low": 17989.0, "close": 17996.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None


def test_rejection_refuses_to_fade_a_trending_market():
    s = build_strategy("sr_rejection")
    ctx = make_context(_rej_features(adx=30.0),
                       bar={"open": 18001.0, "high": 18002.0, "low": 17995.0, "close": 17996.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_TREND


def test_rejection_buys_a_held_support():
    s = build_strategy("sr_rejection")
    ctx = make_context(_rej_features(close=17904.0, rsi=38.0, support=17900.0,
                                     resistance=float("nan")),
                       bar={"open": 17899.0, "high": 17905.0, "low": 17898.0, "close": 17904.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.BUY
    assert intent.stop_price == 17896.0  # min(low, level) - 0.5 x ATR


# ============================================================ breakout and retest


def test_breakout_retest_requires_the_break_before_it_will_enter():
    s = build_strategy("sr_breakout_retest")
    ctx = make_context(
        {"atr": 4.0, "close": 18002.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.5},
        previous_bar={"close": 17998.0}, timestamp=at(11, 0),
    )
    # The break bar itself is never the entry.
    assert s.on_bar(ctx) is None


def test_breakout_retest_enters_when_price_returns_and_holds():
    s = build_strategy("sr_breakout_retest")
    break_ctx = make_context(
        {"atr": 4.0, "close": 18002.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.5},
        previous_bar={"close": 17998.0}, timestamp=at(11, 0),
    )
    s.on_bar(break_ctx)

    retest = make_context(
        {"atr": 4.0, "close": 18001.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.0},
        bar={"open": 18002.0, "high": 18003.0, "low": 17999.5, "close": 18001.0},
        timestamp=at(11, 5), i=break_ctx.i + 5,
    )
    intent = s.on_bar(retest)
    assert intent is not None and intent.side is Side.BUY
    assert "retest_held" in intent.conditions


def test_breakout_retest_abandons_a_retest_that_fails():
    s = build_strategy("sr_breakout_retest")
    s.on_bar(make_context(
        {"atr": 4.0, "close": 18002.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.5},
        previous_bar={"close": 17998.0}, timestamp=at(11, 0)))

    failed = make_context(
        {"atr": 4.0, "close": 17997.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.0},
        bar={"open": 18001.0, "high": 18001.0, "low": 17996.0, "close": 17997.0},
        timestamp=at(11, 5), i=6)
    assert s.on_bar(failed) is None
    assert failed.rejections[-1].reason is RejectReason.FILTER_REGIME
    assert s._pending is None


def test_breakout_retest_forgets_a_break_when_the_session_rolls():
    s = build_strategy("sr_breakout_retest")
    s.on_bar(make_context(
        {"atr": 4.0, "close": 18002.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.5},
        previous_bar={"close": 17998.0}, timestamp=at(11, 0)))
    assert s._pending is not None

    s.on_session_start(datetime(2024, 4, 2, 9, 30, tzinfo=MARKET_TZ).date())
    assert s._pending is None, "a break from yesterday is not a setup today"


def test_breakout_retest_window_expires():
    s = build_strategy("sr_breakout_retest", {"retest_window_bars": 3})
    s.on_bar(make_context(
        {"atr": 4.0, "close": 18002.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.5},
        previous_bar={"close": 17998.0}, timestamp=at(11, 0)))

    late = make_context(
        {"atr": 4.0, "close": 18001.0, "resistance": 18000.0, "support": 17900.0,
         "volume_ratio": 1.0},
        bar={"open": 18002.0, "high": 18003.0, "low": 17999.5, "close": 18001.0},
        timestamp=at(11, 30), i=99)
    assert s.on_bar(late) is None
    assert "window expired" in late.rejections[-1].detail


# ============================================================ VWAP reversion


def _vwap_rev_features(**over):
    base = {
        "atr": 4.0, "close": 17992.0, "vwap": 18000.0,
        "vwap_lower": 17995.0, "vwap_upper": 18005.0,
        "adx": 15.0, "efficiency_ratio": 0.2, "rsi": 35.0,
    }
    return {**base, **over}


def test_vwap_reversion_buys_a_stretched_discount_that_turns_back():
    s = build_strategy("vwap_reversion")
    ctx = make_context(_vwap_rev_features(),
                       bar={"open": 17990.0, "high": 17993.0, "low": 17988.0, "close": 17992.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.BUY
    assert intent.stop_price == 17988.0  # close - 1.0 x ATR
    assert intent.target_price == 18000.0  # VWAP itself


def test_vwap_reversion_ignores_a_stretch_that_has_not_turned():
    s = build_strategy("vwap_reversion")
    ctx = make_context(_vwap_rev_features(),
                       bar={"open": 17995.0, "high": 17995.0, "low": 17990.0, "close": 17992.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert "turned back" in ctx.rejections[-1].detail


def test_vwap_reversion_needs_a_real_stretch():
    s = build_strategy("vwap_reversion")
    ctx = make_context(_vwap_rev_features(close=17999.0),
                       bar={"open": 17998.0, "high": 17999.5, "low": 17997.0, "close": 17999.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None


def test_vwap_reversion_stands_down_in_a_trend():
    s = build_strategy("vwap_reversion")
    ctx = make_context(_vwap_rev_features(adx=30.0),
                       bar={"open": 17990.0, "high": 17993.0, "low": 17988.0, "close": 17992.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_TREND


def test_vwap_reversion_stands_down_when_price_is_travelling_efficiently():
    s = build_strategy("vwap_reversion")
    ctx = make_context(_vwap_rev_features(efficiency_ratio=0.8),
                       bar={"open": 17990.0, "high": 17993.0, "low": 17988.0, "close": 17992.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_REGIME


# ============================================================ VWAP continuation


def _vwap_cont_features(**over):
    base = {
        "atr": 4.0, "close": 18004.0, "vwap": 18000.0, "adx": 28.0,
        "efficiency_ratio": 0.5, "ema_fast": 18005.0, "ema_slow": 18002.0,
    }
    return {**base, **over}


def test_vwap_continuation_buys_a_pullback_that_holds_vwap():
    s = build_strategy("vwap_continuation")
    ctx = make_context(_vwap_cont_features(),
                       bar={"open": 18003.0, "high": 18005.0, "low": 18000.5, "close": 18004.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.BUY
    assert intent.stop_price == 17996.0  # VWAP - 1.0 x ATR
    assert "pulled_back_to_vwap" in intent.conditions


def test_vwap_continuation_requires_a_trend_to_continue():
    s = build_strategy("vwap_continuation")
    ctx = make_context(_vwap_cont_features(adx=15.0),
                       bar={"open": 18003.0, "high": 18005.0, "low": 18000.5, "close": 18004.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None
    assert ctx.rejections[-1].reason is RejectReason.FILTER_TREND


def test_vwap_continuation_needs_price_to_actually_reach_vwap():
    s = build_strategy("vwap_continuation")
    ctx = make_context(_vwap_cont_features(close=18020.0),
                       bar={"open": 18018.0, "high": 18021.0, "low": 18016.0, "close": 18020.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None


def test_vwap_continuation_and_reversion_disagree_on_the_same_bar_by_design():
    """The two rules are opposites gated by opposite regimes; neither may fire on both."""
    rev = build_strategy("vwap_reversion")
    cont = build_strategy("vwap_continuation")
    bar = {"open": 18003.0, "high": 18005.0, "low": 18000.5, "close": 18004.0}

    trending = _vwap_cont_features()
    assert cont.on_bar(make_context(trending, bar=bar, timestamp=at(11, 0))) is not None
    assert rev.on_bar(make_context(trending, bar=bar, timestamp=at(11, 0))) is None


# ============================================================ trend pullback


def _trend_features(**over):
    base = {
        "atr": 4.0, "close": 18010.0, "adx": 30.0, "efficiency_ratio": 0.5,
        "ema_fast": 18012.0, "ema_slow": 18008.0, "ema_trend": 17990.0,
        "swing_low": 17995.0, "swing_high": 18030.0,
    }
    return {**base, **over}


def test_trend_pullback_buys_a_pullback_to_the_slow_ema():
    s = build_strategy("trend_pullback")
    ctx = make_context(_trend_features(),
                       bar={"open": 18009.0, "high": 18011.0, "low": 18008.5, "close": 18010.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    assert intent is not None and intent.side is Side.BUY
    # Structural stop (swing 17995 - 0.4 x ATR = 17993.4 -> 17993.25) is further away than
    # the ATR floor (18010 - 3 = 18007), so the structure wins.
    assert intent.stop_price == 17993.25
    assert "ema_stack_up" in intent.conditions


def test_trend_pullback_applies_an_atr_floor_when_the_structure_is_too_tight():
    s = build_strategy("trend_pullback")
    ctx = make_context(_trend_features(swing_low=18009.0),
                       bar={"open": 18009.0, "high": 18011.0, "low": 18008.5, "close": 18010.0},
                       timestamp=at(11, 0))
    intent = s.on_bar(ctx)
    # min(structure 18007.4, close - 0.75 x ATR = 18007.0) -> 18007.0
    assert intent.stop_price == 18007.0


def test_trend_pullback_requires_the_ema_stack_to_be_ordered():
    s = build_strategy("trend_pullback")
    ctx = make_context(_trend_features(ema_slow=18020.0),  # stack broken
                       bar={"open": 18009.0, "high": 18011.0, "low": 18008.5, "close": 18010.0},
                       timestamp=at(11, 0))
    assert s.on_bar(ctx) is None


def test_trend_pullback_is_invalidated_when_the_stack_breaks(position_factory):
    s = build_strategy("trend_pullback")
    pos = position_factory(side=Side.BUY, entry=18010.0, stop=17993.0)
    ctx = make_context(_trend_features(close=17980.0), timestamp=at(11, 30))
    decision = s.manage(ctx, pos)
    assert decision is not None and "trend EMA" in decision.detail


# ============================================================ shared stop management


def test_the_stop_moves_to_break_even_after_the_configured_r(position_factory):
    s = build_strategy("orb_breakout", {"breakeven_at_r": 1.0})
    pos = position_factory(side=Side.BUY, entry=18000.0, stop=17990.0, risk_points=10.0)
    ctx = make_context(_orb_features(close=18012.0, or_high=17900.0, or_low=17850.0),
                       bar={"open": 18010.0, "high": 18013.0, "low": 18009.0, "close": 18012.0},
                       timestamp=at(11, 30))
    update = s.manage(ctx, pos)
    assert update is not None and update.new_stop == 18000.0


def test_the_stop_is_never_moved_backwards(position_factory):
    s = build_strategy("orb_breakout", {"breakeven_at_r": 1.0})
    pos = position_factory(side=Side.BUY, entry=18000.0, stop=18005.0, risk_points=10.0)
    ctx = make_context(_orb_features(close=18012.0, or_high=17900.0, or_low=17850.0),
                       bar={"open": 18010.0, "high": 18013.0, "low": 18009.0, "close": 18012.0},
                       timestamp=at(11, 30))
    assert s.manage(ctx, pos) is None, "a stop must never widen"


def test_a_trailing_stop_follows_the_best_price(position_factory):
    s = build_strategy("vwap_continuation", {"trail_atr_mult": 2.0, "breakeven_at_r": None})
    pos = position_factory(side=Side.BUY, entry=18000.0, stop=17990.0, risk_points=10.0)
    pos.max_favorable_price = 18040.0
    ctx = make_context(_vwap_cont_features(close=18035.0),
                       bar={"open": 18034.0, "high": 18040.0, "low": 18033.0, "close": 18035.0},
                       timestamp=at(11, 30))
    update = s.manage(ctx, pos)
    assert update is not None and update.new_stop == 18032.0  # 18040 - 2 x ATR 4


def test_the_time_stop_closes_a_stale_position(position_factory):
    s = build_strategy("orb_breakout", {"max_hold_bars": 5})
    pos = position_factory(side=Side.BUY, entry=18000.0, stop=17990.0)
    pos.bars_held = 5
    ctx = make_context(_orb_features(or_high=17900.0, or_low=17850.0), timestamp=at(11, 30))
    decision = s.manage(ctx, pos)
    assert decision is not None and decision.reason.value == "TIME_STOP"


# ============================================================ integration sweep


@pytest.mark.parametrize("name", known_strategies())
def test_every_strategy_survives_a_multi_session_sweep(name, multi_session_bars):
    """Invariants that hold regardless of what the rules decide."""
    strategy = build_strategy(name)
    features = build_features(multi_session_bars, strategy.features)
    frame = features.frame

    intents = []
    for i in range(features.warmup_bars, len(frame)):
        ts = frame.index[i].to_pydatetime()
        ctx = StrategyContext(
            i=i, timestamp=ts, bar=multi_session_bars.iloc[i], features=frame.iloc[i],
            instrument=MNQ, equity=50_000.0, session_date=ts.date(),
            trades_this_session=strategy.trades_this_session,
            _frame=frame, _bars=multi_session_bars,
        )
        intent = strategy.on_bar(ctx)
        if intent is not None:
            intents.append(intent)
            strategy.on_trade_opened(None)

    hours = strategy.spec.trading_hours
    for intent in intents:
        assert hours.may_enter(intent.timestamp), f"{name} entered outside declared hours"
        if intent.side is Side.BUY:
            assert intent.stop_price < intent.reference_price
            if intent.target_price is not None:
                assert intent.target_price > intent.reference_price
        else:
            assert intent.stop_price > intent.reference_price
            if intent.target_price is not None:
                assert intent.target_price < intent.reference_price
        assert intent.conditions, "every intent must name the conditions that fired"
        assert intent.stop_distance > 0


@pytest.mark.parametrize("name", known_strategies())
def test_every_strategy_respects_its_per_session_cap(name, multi_session_bars):
    strategy = build_strategy(name)
    features = build_features(multi_session_bars, strategy.features)
    frame = features.frame

    per_session: dict = {}
    for i in range(features.warmup_bars, len(frame)):
        ts = frame.index[i].to_pydatetime()
        ctx = StrategyContext(
            i=i, timestamp=ts, bar=multi_session_bars.iloc[i], features=frame.iloc[i],
            instrument=MNQ, equity=50_000.0, session_date=ts.date(),
            trades_this_session=strategy.trades_this_session,
            _frame=frame, _bars=multi_session_bars,
        )
        if strategy.on_bar(ctx) is not None:
            strategy.on_trade_opened(None)
            per_session[ts.date()] = per_session.get(ts.date(), 0) + 1

    cap = strategy.spec.max_trades_per_session
    for session, count in per_session.items():
        assert count <= cap, f"{name} took {count} trades on {session}, cap is {cap}"


def test_at_least_some_strategies_actually_fire_on_realistic_data(multi_session_bars):
    """A suite where nothing ever triggers would pass while testing nothing."""
    fired = []
    for name in known_strategies():
        strategy = build_strategy(name)
        features = build_features(multi_session_bars, strategy.features)
        frame = features.frame
        for i in range(features.warmup_bars, len(frame)):
            ts = frame.index[i].to_pydatetime()
            ctx = StrategyContext(
                i=i, timestamp=ts, bar=multi_session_bars.iloc[i], features=frame.iloc[i],
                instrument=MNQ, equity=50_000.0, session_date=ts.date(),
                trades_this_session=strategy.trades_this_session,
                _frame=frame, _bars=multi_session_bars,
            )
            if strategy.on_bar(ctx) is not None:
                fired.append(name)
                break
    assert len(fired) >= 4, f"only {fired} produced a signal on ten sessions"
