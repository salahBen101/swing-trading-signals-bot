"""The sixteen pre-registered hypotheses from RESEARCH_PLAN section 5.

Each is a claim about *market behaviour* with a stated reason the behaviour would exist,
implemented as the simplest deterministic rule that expresses it. That ordering is
deliberate: a rule invented first and rationalised afterwards is indistinguishable from a
rule fitted to noise, and this project has already spent fourteen families finding that
out.

Every signal function returns an array of -1 / 0 / +1 aligned to the bar frame, using only
columns that were knowable at that bar. The harness handles fills, exits and costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .harness import ExitRule, Hypothesis


def _blank(bars: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(bars), dtype="int8")


def _col(bars: pd.DataFrame, name: str) -> np.ndarray:
    return bars[name].to_numpy(dtype="float64")


def _after_minutes(bars: pd.DataFrame, minutes: int) -> np.ndarray:
    return bars["minutes_since_open"].to_numpy() >= minutes


def _within_minutes(bars: pd.DataFrame, minutes: int) -> np.ndarray:
    return bars["minutes_since_open"].to_numpy() < minutes


def _first_per_session(bars: pd.DataFrame, condition: np.ndarray) -> np.ndarray:
    """Keep only a condition's first True in each session.

    Almost every hypothesis here is about an *event* — a break, a sweep, a reclaim — and an
    event that stays true for twenty bars is one event, not twenty signals. Without this,
    trade counts and t-statistics are inflated by the same observation counted repeatedly.
    """
    frame = pd.DataFrame({"session": bars["session"].to_numpy(), "flag": condition})
    first = frame.groupby("session")["flag"].cumsum() == 1
    return (first & condition).to_numpy()


# ======================================================================= A. opening dynamics


def _a1_opening_drive(bars: pd.DataFrame) -> np.ndarray:
    """A session that opens and drives away from the open on strong volume continues.

    Measured at the 30-minute mark: price is more than 0.5 opening-range widths beyond the
    open, in the direction it has been going, on above-average volume.
    """
    out = _blank(bars)
    minutes = bars["minutes_since_open"].to_numpy()
    close = _col(bars, "close")
    session_open = _col(bars, "rth_open")
    or_range = _col(bars, "or_range")
    volume_ratio = _col(bars, "volume_ratio")

    displacement = close - session_open
    strong = np.abs(displacement) >= 0.5 * or_range
    at_mark = minutes == 30
    live = strong & at_mark & (volume_ratio >= 1.0) & np.isfinite(or_range) & (or_range > 0)

    out[live & (displacement > 0)] = 1
    out[live & (displacement < 0)] = -1
    return out


def _a2_range_expansion(bars: pd.DataFrame) -> np.ndarray:
    """Sessions whose opening range is unusually wide trend further.

    The opposite claim to the narrow-open ORB filter that failed here before, and worth
    stating separately rather than as a parameter flip: wide early range means disagreement
    being resolved, narrow means balance.
    """
    out = _blank(bars)
    minutes = bars["minutes_since_open"].to_numpy()
    close = _col(bars, "close")
    or_high = _col(bars, "or_high")
    or_low = _col(bars, "or_low")
    vs_trailing = _col(bars, "or_range_vs_trailing")

    wide = vs_trailing >= 1.3
    at_mark = minutes == 30
    out[at_mark & wide & (close > or_high)] = 1
    out[at_mark & wide & (close < or_low)] = -1
    return out


def _a3_failed_orb(bars: pd.DataFrame) -> np.ndarray:
    """A break of the opening range that closes back inside reverses.

    Trapped breakout traders must exit; their exits are the fuel for the move the other
    way. Requires the break to have happened and then failed, not merely a close inside.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    high = _col(bars, "high")
    low = _col(bars, "low")
    or_high = _col(bars, "or_high")
    or_low = _col(bars, "or_low")
    after = _after_minutes(bars, 30)

    sessions = bars["session"].to_numpy()
    broke_up = pd.Series((high > or_high) & after).groupby(sessions).cummax().to_numpy()
    broke_down = pd.Series((low < or_low) & after).groupby(sessions).cummax().to_numpy()

    failed_up = broke_up & (close < or_high) & after
    failed_down = broke_down & (close > or_low) & after

    out[_first_per_session(bars, failed_up)] = -1
    out[_first_per_session(bars, failed_down)] = 1
    return out


def _a4_opening_reversal(bars: pd.DataFrame) -> np.ndarray:
    """A session that gaps and then reverses in the first 30 minutes keeps reversing.

    Opening gaps overshoot; the early reversal is the correction beginning, and gap fill is
    a well-documented tendency rather than an indicator artefact.
    """
    out = _blank(bars)
    minutes = bars["minutes_since_open"].to_numpy()
    close = _col(bars, "close")
    session_open = _col(bars, "rth_open")
    gap_atr = _col(bars, "gap_atr")

    at_mark = minutes == 30
    moved_against_gap = (close - session_open) * np.sign(gap_atr) < 0
    meaningful_gap = np.abs(gap_atr) >= 0.25

    live = at_mark & moved_against_gap & meaningful_gap
    # Trade the direction of the reversal, i.e. against the gap.
    out[live & (gap_atr > 0)] = -1
    out[live & (gap_atr < 0)] = 1
    return out


def _a5_overnight_inventory(bars: pd.DataFrame) -> np.ndarray:
    """A strong one-way overnight session is partly given back once RTH liquidity arrives.

    Overnight is thin; dealers accumulate one-sided inventory that cannot be offloaded
    until the regular session provides depth. Only testable with the ETH dataset built for
    this phase — every prior family here was RTH-only.
    """
    out = _blank(bars)
    minutes = bars["minutes_since_open"].to_numpy()
    on_move_atr = _col(bars, "on_move_atr")
    on_close_position = _col(bars, "on_close_position")

    # Entered at the open, held; the claim is about the session, not about a trigger.
    at_open = minutes == 0
    strong_up = (on_move_atr >= 0.5) & (on_close_position >= 0.7)
    strong_down = (on_move_atr <= -0.5) & (on_close_position <= 0.3)

    out[at_open & strong_up] = -1
    out[at_open & strong_down] = 1
    return out


# ================================================================== B. VWAP / session structure


def _b1_vwap_continuation(bars: pd.DataFrame) -> np.ndarray:
    """In a session holding one side of VWAP, pullbacks to VWAP resume.

    VWAP is the institutional execution benchmark; a side that keeps defending it is real
    size rather than a chart line.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    low = _col(bars, "low")
    high = _col(bars, "high")
    vwap = _col(bars, "vwap")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 60)

    tolerance = 0.25 * atr
    holding_above = close > vwap
    holding_below = close < vwap
    touched_from_above = low <= vwap + tolerance
    touched_from_below = high >= vwap - tolerance

    out[after & holding_above & touched_from_above] = 1
    out[after & holding_below & touched_from_below] = -1
    return out


def _b2_vwap_rejection(bars: pd.DataFrame) -> np.ndarray:
    """Price reaching VWAP from the wrong side and failing to reclaim it continues away.

    Failure to trade back to the session's average price marks one side as unwilling to
    pay up.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    open_ = _col(bars, "open")
    high = _col(bars, "high")
    low = _col(bars, "low")
    vwap = _col(bars, "vwap")
    after = _after_minutes(bars, 60)

    # Approached VWAP from below and closed back below it: sellers still in control.
    rejected_below = (high >= vwap) & (close < vwap) & (close < open_)
    rejected_above = (low <= vwap) & (close > vwap) & (close > open_)

    out[after & rejected_below] = -1
    out[after & rejected_above] = 1
    return out


def _b3_stretched_reversion(bars: pd.DataFrame) -> np.ndarray:
    """Price a long way from VWAP reverts toward it.

    VWAP-benchmarked execution algorithms mechanically lean against deviation, which is a
    real flow rather than a statistical hope.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    open_ = _col(bars, "open")
    vwap = _col(bars, "vwap")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 45)

    stretch = (close - vwap) / np.where(atr > 0, atr, np.nan)
    turned_up = close > open_
    turned_down = close < open_

    out[after & (stretch <= -2.0) & turned_up] = 1
    out[after & (stretch >= 2.0) & turned_down] = -1
    return out


def _b4_vwap_reclaim(bars: pd.DataFrame) -> np.ndarray:
    """A failed push to a session extreme, followed by a VWAP reclaim, continues.

    Two-stage confirmation: the auction failed at the extreme, and then price traded back
    through the session's fair value. Deliberately rare.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    vwap = _col(bars, "vwap")
    session_high = _col(bars, "session_high")
    session_low = _col(bars, "session_low")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 60)

    sessions = bars["session"].to_numpy()
    previous_close = pd.Series(close).groupby(sessions).shift(1).to_numpy()
    previous_vwap = pd.Series(vwap).groupby(sessions).shift(1).to_numpy()

    crossed_up = (close > vwap) & (previous_close <= previous_vwap)
    crossed_down = (close < vwap) & (previous_close >= previous_vwap)

    # The extreme must be genuinely away, so this is a reclaim rather than chop at VWAP.
    far_from_low = (close - session_low) >= 1.5 * atr
    far_from_high = (session_high - close) >= 1.5 * atr

    out[_first_per_session(bars, after & crossed_up & far_from_low)] = 1
    out[_first_per_session(bars, after & crossed_down & far_from_high)] = -1
    return out


# ================================================================= C. prior-session structure


def _c1_prior_extreme_break(bars: pd.DataFrame) -> np.ndarray:
    """Clearing a prior-session extreme continues.

    Resting stops and breakout orders cluster at obvious prior levels; taking them out is
    itself the move.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    prior_high = _col(bars, "prior_high")
    prior_low = _col(bars, "prior_low")
    after = _after_minutes(bars, 15)

    broke_up = after & (close > prior_high)
    broke_down = after & (close < prior_low)

    out[_first_per_session(bars, broke_up)] = 1
    out[_first_per_session(bars, broke_down)] = -1
    return out


def _c2_failed_prior_break(bars: pd.DataFrame) -> np.ndarray:
    """Clearing a prior extreme and closing back inside reverses.

    The mirror of C1 and the more interesting claim: if the stop cluster *was* the
    liquidity, there is nothing left to push once it is consumed.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    high = _col(bars, "high")
    low = _col(bars, "low")
    prior_high = _col(bars, "prior_high")
    prior_low = _col(bars, "prior_low")
    after = _after_minutes(bars, 15)

    sessions = bars["session"].to_numpy()
    poked_up = pd.Series((high > prior_high) & after).groupby(sessions).cummax().to_numpy()
    poked_down = pd.Series((low < prior_low) & after).groupby(sessions).cummax().to_numpy()

    failed_up = poked_up & (close < prior_high) & after
    failed_down = poked_down & (close > prior_low) & after

    out[_first_per_session(bars, failed_up)] = -1
    out[_first_per_session(bars, failed_down)] = 1
    return out


def _c3_prior_close_interaction(bars: pd.DataFrame) -> np.ndarray:
    """Behaviour around the prior settlement.

    Settlement is the reference for overnight P&L and option strikes, so it attracts and
    then repels flow. The rule takes a rejection of that level.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    open_ = _col(bars, "open")
    high = _col(bars, "high")
    low = _col(bars, "low")
    prior_close = _col(bars, "prior_close")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 30)

    tolerance = 0.3 * atr
    touched = (low <= prior_close + tolerance) & (high >= prior_close - tolerance)
    rejected_up = touched & (close > prior_close) & (close > open_)
    rejected_down = touched & (close < prior_close) & (close < open_)

    out[_first_per_session(bars, after & rejected_up)] = 1
    out[_first_per_session(bars, after & rejected_down)] = -1
    return out


def _c4_overnight_sweep(bars: pd.DataFrame) -> np.ndarray:
    """RTH takes out an overnight extreme and immediately rejects it.

    Overnight extremes are thin-liquidity prints. When the regular session sweeps one and
    fails to hold beyond it, the sweep was liquidation rather than intent. Only testable
    with the ETH dataset built for this phase.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    high = _col(bars, "high")
    low = _col(bars, "low")
    on_high = _col(bars, "on_high")
    on_low = _col(bars, "on_low")
    after = _after_minutes(bars, 5)

    swept_high = after & (high > on_high) & (close < on_high)
    swept_low = after & (low < on_low) & (close > on_low)

    out[_first_per_session(bars, swept_high)] = -1
    out[_first_per_session(bars, swept_low)] = 1
    return out


# ==================================================================== D. trend and pullback


def _d1_strong_session_pullback(bars: pd.DataFrame) -> np.ndarray:
    """In a session with high directional efficiency, pullbacks resume.

    Persistent one-way order flow rarely completes in a single push, so a pause inside a
    directional session is an entry rather than a reversal.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    ema_fast = _col(bars, "ema_fast")
    ema_slow = _col(bars, "ema_slow")
    low = _col(bars, "low")
    high = _col(bars, "high")
    efficiency = _col(bars, "efficiency_ratio")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 45)

    directional = efficiency >= 0.4
    tolerance = 0.3 * atr
    up = (ema_fast > ema_slow) & (low <= ema_slow + tolerance) & (close > ema_slow)
    down = (ema_fast < ema_slow) & (high >= ema_slow - tolerance) & (close < ema_slow)

    out[after & directional & up] = 1
    out[after & directional & down] = -1
    return out


def _d2_breakout_pullback(bars: pd.DataFrame) -> np.ndarray:
    """Break a session extreme, pull back, then continue.

    Requiring the retest trades fewer and later signals than a raw breakout, which is the
    point: the retest is the confirmation the raw break lacks.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    session_high = _col(bars, "session_high")
    session_low = _col(bars, "session_low")
    atr = _col(bars, "atr")
    after = _after_minutes(bars, 60)

    sessions = bars["session"].to_numpy()
    # Pulled back from the session extreme by roughly half an ATR, then closed back up.
    pullback_up = (session_high - close) >= 0.5 * atr
    previous_pullback_up = pd.Series(pullback_up).groupby(sessions).shift(1).to_numpy()
    resumed_up = (previous_pullback_up == 1) & ((session_high - close) < 0.25 * atr)

    pullback_down = (close - session_low) >= 0.5 * atr
    previous_pullback_down = pd.Series(pullback_down).groupby(sessions).shift(1).to_numpy()
    resumed_down = (previous_pullback_down == 1) & ((close - session_low) < 0.25 * atr)

    out[after & resumed_up] = 1
    out[after & resumed_down] = -1
    return out


def _d3_volatility_adjusted_trend(bars: pd.DataFrame) -> np.ndarray:
    """Trend continuation, but only where trailing volatility says trends persist.

    The filter is the hypothesis, not decoration: the claim is that trend persistence is
    regime-dependent and that trailing realized volatility identifies the regime.
    """
    out = _blank(bars)
    close = _col(bars, "close")
    ema_slow = _col(bars, "ema_slow")
    ema_trend = _col(bars, "ema_trend")
    vol_percentile = _col(bars, "vol_percentile")
    adx = _col(bars, "adx")
    after = _after_minutes(bars, 60)

    regime = (vol_percentile >= 0.5) & (adx >= 20)
    up = (close > ema_slow) & (ema_slow > ema_trend)
    down = (close < ema_slow) & (ema_slow < ema_trend)

    out[_first_per_session(bars, after & regime & up)] = 1
    out[_first_per_session(bars, after & regime & down)] = -1
    return out


# ============================================================================== the registry

PRE_REGISTERED: list[Hypothesis] = [
    Hypothesis(
        id="A1_opening_drive", family="A. opening dynamics",
        rationale="Genuine imbalance at the open must be absorbed; a one-sided auction "
                  "that does not immediately fail signals size that is not finished.",
        rules="At the 30-minute mark, |close - session open| >= 0.5 x opening range and "
              "volume ratio >= 1.0; trade the direction of the displacement.",
        signal=_a1_opening_drive, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="A2_range_expansion", family="A. opening dynamics",
        rationale="A wide early range indicates disagreement being resolved directionally, "
                  "rather than balanced rotation.",
        rules="At the 30-minute mark, opening range >= 1.3x its trailing 60-session mean "
              "and close outside the opening range; trade the break direction.",
        signal=_a2_range_expansion, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="A3_failed_orb", family="A. opening dynamics",
        rationale="Trapped breakout traders must exit, supplying fuel in the opposite "
                  "direction.",
        rules="After 30 minutes, price has traded beyond the opening range and then closes "
              "back inside; fade the failed break. First occurrence per session.",
        signal=_a3_failed_orb, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="A4_opening_reversal", family="A. opening dynamics",
        rationale="Opening gaps overshoot; an early reversal is inventory being corrected.",
        rules="At the 30-minute mark, |gap| >= 0.25 daily ATR and price has moved against "
              "the gap from the session open; trade against the gap.",
        signal=_a4_opening_reversal, exit_rule=ExitRule(1.0, 2.0, 36),
    ),
    Hypothesis(
        id="A5_overnight_inventory", family="A. opening dynamics",
        rationale="Overnight is thin and dealer inventory accumulates one-sided; RTH "
                  "liquidity is the first opportunity to offload it.",
        rules="At the RTH open, |overnight move| >= 0.5 daily ATR and the overnight close "
              "sits in the top (bottom) 30% of the overnight range; fade the move.",
        signal=_a5_overnight_inventory, exit_rule=ExitRule(1.0, 2.0, 36),
    ),
    Hypothesis(
        id="B1_vwap_continuation", family="B. VWAP structure",
        rationale="VWAP is the institutional execution benchmark; a side that keeps "
                  "defending it is real size.",
        rules="After 60 minutes, session holding one side of VWAP and price touches within "
              "0.25 ATR of it; trade with the held side.",
        signal=_b1_vwap_continuation, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="B2_vwap_rejection", family="B. VWAP structure",
        rationale="Failure to trade back to the session's average price marks one side as "
                  "unwilling to pay up.",
        rules="After 60 minutes, price reaches VWAP from one side and closes back away "
              "from it with a bar in that direction; trade the rejection.",
        signal=_b2_vwap_rejection, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="B3_stretched_reversion", family="B. VWAP structure",
        rationale="VWAP-benchmarked execution algorithms mechanically lean against "
                  "deviation from it.",
        rules="After 45 minutes, |close - VWAP| >= 2.0 ATR and the bar has turned back "
              "toward VWAP; trade toward VWAP.",
        signal=_b3_stretched_reversion, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="B4_vwap_reclaim", family="B. VWAP structure",
        rationale="A failed auction at a session extreme followed by a reclaim of fair "
                  "value is two-stage confirmation that the extreme was rejected.",
        rules="After 60 minutes, close crosses VWAP while the opposing session extreme is "
              "at least 1.5 ATR away; trade the cross. First occurrence per session.",
        signal=_b4_vwap_reclaim, exit_rule=ExitRule(1.0, 2.0, 30),
    ),
    Hypothesis(
        id="C1_prior_extreme_break", family="C. prior-session structure",
        rationale="Resting stops and breakout orders cluster at obvious prior levels.",
        rules="After 15 minutes, close beyond the prior session's high or low; trade the "
              "break. First occurrence per session.",
        signal=_c1_prior_extreme_break, exit_rule=ExitRule(1.0, 2.0, 30),
    ),
    Hypothesis(
        id="C2_failed_prior_break", family="C. prior-session structure",
        rationale="If the stop cluster was the liquidity, there is no follow-through once "
                  "it has been consumed.",
        rules="After 15 minutes, price trades beyond a prior-session extreme and then "
              "closes back inside; fade it. First occurrence per session.",
        signal=_c2_failed_prior_break, exit_rule=ExitRule(1.0, 2.0, 30),
    ),
    Hypothesis(
        id="C3_prior_close_interaction", family="C. prior-session structure",
        rationale="Settlement is the reference for overnight P&L and option strikes, so it "
                  "attracts and then repels flow.",
        rules="After 30 minutes, the bar spans the prior settlement and closes away from "
              "it in the bar's own direction; trade that way. First per session.",
        signal=_c3_prior_close_interaction, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="C4_overnight_sweep", family="C. prior-session structure",
        rationale="Overnight extremes are thin-liquidity prints; a regular-session sweep "
                  "that fails to hold was liquidation rather than intent.",
        rules="After 5 minutes, price trades beyond an overnight extreme and closes back "
              "inside it; fade the sweep. First occurrence per session.",
        signal=_c4_overnight_sweep, exit_rule=ExitRule(1.0, 2.0, 30),
    ),
    Hypothesis(
        id="D1_strong_session_pullback", family="D. trend and pullback",
        rationale="Persistent one-way order flow rarely completes in a single push.",
        rules="After 45 minutes, efficiency ratio >= 0.4, EMA stack aligned, price pulls "
              "back within 0.3 ATR of the slow EMA and closes back on the trend side.",
        signal=_d1_strong_session_pullback, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="D2_breakout_pullback", family="D. trend and pullback",
        rationale="The retest is the confirmation a raw breakout lacks.",
        rules="After 60 minutes, price pulled back at least 0.5 ATR from the session "
              "extreme on the prior bar and has returned within 0.25 ATR of it.",
        signal=_d2_breakout_pullback, exit_rule=ExitRule(1.0, 2.0, 24),
    ),
    Hypothesis(
        id="D3_volatility_adjusted_trend", family="D. trend and pullback",
        rationale="Trend persistence is regime-dependent; trailing realized volatility "
                  "identifies when trends carry.",
        rules="After 60 minutes, trailing volatility percentile >= 0.5, ADX >= 20 and the "
              "EMA stack is aligned; trade with it. First occurrence per session.",
        signal=_d3_volatility_adjusted_trend, exit_rule=ExitRule(1.0, 2.5, 36),
    ),
]

BY_ID = {h.id: h for h in PRE_REGISTERED}
