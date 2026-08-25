"""Dynamic exits: cut losers early, trail winners.

The fixed 1-ATR-stop / 2-ATR-target exit used in the first screen is symmetric and
regime-blind. This module replaces it with an asymmetric exit — an early cut on trades that
do not work, and a ratcheting trailing stop on trades that do — and, crucially, provides the
**null control** that keeps the exercise honest.

The scientific point this module exists to test:

    For a zero-edge (random-walk) entry, no exit rule produces positive expectancy. A
    trailing stop reshapes the distribution of wins and losses (more small losses, fewer
    large ones, occasional large win) but leaves the *mean* unchanged, and costs then make
    the mean negative. So a dynamic exit can only add value where the ENTRY carries real
    directional persistence that a fixed target was capping.

Therefore every dynamic-exit result must be read against `null_control()`, which applies the
identical exit to random entries. If the dynamic exit "improves" random entries, any
improvement on a real entry is a mirage of the exit, not an edge in the entry.

Fill conventions are identical to `harness.simulate`: next-bar-open entry, gap-through-stop
fills at the open, one position at a time, flat at session end. The trailing stop uses only
information available at or before the bar it is tested on — the favorable extreme of bar
*j* moves the stop that is tested on bar *j+1*, never on bar *j* itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .harness import ROUND_TRIP_POINTS, Trade, summarize


@dataclass(frozen=True, slots=True)
class DynamicExitRule:
    """Asymmetric exit. All distances in ATR multiples, measured at entry.

    The defaults encode the stated idea: a 1-ATR initial stop, move to break-even once the
    trade is 1 ATR in front, then trail 2 ATR behind the best price; and cut a trade that is
    not at least 0.25 ATR in front after 6 bars.
    """

    initial_stop_atr: float = 1.0
    breakeven_at_atr: float = 1.0
    trail_atr: float = 2.0
    early_cut_bars: int = 6
    early_cut_min_progress_atr: float = 0.25
    max_bars: int = 48
    no_entry_within_bars: int = 3


def simulate_dynamic(
    bars: pd.DataFrame,
    direction: np.ndarray,
    atr: np.ndarray,
    rule: DynamicExitRule,
) -> list[Trade]:
    """Walk each signal forward under the dynamic exit."""
    opens = bars["open"].to_numpy(dtype="float64")
    highs = bars["high"].to_numpy(dtype="float64")
    lows = bars["low"].to_numpy(dtype="float64")
    closes = bars["close"].to_numpy(dtype="float64")
    sessions = bars["session"].to_numpy()

    n = len(bars)
    trades: list[Trade] = []
    blocked_until = -1

    for i in np.flatnonzero(direction != 0):
        if i <= blocked_until or i + 1 >= n:
            continue
        if sessions[i + 1] != sessions[i]:
            continue
        bar_atr = atr[i]
        if not np.isfinite(bar_atr) or bar_atr <= 0:
            continue

        side = int(direction[i])
        entry_index = i + 1
        entry = opens[entry_index]
        session = sessions[entry_index]

        stop = entry - side * rule.initial_stop_atr * bar_atr
        best = entry
        moved_to_breakeven = False

        limit = min(n - 1, entry_index + rule.max_bars)
        exit_index = entry_index
        exit_price = closes[entry_index]
        kind = "timeout"

        j = entry_index
        while j <= limit and sessions[j] == session:
            # 1. Test the stop as it stands entering this bar (pessimistic: before the
            #    bar's own favorable extreme is allowed to move it).
            if side > 0 and lows[j] <= stop:
                exit_index, exit_price, kind = j, min(stop, opens[j]), "trail_stop"
                break
            if side < 0 and highs[j] >= stop:
                exit_index, exit_price, kind = j, max(stop, opens[j]), "trail_stop"
                break

            # 2. Update the best price seen, then ratchet the stop from it.
            best = max(best, highs[j]) if side > 0 else min(best, lows[j])
            progress = (best - entry) * side

            if not moved_to_breakeven and progress >= rule.breakeven_at_atr * bar_atr:
                moved_to_breakeven = True
            if moved_to_breakeven:
                be = entry
                trailed = best - side * rule.trail_atr * bar_atr
                candidate = max(be, trailed) if side > 0 else min(be, trailed)
                # A stop only ever tightens.
                stop = max(stop, candidate) if side > 0 else min(stop, candidate)

            # 3. Cut a trade that is not working. Uses the CLOSE, which is known at bar end.
            held = j - entry_index + 1
            close_progress = (closes[j] - entry) * side
            if (
                not moved_to_breakeven
                and held >= rule.early_cut_bars
                and close_progress < rule.early_cut_min_progress_atr * bar_atr
            ):
                exit_index, exit_price, kind = j, closes[j], "early_cut"
                break

            exit_index, exit_price, kind = j, closes[j], "timeout"
            j += 1

        gross = (exit_price - entry) * side
        trades.append(
            Trade(
                entry_index=entry_index,
                exit_index=exit_index,
                direction=side,
                entry_price=entry,
                exit_price=exit_price,
                gross_points=gross,
                bars_held=exit_index - entry_index,
                exit_kind=kind,
                session=session,
                atr=bar_atr,
            )
        )
        blocked_until = exit_index

    return trades


def screen_dynamic(
    signal: np.ndarray,
    bars: pd.DataFrame,
    atr: np.ndarray,
    rule: DynamicExitRule,
    *,
    hypothesis_id: str,
    bar_minutes: float,
    slippage_ticks: float = 1.0,
    keep_trades: bool = False,
):
    """Screen a signal array under the dynamic exit and return the full metric set."""
    direction = _mask_tail(signal, bars, rule.no_entry_within_bars)
    trades = simulate_dynamic(bars, direction, atr, rule)
    return summarize(hypothesis_id, trades, bars, bar_minutes=bar_minutes,
                     slippage_ticks=slippage_ticks, keep_trades=keep_trades)


def _mask_tail(direction: np.ndarray, bars: pd.DataFrame, within: int) -> np.ndarray:
    out = direction.copy()
    order = np.arange(len(bars))
    from_end = (
        pd.DataFrame({"s": bars["session"].to_numpy(), "o": order})
        .groupby("s")["o"].transform("max").to_numpy() - order
    )
    out[from_end < within] = 0
    return out


def null_control(
    bars: pd.DataFrame,
    atr: np.ndarray,
    rule: DynamicExitRule,
    *,
    n_trials: int = 30,
    signals_per_session: float = 0.5,
    seed: int = 0,
    bar_minutes: float = 5.0,
):
    """Apply the dynamic exit to RANDOM entries, many times.

    Returns the distribution of net points per trade across `n_trials` random-entry sets.
    If the dynamic exit had genuine power, this distribution would be centred above zero.
    It is not — for a random entry the exit cannot manufacture expectancy — so this is the
    yardstick every real result is measured against.
    """
    rng = np.random.default_rng(seed)
    n = len(bars)
    sessions = bars["session"].to_numpy()
    n_sessions = pd.Series(sessions).nunique()
    n_signals = int(n_sessions * signals_per_session)

    nets = []
    for _ in range(n_trials):
        signal = np.zeros(n, dtype="int8")
        picks = rng.integers(0, n, size=n_signals)
        signal[picks] = rng.choice([-1, 1], size=n_signals)
        result = screen_dynamic(signal, bars, atr, rule, hypothesis_id="null",
                                bar_minutes=bar_minutes)
        if result.trades > 0:
            nets.append(result.net_points_per_trade)

    nets = np.array(nets)
    return {
        "trials": len(nets),
        "mean_net": float(nets.mean()) if nets.size else 0.0,
        "std_net": float(nets.std(ddof=1)) if nets.size > 1 else 0.0,
        "p05": float(np.percentile(nets, 5)) if nets.size else 0.0,
        "p95": float(np.percentile(nets, 95)) if nets.size else 0.0,
        "max_net": float(nets.max()) if nets.size else 0.0,
        "round_trip_cost": ROUND_TRIP_POINTS,
    }
