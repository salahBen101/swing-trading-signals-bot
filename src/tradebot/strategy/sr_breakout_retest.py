"""Support/resistance breakout and retest.

A level breaks on a close, price returns to it within `retest_window` bars, and the level
holds as the opposite kind of level. The entry is on the hold, not on the break.

The two-stage structure is what distinguishes this from a plain breakout: the break alone
is the entry that pays for everyone else's stop hunt. Requiring a retest costs signals and
is meant to buy confirmation. Whether it does is what the backtest measures.

State is per-session and per-level: the pending break is cleared when the session rolls,
when the retest window expires, or when the level fails outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "retest_window_bars": 20,
    "retest_tolerance_atr": 0.3,
    "stop_buffer_atr": 0.5,
    "target_r": 2.0,
    "min_volume_ratio_on_break": 1.2,
    "max_hold_bars": 90,
    "breakeven_at_r": 1.0,
    "trail_atr_mult": None,
    "pivot_left": 3,
    "pivot_right": 3,
}


@dataclass(slots=True)
class _PendingBreak:
    level: float
    side: Side  # the direction the break implies
    bar_index: int
    session: date


class SupportResistanceBreakoutRetest(Strategy):
    def __init__(self, spec: StrategySpec) -> None:
        super().__init__(spec)
        self._pending: _PendingBreak | None = None

    @classmethod
    def build(cls, **params) -> SupportResistanceBreakoutRetest:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="sr_breakout_retest",
            description="level break, then a retest that holds",
            entry_conditions=(
                "a close broke a confirmed level",
                f"volume_ratio >= {p['min_volume_ratio_on_break']} on the break bar",
                f"price returned within {p['retest_tolerance_atr']} x ATR of the level "
                f"inside {p['retest_window_bars']} bars",
                "the retest bar closed back on the breakout side",
            ),
            invalidation_conditions=("close_back_through_the_broken_level",),
            stop_loss=f"broken level -/+ {p['stop_buffer_atr']} x ATR",
            profit_target=f"{p['target_r']}R",
            filters=("a confirmed pivot level exists", "the break is inside the session"),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            trading_hours=TradingHours(time(9, 45), time(15, 30), time(15, 58)),
            features=FeatureSpec(
                pivot_left=int(p["pivot_left"]), pivot_right=int(p["pivot_right"])
            ),
            params=p,
        )
        return cls(spec)

    def on_session_start(self, session: date) -> None:
        super().on_session_start(session)
        # A break from yesterday is not a setup today: the level's meaning is reset by the
        # overnight session, which this system does not observe.
        self._pending = None

    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        p = self.spec.params
        atr = ctx.f("atr")
        if atr != atr or atr <= 0:
            return None

        c = float(ctx.bar["close"])
        h, l = float(ctx.bar["high"]), float(ctx.bar["low"])
        prev = ctx.previous(1)
        prev_close = float(prev["close"]) if prev is not None else c
        volume_ratio = ctx.f("volume_ratio")

        self._record_break(ctx, c, prev_close, volume_ratio)

        pending = self._pending
        if pending is None:
            return None

        age = ctx.i - pending.bar_index
        if age > int(p["retest_window_bars"]):
            ctx.reject(
                RejectReason.FILTER_REGIME,
                f"retest window expired {age} bars after the break of {pending.level:.2f}",
                level=pending.level,
            )
            self._pending = None
            return None
        if age == 0:
            return None  # the break bar itself is not a retest

        tolerance = float(p["retest_tolerance_atr"]) * atr
        buffer = float(p["stop_buffer_atr"]) * atr

        if pending.side is Side.BUY:
            returned = l <= pending.level + tolerance
            held = c > pending.level
            if returned and not held:
                self._pending = None  # the level failed; it is resistance again
                ctx.reject(
                    RejectReason.FILTER_REGIME,
                    f"retest of {pending.level:.2f} failed to hold",
                    level=pending.level,
                )
                return None
            if returned and held:
                stop = min(l, pending.level) - buffer
                self._pending = None
                return self._make_intent(
                    ctx, Side.BUY, stop,
                    target_price=self._target_from_r(ctx, Side.BUY, stop, float(p["target_r"])),
                    conditions=("level_broke_up", "price_retested_the_level",
                                "retest_held", "volume_confirmed_the_break"),
                    features={"level": pending.level, "retest_age_bars": float(age)},
                )
        else:
            returned = h >= pending.level - tolerance
            held = c < pending.level
            if returned and not held:
                self._pending = None
                ctx.reject(
                    RejectReason.FILTER_REGIME,
                    f"retest of {pending.level:.2f} failed to hold",
                    level=pending.level,
                )
                return None
            if returned and held:
                stop = max(h, pending.level) + buffer
                self._pending = None
                return self._make_intent(
                    ctx, Side.SELL, stop,
                    target_price=self._target_from_r(ctx, Side.SELL, stop, float(p["target_r"])),
                    conditions=("level_broke_down", "price_retested_the_level",
                                "retest_held", "volume_confirmed_the_break"),
                    features={"level": pending.level, "retest_age_bars": float(age)},
                )
        return None

    def _record_break(
        self, ctx: StrategyContext, close: float, prev_close: float, volume_ratio: float
    ) -> None:
        """Latch a fresh break so a later bar can look for its retest."""
        min_volume = float(self.spec.params["min_volume_ratio_on_break"])
        if volume_ratio == volume_ratio and volume_ratio < min_volume:
            return

        resistance = ctx.f("resistance")
        if resistance == resistance and close > resistance and prev_close <= resistance:
            self._pending = _PendingBreak(resistance, Side.BUY, ctx.i, ctx.session_date)
            return

        support = ctx.f("support")
        if support == support and close < support and prev_close >= support:
            self._pending = _PendingBreak(support, Side.SELL, ctx.i, ctx.session_date)

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        level = position.entry_features.get("level")
        if level is None:
            return None
        close = float(ctx.bar["close"])
        if position.is_long and close < float(level):
            return f"close {close:.2f} fell back below the broken level {level:.2f}"
        if not position.is_long and close > float(level):
            return f"close {close:.2f} rose back above the broken level {level:.2f}"
        return None
