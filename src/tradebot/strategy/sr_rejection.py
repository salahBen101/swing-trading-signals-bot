"""Support/resistance rejection.

Price reaches a confirmed swing level, fails to close through it, and prints a bar in the
opposite direction. Faded, with a stop just beyond the level.

The level comes from `features/levels.py`, where a swing high is only reported `right` bars
after it forms — so this rule trades levels that were genuinely visible at the time, not
ones a chart would draw in hindsight.

An ADX ceiling is applied because fading is a range-regime rule: taking it against a strong
trend is the most reliable way to be repeatedly stopped out by design.
"""

from __future__ import annotations

from datetime import time

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "touch_tolerance_atr": 0.25,  # how close to the level counts as a test
    "stop_buffer_atr": 0.5,  # how far beyond the level the stop sits
    "target_r": 2.0,
    "max_adx": 25.0,  # fade only outside a strong trend
    "rsi_extreme": 55.0,  # short side needs RSI above this, long side below 100-this
    "max_hold_bars": 90,
    "breakeven_at_r": 1.0,
    "trail_atr_mult": None,
    "pivot_left": 3,
    "pivot_right": 3,
}


class SupportResistanceRejection(Strategy):
    @classmethod
    def build(cls, **params) -> SupportResistanceRejection:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="sr_rejection",
            description="fade a failed test of a confirmed swing level",
            entry_conditions=(
                f"bar_extreme_within {p['touch_tolerance_atr']} x ATR of the level",
                "close_did_not_break_the_level",
                "bar_closed_against_the_test (rejection candle)",
                f"rsi confirms (>= {p['rsi_extreme']} short / <= {100 - p['rsi_extreme']} long)",
            ),
            invalidation_conditions=("close_through_the_tested_level",),
            stop_loss=f"level -/+ {p['stop_buffer_atr']} x ATR",
            profit_target=f"{p['target_r']}R",
            filters=(f"adx < {p['max_adx']}", "a confirmed level exists on that side"),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            trading_hours=TradingHours(time(9, 45), time(15, 30), time(15, 58)),
            features=FeatureSpec(
                pivot_left=int(p["pivot_left"]), pivot_right=int(p["pivot_right"])
            ),
            params=p,
        )
        return cls(spec)

    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        p = self.spec.params
        atr = ctx.f("atr")
        if atr != atr or atr <= 0:
            return None

        adx = ctx.f("adx")
        if adx == adx and adx >= float(p["max_adx"]):
            ctx.reject(
                RejectReason.FILTER_TREND,
                f"ADX {adx:.1f} >= {p['max_adx']}: too trending to fade",
                adx=adx,
            )
            return None

        tolerance = float(p["touch_tolerance_atr"]) * atr
        buffer = float(p["stop_buffer_atr"]) * atr
        o, h, l, c = (float(ctx.bar[k]) for k in ("open", "high", "low", "close"))
        rsi = ctx.f("rsi")

        resistance = ctx.f("resistance")
        if resistance == resistance:
            tested = h >= resistance - tolerance
            held = c < resistance
            rejection_bar = c < o
            rsi_ok = rsi != rsi or rsi >= float(p["rsi_extreme"])
            if tested and held and rejection_bar and rsi_ok:
                stop = max(h, resistance) + buffer
                return self._make_intent(
                    ctx, Side.SELL, stop,
                    target_price=self._target_from_r(ctx, Side.SELL, stop, float(p["target_r"])),
                    conditions=("tested_resistance", "closed_below_resistance",
                                "bearish_rejection_bar", "rsi_confirms"),
                    features={"level": resistance, "level_kind": 1.0},
                )
            if tested and not held:
                ctx.reject(
                    RejectReason.FILTER_REGIME,
                    f"close {c:.2f} broke resistance {resistance:.2f} rather than rejecting it",
                    level=resistance,
                )

        support = ctx.f("support")
        if support == support:
            tested = l <= support + tolerance
            held = c > support
            rejection_bar = c > o
            rsi_ok = rsi != rsi or rsi <= 100.0 - float(p["rsi_extreme"])
            if tested and held and rejection_bar and rsi_ok:
                stop = min(l, support) - buffer
                return self._make_intent(
                    ctx, Side.BUY, stop,
                    target_price=self._target_from_r(ctx, Side.BUY, stop, float(p["target_r"])),
                    conditions=("tested_support", "closed_above_support",
                                "bullish_rejection_bar", "rsi_confirms"),
                    features={"level": support, "level_kind": -1.0},
                )
            if tested and not held:
                ctx.reject(
                    RejectReason.FILTER_REGIME,
                    f"close {c:.2f} broke support {support:.2f} rather than rejecting it",
                    level=support,
                )
        return None

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        level = position.entry_features.get("level")
        if level is None:
            return None
        close = float(ctx.bar["close"])
        if not position.is_long and close > float(level):
            return f"close {close:.2f} broke back above the rejected level {level:.2f}"
        if position.is_long and close < float(level):
            return f"close {close:.2f} broke back below the held level {level:.2f}"
        return None
