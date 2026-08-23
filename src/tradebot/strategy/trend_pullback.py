"""Trend-following pullback.

Three stacked EMAs establish direction, ADX and the efficiency ratio confirm that the move
is actually going somewhere, and the entry waits for a pullback to the slow EMA that closes
back in the trend's direction.

The stop sits beyond the most recent confirmed swing rather than at a fixed ATR distance:
in a trend the structure is the thing that has to break for the trade to be wrong, and an
ATR stop in a fast trend is frequently inside the noise. An ATR floor is still applied so a
very tight structure cannot produce a stop one tick away.
"""

from __future__ import annotations

from datetime import time

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "min_adx": 25.0,
    "min_efficiency_ratio": 0.35,
    "pullback_tolerance_atr": 0.3,
    "stop_buffer_atr": 0.4,
    "min_stop_atr": 0.75,  # floor, so a tight structure cannot make a one-tick stop
    "target_r": 2.5,
    "max_hold_bars": 120,
    "breakeven_at_r": 1.0,
    "trail_atr_mult": 2.5,
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,
}


class TrendPullback(Strategy):
    @classmethod
    def build(cls, **params) -> TrendPullback:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="trend_pullback",
            description="enter a confirmed trend on a pullback to the slow EMA",
            entry_conditions=(
                f"EMA{p['ema_fast']} > EMA{p['ema_slow']} > EMA{p['ema_trend']} (or inverted)",
                f"price pulled back within {p['pullback_tolerance_atr']} x ATR of EMA{p['ema_slow']}",
                "the bar closed back in the trend direction",
            ),
            invalidation_conditions=(
                f"close through EMA{p['ema_trend']} against the trend",
                "EMA stack lost its ordering",
            ),
            stop_loss=(
                f"beyond the last confirmed swing -/+ {p['stop_buffer_atr']} x ATR, "
                f"at least {p['min_stop_atr']} x ATR from entry"
            ),
            profit_target=f"{p['target_r']}R",
            filters=(
                f"adx >= {p['min_adx']}",
                f"efficiency_ratio >= {p['min_efficiency_ratio']}",
            ),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            trading_hours=TradingHours(time(10, 0), time(15, 30), time(15, 58)),
            features=FeatureSpec(
                ema_fast=int(p["ema_fast"]),
                ema_slow=int(p["ema_slow"]),
                ema_trend=int(p["ema_trend"]),
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
        if adx != adx or adx < float(p["min_adx"]):
            ctx.reject(RejectReason.FILTER_TREND,
                       f"ADX {adx:.1f} < {p['min_adx']}: no trend", adx=adx)
            return None

        er = ctx.f("efficiency_ratio")
        if er != er or er < float(p["min_efficiency_ratio"]):
            ctx.reject(RejectReason.FILTER_REGIME,
                       f"efficiency ratio {er:.2f} < {p['min_efficiency_ratio']}: choppy",
                       efficiency_ratio=er)
            return None

        fast, slow, trend = ctx.f("ema_fast"), ctx.f("ema_slow"), ctx.f("ema_trend")
        if any(x != x for x in (fast, slow, trend)):
            return None

        c, l, h = (float(ctx.bar[k]) for k in ("close", "low", "high"))
        tolerance = float(p["pullback_tolerance_atr"]) * atr
        buffer = float(p["stop_buffer_atr"]) * atr
        floor = float(p["min_stop_atr"]) * atr

        if fast > slow > trend and c > slow and l <= slow + tolerance:
            side = Side.BUY
            swing = ctx.f("swing_low")
            structural = (swing - buffer) if swing == swing else (l - buffer)
            stop = min(structural, c - floor)
            conditions = ("ema_stack_up", "pulled_back_to_ema_slow", "closed_back_above_ema_slow")
        elif fast < slow < trend and c < slow and h >= slow - tolerance:
            side = Side.SELL
            swing = ctx.f("swing_high")
            structural = (swing + buffer) if swing == swing else (h + buffer)
            stop = max(structural, c + floor)
            conditions = ("ema_stack_down", "pulled_back_to_ema_slow", "closed_back_below_ema_slow")
        else:
            return None

        return self._make_intent(
            ctx, side, stop,
            target_price=self._target_from_r(ctx, side, stop, float(p["target_r"])),
            conditions=conditions,
            features={"ema_fast": fast, "ema_slow": slow, "ema_trend": trend,
                      "efficiency_ratio": er},
        )

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        fast, slow, trend = ctx.f("ema_fast"), ctx.f("ema_slow"), ctx.f("ema_trend")
        if any(x != x for x in (fast, slow, trend)):
            return None
        close = float(ctx.bar["close"])
        if position.is_long:
            if close < trend:
                return "close fell through the trend EMA"
            if not fast > slow:
                return "the EMA stack lost its upward ordering"
        else:
            if close > trend:
                return "close rose through the trend EMA"
            if not fast < slow:
                return "the EMA stack lost its downward ordering"
        return None
