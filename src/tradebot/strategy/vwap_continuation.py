"""VWAP continuation.

The mirror image of `vwap_reversion`, and gated by the opposite regime test. In a session
that is trending and holding one side of VWAP, a pullback that touches VWAP and closes back
on the trend side is taken with the trend.

Having both rules in the same repository is the point: they use the same feature, the same
costs and the same risk engine, and differ only in the regime they claim to work in. If
neither shows an edge, that is a much more informative result than testing one of them.
"""

from __future__ import annotations

from datetime import time

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "min_adx": 22.0,
    "min_efficiency_ratio": 0.3,
    "touch_tolerance_atr": 0.25,
    "stop_atr_mult": 1.0,
    "target_r": 2.0,
    "require_ema_alignment": True,
    "max_hold_bars": 90,
    "breakeven_at_r": 1.0,
    "trail_atr_mult": 2.0,
}


class VwapContinuation(Strategy):
    @classmethod
    def build(cls, **params) -> VwapContinuation:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="vwap_continuation",
            description="trade a VWAP pullback in the direction of the session trend",
            entry_conditions=(
                "the session is holding one side of VWAP",
                f"price pulled back within {p['touch_tolerance_atr']} x ATR of VWAP",
                "the bar closed back on the trend side of VWAP",
                "EMA fast/slow aligned with the trend" if p["require_ema_alignment"] else "(EMA ignored)",
            ),
            invalidation_conditions=("close on the wrong side of VWAP",),
            stop_loss=f"VWAP -/+ {p['stop_atr_mult']} x ATR",
            profit_target=f"{p['target_r']}R",
            filters=(
                f"adx >= {p['min_adx']}",
                f"efficiency_ratio >= {p['min_efficiency_ratio']}",
            ),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            trading_hours=TradingHours(time(10, 0), time(15, 30), time(15, 58)),
            features=FeatureSpec(),
            params=p,
        )
        return cls(spec)

    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        p = self.spec.params
        atr, vwap = ctx.f("atr"), ctx.f("vwap")
        if atr != atr or atr <= 0 or vwap != vwap:
            return None

        adx = ctx.f("adx")
        if adx != adx or adx < float(p["min_adx"]):
            ctx.reject(RejectReason.FILTER_TREND,
                       f"ADX {adx:.1f} < {p['min_adx']}: no trend to continue", adx=adx)
            return None

        er = ctx.f("efficiency_ratio")
        if er != er or er < float(p["min_efficiency_ratio"]):
            ctx.reject(RejectReason.FILTER_REGIME,
                       f"efficiency ratio {er:.2f} < {p['min_efficiency_ratio']}: choppy",
                       efficiency_ratio=er)
            return None

        c, l, h = (float(ctx.bar[k]) for k in ("close", "low", "high"))
        tolerance = float(p["touch_tolerance_atr"]) * atr
        ema_fast, ema_slow = ctx.f("ema_fast"), ctx.f("ema_slow")
        aligned_up = (not p["require_ema_alignment"]) or (ema_fast > ema_slow)
        aligned_down = (not p["require_ema_alignment"]) or (ema_fast < ema_slow)

        if c > vwap and l <= vwap + tolerance and aligned_up:
            side = Side.BUY
            stop = vwap - float(p["stop_atr_mult"]) * atr
            conditions = ("session_above_vwap", "pulled_back_to_vwap",
                          "closed_back_above_vwap", "ema_aligned_up")
        elif c < vwap and h >= vwap - tolerance and aligned_down:
            side = Side.SELL
            stop = vwap + float(p["stop_atr_mult"]) * atr
            conditions = ("session_below_vwap", "pulled_back_to_vwap",
                          "closed_back_below_vwap", "ema_aligned_down")
        else:
            return None

        return self._make_intent(
            ctx, side, stop,
            target_price=self._target_from_r(ctx, side, stop, float(p["target_r"])),
            conditions=conditions,
            features={"vwap": vwap, "efficiency_ratio": er,
                      "ema_fast": ema_fast, "ema_slow": ema_slow},
        )

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        vwap = ctx.f("vwap")
        close = float(ctx.bar["close"])
        if vwap != vwap:
            return None
        if position.is_long and close < vwap:
            return "close fell back below VWAP; the continuation premise is gone"
        if not position.is_long and close > vwap:
            return "close rose back above VWAP; the continuation premise is gone"
        return None
