"""VWAP reversion.

Price stretches a configurable number of ATRs away from session VWAP, closes back toward
it, and is faded with VWAP itself as the target.

Measuring the stretch in ATRs rather than points is deliberate: "20 points from VWAP" means
something different in a quiet August session and on an FOMC afternoon, and a fixed-point
threshold silently becomes a volatility filter instead of a distance one.

Like every fade, this is gated to a non-trending regime — both an ADX ceiling and an
efficiency-ratio ceiling, because ADX alone is slow to recognise a fresh trend.
"""

from __future__ import annotations

from datetime import time

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "stretch_atr": 1.5,
    "stop_atr_mult": 1.0,
    "target": "vwap",  # "vwap" | "r"
    "target_r": 1.5,
    "max_adx": 25.0,
    "max_efficiency_ratio": 0.45,
    "require_band_break": True,
    "max_hold_bars": 60,
    "breakeven_at_r": None,
    "trail_atr_mult": None,
    "vwap_band_std": 1.0,
}


class VwapReversion(Strategy):
    @classmethod
    def build(cls, **params) -> VwapReversion:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="vwap_reversion",
            description="fade a stretched deviation from session VWAP",
            entry_conditions=(
                f"|close - vwap| >= {p['stretch_atr']} x ATR",
                "close outside the VWAP band" if p["require_band_break"] else "(band ignored)",
                "the bar closed back toward VWAP",
            ),
            invalidation_conditions=("close reached VWAP", "stretch widened past the stop"),
            stop_loss=f"entry -/+ {p['stop_atr_mult']} x ATR",
            profit_target="session VWAP" if p["target"] == "vwap" else f"{p['target_r']}R",
            filters=(
                f"adx < {p['max_adx']}",
                f"efficiency_ratio < {p['max_efficiency_ratio']}",
            ),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            trading_hours=TradingHours(time(9, 45), time(15, 30), time(15, 58)),
            features=FeatureSpec(vwap_band_std=float(p["vwap_band_std"])),
            params=p,
        )
        return cls(spec)

    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        p = self.spec.params
        atr, vwap = ctx.f("atr"), ctx.f("vwap")
        if atr != atr or atr <= 0 or vwap != vwap:
            return None

        adx = ctx.f("adx")
        if adx == adx and adx >= float(p["max_adx"]):
            ctx.reject(RejectReason.FILTER_TREND,
                       f"ADX {adx:.1f} >= {p['max_adx']}: trending, do not fade", adx=adx)
            return None

        er = ctx.f("efficiency_ratio")
        if er == er and er >= float(p["max_efficiency_ratio"]):
            ctx.reject(RejectReason.FILTER_REGIME,
                       f"efficiency ratio {er:.2f} >= {p['max_efficiency_ratio']}: directional",
                       efficiency_ratio=er)
            return None

        c, o = float(ctx.bar["close"]), float(ctx.bar["open"])
        stretch = (c - vwap) / atr

        if stretch <= -float(p["stretch_atr"]):
            side = Side.BUY
            turned_back = c > o
            band_ok = (not p["require_band_break"]) or c <= ctx.f("vwap_lower")
        elif stretch >= float(p["stretch_atr"]):
            side = Side.SELL
            turned_back = c < o
            band_ok = (not p["require_band_break"]) or c >= ctx.f("vwap_upper")
        else:
            return None

        if not band_ok:
            ctx.reject(RejectReason.FILTER_VOLATILITY,
                       f"stretch {stretch:.2f} ATR did not clear the VWAP band",
                       stretch_atr=stretch)
            return None
        if not turned_back:
            ctx.reject(RejectReason.FILTER_REGIME,
                       "the bar has not yet turned back toward VWAP", stretch_atr=stretch)
            return None

        offset = float(p["stop_atr_mult"]) * atr
        stop = c - offset if side is Side.BUY else c + offset
        target = vwap if p["target"] == "vwap" else self._target_from_r(
            ctx, side, stop, float(p["target_r"])
        )

        return self._make_intent(
            ctx, side, stop, target_price=target,
            conditions=("stretched_from_vwap", "outside_vwap_band", "turned_back_toward_vwap"),
            features={"vwap": vwap, "stretch_atr": stretch,
                      "efficiency_ratio": er, "entry_vwap": vwap},
        )

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        # The trade's whole premise is a return to VWAP. Once VWAP itself has moved past
        # the entry against us, there is nothing left to revert to.
        vwap = ctx.f("vwap")
        if vwap != vwap:
            return None
        if position.is_long and vwap < position.stop_price:
            return "VWAP has fallen below the stop; there is no reversion target left"
        if not position.is_long and vwap > position.stop_price:
            return "VWAP has risen above the stop; there is no reversion target left"
        return None
