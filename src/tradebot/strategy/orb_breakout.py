"""Opening-range breakout.

The first `or_minutes` of the session define a range. A close outside it, on confirming
volume, is taken in the direction of the break.

Earlier work in this repository found the only ORB variant with a DEV-sample signal was the
one filtered to *narrow* opening ranges (`max_or_width_percentile` around 0.4) — and that
it then failed the year-by-year test, drawing 140% of its net profit from 2022 alone. The
filter is exposed here because it is part of the family, not because it is believed. See
PROJECT_SPEC §12.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from ..core.models import OrderIntent, Position
from ..core.types import RejectReason, Side
from ..features.pipeline import FeatureSpec
from .base import Strategy, StrategyContext, StrategySpec, TradingHours

DEFAULTS = {
    "or_minutes": 30,
    "stop_basis": "atr",  # "atr" | "range"
    "stop_atr_mult": 1.0,
    "target_r": 2.0,
    "min_volume_ratio": 1.2,
    "max_or_width_percentile": 1.0,  # 1.0 disables the narrow-open filter
    "max_hold_bars": 120,
    "breakeven_at_r": 1.0,
    "trail_atr_mult": None,
}


class OpeningRangeBreakout(Strategy):
    @classmethod
    def build(cls, **params) -> OpeningRangeBreakout:
        p = {**DEFAULTS, **params}
        spec = StrategySpec(
            name="orb_breakout",
            description=f"{p['or_minutes']}-minute opening-range breakout",
            entry_conditions=(
                "opening_range_complete",
                "close_outside_opening_range",
                "break_is_fresh (previous close was inside)",
                f"volume_ratio >= {p['min_volume_ratio']}",
            ),
            invalidation_conditions=("close_back_inside_opening_range",),
            stop_loss=(
                f"entry -/+ {p['stop_atr_mult']} x ATR"
                if p["stop_basis"] == "atr"
                else "opposite side of the opening range"
            ),
            profit_target=f"{p['target_r']}R",
            filters=(
                f"or_width_percentile <= {p['max_or_width_percentile']}",
                "or_width > 0",
            ),
            max_trades_per_session=int(params.get("max_trades_per_session", 1)),
            # No entry is possible before the opening range has closed, so the earliest
            # entry follows the range length rather than being a second magic number that
            # can drift out of step with it.
            trading_hours=TradingHours(
                earliest_entry=(
                    datetime(2000, 1, 1, 9, 30) + timedelta(minutes=int(p["or_minutes"]))
                ).time(),
                latest_entry=time(15, 30),
                force_flat_at=time(15, 58),
            ),
            features=FeatureSpec(opening_range_minutes=int(p["or_minutes"])),
            params=p,
        )
        return cls(spec)

    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        p = self.spec.params
        or_high, or_low = ctx.f("or_high"), ctx.f("or_low")
        if or_high != or_high or or_low != or_low:  # NaN: the range has not closed yet
            ctx.reject(RejectReason.WARMUP_INCOMPLETE, "opening range not complete")
            return None
        if or_high <= or_low:
            ctx.reject(RejectReason.FILTER_VOLATILITY, "degenerate opening range")
            return None

        width_pct = ctx.f("or_width_percentile")
        max_width = float(p["max_or_width_percentile"])
        if max_width < 1.0 and (width_pct != width_pct or width_pct > max_width):
            ctx.reject(
                RejectReason.FILTER_VOLATILITY,
                f"opening range width percentile {width_pct:.2f} > {max_width}",
                or_width_percentile=width_pct,
            )
            return None

        volume_ratio = ctx.f("volume_ratio")
        if volume_ratio != volume_ratio or volume_ratio < float(p["min_volume_ratio"]):
            ctx.reject(
                RejectReason.FILTER_VOLUME,
                f"volume ratio {volume_ratio:.2f} < {p['min_volume_ratio']}",
                volume_ratio=volume_ratio,
            )
            return None

        close = float(ctx.bar["close"])
        prev = ctx.previous(1)
        prev_close = float(prev["close"]) if prev is not None else close

        if close > or_high and prev_close <= or_high:
            side, level = Side.BUY, or_high
        elif close < or_low and prev_close >= or_low:
            side, level = Side.SELL, or_low
        else:
            return None

        atr = ctx.f("atr")
        if p["stop_basis"] == "range":
            stop = or_low if side is Side.BUY else or_high
        else:
            stop = (
                close - float(p["stop_atr_mult"]) * atr
                if side is Side.BUY
                else close + float(p["stop_atr_mult"]) * atr
            )

        return self._make_intent(
            ctx,
            side,
            stop,
            target_price=self._target_from_r(ctx, side, stop, float(p["target_r"])),
            conditions=(
                "opening_range_complete",
                "close_above_or_high" if side is Side.BUY else "close_below_or_low",
                "break_is_fresh",
                "volume_confirms",
            ),
            features={
                "or_high": or_high, "or_low": or_low,
                "or_width": or_high - or_low, "volume_ratio": volume_ratio,
                "break_level": level,
            },
        )

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        or_high, or_low = ctx.f("or_high"), ctx.f("or_low")
        close = float(ctx.bar["close"])
        if position.is_long and close < or_low:
            return "close fell back through the opposite side of the opening range"
        if not position.is_long and close > or_high:
            return "close rose back through the opposite side of the opening range"
        return None
