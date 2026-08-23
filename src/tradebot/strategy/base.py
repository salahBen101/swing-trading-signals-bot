"""Strategy framework.

Two things this module exists to guarantee:

1. **A strategy cannot reach a broker.** It is constructed with a spec and nothing else,
   and `on_bar` returns an inert `OrderIntent`. There is no client, no session, no
   callback that sends anything. `tests/tradebot/test_no_bypass.py` asserts statically
   that nothing under `strategy/` imports `broker` or `execution`.

2. **A strategy cannot see the future.** The context exposes the current bar, the current
   feature row, and history *up to and including* the current bar. There is no accessor
   that returns a later row.

Beyond that, PROJECT_SPEC §5 requires every strategy to declare its rules as data. The
`StrategySpec` is that declaration: entry conditions, invalidation conditions, stop rule,
target rule, trading hours, filters and a hard per-session trade cap, all serialisable so
the journal can record exactly which rule set produced a trade.

Discretionary language is banned from execution logic. In practice that means every
condition is a named boolean recorded on the intent — `close_above_or_high`, not "looks
strong" — so a trade can always be replayed from its own record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time

import pandas as pd

from ..core.models import OrderIntent, Position, Rejection, Trade
from ..core.types import ExitReason, RejectReason, Side
from ..features.pipeline import FeatureSpec
from ..instruments.registry import InstrumentSpec


@dataclass(frozen=True, slots=True)
class TradingHours:
    """When a strategy may open a position, and when it must be flat.

    `force_flat_at` is not the strategy's choice at runtime — the engine enforces it — but
    it is declared here so the rule set is complete in one object.
    """

    earliest_entry: time = time(9, 35)
    latest_entry: time = time(15, 45)
    force_flat_at: time = time(15, 58)

    def may_enter(self, ts: datetime) -> bool:
        return self.earliest_entry <= ts.time() < self.latest_entry

    def must_be_flat(self, ts: datetime) -> bool:
        return ts.time() >= self.force_flat_at


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """The complete, declarative rule set. Serialisable; dumped into every backtest report."""

    name: str
    description: str
    entry_conditions: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    stop_loss: str
    profit_target: str
    filters: tuple[str, ...]
    max_trades_per_session: int
    trading_hours: TradingHours = field(default_factory=TradingHours)
    features: FeatureSpec = field(default_factory=FeatureSpec)
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        hours = out["trading_hours"]
        out["trading_hours"] = {k: str(v) for k, v in hours.items()}
        return out

    def summary(self) -> str:
        lines = [
            f"{self.name} — {self.description}",
            f"  entry:        {' AND '.join(self.entry_conditions)}",
            f"  filters:      {' AND '.join(self.filters) or '(none)'}",
            f"  invalidation: {' OR '.join(self.invalidation_conditions) or '(none)'}",
            f"  stop:         {self.stop_loss}",
            f"  target:       {self.profit_target}",
            f"  hours:        {self.trading_hours.earliest_entry}"
            f"–{self.trading_hours.latest_entry}"
            f" (flat by {self.trading_hours.force_flat_at})",
            f"  max trades:   {self.max_trades_per_session} per session",
        ]
        if self.params:
            lines.append(f"  params:       {self.params}")
        return "\n".join(lines)


@dataclass(slots=True)
class StrategyContext:
    """Everything a strategy may look at when bar `i` has just closed.

    History slices are built on demand rather than eagerly: materialising `frame[:i+1]` on
    every bar of an 898,000-bar backtest dominates the runtime, and most rules read only
    the current row.
    """

    i: int
    timestamp: datetime
    bar: pd.Series
    features: pd.Series
    instrument: InstrumentSpec
    equity: float
    session_date: date
    trades_this_session: int
    _frame: pd.DataFrame
    _bars: pd.DataFrame
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def bars(self) -> pd.DataFrame:
        """Price history through the current bar. Never further."""
        return self._bars.iloc[: self.i + 1]

    @property
    def feature_history(self) -> pd.DataFrame:
        return self._frame.iloc[: self.i + 1]

    def previous(self, n: int = 1) -> pd.Series | None:
        """The feature row `n` bars back, or None if it is before the start of data."""
        j = self.i - n
        return None if j < 0 else self._frame.iloc[j]

    def f(self, name: str, default: float = float("nan")) -> float:
        value = self.features.get(name, default)
        return default if value is None else float(value)

    def reject(self, reason: RejectReason, detail: str, **context) -> None:
        """Record why a signal did not fire.

        Rejections are first-class output, not debug noise: PROJECT_SPEC §9 requires the
        rejected signals and their reasons to be reproducible, and the dashboard shows
        them. A filter that silently returns None teaches nothing.
        """
        self.rejections.append(
            Rejection(
                timestamp=self.timestamp,
                reason=reason,
                detail=detail,
                stage="STRATEGY",
                instrument=self.instrument.symbol,
                context=context,
            )
        )


@dataclass(frozen=True, slots=True)
class ExitDecision:
    reason: ExitReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class StopUpdate:
    new_stop: float
    detail: str = ""


class Strategy(ABC):
    """Base class. Subclasses implement `_entry_signal` and optionally `_invalidated`.

    The base owns the mechanical gates every strategy shares — warmup, trading hours, the
    per-session trade cap — so that a subclass cannot forget one, and so those refusals are
    recorded uniformly.
    """

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec
        self._session_date: date | None = None
        self._trades_this_session = 0

    # -- identity -----------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def features(self) -> FeatureSpec:
        return self.spec.features

    def param(self, key: str, default=None):
        return self.spec.params.get(key, default)

    # -- lifecycle ----------------------------------------------------------------------
    def on_session_start(self, session: date) -> None:
        self._session_date = session
        self._trades_this_session = 0

    def on_trade_opened(self, position: Position) -> None:
        self._trades_this_session += 1

    def on_trade_closed(self, trade: Trade) -> None:
        return None

    @property
    def trades_this_session(self) -> int:
        return self._trades_this_session

    # -- entry --------------------------------------------------------------------------
    def on_bar(self, ctx: StrategyContext) -> OrderIntent | None:
        """Called only when flat. Returns an intent to open at the next bar's open."""
        if ctx.session_date != self._session_date:
            self.on_session_start(ctx.session_date)

        hours = self.spec.trading_hours
        if not hours.may_enter(ctx.timestamp):
            ctx.reject(
                RejectReason.OUTSIDE_TRADING_HOURS,
                f"{ctx.timestamp.time()} is outside "
                f"{hours.earliest_entry}–{hours.latest_entry}",
            )
            return None

        if self._trades_this_session >= self.spec.max_trades_per_session:
            ctx.reject(
                RejectReason.MAX_TRADES_PER_SESSION,
                f"already took {self._trades_this_session} of "
                f"{self.spec.max_trades_per_session} allowed trades today",
            )
            return None

        return self._entry_signal(ctx)

    @abstractmethod
    def _entry_signal(self, ctx: StrategyContext) -> OrderIntent | None:
        """The strategy's own rules. Return an intent or None."""

    # -- management ---------------------------------------------------------------------
    def manage(self, ctx: StrategyContext, position: Position) -> ExitDecision | StopUpdate | None:
        """Called each bar while a position is open, after the engine has checked the stop
        and target against this bar's range."""
        max_bars = self.param("max_hold_bars")
        if max_bars and position.bars_held >= int(max_bars):
            return ExitDecision(ExitReason.TIME_STOP, f"held {position.bars_held} bars")

        invalidation = self._invalidated(ctx, position)
        if invalidation is not None:
            return ExitDecision(ExitReason.INVALIDATION, invalidation)

        return self._trail_stop(ctx, position)

    def _invalidated(self, ctx: StrategyContext, position: Position) -> str | None:
        """Return a reason string when the trade's premise is gone, else None."""
        return None

    def _trail_stop(self, ctx: StrategyContext, position: Position) -> StopUpdate | None:
        """Shared stop management: move to break-even at `breakeven_at_r`, then trail by
        `trail_atr_mult` ATRs behind the best price seen.

        Both are opt-in per strategy. A stop is only ever moved in the direction that
        reduces risk — a ratchet that could widen a stop is a way to turn a small loss into
        a large one.
        """
        atr = ctx.f("atr")
        if not atr or atr != atr:  # NaN guard
            return None

        instrument = ctx.instrument
        mark = float(ctx.bar["close"])
        r_now = position.r_multiple_at(mark, instrument.multiplier)
        new_stop: float | None = None

        breakeven_at = self.param("breakeven_at_r")
        if breakeven_at and r_now >= float(breakeven_at):
            new_stop = position.entry_price

        trail_mult = self.param("trail_atr_mult")
        if trail_mult:
            offset = float(trail_mult) * atr
            trailed = (
                position.max_favorable_price - offset
                if position.is_long
                else position.max_favorable_price + offset
            )
            new_stop = (
                max(new_stop, trailed) if new_stop is not None and position.is_long
                else min(new_stop, trailed) if new_stop is not None
                else trailed
            )

        if new_stop is None:
            return None

        direction = -1 if position.is_long else +1
        new_stop = instrument.round_away(new_stop, direction=direction)

        improves = (
            new_stop > position.stop_price if position.is_long else new_stop < position.stop_price
        )
        if not improves:
            return None
        return StopUpdate(new_stop, detail=f"trail/breakeven at {r_now:.2f}R")

    # -- helpers for subclasses ---------------------------------------------------------
    def _make_intent(
        self,
        ctx: StrategyContext,
        side: Side,
        stop_price: float,
        *,
        conditions: tuple[str, ...],
        target_price: float | None = None,
        features: dict | None = None,
    ) -> OrderIntent | None:
        """Build an intent with tick-rounded, side-correct protective levels.

        Returns None (with a recorded rejection) when the stop would round onto the wrong
        side of the entry — which happens with a very tight ATR in a quiet market, and
        which would otherwise construct a position with negative risk.
        """
        instrument = ctx.instrument
        reference = instrument.round_to_tick(float(ctx.bar["close"]))
        # Round protective levels away from the entry, never toward it.
        stop = instrument.round_away(stop_price, direction=-1 if side is Side.BUY else +1)
        target = None if target_price is None else instrument.round_to_tick(target_price)

        wrong_side = (
            stop >= reference if side is Side.BUY else stop <= reference
        )
        if wrong_side:
            ctx.reject(
                RejectReason.INVALID_ORDER,
                f"stop {stop} is not on the protective side of {reference} for a {side.name}",
                stop=stop, reference=reference,
            )
            return None

        if target is not None:
            bad_target = target <= reference if side is Side.BUY else target >= reference
            if bad_target:
                ctx.reject(
                    RejectReason.INVALID_ORDER,
                    f"target {target} is not beyond {reference} for a {side.name}",
                    target=target, reference=reference,
                )
                return None

        payload = {"atr": ctx.f("atr"), "rsi": ctx.f("rsi"), "adx": ctx.f("adx")}
        payload.update(features or {})

        return OrderIntent(
            timestamp=ctx.timestamp,
            instrument=instrument.symbol,
            side=side,
            strategy=self.name,
            stop_price=stop,
            target_price=target,
            reference_price=reference,
            conditions=conditions,
            features=payload,
            max_hold_bars=self.param("max_hold_bars"),
        )

    def _target_from_r(self, ctx: StrategyContext, side: Side, stop: float, r: float) -> float:
        entry = float(ctx.bar["close"])
        risk = abs(entry - stop)
        return entry + risk * r if side is Side.BUY else entry - risk * r
