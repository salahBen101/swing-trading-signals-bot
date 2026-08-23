"""Value objects passed between modules.

Two properties matter more than the field lists:

1. **`OrderIntent` is inert.** It is what a strategy returns. It has no `send()`, no
   broker handle, and no way to become an `Order` without passing through the risk
   engine. That is the first of the three layers described in PROJECT_SPEC §3.1.

2. **Everything is frozen** except `Position`, which genuinely mutates as a trade
   breathes (stop ratchets, bars accumulate, excursions widen). Freezing the rest means a
   journalled record cannot be edited after the fact by a later stage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .types import (
    ExitReason,
    OrderPurpose,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed OHLCV bar.

    Construction validates the shape, so a malformed bar cannot travel further into the
    system than the boundary that produced it. `timestamp` is the bar's **open** time and
    must be timezone-aware; a naive datetime is a bug, not something to coerce.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"bar timestamp must be timezone-aware, got {self.timestamp!r}")
        if self.high < self.low:
            raise ValueError(f"high < low at {self.timestamp}: H={self.high} L={self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open outside [low, high] at {self.timestamp}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close outside [low, high] at {self.timestamp}")
        if self.volume < 0:
            raise ValueError(f"negative volume at {self.timestamp}: {self.volume}")

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_up(self) -> bool:
        return self.close >= self.open


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """What a strategy wants, expressed as data it cannot act on.

    `stop_price` is required: a strategy that cannot say where it is wrong does not get to
    open a position, because position sizing is derived from the stop distance and a
    missing stop would mean unbounded risk. `target_price` is optional — some exits are
    managed rather than resting.

    `conditions` records the named predicates that fired. It exists so the journal can
    answer "why was this trade taken" without re-running anything, and so that
    PROJECT_SPEC §5's ban on discretionary language is checkable: every entry here is the
    name of a boolean the strategy actually evaluated.
    """

    timestamp: datetime
    instrument: str
    side: Side
    strategy: str
    stop_price: float
    target_price: float | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reference_price: float = 0.0
    conditions: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    max_hold_bars: int | None = None
    intent_id: str = field(default_factory=lambda: new_id("int"))

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("intent timestamp must be timezone-aware")
        if self.stop_price <= 0:
            raise ValueError("stop_price must be positive")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type} intent requires a limit_price")
        # A stop on the wrong side of the reference is almost always a sign-flip bug in a
        # strategy, and it would size the position off a negative distance.
        if self.reference_price > 0:
            if self.side is Side.BUY and self.stop_price >= self.reference_price:
                raise ValueError(
                    f"long stop {self.stop_price} must sit below reference {self.reference_price}"
                )
            if self.side is Side.SELL and self.stop_price <= self.reference_price:
                raise ValueError(
                    f"short stop {self.stop_price} must sit above reference {self.reference_price}"
                )

    @property
    def stop_distance(self) -> float:
        return abs(self.reference_price - self.stop_price)

    def fingerprint(self) -> tuple:
        """Identity for duplicate detection.

        Deliberately excludes `intent_id` and the feature dump: two strategy evaluations
        that ask for the same thing at the same instant *are* the same order, however they
        were arrived at.
        """
        return (
            self.timestamp,
            self.instrument,
            self.side,
            self.strategy,
            round(self.stop_price, 6),
            None if self.target_price is None else round(self.target_price, 6),
            self.order_type,
        )


@dataclass(frozen=True, slots=True)
class Order:
    """A concrete order, minted by the risk engine and sent by the execution engine.

    Only the risk engine constructs these from an intent (see `risk.limits`). The
    `quantity` is the risk engine's answer, never the strategy's.
    """

    order_id: str
    timestamp: datetime
    instrument: str
    side: Side
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    intent_id: str | None = None
    strategy: str = ""
    purpose: OrderPurpose = OrderPurpose.ENTRY
    # Orders sharing an OCO group cancel each other on fill. Used for the protective
    # stop/target pair, which must never both execute and leave a reversed position.
    oco_group: str | None = None
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, OrderPurpose):
            try:
                object.__setattr__(self, "purpose", OrderPurpose(self.purpose))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown order purpose: {self.purpose!r}") from exc
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type} order requires a limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type} order requires a stop_price")

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.quantity - self.filled_quantity)

    def binding_fields(self) -> tuple:
        """The fields a risk token commits to.

        Anything that changes the economic meaning of the order is in here. Anything that
        does not — the broker id assigned later, the evolving status — is out, so that
        normal lifecycle updates do not invalidate a token that has not yet been spent.
        """
        return (
            self.order_id,
            self.timestamp.isoformat(timespec="microseconds"),
            self.instrument,
            int(self.side),
            self.quantity,
            self.order_type.value,
            self.limit_price,
            self.stop_price,
            self.time_in_force.value,
            self.intent_id,
            self.strategy,
            self.purpose.value,
            self.oco_group,
        )

    def with_status(self, status: OrderStatus, **kwargs: Any) -> Order:
        return replace(self, status=status, **kwargs)


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    timestamp: datetime
    instrument: str
    side: Side
    quantity: int
    price: float
    commission_usd: float = 0.0
    slippage_points: float = 0.0
    broker_fill_id: str | None = None
    is_partial: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if self.price <= 0:
            raise ValueError("fill price must be positive")
        if self.slippage_points < 0:
            raise ValueError("slippage_points must describe adverse slippage and be non-negative")


@dataclass(slots=True)
class Position:
    """An open position. Mutable by design — this is the one object that evolves."""

    instrument: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: datetime
    strategy: str
    initial_stop: float
    stop_price: float
    target_price: float | None = None
    entry_commission_usd: float = 0.0
    entry_slippage_usd: float = 0.0
    # Cumulative fill accounting. ``quantity`` is the contracts still open and therefore
    # shrinks on a scale-out; these fields retain the complete round trip so the eventual
    # Trade and the risk engine do not silently forget earlier partial exits.
    entered_quantity: int = 0
    closed_quantity: int = 0
    realized_gross_pnl_usd: float = 0.0
    exit_commission_usd: float = 0.0
    exit_slippage_usd: float = 0.0
    exit_notional: float = 0.0
    risk_per_contract_points: float = 0.0
    bars_held: int = 0
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    entry_intent_id: str | None = None
    entry_conditions: tuple[str, ...] = ()
    entry_features: dict[str, float] = field(default_factory=dict)
    max_hold_bars: int | None = None
    position_id: str = field(default_factory=lambda: new_id("pos"))

    def __post_init__(self) -> None:
        if self.entered_quantity == 0:
            self.entered_quantity = self.quantity
        if self.max_favorable_price == 0.0:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price == 0.0:
            self.max_adverse_price = self.entry_price

    @property
    def signed_quantity(self) -> int:
        return self.quantity * int(self.side)

    @property
    def is_long(self) -> bool:
        return self.side is Side.BUY

    def unrealized_points(self, mark: float) -> float:
        return (mark - self.entry_price) * int(self.side)

    def unrealized_usd(self, mark: float, multiplier: float) -> float:
        return self.unrealized_points(mark) * self.quantity * multiplier

    def risk_usd(self, multiplier: float) -> float:
        return self.risk_per_contract_points * self.quantity * multiplier

    def r_multiple_at(self, mark: float, multiplier: float) -> float:
        risk = self.risk_usd(multiplier)
        if risk <= 0:
            return 0.0
        return self.unrealized_usd(mark, multiplier) / risk

    def observe(self, bar: Bar) -> None:
        """Fold one bar into the position's excursion record."""
        self.bars_held += 1
        if self.is_long:
            self.max_favorable_price = max(self.max_favorable_price, bar.high)
            self.max_adverse_price = min(self.max_adverse_price, bar.low)
        else:
            self.max_favorable_price = min(self.max_favorable_price, bar.low)
            self.max_adverse_price = max(self.max_adverse_price, bar.high)


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip. This is the unit analytics consumes."""

    trade_id: str
    instrument: str
    strategy: str
    side: Side
    quantity: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: ExitReason
    gross_pnl_usd: float
    commission_usd: float
    net_pnl_usd: float
    r_multiple: float
    bars_held: int
    initial_stop: float
    target_price: float | None = None
    max_favorable_excursion_points: float = 0.0
    max_adverse_excursion_points: float = 0.0
    entry_conditions: tuple[str, ...] = ()
    entry_features: dict[str, float] = field(default_factory=dict)
    position_id: str | None = None
    # Attribution only: actual fill prices already include this execution cost, so it is
    # deliberately *not* subtracted from gross P&L a second time.
    slippage_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.slippage_usd < 0:
            raise ValueError("slippage_usd must describe adverse slippage and be non-negative")

    @property
    def is_winner(self) -> bool:
        return self.net_pnl_usd > 0

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass(frozen=True, slots=True)
class Rejection:
    """A refusal, from any layer. Journalled so the dashboard can show what did not happen."""

    timestamp: datetime
    reason: RejectReason
    detail: str
    stage: str  # STRATEGY | RISK | GUARD | EXECUTION | BROKER
    instrument: str = ""
    strategy: str = ""
    intent_id: str | None = None
    order_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    timestamp: datetime
    equity: float
    realized_pnl_today: float
    unrealized_pnl: float
    open_positions: int
    trades_today: int
