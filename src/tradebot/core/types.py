"""Enumerations shared across every module.

These live in `core` precisely so that no module needs to import another module's
vocabulary. `strategy` and `broker` both talk about a `Side`, and neither imports the
other to say so.
"""

from __future__ import annotations

from enum import Enum, IntEnum


class Side(IntEnum):
    """Order direction, and by extension position direction.

    The integer value is the sign of the exposure, which makes P&L arithmetic read
    directly: ``(exit - entry) * side * qty * multiplier``. A *position* whose side is
    ``BUY`` is long; one whose side is ``SELL`` is short. There is deliberately no second
    long/short enum to keep out of sync with this one.
    """

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def is_long(self) -> bool:
        return self is Side.BUY


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderPurpose(str, Enum):
    """Closed vocabulary for an order's role in the position lifecycle.

    Purpose is part of the signed order identity.  Keeping it closed prevents a typo or a
    caller-chosen string from accidentally entering a privileged exit/protection path.
    """

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    FLATTEN = "FLATTEN"
    STOP = "STOP"
    TARGET = "TARGET"


class OrderStatus(str, Enum):
    """Order lifecycle.

    ``PENDING`` is local-only: the order exists in our process but the broker has not
    acknowledged it. Anything that can leave an order stranded (a timeout, a crash) leaves
    it PENDING, which is what recovery looks for.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    # A cancel command was acknowledged, but the venue has not yet reported a terminal
    # state.  Treating that command acknowledgement as CANCELLED can make the engine drop
    # its only handle to a still-working protective stop.
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES

    @property
    def is_working(self) -> bool:
        return self in _WORKING_STATUSES


_TERMINAL_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)
_WORKING_STATUSES = frozenset(
    {
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCEL_REQUESTED,
    }
)


class ExitReason(str, Enum):
    """Why a position was closed.

    A stop that has been ratcheted away from its initial level is reported as
    ``TRAILING_STOP`` rather than ``STOP_LOSS``: trailing exits are frequently profitable,
    and collapsing both into one label makes trade attribution unreadable.
    """

    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_STOP = "TIME_STOP"
    INVALIDATION = "INVALIDATION"
    SESSION_CLOSE = "SESSION_CLOSE"
    RISK_HALT = "RISK_HALT"
    KILL_SWITCH = "KILL_SWITCH"
    STALE_DATA = "STALE_DATA"
    MANUAL = "MANUAL"


class RejectReason(str, Enum):
    """Machine-readable rejection codes.

    Every refusal anywhere in the system resolves to one of these, so the dashboard's
    "rejected signals" panel and any later attribution work can group without parsing
    prose. The human sentence rides alongside in ``Rejection.detail``.
    """

    # --- risk limits ---
    MAX_RISK_PER_TRADE = "MAX_RISK_PER_TRADE"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_DAILY_LOSS_R = "MAX_DAILY_LOSS_R"
    MAX_TRADES_PER_DAY = "MAX_TRADES_PER_DAY"
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    CONSECUTIVE_LOSS_COOLDOWN = "CONSECUTIVE_LOSS_COOLDOWN"
    TRAILING_DRAWDOWN = "TRAILING_DRAWDOWN"
    SIZE_BELOW_MINIMUM = "SIZE_BELOW_MINIMUM"

    # --- session / state ---
    OUTSIDE_TRADING_HOURS = "OUTSIDE_TRADING_HOURS"
    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    MAX_TRADES_PER_SESSION = "MAX_TRADES_PER_SESSION"
    HALTED_FOR_SESSION = "HALTED_FOR_SESSION"

    # --- integrity / safety ---
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_ALREADY_USED = "TOKEN_ALREADY_USED"
    TOKEN_BINDING_MISMATCH = "TOKEN_BINDING_MISMATCH"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    INVALID_ORDER = "INVALID_ORDER"
    BROKER_ERROR = "BROKER_ERROR"
    LIVE_TRADING_DISABLED = "LIVE_TRADING_DISABLED"

    # --- strategy-side filters (not errors; recorded for attribution) ---
    FILTER_REGIME = "FILTER_REGIME"
    FILTER_VOLATILITY = "FILTER_VOLATILITY"
    FILTER_VOLUME = "FILTER_VOLUME"
    FILTER_TREND = "FILTER_TREND"
    WARMUP_INCOMPLETE = "WARMUP_INCOMPLETE"


class BotState(str, Enum):
    """Top-level runner state, surfaced verbatim on the dashboard."""

    STARTING = "STARTING"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    IN_POSITION = "IN_POSITION"
    HALTED = "HALTED"
    KILLED = "KILLED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class HealthStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class TradingMode(str, Enum):
    """Which broker environment the runner is pointed at.

    ``LIVE`` exists as a value so that configuration can *name* it and the system can
    refuse it explicitly. Nothing in v1 accepts it.
    """

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"
