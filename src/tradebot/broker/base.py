"""Broker adapter protocol and error taxonomy.

The protocol is deliberately narrow. Everything a strategy might want a broker to do —
bracket management, trailing logic, scaling — is done in the execution engine instead, so
that a backtest and a live run take the same code path and can only differ in fill
mechanics. A wide broker interface is how backtest/live divergence gets in.

The error taxonomy exists so callers can make retry decisions without string-matching. A
timeout and a rejection are both "the order did not go through", and treating them the same
way is how you end up sending an order twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from ..core.models import Fill, Order
from ..core.types import OrderStatus, Side


# --------------------------------------------------------------------------- errors


class BrokerError(Exception):
    """Base for anything the broker layer raises."""

    retryable = False


class BrokerConnectionError(BrokerError):
    """The transport failed: socket closed, DNS, connection refused."""

    retryable = True


class BrokerTimeout(BrokerError):
    """No response in time.

    **Retryable only with care.** The order may or may not have reached the exchange, so
    the correct recovery is to reconcile against `get_orders()` before resending, never to
    blindly repeat the request.
    """

    retryable = True


class BrokerAuthError(BrokerError):
    """Credentials rejected or the session expired. Retrying without re-auth is pointless."""

    retryable = False


class BrokerRateLimitError(BrokerError):
    """Too many requests. Retryable after the indicated delay."""

    retryable = True

    def __init__(self, message: str, retry_after_seconds: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class OrderRejected(BrokerError):
    """The broker refused the order on its own terms (margin, symbol, size, session)."""

    retryable = False


class LiveTradingDisabled(BrokerError):
    """Something tried to reach a live endpoint. Not reachable in v1."""

    retryable = False


class NotConnected(BrokerError):
    retryable = True


# --------------------------------------------------------------------------- events


class EventKind(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTED = "RECONNECTED"


@dataclass(frozen=True, slots=True)
class BrokerEvent:
    kind: EventKind
    timestamp: datetime
    order_id: str = ""
    broker_order_id: str = ""
    fill: Fill | None = None
    detail: str = ""


# --------------------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class OrderAck:
    """Immediate response to a submission. An ack is not a fill."""

    order_id: str
    broker_order_id: str
    status: OrderStatus
    accepted_at: datetime
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BrokerOrderState:
    broker_order_id: str
    order_id: str
    status: OrderStatus
    filled_quantity: int
    average_fill_price: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """The broker's view of a position. This is the source of truth on reconciliation.

    `quantity` is signed: positive is long, negative is short, zero is flat.
    """

    instrument: str
    quantity: int
    average_price: float

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def side(self) -> Side | None:
        if self.quantity > 0:
            return Side.BUY
        if self.quantity < 0:
            return Side.SELL
        return None


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    account_id: str
    equity: float
    cash: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    currency: str = "USD"
    is_paper: bool = True
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """Execution-safety semantics implemented by an adapter's venue boundary.

    These flags describe behaviour that is actually implemented and verified.  They are
    not aspirations: an adapter must advertise ``False`` until the corresponding native
    order semantics and terminal-state reconciliation exist. They are necessary evidence,
    not permission by themselves; the guarded execution path may impose a hard stage block
    until it actually consumes those semantics atomically.
    """

    external_execution: bool
    server_side_oco: bool
    reduce_only_or_close_position: bool
    authoritative_cancel_status: bool
    exact_terminal_order_history: bool = False
    authoritative_session_execution_history: bool = False
    account_owner_fencing: bool = False

    @property
    def stage_2_protection_safe(self) -> bool:
        return (
            self.server_side_oco
            and self.reduce_only_or_close_position
            and self.authoritative_cancel_status
        )

    @property
    def stage_2_recovery_safe(self) -> bool:
        """Whether crash recovery and concurrent ownership are proven for paper trading."""

        return (
            self.stage_2_protection_safe
            and self.exact_terminal_order_history
            and self.authoritative_session_execution_history
            and self.account_owner_fencing
        )


# --------------------------------------------------------------------------- protocol


@runtime_checkable
class BrokerAdapter(Protocol):
    """What every broker must provide.

    Note what is *absent*: no `place_bracket`, no `modify`, no strategy hooks. Adding those
    would move decision-making into the adapter, where it could differ between the
    simulator and a live venue.
    """

    name: str
    is_paper: bool
    execution_route: str
    capabilities: BrokerCapabilities

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def place_order(self, order: Order) -> OrderAck: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def get_orders(self) -> list[BrokerOrderState]: ...

    def get_positions(self) -> list[BrokerPosition]: ...

    def get_account(self) -> BrokerAccount: ...

    def poll_events(self) -> list[BrokerEvent]: ...
