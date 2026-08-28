"""Inert broker-state snapshot used by the final entry-risk verification.

Broker adapters own transport-specific models.  The risk layer instead consumes this
small immutable projection, which contains only facts needed to refuse unsafe entries.
It is intentionally constructible without a broker dependency so backtests and focused
tests exercise the same reconciliation boundary as a future live adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.types import OrderStatus


@dataclass(frozen=True, slots=True)
class BrokerRiskAccount:
    account_id: str
    equity: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    currency: str
    is_paper: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "equity": self.equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "currency": self.currency,
            "is_paper": self.is_paper,
        }


@dataclass(frozen=True, slots=True)
class BrokerRiskPosition:
    instrument: str
    quantity: int
    average_price: float

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "quantity": self.quantity,
            "average_price": self.average_price,
        }


@dataclass(frozen=True, slots=True)
class BrokerRiskOrder:
    order_id: str
    broker_order_id: str
    status: OrderStatus
    filled_quantity: int
    average_fill_price: float

    @property
    def is_working(self) -> bool:
        return isinstance(self.status, OrderStatus) and self.status.is_working

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "status": self.status.value if isinstance(self.status, OrderStatus) else str(self.status),
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
        }


@dataclass(frozen=True, slots=True)
class AuthoritativeBrokerSnapshot:
    """One complete adapter read, captured immediately before entry verification."""

    read_started_at: datetime
    captured_at: datetime
    broker_name: str
    broker_is_paper: bool
    execution_route: str
    account: BrokerRiskAccount
    positions: tuple[BrokerRiskPosition, ...]
    orders: tuple[BrokerRiskOrder, ...]

    @property
    def nonflat_positions(self) -> tuple[BrokerRiskPosition, ...]:
        return tuple(position for position in self.positions if not position.is_flat)

    @property
    def working_orders(self) -> tuple[BrokerRiskOrder, ...]:
        return tuple(order for order in self.orders if order.is_working)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_started_at": self.read_started_at.isoformat(),
            "captured_at": self.captured_at.isoformat(),
            "broker_name": self.broker_name,
            "broker_is_paper": self.broker_is_paper,
            "execution_route": self.execution_route,
            "account": self.account.to_dict(),
            "positions": [position.to_dict() for position in self.positions],
            "orders": [order.to_dict() for order in self.orders],
        }
