"""The signal and order lifecycle. Quote, order, fill, position and trade are distinct things.

The pilot collapsed all five into one dict, which is how a displayed quote became a booked fill
with nothing in between. Keeping them separate is what makes it possible to record an order that
was rejected, an order that filled at a worse price than the quote implied, or a signal that
never became an order at all - none of which the old structure could express.

    SIGNAL_DETECTED     the rules fired on a completed session close
          |
    PENDING_ENTRY       waiting for the next session's execution window
          |
    ORDER_SUBMITTED     sent to the broker; we have an order id
          |
    +-- FILLED / PARTIALLY_FILLED -> position exists
    +-- REJECTED        broker refused it
    +-- CANCELLED       we pulled it, or it expired unfilled
    +-- EXPIRED_UNFILLED the signal aged out before it could be executed

Exits mirror this. An exit signal detected at the close of session T submits in session T+1, for
the same reason entries do: the price that closes the trade cannot precede the decision to close
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalState(str, Enum):
    SIGNAL_DETECTED = "SIGNAL_DETECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED_UNFILLED = "EXPIRED_UNFILLED"


class ExitState(str, Enum):
    EXIT_SIGNAL_DETECTED = "EXIT_SIGNAL_DETECTED"
    PENDING_EXIT = "PENDING_EXIT"
    EXIT_ORDER_SUBMITTED = "EXIT_ORDER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


class RejectReason(str, Enum):
    """Every rejection is recorded with a machine-readable reason and surfaced on the dashboard.

    A skipped trade is information. The pilot discarded it, which meant there was no way to tell
    a day with no signals from a day where every signal was refused.
    """

    RISK_PREMIUM_EXCEEDED = "REJECTED_RISK_LIMIT"
    RISK_MAX_OPEN_POSITIONS = "REJECTED_MAX_OPEN_POSITIONS"
    RISK_MAX_TRADES_PER_SESSION = "REJECTED_MAX_TRADES_PER_SESSION"
    RISK_DAILY_LOSS_REACHED = "REJECTED_DAILY_LOSS_LIMIT"
    RISK_TICKER_CONCENTRATION = "REJECTED_TICKER_CONCENTRATION"
    RISK_SECTOR_CONCENTRATION = "REJECTED_SECTOR_CONCENTRATION"
    RISK_AGGREGATE_PREMIUM = "REJECTED_AGGREGATE_PREMIUM"
    INSUFFICIENT_CASH = "REJECTED_INSUFFICIENT_CASH"

    QUOTE_MISSING = "REJECTED_QUOTE_MISSING"
    QUOTE_STALE = "REJECTED_QUOTE_STALE"
    QUOTE_CROSSED = "REJECTED_QUOTE_CROSSED"
    QUOTE_ZERO_BID = "REJECTED_QUOTE_ZERO_BID"
    QUOTE_ZERO_ASK = "REJECTED_QUOTE_ZERO_ASK"
    QUOTE_SPREAD_TOO_WIDE = "REJECTED_SPREAD_TOO_WIDE"
    QUOTE_NO_SIZE = "REJECTED_NO_SIZE"
    QUOTE_INVALID_TIMESTAMP = "REJECTED_INVALID_QUOTE_TIMESTAMP"

    CONTRACT_NONE_SUITABLE = "REJECTED_NO_SUITABLE_CONTRACT"
    CONTRACT_EXPIRED = "REJECTED_CONTRACT_EXPIRED"
    CONTRACT_ADJUSTED = "REJECTED_ADJUSTED_CONTRACT"
    CONTRACT_DTE_OUT_OF_RANGE = "REJECTED_DTE_OUT_OF_RANGE"
    CONTRACT_DELTA_OUT_OF_RANGE = "REJECTED_DELTA_OUT_OF_RANGE"
    CONTRACT_LOW_OPEN_INTEREST = "REJECTED_LOW_OPEN_INTEREST"
    CONTRACT_METADATA_MISSING = "REJECTED_CONTRACT_METADATA_MISSING"
    CONTRACT_MULTIPLIER_UNKNOWN = "REJECTED_MULTIPLIER_UNKNOWN"

    BROKER_UNAVAILABLE = "REJECTED_BROKER_UNAVAILABLE"
    DATA_QUALITY = "REJECTED_DATA_QUALITY"
    SIGNAL_EXPIRED = "REJECTED_SIGNAL_EXPIRED"
    EXPERIMENT_NOT_ACTIVE = "REJECTED_EXPERIMENT_NOT_ACTIVE"


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: RejectReason
    detail: str
    ticker: str = ""
    occurred_at: str = ""
    context: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


# Legal transitions. Anything not listed is a bug, and asserting that here means the bug
# surfaces at the transition rather than three steps later as an impossible ledger row.
ENTRY_TRANSITIONS: dict[SignalState, frozenset[SignalState]] = {
    SignalState.SIGNAL_DETECTED: frozenset({
        SignalState.PENDING_ENTRY, SignalState.REJECTED}),
    SignalState.PENDING_ENTRY: frozenset({
        SignalState.ORDER_SUBMITTED, SignalState.REJECTED,
        SignalState.EXPIRED_UNFILLED, SignalState.CANCELLED}),
    SignalState.ORDER_SUBMITTED: frozenset({
        SignalState.PARTIALLY_FILLED, SignalState.FILLED,
        SignalState.REJECTED, SignalState.CANCELLED}),
    SignalState.PARTIALLY_FILLED: frozenset({
        SignalState.FILLED, SignalState.CANCELLED, SignalState.PARTIALLY_FILLED}),
    SignalState.FILLED: frozenset(),
    SignalState.REJECTED: frozenset(),
    SignalState.CANCELLED: frozenset(),
    SignalState.EXPIRED_UNFILLED: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


def assert_transition(current: SignalState, nxt: SignalState) -> None:
    allowed = ENTRY_TRANSITIONS.get(current, frozenset())
    if nxt not in allowed:
        raise IllegalTransition(
            f"{current.value} -> {nxt.value} is not a legal transition. "
            f"Allowed: {sorted(s.value for s in allowed) or 'none (terminal state)'}"
        )
