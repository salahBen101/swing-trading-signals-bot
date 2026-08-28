"""Quote validation. The gate that decides whether market data may be traded on at all.

The governing rule is that a missing or untrustworthy price produces NO TRADE, never an invented
one. The pilot violated this in a specific, quiet way: when a re-quote failed it left the
previous price in the position and carried on, so the account reported an equity built on a mark
that might have been days old and looked entirely healthy while doing it.

Every check here is a reason to skip. None of them is a reason to substitute a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .states import RejectReason, Rejection


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One option quote, with the provenance needed to judge whether it can be traded on."""

    symbol: str
    underlying: str
    expiry: str
    strike: float
    bid: float
    ask: float
    bid_size: int | None
    ask_size: int | None
    quote_timestamp: datetime | None      # when the market produced it
    data_received_timestamp: datetime     # when we received it
    open_interest: int | None = None
    volume: int | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    contract_multiplier: int | None = 100
    is_adjusted: bool = False

    @property
    def mid(self) -> float | None:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> float | None:
        m, s = self.mid, self.spread
        if m and s is not None and m > 0:
            return s / m * 100
        return None

    def quote_age(self, now: datetime | None = None) -> timedelta | None:
        """How old the quote is. None when the venue gave no timestamp - which is itself a
        rejection reason, because an age that cannot be computed cannot be checked."""
        if self.quote_timestamp is None:
            return None
        now = now or datetime.now(timezone.utc)
        ts = self.quote_timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc) - ts.astimezone(timezone.utc)


def validate_quote(
    q: OptionQuote | None,
    *,
    max_age_seconds: int,
    max_spread_pct: float,
    min_open_interest: int,
    require_size: bool = False,
    now: datetime | None = None,
    ticker: str = "",
) -> Rejection | None:
    """Returns a Rejection if the quote may not be traded on, else None.

    Ordered cheapest-first, and deliberately strict: `ask < bid` is a crossed book, which means
    the feed is broken or the data is mid-update, not that free money is available.
    """
    now = now or datetime.now(timezone.utc)
    tk = ticker or (q.underlying if q else "")

    if q is None:
        return Rejection(RejectReason.QUOTE_MISSING, "no quote returned by the data source",
                         ticker=tk, occurred_at=now.isoformat())

    if q.contract_multiplier is None or q.contract_multiplier <= 0:
        return Rejection(RejectReason.CONTRACT_MULTIPLIER_UNKNOWN,
                         "contract multiplier unknown; premium at risk cannot be computed",
                         ticker=tk, occurred_at=now.isoformat())

    if q.is_adjusted:
        return Rejection(RejectReason.CONTRACT_ADJUSTED,
                         "adjusted contract - deliverable is not 100 shares of the underlying",
                         ticker=tk, occurred_at=now.isoformat())

    if q.quote_timestamp is None:
        return Rejection(RejectReason.QUOTE_INVALID_TIMESTAMP,
                         "quote carries no timestamp, so its age cannot be established",
                         ticker=tk, occurred_at=now.isoformat())

    age = q.quote_age(now)
    if age is None or age.total_seconds() < 0:
        return Rejection(RejectReason.QUOTE_INVALID_TIMESTAMP,
                         f"quote timestamp is in the future ({q.quote_timestamp})",
                         ticker=tk, occurred_at=now.isoformat())
    if age.total_seconds() > max_age_seconds:
        return Rejection(RejectReason.QUOTE_STALE,
                         f"quote is {age.total_seconds():.0f}s old, limit is {max_age_seconds}s",
                         ticker=tk, occurred_at=now.isoformat(),
                         context={"quote_age_s": age.total_seconds()})

    if q.bid is None or q.bid <= 0:
        return Rejection(RejectReason.QUOTE_ZERO_BID,
                         f"bid is {q.bid} - nothing is willing to buy this contract",
                         ticker=tk, occurred_at=now.isoformat())
    if q.ask is None or q.ask <= 0:
        return Rejection(RejectReason.QUOTE_ZERO_ASK, f"ask is {q.ask}",
                         ticker=tk, occurred_at=now.isoformat())
    if q.ask < q.bid:
        return Rejection(RejectReason.QUOTE_CROSSED,
                         f"crossed book: bid {q.bid} > ask {q.ask}; the feed is not trustworthy",
                         ticker=tk, occurred_at=now.isoformat())

    if require_size:
        if not q.bid_size or not q.ask_size or q.bid_size <= 0 or q.ask_size <= 0:
            return Rejection(RejectReason.QUOTE_NO_SIZE,
                             f"no displayed size (bid {q.bid_size}, ask {q.ask_size})",
                             ticker=tk, occurred_at=now.isoformat())

    sp = q.spread_pct
    if sp is None:
        return Rejection(RejectReason.QUOTE_MISSING, "spread could not be computed",
                         ticker=tk, occurred_at=now.isoformat())
    if sp > max_spread_pct:
        return Rejection(RejectReason.QUOTE_SPREAD_TOO_WIDE,
                         f"spread {sp:.1f}% exceeds the {max_spread_pct:.1f}% limit",
                         ticker=tk, occurred_at=now.isoformat(), context={"spread_pct": sp})

    if q.open_interest is not None and q.open_interest < min_open_interest:
        return Rejection(RejectReason.CONTRACT_LOW_OPEN_INTEREST,
                         f"open interest {q.open_interest} below {min_open_interest}",
                         ticker=tk, occurred_at=now.isoformat())

    return None


def validate_contract_terms(
    q: OptionQuote,
    *,
    as_of: datetime,
    min_dte: int,
    max_dte: int,
    min_delta: float,
    max_delta: float,
    ticker: str = "",
) -> Rejection | None:
    """Contract-level gates that are about the instrument rather than its current quote."""
    tk = ticker or q.underlying
    try:
        exp = datetime.strptime(q.expiry, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return Rejection(RejectReason.CONTRACT_METADATA_MISSING,
                         f"expiry {q.expiry!r} is not a usable date",
                         ticker=tk, occurred_at=as_of.isoformat())

    dte = (exp - as_of.date()).days
    if dte <= 0:
        return Rejection(RejectReason.CONTRACT_EXPIRED, f"contract expired on {q.expiry}",
                         ticker=tk, occurred_at=as_of.isoformat())
    if not (min_dte <= dte <= max_dte):
        return Rejection(RejectReason.CONTRACT_DTE_OUT_OF_RANGE,
                         f"{dte} DTE is outside the {min_dte}-{max_dte} window",
                         ticker=tk, occurred_at=as_of.isoformat(), context={"dte": dte})

    if q.delta is None:
        return Rejection(RejectReason.CONTRACT_METADATA_MISSING,
                         "delta unavailable; the contract cannot be validated against policy",
                         ticker=tk, occurred_at=as_of.isoformat())
    if not (min_delta <= q.delta <= max_delta):
        return Rejection(RejectReason.CONTRACT_DELTA_OUT_OF_RANGE,
                         f"delta {q.delta:.2f} outside the {min_delta:.2f}-{max_delta:.2f} window",
                         ticker=tk, occurred_at=as_of.isoformat(), context={"delta": q.delta})

    return None
