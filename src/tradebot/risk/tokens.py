"""Risk approval tokens.

This is layer 2 of the non-bypass property in PROJECT_SPEC §3.1. The risk engine does not
merely *say* yes; it mints a token bound to the exact order it approved. The broker's
signature requires one, so an unapproved order is unrepresentable rather than merely
discouraged.

Three properties make the token worth having:

* **Bound.** The signature covers the order's economic fields *and* the risk facts the
  approval was based on. Changing the quantity, the side, the price or the claimed risk
  after approval invalidates it.
* **Unforgeable.** The signature is an HMAC under a secret generated per engine instance.
  Nothing outside the engine can produce a valid token, so "just construct one" is not a
  workaround available to a strategy or to a careless caller.
* **Single-use.** Spending a token consumes it. A replayed approval is rejected, which is
  where duplicate-order protection comes from for free.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..core.models import Order, new_id
from ..core.types import OrderPurpose, RejectReason


class AuthorizationKind(str, Enum):
    """The privilege a token grants, independently of caller-controlled order metadata."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    PROTECTIVE = "PROTECTIVE"


@dataclass(frozen=True, slots=True)
class RiskToken:
    """Proof that the risk engine approved this exact order, now.

    `risk_usd` and `stop_price` are carried so the broker-side re-validation can check the
    approval's own arithmetic rather than trusting a number recomputed by the caller. They
    are inside the signature, so they cannot be edited in transit.
    """

    token_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str
    risk_usd: float
    stop_price: float
    authorization_kind: AuthorizationKind

    def __post_init__(self) -> None:
        if not isinstance(self.authorization_kind, AuthorizationKind):
            try:
                object.__setattr__(
                    self,
                    "authorization_kind",
                    AuthorizationKind(self.authorization_kind),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"unknown authorization kind: {self.authorization_kind!r}"
                ) from exc

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class TokenError(Exception):
    """Raised when a token fails verification. Carries a machine-readable reason."""

    def __init__(self, reason: RejectReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class TokenMint:
    """Issues and verifies tokens for one risk engine instance.

    The secret lives only in memory and is regenerated on every start, which also means a
    token cannot survive a restart — correct, because the risk state it attested to did not
    survive either.
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._secret = secrets.token_bytes(32)
        self._ttl = timedelta(seconds=ttl_seconds)
        self._spent: set[str] = set()
        self.issued_count = 0

    def _sign(
        self,
        order: Order,
        *,
        token_id: str,
        issued_at: datetime,
        expires_at: datetime,
        authorization_kind: AuthorizationKind,
        risk_usd: float,
        stop_price: float,
    ) -> str:
        # JSON gives the payload an unambiguous structure.  A delimiter-joined string can
        # collide when a free-text field itself contains the delimiter, and rounding money
        # or prices before signing leaves small mutations unauthenticated.
        payload = json.dumps(
            {
                "schema": 1,
                "token_id": token_id,
                "issued_at": issued_at.isoformat(timespec="microseconds"),
                "expires_at": expires_at.isoformat(timespec="microseconds"),
                "authorization_kind": authorization_kind.value,
                "order": list(order.binding_fields()),
                "risk_usd": risk_usd,
                "protective_stop": stop_price,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_kind(order: Order, authorization_kind: AuthorizationKind) -> None:
        allowed = {
            AuthorizationKind.ENTRY: frozenset({OrderPurpose.ENTRY}),
            AuthorizationKind.EXIT: frozenset({OrderPurpose.EXIT, OrderPurpose.FLATTEN}),
            AuthorizationKind.PROTECTIVE: frozenset(
                {OrderPurpose.STOP, OrderPurpose.TARGET}
            ),
        }
        if order.purpose not in allowed[authorization_kind]:
            raise ValueError(
                f"{authorization_kind.value} authorization cannot be issued for "
                f"{order.purpose.value} order"
            )

    def issue(
        self,
        order: Order,
        *,
        now: datetime,
        risk_usd: float,
        stop_price: float,
        authorization_kind: AuthorizationKind = AuthorizationKind.ENTRY,
    ) -> RiskToken:
        if not isinstance(authorization_kind, AuthorizationKind):
            authorization_kind = AuthorizationKind(authorization_kind)
        self._validate_kind(order, authorization_kind)
        token_id = new_id("tok")
        expires_at = now + self._ttl
        signature = self._sign(
            order,
            token_id=token_id,
            issued_at=now,
            expires_at=expires_at,
            authorization_kind=authorization_kind,
            risk_usd=risk_usd,
            stop_price=stop_price,
        )
        self.issued_count += 1
        return RiskToken(
            token_id=token_id,
            issued_at=now,
            expires_at=expires_at,
            signature=signature,
            risk_usd=risk_usd,
            stop_price=stop_price,
            authorization_kind=authorization_kind,
        )

    def verify(
        self, order: Order, token: RiskToken, *, now: datetime
    ) -> AuthorizationKind:
        """Raise `TokenError` unless the token authorises exactly this order, right now."""
        expected = self._sign(
            order,
            token_id=token.token_id,
            issued_at=token.issued_at,
            expires_at=token.expires_at,
            authorization_kind=token.authorization_kind,
            risk_usd=token.risk_usd,
            stop_price=token.stop_price,
        )
        if not hmac.compare_digest(expected, token.signature):
            raise TokenError(
                RejectReason.TOKEN_BINDING_MISMATCH,
                "the token does not authorise this order: its authorization, timestamps, "
                "risk facts or immutable order fields differ",
            )
        if token.token_id in self._spent:
            raise TokenError(
                RejectReason.TOKEN_ALREADY_USED,
                f"token {token.token_id} has already been spent; this is a duplicate order",
            )
        if token.issued_at > now:
            raise TokenError(
                RejectReason.INVALID_TOKEN,
                f"token {token.token_id} was issued in the future at "
                f"{token.issued_at.isoformat()}",
            )
        if token.is_expired(now):
            raise TokenError(
                RejectReason.TOKEN_EXPIRED,
                f"token {token.token_id} expired at {token.expires_at.isoformat()}",
            )
        if token.expires_at <= token.issued_at:
            raise TokenError(
                RejectReason.INVALID_TOKEN,
                f"token {token.token_id} has a non-positive validity interval",
            )
        return token.authorization_kind

    def spend(self, token: RiskToken) -> None:
        self._spent.add(token.token_id)

    def is_spent(self, token: RiskToken) -> bool:
        return token.token_id in self._spent

    @property
    def spent_count(self) -> int:
        return len(self._spent)
