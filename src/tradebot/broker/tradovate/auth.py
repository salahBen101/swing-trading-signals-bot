"""Tradovate authentication and environment selection.

Two things this module exists to guarantee.

**Live is unreachable.** `Environment.LIVE` exists so configuration can name it and the
system can refuse it by name rather than by omission. Resolving it raises
`LiveTradingDisabled` unconditionally in v1, and it additionally requires a *separate*
credential set — `TRADOVATE_LIVE_*`, never the demo one — so enabling it later is a
deliberate act rather than a flipped boolean.

**Tokens are renewed, not re-requested.** An access token lasts about 90 minutes.
Tradovate caps concurrent sessions and answers a re-authentication storm with 4xx/5xx, so
the client renews ahead of expiry via `/auth/renewaccesstoken` and only calls
`/auth/accesstokenrequest` when it genuinely has no session.

Credentials come from the environment only. Nothing here reads a literal, and
`tests/tradebot/test_tradovate.py` asserts the source contains no credential-shaped string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from ...core.clock import MARKET_TZ, Clock, SystemClock
from ..base import BrokerAuthError, LiveTradingDisabled
from .transport import RestClient

# Renew this far ahead of expiry. Tradovate's own guidance is roughly 15 minutes; leaving
# it later risks a request landing on an expired token during a volatile minute.
RENEW_MARGIN = timedelta(minutes=15)
TOKEN_LIFETIME = timedelta(minutes=90)


class Environment(str, Enum):
    DEMO = "demo"
    LIVE = "live"


_HOSTS = {
    Environment.DEMO: "https://demo.tradovateapi.com/v1",
    Environment.LIVE: "https://live.tradovateapi.com/v1",
}
_WS_HOSTS = {
    Environment.DEMO: "wss://demo.tradovateapi.com/v1/websocket",
    Environment.LIVE: "wss://live.tradovateapi.com/v1/websocket",
}
_MD_WS_HOSTS = {
    Environment.DEMO: "wss://md.tradovateapi.com/v1/websocket",
    Environment.LIVE: "wss://md.tradovateapi.com/v1/websocket",
}


def rest_base_url(environment: Environment) -> str:
    _refuse_live(environment)
    return _HOSTS[environment]


def websocket_url(environment: Environment) -> str:
    _refuse_live(environment)
    return _WS_HOSTS[environment]


def market_data_websocket_url(environment: Environment) -> str:
    _refuse_live(environment)
    return _MD_WS_HOSTS[environment]


def _refuse_live(environment: Environment) -> None:
    if environment is Environment.LIVE:
        raise LiveTradingDisabled(
            "live trading is not implemented in this version. Reaching a live Tradovate "
            "endpoint requires an explicit code change, ALLOW_LIVE_TRADING=true, and a "
            "separate TRADOVATE_LIVE_* credential set. See PROJECT_SPEC section 8."
        )


@dataclass(frozen=True, slots=True)
class Credentials:
    name: str
    password: str
    app_id: str = "tradebot"
    app_version: str = "0.2.0"
    cid: str = ""
    sec: str = ""
    device_id: str = ""
    account_id: str = ""
    account_spec: str = ""

    @property
    def is_complete(self) -> bool:
        return bool(self.name and self.password)

    def auth_body(self) -> dict:
        """Exactly the fields `/auth/accesstokenrequest` documents.

        Optional fields are omitted when empty rather than sent blank: Tradovate rejects
        some empty-string values that it accepts as absent.
        """
        body = {
            "name": self.name,
            "password": self.password,
            "appId": self.app_id,
            "appVersion": self.app_version,
        }
        for key, value in (("cid", self.cid), ("sec", self.sec),
                           ("deviceId", self.device_id)):
            if value:
                body[key] = value
        return body

    def redacted(self) -> dict:
        return {
            "name": self.name,
            "password": "***" if self.password else "",
            "app_id": self.app_id,
            "account_id": self.account_id,
        }


def load_credentials(environment: Environment, env: dict | None = None) -> Credentials:
    """Read credentials from the environment. Demo and live use disjoint variable names."""
    env = env if env is not None else os.environ
    prefix = "TRADOVATE_LIVE_" if environment is Environment.LIVE else "TRADOVATE_DEMO_"
    return Credentials(
        name=env.get(f"{prefix}NAME", ""),
        password=env.get(f"{prefix}PASSWORD", ""),
        app_id=env.get(f"{prefix}APP_ID", "tradebot"),
        app_version=env.get(f"{prefix}APP_VERSION", "0.2.0"),
        cid=env.get(f"{prefix}CID", ""),
        sec=env.get(f"{prefix}SEC", ""),
        device_id=env.get(f"{prefix}DEVICE_ID", ""),
        account_id=env.get(f"{prefix}ACCOUNT_ID", ""),
        account_spec=env.get(f"{prefix}ACCOUNT_SPEC", env.get(f"{prefix}NAME", "")),
    )


@dataclass(slots=True)
class Session:
    access_token: str
    md_access_token: str
    expires_at: datetime
    user_id: int | None = None

    def needs_renewal(self, now: datetime) -> bool:
        return now >= self.expires_at - RENEW_MARGIN

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class Authenticator:
    """Owns the session and its lifecycle. Nothing else calls the auth endpoints."""

    def __init__(
        self,
        client: RestClient,
        credentials: Credentials,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.client = client
        self.credentials = credentials
        self.clock = clock or SystemClock()
        self.session: Session | None = None
        self.authenticate_count = 0
        self.renew_count = 0

    def token(self) -> str:
        """The current access token, authenticating or renewing as needed."""
        now = self.clock.now().astimezone(MARKET_TZ)

        if self.session is None:
            self._authenticate()
        elif self.session.is_expired(now):
            # Past expiry there is nothing to renew; a fresh session is the only option.
            self._authenticate()
        elif self.session.needs_renewal(now):
            self._renew()

        return self.session.access_token

    def _authenticate(self) -> None:
        if not self.credentials.is_complete:
            raise BrokerAuthError(
                "no Tradovate credentials. Set TRADOVATE_DEMO_NAME and "
                "TRADOVATE_DEMO_PASSWORD in your environment (see .env.example). "
                "Credentials are never read from source."
            )
        self.authenticate_count += 1
        body = self.client.post("auth/accesstokenrequest", self.credentials.auth_body())
        self.session = self._session_from(body)

    def _renew(self) -> None:
        self.renew_count += 1
        try:
            body = self.client.post("auth/renewaccesstoken", {},
                                    token=self.session.access_token)
            self.session = self._session_from(body)
        except BrokerAuthError:
            # The session went away before the renewal landed. Fall back to a full
            # authentication rather than leaving the caller with a dead token.
            self._authenticate()

    def _session_from(self, body: dict) -> Session:
        token = body.get("accessToken")
        if not token:
            raise BrokerAuthError(
                f"no accessToken in the response: {body.get('errorText') or body}"
            )
        expires_at = _parse_expiry(body.get("expirationTime"), self.clock)
        return Session(
            access_token=token,
            md_access_token=body.get("mdAccessToken") or token,
            expires_at=expires_at,
            user_id=body.get("userId"),
        )

    def invalidate(self) -> None:
        self.session = None


def _parse_expiry(raw: str | None, clock: Clock) -> datetime:
    now = clock.now().astimezone(MARKET_TZ)
    if not raw:
        return now + TOKEN_LIFETIME
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return now + TOKEN_LIFETIME
    if parsed.tzinfo is None:
        # Tradovate returns UTC. Reading a naive value as local time would shift expiry by
        # hours and leave a token that looks valid long after it is not.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(MARKET_TZ)
