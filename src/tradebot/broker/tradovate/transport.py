"""HTTP transport for the Tradovate REST API.

A thin seam over `urllib.request`, for two reasons. It keeps `requests` out of the
dependency list, so a clean-machine install is one `pip install -r` with nothing that
needs compiling. And it gives tests a place to inject a fake without a socket, which is how
the 401/429/500 handling below is actually exercised rather than merely written.

Tradovate's documented failure modes are handled explicitly:

* **429 with a penalty ticket.** The API answers rate limiting with `p-ticket` and
  `p-time`; the correct response is to wait `p-time` seconds and re-present the ticket,
  not to retry immediately and dig deeper.
* **401.** The session is gone. Retrying the same call cannot help — the caller has to
  re-authenticate, so this raises a distinct error type.
* **A 200 carrying `errorText`.** Tradovate reports several failures with a success status
  code and an error body. Treating HTTP 200 as success is the mistake this guards against.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..base import (
    BrokerAuthError,
    BrokerConnectionError,
    BrokerError,
    BrokerRateLimitError,
    BrokerTimeout,
    OrderRejected,
)


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: dict
    headers: dict


class Transport(Protocol):
    def request(
        self, method: str, url: str, *, body: dict | None = None,
        headers: dict | None = None, timeout: float = 15.0,
    ) -> Response: ...


class UrllibTransport:
    """The real one."""

    def request(
        self, method: str, url: str, *, body: dict | None = None,
        headers: dict | None = None, timeout: float = 15.0,
    ) -> Response:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=payload, method=method)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8") or "{}"
                return Response(response.status, _loads(raw), dict(response.headers))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8") or "{}"
            return Response(exc.code, _loads(raw), dict(exc.headers or {}))
        except urllib.error.URLError as exc:
            raise BrokerConnectionError(f"{method} {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise BrokerTimeout(f"{method} {url} timed out after {timeout}s") from exc


def _loads(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"items": parsed}


class RestClient:
    """Applies Tradovate's error conventions on top of a `Transport`."""

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        *,
        timeout: float = 15.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or UrllibTransport()
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = sleep
        self.request_count = 0

    def call(
        self, method: str, path: str, *, body: dict | None = None, token: str | None = None,
        retry_on_rate_limit: bool = True,
    ) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        attempt = 0

        while True:
            attempt += 1
            self.request_count += 1
            response = self.transport.request(method, url, body=body, headers=headers,
                                              timeout=self.timeout)

            if response.status == 429 or "p-ticket" in response.body:
                ticket = response.body.get("p-ticket")
                wait = float(response.body.get("p-time", 1) or 1)
                if not retry_on_rate_limit or attempt > self.max_retries:
                    raise BrokerRateLimitError(
                        f"rate limited on {path} after {attempt} attempt(s)",
                        retry_after_seconds=wait,
                    )
                # Re-present the ticket after waiting, as the API asks. Hammering without
                # the ticket extends the penalty rather than clearing it.
                self._sleep(wait)
                body = {**(body or {}), "p-ticket": ticket} if ticket else body
                continue

            if response.status in (401, 403):
                raise BrokerAuthError(
                    f"{path} refused ({response.status}): "
                    f"{response.body.get('errorText') or 'not authorised'}"
                )
            if response.status == 408 or response.status == 504:
                raise BrokerTimeout(f"{path} timed out ({response.status})")
            if response.status >= 500:
                if attempt <= self.max_retries:
                    self._sleep(min(2 ** attempt * 0.25, 5.0))
                    continue
                raise BrokerConnectionError(
                    f"{path} failed with {response.status} after {attempt} attempts"
                )
            if response.status >= 400:
                raise BrokerError(f"{path} failed with {response.status}: {response.body}")

            # A 200 that carries an error. Tradovate does this, and treating the status
            # code alone as success is exactly how a rejected order looks like a placed one.
            if "errorText" in response.body and response.body["errorText"]:
                raise OrderRejected(f"{path}: {response.body['errorText']}")

            return response.body

    def get(self, path: str, *, token: str | None = None) -> dict:
        return self.call("GET", path, token=token)

    def post(self, path: str, body: dict, *, token: str | None = None, **kwargs) -> dict:
        return self.call("POST", path, body=body, token=token, **kwargs)


class FakeTransport:
    """A scripted transport for tests. Records every request it was given.

    Lives in the package rather than the test file because the WebSocket client and the
    adapter both need it, and a shared fake that drifts from the real client's expectations
    is worse than no fake at all.
    """

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes: dict[str, Any] = routes or {}
        self.requests: list[dict] = []

    def add(self, path: str, response: Any) -> None:
        """`response` may be a dict, a `Response`, an Exception, or a list to pop from."""
        self.routes[path] = response

    def request(
        self, method: str, url: str, *, body: dict | None = None,
        headers: dict | None = None, timeout: float = 15.0,
    ) -> Response:
        path = url.split("/v1/", 1)[-1] if "/v1/" in url else url
        self.requests.append(
            {"method": method, "url": url, "path": path, "body": body,
             "headers": dict(headers or {})}
        )
        route = self.routes.get(path)
        if route is None:
            return Response(404, {"errorText": f"no route for {path}"}, {})
        if isinstance(route, list):
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, Exception):
            raise route
        if isinstance(route, Response):
            return route
        return Response(200, route, {})

    def last(self, path: str) -> dict | None:
        for record in reversed(self.requests):
            if record["path"] == path:
                return record
        return None
