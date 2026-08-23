"""Local web dashboard.

Standard library only — `http.server` plus a single static page. A web framework would be
one more dependency between a clean machine and a working setup, for a page that serves one
user on loopback.

**Security posture.** There is no authentication, so the server refuses to bind to anything
but loopback (enforced in `config.DashboardConfig.validate`). Everything is read-only except
one endpoint, which can only *reduce* what the system is allowed to do:

* `POST /api/stop` trips the kill switch — flatten and refuse new entries.

There is deliberately no endpoint that opens a position, changes a limit, or picks a
strategy, and no web endpoint can clear the kill switch. A dashboard that can trade or
silently restore permission is a second, untested trading interface.
"""

from __future__ import annotations

import json
from copy import deepcopy
import threading
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..core.clock import MARKET_TZ

STATIC_DIR = Path(__file__).parent / "static"


class DashboardState:
    """Everything the page shows, behind one lock.

    The runner pushes a snapshot after each bar; the HTTP threads read it. Copying under a
    lock rather than sharing live objects means a request can never observe the engine
    halfway through updating a position.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {"state": "STARTING"}
        self._health: dict[str, Any] = {}
        self._trades: list[dict] = []
        self._rejections: list[dict] = []
        self._equity: list[dict] = []
        self._logs: list[dict] = []

    def update(
        self, *, snapshot=None, health=None, trades=None, rejections=None,
        equity=None, logs=None,
    ) -> None:
        with self._lock:
            if snapshot is not None:
                self._snapshot = snapshot
            if health is not None:
                self._health = health
            if trades is not None:
                self._trades = trades
            if rejections is not None:
                self._rejections = rejections
            if equity is not None:
                self._equity = equity
            if logs is not None:
                self._logs = logs

    def read(self) -> dict:
        with self._lock:
            return {
                # A shallow copy still shares nested position/prop dictionaries with the
                # runner. Deep-copy under the lock so an HTTP request always sees one
                # coherent account snapshot rather than half of two bars.
                "snapshot": deepcopy(self._snapshot),
                "health": deepcopy(self._health),
                "trades": deepcopy(self._trades),
                "rejections": deepcopy(self._rejections),
                "equity": deepcopy(self._equity),
                "logs": deepcopy(self._logs),
                "served_at": datetime.now(tz=MARKET_TZ).isoformat(timespec="seconds"),
            }


class _Handler(BaseHTTPRequestHandler):
    server_version = "tradebot-dashboard"

    def __init__(self, state: DashboardState, on_stop: Callable[[str], dict],
                 *args, **kwargs) -> None:
        self.state = state
        self.on_stop = on_stop
        super().__init__(*args, **kwargs)

    # -- plumbing ------------------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        """Silence the default stderr access log; the journal is the record."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # The page is self-contained; refusing external resources outright means a stray
        # copy-pasted CDN link cannot quietly start phoning out from a trading machine.
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- routes --------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if route == "/api/state":
            return self._json(self.state.read())
        if route == "/api/health":
            return self._json(self.state.read()["health"])
        return self._json({"error": "not found", "path": route}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0].rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        reason = str(body.get("reason") or "dashboard")

        if route == "/api/stop":
            return self._json(self.on_stop(reason))
        return self._json({"error": "not found", "path": route}, code=404)


class Dashboard:
    """Runs the HTTP server on a background thread."""

    def __init__(
        self,
        state: DashboardState,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        on_stop: Callable[[str], dict] | None = None,
    ) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"refusing to bind the dashboard to {host!r}: it has no authentication "
                f"and exposes an emergency STOP"
            )
        self.state = state
        self.host = host
        self.port = port
        self._on_stop = on_stop or (lambda reason: {"ok": False, "detail": "no runner attached"})
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        handler = partial(_Handler, self.state, self._on_stop)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        # Port 0 asks the OS for a free port, which is how tests avoid fighting over one.
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name="tradebot-dashboard")
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> Dashboard:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
