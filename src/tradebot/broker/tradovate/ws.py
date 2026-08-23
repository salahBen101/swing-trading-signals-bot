"""Tradovate WebSocket framing.

Tradovate does not send plain JSON over its socket. Every message is prefixed with a
single-character frame type, and getting this wrong is the most common reason a client
"connects and then nothing happens":

| frame | meaning | correct response |
|---|---|---|
| `o` | socket opened | send `authorize\\n{id}\\n\\n{accessToken}` |
| `h` | heartbeat from the server | reply with `[]` to keep the connection alive |
| `a` | an array of JSON payloads | dispatch each element |
| `c` | close | reconnect with backoff |

Requests are newline-delimited: `endpoint\\nrequestId\\nquery\\nbody`.

The parsing and the state machine live here, separate from any socket, so they can be
tested exhaustively without a network. `TradovateSocket` then wires them to a real
connection; v1 ships the framing and the reconnect policy but does not consume the socket
in the runner (see docs/LIMITATIONS.md).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class FrameType(str, Enum):
    OPEN = "o"
    HEARTBEAT = "h"
    DATA = "a"
    CLOSE = "c"


@dataclass(frozen=True, slots=True)
class Frame:
    type: FrameType
    payload: list

    @property
    def is_data(self) -> bool:
        return self.type is FrameType.DATA


class FrameError(ValueError):
    pass


def parse_frame(raw: str) -> Frame:
    """Decode one socket message."""
    if not raw:
        raise FrameError("empty frame")
    head, body = raw[0], raw[1:].strip()
    try:
        frame_type = FrameType(head)
    except ValueError as exc:
        raise FrameError(f"unknown frame type {head!r} in {raw[:40]!r}") from exc

    if frame_type is not FrameType.DATA:
        # `o`, `h` and `c` carry no payload; anything trailing is noise.
        return Frame(frame_type, [])

    try:
        payload = json.loads(body or "[]")
    except json.JSONDecodeError as exc:
        raise FrameError(f"data frame is not JSON: {body[:80]!r}") from exc
    if not isinstance(payload, list):
        payload = [payload]
    return Frame(frame_type, payload)


def build_request(endpoint: str, request_id: int, query: str = "", body: dict | None = None) -> str:
    """`endpoint\\nid\\nquery\\nbody` — the wire format for every socket request."""
    encoded = json.dumps(body) if body is not None else ""
    return f"{endpoint}\n{request_id}\n{query}\n{encoded}"


def build_authorize(request_id: int, access_token: str) -> str:
    """The reply to the `o` frame. The token goes in the body slot, unquoted."""
    return f"authorize\n{request_id}\n\n{access_token}"


HEARTBEAT_REPLY = "[]"


@dataclass
class ReconnectPolicy:
    """Exponential backoff with a ceiling.

    A socket that reconnects instantly and fails instantly becomes a request flood, which
    is how a rate limit turns into a session ban.
    """

    initial_seconds: float = 1.0
    max_seconds: float = 30.0
    factor: float = 2.0
    max_attempts: int = 0  # 0 = unlimited
    attempts: int = 0

    def next_delay(self) -> float:
        delay = min(self.initial_seconds * (self.factor ** self.attempts), self.max_seconds)
        self.attempts += 1
        return delay

    def reset(self) -> None:
        self.attempts = 0

    @property
    def exhausted(self) -> bool:
        return bool(self.max_attempts) and self.attempts >= self.max_attempts


@dataclass
class SocketState:
    """The protocol state machine, with no socket attached.

    `handle(raw)` returns the string that should be written back, or None. Every
    transition in the table above is exercised in `test_tradovate.py`.
    """

    access_token: str
    on_data: Callable[[dict], None] | None = None
    authorized: bool = False
    request_id: int = 0
    heartbeats: int = 0
    data_messages: int = 0
    closes: int = 0
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)

    def next_request_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def handle(self, raw: str) -> str | None:
        frame = parse_frame(raw)

        if frame.type is FrameType.OPEN:
            self.authorized = False
            self.reconnect.reset()
            return build_authorize(self.next_request_id(), self.access_token)

        if frame.type is FrameType.HEARTBEAT:
            self.heartbeats += 1
            # The server drops a socket that stops answering heartbeats.
            return HEARTBEAT_REPLY

        if frame.type is FrameType.CLOSE:
            self.closes += 1
            self.authorized = False
            return None

        for message in frame.payload:
            self.data_messages += 1
            if isinstance(message, dict):
                # The authorize response comes back as an ordinary data frame carrying the
                # request id we sent, so authorisation is confirmed here rather than
                # assumed after writing the request.
                if message.get("s") == 200 and message.get("i") == 1:
                    self.authorized = True
                if self.on_data is not None:
                    self.on_data(message)
        return None

    def subscribe_user_events(self, user_id: int) -> str:
        return build_request("user/syncrequest", self.next_request_id(),
                             body={"users": [user_id]})
