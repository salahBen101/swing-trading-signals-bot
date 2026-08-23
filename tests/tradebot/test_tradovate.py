"""M10: the Tradovate demo adapter.

No network. The transport is injected, so the documented request shapes, the token
lifecycle, the error mapping and the WebSocket framing are all asserted directly.

The tests that matter most are the ones proving a live endpoint is unreachable and that no
credential is ever read from source.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta

import pytest

from tradebot.broker.base import (
    BrokerAuthError,
    BrokerConnectionError,
    BrokerRateLimitError,
    BrokerTimeout,
    LiveTradingDisabled,
    NotConnected,
    OrderRejected,
)
from tradebot.broker.tradovate import adapter as adapter_module
from tradebot.broker.tradovate import auth as auth_module
from tradebot.broker.tradovate import ws as ws_module
from tradebot.broker.tradovate.adapter import TradovateBroker
from tradebot.broker.tradovate.auth import (
    Authenticator,
    Credentials,
    Environment,
    load_credentials,
    market_data_websocket_url,
    rest_base_url,
    websocket_url,
)
from tradebot.broker.tradovate.transport import FakeTransport, Response, RestClient
from tradebot.broker.tradovate.ws import (
    FrameError,
    FrameType,
    HEARTBEAT_REPLY,
    ReconnectPolicy,
    SocketState,
    build_authorize,
    build_request,
    parse_frame,
)
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.models import Order
from tradebot.core.types import OrderStatus, OrderType, Side
from tradebot.instruments.registry import get_instrument

MNQ = get_instrument("MNQ")
DEMO = "https://demo.tradovateapi.com/v1"


def at(hour=11, minute=0) -> datetime:
    return datetime(2024, 4, 1, hour, minute, tzinfo=MARKET_TZ)


def creds(**over) -> Credentials:
    base = dict(name="demo-user", password="demo-pass", app_id="tradebot",
                app_version="0.2.0", account_id="12345", account_spec="demo-user")
    base.update(over)
    return Credentials(**base)


def token_response(minutes: int = 90) -> dict:
    expiry = (at() + timedelta(minutes=minutes)).astimezone().isoformat()
    return {"accessToken": "tok-abc", "mdAccessToken": "md-abc",
            "expirationTime": expiry, "userId": 777}


# ============================================================ live is unreachable


def test_the_live_rest_host_cannot_be_resolved():
    with pytest.raises(LiveTradingDisabled, match="not implemented"):
        rest_base_url(Environment.LIVE)


def test_the_live_websocket_hosts_cannot_be_resolved():
    with pytest.raises(LiveTradingDisabled):
        websocket_url(Environment.LIVE)
    with pytest.raises(LiveTradingDisabled):
        market_data_websocket_url(Environment.LIVE)


def test_constructing_a_live_adapter_raises_before_any_request():
    fake = FakeTransport()
    with pytest.raises(LiveTradingDisabled):
        TradovateBroker(MNQ, environment=Environment.LIVE, transport=fake,
                        credentials=creds())
    assert fake.requests == [], "nothing may be sent toward a live host"


def test_the_demo_host_is_the_documented_one():
    assert rest_base_url(Environment.DEMO) == DEMO
    assert websocket_url(Environment.DEMO) == "wss://demo.tradovateapi.com/v1/websocket"


def test_demo_and_live_credentials_come_from_disjoint_variables():
    env = {
        "TRADOVATE_DEMO_NAME": "demo-user", "TRADOVATE_DEMO_PASSWORD": "demo-pass",
        "TRADOVATE_LIVE_NAME": "live-user", "TRADOVATE_LIVE_PASSWORD": "live-pass",
    }
    assert load_credentials(Environment.DEMO, env).name == "demo-user"
    assert load_credentials(Environment.LIVE, env).name == "live-user"


def test_missing_credentials_produce_an_actionable_error():
    broker = TradovateBroker(MNQ, transport=FakeTransport(),
                             credentials=Credentials(name="", password=""))
    with pytest.raises(BrokerAuthError, match="TRADOVATE_DEMO_NAME"):
        broker.connect()


@pytest.mark.parametrize("module", [adapter_module, auth_module, ws_module])
def test_no_credential_literal_exists_in_the_source(module):
    """Every string assigned to a credential-shaped name must come from the environment."""
    tree = ast.parse(inspect.getsource(module))
    suspicious = {"password", "secret", "api_key", "apikey", "access_token", "token",
                  "sec", "cid"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        if not node.value.value:
            continue
        for target in node.targets:
            name = getattr(target, "id", "") or getattr(target, "attr", "")
            assert name.lower() not in suspicious, (
                f"{module.__name__} assigns a literal to {name!r}"
            )


def test_credentials_redact_the_password():
    assert creds().redacted()["password"] == "***"
    assert "demo-pass" not in str(creds().redacted())


# ============================================================ authentication


def test_the_auth_request_matches_the_documented_shape():
    fake = FakeTransport({"auth/accesstokenrequest": token_response()})
    client = RestClient(DEMO, fake)
    auth = Authenticator(client, creds(cid="c1", sec="s1", device_id="d1"),
                         clock=SimulatedClock(at()))

    assert auth.token() == "tok-abc"
    sent = fake.last("auth/accesstokenrequest")
    assert sent["method"] == "POST"
    assert sent["url"] == f"{DEMO}/auth/accesstokenrequest"
    assert sent["body"] == {
        "name": "demo-user", "password": "demo-pass", "appId": "tradebot",
        "appVersion": "0.2.0", "cid": "c1", "sec": "s1", "deviceId": "d1",
    }
    assert "Authorization" not in sent["headers"], "the auth call carries no bearer token"


def test_optional_auth_fields_are_omitted_rather_than_sent_empty():
    fake = FakeTransport({"auth/accesstokenrequest": token_response()})
    Authenticator(RestClient(DEMO, fake), creds(), clock=SimulatedClock(at())).token()
    body = fake.last("auth/accesstokenrequest")["body"]
    assert "cid" not in body and "sec" not in body and "deviceId" not in body


def test_a_valid_token_is_reused_rather_than_re_requested():
    fake = FakeTransport({"auth/accesstokenrequest": token_response()})
    auth = Authenticator(RestClient(DEMO, fake), creds(), clock=SimulatedClock(at()))

    for _ in range(5):
        auth.token()
    assert auth.authenticate_count == 1, "re-auth storms cause 4xx/5xx and session limits"


def test_the_token_is_renewed_before_it_expires_not_after():
    clock = SimulatedClock(at())
    fake = FakeTransport({
        "auth/accesstokenrequest": token_response(90),
        "auth/renewaccesstoken": {"accessToken": "tok-renewed",
                                  "expirationTime": (at(13, 0)).isoformat()},
    })
    auth = Authenticator(RestClient(DEMO, fake), creds(), clock=clock)
    auth.token()

    clock.set(at() + timedelta(minutes=60))       # 30 min left, margin is 15
    auth.token()
    assert auth.renew_count == 0

    clock.set(at() + timedelta(minutes=76))       # 14 min left, inside the margin
    assert auth.token() == "tok-renewed"
    assert auth.renew_count == 1
    assert auth.authenticate_count == 1

    renewal = fake.last("auth/renewaccesstoken")
    assert renewal["headers"]["Authorization"] == "Bearer tok-abc"


def test_a_fully_expired_session_authenticates_again_rather_than_renewing():
    clock = SimulatedClock(at())
    fake = FakeTransport({"auth/accesstokenrequest": token_response(90)})
    auth = Authenticator(RestClient(DEMO, fake), creds(), clock=clock)
    auth.token()

    clock.set(at() + timedelta(minutes=200))
    auth.token()
    assert auth.authenticate_count == 2
    assert auth.renew_count == 0


def test_a_renewal_that_is_refused_falls_back_to_a_full_authentication():
    clock = SimulatedClock(at())
    fake = FakeTransport({
        "auth/accesstokenrequest": token_response(90),
        "auth/renewaccesstoken": Response(401, {"errorText": "session gone"}, {}),
    })
    auth = Authenticator(RestClient(DEMO, fake), creds(), clock=clock)
    auth.token()

    clock.set(at() + timedelta(minutes=80))
    auth.token()
    assert auth.renew_count == 1
    assert auth.authenticate_count == 2


def test_a_response_without_a_token_is_an_auth_error():
    fake = FakeTransport({"auth/accesstokenrequest": {"userStatus": "TemporaryLocked"}})
    auth = Authenticator(RestClient(DEMO, fake), creds(), clock=SimulatedClock(at()))
    with pytest.raises(BrokerAuthError, match="no accessToken"):
        auth.token()


# ============================================================ transport error mapping


def test_a_401_becomes_an_auth_error_not_a_retry():
    fake = FakeTransport({"order/list": Response(401, {"errorText": "denied"}, {})})
    with pytest.raises(BrokerAuthError):
        RestClient(DEMO, fake).get("order/list", token="t")
    assert len(fake.requests) == 1, "retrying an auth failure cannot help"


def test_a_429_waits_the_penalty_time_and_re_presents_the_ticket():
    slept: list[float] = []
    fake = FakeTransport({"order/placeorder": [
        Response(429, {"p-ticket": "TICKET", "p-time": 3}, {}),
        Response(200, {"orderId": 99}, {}),
    ]})
    client = RestClient(DEMO, fake, sleep=slept.append)

    assert client.post("order/placeorder", {"a": 1}, token="t") == {"orderId": 99}
    assert slept == [3.0]
    assert fake.requests[1]["body"]["p-ticket"] == "TICKET"


def test_persistent_rate_limiting_gives_up_with_the_backoff_attached():
    fake = FakeTransport({"order/list": Response(429, {"p-ticket": "T", "p-time": 2}, {})})
    client = RestClient(DEMO, fake, max_retries=2, sleep=lambda _: None)
    with pytest.raises(BrokerRateLimitError) as err:
        client.get("order/list", token="t")
    assert err.value.retry_after_seconds == 2.0
    assert err.value.retryable


def test_a_500_is_retried_then_reported_as_a_connection_error():
    fake = FakeTransport({"order/list": Response(500, {}, {})})
    client = RestClient(DEMO, fake, max_retries=2, sleep=lambda _: None)
    with pytest.raises(BrokerConnectionError, match="after 3 attempts"):
        client.get("order/list", token="t")
    assert len(fake.requests) == 3


def test_a_408_is_a_timeout():
    fake = FakeTransport({"order/list": Response(408, {}, {})})
    with pytest.raises(BrokerTimeout):
        RestClient(DEMO, fake).get("order/list", token="t")


def test_a_200_carrying_an_error_body_is_still_a_failure():
    """Tradovate reports some failures with a success status code."""
    fake = FakeTransport({"order/placeorder": {"errorText": "insufficient margin"}})
    with pytest.raises(OrderRejected, match="insufficient margin"):
        RestClient(DEMO, fake).post("order/placeorder", {}, token="t")


def test_a_transport_level_failure_surfaces_as_a_connection_error():
    fake = FakeTransport({"order/list": BrokerConnectionError("socket closed")})
    with pytest.raises(BrokerConnectionError):
        RestClient(DEMO, fake).get("order/list", token="t")


# ============================================================ the adapter


@pytest.fixture
def wired():
    fake = FakeTransport({
        "auth/accesstokenrequest": token_response(),
        "account/list": {"items": [{"id": 12345, "name": "DEMO1",
                                    "cashBalance": 50000.0, "totalCashValue": 50120.0}]},
        "order/placeorder": {"orderId": 555},
        "order/cancelorder": {"commandId": 1},
        "order/list": {"items": []},
        "position/list": {"items": []},
        "fill/list": {"items": []},
    })
    broker = TradovateBroker(MNQ, transport=fake, credentials=creds(),
                             clock=SimulatedClock(at()), symbol="MNQZ6")
    return broker, fake


def test_the_adapter_reports_itself_as_paper(wired):
    broker, _ = wired
    assert broker.is_paper is True
    assert broker.name == "tradovate"


def test_nothing_can_be_sent_before_connecting(wired):
    broker, _ = wired
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.BUY,
                  quantity=2, order_type=OrderType.MARKET)
    with pytest.raises(NotConnected):
        broker.place_order(order)


def test_placeorder_sends_exactly_the_documented_body(wired):
    broker, fake = wired
    broker.connect()
    order = Order(order_id="o1", timestamp=at(), instrument="MNQ", side=Side.BUY,
                  quantity=2, order_type=OrderType.MARKET)

    ack = broker.place_order(order)
    body = fake.last("order/placeorder")["body"]

    assert body == {
        "accountSpec": "demo-user", "accountId": 12345, "action": "Buy",
        "symbol": "MNQZ6", "orderQty": 2, "orderType": "Market", "isAutomated": True,
    }
    assert ack.broker_order_id == "555"
    assert ack.status is OrderStatus.SUBMITTED


def test_is_automated_is_always_true_and_never_configurable(wired):
    """Exchange policy: anything not triggered by a human must be flagged."""
    broker, fake = wired
    broker.connect()
    for side in (Side.BUY, Side.SELL):
        for order_type, extra in (
            (OrderType.MARKET, {}),
            (OrderType.LIMIT, {"limit_price": 18000.0}),
            (OrderType.STOP, {"stop_price": 17900.0}),
        ):
            broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ",
                                     side=side, quantity=1, order_type=order_type, **extra))
            assert fake.last("order/placeorder")["body"]["isAutomated"] is True

    source = inspect.getsource(adapter_module)
    assert "isAutomated\": False" not in source and "isAutomated': False" not in source


def test_limit_and_stop_prices_are_snapped_to_the_tick_grid(wired):
    broker, fake = wired
    broker.connect()
    broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ",
                             side=Side.BUY, quantity=1, order_type=OrderType.STOP_LIMIT,
                             limit_price=18000.1873, stop_price=17999.1873))
    body = fake.last("order/placeorder")["body"]
    assert body["price"] == 18000.25
    assert body["stopPrice"] == 17999.25
    assert body["orderType"] == "StopLimit"


def test_a_failure_reason_in_the_response_raises(wired):
    broker, fake = wired
    broker.connect()
    fake.add("order/placeorder", {"failureReason": "MaxOrderQty",
                                  "failureText": "too many contracts"})
    with pytest.raises(OrderRejected, match="MaxOrderQty"):
        broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ",
                                 side=Side.BUY, quantity=99, order_type=OrderType.MARKET))


def test_a_response_with_no_order_id_is_a_rejection(wired):
    broker, fake = wired
    broker.connect()
    fake.add("order/placeorder", {"status": "ok"})
    with pytest.raises(OrderRejected, match="no orderId"):
        broker.place_order(Order(order_id="o", timestamp=at(), instrument="MNQ",
                                 side=Side.BUY, quantity=1, order_type=OrderType.MARKET))


def test_cancel_sends_the_broker_order_id_as_an_integer(wired):
    broker, fake = wired
    broker.connect()
    broker.cancel_order("555")
    assert fake.last("order/cancelorder")["body"] == {"orderId": 555}


def test_positions_are_signed_and_flat_rows_are_dropped(wired):
    broker, fake = wired
    broker.connect()
    fake.add("position/list", {"items": [
        {"netPos": 3, "netPrice": 18000.25},
        {"netPos": 0, "netPrice": 0.0},
        {"netPos": -2, "netPrice": 17990.0},
    ]})
    positions = broker.get_positions()
    assert [p.quantity for p in positions] == [3, -2]
    assert positions[0].side is Side.BUY and positions[1].side is Side.SELL


def test_the_account_is_read_from_the_configured_account_id(wired):
    broker, fake = wired
    broker.connect()
    account = broker.get_account()
    assert account.account_id == "12345"
    assert account.is_paper
    assert account.cash == 50000.0


def test_order_status_words_map_onto_the_shared_vocabulary(wired):
    broker, fake = wired
    broker.connect()
    fake.add("order/list", {"items": [
        {"id": 1, "ordStatus": "Working", "cumQty": 0, "avgPx": 0},
        {"id": 2, "ordStatus": "Filled", "cumQty": 2, "avgPx": 18000.25},
        {"id": 3, "ordStatus": "Rejected", "cumQty": 0, "avgPx": 0},
    ]})
    statuses = [o.status for o in broker.get_orders()]
    assert statuses == [OrderStatus.ACCEPTED, OrderStatus.FILLED, OrderStatus.REJECTED]


def test_fills_are_reported_once_each(wired):
    broker, fake = wired
    broker.connect()
    broker.poll_events()  # drain the connect event
    fake.add("order/placeorder", {"orderId": 555})
    broker.place_order(Order(order_id="o1", timestamp=at(), instrument="MNQ",
                             side=Side.BUY, quantity=2, order_type=OrderType.MARKET))
    broker.poll_events()

    fake.add("fill/list", {"items": [
        {"id": 900, "orderId": 555, "qty": 2, "price": 18000.25, "action": "Buy",
         "commission": 1.24, "timestamp": "2024-04-01T15:00:00Z"},
    ]})
    first = [e for e in broker.poll_events() if e.fill]
    assert len(first) == 1
    assert first[0].fill.quantity == 2
    assert first[0].fill.order_id == "o1"
    assert first[0].fill.slippage_points == 0.0, (
        "the venue response has no honest benchmark, so the adapter must not invent one"
    )

    second = [e for e in broker.poll_events() if e.fill]
    assert second == [], "a fill must not be replayed on the next poll"


def test_a_failing_fill_poll_does_not_take_the_runner_down(wired):
    broker, fake = wired
    broker.connect()
    fake.add("fill/list", BrokerConnectionError("socket closed"))
    events = broker.poll_events()  # must not raise
    assert any("fill poll failed" in e.detail for e in events)


def test_disconnect_invalidates_the_session(wired):
    broker, _ = wired
    broker.connect()
    assert broker.is_connected()
    broker.disconnect()
    assert not broker.is_connected()
    assert broker.auth.session is None


# ============================================================ websocket framing


def test_the_open_frame_is_answered_with_an_authorize_request():
    state = SocketState(access_token="tok-abc")
    reply = state.handle("o")
    assert reply == "authorize\n1\n\ntok-abc"
    assert not state.authorized, "authorisation is confirmed by the response, not the send"


def test_a_heartbeat_must_be_answered_or_the_socket_is_dropped():
    state = SocketState(access_token="t")
    assert state.handle("h") == HEARTBEAT_REPLY
    assert state.heartbeats == 1


def test_a_data_frame_dispatches_every_element():
    seen = []
    state = SocketState(access_token="t", on_data=seen.append)
    assert state.handle('a[{"e":"props","d":{"x":1}},{"e":"props","d":{"x":2}}]') is None
    assert len(seen) == 2 and state.data_messages == 2


def test_the_authorize_response_flips_the_authorized_flag():
    state = SocketState(access_token="t")
    state.handle("o")
    state.handle('a[{"s":200,"i":1}]')
    assert state.authorized


def test_a_close_frame_clears_authorisation():
    state = SocketState(access_token="t")
    state.handle("o")
    state.handle('a[{"s":200,"i":1}]')
    assert state.handle("c") is None
    assert not state.authorized and state.closes == 1


def test_an_unknown_frame_type_is_an_error_not_a_silent_skip():
    with pytest.raises(FrameError, match="unknown frame type"):
        parse_frame("x[]")
    with pytest.raises(FrameError, match="empty frame"):
        parse_frame("")


def test_a_malformed_data_frame_is_an_error():
    with pytest.raises(FrameError, match="not JSON"):
        parse_frame("a{not json")


def test_a_data_frame_carrying_one_object_is_normalised_to_a_list():
    frame = parse_frame('a{"e":"props"}')
    assert frame.type is FrameType.DATA
    assert frame.payload == [{"e": "props"}]


def test_requests_use_the_newline_delimited_wire_format():
    assert build_request("user/syncrequest", 3, body={"users": [7]}) == (
        'user/syncrequest\n3\n\n{"users": [7]}'
    )
    assert build_request("order/list", 4) == "order/list\n4\n\n"
    assert build_authorize(1, "tok") == "authorize\n1\n\ntok"


def test_request_ids_increment():
    state = SocketState(access_token="t")
    assert [state.next_request_id() for _ in range(3)] == [1, 2, 3]


def test_reconnect_backs_off_exponentially_to_a_ceiling():
    policy = ReconnectPolicy(initial_seconds=1.0, max_seconds=8.0, factor=2.0)
    assert [policy.next_delay() for _ in range(6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    policy.reset()
    assert policy.next_delay() == 1.0


def test_reconnect_attempts_can_be_capped():
    policy = ReconnectPolicy(max_attempts=2)
    policy.next_delay()
    assert not policy.exhausted
    policy.next_delay()
    assert policy.exhausted


def test_a_reconnect_after_a_drop_resets_the_backoff():
    state = SocketState(access_token="t")
    state.reconnect.next_delay()
    state.reconnect.next_delay()
    state.handle("o")  # the socket came back
    assert state.reconnect.attempts == 0
