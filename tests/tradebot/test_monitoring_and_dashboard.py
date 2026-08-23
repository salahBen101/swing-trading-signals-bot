"""M9: health monitoring and the dashboard.

The dashboard tests exist mainly to pin what the API must *not* do. A page that can place
an order is a second trading interface with none of the risk engine's protections in front
of it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

from tradebot.config import ConfigError, DashboardConfig
from tradebot.core.clock import MARKET_TZ, SimulatedClock
from tradebot.core.types import HealthStatus
from tradebot.dashboard.server import Dashboard, DashboardState
from tradebot.monitoring.health import HealthMonitor
from tradebot.risk.killswitch import KillSwitch


def at(hour=11, minute=0, second=0) -> datetime:
    return datetime(2024, 4, 1, hour, minute, second, tzinfo=MARKET_TZ)


# ============================================================ health monitor


@pytest.fixture
def monitor(tmp_path):
    clock = SimulatedClock(at())
    return HealthMonitor(
        bar_seconds=60, stale_multiple=3.0, hard_stale_multiple=10.0,
        heartbeat_timeout_seconds=120.0, clock=clock,
        kill_switch=KillSwitch(tmp_path / "K.flag", max_consecutive_errors=3, clock=clock),
    )


def test_a_fresh_process_is_healthy_not_stale(monitor):
    report = monitor.report(at())
    assert report.status is HealthStatus.OK
    assert report.may_trade
    assert "no bars yet" in next(c.detail for c in report.checks if c.name == "market_data")


def test_data_degrades_then_dies_as_the_gap_widens(monitor):
    """A running loop keeps beating even when no bar arrives, so only data goes stale."""
    monitor.on_bar(at(), arrived_at=at())

    def look(moment):
        monitor.heartbeat(moment)  # the loop is alive; the feed is what is quiet
        return monitor.report(moment)

    assert look(at(11, 2)).status is HealthStatus.OK

    degraded = look(at(11, 3, 1))
    assert degraded.status is HealthStatus.DEGRADED
    assert degraded.may_trade, "a slow feed is not a broken one"

    down = look(at(11, 10, 1))
    assert down.status is HealthStatus.DOWN
    assert not down.may_trade


def test_a_replayed_bar_stamps_the_heartbeat_at_the_same_instant(monitor):
    """Otherwise a fast replay looks like a hung process on its very first bar."""
    monitor.on_bar(at(), arrived_at=at())
    assert monitor.last_heartbeat_at == monitor.last_bar_at


def test_stale_helpers_agree_with_the_report(monitor):
    monitor.on_bar(at(), arrived_at=at())
    assert not monitor.data_is_stale(at(11, 1))
    assert monitor.data_is_stale(at(11, 4))
    assert not monitor.data_is_hard_stale(at(11, 4))
    assert monitor.data_is_hard_stale(at(11, 15))


def test_a_dead_heartbeat_is_reported_separately_from_stale_data(monitor):
    monitor.heartbeat(at())
    assert monitor.report(at(11, 1)).status is HealthStatus.OK

    report = monitor.report(at(11, 5))
    beat = next(c for c in report.checks if c.name == "heartbeat")
    assert beat.status is HealthStatus.DOWN
    assert report.status is HealthStatus.DOWN


def test_a_disconnected_broker_takes_the_system_down(monitor):
    monitor.on_broker_state(False)
    report = monitor.report(at())
    assert report.status is HealthStatus.DOWN
    assert not report.may_trade


def test_an_engaged_kill_switch_takes_the_system_down(monitor):
    monitor.kill_switch.trip("operator")
    report = monitor.report(at())
    assert report.status is HealthStatus.DOWN
    assert "operator" in next(c.detail for c in report.checks if c.name == "kill_switch")


def test_a_run_of_errors_trips_the_kill_switch(monitor):
    assert not monitor.record_error("broker timeout", at=at())
    assert not monitor.record_error("broker timeout", at=at(11, 1))
    assert monitor.record_error("broker timeout", at=at(11, 2))
    assert monitor.kill_switch.is_active()
    assert len(monitor.recent_errors()) == 3


def test_a_success_resets_the_error_run(monitor):
    monitor.record_error("a", at=at())
    monitor.record_error("b", at=at())
    monitor.record_success()
    assert not monitor.record_error("c", at=at())
    assert not monitor.kill_switch.is_active()


def test_the_health_report_serialises_for_the_dashboard(monitor):
    monitor.on_bar(at(), arrived_at=at())
    payload = monitor.report(at()).to_dict()
    assert payload["status"] == "OK"
    assert payload["may_trade"] is True
    assert {c["name"] for c in payload["checks"]} == {
        "market_data", "heartbeat", "broker", "kill_switch"
    }
    json.dumps(payload)  # must be JSON-serialisable


def test_monitoring_cannot_open_a_position(monitor):
    forbidden = {"place_order", "buy", "sell", "enter", "submit", "broker_place"}
    assert forbidden.isdisjoint(dir(monitor))


# ============================================================ dashboard


@pytest.fixture
def served(tmp_path):
    """A running dashboard on an OS-assigned port, with a real kill switch behind STOP."""
    switch = KillSwitch(tmp_path / "K.flag")
    state = DashboardState()
    state.update(
        snapshot={"state": "RUNNING", "paper": True, "strategy": "orb_breakout",
                  "instrument": "MNQ", "broker": "simulated", "equity": 50_000.0,
                  "position": None, "risk": {"daily_pnl": -120.0}},
        health={"status": "OK", "may_trade": True, "checks": []},
        trades=[{"entry_time": at().isoformat(), "net_pnl": 12.5}],
        rejections=[{"stage": "RISK", "reason": "MAX_TRADES_PER_DAY", "n": 3}],
        equity=[{"ts": at().isoformat(), "equity": 50_000.0}],
        logs=[{"ts": at().isoformat(), "level": "INFO", "kind": "STARTED", "detail": ""}],
    )

    def on_stop(reason):
        switch.trip(reason, by="dashboard")
        return {"ok": True, "detail": switch.state().describe()}

    dashboard = Dashboard(state, host="127.0.0.1", port=0, on_stop=on_stop)
    dashboard.start()
    yield dashboard, switch, state
    dashboard.stop()


def get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_the_page_is_served_and_is_self_contained(served):
    dashboard, _, _ = served
    with urllib.request.urlopen(dashboard.url, timeout=5) as response:
        body = response.read().decode()
        assert response.status == 200
        assert response.headers["Content-Security-Policy"]
    assert "STOP TRADING" in body
    assert "Prop account" in body
    assert "evaluation pass rate" in body
    assert "http://" not in body.replace('http://127.0.0.1', '')
    assert "<script src=" not in body, "the page must not load anything external"
    assert 'id="resume"' not in body
    assert "/api/resume" not in body
    assert "const esc =" in body, "journal/broker labels must be HTML-escaped"


def test_the_state_endpoint_returns_everything_the_page_shows(served):
    dashboard, _, _ = served
    status, payload = get(dashboard.url + "api/state")
    assert status == 200
    for key in ("snapshot", "health", "trades", "rejections", "equity", "logs"):
        assert key in payload
    assert payload["snapshot"]["strategy"] == "orb_breakout"


def test_the_health_endpoint_is_available_on_its_own(served):
    dashboard, _, _ = served
    status, payload = get(dashboard.url + "api/health")
    assert status == 200 and payload["status"] == "OK"


def test_the_stop_button_trips_the_kill_switch(served):
    dashboard, switch, _ = served
    assert not switch.is_active()

    status, payload = post(dashboard.url + "api/stop", {"reason": "operator panicked"})
    assert status == 200 and payload["ok"]
    assert switch.is_active()
    assert "operator panicked" in switch.state().reason


def test_dashboard_cannot_clear_the_kill_switch(served):
    dashboard, switch, _ = served
    post(dashboard.url + "api/stop", {"reason": "stop"})
    assert switch.is_active()

    with pytest.raises(urllib.error.HTTPError) as err:
        post(dashboard.url + "api/resume", {"reason": "all clear"})
    assert err.value.code == 404
    assert switch.is_active()


def test_there_is_no_endpoint_that_can_place_a_trade(served):
    """The API surface is the security boundary; enumerate it."""
    dashboard, _, _ = served
    for path in ("api/order", "api/buy", "api/sell", "api/place", "api/position",
                 "api/config", "api/limits", "api/resume"):
        with pytest.raises(urllib.error.HTTPError) as err:
            post(dashboard.url + path, {})
        assert err.value.code == 404


def test_unknown_get_routes_are_not_found(served):
    dashboard, _, _ = served
    with pytest.raises(urllib.error.HTTPError) as err:
        get(dashboard.url + "api/secrets")
    assert err.value.code == 404


def test_state_updates_are_visible_on_the_next_request(served):
    dashboard, _, state = served
    state.update(snapshot={"state": "IN_POSITION", "paper": True,
                           "position": {"side": "BUY", "quantity": 2}})
    _, payload = get(dashboard.url + "api/state")
    assert payload["snapshot"]["state"] == "IN_POSITION"
    assert payload["snapshot"]["position"]["quantity"] == 2


def test_state_reads_are_deep_copies_of_nested_account_state():
    state = DashboardState()
    original = {"prop_account": {"current_balance_usd": 50_000}}
    state.update(snapshot=original)
    first = state.read()
    first["snapshot"]["prop_account"]["current_balance_usd"] = 1
    assert state.read()["snapshot"]["prop_account"]["current_balance_usd"] == 50_000


def test_the_dashboard_refuses_to_bind_to_a_routable_address():
    with pytest.raises(ValueError, match="no authentication"):
        Dashboard(DashboardState(), host="0.0.0.0", port=0)


def test_the_config_also_refuses_a_routable_bind_address():
    with pytest.raises(ConfigError, match="loopback"):
        DashboardConfig(host="192.168.1.10", port=8787).validate("dashboard")


def test_malformed_post_bodies_do_not_crash_the_server(served):
    dashboard, switch, _ = served
    request = urllib.request.Request(
        dashboard.url + "api/stop", data=b"{not json", method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    assert switch.is_active(), "a malformed body still stops; STOP fails safe"
