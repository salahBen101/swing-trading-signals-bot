"""Health monitoring.

Answers one question for the dashboard and one for the runner: *is this system in a fit
state to be trading right now?*

The three conditions that matter for an intraday bot, in order of how quietly they fail:

1. **Stale market data.** The most dangerous, because nothing looks wrong. The process is
   up, the strategy is evaluating, the risk engine is happy — and the last price it saw was
   eleven minutes ago. Managing a stop against a price that no longer exists is worse than
   not trading at all.
2. **A dead heartbeat.** The loop has stopped advancing. Distinguished from staleness
   because the fix is different: one is a data problem, the other is a process problem.
3. **A run of errors.** Individually recoverable, collectively a sign that something
   structural is broken. Past a threshold this trips the kill switch.

Monitoring can only ever *reduce* permission. It has no path to opening a position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.clock import Clock, SystemClock, ensure_aware
from ..core.types import HealthStatus
from ..risk.killswitch import KillSwitch


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: HealthStatus
    detail: str

    @property
    def ok(self) -> bool:
        return self.status is HealthStatus.OK


@dataclass(frozen=True, slots=True)
class HealthReport:
    status: HealthStatus
    checks: tuple[HealthCheck, ...]
    timestamp: datetime

    @property
    def ok(self) -> bool:
        return self.status is HealthStatus.OK

    @property
    def may_trade(self) -> bool:
        """DEGRADED still trades — a slow feed is not a broken one. DOWN does not."""
        return self.status is not HealthStatus.DOWN

    def failures(self) -> list[HealthCheck]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "may_trade": self.may_trade,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {"name": c.name, "status": c.status.value, "detail": c.detail}
                for c in self.checks
            ],
        }


@dataclass(slots=True)
class _ErrorRecord:
    timestamp: datetime
    detail: str


class HealthMonitor:
    def __init__(
        self,
        *,
        bar_seconds: float,
        stale_multiple: float = 3.0,
        hard_stale_multiple: float = 10.0,
        heartbeat_timeout_seconds: float = 120.0,
        kill_switch: KillSwitch | None = None,
        clock: Clock | None = None,
        error_window: int = 20,
    ) -> None:
        self.bar_seconds = bar_seconds
        self.stale_multiple = stale_multiple
        self.hard_stale_multiple = hard_stale_multiple
        self.heartbeat_timeout = timedelta(seconds=heartbeat_timeout_seconds)
        self.kill_switch = kill_switch
        self.clock = clock or SystemClock()

        self.last_bar_at: datetime | None = None
        self.last_heartbeat_at: datetime | None = None
        self.broker_connected = True
        self.errors: deque[_ErrorRecord] = deque(maxlen=error_window)
        self.bars_seen = 0

    # ------------------------------------------------------------------ signals in

    def on_bar(self, bar_time: datetime, *, arrived_at: datetime | None = None) -> None:
        arrival = ensure_aware(arrived_at or self.clock.now())
        self.last_bar_at = arrival
        self.bars_seen += 1
        # Stamp the heartbeat at the same instant, not at wall-clock now. A caller that
        # says when the bar arrived is replaying or catching up, and letting the two
        # timestamps drift apart makes the heartbeat check fire on a healthy replay.
        self.heartbeat(arrival)

    def heartbeat(self, at: datetime | None = None) -> None:
        self.last_heartbeat_at = ensure_aware(at or self.clock.now())

    def on_broker_state(self, connected: bool) -> None:
        self.broker_connected = connected

    def record_error(self, detail: str, *, at: datetime | None = None) -> bool:
        """Record an error. Returns True if this one tripped the kill switch."""
        self.errors.append(_ErrorRecord(ensure_aware(at or self.clock.now()), detail))
        if self.kill_switch is not None:
            return self.kill_switch.record_error(detail)
        return False

    def record_success(self) -> None:
        if self.kill_switch is not None:
            self.kill_switch.record_success()

    # ------------------------------------------------------------------ report out

    def report(self, now: datetime | None = None) -> HealthReport:
        now = ensure_aware(now or self.clock.now())
        checks = [
            self._check_data(now),
            self._check_heartbeat(now),
            self._check_broker(),
            self._check_kill_switch(),
        ]
        if any(c.status is HealthStatus.DOWN for c in checks):
            overall = HealthStatus.DOWN
        elif any(c.status is HealthStatus.DEGRADED for c in checks):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.OK
        return HealthReport(status=overall, checks=tuple(checks), timestamp=now)

    def _check_data(self, now: datetime) -> HealthCheck:
        if self.last_bar_at is None:
            # Not started is not stale. Reporting a fresh process as DOWN would trip the
            # monitor before the first bar ever arrives.
            return HealthCheck("market_data", HealthStatus.OK, "no bars yet")
        gap = (now - self.last_bar_at).total_seconds()
        multiple = gap / self.bar_seconds
        detail = (
            f"last bar {gap:.0f}s ago ({multiple:.1f}x the {self.bar_seconds:g}s interval)"
        )
        if multiple >= self.hard_stale_multiple:
            return HealthCheck("market_data", HealthStatus.DOWN, detail)
        if multiple >= self.stale_multiple:
            return HealthCheck("market_data", HealthStatus.DEGRADED, detail)
        return HealthCheck("market_data", HealthStatus.OK, detail)

    def _check_heartbeat(self, now: datetime) -> HealthCheck:
        if self.last_heartbeat_at is None:
            return HealthCheck("heartbeat", HealthStatus.OK, "not started")
        gap = now - self.last_heartbeat_at
        detail = f"last beat {gap.total_seconds():.0f}s ago"
        if gap >= self.heartbeat_timeout:
            return HealthCheck("heartbeat", HealthStatus.DOWN, detail)
        return HealthCheck("heartbeat", HealthStatus.OK, detail)

    def _check_broker(self) -> HealthCheck:
        if self.broker_connected:
            return HealthCheck("broker", HealthStatus.OK, "connected")
        return HealthCheck("broker", HealthStatus.DOWN, "disconnected")

    def _check_kill_switch(self) -> HealthCheck:
        if self.kill_switch is None:
            return HealthCheck("kill_switch", HealthStatus.OK, "not configured")
        state = self.kill_switch.state()
        if state.active:
            return HealthCheck("kill_switch", HealthStatus.DOWN, state.describe())
        return HealthCheck(
            "kill_switch", HealthStatus.OK,
            f"clear ({self.kill_switch.consecutive_errors} consecutive errors)",
        )

    # ------------------------------------------------------------------ convenience

    def data_is_stale(self, now: datetime | None = None) -> bool:
        return self._check_data(ensure_aware(now or self.clock.now())).status is not HealthStatus.OK

    def data_is_hard_stale(self, now: datetime | None = None) -> bool:
        check = self._check_data(ensure_aware(now or self.clock.now()))
        return check.status is HealthStatus.DOWN

    def recent_errors(self, limit: int = 10) -> list[dict]:
        return [
            {"timestamp": e.timestamp.isoformat(), "detail": e.detail}
            for e in list(self.errors)[-limit:]
        ]
