"""The emergency kill switch.

Backed by a file rather than a variable, for two reasons. A flag on disk survives a crash
and a restart, so a process that died mid-incident comes back refusing to trade instead of
cheerfully resuming. And it can be tripped by something that is not this process — the
dashboard, a scheduled job, or a human with a terminal — which is the whole point of an
emergency stop.

Clearing it is deliberately manual. An automatic reset would turn the one control that
means "stop until a person looks at this" into a speed bump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.clock import MARKET_TZ, Clock, SystemClock


@dataclass(frozen=True, slots=True)
class KillSwitchState:
    active: bool
    reason: str = ""
    tripped_at: str = ""
    tripped_by: str = ""

    def describe(self) -> str:
        if not self.active:
            return "clear"
        return f"ACTIVE since {self.tripped_at} ({self.tripped_by}): {self.reason}"


class KillSwitch:
    def __init__(
        self,
        flag_file: str | Path = "logs/KILL_SWITCH.flag",
        *,
        max_consecutive_errors: int = 5,
        clock: Clock | None = None,
    ) -> None:
        self.path = Path(flag_file)
        self.max_consecutive_errors = max_consecutive_errors
        self._clock = clock or SystemClock()
        self._consecutive_errors = 0

    # -- state ---------------------------------------------------------------------------
    def state(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState(active=False)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A flag file we cannot parse still means someone put a flag file there.
            # Failing open here would defeat the entire mechanism.
            return KillSwitchState(True, "unreadable flag file", "", "unknown")
        return KillSwitchState(
            active=True,
            reason=payload.get("reason", ""),
            tripped_at=payload.get("tripped_at", ""),
            tripped_by=payload.get("tripped_by", ""),
        )

    def is_active(self) -> bool:
        return self.path.exists()

    # -- control -------------------------------------------------------------------------
    def trip(self, reason: str, *, by: str = "system") -> KillSwitchState:
        """Engage the switch. Idempotent — the first reason is kept.

        Keeping the first reason matters during a cascade: the third error is rarely the
        interesting one, and overwriting would bury the trigger.
        """
        if self.path.exists():
            return self.state()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reason": reason,
            "tripped_at": self._now().isoformat(timespec="seconds"),
            "tripped_by": by,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.state()

    def clear(self, *, by: str = "human") -> None:
        """Release the switch. Intended for a person, never for automatic recovery."""
        self.path.unlink(missing_ok=True)
        self._consecutive_errors = 0

    # -- error tracking ------------------------------------------------------------------
    def record_error(self, detail: str) -> bool:
        """Count a runtime error; trip once they run consecutively past the threshold.

        Returns True if this error tripped the switch. A run of errors usually means the
        broker, the feed or the clock is broken, and continuing to send orders into that
        is how a bad afternoon becomes an expensive one.
        """
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_consecutive_errors:
            self.trip(
                f"{self._consecutive_errors} consecutive errors, last: {detail}",
                by="error-monitor",
            )
            return True
        return False

    def record_success(self) -> None:
        self._consecutive_errors = 0

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    def _now(self) -> datetime:
        return self._clock.now().astimezone(MARKET_TZ)
