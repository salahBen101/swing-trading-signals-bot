"""Strict durable state for the personal-risk gate.

The trade journal is an audit record, not a safe source from which to reconstruct every
in-flight risk decision after a crash.  This module owns a deliberately small, versioned
snapshot that can later replace ``limits.RiskState`` without changing its existing field
or ``snapshot`` interface.

The persisted binding prevents a state file from being reused for a different instrument,
risk policy, broker route, or account.  Unknown fields and malformed values fail closed;
there is intentionally no best-effort migration or permissive coercion here.
"""

from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, runtime_checkable

if os.name == "nt":  # pragma: no cover - platform-specific branch
    import msvcrt
else:  # pragma: no cover - platform-specific branch
    import fcntl


_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_IDEMPOTENCY_IDS = 256
_MAX_RECENT_FINGERPRINTS = 256


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_aware(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _optional_identifier(value: object, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError(f"{label} must be null or a non-empty string")


def _validate_identifier_tuple(value: object, label: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    if len(value) > _MAX_IDEMPOTENCY_IDS:
        raise ValueError(
            f"{label} may contain at most {_MAX_IDEMPOTENCY_IDS} identifiers"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicate identifiers")


def _require_exact_keys(payload: object, expected: set[str], label: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} keys differ; missing={missing}, unknown={unknown}"
        )
    return payload


@contextmanager
def _exclusive_store_lock(path: Path):
    """Serialize compare-and-swap updates across processes using a sibling lock file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        raise RiskStateStoreError(
            f"could not open personal-risk lock for {path}: {exc}"
        ) from exc
    with handle:
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise RiskStateStoreError(
                f"could not lock personal-risk state at {path}: {exc}"
            ) from exc
        try:
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise RiskStateStoreError(
                    f"could not unlock personal-risk state at {path}: {exc}"
                ) from exc


def _binding_update_is_monotonic(
    previous: RiskStateBinding,
    proposed: RiskStateBinding,
) -> bool:
    """Permit only first-use identity binding; an established identity cannot rotate."""

    if (
        previous.instrument != proposed.instrument
        or previous.risk_policy_sha256 != proposed.risk_policy_sha256
        or previous.deployment_context_id != proposed.deployment_context_id
    ):
        return False
    for old, new in (
        (previous.broker_account_id, proposed.broker_account_id),
        (previous.broker_name, proposed.broker_name),
        (previous.broker_is_paper, proposed.broker_is_paper),
        (previous.broker_execution_route, proposed.broker_execution_route),
    ):
        if old is not None and old != new:
            return False
    return True


@dataclass(slots=True)
class RiskState:
    """Durable personal-risk bookkeeping.

    The fields through ``recent_fingerprints`` intentionally match the former in-memory
    state in :mod:`tradebot.risk.limits`.  The remaining fields make entry accounting,
    fill/trade application, and the post-loss size ceiling recoverable and idempotent.
    """

    equity: float
    peak_equity: float
    session_date: date | None = None
    session_start_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_r: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    open_position_id: str | None = None
    recent_fingerprints: dict[str, datetime] = field(default_factory=dict)

    revision: int = 0
    updated_at: datetime = field(default_factory=_utc_now)

    active_entry_order_id: str | None = None
    active_entry_intent_id: str | None = None
    active_entry_fingerprint: str | None = None
    active_entry_risk_usd: float = 0.0
    active_approved_quantity: int = 0
    entry_ever_filled: bool = False

    consumed_entry_order_ids: tuple[str, ...] = ()
    applied_trade_ids: tuple[str, ...] = ()

    last_trade_id: str | None = None
    last_trade_closed_at: datetime | None = None
    last_trade_net_pnl_usd: float | None = None
    last_trade_contracts: int = 0
    last_trade_was_loss: bool = False
    last_trade_approved_risk_usd: float = 0.0
    last_trade_quantity: int = 0

    def validate(self) -> None:
        for label in (
            "equity",
            "peak_equity",
            "session_start_equity",
            "daily_realized_pnl",
            "daily_r",
        ):
            if not _finite_number(getattr(self, label)):
                raise ValueError(f"risk state {label} must be finite")

        if self.session_date is not None and type(self.session_date) is not date:
            raise ValueError("risk state session_date must be null or a date")
        for label in ("consecutive_losses", "trades_today", "revision"):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"risk state {label} must be a non-negative integer")
        if self.cooldown_until is not None and not _is_aware(self.cooldown_until):
            raise ValueError("risk state cooldown_until must be null or timezone-aware")
        if not _is_aware(self.updated_at):
            raise ValueError("risk state updated_at must be timezone-aware")
        if type(self.halted) is not bool:
            raise ValueError("risk state halted must be boolean")
        if not isinstance(self.halt_reason, str):
            raise ValueError("risk state halt_reason must be a string")
        if self.halted and not self.halt_reason.strip():
            raise ValueError("a halted risk state must include a halt_reason")
        if not self.halted and self.halt_reason:
            raise ValueError("a non-halted risk state must not include a halt_reason")
        _optional_identifier(self.open_position_id, "risk state open_position_id")

        if type(self.recent_fingerprints) is not dict:
            raise ValueError("risk state recent_fingerprints must be an object")
        if len(self.recent_fingerprints) > _MAX_RECENT_FINGERPRINTS:
            raise ValueError(
                "risk state recent_fingerprints exceeds the bounded retention limit"
            )
        for fingerprint, seen_at in self.recent_fingerprints.items():
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                raise ValueError(
                    "risk state recent_fingerprints keys must be non-empty strings"
                )
            if not _is_aware(seen_at):
                raise ValueError(
                    "risk state recent_fingerprints values must be timezone-aware"
                )

        for label in (
            "active_entry_order_id",
            "active_entry_intent_id",
            "active_entry_fingerprint",
        ):
            _optional_identifier(getattr(self, label), f"risk state {label}")
        if not _finite_number(self.active_entry_risk_usd) or self.active_entry_risk_usd < 0:
            raise ValueError("risk state active_entry_risk_usd must be finite and non-negative")
        if type(self.active_approved_quantity) is not int or self.active_approved_quantity < 0:
            raise ValueError(
                "risk state active_approved_quantity must be a non-negative integer"
            )
        if type(self.entry_ever_filled) is not bool:
            raise ValueError("risk state entry_ever_filled must be boolean")

        active = self.active_entry_order_id is not None
        if active:
            if self.active_approved_quantity <= 0:
                raise ValueError("an active entry must have a positive approved quantity")
            if self.active_entry_risk_usd <= 0:
                raise ValueError("an active entry must have positive approved risk")
        elif (
            self.active_entry_intent_id is not None
            or self.active_entry_fingerprint is not None
            or self.active_entry_risk_usd != 0
            or self.active_approved_quantity != 0
            or self.entry_ever_filled
        ):
            raise ValueError("inactive entry metadata must be empty")

        _validate_identifier_tuple(
            self.consumed_entry_order_ids, "risk state consumed_entry_order_ids"
        )
        _validate_identifier_tuple(
            self.applied_trade_ids, "risk state applied_trade_ids"
        )

        _optional_identifier(self.last_trade_id, "risk state last_trade_id")
        if self.last_trade_closed_at is not None and not _is_aware(
            self.last_trade_closed_at
        ):
            raise ValueError(
                "risk state last_trade_closed_at must be null or timezone-aware"
            )
        if self.last_trade_net_pnl_usd is not None and not _finite_number(
            self.last_trade_net_pnl_usd
        ):
            raise ValueError("risk state last_trade_net_pnl_usd must be null or finite")
        if type(self.last_trade_contracts) is not int or self.last_trade_contracts < 0:
            raise ValueError("risk state last_trade_contracts must be a non-negative integer")
        if type(self.last_trade_was_loss) is not bool:
            raise ValueError("risk state last_trade_was_loss must be boolean")
        if (
            not _finite_number(self.last_trade_approved_risk_usd)
            or self.last_trade_approved_risk_usd < 0
        ):
            raise ValueError(
                "risk state last_trade_approved_risk_usd must be finite and non-negative"
            )
        if type(self.last_trade_quantity) is not int or self.last_trade_quantity < 0:
            raise ValueError("risk state last_trade_quantity must be a non-negative integer")

        has_last_trade = self.last_trade_id is not None
        if has_last_trade:
            if self.last_trade_closed_at is None or self.last_trade_net_pnl_usd is None:
                raise ValueError("last trade id requires its close time and net P&L")
            if self.last_trade_contracts <= 0 or self.last_trade_quantity <= 0:
                raise ValueError("last trade quantities must be positive")
            if self.last_trade_contracts != self.last_trade_quantity:
                raise ValueError("last trade quantity fields must agree")
            if self.last_trade_approved_risk_usd <= 0:
                raise ValueError("last trade approved risk must be positive")
            if self.last_trade_was_loss is not (self.last_trade_net_pnl_usd <= 0):
                raise ValueError("last_trade_was_loss must agree with last trade net P&L")
        elif (
            self.last_trade_closed_at is not None
            or self.last_trade_net_pnl_usd is not None
            or self.last_trade_contracts != 0
            or self.last_trade_was_loss
            or self.last_trade_approved_risk_usd != 0
            or self.last_trade_quantity != 0
        ):
            raise ValueError("last trade metadata must be empty when last_trade_id is null")

    def snapshot(self) -> dict:
        """Return the legacy dashboard/audit view used by the current risk engine."""

        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "session_date": str(self.session_date) if self.session_date else None,
            "session_start_equity": self.session_start_equity,
            "daily_realized_pnl": self.daily_realized_pnl,
            "daily_r": self.daily_r,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat()
            if self.cooldown_until
            else None,
            "trades_today": self.trades_today,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    def to_dict(self) -> dict:
        self.validate()
        return {
            "equity": float(self.equity),
            "peak_equity": float(self.peak_equity),
            "session_date": self.session_date.isoformat() if self.session_date else None,
            "session_start_equity": float(self.session_start_equity),
            "daily_realized_pnl": float(self.daily_realized_pnl),
            "daily_r": float(self.daily_r),
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat(timespec="microseconds")
            if self.cooldown_until
            else None,
            "trades_today": self.trades_today,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "open_position_id": self.open_position_id,
            "recent_fingerprints": {
                key: value.isoformat(timespec="microseconds")
                for key, value in sorted(self.recent_fingerprints.items())
            },
            "revision": self.revision,
            "updated_at": self.updated_at.isoformat(timespec="microseconds"),
            "active_entry_order_id": self.active_entry_order_id,
            "active_entry_intent_id": self.active_entry_intent_id,
            "active_entry_fingerprint": self.active_entry_fingerprint,
            "active_entry_risk_usd": float(self.active_entry_risk_usd),
            "active_approved_quantity": self.active_approved_quantity,
            "entry_ever_filled": self.entry_ever_filled,
            "consumed_entry_order_ids": list(self.consumed_entry_order_ids),
            "applied_trade_ids": list(self.applied_trade_ids),
            "last_trade_id": self.last_trade_id,
            "last_trade_closed_at": self.last_trade_closed_at.isoformat(
                timespec="microseconds"
            )
            if self.last_trade_closed_at
            else None,
            "last_trade_net_pnl_usd": float(self.last_trade_net_pnl_usd)
            if self.last_trade_net_pnl_usd is not None
            else None,
            "last_trade_contracts": self.last_trade_contracts,
            "last_trade_was_loss": self.last_trade_was_loss,
            "last_trade_approved_risk_usd": float(self.last_trade_approved_risk_usd),
            "last_trade_quantity": self.last_trade_quantity,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RiskState:
        expected = {
            "equity",
            "peak_equity",
            "session_date",
            "session_start_equity",
            "daily_realized_pnl",
            "daily_r",
            "consecutive_losses",
            "cooldown_until",
            "trades_today",
            "halted",
            "halt_reason",
            "open_position_id",
            "recent_fingerprints",
            "revision",
            "updated_at",
            "active_entry_order_id",
            "active_entry_intent_id",
            "active_entry_fingerprint",
            "active_entry_risk_usd",
            "active_approved_quantity",
            "entry_ever_filled",
            "consumed_entry_order_ids",
            "applied_trade_ids",
            "last_trade_id",
            "last_trade_closed_at",
            "last_trade_net_pnl_usd",
            "last_trade_contracts",
            "last_trade_was_loss",
            "last_trade_approved_risk_usd",
            "last_trade_quantity",
        }
        raw = _require_exact_keys(payload, expected, "risk state payload")
        fingerprints = raw["recent_fingerprints"]
        if not isinstance(fingerprints, dict):
            raise ValueError("risk state recent_fingerprints must be an object")
        if type(raw["consumed_entry_order_ids"]) is not list:
            raise ValueError("risk state consumed_entry_order_ids must be an array")
        if type(raw["applied_trade_ids"]) is not list:
            raise ValueError("risk state applied_trade_ids must be an array")
        try:
            state = cls(
                equity=raw["equity"],
                peak_equity=raw["peak_equity"],
                session_date=date.fromisoformat(raw["session_date"])
                if raw["session_date"] is not None
                else None,
                session_start_equity=raw["session_start_equity"],
                daily_realized_pnl=raw["daily_realized_pnl"],
                daily_r=raw["daily_r"],
                consecutive_losses=raw["consecutive_losses"],
                cooldown_until=datetime.fromisoformat(raw["cooldown_until"])
                if raw["cooldown_until"] is not None
                else None,
                trades_today=raw["trades_today"],
                halted=raw["halted"],
                halt_reason=raw["halt_reason"],
                open_position_id=raw["open_position_id"],
                recent_fingerprints={
                    key: datetime.fromisoformat(value)
                    for key, value in fingerprints.items()
                },
                revision=raw["revision"],
                updated_at=datetime.fromisoformat(raw["updated_at"]),
                active_entry_order_id=raw["active_entry_order_id"],
                active_entry_intent_id=raw["active_entry_intent_id"],
                active_entry_fingerprint=raw["active_entry_fingerprint"],
                active_entry_risk_usd=raw["active_entry_risk_usd"],
                active_approved_quantity=raw["active_approved_quantity"],
                entry_ever_filled=raw["entry_ever_filled"],
                consumed_entry_order_ids=tuple(raw["consumed_entry_order_ids"]),
                applied_trade_ids=tuple(raw["applied_trade_ids"]),
                last_trade_id=raw["last_trade_id"],
                last_trade_closed_at=datetime.fromisoformat(raw["last_trade_closed_at"])
                if raw["last_trade_closed_at"] is not None
                else None,
                last_trade_net_pnl_usd=raw["last_trade_net_pnl_usd"],
                last_trade_contracts=raw["last_trade_contracts"],
                last_trade_was_loss=raw["last_trade_was_loss"],
                last_trade_approved_risk_usd=raw["last_trade_approved_risk_usd"],
                last_trade_quantity=raw["last_trade_quantity"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid risk state payload: {exc}") from exc
        state.validate()
        return state


@dataclass(frozen=True, slots=True)
class RiskStateBinding:
    """Identity whose personal-risk history a state file represents."""

    instrument: str
    risk_policy_sha256: str
    broker_account_id: str | None = None
    broker_name: str | None = None
    broker_is_paper: bool | None = None
    broker_execution_route: str | None = None
    deployment_context_id: str | None = None

    def validate(self) -> None:
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("risk state binding instrument must be non-empty")
        if (
            not isinstance(self.risk_policy_sha256, str)
            or _SHA256.fullmatch(self.risk_policy_sha256) is None
        ):
            raise ValueError(
                "risk state binding risk_policy_sha256 must be a lowercase SHA-256 digest"
            )
        _optional_identifier(
            self.broker_account_id, "risk state binding broker_account_id"
        )
        _optional_identifier(self.broker_name, "risk state binding broker_name")
        _optional_identifier(
            self.broker_execution_route,
            "risk state binding broker_execution_route",
        )
        _optional_identifier(
            self.deployment_context_id,
            "risk state binding deployment_context_id",
        )
        if self.broker_is_paper is not None and type(self.broker_is_paper) is not bool:
            raise ValueError("risk state binding broker_is_paper must be null or boolean")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "instrument": self.instrument,
            "risk_policy_sha256": self.risk_policy_sha256,
            "broker_account_id": self.broker_account_id,
            "broker_name": self.broker_name,
            "broker_is_paper": self.broker_is_paper,
            "broker_execution_route": self.broker_execution_route,
            "deployment_context_id": self.deployment_context_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RiskStateBinding:
        raw = _require_exact_keys(
            payload,
            {
                "instrument",
                "risk_policy_sha256",
                "broker_account_id",
                "broker_name",
                "broker_is_paper",
                "broker_execution_route",
                "deployment_context_id",
            },
            "risk state binding",
        )
        binding = cls(
            instrument=raw["instrument"],
            risk_policy_sha256=raw["risk_policy_sha256"],
            broker_account_id=raw["broker_account_id"],
            broker_name=raw["broker_name"],
            broker_is_paper=raw["broker_is_paper"],
            broker_execution_route=raw["broker_execution_route"],
            deployment_context_id=raw["deployment_context_id"],
        )
        binding.validate()
        return binding


@dataclass(frozen=True, slots=True)
class StoredRiskState:
    binding: RiskStateBinding
    state: RiskState

    def validate(self) -> None:
        if not isinstance(self.binding, RiskStateBinding):
            raise ValueError("stored risk state binding has the wrong type")
        if not isinstance(self.state, RiskState):
            raise ValueError("stored risk state payload has the wrong type")
        self.binding.validate()
        self.state.validate()

    def to_dict(self) -> dict:
        self.validate()
        return {"binding": self.binding.to_dict(), "state": self.state.to_dict()}

    @classmethod
    def from_dict(cls, payload: object) -> StoredRiskState:
        raw = _require_exact_keys(payload, {"binding", "state"}, "stored risk state")
        stored = cls(
            binding=RiskStateBinding.from_dict(raw["binding"]),
            state=RiskState.from_dict(raw["state"]),
        )
        stored.validate()
        return stored


class RiskStateStoreError(RuntimeError):
    """Configured personal-risk state could not be read or durably changed."""


@runtime_checkable
class RiskStateStore(Protocol):
    def load(
        self, *, expected_binding: RiskStateBinding | None = None
    ) -> StoredRiskState | None: ...

    def initialize(self, stored_state: StoredRiskState) -> None: ...

    def save(self, stored_state: StoredRiskState) -> None: ...


class FileRiskStateStore:
    """Versioned JSON store using write/fsync/atomic-replace in one directory."""

    is_durable = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def initialization_marker_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.initialized")

    def load(
        self, *, expected_binding: RiskStateBinding | None = None
    ) -> StoredRiskState | None:
        if expected_binding is not None:
            try:
                expected_binding.validate()
            except (TypeError, ValueError) as exc:
                raise RiskStateStoreError(f"invalid expected risk-state binding: {exc}") from exc
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RiskStateStoreError(
                f"could not read personal-risk state at {self.path}: {exc}"
            ) from exc
        try:
            document = json.loads(raw)
            root = _require_exact_keys(
                document, {"schema_version", "stored_state"}, "risk state document"
            )
            if type(root["schema_version"]) is not int:
                raise ValueError("risk state schema version must be an integer")
            if root["schema_version"] != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported risk state schema version {root['schema_version']!r}"
                )
            stored = StoredRiskState.from_dict(root["stored_state"])
            if expected_binding is not None and stored.binding != expected_binding:
                raise ValueError("persisted risk state binding does not match this runtime")
            return stored
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RiskStateStoreError(
                f"personal-risk state at {self.path} is corrupt or incompatible: {exc}"
            ) from exc

    @staticmethod
    def _serialized_payload(stored_state: StoredRiskState) -> str:
        if not isinstance(stored_state, StoredRiskState):
            raise RiskStateStoreError("can only persist a StoredRiskState")
        return json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "stored_state": stored_state.to_dict(),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"

    def _write_payload_locked(self, payload: str) -> None:
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            if os.name != "nt":  # directory fsync is not supported on Windows
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def initialize(self, stored_state: StoredRiskState) -> None:
        """Create a fresh-account state exactly once for this durable store path."""

        try:
            payload = self._serialized_payload(stored_state)
            marker_payload = json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "initialized_at": stored_state.state.updated_at.isoformat(
                        timespec="microseconds"
                    ),
                    "binding": stored_state.binding.to_dict(),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _exclusive_store_lock(self.path):
                if self.load() is not None:
                    raise ValueError("personal-risk state is already initialized")
                try:
                    with self.initialization_marker_path.open(
                        "x", encoding="utf-8", newline="\n"
                    ) as marker:
                        marker.write(marker_payload)
                        marker.flush()
                        os.fsync(marker.fileno())
                except FileExistsError as exc:
                    raise ValueError(
                        "personal-risk state was initialized before; missing state "
                        "requires operator recovery, not bootstrap"
                    ) from exc
                # The marker is deliberately written first. A crash after this point can
                # require manual recovery, but it can never silently reset daily history.
                self._write_payload_locked(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RiskStateStoreError(
                f"could not initialize personal-risk state at {self.path}: {exc}"
            ) from exc

    def save(self, stored_state: StoredRiskState) -> None:
        try:
            payload = self._serialized_payload(stored_state)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _exclusive_store_lock(self.path):
                current = self.load()
                if current is None:
                    raise ValueError(
                        "personal-risk state is not initialized; use the one-time "
                        "initialize operation"
                    )
                expected_revision = current.state.revision + 1
                if stored_state.state.revision != expected_revision:
                    raise ValueError(
                        "stale personal-risk revision "
                        f"{stored_state.state.revision}; expected {expected_revision}"
                    )
                if not _binding_update_is_monotonic(
                    current.binding,
                    stored_state.binding,
                ):
                    raise ValueError(
                        "personal-risk state binding cannot rotate or become less specific"
                    )
                self._write_payload_locked(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RiskStateStoreError(
                f"could not persist personal-risk state at {self.path}: {exc}"
            ) from exc


__all__ = [
    "FileRiskStateStore",
    "RiskState",
    "RiskStateBinding",
    "RiskStateStore",
    "RiskStateStoreError",
    "StoredRiskState",
]
