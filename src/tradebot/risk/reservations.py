"""Durable fail-closed state for an entry that may have reached a broker.

An approval token is intentionally process-local, but a submitted entry is not: a timeout
or crash can leave a working order or position at the venue.  This module persists the
smallest conservative fact needed across restart.  It contains no broker dependency, so
the guard can use the same abstraction with a local file today and a transactional store
later.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, runtime_checkable


_SCHEMA_VERSION = 1


class PendingEntryState(str, Enum):
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    TERMINAL_REPORTED = "TERMINAL_REPORTED"


@dataclass(frozen=True, slots=True)
class PendingEntryReservation:
    """Worst-case reservation retained until fresh venue truth proves it safe to clear."""

    order_id: str
    broker_order_id: str | None
    state: PendingEntryState
    reserved_at: datetime
    updated_at: datetime
    quantity: int
    risk_usd: float
    instrument: str = ""

    def validate(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("pending entry order_id must not be empty")
        if self.broker_order_id is not None and (
            not isinstance(self.broker_order_id, str) or not self.broker_order_id.strip()
        ):
            raise ValueError("pending entry broker_order_id must be null or non-empty")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("pending entry instrument must not be empty")
        if not isinstance(self.state, PendingEntryState):
            raise ValueError("pending entry state is invalid")
        for label, moment in (
            ("reserved_at", self.reserved_at),
            ("updated_at", self.updated_at),
        ):
            if (
                not isinstance(moment, datetime)
                or moment.tzinfo is None
                or moment.utcoffset() is None
            ):
                raise ValueError(f"pending entry {label} must be timezone-aware")
        if self.updated_at < self.reserved_at:
            raise ValueError("pending entry updated_at cannot precede reserved_at")
        if (
            not isinstance(self.quantity, int)
            or isinstance(self.quantity, bool)
            or self.quantity <= 0
        ):
            raise ValueError("pending entry quantity must be a positive integer")
        if (
            isinstance(self.risk_usd, bool)
            or not isinstance(self.risk_usd, (int, float))
            or not math.isfinite(float(self.risk_usd))
            or self.risk_usd <= 0
        ):
            raise ValueError("pending entry risk_usd must be finite and positive")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "state": self.state.value,
            "reserved_at": self.reserved_at.isoformat(timespec="microseconds"),
            "updated_at": self.updated_at.isoformat(timespec="microseconds"),
            "quantity": self.quantity,
            "risk_usd": float(self.risk_usd),
            "instrument": self.instrument,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PendingEntryReservation:
        if not isinstance(payload, dict):
            raise ValueError("pending entry payload must be an object")
        expected = {
            "order_id",
            "broker_order_id",
            "state",
            "reserved_at",
            "updated_at",
            "quantity",
            "risk_usd",
            "instrument",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            unknown = sorted(set(payload) - expected)
            raise ValueError(
                f"pending entry payload keys differ; missing={missing}, unknown={unknown}"
            )
        try:
            reservation = cls(
                order_id=payload["order_id"],
                broker_order_id=payload["broker_order_id"],
                state=PendingEntryState(payload["state"]),
                reserved_at=datetime.fromisoformat(payload["reserved_at"]),
                updated_at=datetime.fromisoformat(payload["updated_at"]),
                quantity=payload["quantity"],
                risk_usd=payload["risk_usd"],
                instrument=payload["instrument"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pending entry payload: {exc}") from exc
        reservation.validate()
        return reservation


class ReservationStoreError(RuntimeError):
    """Configured reservation state could not be read or durably changed."""


@runtime_checkable
class PendingEntryReservationStore(Protocol):
    def load(self) -> PendingEntryReservation | None: ...

    def save(self, reservation: PendingEntryReservation) -> None: ...

    def clear(self) -> None: ...


class FilePendingEntryReservationStore:
    """Versioned JSON store using write/fsync/atomic-replace in one directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> PendingEntryReservation | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ReservationStoreError(
                f"could not read pending-entry reservation at {self.path}: {exc}"
            ) from exc
        try:
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("reservation document must be an object")
            if set(document) != {"schema_version", "reservation"}:
                raise ValueError("reservation document has missing or unknown keys")
            if document["schema_version"] != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported reservation schema version {document['schema_version']!r}"
                )
            return PendingEntryReservation.from_dict(document["reservation"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReservationStoreError(
                f"pending-entry reservation at {self.path} is corrupt or incompatible: {exc}"
            ) from exc

    def save(self, reservation: PendingEntryReservation) -> None:
        if not isinstance(reservation, PendingEntryReservation):
            raise ReservationStoreError("can only persist a PendingEntryReservation")
        try:
            payload = json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "reservation": reservation.to_dict(),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
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
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass
        except (OSError, TypeError, ValueError) as exc:
            raise ReservationStoreError(
                f"could not persist pending-entry reservation at {self.path}: {exc}"
            ) from exc

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise ReservationStoreError(
                f"could not clear pending-entry reservation at {self.path}: {exc}"
            ) from exc


__all__ = [
    "FilePendingEntryReservationStore",
    "PendingEntryReservation",
    "PendingEntryReservationStore",
    "PendingEntryState",
    "ReservationStoreError",
]
