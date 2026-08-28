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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, runtime_checkable

from ..core.types import OrderType, Side

if os.name == "nt":  # pragma: no cover - platform-specific branch
    import msvcrt
else:  # pragma: no cover - platform-specific branch
    import fcntl


_SCHEMA_VERSION = 3
_MAX_FILL_IDS = 256


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
    instrument: str
    side: Side
    order_type: OrderType
    limit_price: float | None
    protective_stop_price: float
    broker_account_id: str
    broker_execution_route: str
    cumulative_filled_quantity: int
    ever_fill_observed: bool
    observed_fill_ids: tuple[str, ...]
    revision: int

    def validate(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("pending entry order_id must not be empty")
        if self.broker_order_id is not None and (
            not isinstance(self.broker_order_id, str) or not self.broker_order_id.strip()
        ):
            raise ValueError("pending entry broker_order_id must be null or non-empty")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError("pending entry instrument must not be empty")
        if self.instrument != self.instrument.strip():
            raise ValueError("pending entry instrument must not contain outer whitespace")
        if type(self.side) is not Side:
            raise ValueError("pending entry side must be a signed Side")
        if type(self.order_type) is not OrderType:
            raise ValueError("pending entry order_type is invalid")
        if self.limit_price is not None and (
            isinstance(self.limit_price, bool)
            or not isinstance(self.limit_price, (int, float))
            or not math.isfinite(float(self.limit_price))
            or self.limit_price <= 0
        ):
            raise ValueError(
                "pending entry limit_price must be null or finite and positive"
            )
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if self.limit_price is None:
                raise ValueError(
                    f"pending entry {self.order_type.value} requires a limit_price"
                )
        elif self.limit_price is not None:
            raise ValueError(
                f"pending entry {self.order_type.value} cannot carry a limit_price"
            )
        if (
            isinstance(self.protective_stop_price, bool)
            or not isinstance(self.protective_stop_price, (int, float))
            or not math.isfinite(float(self.protective_stop_price))
            or self.protective_stop_price <= 0
        ):
            raise ValueError(
                "pending entry protective_stop_price must be finite and positive"
            )
        if self.limit_price is not None:
            stop_is_protective = (
                self.side is Side.BUY
                and self.protective_stop_price < self.limit_price
            ) or (
                self.side is Side.SELL
                and self.protective_stop_price > self.limit_price
            )
            if not stop_is_protective:
                raise ValueError(
                    "pending entry protective stop is not on the signed protective side "
                    "of its limit"
                )
        if (
            not isinstance(self.broker_account_id, str)
            or not self.broker_account_id.strip()
        ):
            raise ValueError("pending entry broker_account_id must not be empty")
        if self.broker_account_id != self.broker_account_id.strip():
            raise ValueError(
                "pending entry broker_account_id must not contain outer whitespace"
            )
        if (
            not isinstance(self.broker_execution_route, str)
            or not self.broker_execution_route.strip()
        ):
            raise ValueError("pending entry broker_execution_route must not be empty")
        if self.broker_execution_route != self.broker_execution_route.strip().casefold():
            raise ValueError(
                "pending entry broker_execution_route must be canonical lower-case text"
            )
        if not isinstance(self.state, PendingEntryState):
            raise ValueError("pending entry state is invalid")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision <= 0
        ):
            raise ValueError("pending entry revision must be a positive integer")
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
        if (
            not isinstance(self.cumulative_filled_quantity, int)
            or isinstance(self.cumulative_filled_quantity, bool)
            or self.cumulative_filled_quantity < 0
            or self.cumulative_filled_quantity > self.quantity
        ):
            raise ValueError(
                "pending entry cumulative_filled_quantity must be an integer between "
                "zero and quantity"
            )
        if type(self.ever_fill_observed) is not bool:
            raise ValueError("pending entry ever_fill_observed must be a boolean")
        if self.cumulative_filled_quantity > 0 and not self.ever_fill_observed:
            raise ValueError(
                "pending entry positive filled quantity requires fill evidence"
            )
        if self.state in {
            PendingEntryState.PARTIALLY_FILLED,
            PendingEntryState.FILLED,
        } and not self.ever_fill_observed:
            raise ValueError(
                f"pending entry {self.state.value} state requires fill evidence"
            )
        if (
            self.state is PendingEntryState.FILLED
            and self.cumulative_filled_quantity != self.quantity
        ):
            raise ValueError(
                "pending entry FILLED state requires the complete order quantity"
            )
        if not isinstance(self.observed_fill_ids, tuple):
            raise ValueError("pending entry observed_fill_ids must be a tuple")
        if len(self.observed_fill_ids) > _MAX_FILL_IDS:
            raise ValueError(
                f"pending entry may retain at most {_MAX_FILL_IDS} unique fill ids"
            )
        if any(
            not isinstance(fill_id, str)
            or not fill_id.strip()
            or fill_id != fill_id.strip()
            for fill_id in self.observed_fill_ids
        ):
            raise ValueError(
                "pending entry observed_fill_ids must contain non-empty stripped strings"
            )
        if len(set(self.observed_fill_ids)) != len(self.observed_fill_ids):
            raise ValueError("pending entry observed_fill_ids must be unique")
        if self.observed_fill_ids and not self.ever_fill_observed:
            raise ValueError(
                "pending entry observed_fill_ids require fill evidence"
            )

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
            "side": int(self.side),
            "order_type": self.order_type.value,
            "limit_price": (
                None if self.limit_price is None else float(self.limit_price)
            ),
            "protective_stop_price": float(self.protective_stop_price),
            "broker_account_id": self.broker_account_id,
            "broker_execution_route": self.broker_execution_route,
            "cumulative_filled_quantity": self.cumulative_filled_quantity,
            "ever_fill_observed": self.ever_fill_observed,
            "observed_fill_ids": list(self.observed_fill_ids),
            "revision": self.revision,
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
            "side",
            "order_type",
            "limit_price",
            "protective_stop_price",
            "broker_account_id",
            "broker_execution_route",
            "cumulative_filled_quantity",
            "ever_fill_observed",
            "observed_fill_ids",
            "revision",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            unknown = sorted(set(payload) - expected)
            raise ValueError(
                f"pending entry payload keys differ; missing={missing}, unknown={unknown}"
            )
        try:
            if not isinstance(payload["observed_fill_ids"], list):
                raise TypeError("observed_fill_ids must be a JSON array")
            if type(payload["side"]) is not int:
                raise TypeError("side must be a signed JSON integer")
            reservation = cls(
                order_id=payload["order_id"],
                broker_order_id=payload["broker_order_id"],
                state=PendingEntryState(payload["state"]),
                reserved_at=datetime.fromisoformat(payload["reserved_at"]),
                updated_at=datetime.fromisoformat(payload["updated_at"]),
                quantity=payload["quantity"],
                risk_usd=payload["risk_usd"],
                instrument=payload["instrument"],
                side=Side(payload["side"]),
                order_type=OrderType(payload["order_type"]),
                limit_price=payload["limit_price"],
                protective_stop_price=payload["protective_stop_price"],
                broker_account_id=payload["broker_account_id"],
                broker_execution_route=payload["broker_execution_route"],
                cumulative_filled_quantity=payload[
                    "cumulative_filled_quantity"
                ],
                ever_fill_observed=payload["ever_fill_observed"],
                observed_fill_ids=tuple(payload["observed_fill_ids"]),
                revision=payload["revision"],
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

    def save(
        self, reservation: PendingEntryReservation
    ) -> PendingEntryReservation: ...

    def clear(self, expected: PendingEntryReservation) -> None: ...


class FilePendingEntryReservationStore:
    """Versioned JSON store with an exclusive writer lease and monotonic CAS.

    Atomic replace protects readers from torn JSON. An OS advisory lock on the adjacent
    lock file serializes writers across processes and is released by the kernel on process
    death. ``revision`` and transition validation stop a guard that loaded stale state
    from erasing newer fill evidence.
    """

    is_durable = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def load(self) -> PendingEntryReservation | None:
        return self._load_unlocked()

    def _load_unlocked(self) -> PendingEntryReservation | None:
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
            if (
                type(document["schema_version"]) is not int
                or document["schema_version"] != _SCHEMA_VERSION
            ):
                raise ValueError(
                    f"unsupported reservation schema version {document['schema_version']!r}"
                )
            return PendingEntryReservation.from_dict(document["reservation"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReservationStoreError(
                f"pending-entry reservation at {self.path} is corrupt or incompatible: {exc}"
            ) from exc

    def save(
        self, reservation: PendingEntryReservation
    ) -> PendingEntryReservation:
        if not isinstance(reservation, PendingEntryReservation):
            raise ReservationStoreError("can only persist a PendingEntryReservation")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._exclusive_writer():
                current = self._load_unlocked()
                if current is None:
                    if reservation.revision != 1:
                        raise ValueError(
                            "a new reservation must begin at revision 1"
                        )
                else:
                    if reservation.revision == current.revision + 1:
                        self._validate_transition(current, reservation)
                    elif reservation.revision <= current.revision:
                        reservation = self._merge_stale_fill_evidence(
                            current,
                            reservation,
                        )
                    else:
                        raise ValueError(
                            f"future reservation revision {reservation.revision}; "
                            f"expected {current.revision + 1}"
                        )
                payload = json.dumps(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "reservation": reservation.to_dict(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ) + "\n"
                self._atomic_write(payload)
                return reservation
        except ReservationStoreError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ReservationStoreError(
                f"could not persist pending-entry reservation at {self.path}: {exc}"
            ) from exc

    def clear(self, expected: PendingEntryReservation) -> None:
        if not isinstance(expected, PendingEntryReservation):
            raise ReservationStoreError(
                "clearing a pending entry requires the exact expected reservation"
            )
        try:
            expected.validate()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._exclusive_writer():
                current = self._load_unlocked()
                if current is None:
                    raise ValueError(
                        "reservation disappeared before compare-and-clear"
                    )
                if current != expected:
                    raise ValueError(
                        "reservation changed before compare-and-clear"
                    )
                self.path.unlink()
                self._fsync_parent_directory()
        except ReservationStoreError:
            raise
        except (TypeError, ValueError) as exc:
            raise ReservationStoreError(
                f"could not clear pending-entry reservation at {self.path}: {exc}"
            ) from exc
        except OSError as exc:
            raise ReservationStoreError(
                f"could not clear pending-entry reservation at {self.path}: {exc}"
            ) from exc

    @contextmanager
    def _exclusive_writer(self):
        try:
            handle = self.lock_path.open("a+b")
        except OSError as exc:
            raise ReservationStoreError(
                f"could not open pending-entry writer lock at {self.lock_path}: {exc}"
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
                raise ReservationStoreError(
                    f"could not lock pending-entry state at {self.path}: {exc}"
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
                    raise ReservationStoreError(
                        f"could not unlock pending-entry state at {self.path}: {exc}"
                    ) from exc

    def _atomic_write(self, payload: str) -> None:
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
            self._fsync_parent_directory()
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _fsync_parent_directory(self) -> None:
        if os.name == "nt":
            return
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _merge_stale_fill_evidence(
        cls,
        current: PendingEntryReservation,
        stale: PendingEntryReservation,
    ) -> PendingEntryReservation:
        """Merge only monotonic economic evidence from a stale concurrent writer."""
        stale.validate()
        immutable_fields = (
            "order_id",
            "reserved_at",
            "quantity",
            "risk_usd",
            "instrument",
            "side",
            "order_type",
            "limit_price",
            "protective_stop_price",
            "broker_account_id",
            "broker_execution_route",
        )
        conflicts = tuple(
            field
            for field in immutable_fields
            if getattr(stale, field) != getattr(current, field)
        )
        if conflicts:
            raise ValueError(
                "stale reservation identity/risk conflicts with durable state: "
                + ", ".join(conflicts)
            )
        if (
            current.broker_order_id is not None
            and stale.broker_order_id is not None
            and current.broker_order_id != stale.broker_order_id
        ):
            raise ValueError("stale reservation broker_order_id conflicts")

        new_ids = tuple(
            fill_id
            for fill_id in stale.observed_fill_ids
            if fill_id not in current.observed_fill_ids
        )
        stronger_fill = (
            stale.cumulative_filled_quantity
            > current.cumulative_filled_quantity
            or (stale.ever_fill_observed and not current.ever_fill_observed)
            or bool(new_ids)
        )
        new_broker_binding = (
            current.broker_order_id is None
            and stale.broker_order_id is not None
        )
        if not stronger_fill and not new_broker_binding:
            raise ValueError(
                f"stale reservation revision {stale.revision}; durable revision is "
                f"{current.revision}"
            )

        observed_fill_ids = (*current.observed_fill_ids, *new_ids)
        filled_quantity = max(
            current.cumulative_filled_quantity,
            stale.cumulative_filled_quantity,
        )
        # Two branches carrying different fill identifiers may each have started from
        # the same lower revision, so their cumulative numbers cannot safely be added.
        # Reserving the full order is the conservative bound and cannot permit an entry.
        if (
            any(
                fill_id not in stale.observed_fill_ids
                for fill_id in current.observed_fill_ids
            )
            and new_ids
        ):
            filled_quantity = current.quantity
        ever_fill_observed = (
            current.ever_fill_observed
            or stale.ever_fill_observed
            or filled_quantity > 0
        )
        state = cls._merge_pending_states(
            current.state,
            stale.state,
            filled_quantity=filled_quantity,
            quantity=current.quantity,
            ever_fill_observed=ever_fill_observed,
        )
        merged = PendingEntryReservation(
            order_id=current.order_id,
            broker_order_id=current.broker_order_id or stale.broker_order_id,
            state=state,
            reserved_at=current.reserved_at,
            updated_at=max(current.updated_at, stale.updated_at),
            quantity=current.quantity,
            risk_usd=current.risk_usd,
            instrument=current.instrument,
            side=current.side,
            order_type=current.order_type,
            limit_price=current.limit_price,
            protective_stop_price=current.protective_stop_price,
            broker_account_id=current.broker_account_id,
            broker_execution_route=current.broker_execution_route,
            cumulative_filled_quantity=filled_quantity,
            ever_fill_observed=ever_fill_observed,
            observed_fill_ids=observed_fill_ids,
            revision=current.revision + 1,
        )
        cls._validate_transition(current, merged)
        return merged

    @staticmethod
    def _merge_pending_states(
        current: PendingEntryState,
        stale: PendingEntryState,
        *,
        filled_quantity: int,
        quantity: int,
        ever_fill_observed: bool,
    ) -> PendingEntryState:
        if current is PendingEntryState.FILLED:
            return PendingEntryState.FILLED
        if current is PendingEntryState.TERMINAL_REPORTED:
            return PendingEntryState.TERMINAL_REPORTED
        if stale is PendingEntryState.TERMINAL_REPORTED:
            return PendingEntryState.TERMINAL_REPORTED
        if stale is PendingEntryState.FILLED or filled_quantity >= quantity:
            return PendingEntryState.FILLED
        if (
            current is PendingEntryState.CANCEL_REQUESTED
            or stale is PendingEntryState.CANCEL_REQUESTED
        ):
            return PendingEntryState.CANCEL_REQUESTED
        if ever_fill_observed:
            return PendingEntryState.PARTIALLY_FILLED
        if current is PendingEntryState.OUTCOME_UNKNOWN:
            return PendingEntryState.OUTCOME_UNKNOWN
        return current

    @staticmethod
    def _validate_transition(
        current: PendingEntryReservation,
        proposed: PendingEntryReservation,
    ) -> None:
        proposed.validate()
        if proposed.revision != current.revision + 1:
            raise ValueError(
                f"stale reservation revision {proposed.revision}; expected "
                f"{current.revision + 1}"
            )
        immutable_fields = (
            "order_id",
            "reserved_at",
            "quantity",
            "risk_usd",
            "instrument",
            "side",
            "order_type",
            "limit_price",
            "protective_stop_price",
            "broker_account_id",
            "broker_execution_route",
        )
        rotated = tuple(
            field
            for field in immutable_fields
            if getattr(proposed, field) != getattr(current, field)
        )
        if rotated:
            raise ValueError(
                "reservation immutable identity/risk fields changed: "
                + ", ".join(rotated)
            )
        if (
            current.broker_order_id is not None
            and proposed.broker_order_id != current.broker_order_id
        ):
            raise ValueError("reservation broker_order_id cannot rotate or disappear")
        if proposed.updated_at < current.updated_at:
            raise ValueError("reservation updated_at cannot move backwards")
        if (
            proposed.cumulative_filled_quantity
            < current.cumulative_filled_quantity
        ):
            raise ValueError("reservation filled quantity cannot decrease")
        if current.ever_fill_observed and not proposed.ever_fill_observed:
            raise ValueError("reservation fill evidence cannot be erased")
        if proposed.observed_fill_ids[: len(current.observed_fill_ids)] != (
            current.observed_fill_ids
        ):
            raise ValueError("reservation fill identifiers must be append-only")
        allowed_states = {
            PendingEntryState.SUBMITTING: {
                PendingEntryState.SUBMITTING,
                PendingEntryState.ACKNOWLEDGED,
                PendingEntryState.PARTIALLY_FILLED,
                PendingEntryState.FILLED,
                PendingEntryState.OUTCOME_UNKNOWN,
                PendingEntryState.TERMINAL_REPORTED,
            },
            PendingEntryState.OUTCOME_UNKNOWN: {
                PendingEntryState.OUTCOME_UNKNOWN,
                PendingEntryState.ACKNOWLEDGED,
                PendingEntryState.PARTIALLY_FILLED,
                PendingEntryState.FILLED,
                PendingEntryState.TERMINAL_REPORTED,
            },
            PendingEntryState.ACKNOWLEDGED: {
                PendingEntryState.ACKNOWLEDGED,
                PendingEntryState.PARTIALLY_FILLED,
                PendingEntryState.FILLED,
                PendingEntryState.CANCEL_REQUESTED,
                PendingEntryState.OUTCOME_UNKNOWN,
                PendingEntryState.TERMINAL_REPORTED,
            },
            PendingEntryState.PARTIALLY_FILLED: {
                PendingEntryState.PARTIALLY_FILLED,
                PendingEntryState.FILLED,
                PendingEntryState.CANCEL_REQUESTED,
                PendingEntryState.OUTCOME_UNKNOWN,
                PendingEntryState.TERMINAL_REPORTED,
            },
            PendingEntryState.CANCEL_REQUESTED: {
                PendingEntryState.CANCEL_REQUESTED,
                PendingEntryState.FILLED,
                PendingEntryState.OUTCOME_UNKNOWN,
                PendingEntryState.TERMINAL_REPORTED,
            },
            PendingEntryState.FILLED: {PendingEntryState.FILLED},
            PendingEntryState.TERMINAL_REPORTED: {
                PendingEntryState.TERMINAL_REPORTED
            },
        }
        if proposed.state not in allowed_states[current.state]:
            raise ValueError(
                f"reservation state cannot regress from {current.state.value} to "
                f"{proposed.state.value}"
            )


__all__ = [
    "FilePendingEntryReservationStore",
    "PendingEntryReservation",
    "PendingEntryReservationStore",
    "PendingEntryState",
    "ReservationStoreError",
]
