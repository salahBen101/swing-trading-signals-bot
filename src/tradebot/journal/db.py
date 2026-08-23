"""The trade journal: a SQLite record of every decision the system made.

PROJECT_SPEC §9 requires that every strategy decision be reproducible from the record
alone. That means recording not only what happened but what *nearly* happened: the signals
that fired, the ones that were refused, and the reason code for each refusal. A journal
that only contains fills can tell you what you did; it cannot tell you why.

Two deliberate choices:

* **Append-only in spirit.** Rows are inserted, not edited. Orders move through statuses by
  gaining rows in `order_events`, so the history of an order is recoverable rather than
  overwritten. The one exception is `orders.status`, a denormalised cache of the latest
  event, kept because every read wants it.
* **The journal is not the source of truth for live position state.** The broker is. On
  restart the two are reconciled and any difference is itself journalled.

WAL mode is on so the dashboard can read while the runner writes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import (
    AccountSnapshot,
    Bar,
    Fill,
    Order,
    OrderIntent,
    Rejection,
    Trade,
)
from ..core.types import OrderStatus

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    instrument  TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    broker      TEXT NOT NULL,
    config_json TEXT NOT NULL,
    ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    intent_id    TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    instrument   TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    side         INTEGER NOT NULL,
    reference    REAL NOT NULL,
    stop_price   REAL NOT NULL,
    target_price REAL,
    conditions   TEXT NOT NULL,
    features     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    reason     TEXT NOT NULL,
    detail     TEXT NOT NULL,
    stage      TEXT NOT NULL,
    instrument TEXT,
    strategy   TEXT,
    intent_id  TEXT,
    order_id   TEXT,
    context    TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    ts              TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    strategy        TEXT,
    intent_id       TEXT,
    side            INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    purpose         TEXT NOT NULL,
    oco_group       TEXT,
    status          TEXT NOT NULL,
    broker_order_id TEXT,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    avg_fill_price  REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS order_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    order_id TEXT NOT NULL,
    ts       TEXT NOT NULL,
    status   TEXT NOT NULL,
    detail   TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    order_id      TEXT NOT NULL,
    ts            TEXT NOT NULL,
    instrument    TEXT NOT NULL,
    side          INTEGER NOT NULL,
    quantity      INTEGER NOT NULL,
    price         REAL NOT NULL,
    commission    REAL NOT NULL,
    slippage_points REAL NOT NULL DEFAULT 0,
    is_partial    INTEGER NOT NULL DEFAULT 0,
    broker_fill_id TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    instrument   TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    side         INTEGER NOT NULL,
    quantity     INTEGER NOT NULL,
    entry_time   TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    exit_time    TEXT NOT NULL,
    exit_price   REAL NOT NULL,
    exit_reason  TEXT NOT NULL,
    gross_pnl    REAL NOT NULL,
    commission   REAL NOT NULL,
    slippage_usd REAL NOT NULL DEFAULT 0,
    net_pnl      REAL NOT NULL,
    r_multiple   REAL NOT NULL,
    bars_held    INTEGER NOT NULL,
    initial_stop REAL NOT NULL,
    target_price REAL,
    mfe_points   REAL NOT NULL DEFAULT 0,
    mae_points   REAL NOT NULL DEFAULT 0,
    conditions   TEXT,
    features     TEXT
);

CREATE TABLE IF NOT EXISTS equity (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    equity     REAL NOT NULL,
    realized   REAL NOT NULL,
    unrealized REAL NOT NULL,
    positions  INTEGER NOT NULL,
    trades     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL,
    kind    TEXT NOT NULL,
    detail  TEXT NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS bars (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    instrument TEXT NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    features   TEXT
);

CREATE INDEX IF NOT EXISTS ix_rejections_run_ts ON rejections(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_trades_run_exit   ON trades(run_id, exit_time);
CREATE INDEX IF NOT EXISTS ix_equity_run_ts     ON equity(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_events_run_ts     ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_fills_order       ON fills(order_id);
CREATE INDEX IF NOT EXISTS ix_orders_run        ON orders(run_id);
"""


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


class Journal:
    """Write side. `journal.queries` holds the read side."""

    def __init__(self, path: str | Path, *, run_id: str = "default") -> None:
        self.path = Path(path)
        self.run_id = run_id
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self.conn:
            # WAL lets the dashboard read while the runner writes. Not available for
            # in-memory databases, where the pragma is simply ignored.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            prior_version = self._schema_version()
            if prior_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"journal schema {prior_version} is newer than supported "
                    f"version {SCHEMA_VERSION}"
                )
            self.conn.executescript(_SCHEMA)
            if prior_version < 2:
                self._migrate_to_v2()
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _schema_version(self) -> int:
        """Read the on-disk version without assuming the ``meta`` table exists."""
        table = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        if table is None:
            return 0
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid journal schema version {row[0]!r}") from exc

    def _migrate_to_v2(self) -> None:
        """Add execution-cost attribution to a v1 journal, safely on retry.

        ``ALTER TABLE ... ADD COLUMN`` has no portable ``IF NOT EXISTS`` form in the
        supported SQLite versions.  Inspecting each table first makes this migration
        idempotent, including recovery from an interrupted attempt that added only one of
        the two columns.
        """
        self._add_column_if_missing(
            "fills", "slippage_points", "REAL NOT NULL DEFAULT 0"
        )
        self._add_column_if_missing(
            "trades", "slippage_usd", "REAL NOT NULL DEFAULT 0"
        )

    def _add_column_if_missing(
        self, table: str, column: str, declaration: str
    ) -> None:
        allowed = {"fills", "trades"}
        if table not in allowed:
            raise ValueError(f"unsupported migration table {table!r}")
        columns = {
            row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ runs

    def start_run(
        self, *, mode: str, instrument: str, strategy: str, broker: str, config: dict,
        started_at: datetime,
    ) -> str:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, started_at, mode, instrument, strategy,"
                " broker, config_json) VALUES (?,?,?,?,?,?,?)",
                (self.run_id, _iso(started_at), mode, instrument, strategy, broker,
                 _json(config)),
            )
        return self.run_id

    def end_run(self, ended_at: datetime) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET ended_at = ? WHERE run_id = ?", (_iso(ended_at), self.run_id)
            )

    # ------------------------------------------------------------------ writes

    def record_signal(self, intent: OrderIntent) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO signals(intent_id, run_id, ts, instrument, strategy,"
                " side, reference, stop_price, target_price, conditions, features)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (intent.intent_id, self.run_id, _iso(intent.timestamp), intent.instrument,
                 intent.strategy, int(intent.side), intent.reference_price, intent.stop_price,
                 intent.target_price, _json(list(intent.conditions)), _json(intent.features)),
            )

    def record_rejection(self, rejection: Rejection) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO rejections(run_id, ts, reason, detail, stage, instrument,"
                " strategy, intent_id, order_id, context) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, _iso(rejection.timestamp), rejection.reason.value,
                 rejection.detail, rejection.stage, rejection.instrument, rejection.strategy,
                 rejection.intent_id, rejection.order_id, _json(rejection.context)),
            )

    def record_rejections(self, rejections: Iterable[Rejection]) -> None:
        for rejection in rejections:
            self.record_rejection(rejection)

    def record_order(self, order: Order) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO orders(order_id, run_id, ts, instrument, strategy,"
                " intent_id, side, quantity, order_type, limit_price, stop_price, purpose,"
                " oco_group, status, broker_order_id, filled_quantity, avg_fill_price)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (order.order_id, self.run_id, _iso(order.timestamp), order.instrument,
                 order.strategy, order.intent_id, int(order.side), order.quantity,
                 order.order_type.value, order.limit_price, order.stop_price, order.purpose,
                 order.oco_group, order.status.value, order.broker_order_id,
                 order.filled_quantity, order.average_fill_price),
            )
            self.conn.execute(
                "INSERT INTO order_events(run_id, order_id, ts, status, detail)"
                " VALUES (?,?,?,?,?)",
                (self.run_id, order.order_id, _iso(order.timestamp), order.status.value,
                 "created"),
            )

    def update_order(
        self, order_id: str, status: OrderStatus, *, ts: datetime,
        broker_order_id: str | None = None, filled_quantity: int | None = None,
        average_fill_price: float | None = None, detail: str = "",
    ) -> None:
        sets = ["status = ?"]
        params: list[Any] = [status.value]
        if broker_order_id is not None:
            sets.append("broker_order_id = ?")
            params.append(broker_order_id)
        if filled_quantity is not None:
            sets.append("filled_quantity = ?")
            params.append(filled_quantity)
        if average_fill_price is not None:
            sets.append("avg_fill_price = ?")
            params.append(average_fill_price)
        params.extend([order_id, self.run_id])

        with self.conn:
            self.conn.execute(
                f"UPDATE orders SET {', '.join(sets)} WHERE order_id = ? AND run_id = ?",
                params,
            )
            self.conn.execute(
                "INSERT INTO order_events(run_id, order_id, ts, status, detail)"
                " VALUES (?,?,?,?,?)",
                (self.run_id, order_id, _iso(ts), status.value, detail),
            )

    def record_fill(self, fill: Fill) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO fills(fill_id, run_id, order_id, ts, instrument,"
                " side, quantity, price, commission, slippage_points, is_partial,"
                " broker_fill_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (fill.fill_id, self.run_id, fill.order_id, _iso(fill.timestamp),
                 fill.instrument, int(fill.side), fill.quantity, fill.price,
                 fill.commission_usd, fill.slippage_points, int(fill.is_partial),
                 fill.broker_fill_id),
            )

    def record_trade(self, trade: Trade) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO trades(trade_id, run_id, instrument, strategy, side,"
                " quantity, entry_time, entry_price, exit_time, exit_price, exit_reason,"
                " gross_pnl, commission, slippage_usd, net_pnl, r_multiple, bars_held,"
                " initial_stop,"
                " target_price, mfe_points, mae_points, conditions, features)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade.trade_id, self.run_id, trade.instrument, trade.strategy,
                 int(trade.side), trade.quantity, _iso(trade.entry_time), trade.entry_price,
                 _iso(trade.exit_time), trade.exit_price, trade.exit_reason.value,
                 trade.gross_pnl_usd, trade.commission_usd, trade.slippage_usd,
                 trade.net_pnl_usd, trade.r_multiple, trade.bars_held, trade.initial_stop,
                 trade.target_price, trade.max_favorable_excursion_points,
                 trade.max_adverse_excursion_points, _json(list(trade.entry_conditions)),
                 _json(trade.entry_features)),
            )

    def record_equity(self, snapshot: AccountSnapshot) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO equity(run_id, ts, equity, realized, unrealized, positions,"
                " trades) VALUES (?,?,?,?,?,?,?)",
                (self.run_id, _iso(snapshot.timestamp), snapshot.equity,
                 snapshot.realized_pnl_today, snapshot.unrealized_pnl,
                 snapshot.open_positions, snapshot.trades_today),
            )

    def record_event(
        self, ts: datetime, kind: str, detail: str, *, level: str = "INFO",
        payload: dict | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO events(run_id, ts, level, kind, detail, payload)"
                " VALUES (?,?,?,?,?,?)",
                (self.run_id, _iso(ts), level, kind, detail,
                 _json(payload) if payload else None),
            )

    def record_bar(self, bar: Bar, instrument: str, features: dict | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO bars(run_id, ts, instrument, open, high, low, close, volume,"
                " features) VALUES (?,?,?,?,?,?,?,?,?)",
                (self.run_id, _iso(bar.timestamp), instrument, bar.open, bar.high, bar.low,
                 bar.close, bar.volume, _json(features) if features else None),
            )

    # ------------------------------------------------------------------ counts

    def count(self, table: str) -> int:
        allowed = {
            "signals", "rejections", "orders", "order_events", "fills", "trades",
            "equity", "events", "bars", "runs",
        }
        if table not in allowed:
            raise ValueError(f"unknown table {table!r}")
        with closing(self.conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE run_id = ?"
            if table != "runs" else "SELECT COUNT(*) AS n FROM runs",
            () if table == "runs" else (self.run_id,),
        )) as cursor:
            return int(cursor.fetchone()["n"])
