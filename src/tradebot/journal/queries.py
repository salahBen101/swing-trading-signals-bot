"""Read side of the journal.

Kept apart from the writer so the dashboard and the analytics module can open the database
read-only while the runner is writing to it. Everything returns plain dicts or reconstructed
domain objects; no SQL escapes this module.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ..core.models import Trade
from ..core.types import ExitReason, Side


class JournalReader:
    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        # `uri=True` with mode=ro means a corrupt or missing file fails loudly here rather
        # than being silently created as an empty database, which would make the dashboard
        # show a healthy-looking zero-trade run.
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> JournalReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _scope(self, where: str = "") -> tuple[str, tuple]:
        if self.run_id is None:
            return (where, ())
        clause = f"{where} AND run_id = ?" if where else "WHERE run_id = ?"
        return (clause, (self.run_id,))

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with closing(self.conn.execute(sql, params)) as cursor:
            return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------ queries

    def runs(self) -> list[dict]:
        return self._rows("SELECT * FROM runs ORDER BY started_at DESC")

    def latest_run_id(self) -> str | None:
        rows = self._rows("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
        return rows[0]["run_id"] if rows else None

    def trades(self, limit: int | None = None) -> list[Trade]:
        where, params = self._scope()
        sql = f"SELECT * FROM trades {where} ORDER BY exit_time ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._to_trade(row) for row in self._rows(sql, params)]

    @staticmethod
    def _to_trade(row: dict) -> Trade:
        return Trade(
            trade_id=row["trade_id"],
            instrument=row["instrument"],
            strategy=row["strategy"],
            side=Side(row["side"]),
            quantity=row["quantity"],
            entry_time=datetime.fromisoformat(row["entry_time"]),
            entry_price=row["entry_price"],
            exit_time=datetime.fromisoformat(row["exit_time"]),
            exit_price=row["exit_price"],
            exit_reason=ExitReason(row["exit_reason"]),
            gross_pnl_usd=row["gross_pnl"],
            commission_usd=row["commission"],
            net_pnl_usd=row["net_pnl"],
            r_multiple=row["r_multiple"],
            bars_held=row["bars_held"],
            initial_stop=row["initial_stop"],
            target_price=row["target_price"],
            max_favorable_excursion_points=row["mfe_points"],
            max_adverse_excursion_points=row["mae_points"],
            entry_conditions=tuple(json.loads(row["conditions"] or "[]")),
            entry_features=json.loads(row["features"] or "{}"),
            slippage_usd=row.get("slippage_usd", 0.0) or 0.0,
        )

    def rejections(self, limit: int = 100) -> list[dict]:
        where, params = self._scope()
        return self._rows(
            f"SELECT * FROM rejections {where} ORDER BY id DESC LIMIT {int(limit)}", params
        )

    def rejection_counts(self) -> list[dict]:
        """Rejections grouped by reason — the dashboard's "why nothing happened" panel."""
        where, params = self._scope()
        return self._rows(
            f"SELECT reason, stage, COUNT(*) AS n FROM rejections {where}"
            f" GROUP BY reason, stage ORDER BY n DESC", params
        )

    def orders(self, limit: int = 100) -> list[dict]:
        where, params = self._scope()
        return self._rows(
            f"SELECT * FROM orders {where} ORDER BY ts DESC LIMIT {int(limit)}", params
        )

    def order_history(self, order_id: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM order_events WHERE order_id = ? ORDER BY id ASC", (order_id,)
        )

    def fills(self, limit: int = 100) -> list[dict]:
        where, params = self._scope()
        return self._rows(
            f"SELECT * FROM fills {where} ORDER BY ts DESC LIMIT {int(limit)}", params
        )

    def signals(self, limit: int = 100) -> list[dict]:
        where, params = self._scope()
        return self._rows(
            f"SELECT * FROM signals {where} ORDER BY ts DESC LIMIT {int(limit)}", params
        )

    def equity_curve(self) -> list[dict]:
        where, params = self._scope()
        return self._rows(f"SELECT ts, equity FROM equity {where} ORDER BY ts ASC", params)

    def events(self, limit: int = 200, level: str | None = None) -> list[dict]:
        where, params = self._scope()
        if level:
            where = f"{where} AND level = ?" if where else "WHERE level = ?"
            params = (*params, level)
        return self._rows(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT {int(limit)}", params
        )

    def summary(self) -> dict:
        where, params = self._scope()
        rows = self._rows(
            f"SELECT COUNT(*) AS trades, COALESCE(SUM(net_pnl), 0) AS net_pnl,"
            f" COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS winners"
            f" FROM trades {where}", params
        )
        row = rows[0] if rows else {"trades": 0, "net_pnl": 0.0, "winners": 0}
        trades = row["trades"] or 0
        return {
            "trades": trades,
            "net_pnl": row["net_pnl"] or 0.0,
            "winners": row["winners"] or 0,
            "win_rate": (row["winners"] / trades) if trades else 0.0,
        }
