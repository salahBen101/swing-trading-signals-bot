"""Focused journal schema migration and execution-cost attribution tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from tradebot.core.clock import MARKET_TZ
from tradebot.core.models import Fill, Trade
from tradebot.core.types import ExitReason, Side
from tradebot.journal.db import SCHEMA_VERSION, Journal
from tradebot.journal.queries import JournalReader


def at(hour: int = 11, minute: int = 0) -> datetime:
    return datetime(2024, 4, 1, hour, minute, tzinfo=MARKET_TZ)


def completed_trade(*, trade_id: str, slippage_usd: float) -> Trade:
    return Trade(
        trade_id=trade_id,
        instrument="MNQ",
        strategy="migration-test",
        side=Side.BUY,
        quantity=1,
        entry_time=at(),
        entry_price=18000.25,
        exit_time=at(11, 5),
        exit_price=18009.75,
        exit_reason=ExitReason.MANUAL,
        gross_pnl_usd=19.0,
        commission_usd=1.24,
        net_pnl_usd=17.76,
        r_multiple=0.888,
        bars_held=5,
        initial_stop=17990.0,
        slippage_usd=slippage_usd,
    )


def create_v1_journal(path: Path, *, interrupted_upgrade: bool = False) -> None:
    """Create the two v1 tables changed by schema v2, including existing rows."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO meta(key, value) VALUES ('schema_version', '1');

        CREATE TABLE fills (
            fill_id        TEXT PRIMARY KEY,
            run_id         TEXT NOT NULL,
            order_id       TEXT NOT NULL,
            ts             TEXT NOT NULL,
            instrument     TEXT NOT NULL,
            side           INTEGER NOT NULL,
            quantity       INTEGER NOT NULL,
            price          REAL NOT NULL,
            commission     REAL NOT NULL,
            is_partial     INTEGER NOT NULL DEFAULT 0,
            broker_fill_id TEXT
        );

        CREATE TABLE trades (
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
        """
    )
    if interrupted_upgrade:
        conn.execute(
            "ALTER TABLE fills ADD COLUMN slippage_points REAL NOT NULL DEFAULT 0"
        )
    conn.execute(
        "INSERT INTO fills(fill_id, run_id, order_id, ts, instrument, side, quantity,"
        " price, commission, is_partial, broker_fill_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "old-fill",
            "old-run",
            "old-order",
            at().isoformat(),
            "MNQ",
            int(Side.BUY),
            1,
            18000.25,
            0.62,
            0,
            "venue-old",
        ),
    )
    conn.execute(
        "INSERT INTO trades(trade_id, run_id, instrument, strategy, side, quantity,"
        " entry_time, entry_price, exit_time, exit_price, exit_reason, gross_pnl,"
        " commission, net_pnl, r_multiple, bars_held, initial_stop, target_price,"
        " mfe_points, mae_points, conditions, features)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "old-trade",
            "old-run",
            "MNQ",
            "legacy",
            int(Side.BUY),
            1,
            at().isoformat(),
            18000.0,
            at(11, 5).isoformat(),
            18010.0,
            ExitReason.MANUAL.value,
            20.0,
            1.24,
            18.76,
            0.938,
            5,
            17990.0,
            None,
            10.0,
            0.0,
            "[]",
            "{}",
        ),
    )
    conn.commit()
    conn.close()


def test_fresh_v2_journal_persists_fill_and_trade_slippage(tmp_path: Path) -> None:
    path = tmp_path / "fresh.sqlite3"
    with Journal(path, run_id="run") as journal:
        journal.record_fill(
            Fill(
                fill_id="fill",
                order_id="order",
                timestamp=at(),
                instrument="MNQ",
                side=Side.BUY,
                quantity=2,
                price=18000.25,
                commission_usd=1.24,
                slippage_points=0.25,
                is_partial=True,
            )
        )
        journal.record_trade(completed_trade(trade_id="trade", slippage_usd=1.0))

    with JournalReader(path, run_id="run") as reader:
        assert reader.fills()[0]["slippage_points"] == pytest.approx(0.25)
        trade = reader.trades()[0]
        assert trade.slippage_usd == pytest.approx(1.0)
        assert trade.net_pnl_usd == pytest.approx(17.76)


@pytest.mark.parametrize("interrupted_upgrade", [False, True])
def test_v1_journal_upgrade_is_lossless_and_idempotent(
    tmp_path: Path, interrupted_upgrade: bool
) -> None:
    path = tmp_path / "old.sqlite3"
    create_v1_journal(path, interrupted_upgrade=interrupted_upgrade)

    # Opening once performs v1 -> v2. Opening again must be a no-op, including after an
    # interrupted prior attempt where one column already existed.
    with Journal(path, run_id="old-run") as journal:
        assert journal.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == str(SCHEMA_VERSION)
    with Journal(path, run_id="old-run"):
        pass

    conn = sqlite3.connect(path)
    fill_columns = [row[1] for row in conn.execute("PRAGMA table_info(fills)")]
    trade_columns = [row[1] for row in conn.execute("PRAGMA table_info(trades)")]
    assert fill_columns.count("slippage_points") == 1
    assert trade_columns.count("slippage_usd") == 1
    conn.close()

    with JournalReader(path, run_id="old-run") as reader:
        fills = reader.fills()
        trades = reader.trades()
    assert fills[0]["fill_id"] == "old-fill"
    assert fills[0]["slippage_points"] == 0.0
    assert trades[0].trade_id == "old-trade"
    assert trades[0].slippage_usd == 0.0
    assert trades[0].net_pnl_usd == pytest.approx(18.76)
