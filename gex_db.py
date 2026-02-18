"""SQLite storage and query functions for GEX snapshots."""

from __future__ import annotations

import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pytz

from gex_parser import GEXSnapshot

PT_TZ = pytz.timezone("US/Pacific")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gex_data.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS gex_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_pt REAL NOT NULL,
            date_pt TEXT NOT NULL,
            time_pt TEXT NOT NULL,
            session_tag TEXT NOT NULL,
            curr_price REAL,
            net_gamma INTEGER,
            total_call_gamma INTEGER,
            total_put_gamma INTEGER,
            call_wall INTEGER,
            put_floor INTEGER,
            top5_calls TEXT,
            top5_puts TEXT,
            raw_text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON gex_snapshots(timestamp_pt);
        CREATE INDEX IF NOT EXISTS idx_snapshots_date ON gex_snapshots(date_pt);
        CREATE INDEX IF NOT EXISTS idx_snapshots_session ON gex_snapshots(session_tag);

        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_pt TEXT NOT NULL UNIQUE,
            open_price REAL,
            close_price REAL,
            high_gamma INTEGER,
            low_gamma INTEGER,
            close_gamma INTEGER,
            open_call_wall INTEGER,
            close_call_wall INTEGER,
            open_put_floor INTEGER,
            close_put_floor INTEGER,
            regime_sequence TEXT,
            num_snapshots INTEGER,
            created_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summaries(date_pt);
    """)
    conn.commit()
    conn.close()


def save_snapshot(snapshot: GEXSnapshot) -> int:
    """Save a GEXSnapshot to the database. Returns the row id."""
    conn = get_conn()
    cursor = conn.execute(
        """INSERT INTO gex_snapshots
        (timestamp_pt, date_pt, time_pt, session_tag,
         curr_price, net_gamma, total_call_gamma, total_put_gamma,
         call_wall, put_floor, top5_calls, top5_puts, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot.timestamp_pt.timestamp(),
            snapshot.timestamp_pt.strftime("%Y-%m-%d"),
            snapshot.timestamp_pt.strftime("%H:%M:%S"),
            snapshot.session_tag,
            snapshot.curr_price,
            snapshot.net_gamma,
            snapshot.total_call_gamma,
            snapshot.total_put_gamma,
            snapshot.call_wall,
            snapshot.put_floor,
            json.dumps(snapshot.top5_calls),
            json.dumps(snapshot.top5_puts),
            snapshot.raw_text,
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _row_to_snapshot(row: sqlite3.Row) -> GEXSnapshot:
    """Convert a database row back to a GEXSnapshot."""
    ts = datetime.fromtimestamp(row["timestamp_pt"], tz=PT_TZ)
    return GEXSnapshot(
        timestamp_pt=ts,
        session_tag=row["session_tag"],
        curr_price=row["curr_price"] or 0.0,
        net_gamma=row["net_gamma"] or 0,
        total_call_gamma=row["total_call_gamma"] or 0,
        total_put_gamma=row["total_put_gamma"] or 0,
        call_wall=row["call_wall"] or 0,
        put_floor=row["put_floor"] or 0,
        top5_calls=json.loads(row["top5_calls"]) if row["top5_calls"] else [],
        top5_puts=json.loads(row["top5_puts"]) if row["top5_puts"] else [],
        raw_text=row["raw_text"] or "",
    )


def get_recent_snapshots(minutes: int, session_tag: Optional[str] = None) -> List[GEXSnapshot]:
    """Get snapshots from the last N minutes up to now, optionally filtered by session tag."""
    conn = get_conn()
    now_ts = datetime.now(PT_TZ).timestamp()
    cutoff = now_ts - (minutes * 60)
    if session_tag:
        rows = conn.execute(
            "SELECT * FROM gex_snapshots "
            "WHERE timestamp_pt >= ? AND timestamp_pt <= ? AND session_tag = ? "
            "ORDER BY timestamp_pt ASC",
            (cutoff, now_ts, session_tag),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM gex_snapshots "
            "WHERE timestamp_pt >= ? AND timestamp_pt <= ? ORDER BY timestamp_pt ASC",
            (cutoff, now_ts),
        ).fetchall()
    conn.close()
    return [_row_to_snapshot(r) for r in rows]


def get_snapshots_range(start_ts: float, end_ts: float) -> List[GEXSnapshot]:
    """Get snapshots between two unix timestamps."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM gex_snapshots WHERE timestamp_pt >= ? AND timestamp_pt <= ? ORDER BY timestamp_pt ASC",
        (start_ts, end_ts),
    ).fetchall()
    conn.close()
    return [_row_to_snapshot(r) for r in rows]


def get_latest_snapshot() -> Optional[GEXSnapshot]:
    """Get the most recent snapshot at or before now."""
    conn = get_conn()
    now_ts = datetime.now(PT_TZ).timestamp()
    row = conn.execute(
        "SELECT * FROM gex_snapshots WHERE timestamp_pt <= ? ORDER BY timestamp_pt DESC LIMIT 1",
        (now_ts,),
    ).fetchone()
    conn.close()
    return _row_to_snapshot(row) if row else None


def get_today_snapshots(session_tag: Optional[str] = None) -> List[GEXSnapshot]:
    """Get today's snapshots (PT) up to now."""
    today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
    now_ts = datetime.now(PT_TZ).timestamp()
    conn = get_conn()
    if session_tag:
        rows = conn.execute(
            "SELECT * FROM gex_snapshots "
            "WHERE date_pt = ? AND timestamp_pt <= ? AND session_tag = ? "
            "ORDER BY timestamp_pt ASC",
            (today, now_ts, session_tag),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM gex_snapshots "
            "WHERE date_pt = ? AND timestamp_pt <= ? ORDER BY timestamp_pt ASC",
            (today, now_ts),
        ).fetchall()
    conn.close()
    return [_row_to_snapshot(r) for r in rows]


def get_eod_snapshot(date_pt: str) -> Optional[GEXSnapshot]:
    """Get the last RTH snapshot for a given date."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM gex_snapshots WHERE date_pt = ? AND session_tag = 'RTH' ORDER BY timestamp_pt DESC LIMIT 1",
        (date_pt,),
    ).fetchone()
    conn.close()
    return _row_to_snapshot(row) if row else None


def get_previous_trading_day_eod() -> Optional[GEXSnapshot]:
    """Get the EOD snapshot from the most recent previous trading day."""
    today = datetime.now(PT_TZ).strftime("%Y-%m-%d")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM gex_snapshots WHERE date_pt < ? AND session_tag = 'RTH' ORDER BY timestamp_pt DESC LIMIT 1",
        (today,),
    ).fetchone()
    conn.close()
    return _row_to_snapshot(row) if row else None


def save_daily_summary(date_pt: str, summary: dict):
    """Save or update a daily summary."""
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_summaries
        (date_pt, open_price, close_price, high_gamma, low_gamma, close_gamma,
         open_call_wall, close_call_wall, open_put_floor, close_put_floor,
         regime_sequence, num_snapshots, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            date_pt,
            summary.get("open_price"),
            summary.get("close_price"),
            summary.get("high_gamma"),
            summary.get("low_gamma"),
            summary.get("close_gamma"),
            summary.get("open_call_wall"),
            summary.get("close_call_wall"),
            summary.get("open_put_floor"),
            summary.get("close_put_floor"),
            json.dumps(summary.get("regime_sequence", [])),
            summary.get("num_snapshots", 0),
            datetime.now(PT_TZ).timestamp(),
        ),
    )
    conn.commit()
    conn.close()


def purge_old_data(retention_days: int = 30):
    """Delete snapshots and summaries older than retention_days."""
    cutoff = (datetime.now(PT_TZ) - timedelta(days=retention_days)).timestamp()
    cutoff_date = (datetime.now(PT_TZ) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    conn = get_conn()
    conn.execute("DELETE FROM gex_snapshots WHERE timestamp_pt < ?", (cutoff,))
    conn.execute("DELETE FROM daily_summaries WHERE date_pt < ?", (cutoff_date,))
    conn.commit()
    conn.close()


def get_snapshot_count(date_pt: Optional[str] = None) -> int:
    """Get count of snapshots, optionally for a specific date."""
    conn = get_conn()
    if date_pt:
        row = conn.execute(
            "SELECT COUNT(*) FROM gex_snapshots WHERE date_pt = ?", (date_pt,)
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM gex_snapshots").fetchone()
    conn.close()
    return row[0]


# Initialize on import
init_db()
