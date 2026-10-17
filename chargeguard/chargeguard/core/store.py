"""SQLite record of sessions, decisions, incidents and an audit trail of operator actions."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  tx INTEGER PRIMARY KEY, cp TEXT, connector INTEGER, driver TEXT, battery TEXT, smart INTEGER,
  started TEXT, ended TEXT, minutes INTEGER DEFAULT 0, kwh REAL DEFAULT 0, cost REAL DEFAULT 0,
  level INTEGER DEFAULT 0, max_risk REAL DEFAULT 0, outcome TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, tx INTEGER, ts TEXT, minute INTEGER, level INTEGER, risk REAL,
  percentile REAL, reasons TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY, tx INTEGER, cp TEXT, connector INTEGER, battery TEXT, minute INTEGER, percentile REAL,
  opened TEXT, status TEXT, risk REAL, reasons TEXT, note TEXT, note_source TEXT, decided TEXT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, detail TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            if path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)

    def _x(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.db.execute(sql, args)

    def start(self, tx, cp, connector, driver, battery, smart, started) -> None:
        self._x("INSERT OR REPLACE INTO sessions (tx, cp, connector, driver, battery, smart, started) "
                "VALUES (?,?,?,?,?,?,?)", (tx, cp, connector, driver, battery, int(smart), started))

    def progress(self, tx, minutes, kwh, cost, level, max_risk) -> None:
        self._x("UPDATE sessions SET minutes=?, kwh=?, cost=?, level=?, max_risk=? WHERE tx=?",
                (minutes, kwh, cost, level, max_risk, tx))

    def end(self, tx, ended, outcome) -> None:
        self._x("UPDATE sessions SET ended=?, outcome=? WHERE tx=?", (ended, outcome, tx))

    def decision(self, tx, ts, minute, level, risk, percentile, reasons, action) -> None:
        self._x("INSERT INTO decisions (tx, ts, minute, level, risk, percentile, reasons, action) "
                "VALUES (?,?,?,?,?,?,?,?)", (tx, ts, minute, level, risk, percentile, json.dumps(reasons), action))

    def open_incident(self, tx, cp, connector, risk, reasons, note, note_source, battery=None, minute=None,
                      percentile=None) -> int:
        cur = self._x("INSERT INTO incidents (tx, cp, connector, battery, minute, percentile, opened, status, risk, "
                      "reasons, note, note_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (tx, cp, connector, battery, minute, percentile, now(), "awaiting_approval", risk,
                       json.dumps(reasons), note, note_source))
        return int(cur.lastrowid)

    def set_note(self, incident_id, note, source) -> None:
        self._x("UPDATE incidents SET note=?, note_source=? WHERE id=?", (note, source, incident_id))

    def decide(self, incident_id, status, operator) -> None:
        self._x("UPDATE incidents SET status=?, decided=?, decided_by=? WHERE id=?",
                (status, now(), operator, incident_id))

    def incident(self, incident_id) -> dict | None:
        r = self._x("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        return self._incident(r) if r else None

    def incidents(self, limit: int = 50) -> list[dict]:
        rows = self._x("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._incident(r) for r in rows]

    @staticmethod
    def _incident(r) -> dict:
        d = dict(r)
        d["reasons"] = json.loads(d["reasons"] or "[]")
        return d

    def audit(self, actor, action, detail: dict) -> None:
        self._x("INSERT INTO audit (ts, actor, action, detail) VALUES (?,?,?,?)",
                (now(), actor, action, json.dumps(detail)))

    def audit_log(self, limit: int = 100) -> list[dict]:
        rows = self._x("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]

    def summary(self) -> dict:
        r = self._x("SELECT COUNT(*) n, COALESCE(SUM(kwh),0) kwh, COALESCE(SUM(cost),0) cost, "
                    "SUM(level>=1) watch, SUM(level>=2) slow, SUM(level>=3) stop FROM sessions").fetchone()
        return {k: (r[k] or 0) for k in r.keys()}
