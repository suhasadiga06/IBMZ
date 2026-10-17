"""Durable state: relationship states, cases, evidence, recipient choices, audit.

SQLite in WAL mode: a single file, transactional, and fast enough for a
prototype. Only pseudonymous tokens are stored; evidence text is encrypted.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (sender TEXT, recipient TEXT, state TEXT, updated REAL,
                                  PRIMARY KEY (sender, recipient));
CREATE TABLE IF NOT EXISTS cases (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, recipient TEXT,
                                  opened REAL, updated REAL, status TEXT, priority TEXT, score REAL,
                                  reasons TEXT, summary TEXT, summary_by TEXT, second_opinion TEXT,
                                  label TEXT, UNIQUE (sender, recipient));
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, ts REAL,
                                     amount REAL, sealed_ref TEXT, harm REAL, threat REAL, hidden INTEGER);
CREATE TABLE IF NOT EXISTS hides (recipient TEXT, sender TEXT, ts REAL, PRIMARY KEY (recipient, sender));
CREATE TABLE IF NOT EXISTS audit (ts REAL, actor TEXT, action TEXT, case_id INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS labels (case_id INTEGER, ts REAL, analyst TEXT, label TEXT, features TEXT);
CREATE INDEX IF NOT EXISTS ev_case ON evidence(case_id);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def x(self, sql: str, args=()):
        with self._lock:
            return self.db.execute(sql, args)

    def q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # -- relationship state ----------------------------------------------------
    def save_state(self, sender: str, recipient: str, state_json: str) -> None:
        self.x("INSERT INTO pairs VALUES (?,?,?,?) ON CONFLICT(sender, recipient) DO UPDATE SET "
               "state=excluded.state, updated=excluded.updated", (sender, recipient, state_json, time.time()))

    def load_states(self) -> list[dict]:
        return self.q("SELECT sender, recipient, state FROM pairs")

    # -- cases --------------------------------------------------------------------
    def case_for(self, sender: str, recipient: str) -> dict | None:
        rows = self.q("SELECT * FROM cases WHERE sender=? AND recipient=?", (sender, recipient))
        return rows[0] if rows else None

    def open_or_update(self, sender: str, recipient: str, ts: float, priority: str, score: float,
                       reasons: list[str]) -> int:
        c = self.case_for(sender, recipient)
        if c is None:
            cur = self.x("INSERT INTO cases (sender, recipient, opened, updated, status, priority, score, reasons) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (sender, recipient, ts, ts, "open", priority, score, json.dumps(reasons)))
            return cur.lastrowid
        rank = {"P1": 1, "P2": 2, "P3": 3, "": 9}
        best = min([c["priority"] or "", priority or ""], key=lambda p: rank.get(p, 9))
        merged = list(dict.fromkeys(json.loads(c["reasons"] or "[]") + reasons))[-12:]
        self.x("UPDATE cases SET updated=?, priority=?, score=MAX(score, ?), reasons=? WHERE id=?",
               (ts, best, score, json.dumps(merged), c["id"]))
        return c["id"]

    def add_evidence(self, case_id: int, ts: float, amount: float, sealed: str, harm: float, threat: float,
                     hidden: bool) -> None:
        self.x("INSERT INTO evidence (case_id, ts, amount, sealed_ref, harm, threat, hidden) VALUES (?,?,?,?,?,?,?)",
               (case_id, ts, amount, sealed, harm, threat, int(hidden)))

    def cases(self, status: str | None = None, limit: int = 200) -> list[dict]:
        rank = "CASE priority WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 WHEN 'P3' THEN 3 ELSE 9 END"
        where, args = ("WHERE status=?", (status,)) if status else ("", ())
        return self.q(f"SELECT c.*, (SELECT COUNT(*) FROM evidence e WHERE e.case_id=c.id) AS n_evidence "
                      f"FROM cases c {where} ORDER BY {rank}, score DESC, updated DESC LIMIT ?", (*args, limit))

    def case(self, case_id: int) -> dict | None:
        rows = self.q("SELECT * FROM cases WHERE id=?", (case_id,))
        return rows[0] if rows else None

    def evidence(self, case_id: int) -> list[dict]:
        return self.q("SELECT * FROM evidence WHERE case_id=? ORDER BY ts", (case_id,))

    def set_case(self, case_id: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE cases SET {cols} WHERE id=?", (*fields.values(), case_id))

    # -- recipient choices, labels, audit ------------------------------------
    def hide(self, recipient: str, sender: str, ts: float) -> None:
        self.x("INSERT OR IGNORE INTO hides VALUES (?,?,?)", (recipient, sender, ts))

    def label(self, case_id: int, analyst: str, label: str, features: list[float]) -> None:
        self.x("INSERT INTO labels VALUES (?,?,?,?,?)", (case_id, time.time(), analyst, label, json.dumps(features)))
        self.set_case(case_id, label=label, status="closed" if label == "not_abuse" else "confirmed")

    def audit(self, actor: str, action: str, case_id: int | None = None, detail: str = "") -> None:
        self.x("INSERT INTO audit VALUES (?,?,?,?,?)", (time.time(), actor, action, case_id, detail))

    def audit_log(self, limit: int = 100) -> list[dict]:
        return self.q("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,))

    def reset(self) -> None:
        with self._lock:
            for t in ("pairs", "cases", "evidence", "hides", "audit", "labels"):
                self.db.execute(f"DELETE FROM {t}")
