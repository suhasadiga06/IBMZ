"""Durable state: relationship states, cases, evidence, recipient choices, audit.

SQLite in WAL mode: a single file, transactional, and fast enough for a
prototype. Only pseudonymous tokens are stored; evidence text is encrypted.

The evidence log is a hash chain per case: each row stores
SHA-256(previous row's hash + this row), so changing or deleting any stored
row breaks every hash after it, and verify_chain() finds the first bad row.
"""
from __future__ import annotations

import hashlib
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
                                     amount REAL, sealed_ref TEXT, harm REAL, threat REAL, hidden INTEGER,
                                     chain TEXT);
CREATE TABLE IF NOT EXISTS hides (recipient TEXT, sender TEXT, ts REAL, PRIMARY KEY (recipient, sender));
CREATE TABLE IF NOT EXISTS audit (ts REAL, actor TEXT, action TEXT, case_id INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS labels (case_id INTEGER, ts REAL, analyst TEXT, label TEXT, features TEXT);
CREATE TABLE IF NOT EXISTS prefs (recipient TEXT PRIMARY KEY, data TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS consents (id INTEGER PRIMARY KEY AUTOINCREMENT, recipient TEXT, sender TEXT,
                                     purpose TEXT, granted REAL, withdrawn REAL);
CREATE TABLE IF NOT EXISTS packs (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, created REAL,
                                  analyst TEXT, sha256 TEXT UNIQUE, sealed_body TEXT, consent_id INTEGER);
CREATE INDEX IF NOT EXISTS ev_case ON evidence(case_id);
"""
TABLES = ("pairs", "cases", "evidence", "hides", "audit", "labels", "prefs", "consents", "packs")


def genesis(case_id: int) -> str:
    return hashlib.sha256(f"safepay:evidence:case:{case_id}".encode()).hexdigest()


def link(prev: str, case_id: int, ts: float, amount: float, sealed: str, harm: float, threat: float,
         hidden: bool) -> str:
    # Explicit types: SQLite returns REAL columns as floats, so 1780000000 must hash as 1780000000.0.
    row = json.dumps([int(case_id), round(float(ts), 3), round(float(amount), 2), str(sealed),
                      round(float(harm), 6), round(float(threat), 6), int(bool(hidden))], separators=(",", ":"))
    return hashlib.sha256((prev + row).encode()).hexdigest()


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        try:  # databases created before the hash chain existed
            self.db.execute("ALTER TABLE evidence ADD COLUMN chain TEXT")
        except sqlite3.OperationalError:
            pass

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
                     hidden: bool) -> str:
        """Append one row to the case's hash chain. Returns the new chain head."""
        with self._lock:
            last = self.db.execute("SELECT chain FROM evidence WHERE case_id=? ORDER BY id DESC LIMIT 1",
                                   (case_id,)).fetchone()
            head = link(last["chain"] if last and last["chain"] else genesis(case_id),
                        case_id, ts, amount, sealed, harm, threat, hidden)
            self.db.execute("INSERT INTO evidence (case_id, ts, amount, sealed_ref, harm, threat, hidden, chain) "
                            "VALUES (?,?,?,?,?,?,?,?)", (case_id, ts, amount, sealed, harm, threat, int(hidden), head))
        return head

    def verify_chain(self, case_id: int) -> dict:
        """Recompute the case's chain from the stored rows."""
        rows = self.q("SELECT * FROM evidence WHERE case_id=? ORDER BY id", (case_id,))
        prev = genesis(case_id)
        for r in rows:
            prev = link(prev, case_id, r["ts"], r["amount"], r["sealed_ref"], r["harm"], r["threat"], bool(r["hidden"]))
            if prev != r["chain"]:
                return {"intact": False, "entries": len(rows), "first_bad_entry": r["id"], "head": None}
        return {"intact": True, "entries": len(rows), "first_bad_entry": None, "head": prev}

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

    # -- victim safety: preferences, consent, evidence packs -------------------
    def set_prefs(self, recipient: str, data: dict) -> None:
        self.x("INSERT INTO prefs VALUES (?,?,?) ON CONFLICT(recipient) DO UPDATE SET data=excluded.data, "
               "updated=excluded.updated", (recipient, json.dumps(data), time.time()))

    def prefs(self, recipient: str) -> dict | None:
        rows = self.q("SELECT data, updated FROM prefs WHERE recipient=?", (recipient,))
        return {**json.loads(rows[0]["data"]), "updated": rows[0]["updated"]} if rows else None

    def consent(self, recipient: str, sender: str) -> dict | None:
        """The active (not withdrawn) consent for this sender, if any."""
        rows = self.q("SELECT * FROM consents WHERE recipient=? AND sender=? AND withdrawn IS NULL "
                      "ORDER BY id DESC LIMIT 1", (recipient, sender))
        return rows[0] if rows else None

    def grant_consent(self, recipient: str, sender: str, purpose: str) -> int:
        self.withdraw_consent(recipient, sender)
        return self.x("INSERT INTO consents (recipient, sender, purpose, granted) VALUES (?,?,?,?)",
                      (recipient, sender, purpose, time.time())).lastrowid

    def withdraw_consent(self, recipient: str, sender: str) -> None:
        self.x("UPDATE consents SET withdrawn=? WHERE recipient=? AND sender=? AND withdrawn IS NULL",
               (time.time(), recipient, sender))

    def add_pack(self, case_id: int, analyst: str, sha256: str, sealed_body: str, consent_id: int) -> int:
        return self.x("INSERT INTO packs (case_id, created, analyst, sha256, sealed_body, consent_id) "
                      "VALUES (?,?,?,?,?,?)", (case_id, time.time(), analyst, sha256, sealed_body, consent_id)).lastrowid

    def packs(self, case_id: int) -> list[dict]:
        return self.q("SELECT id, case_id, created, analyst, sha256, consent_id FROM packs WHERE case_id=? "
                      "ORDER BY id DESC", (case_id,))

    def pack(self, pack_id: int) -> dict | None:
        rows = self.q("SELECT * FROM packs WHERE id=?", (pack_id,))
        return rows[0] if rows else None

    def pack_by_hash(self, sha256: str) -> dict | None:
        rows = self.q("SELECT id, case_id, created, analyst FROM packs WHERE sha256=?", (sha256,))
        return rows[0] if rows else None

    def audit(self, actor: str, action: str, case_id: int | None = None, detail: str = "") -> None:
        self.x("INSERT INTO audit VALUES (?,?,?,?,?)", (time.time(), actor, action, case_id, detail))

    def audit_log(self, limit: int = 100) -> list[dict]:
        return self.q("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,))

    def reset(self) -> None:
        with self._lock:
            for t in TABLES:
                self.db.execute(f"DELETE FROM {t}")
