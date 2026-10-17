"""Incident notes for station staff, written by IBM Granite running on the same LinuxONE.

Granite (via llama.cpp's OpenAI-compatible server) turns the facts of an incident into a
short note. It never decides anything: the risk model and the ladder have already acted,
and an operator approves any stop. Notes are written in a background thread; until one
arrives (or if Granite is unavailable) the incident carries a plain template note.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

URL = os.environ.get("CHARGEGUARD_LLM_URL", "http://127.0.0.1:8081")

# The fixed instructions come *before* the incident facts, and are identical every time:
# llama.cpp reuses the cached work for a shared prompt prefix (cache_prompt), so only the
# facts are processed per incident. On 3 shared cores, that and a shorter note are most of
# the speed-up (same approach as SafePay's Granite briefs).
SYSTEM = ("You write short incident notes for EV charging station staff. This is an early warning from "
          "charging data, not a diagnosis: never say the battery will catch fire or that a fault is confirmed.")
TASK = ("Write two short sentences of plain prose for the operator (no numbering, no lists): first what was "
        "unusual about this charge, using only the facts below; then what to check before approving or "
        "dismissing the stop.\n\nThe flagged charging session:\n")
# Granite on 3 shared cores generates about 1 token a second at best: keep notes short.
MAX_TOKENS = 50

ACTIONS = {"set_charging_profile": "capped the charging current through the charger's smart-charging interface",
           "escalate_operator": "could not cap the current (this charger has no smart charging) and alerted an operator",
           "await_approval": "kept the charge running and is waiting for an operator"}


def riskier_than(p: float) -> str:
    """'99.4%', or 'more than 99.9%' instead of a rounded-up, impossible '100.0%'."""
    return "more than 99.9%" if p >= 0.999 else f"{p:.1%}"


def facts_text(f: dict) -> str:
    """The incident as plain sentences: a small model misreads raw scores and field names."""
    lines = [f"Charger {f['cp']}, connector {f['connector']}, {f['battery']} battery, minute {f['minute']} of the charge.",
             f"This charge is riskier than {riskier_than(f['percentile'])} of normal charges.",
             "Unusual readings: " + "; ".join(f["reasons"]) + ".",
             f"Gun temperatures now: {f['gun_temperatures_c']} °C. Energy so far: {f['energy_kwh']} kWh.",
             f"The system {ACTIONS.get(f['action'], 'is watching the charge')}" +
             (f" (limit {f['current_limit_amps']:.0f} A)." if f.get("current_limit_amps") else "."),
             "A stop has been recommended and needs an operator's approval."]
    return "\n".join(lines)


def template(f: dict) -> str:
    did = ACTIONS.get(f["action"], "kept watching")
    return (f"{f['cp']} connector {f['connector']} ({f['battery']} battery, minute {f['minute']}): "
            f"riskier than {riskier_than(f['percentile'])} of normal charges. Unusual: {'; '.join(f['reasons'])}. "
            f"ChargeGuard {did}. Check the vehicle and connector before approving the stop.")


def available() -> bool:
    try:
        with urllib.request.urlopen(URL + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def complete_sentences(text: str) -> str:
    """The token budget can end mid-sentence; never show a broken one."""
    text = text.strip()
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind(".\n"))
    if text and text[-1] not in ".!?" and cut > 0:
        text = text[:cut + 1]
    return text


def write(f: dict, timeout: int = 300) -> str:
    return write_timed(f, timeout)[0]


def write_timed(f: dict, timeout: int = 300) -> tuple[str, dict]:
    """The note and llama.cpp's timings (prompt tokens processed, tokens generated, ms)."""
    body = json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": TASK + facts_text(f)}],
                       "max_tokens": MAX_TOKENS, "temperature": 0.1, "cache_prompt": True}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return complete_sentences(d["choices"][0]["message"]["content"]), d.get("timings", {})


class Writer:
    """One background thread that always works on the newest incident.

    On a small VM Granite can take minutes per note, so a backlog is never queued: when a
    new incident arrives, older ones that haven't started keep their template note.
    Calls done(incident_id, note) when Granite answers.
    """

    def __init__(self, done, timeout: int = 900):
        self.done, self.timeout = done, timeout
        self.next: tuple | None = None
        self.cv = threading.Condition()
        self.skipped = 0
        self.enabled = available()
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def submit(self, incident_id: int, facts: dict) -> None:
        if not self.enabled:
            return
        with self.cv:
            if self.next is not None:
                self.skipped += 1
            self.next = (incident_id, facts)
            self.cv.notify()

    def _run(self) -> None:
        while True:
            with self.cv:
                while self.next is None:
                    self.cv.wait()
                incident_id, facts = self.next
                self.next = None
            try:
                self.done(incident_id, write(facts, self.timeout))
            except Exception:
                pass  # the template note stays
