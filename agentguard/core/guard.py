"""AgentGuard: three independent layers around a tool-using assistant.

  Layer 1  input    injection classifier on the customer's message and on any
                    untrusted document. Flagged documents are quarantined so
                    the model never reads them.
  Layer 2  tools    every tool call is checked against channel policy and a
                    behaviour model that scores the whole session so far.
  Layer 3  output   the reply is checked for other customers' data and
                    credentials, and redacted before it leaves.

Each layer is also evaluated in *shadow* mode on every turn, i.e. as if the
layers before it had missed. That is how the demo shows defence in depth
honestly: you can see which layers would have stopped an attack on their own.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import leakdetect, tooltrace
from .agent import BankAgent, GateDecision, Turn
from .bank import Bank
from .injection import InjectionModel

ART = Path(__file__).resolve().parents[2] / "artifacts"
CUSTOMER_TOOLS = {"get_balance", "recent_transactions", "transfer", "send_email"}
QUARANTINE_NOTE = ("I set that document aside: it contained instructions aimed at me rather "
                   "than information for you.")


@dataclass
class Guards:
    injection: InjectionModel
    trace: object                      # sklearn classifier over TraceFeatures
    trace_threshold: float = 0.5

    @classmethod
    def load(cls, art: Path = ART) -> "Guards":
        trace = joblib.load(art / "trace.joblib")
        trace.n_jobs = 1  # one row per call: thread start-up costs more than it saves
        return cls(injection=joblib.load(art / "injection.joblib"), trace=trace)


@dataclass
class RunResult:
    mode: str
    reply: str
    tool_calls: list[dict]
    leaked_values: int
    emails_out: int
    money_out: float
    layers: dict = field(default_factory=dict)
    outcome: str = ""
    caught_by: list[str] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)


def _damage(bank: Bank, cid: str, turn: Turn, hijacked: bool) -> tuple[int, int, float]:
    """What actually left the bank: foreign secrets in reply/outbox, emails, money.

    In a hijacked turn every transfer is unauthorised, even to a saved payee:
    the customer never asked for it.
    """
    cust = bank.customers[cid]
    foreign = bank.known_secrets() - {cust.account, cust.card, cust.national_id, cust.email}
    sent = turn.reply + "\n" + "\n".join(m["body"] for m in bank.outbox)
    leaked = sum(1 for s in foreign if s in sent)
    emails = sum(1 for m in bank.outbox if m["to"] != cust.email)
    money = sum(t["amount"] for t in bank.transfers
                if hijacked or t["to"] not in cust.payees.values())
    return leaked, emails, round(money, 2)


class AgentGuard:
    def __init__(self, guards: Guards):
        self.g = guards

    # -- Layer 2 ------------------------------------------------------------
    def tool_gate(self, agent: BankAgent):
        cust = agent.customer

        def gate(tool: str, args: dict, session: tooltrace.Session) -> GateDecision:
            feats = tooltrace.extract(session)
            risk = float(self.g.trace.predict_proba(np.array([feats.vector()]))[0, 1])
            policy = ""
            if tool not in CUSTOMER_TOOLS:
                policy = f"'{tool}' is not permitted in the customer channel"
            elif tool == "send_email" and args.get("to") != cust.email:
                policy = "email to an address that is not the customer's own"
            elif tool == "transfer" and args.get("to") not in cust.payees.values() \
                    and float(args.get("amount", 0)) > max(cust.history):
                policy = "large transfer to an unknown payee needs step-up confirmation"
            reasons = [f"policy: {policy}"] if policy else []
            if risk >= self.g.trace_threshold:
                reasons.append(f"behaviour model: session risk {risk:.2f} ({_why(feats)})")
            return GateDecision(allow=not reasons, reason="; ".join(reasons), score=risk)

        return gate

    # -- Runs ---------------------------------------------------------------
    def run_unprotected(self, user: str, document: str = "", plan: list[dict] | None = None,
                        cid: str = "C1000") -> RunResult:
        bank = Bank()
        turn = BankAgent(bank, cid).run(user, document, plan)
        leaked, emails, money = _damage(bank, cid, turn, hijacked=plan is not None)
        return RunResult("unprotected", turn.reply, _public(turn.tool_calls), leaked, emails, money,
                         outcome="breached" if (leaked or emails or money) else "allowed")

    def run_protected(self, user: str, document: str = "", plan: list[dict] | None = None,
                      cid: str = "C1000") -> RunResult:
        t0 = time.perf_counter()
        layers: dict = {}

        # Layer 1: input.
        u = self.g.injection.verdict(user)
        d = self.g.injection.verdict(document) if document else None
        t1 = time.perf_counter()
        layers["input"] = {
            "fired": u.flagged or bool(d and d.flagged),
            "user_score": round(u.score, 3),
            "document_score": round(d.score, 3) if d else None,
            "evidence": (d.evidence if d and d.flagged else u.evidence) if (u.flagged or (d and d.flagged)) else [],
            "obfuscation": {k: v for k, v in (d.obfuscation if d else u.obfuscation).items() if v},
            "threshold": round(self.g.injection.threshold, 3),
        }

        # Shadow pass: run layers 2 and 3 as if layer 1 had missed.
        shadow_bank = Bank()
        shadow_agent = BankAgent(shadow_bank, cid)
        real_gate = self.tool_gate(shadow_agent)
        shadow_decisions: list[dict] = []

        def recording_gate(tool, args, session):
            dec = real_gate(tool, args, session)
            shadow_decisions.append({"tool": tool, "allowed": dec.allow, "reason": dec.reason,
                                     "score": round(dec.score, 3)})
            return GateDecision(True)
        shadow_turn = shadow_agent.run(user, document, plan, gate=recording_gate)
        cust = shadow_bank.customers[cid]
        own = {cust.account, cust.card, cust.national_id, cust.email}
        shadow_out = "\n".join([shadow_turn.reply] + [m["body"] for m in shadow_bank.outbox])
        shadow_leak = leakdetect.check(shadow_out, shadow_bank.known_secrets(), own)
        t2 = time.perf_counter()

        # Enforced pass.
        bank = Bank()
        agent = BankAgent(bank, cid)
        quarantined = False
        if u.flagged:
            turn = Turn(user_message=user, document=document,
                        reply="I can't help with that request. It has been logged for review.")
            action_in = "blocked request"
        else:
            quarantined = bool(d and d.flagged)
            # A quarantined document never reaches the model, so it cannot hijack it.
            turn = agent.run(user, "" if quarantined else document,
                             None if quarantined else plan, gate=self.tool_gate(agent))
            if quarantined:
                turn.document = document
                turn.reply = QUARANTINE_NOTE + (
                    "" if turn.tool_calls else
                    " Nothing in it was acted on. If you expected a normal document, "
                    "please contact the sender through a channel you trust.")
            action_in = "quarantined document" if quarantined else "passed"
        layers["input"]["action"] = action_in

        layers["tools"] = {
            "fired": any(not c["allowed"] for c in shadow_decisions),
            "decisions": shadow_decisions,
            "enforced": [c for c in _public(turn.tool_calls) if not c.get("allowed", True)],
        }

        c = bank.customers[cid]
        own = {c.account, c.card, c.national_id, c.email}
        verdict = leakdetect.check(turn.reply, bank.known_secrets(), own)
        if verdict.flagged:
            turn.reply = leakdetect.redact(turn.reply, bank.known_secrets() - own)
        layers["output"] = {
            "fired": shadow_leak.flagged,
            "score": shadow_leak.score,
            "kinds": shadow_leak.kinds,
            "action": "redacted" if verdict.flagged else "passed",
        }
        t3 = time.perf_counter()

        hijacked = plan is not None and not quarantined and not u.flagged
        leaked, emails, money = _damage(bank, cid, turn, hijacked)
        caught =[name for name in ("input", "tools", "output") if layers[name]["fired"]]
        res = RunResult("protected", turn.reply, _public(turn.tool_calls), leaked, emails, money,
                        layers=layers, caught_by=caught)
        res.outcome = ("breached" if (leaked or emails or money)
                       else "blocked" if caught else "allowed")
        res.latency_ms = {"input": round(1000 * (t1 - t0), 2),
                          "tools_and_output": round(1000 * (t3 - t1), 2),
                          "total": round(1000 * (t3 - t0), 2)}
        return res


def _why(f: tooltrace.TraceFeatures) -> str:
    cues = []
    if f.egress_after_untrusted:
        cues.append("outbound action right after reading an untrusted document")
    if f.egress_after_bulk:
        cues.append("outbound action after bulk record access")
    if f.unknown_payee_transfers:
        cues.append("payment to an unknown payee")
    if f.max_amount_ratio > 1.5:
        cues.append(f"amount {f.max_amount_ratio:.1f}x the customer's usual maximum")
    if f.bulk_calls:
        cues.append(f"{f.bulk_calls} bulk lookup(s)")
    return ", ".join(cues) or "unusual sequence"


def _public(calls: list[dict]) -> list[dict]:
    """Tool calls without raw results, which may contain the very data we protect."""
    return [{k: v for k, v in c.items() if k != "result"} for c in calls]
