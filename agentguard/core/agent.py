"""The assistant being protected: a tool-using bank agent.

Two ways a turn can be driven:

* **Normal**: the agent maps the customer's own message to tool calls.
* **Hijacked**: we assume an injected document has already taken over the
  model (the worst case) and replay the tool calls it would make. This lets us
  measure how much damage is contained *after* the language model has failed,
  which is the question a bank's risk team actually asks.

Guards never live inside the agent. They are handed in as a `gate` callback
that is consulted before every tool call, plus checks on the input and output.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Callable

from .bank import Bank, SYSTEM_PROMPT, mask
from .tooltrace import Session

# Customer-intent patterns for normal (non-hijacked) turns.
INTENTS: list[tuple[str, re.Pattern]] = [
    ("transfer", re.compile(r"\b(?:transfer|send|pay|wire)\b[^.?!]{0,30}\$\s?([\d,]+(?:\.\d{2})?)", re.I)),
    ("recent_transactions", re.compile(
        r"\b(?:recent|last|past|previous)\b[^.?!]{0,25}\b(?:transaction|transfer|payment|activity)s?\b", re.I)),
    ("get_balance", re.compile(r"\b(?:balance|how much (?:do i have|money))\b", re.I)),
]
PAYEE_RE = re.compile(r"\bto\s+(?:my\s+)?([a-z]+)\b", re.I)

# The gate sees the tool, its arguments and the session *including* this call.
Gate = Callable[[str, dict, Session], "GateDecision"]


@dataclass
class GateDecision:
    allow: bool
    reason: str = ""
    score: float = 0.0


@dataclass
class Turn:
    user_message: str
    document: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    reply: str = ""
    session: Session = field(default_factory=Session)


class BankAgent:
    def __init__(self, bank: Bank, cid: str = "C1000"):
        self.bank = bank
        self.cid = cid
        self.customer = bank.customers[cid]

    def run(self, user_message: str, document: str = "", hijack_plan: list[dict] | None = None,
            gate: Gate | None = None) -> Turn:
        turn = Turn(user_message=user_message, document=document)
        turn.session.hist_mean = sum(self.customer.history) / len(self.customer.history)
        turn.session.hist_max = max(self.customer.history)
        if document:
            turn.session.add("read_document", rows_returned=1)

        steps = hijack_plan if hijack_plan is not None else self._plan(user_message)
        parts = [self._execute(step["tool"], dict(step.get("args", {})), turn, gate) for step in steps]
        if document and hijack_plan is None:
            parts.append(f"I've read the document you shared ({len(document.split())} words).")
        turn.reply = "\n".join(p for p in parts if p) or self._smalltalk()
        return turn

    def _plan(self, msg: str) -> list[dict]:
        steps = []
        for tool, pat in INTENTS:
            m = pat.search(msg)
            if not m:
                continue
            if tool == "transfer":
                payee = PAYEE_RE.search(msg)
                name = payee.group(1).lower() if payee else ""
                steps.append({"tool": "transfer", "args": {
                    "to": self.customer.payees.get(name, "unknown-payee"),
                    "amount": float(m.group(1).replace(",", ""))}})
            else:
                steps.append({"tool": tool})
        return steps

    def _execute(self, tool: str, args: dict, turn: Turn, gate: Gate | None) -> str:
        record = {"tool": tool, "args": args, "allowed": True, "reason": ""}
        turn.tool_calls.append(record)

        # Build the session as it would look if this call ran, and let the gate judge it.
        trial = copy.deepcopy(turn.session)
        self._log(trial, tool, args)
        # Revealing the prompt is the model talking, not a tool call, so there is
        # nothing for the gate to intercept. Only the output layer can stop it.
        if gate is not None and tool != "reveal_prompt":
            decision = gate(tool, args, trial)
            record.update(allowed=decision.allow, reason=decision.reason, score=round(decision.score, 3))
            if not decision.allow:
                return f"[AgentGuard blocked {tool}: {decision.reason}]"
        turn.session = trial
        record["result"] = self._call(tool, args, turn)
        return record["result"]

    def _log(self, s: Session, tool: str, args: dict) -> None:
        if tool == "transfer":
            s.add("transfer", amount=float(args.get("amount", 0)),
                  payee_known=args.get("to") in self.customer.payees.values())
        elif tool == "export_all_customers":
            s.add(tool, rows_returned=len(self.bank.customers))
        elif tool == "lookup_customer":
            s.add(tool, rows_returned=5)
        elif tool == "recent_transactions":
            s.add(tool, rows_returned=5)
        elif tool in ("get_balance", "send_email"):
            s.add(tool, rows_returned=1 if tool == "get_balance" else 0)
        # reveal_prompt is not a bank tool, so it leaves no trace: only the
        # output guard can catch it. That is deliberate, and shows why layers matter.

    def _call(self, tool: str, args: dict, turn: Turn) -> str:
        b = self.bank
        if tool == "get_balance":
            return b.get_balance(self.cid)
        if tool == "recent_transactions":
            return b.recent_transactions(self.cid)
        if tool == "transfer":
            return b.transfer(self.cid, args.get("to", "unknown-payee"), float(args.get("amount", 0)))
        if tool == "lookup_customer":
            return b.lookup_customer(args.get("query", ""))
        if tool == "export_all_customers":
            return b.export_all_customers()
        if tool == "send_email":
            earlier = "\n".join(c.get("result", "") for c in turn.tool_calls[:-1])
            return b.send_email(args.get("to", "unknown@example"), earlier or "(empty)")
        if tool == "reveal_prompt":
            return SYSTEM_PROMPT
        return ""

    def _smalltalk(self) -> str:
        return (f"Hello {self.customer.name.split()[0]}, I can help with account "
                f"{mask(self.customer.account)}: balance, recent transactions or a payment.")
