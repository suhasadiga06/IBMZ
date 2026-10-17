"""Turn a guard verdict into an incident report a human can act on.

The facts (scores, tools, evidence) always come from the guards. The language
model only phrases them, and if no model is available the same facts are laid
out with a template. Either way, nothing is invented and nothing leaves the
machine.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .guard import RunResult
from .llm import LLM, TemplateLLM

LAYER_NAMES = {"input": "Input screening", "tools": "Tool-call firewall", "output": "Output leak filter"}


def facts(scenario: dict, unprot: RunResult, prot: RunResult) -> dict:
    L = prot.layers
    return {
        "scenario": scenario["title"],
        "kind": scenario["kind"],
        "without_guard": {"leaked_values": unprot.leaked_values, "emails_out": unprot.emails_out,
                          "money_out": unprot.money_out, "tools": [c["tool"] for c in unprot.tool_calls]},
        "with_guard": {"leaked_values": prot.leaked_values, "emails_out": prot.emails_out,
                       "money_out": prot.money_out, "outcome": prot.outcome},
        "input": {"document_score": L["input"]["document_score"], "user_score": L["input"]["user_score"],
                  "threshold": L["input"]["threshold"], "action": L["input"]["action"],
                  "evidence": L["input"]["evidence"], "obfuscation": L["input"]["obfuscation"]},
        "tools_blocked": [d for d in L["tools"]["decisions"] if not d["allowed"]],
        "output": {"kinds": L["output"]["kinds"], "action": L["output"]["action"]},
        "caught_by": [LAYER_NAMES[x] for x in prot.caught_by],
        "latency_ms": prot.latency_ms.get("total"),
    }


def _template(f: dict) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w, g = f["without_guard"], f["with_guard"]
    if f["kind"] == "benign" and not f["caught_by"]:
        return (f"No incident. '{f['scenario']}' was a normal request and passed all three layers "
                f"in {f['latency_ms']} ms. Nothing was blocked.")
    if f["kind"] == "benign":
        return (f"False alarm on a normal request ('{f['scenario']}'), flagged by "
                f"{', '.join(f['caught_by'])}. The customer would see friction here; "
                f"this case should be reviewed to tune thresholds.")

    lines = [f"INCIDENT REPORT  ·  {when}", f"Scenario: {f['scenario']}", ""]
    impact = []
    if w["leaked_values"]:
        impact.append(f"{w['leaked_values']} confidential values from other customers")
    if w["emails_out"]:
        impact.append(f"{w['emails_out']} email(s) to an outside address")
    if w["money_out"]:
        impact.append(f"${w['money_out']:,.2f} sent to an unknown account")
    lines.append("What would have happened without AgentGuard: "
                 + ("; ".join(impact) if impact else "no measurable damage") + ".")
    lines.append(f"With AgentGuard: {g['leaked_values']} values leaked, {g['emails_out']} emails out, "
                 f"${g['money_out']:,.2f} moved. Outcome: {g['outcome'].upper()}.")
    lines.append("")
    i = f["input"]
    if i["action"] != "passed":
        ev = ", ".join(f"'{e}'" for e in i["evidence"][:4])
        lines.append(f"1. Input screening {i['action']} (score {i['document_score'] or i['user_score']:.2f} "
                     f"vs threshold {i['threshold']:.2f}). Strongest signals: {ev}.")
        if i["obfuscation"]:
            lines.append(f"   Obfuscation detected: {', '.join(f'{k}={v}' for k, v in i['obfuscation'].items())}.")
    else:
        lines.append(f"1. Input screening let the document through (score "
                     f"{(i['document_score'] or 0):.2f} < {i['threshold']:.2f}). Later layers had to act.")
    if f["tools_blocked"]:
        for d in f["tools_blocked"]:
            lines.append(f"2. Tool firewall would block {d['tool']}: {d['reason']}.")
    else:
        lines.append("2. Tool firewall saw no dangerous tool calls.")
    if f["output"]["kinds"]:
        lines.append(f"3. Output filter found {', '.join(f['output']['kinds'])}.")
    else:
        lines.append("3. Output filter found nothing sensitive in the reply.")
    lines += ["", f"Independent layers that would have stopped this alone: "
              f"{', '.join(f['caught_by']) or 'none'}.",
              "Recommended action: review the source of the document, keep the sender on watch, "
              "and confirm no customer data left the platform (audit log attached)."]
    return "\n".join(lines)


def incident_report(scenario: dict, unprot: RunResult, prot: RunResult, llm: LLM | None = None) -> dict:
    f = facts(scenario, unprot, prot)
    base = _template(f)
    llm = llm or TemplateLLM()
    text = ""
    if not isinstance(llm, TemplateLLM) and f["kind"] == "attack":
        prompt = ("You are a bank security analyst. Rewrite the incident notes below as a short, calm "
                  "report for a branch manager: what happened, what was stopped, what to do next. "
                  "Use only the facts given. Do not add numbers.\n\nNOTES:\n" + base + "\n\nREPORT:\n")
        text = llm.complete(prompt, max_tokens=220)
    return {"text": text or base, "generator": llm.name if text else "template", "facts": f}
