"""Run every scenario with and without AgentGuard and print a scoreboard.

Usage: python3 -m agentguard.scripts.demo_cli
"""
from __future__ import annotations

from agentguard.core import scenarios
from agentguard.core.guard import AgentGuard, Guards

MARK = {True: "X", False: "."}


def main() -> None:
    guard = AgentGuard(Guards.load())
    print(f"{'scenario':18s} {'kind':6s} | {'unprotected':24s} | in tool out | {'protected':10s} ms")
    print("-" * 86)
    totals = {"attacks": 0, "breached_unprot": 0, "breached_prot": 0, "benign": 0, "false_alarms": 0}
    for s in scenarios.all_scenarios():
        u = guard.run_unprotected(s["user"], s["document"], s["plan"])
        p = guard.run_protected(s["user"], s["document"], s["plan"])
        dmg = f"{u.leaked_values} leaked, {u.emails_out} mail, ${u.money_out:,.0f}"
        L = p.layers
        lights = " ".join(f" {MARK[L[k]['fired']]} " for k in ("input", "tools", "output"))
        print(f"{s['id']:18s} {s['kind']:6s} | {dmg:24s} | {lights} | {p.outcome:10s} {p.latency_ms['total']:.0f}")
        if s["kind"] == "attack":
            totals["attacks"] += 1
            totals["breached_unprot"] += u.outcome == "breached"
            totals["breached_prot"] += p.outcome == "breached"
        else:
            totals["benign"] += 1
            totals["false_alarms"] += bool(p.caught_by)
    print("-" * 86)
    print(f"attacks: {totals['breached_unprot']}/{totals['attacks']} breached without guard, "
          f"{totals['breached_prot']}/{totals['attacks']} with guard | "
          f"benign: {totals['false_alarms']}/{totals['benign']} false alarms")


if __name__ == "__main__":
    main()
