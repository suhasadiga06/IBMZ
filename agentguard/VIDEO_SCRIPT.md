# AgentGuard: 3-minute demo video script

Hard limit is 3:00. Timings below add up to 2:50, leaving 10 s of slack.
Record the browser at 1440×900 or larger. Have the SSH terminal ready in a second window.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Title card: "AgentGuard · a firewall for AI assistants · IBM Z Datathon 2026 · SSF2026" | "Banks are giving AI assistants access to customer accounts. The scary part: an attacker doesn't need to hack the bank. They just need the assistant to *read* a document they wrote." |
| 0:20–0:40 | Demo page. Click **Bulk export, then email it out**. Point at the left panel. | "Here's a real attack text from a public dataset, hidden in a document. The customer only asked for a summary. The unprotected assistant exported 28 confidential values and emailed them to an outside address." |
| 0:40–1:05 | Point at the right panel, then the three layer cards. | "Same attack, protected by AgentGuard. Three independent layers: input screening quarantined the document; the tool firewall would have blocked the export; and the output filter would have caught the data anyway. Each layer works on its own." |
| 1:05–1:25 | Click **Small transfer to a known payee**. Point at layer 2. | "This one is subtle: $215 to a saved payee, under the usual limit. Every rule passes. Our behaviour model still blocks it at risk 0.99, because money is moving right after the assistant read an outside document." |
| 1:25–1:40 | Click **Reveal the system prompt and API key**. Point at layer 3. | "And when no tool is involved, when the model just talks, the output filter catches the leaked key. That's defence in depth." |
| 1:40–1:55 | Click **Run all scenarios**. Let the scoreboard fill. | "Across all scenarios: eight of eight attacks breach the unprotected assistant. Zero get through AgentGuard. And zero false alarms on normal requests." |
| 1:55–2:15 | Scroll to the metrics table. | "We measured honestly. The classifier is scored on a different dataset it never saw: F1 0.97, 2% false alarms. A single-source model only reaches 0.90, which is why we train on two sources." |
| 2:15–2:40 | Switch to the SSH terminal. Run `uname -m`, `grep machine /proc/cpuinfo \| head -1`, then `python3 -m agentguard.scripts.demo_cli`. | "Everything runs on IBM LinuxONE: this is an IBM z17, s390x, big-endian. About one millisecond per prompt. Customer data, models and audit logs never leave the platform. No external AI API." |
| 2:40–2:50 | Back to the demo header, then title card. | "AgentGuard: let AI help customers, without letting it be turned against them. Thank you." |

## Checklist before recording

- [ ] Server running on the VM (`bash run.sh`) and SSH tunnel open
- [ ] Browser zoom at 100%, no other tabs visible
- [ ] Terminal font large enough to read in the video
- [ ] Both mandatory labs completed and screenshots saved (they're required for the technical score)
- [ ] Screenshot of the LinuxONE Community Cloud "Manage Instances" page showing the VM
