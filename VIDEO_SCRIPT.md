# SafePay: 3-minute demo video script

The hard limit is 3:00. Timings add up to 2:50, leaving 10 seconds of slack. Record the browser at 1440×900 or larger, with the SSH terminal ready in a second window.

**Before recording (about 15 minutes ahead):**
1. Free memory for Granite: `ssh ibmz "podman stop notebook"` (the lab container; restart it later with `bash run_labs.sh`). With it running, Granite slows from about 1.2 to 0.4 tokens/s.
2. On the VM, run `bash run_safepay.sh`. On the laptop, open the tunnel: `ssh -N -L 8090:127.0.0.1:8090 ibmz`.
3. On the demo page press **Reset**, then replay all four scenarios once, in order. Cases: #1 split threat (Jordan → Sam), #2 coercive control (Riley → Sam), #3 child support (Chris → Kim).
4. Wait until `/analyst` shows a "Granite brief" on cases #1 and #2 (1–2 minutes each).
5. Don't set Sam's message settings yet. You'll do that on camera.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Title card: "SafePay · IBM Z Datathon 2026 · SSF2026" | "When someone escapes an abuser, they block them everywhere. But you can't block a bank transfer. Abusers send one-cent payments with threats in the reference. Australia's biggest bank finds about 1,500 high-risk cases a year." |
| 0:20–0:42 | Demo page. Click **Threat split across 9 payments**. Point at Sam's app, then the joined-text box. | "A real reported case: one threat split across nine transfers so no single message trips a filter. SafePay joins them up, reads 'I am going to find you and kill you', and hides all nine. The money still arrives, because it could be child support she depends on." |
| 0:42–1:00 | Click **Polite but menacing**. Point at message score 0.0x and relationship score. | "Now no rude words at all: 'saw you at the gym today', 'I'm outside'. A toxicity model calls these harmless; our relationship model sees penny payments at night that are never answered. Friends calling each other 'loser' don't open a case." |
| 1:00–1:20 | Sam's app → **Message settings**. Set "In the app only", 10:00–14:00, tick "Someone else may see my phone". On Riley, choose "protection order" and **Allow a record**. Press **Quick exit**. | "Sam decides how the bank may contact her, safely, and can allow a signed record of Riley's payments for a protection order. One tap on quick exit, and her screen looks like an ordinary account." |
| 1:20–1:40 | Analyst console, case #2. Point at the safe-contact panel, then the Granite brief. | "The specialist team sees her rules first: in the app, ten till two, neutral wording. IBM Granite, on the same mainframe, drafts the brief and never writes to the sender, because tipping off an abuser is dangerous." |
| 1:40–1:58 | **Prepare signed evidence pack** → pack page → **Verify** → **Simulate tampering**. | "With her consent: a record of all eight payments, including those sent before the case opened, hash-chained and signed with Ed25519. Police can verify it. Add one full stop, and it fails." |
| 1:58–2:20 | Evidence page: real-time vs monthly table, then fairness table. | "Against the bank's published monthly approach: a median of 3 days to catch abuse instead of 20, and 94% of abusive messages withheld as they arrive. We caught our own model hiding 99.7% of harmless messages that mention 'gay', and fixed it." |
| 2:20–2:42 | SSH terminal: `uname -m`, `grep -m1 machine /proc/cpuinfo`, then the speed table. | "All of it runs on an IBM z17 in LinuxONE Community Cloud: big-endian s390x, about 900 payments a second in one process, well under a millisecond typically. We benchmarked the on-chip AI accelerator honestly: for our small model the CPU was faster, so that's what we ship. No data leaves the platform." |
| 2:42–2:50 | Back to Sam's app, then the title card. | "SafePay: the money gets through, the abuse doesn't, and the evidence does. Thank you." |

## Checklist

- [ ] Both mandatory labs done, with screenshots (they're required for the technical score)
- [ ] Screenshot of LinuxONE Community Cloud "Manage Instances" showing the VM
- [ ] Lab container stopped before recording, and Granite briefs present on cases #1 and #2
- [ ] Terminal font large enough to read
- [ ] Project title `[Team Name_IBM Z Datathon 2026 Project]`, and "SSF2026" in the description
