# SafePay: 3-minute demo video script

The hard limit is 3:00. Timings add up to 2:50, leaving 10 seconds of slack. Record the browser at 1440×900 or larger, with the SSH terminal ready in a second window.

**Before recording (about 10 minutes ahead):**
1. On the VM, run `bash run_safepay.sh`. On the laptop, open the tunnel: `ssh -N -L 8090:127.0.0.1:8090 ibmz`.
2. On the demo page press **Reset**, then replay all four scenarios once, so Granite has time to write the briefs (about 1–2 minutes per case). Check `/analyst` shows a "Granite brief" on case #1.
3. Press **Reset** again only if you want to show the scenarios live from scratch. Then let Granite finish again before you show the analyst console.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Title card: "SafePay · IBM Z Datathon 2026 · SSF2026" | "When someone escapes an abuser, they block them everywhere. But you can't block a bank transfer. Abusers send one-cent payments with threats in the reference. Australia's biggest bank finds about 1,500 high-risk cases a year." |
| 0:20–0:45 | Demo page. Click **Threat split across 9 payments**. Point at Sam's app, then the joined-text box. | "This is a real reported case: one threat split across nine transfers so no single message trips a filter. SafePay joins them up, reads 'I am going to find you and kill you', and hides all nine from Sam. The money still arrives, because it could be child support she depends on." |
| 0:45–1:05 | Click **Polite but menacing**. Point at message score 0.0x and relationship score. | "Now no rude words at all: 'saw you at the gym today', 'I'm outside'. A toxicity model scores these as harmless. Our relationship model sees the pattern: penny payments at night, no replies, contact after she hid him." |
| 1:05–1:20 | Click **Friends' banter**. | "And friends calling each other 'loser' get a soft hide at most. No case, nobody investigated." |
| 1:20–1:45 | Analyst console, case #1. Timeline, evidence, Granite brief. | "The specialist team sees cases ranked by risk, with decrypted evidence and an audit trail. IBM Granite, running on the same mainframe, writes the brief and gives a second opinion. It never writes anything to the sender, because tipping off an abuser can be dangerous." |
| 1:45–2:15 | Evidence page: the real-time vs monthly table, then the fairness table. | "We compared against the bank's published monthly approach on a held-out world. Same accuracy, but a median of 3.2 days to catch abuse instead of 20, and 94% of abusive messages withheld as they arrive. We also caught our own model hiding 99.7% of harmless messages that mention 'gay', and fixed it to 13%, the same as any sentence." |
| 2:15–2:40 | SSH terminal: `uname -m`, `grep -m1 machine /proc/cpuinfo`, then the speed table on the evidence page. | "All of it runs on an IBM z17 in LinuxONE Community Cloud: big-endian s390x, 700 payments a second in one process, under a millisecond typically. We benchmarked the on-chip AI accelerator honestly: for our small model the CPU was faster, so that's what we ship. No data leaves the platform." |
| 2:40–2:50 | Back to the demo, Sam's app, then the title card. | "SafePay: the money gets through, the abuse doesn't. Thank you." |

## Checklist

- [ ] Both mandatory labs done, with screenshots (they're required for the technical score)
- [ ] Screenshot of LinuxONE Community Cloud "Manage Instances" showing the VM
- [ ] Granite briefs present before you open the analyst console
- [ ] Terminal font large enough to read
- [ ] Project title `[Team Name_IBM Z Datathon 2026 Project]`, and "SSF2026" in the description
