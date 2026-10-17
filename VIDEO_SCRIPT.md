# SafePay: 3-minute demo video script

The hard limit is 3:00. Timings add up to 2:50, leaving 10 seconds of slack. Record the browser at 1440×900 or larger, with the SSH terminal ready in a second window.

**Before recording (about 15 minutes ahead):**
1. Make sure the lab notebook container is stopped (`ssh ibmz "podman ps"` should not list `notebook`). With it running, Granite slows from about 1.2 to 0.4 tokens/s.
2. On the VM, run `bash run_safepay.sh`. On the laptop, open the tunnel: `ssh -N -L 8090:127.0.0.1:8090 ibmz`.
3. On the demo page press **Reset** just before the first take, then record the scenes in order. Scene 2 creates case #1 (Jordan → Sam) and scene 3 creates case #2 (Riley → Sam).
4. Between scenes 3 and 5, wait until `/analyst` shows a "Granite brief" on case #2 (1–2 minutes).
5. Click each scenario only once: a second click adds more payments to the same case, and the pack would no longer have "all eight".

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Title card: "SafePay · IBM Z Datathon 2026 · SSF2026" | "Hi, I'm Suhas, and this is SafePay. When someone escapes an abuser, they block them everywhere. But you can't block a bank transfer. Abusers send one-cent payments with threats in the reference, and Australia's biggest bank finds about 1,500 high-risk cases a year." |
| 0:20–0:42 | Demo page. Click **Threat split across 9 payments**. Point at Sam's app, then the joined-text box. | "This is a real reported pattern: one threat split across nine transfers, so no single message trips a filter. SafePay joins them up, reads 'I am going to find you and kill you', and hides all nine. The money still arrives, because it could be child support she depends on." |
| 0:42–1:00 | Click **Polite but menacing**. Point at the message score (0.0x), then the relationship score. | "Now there are no rude words at all: 'saw you at the gym today', 'I'm outside'. A toxicity model calls these harmless. My relationship model sees penny payments at night that are never answered, and opens a case." |
| 1:00–1:20 | Sam's app → **Message settings**. Set "In the app only", 10:00–14:00, tick "Someone else may see my phone", **Save**. On Riley, choose "protection order" → **Allow a record**. Press **Quick exit**. | "Sam decides how her bank may contact her: in the app only, ten till two, because someone else may see her phone. She allows a signed record of Riley's payments for a protection order. And one tap on quick exit makes her screen look like an ordinary account." |
| 1:20–1:40 | Analyst console, case #2. Point at the safe-contact panel, then the Granite brief. | "The bank's specialist team sees her rules first. IBM Granite, running on the same mainframe, drafts their brief, and it never writes to the sender, because tipping off an abuser can be dangerous." |
| 1:40–1:58 | **Prepare signed evidence pack** → pack page → **Verify** → **Simulate tampering**. | "With her consent, I prepare a record of all eight payments, including the ones sent before the case opened. It's hash-chained and signed with Ed25519, so police can verify it. Add one full stop, and verification fails." |
| 1:58–2:20 | Evidence page: real-time vs monthly table, then the stress-test section. | "Against the bank's published monthly approach, SafePay catches abuse in a median of 3 days instead of 19, and withholds 92% of abusive messages as they arrive. At a realistic scale, it raised 15 false alarms in 305,000 innocent relationships. And I found and fixed my model's bias against messages that mention 'gay'." |
| 2:20–2:42 | SSH terminal: `uname -m`, `grep -m1 machine /proc/cpuinfo`, then the speed table. | "All of this runs on an IBM z17 in LinuxONE Community Cloud: big-endian s390x, over 900 payments a second in one process, usually well under a millisecond each. I benchmarked the on-chip AI accelerator honestly: for this small model the CPU was faster, so that's what I ship. No data leaves the platform." |
| 2:42–2:50 | Back to Sam's app, then the title card. | "SafePay: the money gets through, the abuse doesn't, and the evidence does. Thank you." |

**Optional beat (swap for 0:42–1:00 if you prefer):** click **Hidden, then switches account**. Say: "Here Sam hides Casey, so Casey comes back from a new account. To any model that's a stranger. But for 14 days after a hide, new senders are held back, so even his first message is withheld."

The same script, formatted for reading while you record, is in `pitch/SafePay_recording_script.pdf`.

## How to record (Windows 11)

**Tool: Clipchamp**, Microsoft's free video editor, already on Windows 11 (Start → type "Clipchamp"). It records the screen and webcam together and lets you trim and export without a watermark at 1080p.

1. **Set up the screen.** Plug in the charger. Close other apps (Clipchamp itself uses about 1 GB of RAM). Press **Win + N** and turn on *Do not disturb*. Open the demo in Edge, maximise it, and keep zoom at 100%. If the three demo columns stack on top of each other, press **Ctrl + −** once.
2. **Start a recording.** Clipchamp → *Create a new video* → *Record & create* → **Screen and camera** (or just *Screen* if you'd rather not appear). Allow the camera and microphone, then pick the Edge window.
3. **Record one script row per take,** not the whole video in one go. A mistake then costs 20 seconds, not 3 minutes. Pause for a second before and after speaking, which makes trimming easy.
4. **The terminal shot (2:20).** Record the SSH window separately with *Screen*. Make the font large first (Ctrl + scroll wheel in Windows Terminal).
5. **Arrange.** Drag the takes onto the timeline in order and trim the gaps. If you used the camera, shrink it to a small circle in the **bottom-left of the middle panel** (the empty area under the screening trace), so it never covers Sam's app or the IBM z17 chip.
6. **Title card.** *Text* → add "SafePay · IBM Z Datathon 2026 · SSF2026" on a plain background for the first 5 seconds.
7. **Check the length** in the timeline: **3:00 or less.**
8. **Export** → **1080p**. Watch the whole file once before uploading.
9. **Upload** to YouTube as **Unlisted** (or Google Drive, shared with *Anyone with the link*), and paste the link into Hack2Skill.

**Voice:** earphones with a mic held near your mouth beat the laptop mic. Record in a quiet room, and do a 10-second test take first to check the level. The script is about 380 words for 2:50, a calm pace; don't rush it.

**Don't show:** the `.pem` key file or its folder, the Jupyter lab token, or your email inbox and notifications.

**Alternatives:** Snipping Tool (**Win + Shift + R**) records the screen and microphone with no editing. OBS Studio (free, obsproject.com) gives the most control if you're comfortable installing it.

## Checklist

- [ ] Both mandatory labs done, with screenshots (they're required for the technical score)
- [ ] Screenshot of LinuxONE Community Cloud "Manage Instances" showing the VM
- [ ] Lab container stopped before recording, and the Granite brief present on case #2 before scene 5
- [ ] Terminal font large enough to read
- [ ] Project title `Suhas Adiga_IBM Z Datathon 2026 Project`, and "SSF2026" in the description
- [ ] Live demo running (`bash run_safepay.sh public`), port 8090 open, and http://148.100.76.164:8090 opens from your own browser; the login (user `judge` and the password) goes in the description
