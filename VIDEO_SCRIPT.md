# SafePay: 3-minute video script

The video moves between the pitch deck and the live demo while you speak: a slide explains each idea, then the demo shows it working. The hard limit is 3:00; the timings add up to 2:52. The same script, formatted for reading while you record, is in `pitch/SafePay_recording_script.pdf`.

**Before recording (about 15 minutes ahead):**
1. Make sure the lab notebook container is stopped (`ssh ibmz "podman ps"` should not list `notebook`). With it running, Granite slows from about 1.2 to 0.4 tokens/s.
2. On the VM, run `bash ~/safepay/run_safepay.sh`. On the laptop, open the tunnel: `ssh -N -L 8090:127.0.0.1:8090 ibmz`.
3. In one Edge window, open two tabs: tab 1 the deck (`pitch/SafePay_pitch_deck.pdf`, **Ctrl + \\** for one page at a time), tab 2 the demo (http://localhost:8090). Switch with **Ctrl + 1** / **Ctrl + 2**.
4. Press **Reset** on the demo just before the first take, then record the scenes in order. Scene 4 creates case #1 (Jordan → Sam) and scene 6 creates case #2 (Riley → Sam).
5. Click each scenario only once: a second click adds payments to the same case, and the pack would no longer have "all eight".
6. After scene 6, wait until `/analyst` shows a "Granite brief" on case #2 (1–2 minutes) before recording scene 8.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | Slides 1–2 | "Hi, I'm Suhas, and this is SafePay. When someone escapes an abuser, they block them everywhere. But you can't block a bank transfer, so abusers send one-cent payments with threats in the reference." |
| 0:15–0:30 | Slide 3 | "Banks use banned-word lists, which abusers dodge with disguised spellings, or by splitting one threat across nine transfers. The best published model scores relationships once a month, so victims are exposed for weeks." |
| 0:30–0:45 | Slide 4, then switch to the demo | "SafePay sits inside the payment flow on IBM Z. It reads real ISO 20022 payment messages and makes a decision on every payment in about a millisecond. Here it is, running live." |
| 0:45–1:05 | **Demo:** click **Threat split across 9 payments**. Point at the joined-text box, then Sam's app. | "This threat is split across nine transfers, so no single message trips a filter. SafePay joins them up, reads 'I am going to find you and kill you', and hides all nine. The money still arrives, because it could be child support." |
| 1:05–1:17 | Slide 5 | "That's one of three layers: a message model that undoes disguised spellings, a joiner for split messages, and a relationship model on IBM Snap ML." |
| 1:17–1:35 | **Demo:** click **Polite but menacing**. Point at the message score, then the relationship score. | "Here there are no rude words: 'saw you at the gym today', 'I'm outside'. The message scores near zero. But the relationship model sees penny payments at night, never answered, and opens a case." |
| 1:35–1:55 | **Demo:** Sam's app → **Message settings**. "In the app only", tick "Someone else may see my phone", **Save**. On Riley, "protection order" → **Allow a record**. **Quick exit**. | "The victim stays in control. Sam chooses in-app contact only, because someone else may see her phone, and allows a record of Riley's payments for a protection order. One tap on quick exit hides everything." |
| 1:55–2:17 | **Demo:** analyst console, case #2: safe-contact panel, Granite brief → **Prepare signed evidence pack** → **Verify** → **Simulate tampering**. | "The specialist team sees her rules first, and IBM Granite, running on the same machine, drafts their brief. With her consent, they prepare a signed evidence pack of all eight payments. Change one full stop, and verification fails." |
| 2:17–2:32 | Slide 7 | "Against the bank's monthly approach, SafePay opens a case in a median of 3 days instead of 19, and withholds 92% of abusive messages as they arrive." |
| 2:32–2:45 | Slide 11 | "It all runs on an IBM z17 in LinuxONE Community Cloud: over 900 payments a second, encrypted and pseudonymised, with no data leaving the platform." |
| 2:45–2:52 | Slide 13 | "SafePay: the money gets through, the abuse doesn't, and the evidence does. Thank you." |

## How to record (Windows 11)

**Tool: Clipchamp**, Microsoft's free video editor, already on Windows 11 (Start → type "Clipchamp"). It exports at 1080p without a watermark.

1. **Set up the screen.** Plug in the charger, close other apps, press **Win + N** → *Do not disturb*. Edge maximised, zoom 100%. If the three demo columns stack on top of each other, press **Ctrl + −** once.
2. **Record.** *Create a new video* → *Record & create* → **Screen and camera** (or just *Screen*). Pick the Edge window with both tabs, so switching tabs mid-take is recorded too.
3. **One scene per take.** Pause for a second before and after speaking, which makes trimming easy.
4. **Arrange.** Put the takes on the timeline in order and trim the gaps. If you used the camera, shrink it to a small circle under the demo's middle panel, so it never covers Sam's app or a slide's numbers.
5. **Check the length: 3:00 or less.** Export at **1080p** and watch it once.
6. **Upload** to YouTube as **Unlisted** (or Google Drive, *Anyone with the link*), and paste the link into Hack2Skill.

**Voice:** earphones with a mic beat the laptop mic. Quiet room, and a 10-second test take first.

**Don't show:** the `.pem` key file or its folder, the Jupyter lab token, or your email and notifications.

## Checklist

- [ ] Both mandatory labs done, with screenshots (they're required for the technical score)
- [ ] Screenshot of LinuxONE Community Cloud "Manage Instances" showing the VM
- [ ] Lab container stopped before recording, and the Granite brief present on case #2 before scene 8
- [ ] Project title `Suhas Adiga_IBM Z Datathon 2026 Project`, and "SSF2026" in the description
- [ ] Live demo running (`bash run_safepay.sh public`), port 8090 open, and http://148.100.99.226:8090 opens from your own browser; the login (user `judge` and the password) goes in the description
