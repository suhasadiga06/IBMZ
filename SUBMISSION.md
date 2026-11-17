# Submission pack (deadline Sun 18 Oct 2026, 10:00 PM IST)

## Fields to fill on Hack2Skill

**Project title**
```
Suhas Adiga_IBM Z Datathon 2026 Project
```

**Repository**
```
https://github.com/suhasadiga06/IBMZ
```

**Live demo / how to run** (if there is a field for it)
```
No hosted demo. Run it locally in about 30 minutes: https://github.com/suhasadiga06/IBMZ/blob/main/JUDGES.md
```

**Files to upload**
- `pitch/SafePay_pitch_deck.pdf` (pitch deck, required)
- `pitch/SafePay_architecture.png` (architecture diagram, if there is an image field)
- Demo video, 3 minutes or less (record with `pitch/SafePay_recording_script.pdf`)
- Lab screenshots from `lab_screenshots/`, plus the "Manage Instances" screenshot, if asked

## Description (paste as-is)

```
SSF2026 · Theme 1: Real-Time AI for Critical Decisions (with Theme 2: sensitive data stays on-platform)

SafePay stops abuse hidden in payment messages, in real time, on IBM Z.

When victims block an abuser everywhere else, some abusers switch to the one channel that can't be blocked: one-cent bank transfers with threats in the payment reference. Commonwealth Bank finds about 1,500 high-risk cases a year, but its model scores relationships once a month.

SafePay is a screening service in the payment flow. It reads real ISO 20022 messages (pain.001 / pacs.008) and returns a decision in about a millisecond:
- A message model (harm + threat) that undoes evasion tricks like k.i.l.l, d3ad, look-alike letters and emoji, and joins threats split across several transfers.
- A relationship model on IBM Snap ML (23 features updated per payment) that catches polite-but-menacing coercive control no text model can see.
- Money is never blocked; only the message is withheld, and the sender is never told.
- IBM Granite 3.3 2B runs locally on the VM to write a brief for the specialist team and give a second opinion.
- The victim sets safe-contact rules and can consent to an Ed25519-signed, hash-chained evidence pack for police or a court.

Results (synthetic test world, real message text):
- Median time to a case: 3.3 days vs 19.3 days for a CBA-style monthly model; 92% of abusive messages withheld as they arrive.
- 94% of abusive relationships caught with 0 false alarms in 6,390 innocent ones; 0.49 false alarms per 10,000 on 305,163 innocent relationships.
- Outside check against Commonwealth Bank's open-source production model: PR AUC 0.993 vs 0.860.

Why IBM Z (measured on a z17): 933 payments/s end to end, median 0.27 ms; Snap ML scoring in 11 µs; Granite on the CPU. Accounts are HMAC-pseudonymised, messages AES-256-GCM encrypted, and no data leaves the platform: no external AI API.

Run it: https://github.com/suhasadiga06/IBMZ/blob/main/JUDGES.md
Code: https://github.com/suhasadiga06/IBMZ
```

## Before you submit

1. Record the video (3 min max) following `pitch/SafePay_recording_script.pdf`. Don't show the .pem key, the lab token or your email.
2. Upload the video (YouTube unlisted or Google Drive set to "anyone with the link") and paste the link.
