# ChargeGuard: 3-minute video script

The rules ask for: the problem statement, supporting data, the solution overview, and how it uses IBM Z (tech stack, plus a demo of the LinuxONE instance and the model). Keep it **under 3:00**.

Before recording:
- On the VM: `bash run_chargeguard.sh demo` about 5 minutes ahead, so incidents and a Granite note already exist.
- On the laptop: `ssh -N -L 8095:localhost:8095 ibmz`, then open http://localhost:8095 and http://localhost:8095/evidence in two tabs.
- Keep a terminal open with `ssh ibmz` for the IBM Z shots.

---

**0:00–0:25 · The problem** (slide or evidence page header)

> "When an EV battery starts to go wrong, it often shows it while charging. China's emergency ministry counted 640 new-energy-vehicle fires in just the first quarter of 2022, and media tallies put about a quarter of them during charging. Chargers already have an over-temperature trip. On real data, that trip catches two percent of faulty-battery charges."

**0:25–0:45 · The data**

> "We used 18,531 real charging sessions from 30 public stations in Shenzhen, recorded every minute, released by Hong Kong PolyU. 747 came from batteries the authors labelled faulty. Those labels are faults, not fires, so ChargeGuard is an early warning, not a fire predictor."

**0:45–1:35 · Live demo on LinuxONE** (terminal, then the live page)

- Terminal: `lscpu | grep "Machine type"` shows `9175` (IBM z17). Then `curl -s localhost:8095/healthz` shows `engine: snapml`.
> "This is our LinuxONE VM, an IBM z17. ChargeGuard is the charging network's central system. These are simulated chargers speaking OCPP, the real charger protocol. They replay six stations the model never saw."
- Live page: point at a connector turning **watch → slow**.
> "Every reading is scored as it arrives. Here the gun temperature climbs faster than the current explains, so ChargeGuard caps the current through OCPP's SetChargingProfile. It only does that on chargers that support smart charging; this one doesn't, so it goes to an operator instead."
- Incident queue: show the **reasons** and the **Granite note**, then click **Approve stop** and show the charger answering `Accepted`.
> "A stop is never automatic. IBM Granite, running on the same machine, writes a note for station staff. A person approves, and only then does the system send RemoteStopTransaction."

**1:35–2:15 · Does it work?** (evidence page)

> "We locked six stations away before training. On those, ChargeGuard ranks faulty charges with an AUC of 0.96, and 0.95 from just the first five minutes. It recommends a stop for 89% of faulty-battery sessions; the temperature trip manages 2%. We found faults concentrated in big commercial packs, so our model ignores vehicle size: for typical cars it still scores 0.90.
> Honestly: at two stations whose temperature sensors read differently, false alarms are far too high. ChargeGuard flags those chargers for a hardware check, and that's why stops need a human."

**2:15–2:45 · Why IBM Z** (benchmark section, terminal)

> "Every charge is already a transaction: authorised, metered every minute, billed. So the screening belongs in that path. On the z17, IBM Snap ML scores a reading in 12 microseconds. The whole pipeline takes 94, which is about 5,500 readings a second on one core, enough for 330,000 connectors. Granite, the model and driver data all stay on this machine, and drivers are pseudonymised."

**2:45–3:00 · Close**

> "ChargeGuard: early warning of hazardous charging, inside the charging transaction, on IBM Z. Slow first, people decide stops, and every number is from stations the model never saw."

---

## Checklist

- [ ] Both mandatory labs done, with screenshots (needed for the technical score)
- [ ] Screenshot of LinuxONE Community Cloud "Manage Instances" showing the VM
- [ ] Video under 3:00, showing the LinuxONE terminal and the live model
- [ ] ChargeGuard in its **own** public GitHub repo (not inside SafePay's), README first
- [ ] Submission title: `[Team Name_IBM Z Datathon 2026 Project]`, track: Wildcard
- [ ] Confirm on Discord that a team may submit one project per track
- [ ] Submit on Hack2Skill before **Sun 18 Oct, 10:00 PM IST** (aim for 9 PM)
