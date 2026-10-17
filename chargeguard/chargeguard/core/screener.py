"""The central-system side of ChargeGuard: one live state per charging session.

Every MeterValues reading goes through the same path:
features (O(1) update) -> risk model -> decision ladder -> action -> event for the dashboard.

Actions, by level:
  watch             nothing is sent to the charger; the session is highlighted.
  slow              SetChargingProfile caps the current at SLOW_FRACTION of its peak, if the
                    charger supports SmartCharging; otherwise the operator is alerted instead.
  stop_recommended  an incident opens with the reasons and a note; the charge continues until
                    an operator approves, and only then is RemoteStopTransaction sent.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from chargeguard.core import explain, granite, model as M, privacy
from chargeguard.core.features import BATTERY_TYPES, FEATURES, Reading, SessionFeatures
from chargeguard.core.policy import LEVELS, STATION_MIN_SESSIONS, STATION_WINDOW, Ladder, Thresholds, station_outlier
from chargeguard.core.store import Store, now
from chargeguard.core.tariff import Tariff

SLOW_FRACTION = 0.5
MIN_LIMIT_A = 10.0


@dataclass
class Live:
    cp: str
    connector: int
    tx: int
    driver: str
    battery: str
    smart: bool
    meter_start: float
    feats: SessionFeatures
    ladder: Ladder
    kwh: float = 0.0
    cost: float = 0.0
    max_risk: float = 0.0
    incident: int | None = None
    limit_amps: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=240))


@dataclass
class Decision:
    tx: int
    minute: int
    risk: float
    percentile: float
    level: int
    escalated: bool
    action: str = "none"           # none | set_charging_profile | escalate_operator | await_approval
    limit_amps: float | None = None
    reasons: list = field(default_factory=list)
    incident: int | None = None


class Screener:
    def __init__(self, artifact: str | Path, store: Store, tariff: Tariff, emit=None, notes: bool = True):
        a = joblib.load(artifact)
        self.model = a["model"]
        self.engine = a["engine"]
        self.th = Thresholds(**a["thresholds"])
        self.reference = a["reference"]
        self.normal = np.asarray(a["normal_max_scores"], dtype=float)
        self.flagged_training = set(a.get("flagged_stations", []))
        self.features = list(a["features"])
        self.cols = [FEATURES.index(f) for f in self.features]
        ref = np.array([[self.reference[f][2] for f in self.features]] * 32, dtype=np.float32)  # medians
        self.tree_format = M.optimise(self.model, ref)
        self.store, self.tariff = store, tariff
        self.emit = emit or (lambda e: None)
        self.live: dict[int, Live] = {}
        self.pending_battery: dict[tuple[str, int], int] = {}
        self.station_recent: dict[str, deque] = {}
        self.station_flagged: set[str] = set()
        self.latency_us: deque = deque(maxlen=5000)
        self.readings = 0
        self.writer = granite.Writer(self._note_done) if notes else None

    # --- session lifecycle ----------------------------------------------------------------

    def battery_reported(self, cp: str, connector: int, code: int) -> None:
        self.pending_battery[(cp, connector)] = code

    def start(self, cp: str, connector: int, tx: int, id_tag: str, ts: str, meter_start_wh: float,
              smart: bool) -> None:
        code = self.pending_battery.pop((cp, connector), None)
        battery = BATTERY_TYPES.get(code, "unknown") if code is not None else "unknown"
        s = Live(cp, connector, tx, privacy.token(id_tag), battery, smart, meter_start_wh,
                 SessionFeatures(code), Ladder(self.th))
        self.live[tx] = s
        self.store.start(tx, cp, connector, s.driver, battery, smart, ts)
        self.emit({"type": "session_start", "cp": cp, "connector": connector, "tx": tx, "driver": s.driver,
                   "battery": battery, "smart_charging": smart, "ts": ts})

    def reading(self, tx: int, ts: str, r: Reading) -> Decision | None:
        s = self.live.get(tx)
        if s is None:
            return None
        hour = _hour(ts)
        s.cost += self.tariff.cost(max(r.energy_kwh - s.kwh, 0.0), hour)
        s.kwh = r.energy_kwh
        vec = s.feats.update(r)
        x = np.asarray([[vec[i] for i in self.cols]], dtype=np.float32)
        t0 = time.perf_counter()
        risk = float(M.proba(self.model, x)[0])
        self.latency_us.append((time.perf_counter() - t0) * 1e6)
        self.readings += 1
        minute = s.feats.minutes
        new_level = s.ladder.step(minute, risk)
        eff = min(s.ladder.recent) if len(s.ladder.recent) == s.ladder.th.persist else risk
        s.max_risk = max(s.max_risk, eff)
        pct = float(np.searchsorted(self.normal, eff, side="right") / max(len(self.normal), 1))
        d = Decision(tx, minute, risk, pct, s.ladder.level, new_level is not None)
        if d.escalated or s.ladder.level >= 1:
            d.reasons = explain.reasons(vec, self.reference, set(self.features))
        if d.escalated:
            self._act(s, d, r, ts)
        point = {"minute": minute, "ts": ts, "risk": round(risk, 5), "percentile": round(pct, 4),
                 "level": s.ladder.level, "v": r.voltage, "i": r.current, "p": r.power, "soc": r.soc,
                 "t1": r.temp1, "t2": r.temp2, "kwh": round(s.kwh, 2), "cost": round(s.cost, 2)}
        s.history.append(point)
        if minute % 5 == 0 or d.escalated:
            self.store.progress(tx, minute, s.kwh, s.cost, s.ladder.level, s.max_risk)
        self.emit({"type": "reading", "cp": s.cp, "connector": s.connector, "tx": tx, **point,
                   "reasons": d.reasons, "limit_amps": s.limit_amps})
        return d

    def end(self, tx: int, ts: str, reason: str) -> None:
        s = self.live.pop(tx, None)
        if s is None:
            return
        outcome = {0: "normal", 1: "watched", 2: "slowed", 3: "stop_recommended"}[s.ladder.level]
        if reason == "Remote":
            outcome = "stopped_by_operator"
        self.store.progress(tx, s.feats.minutes, s.kwh, s.cost, s.ladder.level, s.max_risk)
        self.store.end(tx, ts, outcome)
        self._station_update(s)
        self.emit({"type": "session_end", "cp": s.cp, "connector": s.connector, "tx": tx, "outcome": outcome,
                   "minutes": s.feats.minutes, "kwh": round(s.kwh, 2), "cost": round(s.cost, 2), "ts": ts})

    # --- actions --------------------------------------------------------------------------

    def _act(self, s: Live, d: Decision, r: Reading, ts: str) -> None:
        level = LEVELS[d.level]
        if level == "slow" or (level == "stop_recommended" and s.limit_amps is None):
            if s.smart:
                s.limit_amps = max(MIN_LIMIT_A, SLOW_FRACTION * s.feats.i.max)
                d.action, d.limit_amps = "set_charging_profile", s.limit_amps
            else:
                d.action = "escalate_operator"
        if level == "stop_recommended":
            facts = {"cp": s.cp, "connector": s.connector, "battery": s.battery, "minute": d.minute,
                     "percentile": d.percentile, "risk_score": round(d.risk, 4), "reasons": d.reasons,
                     "action": d.action, "current_limit_amps": s.limit_amps,
                     "gun_temperatures_c": f"{r.temp1:.0f} and {r.temp2:.0f}", "energy_kwh": round(s.kwh, 2)}
            note = granite.template(facts)
            s.incident = d.incident = self.store.open_incident(s.tx, s.cp, s.connector, d.risk, d.reasons, note,
                                                               "template", s.battery, d.minute, d.percentile)
            d.action = d.action if d.action != "none" else "await_approval"
            if self.writer:
                self.writer.submit(s.incident, facts)
            self.emit({"type": "incident", "id": s.incident, "status": "awaiting_approval", "cp": s.cp,
                       "connector": s.connector, "tx": s.tx, "battery": s.battery, "minute": d.minute,
                       "percentile": d.percentile, "reasons": d.reasons, "note": note, "note_source": "template",
                       "ts": ts})
        self.store.decision(s.tx, ts, d.minute, d.level, d.risk, d.percentile, d.reasons, d.action)
        self.emit({"type": "decision", "cp": s.cp, "connector": s.connector, "tx": s.tx, "level": d.level,
                   "level_name": level, "action": d.action, "limit_amps": d.limit_amps, "minute": d.minute,
                   "percentile": d.percentile, "reasons": d.reasons, "incident": d.incident, "ts": ts})

    def _note_done(self, incident_id: int, note: str) -> None:
        self.store.set_note(incident_id, note, "granite")
        self.emit({"type": "incident_note", "id": incident_id, "note": note, "note_source": "granite"})

    def approve(self, incident_id: int, operator: str) -> int | None:
        """Operator approves a stop. Returns the transaction to stop (if still charging)."""
        inc = self.store.incident(incident_id)
        if not inc or inc["status"] != "awaiting_approval":
            return None
        self.store.decide(incident_id, "approved", operator)
        self.store.audit(operator, "approve_stop", {"incident": incident_id, "tx": inc["tx"]})
        self.emit({"type": "incident", **self.store.incident(incident_id)})
        return inc["tx"] if inc["tx"] in self.live else None

    def dismiss(self, incident_id: int, operator: str) -> bool:
        inc = self.store.incident(incident_id)
        if not inc or inc["status"] != "awaiting_approval":
            return False
        self.store.decide(incident_id, "dismissed", operator)
        self.store.audit(operator, "dismiss_stop", {"incident": incident_id, "tx": inc["tx"]})
        self.emit({"type": "incident", **self.store.incident(incident_id)})
        return True

    # --- station check --------------------------------------------------------------------

    def _station_update(self, s: Live) -> None:
        q = self.station_recent.setdefault(s.cp, deque(maxlen=STATION_WINDOW))
        q.append(s.ladder.level >= 2)
        rate = sum(q) / len(q)
        others = [sum(o) / len(o) for cp, o in self.station_recent.items()
                  if cp != s.cp and len(o) >= STATION_MIN_SESSIONS]
        if s.cp not in self.station_flagged and station_outlier(rate, len(q), others):
            self.station_flagged.add(s.cp)
            fleet = round(float(np.median(others)), 3) if others else None
            self.store.audit("chargeguard", "station_flag", {"cp": s.cp, "slow_rate": rate, "fleet_median": fleet})
            self.emit({"type": "station_flag", "cp": s.cp, "slow_rate": round(rate, 3), "fleet_median": fleet,
                       "message": "This charger slows far more sessions than the rest of the fleet. Check its "
                                  "sensors before trusting its alerts."})

    def stats(self) -> dict:
        lat = np.array(self.latency_us) if self.latency_us else np.zeros(1)
        return {"readings": self.readings, "live_sessions": len(self.live), "engine": self.engine,
                "tree_format": self.tree_format, "latency_us_p50": round(float(np.percentile(lat, 50)), 1),
                "latency_us_p99": round(float(np.percentile(lat, 99)), 1),
                "flagged_stations": sorted(self.station_flagged)}


def _hour(ts: str) -> int:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except ValueError:
        return 12
