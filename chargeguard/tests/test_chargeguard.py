"""ChargeGuard tests: feature parity, ladder parity, OCPP round trips, explanations, and the
service flow over a real OCPP WebSocket (no stop without operator approval).

    python -m pytest -q tests
"""
from __future__ import annotations

import importlib
import json
import math
import os

import joblib
import numpy as np
import pytest

from chargeguard.core import explain, ocpp, policy, privacy
from chargeguard.core.features import BEHAVIOUR_FEATURES, FEATURES, Reading, SessionFeatures


def _session(n=40, seed=0, hot=False):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = 30 + (1.5 * k if hot else 0.05 * k) + rng.normal(0, 0.3)
        out.append(Reading(voltage=350 + 0.4 * k + rng.normal(0, 1), current=90 + rng.normal(0, 3),
                           power=32 + rng.normal(0, 1), soc=40 + 0.6 * k, energy_kwh=0.55 * (k + 1),
                           temp1=t, temp2=t + 6))
    return out


# --- features ---------------------------------------------------------------------------

def test_streaming_features_match_batch():
    rs = _session()
    f = SessionFeatures(3)
    for r in rs:
        vec = f.update(r)
    v = np.array([r.voltage for r in rs])
    t1 = np.array([r.temp1 for r in rs])
    x = dict(zip(FEATURES, vec))
    assert x["minutes"] == len(rs)
    assert math.isclose(x["v_mean"], v.mean(), rel_tol=1e-9)
    assert math.isclose(x["v_std"], v.std(ddof=1), rel_tol=1e-9)
    assert math.isclose(x["v_slope"], np.polyfit(np.arange(len(v)), v, 1)[0], rel_tol=1e-6)
    assert math.isclose(x["t1_slope"], np.polyfit(np.arange(len(t1)), t1, 1)[0], rel_tol=1e-6)
    assert math.isclose(x["t1_d5"], t1[-1] - t1[-6], rel_tol=1e-9)
    assert math.isclose(x["kwh_per_soc"], rs[-1].energy_kwh / (rs[-1].soc - rs[0].soc), rel_tol=1e-9)


def test_undefined_values_early_in_session():
    f = SessionFeatures(None)
    x = dict(zip(FEATURES, f.update(_session(1)[0])))
    assert x["battery_type"] == -1 and x["kwh_per_soc"] == -1 and x["v_std"] == 0


def test_behaviour_features_leave_out_vehicle_size():
    assert "kwh_per_soc" not in BEHAVIOUR_FEATURES and "v_mean" not in BEHAVIOUR_FEATURES
    assert "t1_d5" in BEHAVIOUR_FEATURES


# --- policy -----------------------------------------------------------------------------

def test_live_ladder_matches_offline_evaluation():
    """The service escalates at exactly the minute the evaluation assumes."""
    rng = np.random.default_rng(1)
    th = policy.Thresholds(watch=0.3, slow=0.6, stop=0.85)
    for _ in range(200):
        p = np.clip(np.cumsum(rng.normal(0.02, 0.08, 60)), 0, 1)
        eff = policy.effective(p, th.persist, th.min_minutes)
        lad = policy.Ladder(th)
        first = {}
        for minute, x in enumerate(p, start=1):
            new = lad.step(minute, float(x))
            if new is not None:
                for lv in range(1, new + 1):
                    first.setdefault(lv, minute)
        for lv, key in ((1, "watch"), (2, "slow"), (3, "stop")):
            idx = policy.first_at_or_above(eff, getattr(th, key))
            assert first.get(lv, 0) == (idx + 1 if idx >= 0 else 0)


def test_one_noisy_reading_never_escalates():
    lad = policy.Ladder(policy.Thresholds(watch=0.3, slow=0.6, stop=0.85))
    assert [lad.step(m, p) for m, p in enumerate([0.1, 0.1, 0.1, 0.99, 0.1, 0.1], start=1)] == [None] * 6


def test_thresholds_are_ordered_and_meet_budget():
    rng = np.random.default_rng(2)
    m = rng.beta(1, 8, 5000)
    th = policy.choose(m)
    assert th.watch < th.slow < th.stop
    assert abs((m >= th.slow).mean() - policy.BUDGETS["slow"]) < 0.005


def test_station_check_flags_outlier():
    assert policy.station_check({1: 0.01, 2: 0.02, 20: 0.6}) == [20]
    # Live: a charger stands out only against the rest of the fleet.
    assert policy.station_outlier(0.45, 50, [0.05, 0.08, 0.06])
    assert not policy.station_outlier(0.45, 50, [0.30, 0.35, 0.28])  # the whole fleet is busy
    assert not policy.station_outlier(0.45, 10, [0.05])               # too few sessions yet
    assert not policy.station_outlier(0.08, 50, [0.01])               # under 5x the slow budget


# --- OCPP -------------------------------------------------------------------------------

def test_meter_values_round_trip():
    r = _session(3)[2]
    payload = ocpp.meter_values(2, 1001, "2021-05-01T10:00:00+08:00", r, register_wh=500_000 + 1650)
    ts, back, reg = ocpp.reading_from_meter_values(json.loads(json.dumps(payload)), meter_start_wh=500_000)
    assert ts.startswith("2021-05-01") and reg == 501_650
    assert math.isclose(back.energy_kwh, 1.65) and round(back.temp2) == round(r.temp2)
    assert abs(back.voltage - r.voltage) < 0.06 and abs(back.current - r.current) < 0.006


def test_feature_profiles_and_frames():
    conf = {"configurationKey": [{"key": "SupportedFeatureProfiles", "value": "Core, SmartCharging"}]}
    assert ocpp.feature_profiles(conf) == {"Core", "SmartCharging"}
    uid, text = ocpp.call("RemoteStopTransaction", ocpp.remote_stop(7))
    m = ocpp.parse(text)
    assert (m.kind, m.uid, m.action, m.payload) == (ocpp.CALL, uid, "RemoteStopTransaction", {"transactionId": 7})
    with pytest.raises(ValueError):
        ocpp.parse('{"not": "ocpp"}')


# --- explanations and privacy -----------------------------------------------------------

def test_reasons_quote_a_line_the_value_crosses():
    ref = {f: [-10.0, -5.0, 0.0, 5.0, 10.0] for f in FEATURES}  # p1, p5, p50, p95, p99
    vec = [0.0] * len(FEATURES)
    vec[FEATURES.index("t1_d5")] = 7.0         # past p95 only
    vec[FEATURES.index("v_std")] = 20.0        # past p99
    vec[FEATURES.index("i_last_ratio")] = -7.0  # below p5 only: an unusually low value
    rs = explain.reasons(vec, ref)
    assert rs[0].startswith("pack voltage is unsteady (spread 20.0 V)") and "99% of normal charges stay under 10.0 V" in rs[0]
    assert any("rose 7 °C" in r and "95% of normal charges stay under 5 °C" in r for r in rs)
    assert any(r.startswith("current has fallen to") and "95% of normal charges stay above" in r for r in rs)
    assert explain.reasons(vec, ref, allowed={"soc0"}) == [explain.FALLBACK]


def test_results_json_has_no_nan():
    from chargeguard.core.jsonsafe import clean
    import numpy as np
    out = clean({"a": float("nan"), "b": [np.float32(1.5), np.int64(3), float("inf")], "c": "x"})
    assert out == {"a": None, "b": [1.5, 3, None], "c": "x"}
    json.dumps(out, allow_nan=False)


def test_pseudonyms_are_stable_and_opaque(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARGEGUARD_KEY_FILE", str(tmp_path / "key"))
    monkeypatch.setattr(privacy, "_KEY", None)
    a, b = privacy.token("card-123"), privacy.token("card-123")
    assert a == b and a.startswith("drv_") and "123" not in a and privacy.token("card-124") != a


def test_granite_writer_skips_backlog_for_newest(monkeypatch):
    """Granite can take minutes per note on a small VM: only the newest waiting incident gets one."""
    import threading
    from chargeguard.core import granite
    started, release, done = threading.Event(), threading.Event(), []
    monkeypatch.setattr(granite, "available", lambda: True)

    def slow_write(facts, timeout=0):
        started.set()
        release.wait(5)
        return f"note {facts['n']}"
    monkeypatch.setattr(granite, "write", slow_write)
    w = granite.Writer(lambda i, note: done.append((i, note)))
    w.submit(1, {"n": 1})
    assert started.wait(5)
    w.submit(2, {"n": 2})
    w.submit(3, {"n": 3})  # replaces 2 while 1 is still being written
    release.set()
    for _ in range(100):
        if len(done) == 2:
            break
        threading.Event().wait(0.05)
    assert done == [(1, "note 1"), (3, "note 3")] and w.skipped == 1


# --- service over a real OCPP WebSocket ---------------------------------------------------

class HeatModel:
    """Stand-in model: risk rises with the gun temperature climb over 5 minutes."""
    def predict_proba(self, X):
        j = BEHAVIOUR_FEATURES.index("t1_d5")
        p = 1 / (1 + np.exp(-(X[:, j] - 2.0)))
        return np.c_[1 - p, p]


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    ref = {f: [-2.0, -1.0, 0.0, 1.0, 2.0] for f in FEATURES}
    joblib.dump({"model": HeatModel(), "engine": "test", "features": BEHAVIOUR_FEATURES,
                 "thresholds": policy.Thresholds(watch=0.5, slow=0.8, stop=0.95).to_dict(),
                 "test_stations": [1], "reference": ref, "flagged_stations": [],
                 "normal_max_scores": np.linspace(0, 0.6, 100).astype(np.float32)}, tmp_path / "risk.joblib")
    monkeypatch.setenv("CHARGEGUARD_ARTIFACT", str(tmp_path / "risk.joblib"))
    monkeypatch.setenv("CHARGEGUARD_DB", ":memory:")
    monkeypatch.setenv("CHARGEGUARD_NOTES", "0")
    monkeypatch.setenv("CHARGEGUARD_KEY_FILE", str(tmp_path / "key"))
    from fastapi.testclient import TestClient
    import chargeguard.app.main as main
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def _cp_call(ws, action, payload, smart=True):
    """Send a CALL as the charge point, answering any CALLs the central system sends first."""
    uid, text = ocpp.call(action, payload)
    ws.send_text(text)
    commands = []
    while True:
        m = ocpp.parse(ws.receive_text())
        if m.kind == ocpp.CALL:
            commands.append(m)
            if m.action == "GetConfiguration":
                val = "Core,SmartCharging" if smart else "Core"
                ans = {"configurationKey": [{"key": "SupportedFeatureProfiles", "value": val}]}
            else:
                ans = {"status": "Accepted"}
            ws.send_text(ocpp.result(m.uid, ans))
            continue
        assert m.uid == uid
        return m.payload, commands


@pytest.mark.parametrize("smart", [True, False])
def test_ocpp_flow_slows_then_waits_for_operator(app_client, smart):
    with app_client.websocket_connect("/ocpp/SZ-01", subprotocols=["ocpp1.6"]) as ws:
        _cp_call(ws, "BootNotification", {"chargePointVendor": "t", "chargePointModel": "t"}, smart)
        _cp_call(ws, "DataTransfer", {"vendorId": ocpp.GBT_VENDOR, "messageId": "BRM",
                                      "data": json.dumps({"connectorId": 1, "batteryType": 6})}, smart)
        res, _ = _cp_call(ws, "StartTransaction", {"connectorId": 1, "idTag": "card-9", "meterStart": 1000,
                                                    "timestamp": "2021-05-01T10:00:00+08:00"}, smart)
        tx = res["transactionId"]
        seen = []
        for k, r in enumerate(_session(25, hot=True)):
            reg = 1000 + round(r.energy_kwh * 1000)
            _, cmds = _cp_call(ws, "MeterValues",
                               ocpp.meter_values(1, tx, f"2021-05-01T10:{k:02d}:00+08:00", r, reg), smart)
            seen += [c.action for c in cmds]
        state = app_client.get("/api/state").json()
        inc = [i for i in state["incidents"] if i["tx"] == tx]
        assert inc and inc[0]["status"] == "awaiting_approval"
        assert "RemoteStopTransaction" not in seen  # never without approval
        assert ("SetChargingProfile" in seen) == smart
        conn = state["chargers"][0]["connectors"][0]
        assert conn["battery"] == "NMC" and conn["driver"].startswith("drv_") and conn["level"] == 3
        # Operator approves: the central system sends RemoteStopTransaction to the charger.
        import threading
        out = {}
        t = threading.Thread(target=lambda: out.update(app_client.post(
            f"/api/incidents/{inc[0]['id']}/approve", json={"operator": "tester"}).json()))
        t.start()
        m = ocpp.parse(ws.receive_text())
        assert m.action == "RemoteStopTransaction" and m.payload == {"transactionId": tx}
        ws.send_text(ocpp.result(m.uid, {"status": "Accepted"}))
        t.join(10)
        assert out["stop_sent"] and out["charger_answer"] == "Accepted"
        _cp_call(ws, "StopTransaction", {"transactionId": tx, "meterStop": 9000,
                                         "timestamp": "2021-05-01T10:30:00+08:00", "reason": "Remote"}, smart)
    audit = app_client.get("/api/audit").json()
    assert any(a["action"] == "approve_stop" and a["actor"] == "tester" for a in audit)
