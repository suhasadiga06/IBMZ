"""ChargeGuard central system: OCPP 1.6-J endpoint for charge points, plus the operator dashboard.

    uvicorn chargeguard.app.main:app --host 127.0.0.1 --port 8095

Charge points connect to  ws://<host>/ocpp/<charge point id>  (subprotocol "ocpp1.6").
Operator API:
  GET  /api/state                     snapshot of chargers, live sessions, incidents, totals
  GET  /api/events                    server-sent events (see EVENTS below)
  POST /api/incidents/{id}/approve    {"operator": "..."} -> RemoteStopTransaction is sent
  POST /api/incidents/{id}/dismiss    {"operator": "..."}
  GET  /api/metrics | /api/examples | /api/benchmark | /api/audit
  GET  /healthz, /metrics (Prometheus text)
Pages: /  live operations,  /evidence  evaluation results.

EVENTS (JSON in SSE "data:"), each with a "type":
  charger         {cp, connected, smart_charging, profiles}
  session_start   {cp, connector, tx, driver, battery, smart_charging, ts}
  reading         {cp, connector, tx, minute, ts, risk, percentile, level, v, i, p, soc, t1, t2, kwh, cost,
                   reasons, limit_amps}
  decision        {cp, connector, tx, level, level_name, action, limit_amps, minute, percentile, reasons,
                   incident, ts}
  incident        {id, status, cp, connector, tx, ... , note, note_source}
  incident_note   {id, note, note_source}
  command         {cp, action, tx, status}        an OCPP command sent to a charger and its answer
  session_end     {cp, connector, tx, outcome, minutes, kwh, cost, ts}
  station_flag    {cp, slow_rate, message}
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from chargeguard.core import ocpp
from chargeguard.core.screener import Screener
from chargeguard.core.store import Store, now
from chargeguard.core.tariff import Tariff
from chargeguard.core.jsonsafe import clean

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
ARTIFACT = Path(os.environ.get("CHARGEGUARD_ARTIFACT", ROOT / "artifacts" / "risk.joblib"))
DB = os.environ.get("CHARGEGUARD_DB", str(ROOT / "data" / "chargeguard.db"))
NOTES = os.environ.get("CHARGEGUARD_NOTES", "1") == "1"
COMMAND_TIMEOUT = 15


class ChargePoint:
    def __init__(self, cp_id: str, ws: WebSocket):
        self.id, self.ws = cp_id, ws
        self.profiles: set[str] = set()
        self.pending: dict[str, asyncio.Future] = {}
        self.connectors: dict[int, int] = {}  # connector -> live transaction

    @property
    def smart(self) -> bool:
        return "SmartCharging" in self.profiles

    async def send_call(self, action: str, payload: dict) -> dict:
        uid, text = ocpp.call(action, payload)
        fut = asyncio.get_running_loop().create_future()
        self.pending[uid] = fut
        await self.ws.send_text(text)
        try:
            return await asyncio.wait_for(fut, COMMAND_TIMEOUT)
        finally:
            self.pending.pop(uid, None)


class Hub:
    """Fans events out to dashboard subscribers; safe to call from any thread."""

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, event: dict) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict) -> None:
        for q in list(self.subs):
            if q.qsize() < 2000:
                q.put_nowait(event)


hub = Hub()
state: dict = {}
tx_ids = itertools.count(int(os.environ.get("CHARGEGUARD_FIRST_TX", "1001")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    store = Store(DB)
    state["store"] = store
    state["screener"] = Screener(ARTIFACT, store, Tariff(ROOT / "artifacts" / "tariff.json"), hub.emit, notes=NOTES)
    state["cps"] = {}
    yield


app = FastAPI(title="ChargeGuard", lifespan=lifespan)


# --- OCPP --------------------------------------------------------------------------------

@app.websocket("/ocpp/{cp_id}")
async def ocpp_endpoint(ws: WebSocket, cp_id: str):
    await ws.accept(subprotocol=ocpp.SUBPROTOCOL)
    cp = ChargePoint(cp_id, ws)
    state["cps"][cp_id] = cp
    asyncio.create_task(_discover(cp))
    scr: Screener = state["screener"]
    try:
        while True:
            msg = ocpp.parse(await ws.receive_text())
            if msg.kind in (ocpp.CALLRESULT, ocpp.CALLERROR):
                fut = cp.pending.get(msg.uid)
                if fut and not fut.done():
                    fut.set_result(msg.payload if msg.kind == ocpp.CALLRESULT else {"error": msg.action})
                continue
            await ws.send_text(_handle(cp, scr, msg))
    except WebSocketDisconnect:
        pass
    finally:
        for tx in list(cp.connectors.values()):
            scr.end(tx, now(), "EVDisconnected")
        state["cps"].pop(cp_id, None)
        hub.emit({"type": "charger", "cp": cp_id, "connected": False})


async def _discover(cp: ChargePoint) -> None:
    try:
        res = await cp.send_call("GetConfiguration", ocpp.get_configuration())
        cp.profiles = ocpp.feature_profiles(res)
    except asyncio.TimeoutError:
        cp.profiles = set()
    hub.emit({"type": "charger", "cp": cp.id, "connected": True, "smart_charging": cp.smart,
              "profiles": sorted(cp.profiles)})


def _handle(cp: ChargePoint, scr: Screener, msg: ocpp.Message) -> str:
    a, p = msg.action, msg.payload
    if a == "BootNotification":
        return ocpp.result(msg.uid, {"status": "Accepted", "currentTime": now(), "interval": 300})
    if a == "Heartbeat":
        return ocpp.result(msg.uid, {"currentTime": now()})
    if a in ("StatusNotification", "Authorize"):
        return ocpp.result(msg.uid, {} if a == "StatusNotification" else {"idTagInfo": {"status": "Accepted"}})
    if a == "DataTransfer":
        got = ocpp.battery_type_from_data_transfer(p)
        if got:
            scr.battery_reported(cp.id, *got)
        return ocpp.result(msg.uid, {"status": "Accepted" if got else "UnknownVendorId"})
    if a == "StartTransaction":
        tx = next(tx_ids)
        cp.connectors[int(p["connectorId"])] = tx
        scr.start(cp.id, int(p["connectorId"]), tx, p["idTag"], p["timestamp"], float(p["meterStart"]), cp.smart)
        return ocpp.result(msg.uid, {"transactionId": tx, "idTagInfo": {"status": "Accepted"}})
    if a == "MeterValues":
        tx = int(p["transactionId"])
        live = scr.live.get(tx)
        if live is not None:
            ts, reading, _ = ocpp.reading_from_meter_values(p, live.meter_start)
            d = scr.reading(tx, ts, reading)
            if d and d.action == "set_charging_profile":
                asyncio.create_task(_command(cp, "SetChargingProfile",
                                             ocpp.set_charging_profile(live.connector, tx, d.limit_amps), tx))
        return ocpp.result(msg.uid, {})
    if a == "StopTransaction":
        tx = int(p["transactionId"])
        for c, t in list(cp.connectors.items()):
            if t == tx:
                cp.connectors.pop(c)
        scr.end(tx, p.get("timestamp", now()), p.get("reason", "Local"))
        return ocpp.result(msg.uid, {"idTagInfo": {"status": "Accepted"}})
    return ocpp.error(msg.uid, "NotImplemented", f"{a} is not handled by ChargeGuard")


async def _command(cp: ChargePoint, action: str, payload: dict, tx: int) -> dict:
    try:
        res = await cp.send_call(action, payload)
        status = res.get("status", "error")
    except asyncio.TimeoutError:
        res, status = {}, "timeout"
    state["store"].audit("chargeguard", action, {"cp": cp.id, "tx": tx, "status": status})
    hub.emit({"type": "command", "cp": cp.id, "action": action, "tx": tx, "status": status})
    return res


# --- operator API ------------------------------------------------------------------------

class OperatorAction(BaseModel):
    operator: str = "operator"


@app.post("/api/incidents/{incident_id}/approve")
async def approve(incident_id: int, body: OperatorAction):
    scr: Screener = state["screener"]
    inc = state["store"].incident(incident_id)
    if inc is None:
        raise HTTPException(404, "no such incident")
    tx = scr.approve(incident_id, body.operator)
    if tx is None:
        return {"status": state["store"].incident(incident_id)["status"], "stop_sent": False}
    cp = state["cps"].get(inc["cp"])
    if cp is None:
        return {"status": "approved", "stop_sent": False, "reason": "charger not connected"}
    res = await _command(cp, "RemoteStopTransaction", ocpp.remote_stop(tx), tx)
    return {"status": "approved", "stop_sent": True, "charger_answer": res.get("status", "timeout")}


@app.post("/api/incidents/{incident_id}/dismiss")
async def dismiss(incident_id: int, body: OperatorAction):
    if not state["screener"].dismiss(incident_id, body.operator):
        raise HTTPException(409, "incident is not awaiting a decision")
    return {"status": "dismissed"}


@app.get("/api/state")
async def snapshot():
    scr: Screener = state["screener"]
    chargers = []
    for cp in state["cps"].values():
        conns = []
        for c, tx in sorted(cp.connectors.items()):
            s = scr.live.get(tx)
            if s:
                conns.append({"connector": c, "tx": tx, "driver": s.driver, "battery": s.battery,
                              "level": s.ladder.level, "limit_amps": s.limit_amps, "incident": s.incident,
                              "history": list(s.history)[-60:]})
        chargers.append({"cp": cp.id, "smart_charging": cp.smart, "profiles": sorted(cp.profiles),
                         "flagged": cp.id in scr.station_flagged, "connectors": conns})
    return {"chargers": chargers, "incidents": state["store"].incidents(30), "summary": state["store"].summary(),
            "stats": scr.stats(), "thresholds": scr.th.to_dict()}


@app.get("/api/events")
async def events():
    q: asyncio.Queue = asyncio.Queue()
    hub.subs.add(q)

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    e = await asyncio.wait_for(q.get(), 15)
                    yield f"data: {json.dumps(e)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.subs.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _json_file(name: str):
    path = ROOT / "results" / name
    if not path.exists():
        raise HTTPException(404, f"{name} not generated yet")
    return JSONResponse(clean(json.loads(path.read_text())))  # older files may hold NaN


@app.get("/api/metrics")
async def metrics_json():
    return _json_file("metrics.json")


@app.get("/api/examples")
async def examples_json():
    return _json_file("examples.json")


@app.get("/api/benchmark")
async def benchmark_json():
    return _json_file("benchmark.json")


@app.get("/api/vehicle")
async def vehicle_json():
    return _json_file("vehicle_metrics.json")


@app.get("/api/audit")
async def audit():
    return state["store"].audit_log(100)


@app.get("/healthz")
async def healthz():
    return {"ok": True, **state["screener"].stats()}


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus():
    s = state["screener"].stats()
    return (f"chargeguard_readings_total {s['readings']}\nchargeguard_live_sessions {s['live_sessions']}\n"
            f"chargeguard_score_latency_us{{quantile=\"0.5\"}} {s['latency_us_p50']}\n"
            f"chargeguard_score_latency_us{{quantile=\"0.99\"}} {s['latency_us_p99']}\n")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/evidence")
async def evidence():
    return FileResponse(STATIC / "evidence.html")
