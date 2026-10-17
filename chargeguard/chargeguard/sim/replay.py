"""Simulated charge points that replay recorded Shenzhen charging sessions over OCPP 1.6-J.

    python -m chargeguard.sim.replay --url ws://127.0.0.1:8095/ocpp --minute-seconds 0.5 --fault-share 0.25

- Only the stations locked away from training are replayed (artifact "test_stations"), so
  everything the dashboard shows is out-of-sample.
- Each station is one charge point with CONNECTORS connectors. One charge point (--no-smart)
  does not offer the SmartCharging profile, to show the operator-escalation path.
- --fault-share sets the demo mix of faulty sessions (the real rate is about 4.6%); 0 replays
  sessions in recorded order. The dashboard shows the mix in use.
- Readings are the recorded ones. When ChargeGuard caps the current, the simulator accepts the
  profile but keeps replaying the recording: it cannot re-simulate the battery's physics.
- Round-trip time of every MeterValues -> answer is recorded (results/replay_latency.json).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import websockets

from chargeguard.core import ocpp
from chargeguard.core.features import COLS, Reading

ROOT = Path(__file__).resolve().parents[2]
CONNECTORS = 4


def load_sessions(data: Path, stations: list[int]) -> dict[int, list[pd.DataFrame]]:
    df = pd.read_csv(data, dtype={"transaction_id": str})
    df = df[df["class_judge"].isin(stations)]
    df["end_time"] = pd.to_datetime(df["end_time"])
    df = df.sort_values(["transaction_id", "end_time", "id"], kind="stable")
    out: dict[int, list] = {s: [] for s in stations}
    for _, g in df.groupby("transaction_id", sort=False):
        out[int(g["class_judge"].iloc[0])].append(g)
    for s in out:
        out[s].sort(key=lambda g: g["end_time"].iloc[0])
    return out


def demo_order(sessions: list[pd.DataFrame], share: float, rng: random.Random) -> list[pd.DataFrame]:
    if share <= 0:
        return sessions
    faulty = [g for g in sessions if g["label"].iloc[0] == 1]
    normal = [g for g in sessions if g["label"].iloc[0] == 0]
    rng.shuffle(faulty)
    rng.shuffle(normal)
    out = []
    while faulty or normal:
        pick = faulty if faulty and (not normal or rng.random() < share) else normal
        out.append(pick.pop())
    return out


class SimChargePoint:
    def __init__(self, cp_id: str, smart: bool, sessions: list[pd.DataFrame], minute_s: float, max_sessions: int):
        self.id, self.smart, self.minute_s, self.max_sessions = cp_id, smart, minute_s, max_sessions
        self.queue = list(reversed(sessions))
        self.pending: dict[str, asyncio.Future] = {}
        self.stop_requested: set[int] = set()
        self.rtt_ms: list[float] = []
        self.done = 0

    async def call(self, ws, action: str, payload: dict) -> dict:
        uid, text = ocpp.call(action, payload)
        fut = asyncio.get_running_loop().create_future()
        self.pending[uid] = fut
        await ws.send(text)
        return await asyncio.wait_for(fut, 30)

    async def receiver(self, ws) -> None:
        async for text in ws:
            m = ocpp.parse(text)
            if m.kind in (ocpp.CALLRESULT, ocpp.CALLERROR):
                fut = self.pending.pop(m.uid, None)
                if fut and not fut.done():
                    fut.set_result(m.payload)
                continue
            if m.action == "GetConfiguration":
                profiles = "Core,RemoteTrigger" + (",SmartCharging" if self.smart else "")
                ans = {"configurationKey": [{"key": "SupportedFeatureProfiles", "readonly": True, "value": profiles}]}
            elif m.action == "SetChargingProfile":
                ans = {"status": "Accepted" if self.smart else "NotSupported"}
            elif m.action == "RemoteStopTransaction":
                self.stop_requested.add(int(m.payload["transactionId"]))
                ans = {"status": "Accepted"}
            else:
                await ws.send(ocpp.error(m.uid, "NotImplemented", m.action))
                continue
            await ws.send(ocpp.result(m.uid, ans))

    async def connector(self, ws, connector: int) -> None:
        while self.queue and self.done < self.max_sessions:
            g = self.queue.pop()
            self.done += 1
            await self.charge(ws, connector, g)
            await asyncio.sleep(self.minute_s * 2)

    async def charge(self, ws, connector: int, g: pd.DataFrame) -> None:
        first = g.iloc[0]
        meter_start = round((float(first["current_energy_meter_value"]) - float(first["total_charging_kwh"])) * 1000)
        ts = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S+08:00")
        await self.call(ws, "DataTransfer", {"vendorId": ocpp.GBT_VENDOR, "messageId": "BRM",
                                             "data": json.dumps({"connectorId": connector,
                                                                 "batteryType": int(first["types"])})})
        res = await self.call(ws, "StartTransaction", {"connectorId": connector, "idTag": f"card-{first['transaction_id'][-10:]}",
                                                       "meterStart": meter_start, "timestamp": ts(first["begin_time"])})
        tx = int(res["transactionId"])
        reason, last_t, register = "EVDisconnected", first["end_time"], meter_start
        for row in g.itertuples(index=False):
            if tx in self.stop_requested:
                reason = "Remote"
                break
            r = Reading(**{k: float(getattr(row, c)) for k, c in COLS.items()})
            register = meter_start + round(r.energy_kwh * 1000)
            t0 = time.perf_counter()
            await self.call(ws, "MeterValues", ocpp.meter_values(connector, tx, ts(row.end_time), r, register))
            self.rtt_ms.append((time.perf_counter() - t0) * 1000)
            last_t = row.end_time
            await asyncio.sleep(self.minute_s)
        await self.call(ws, "StopTransaction", {"transactionId": tx, "meterStop": register,
                                                "timestamp": ts(last_t), "reason": reason})

    async def run(self, url: str) -> None:
        async with websockets.connect(f"{url}/{self.id}", subprotocols=[ocpp.SUBPROTOCOL],
                                      ping_interval=20, max_queue=None) as ws:
            rx = asyncio.create_task(self.receiver(ws))
            await self.call(ws, "BootNotification", {"chargePointVendor": "Simulated (ChargeGuard)",
                                                     "chargePointModel": "DC replay"})
            await asyncio.gather(*(self.connector(ws, c) for c in range(1, CONNECTORS + 1)))
            rx.cancel()


async def main_async(args) -> None:
    art = joblib.load(ROOT / "artifacts" / "risk.joblib")
    stations = [int(s) for s in art["test_stations"]]
    by_station = load_sessions(Path(args.data), stations)
    rng = random.Random(args.seed)
    no_smart = args.no_smart or f"SZ-{stations[-1]:02d}"
    cps = [SimChargePoint(f"SZ-{s:02d}", f"SZ-{s:02d}" != no_smart, demo_order(by_station[s], args.fault_share, rng),
                          args.minute_seconds, args.sessions) for s in stations]
    print(f"replaying stations {stations} as {len(cps)} charge points x {CONNECTORS} connectors; "
          f"fault share {'recorded' if args.fault_share <= 0 else args.fault_share}; no SmartCharging on {no_smart}")
    await asyncio.gather(*(cp.run(args.url) for cp in cps))
    rtt = np.concatenate([cp.rtt_ms for cp in cps]) if any(cp.rtt_ms for cp in cps) else np.zeros(1)
    out = {"meter_values": int(len(rtt)), "rtt_ms_p50": round(float(np.percentile(rtt, 50)), 2),
           "rtt_ms_p95": round(float(np.percentile(rtt, 95)), 2), "rtt_ms_p99": round(float(np.percentile(rtt, 99)), 2),
           "charge_points": len(cps), "connectors": len(cps) * CONNECTORS, "minute_seconds": args.minute_seconds}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "replay_latency.json").write_text(json.dumps(out, indent=2))
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8095/ocpp")
    ap.add_argument("--data", default=str(ROOT / "data" / "ev30.csv.gz"))
    ap.add_argument("--minute-seconds", type=float, default=0.5, help="real seconds per recorded minute")
    ap.add_argument("--fault-share", type=float, default=0.25, help="demo share of faulty sessions; 0 = recorded order")
    ap.add_argument("--sessions", type=int, default=10**9, help="max sessions per charge point")
    ap.add_argument("--no-smart", default=None, help="charge point without SmartCharging (default: last station)")
    ap.add_argument("--seed", type=int, default=7)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
