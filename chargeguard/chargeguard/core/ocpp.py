"""The OCPP 1.6-J (JSON over WebSocket, subprotocol "ocpp1.6") messages ChargeGuard uses.

Charge point -> central system:
  BootNotification, StartTransaction, MeterValues, StopTransaction, and DataTransfer
  with vendorId "GBT27930" carrying the battery type from the BMS handshake that
  GB/T 27930 (China's DC charging protocol) performs before a charge starts.
Central system -> charge point:
  GetConfiguration("SupportedFeatureProfiles") once per connection;
  SetChargingProfile to cap the current, sent only when the charger lists the optional
  SmartCharging feature profile; RemoteStopTransaction, only after an operator approves.

ChargeGuard is a prototype: the charge points in this repo are simulated, and nothing
here has been certified for controlling real chargers.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from chargeguard.core.features import Reading

CALL, CALLRESULT, CALLERROR = 2, 3, 4
SUBPROTOCOL = "ocpp1.6"
GBT_VENDOR = "GBT27930"


@dataclass
class Message:
    kind: int
    uid: str
    action: str | None
    payload: dict


def parse(text: str) -> Message:
    m = json.loads(text)
    if not isinstance(m, list) or len(m) < 3 or m[0] not in (CALL, CALLRESULT, CALLERROR):
        raise ValueError("not an OCPP-J message")
    if m[0] == CALL:
        return Message(CALL, str(m[1]), str(m[2]), m[3] if len(m) > 3 else {})
    if m[0] == CALLRESULT:
        return Message(CALLRESULT, str(m[1]), None, m[2])
    return Message(CALLERROR, str(m[1]), str(m[2]), {"description": m[3] if len(m) > 3 else ""})


def call(action: str, payload: dict, uid: str | None = None) -> tuple[str, str]:
    uid = uid or uuid.uuid4().hex[:12]
    return uid, json.dumps([CALL, uid, action, payload])


def result(uid: str, payload: dict) -> str:
    return json.dumps([CALLRESULT, uid, payload])


def error(uid: str, code: str, description: str) -> str:
    return json.dumps([CALLERROR, uid, code, description, {}])


# --- MeterValues <-> Reading -------------------------------------------------------------

def _num(v) -> float:
    return float(v)


def reading_from_meter_values(payload: dict, meter_start_wh: float) -> tuple[str, Reading, float]:
    """Latest sampled values -> (timestamp, Reading, energy register in Wh)."""
    mv = payload["meterValue"][-1]
    vals: dict[str, float] = {}
    for s in mv["sampledValue"]:
        meas = s.get("measurand", "Energy.Active.Import.Register")
        x = _num(s["value"])
        unit = s.get("unit", "")
        if meas == "Power.Active.Import" and unit == "W":
            x /= 1000.0
        if meas == "Energy.Active.Import.Register" and unit == "kWh":
            x *= 1000.0
        key = f"{meas}:{s['location']}" if meas == "Temperature" else meas
        vals[key] = x
    register = vals.get("Energy.Active.Import.Register", meter_start_wh)
    r = Reading(voltage=vals.get("Voltage", 0.0), current=vals.get("Current.Import", 0.0),
                power=vals.get("Power.Active.Import", 0.0), soc=vals.get("SoC", 0.0),
                energy_kwh=(register - meter_start_wh) / 1000.0,
                temp1=vals.get("Temperature:Outlet", 0.0), temp2=vals.get("Temperature:Cable", 0.0))
    return mv["timestamp"], r, register


def meter_values(connector_id: int, transaction_id: int, timestamp: str, r: Reading, register_wh: float) -> dict:
    sv = [
        {"value": f"{r.voltage:.1f}", "measurand": "Voltage", "unit": "V"},
        {"value": f"{r.current:.2f}", "measurand": "Current.Import", "unit": "A"},
        {"value": f"{r.power:.2f}", "measurand": "Power.Active.Import", "unit": "kW"},
        {"value": f"{r.soc:.0f}", "measurand": "SoC", "unit": "Percent"},
        {"value": f"{register_wh:.0f}", "measurand": "Energy.Active.Import.Register", "unit": "Wh"},
        {"value": f"{r.temp1:.0f}", "measurand": "Temperature", "location": "Outlet", "unit": "Celsius"},
        {"value": f"{r.temp2:.0f}", "measurand": "Temperature", "location": "Cable", "unit": "Celsius"},
    ]
    return {"connectorId": connector_id, "transactionId": transaction_id,
            "meterValue": [{"timestamp": timestamp, "sampledValue": sv}]}


# --- Central system commands -------------------------------------------------------------

def get_configuration() -> dict:
    return {"key": ["SupportedFeatureProfiles"]}


def feature_profiles(conf_result: dict) -> set[str]:
    for k in conf_result.get("configurationKey", []):
        if k.get("key") == "SupportedFeatureProfiles":
            return {p.strip() for p in str(k.get("value", "")).split(",") if p.strip()}
    return set()


def set_charging_profile(connector_id: int, transaction_id: int, limit_amps: float, profile_id: int = 1) -> dict:
    return {"connectorId": connector_id, "csChargingProfiles": {
        "chargingProfileId": profile_id, "transactionId": transaction_id, "stackLevel": 1,
        "chargingProfilePurpose": "TxProfile", "chargingProfileKind": "Relative",
        "chargingSchedule": {"chargingRateUnit": "A",
                             "chargingSchedulePeriod": [{"startPeriod": 0, "limit": round(limit_amps, 1)}]}}}


def remote_stop(transaction_id: int) -> dict:
    return {"transactionId": transaction_id}


def battery_type_from_data_transfer(payload: dict) -> tuple[int, int] | None:
    """GB/T 27930 handshake forwarded as DataTransfer -> (connectorId, battery type code)."""
    if payload.get("vendorId") != GBT_VENDOR:
        return None
    d = json.loads(payload.get("data") or "{}")
    return int(d["connectorId"]), int(d["batteryType"])
