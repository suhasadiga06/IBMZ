"""Streaming features for one charging session, updated once per meter reading.

The same SessionFeatures object builds the training set (by replaying recorded
sessions reading by reading) and runs inside the live service, so the model is
never served a feature computed differently from the one it was trained on.
Every update is O(1): running sums, maxima and a short window of recent readings.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Values that are undefined early in a session (e.g. kWh per % of charge before the
# charge level has moved) use -1, which tree models handle without imputation.
UNDEFINED = -1.0
WINDOW = 5  # minutes, for "change over the last few minutes" features

FEATURES = [
    "minutes", "battery_type",
    "v_last", "v_mean", "v_std", "v_max", "v_slope", "v_d5",
    "i_last", "i_mean", "i_std", "i_max", "i_last_ratio", "i_d5",
    "p_mean", "p_max",
    "soc0", "soc_now", "soc_gain", "kwh", "kwh_per_soc",
    "t1_last", "t2_last", "t1_max", "t2_max", "t1_rise", "t2_rise", "t_gap_max",
    "t1_slope", "t2_slope", "t1_d5", "t2_d5", "t_per_kw",
]
# Absolute levels that mostly say how big the vehicle is (pack voltage, power, current,
# energy, kWh per % of charge ~ pack capacity) rather than how this charge is behaving.
# In the Shenzhen data almost every session with a 150-300 kWh apparent pack is labelled
# faulty, so a model given these can score well by recognising large vehicles.
SIZE_FEATURES = {"v_last", "v_mean", "v_max", "i_last", "i_mean", "i_max", "p_mean", "p_max",
                 "kwh", "kwh_per_soc"}
# The served model: how this charge behaves, without vehicle-size levels. Absolute gun
# temperatures stay in: making them relative to each charger's own baseline cut false alarms
# but lost too many faults in cross-validation (see README, development history). Chargers
# whose sensors read unusually are caught by the runtime station check instead.
BEHAVIOUR_FEATURES = [f for f in FEATURES if f not in SIZE_FEATURES]

# GB/T 27930 battery type codes as they appear in the Shenzhen data.
BATTERY_TYPES = {3: "LFP", 4: "LMO", 5: "LCO", 6: "NMC", 7: "LP"}


@dataclass
class Reading:
    voltage: float      # V
    current: float      # A
    power: float        # kW
    soc: float          # %
    energy_kwh: float   # kWh delivered so far in this session
    temp1: float        # charging gun temperature sensor 1, deg C
    temp2: float        # charging gun temperature sensor 2, deg C


class _Run:
    """Running mean/std (Welford), max, and least-squares slope against the minute index."""
    __slots__ = ("n", "mean", "m2", "max", "sx", "sy", "sxy", "sxx", "first", "last")

    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.0
        self.max = -math.inf
        self.sx = self.sy = self.sxy = self.sxx = 0.0
        self.first = self.last = 0.0

    def add(self, x: float) -> None:
        if self.n == 0:
            self.first = x
        t = float(self.n)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if x > self.max:
            self.max = x
        self.sx += t; self.sy += x; self.sxy += t * x; self.sxx += t * t
        self.last = x

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def slope(self) -> float:
        if self.n < 3:
            return 0.0
        den = self.n * self.sxx - self.sx * self.sx
        return (self.n * self.sxy - self.sx * self.sy) / den if den else 0.0


class SessionFeatures:
    __slots__ = ("battery_type", "v", "i", "p", "t1", "t2", "soc0", "soc", "kwh", "gap_max", "recent")

    def __init__(self, battery_type: int | None):
        self.battery_type = float(battery_type) if battery_type is not None else UNDEFINED
        self.v, self.i, self.p, self.t1, self.t2 = _Run(), _Run(), _Run(), _Run(), _Run()
        self.soc0 = None
        self.soc = 0.0
        self.kwh = 0.0
        self.gap_max = 0.0
        self.recent: deque = deque(maxlen=WINDOW + 1)

    @property
    def minutes(self) -> int:
        return self.v.n

    def update(self, r: Reading) -> list[float]:
        self.v.add(r.voltage); self.i.add(r.current); self.p.add(r.power)
        self.t1.add(r.temp1); self.t2.add(r.temp2)
        if self.soc0 is None:
            self.soc0 = r.soc
        self.soc, self.kwh = r.soc, r.energy_kwh
        self.gap_max = max(self.gap_max, abs(r.temp2 - r.temp1))
        self.recent.append((r.voltage, r.current, r.temp1, r.temp2))
        return self.vector()

    def vector(self) -> list[float]:
        v, i, p, t1, t2 = self.v, self.i, self.p, self.t1, self.t2
        old = self.recent[0]
        soc_gain = self.soc - self.soc0
        return [
            float(v.n), self.battery_type,
            v.last, v.mean, v.std, v.max, v.slope, v.last - old[0],
            i.last, i.mean, i.std, i.max, i.last / i.max if i.max > 0 else UNDEFINED, i.last - old[1],
            p.mean, p.max,
            self.soc0, self.soc, soc_gain, self.kwh, self.kwh / soc_gain if soc_gain >= 1 else UNDEFINED,
            t1.last, t2.last, t1.max, t2.max, t1.max - t1.first, t2.max - t2.first, self.gap_max,
            t1.slope, t2.slope, t1.last - old[2], t2.last - old[3],
            (t1.max - t1.first) / max(p.mean, 1.0),
        ]


# Column names in the Shenzhen PolyU/Autosun charging dataset.
COLS = dict(voltage="chargingv", current="charginga", power="out_power", soc="current_soc",
            energy_kwh="total_charging_kwh", temp1="charging_gun_temperature1",
            temp2="charging_gun_temperature2")


def replay(df) -> tuple[np.ndarray, np.ndarray]:
    """Features for every reading of every session in df, as the live service would see them.

    df must be sorted by transaction_id, then time. Returns (X float32, minute index int32).
    """
    tx = df["transaction_id"].to_numpy()
    bt = df["types"].to_numpy()
    cols = [df[COLS[k]].to_numpy(dtype=float) for k in Reading.__dataclass_fields__]
    X = np.empty((len(df), len(FEATURES)), dtype=np.float32)
    minute = np.empty(len(df), dtype=np.int32)
    state, current = None, None
    for n in range(len(df)):
        if tx[n] != current:
            current, state = tx[n], SessionFeatures(int(bt[n]))
        X[n] = state.update(Reading(*(c[n] for c in cols)))
        minute[n] = state.minutes
    return X, minute
