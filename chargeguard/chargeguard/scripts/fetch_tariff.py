"""Build artifacts/tariff.json: Shenzhen charging price by hour of day, from UrbanEV (CC0).

    python -m chargeguard.scripts.fetch_tariff
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data/"


def hourly(name: str) -> list[float]:
    with urllib.request.urlopen(BASE + name, timeout=120) as r:
        df = pd.read_csv(io.BytesIO(r.read()), parse_dates=["time"])
    by_hour = df.set_index("time").mean(axis=1).groupby(lambda t: t.hour).mean()
    return [round(float(by_hour.get(h, by_hour.mean())), 4) for h in range(24)]


def main() -> None:
    out = {"electricity": hourly("e_price.csv"), "service": hourly("s_price.csv"),
           "unit": "CNY/kWh", "source": "UrbanEV (Shenzhen, Sep 2022-Feb 2023), mean over zones by hour"}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "tariff.json").write_text(json.dumps(out, indent=1))
    print("hour  electricity  service")
    for h in range(0, 24, 3):
        print(f"{h:4d}  {out['electricity'][h]:11.3f}  {out['service'][h]:7.3f}")


if __name__ == "__main__":
    main()
