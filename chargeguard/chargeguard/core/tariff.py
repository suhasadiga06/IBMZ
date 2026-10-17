"""What a charge costs: Shenzhen public charging price per kWh by hour of day.

Electricity price plus service fee, averaged over UrbanEV's Shenzhen zones (Sep 2022 to
Feb 2023, CNY/kWh), written to artifacts/tariff.json by scripts/fetch_tariff.py.
"""
from __future__ import annotations

import json
from pathlib import Path

FALLBACK = {"electricity": [0.9] * 24, "service": [0.76] * 24, "source": "flat fallback (UrbanEV medians)"}


class Tariff:
    def __init__(self, path: Path | None = None):
        d = FALLBACK
        if path is not None and Path(path).exists():
            d = json.loads(Path(path).read_text())
        self.electricity, self.service, self.source = d["electricity"], d["service"], d["source"]

    def per_kwh(self, hour: int) -> float:
        return self.electricity[hour % 24] + self.service[hour % 24]

    def cost(self, kwh: float, hour: int) -> float:
        return kwh * self.per_kwh(hour)
