"""Plain-language reasons for a risk score: which readings are unusual next to normal charges.

`reference` holds, per feature, the 1st/5th/50th/95th/99th percentile over normal sessions of
the training stations (written by scripts/train.py). A reading counts as unusual when it is
past the normal 5th or 95th percentile in the direction that phrase describes; the strongest
few are reported, each quoting the percentile line it actually crosses. These describe what
the model saw, not a causal diagnosis of the battery.
"""
from __future__ import annotations

from chargeguard.core.features import FEATURES

# feature -> list of (direction, what was seen, number format, unit)
PHRASES = {
    "t_gap_max": [("low", "the two gun temperature sensors read unusually close together (at most {x} apart)", "{:.0f}", " °C"),
                  ("high", "the two gun temperature sensors disagree by {x}", "{:.0f}", " °C")],
    "t1_last": [("high", "gun temperature (sensor 1) is now {x}", "{:.0f}", " °C")],
    "t2_last": [("high", "gun temperature (sensor 2) is now {x}", "{:.0f}", " °C")],
    "t1_max": [("high", "gun temperature (sensor 1) reached {x}", "{:.0f}", " °C")],
    "t2_max": [("high", "gun temperature (sensor 2) reached {x}", "{:.0f}", " °C")],
    "t1_d5": [("high", "gun temperature (sensor 1) rose {x} in the last 5 minutes", "{:.0f}", " °C")],
    "t2_d5": [("high", "gun temperature (sensor 2) rose {x} in the last 5 minutes", "{:.0f}", " °C")],
    "t1_rise": [("high", "gun temperature (sensor 1) is {x} above where it started", "{:.0f}", " °C")],
    "t2_rise": [("high", "gun temperature (sensor 2) is {x} above where it started", "{:.0f}", " °C")],
    "t1_slope": [("high", "gun temperature (sensor 1) is climbing {x} per minute", "{:.2f}", " °C")],
    "t2_slope": [("high", "gun temperature (sensor 2) is climbing {x} per minute", "{:.2f}", " °C")],
    "t_per_kw": [("high", "temperature is rising faster than the power explains ({x} per kW)", "{:.2f}", " °C")],
    "i_std": [("high", "current is unsteady (spread {x})", "{:.0f}", " A")],
    "i_d5": [("high", "current changed by {x} in the last 5 minutes", "{:.0f}", " A")],
    "i_last_ratio": [("low", "current has fallen to {x} of its peak", "{:.0%}", "")],
    "v_std": [("high", "pack voltage is unsteady (spread {x})", "{:.1f}", " V")],
    "v_d5": [("high", "pack voltage moved {x} in the last 5 minutes", "{:.1f}", " V")],
    "v_slope": [("high", "pack voltage is climbing {x} per minute", "{:.2f}", " V")],
}
_INDEX = {f: i for i, f in enumerate(FEATURES)}
FALLBACK = "an unusual combination of readings; no single value is outside the normal range"


def reasons(vector: list[float], reference: dict, allowed: set | None = None, k: int = 3) -> list[str]:
    """Up to k plain-language reasons, strongest first. `allowed`: features the model uses."""
    scored = []
    for f, variants in PHRASES.items():
        if (allowed is not None and f not in allowed) or f not in reference:
            continue
        x = vector[_INDEX[f]]
        p1, p5, p50, p95, p99 = reference[f]
        for direction, seen, fmt, unit in variants:
            if direction == "high" and x > p95:
                pct, line, word = (99, p99, "under") if x > p99 else (95, p95, "under")
                strength = (x - p50) / max(p99 - p50, 1e-6)
            elif direction == "low" and x < p5:
                pct, line, word = (99, p1, "above") if x < p1 else (95, p5, "above")
                strength = (p50 - x) / max(p50 - p1, 1e-6)
            else:
                continue
            text = f"{seen.format(x=fmt.format(x) + unit)}; {pct}% of normal charges stay {word} {fmt.format(line)}{unit}"
            scored.append((strength, text))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored[:k]] or [FALLBACK]
