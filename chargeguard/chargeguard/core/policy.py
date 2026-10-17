"""Decision ladder for one charging session: continue -> watch -> slow -> stop recommended.

- A level is entered only when the risk score stays at or above its threshold for
  `persist` consecutive readings, and never before `min_minutes`: one noisy reading
  never changes a charge.
- Levels only go up within a session. Once a charge is slowed, a person decides
  what happens next.
- "slow" is automatic (OCPP SetChargingProfile caps the current) but only on
  chargers that report the SmartCharging feature profile; elsewhere the session is
  escalated to an operator instead.
- "stop recommended" never stops a charge by itself. An operator approves it, and
  only then does the central system send RemoteStopTransaction.

Thresholds are chosen on out-of-fold scores from the training stations, as the
score that a set share of normal sessions ever reaches (the false-action budget).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

LEVELS = ("continue", "watch", "slow", "stop_recommended")
# Share of normal (non-faulty) training sessions allowed to ever reach each level.
BUDGETS = {"watch": 0.05, "slow": 0.02, "stop": 0.005}


@dataclass(frozen=True)
class Thresholds:
    watch: float
    slow: float
    stop: float
    persist: int = 2
    min_minutes: int = 3

    def level_for(self, eff: float) -> int:
        return 3 if eff >= self.stop else 2 if eff >= self.slow else 1 if eff >= self.watch else 0

    def to_dict(self) -> dict:
        return asdict(self)


class Ladder:
    __slots__ = ("th", "level", "recent")

    def __init__(self, th: Thresholds):
        self.th = th
        self.level = 0
        self.recent: deque = deque(maxlen=th.persist)

    def step(self, minute: int, p: float) -> int | None:
        """Feed one score. Returns the new level if the session escalated, else None."""
        self.recent.append(p)
        if minute < self.th.min_minutes or len(self.recent) < self.th.persist:
            return None
        new = self.th.level_for(min(self.recent))
        if new > self.level:
            self.level = new
            return new
        return None


def effective(p: np.ndarray, persist: int = 2, min_minutes: int = 3) -> np.ndarray:
    """Per-reading score after the persistence rule, for one session's scores in time order.

    Reading k (1-based minute k+1) counts only from min_minutes on; NaN before.
    """
    p = np.asarray(p, dtype=float)
    eff = p.copy()
    for lag in range(1, persist):
        eff[lag:] = np.minimum(eff[lag:], p[:-lag])
        eff[:lag] = np.nan
    eff[: max(min_minutes - 1, 0)] = np.nan
    return eff


def first_at_or_above(eff: np.ndarray, thr: float) -> int:
    """Index of the first reading whose effective score reaches thr, or -1."""
    hit = np.flatnonzero(np.nan_to_num(eff, nan=-np.inf) >= thr)
    return int(hit[0]) if len(hit) else -1


def choose(max_eff_normal: np.ndarray, budgets: dict = BUDGETS, persist: int = 2, min_minutes: int = 3,
           stations: np.ndarray | None = None, min_sessions: int = 100) -> Thresholds:
    """Thresholds such that each budget's share of normal sessions ever reaches that level.

    With `stations`, each threshold is the median over stations of that station's own
    quantile, so a typical station sees the budgeted false-action rate and one station
    with unusual hardware can't set the thresholds for everyone (see station_check).
    """
    m = np.asarray(max_eff_normal, dtype=float)
    if stations is None:
        q = {k: float(np.quantile(m, 1 - b)) for k, b in budgets.items()}
    else:
        groups = [m[stations == s] for s in np.unique(stations)]
        groups = [g for g in groups if len(g) >= min_sessions] or [m]
        q = {k: float(np.median([np.quantile(g, 1 - b) for g in groups])) for k, b in budgets.items()}
    # Keep the ladder strictly ordered even if quantiles tie.
    slow = max(q["slow"], np.nextafter(q["watch"], 1))
    stop = max(q["stop"], np.nextafter(slow, 1))
    return Thresholds(watch=q["watch"], slow=float(slow), stop=float(stop), persist=persist, min_minutes=min_minutes)


def station_check(slow_rate_by_station: dict, budget: float = BUDGETS["slow"], factor: float = 5.0) -> list:
    """Training: stations whose *normal* sessions reach "slow" far more often than budgeted.

    Their alerts say more about the charger (sensor layout, calibration) than about the
    batteries, so they are left out of threshold calibration and need a hardware check.
    """
    return sorted(s for s, r in slow_rate_by_station.items() if r > factor * budget)


# Live, labels are unknown, so a charger is compared with the rest of the fleet.
STATION_WINDOW, STATION_MIN_SESSIONS, FLEET_FACTOR = 50, 20, 3.0


def station_outlier(rate: float, n: int, other_rates: list[float], budget: float = BUDGETS["slow"]) -> bool:
    """Live station check: this charger slows far more of its recent sessions than the others.

    Flag when, over its last STATION_WINDOW sessions (at least STATION_MIN_SESSIONS), more than
    5x the "slow" budget were slowed AND more than FLEET_FACTOR x the median of the other
    chargers that have enough sessions. A site that simply sees many faulty cars raises the
    rate everywhere it is compared, but a charger whose sensors read differently stands out.
    """
    if n < STATION_MIN_SESSIONS or rate <= 5 * budget:
        return False
    if not other_rates:
        return True
    return rate > FLEET_FACTOR * float(np.median(other_rates))
