"""Model 2's inputs: the running state of one sender -> recipient relationship.

Updated one payment at a time, in O(1), so it can sit inline in a real-time
payment flow. This is the main difference from monthly batch scoring: the
features for a payment are ready the moment the payment arrives.

Split messages: abusers spread one threat over several transfers so that each
reference looks harmless ("I am" / "going to" / "find you"). We keep the last
few references sent within SPLIT_WINDOW and score them joined together.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

SPLIT_WINDOW = 30 * 60        # seconds
SPLIT_KEEP = 8
# Recent references kept per relationship, so that if a case opens, its record
# starts before the case did. Stored encrypted with the rest of the state.
HISTORY_KEEP = 20
RATE_WINDOW = 7 * 86400
CUSUM_K = 0.25                # harm above this level accumulates evidence
MICRO = 1.00                  # "penny payment" cut-off in currency units
HARM_CUT = 0.5
# Below this, message scores carry no reliable signal. Flooring them stops the
# relationship model from memorising the exact score of particular harmless
# phrases (our first version did, and failed on unseen wording).
SCORE_FLOOR = 0.25


def _floor(x: float) -> float:
    return x if x >= SCORE_FLOOR else 0.0


@dataclass
class PairState:
    first_ts: float = 0.0
    last_ts: float = 0.0
    n_tx: int = 0
    n_msg: int = 0
    n_micro: int = 0
    n_night: int = 0
    n_reply: int = 0
    n_harmful: int = 0
    harm_sum: float = 0.0
    harm_max: float = 0.0
    threat_max: float = 0.0
    ewm_harm: float = 0.0
    cusum: float = 0.0
    split_harm_max: float = 0.0
    split_threat_max: float = 0.0
    amount_sum: float = 0.0
    amount_sq: float = 0.0
    hidden_at: float = 0.0          # recipient chose "hide messages from this sender"
    n_after_hide: int = 0
    burst_max: int = 0
    recent: list[float] = field(default_factory=list)      # timestamps in RATE_WINDOW
    buffer: list[list] = field(default_factory=list)       # [ts, reference] in SPLIT_WINDOW
    history: list[list] = field(default_factory=list)      # [ts, amount, reference, hidden], last HISTORY_KEEP

    # -- updates ----------------------------------------------------------------
    def split_text(self, ts: float, reference: str) -> str | None:
        """The joined recent references including this one, if there is more than one."""
        if not reference:
            return None
        recent = [r for t, r in self.buffer if ts - t <= SPLIT_WINDOW]
        return " ".join(recent + [reference]) if recent else None

    def update(self, ts: float, hour: int, amount: float, reference: str, harm: float, threat: float,
               split_harm: float = 0.0, split_threat: float = 0.0) -> None:
        if self.n_tx == 0:
            self.first_ts = ts
        self.n_tx += 1
        self.last_ts = ts
        self.n_micro += amount < MICRO
        self.n_night += 0 <= hour < 5
        self.amount_sum += amount
        self.amount_sq += amount * amount
        if self.hidden_at and ts >= self.hidden_at:
            self.n_after_hide += 1

        self.recent = [t for t in self.recent if ts - t <= RATE_WINDOW] + [ts]
        self.burst_max = max(self.burst_max, sum(1 for t in self.recent if ts - t <= 3600))

        if reference:
            harm, threat = _floor(harm), _floor(threat)
            split_harm, split_threat = _floor(split_harm), _floor(split_threat)
            self.n_msg += 1
            self.harm_sum += harm
            self.harm_max = max(self.harm_max, harm)
            self.threat_max = max(self.threat_max, threat)
            self.n_harmful += harm >= HARM_CUT
            self.ewm_harm = 0.7 * self.ewm_harm + 0.3 * harm
            self.cusum = max(0.0, self.cusum + harm - CUSUM_K)
            self.split_harm_max = max(self.split_harm_max, split_harm)
            self.split_threat_max = max(self.split_threat_max, split_threat)
            self.buffer = [b for b in self.buffer if ts - b[0] <= SPLIT_WINDOW][-(SPLIT_KEEP - 1):] + [[ts, reference]]
            self.history = (self.history + [[ts, round(amount, 2), reference, False]])[-HISTORY_KEEP:]

    def reply(self) -> None:
        """The recipient paid the sender back: two-way contact."""
        self.n_reply += 1

    # -- model input ------------------------------------------------------------
    def features(self, now: float) -> list[float]:
        n = max(self.n_tx, 1)
        days = max((self.last_ts - self.first_ts) / 86400, 1 / 24)
        recent = [t for t in self.recent if now - t <= RATE_WINDOW]
        prev = max(self.n_tx - len(recent), 0)
        mean = self.amount_sum / n
        var = max(self.amount_sq / n - mean * mean, 0.0)
        return [
            math.log1p(self.n_tx), days, self.n_tx / days,
            self.n_micro / n, self.n_night / n, self.n_msg / n, min(self.n_reply / n, 1.0),
            self.harm_max, self.harm_sum / max(self.n_msg, 1), self.n_harmful / max(self.n_msg, 1),
            self.threat_max, self.ewm_harm, self.split_harm_max, self.split_threat_max,
            self.cusum, float(self.burst_max), float(len(recent)), len(recent) / max(prev, 1),
            float(self.hidden_at > 0), math.log1p(self.n_after_hide),
            math.log1p(mean), math.sqrt(var) / max(mean, 1e-6), math.log1p(self.n_harmful),
        ]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "PairState":
        return cls(**json.loads(s))


PAIR_FEATURES = [
    "log_payments", "days_active", "payments_per_day",
    "share_penny_payments", "share_night", "share_with_message", "reply_ratio",
    "harm_max", "harm_mean", "share_harmful", "threat_max", "harm_recent_ewm",
    "split_message_harm", "split_message_threat", "escalation_cusum", "max_payments_in_1h",
    "payments_last_7d", "frequency_trend", "recipient_hid_sender", "log_contact_after_hide",
    "log_amount_mean", "amount_cv", "log_harmful_messages",
]
# Context from the recipient's side, so an abuser who switches to a new account
# after being hidden is not a stranger with a clean slate (computed by Engine).
RECIPIENT_FEATURES = ["recipient_hid_another_sender", "days_from_that_hide_to_first_contact",
                      "new_senders_since_that_hide"]
FEATURES = PAIR_FEATURES + RECIPIENT_FEATURES
NO_HIDE_DAYS = 60.0

