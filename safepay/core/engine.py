"""The scoring engine shared by training, evaluation and the live API.

One call per payment, in arrival order:
  1. score the reference (Model 1), memoised for repeated references
  2. join it with the sender's last few references and score that (split messages)
  3. update the relationship state; a payment back from the recipient counts as a reply
  4. score the relationship (Model 2) from its state

Training and the live service run exactly this code, so offline numbers
describe what the service actually does.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .normalize import EMOJI_RE, canonicalize
from .state import PairState
from .textmodel import MessageModel

MIN_LETTERS = 4


def informative(text: str) -> bool:
    """Too little text to judge on its own ("I", "a", "you")? Then it is only
    judged as part of the joined recent messages. With one or two features a
    linear model's score is dominated by a single weight: "I" scored 0.96 as a
    threat before this rule."""
    if not text:
        return False
    return EMOJI_RE.search(text) is not None or sum(c.isalpha() for c in canonicalize(text)) >= MIN_LETTERS


@dataclass
class Observation:
    harm: float
    threat: float
    split_text: str | None
    split_harm: float
    split_threat: float
    rel_score: float
    features: list[float]
    state: PairState
    n_before: int


class Engine:
    def __init__(self, text: MessageModel, rel=None, cache_size: int = 50_000):
        self.text = text
        self.rel = rel
        self.states: dict[tuple[str, str], PairState] = {}
        self._cache: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._cache_size = cache_size

    def score_text(self, text: str) -> tuple[float, float]:
        if not informative(text):
            return 0.0, 0.0
        hit = self._cache.get(text)
        if hit is not None:
            self._cache.move_to_end(text)
            return hit
        p = self.text.proba([text])
        out = (float(p["harm"][0]), float(p["threat"][0]))
        self._cache[text] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def state(self, sender: str, recipient: str) -> PairState:
        key = (sender, recipient)
        st = self.states.get(key)
        if st is None:
            st = self.states[key] = PairState()
        return st

    def hide(self, recipient: str, sender: str, ts: float) -> None:
        st = self.state(sender, recipient)
        st.hidden_at = st.hidden_at or ts

    def observe(self, when: datetime, sender: str, recipient: str, amount: float, reference: str) -> Observation:
        ts = when.timestamp()
        st = self.state(sender, recipient)
        n_before = st.n_tx
        harm, threat = self.score_text(reference)
        joined = st.split_text(ts, reference)
        s_harm, s_threat = self.score_text(joined) if joined else (0.0, 0.0)
        st.update(ts, when.hour, amount, reference, harm, threat, s_harm, s_threat)
        # This payment is a reply in the eyes of the opposite pair, if that pair exists.
        back = self.states.get((recipient, sender))
        if back is not None:
            back.reply()
        feats = st.features(ts)
        score = float(self.rel.predict_proba(np.asarray([feats], dtype=np.float32))[0, 1]) if self.rel else 0.0
        return Observation(harm, threat, joined, s_harm, s_threat, score, feats, st, n_before)
