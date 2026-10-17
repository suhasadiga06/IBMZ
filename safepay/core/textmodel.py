"""Model 1: how harmful is this payment reference on its own?

Two heads share one feature space:
  harm    - abusive, harassing or insulting (Civil Comments toxicity >= 0.5)
  threat  - threatens harm (Civil Comments threat label)

Word n-grams catch phrasing; character n-grams catch misspellings that survive
canonicalisation. Linear heads keep inference around a millisecond on one core
and let every flag point at the words that caused it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .normalize import canonicalize, obfuscation_signals

# Identity terms are replaced by one neutral token before the model sees text.
# In online comments these words co-occur with toxicity, so an unmasked model
# learns "mentions a group" = "abusive" (our first model flagged 99.7% of
# harmless template sentences containing "gay"). Masking makes it judge how
# something is said, not who is mentioned. Slurs are not identity terms and
# stay visible to the model.
IDENTITY_TERMS = [
    "gay", "gays", "lesbian", "lesbians", "bisexual", "bi", "transgender", "trans", "queer", "lgbt", "lgbtq",
    "homosexual", "homosexuals", "heterosexual", "straight", "nonbinary", "non-binary",
    "black", "blacks", "white", "whites", "african", "africans", "african-american", "asian", "asians",
    "hispanic", "hispanics", "latino", "latina", "latinos", "latinas", "latinx", "mexican", "mexicans",
    "european", "europeans", "indian", "indians", "chinese", "japanese", "arab", "arabs", "middle eastern",
    "muslim", "muslims", "islam", "islamic", "christian", "christians", "jew", "jews", "jewish",
    "buddhist", "buddhists", "catholic", "catholics", "protestant", "protestants", "sikh", "sikhs",
    "hindu", "hindus", "atheist", "atheists", "immigrant", "immigrants", "refugee", "refugees",
    "female", "male", "woman", "women", "man", "men", "elderly", "old", "older", "young", "younger",
    "teenage", "teenager", "blind", "deaf", "paralyzed", "disabled", "american", "canadian",
]
_IDENTITY_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(IDENTITY_TERMS, key=len, reverse=True)))
                          + r")\b")


_REPEATED_MASK = re.compile(r"identityterm(?:[\s\-]+identityterm)+")


def mask_identity(text: str) -> str:
    # "african american women" -> one token, not three: repeats carried their own bias.
    return _REPEATED_MASK.sub("identityterm", _IDENTITY_RE.sub("identityterm", text))


def prepare(text: str) -> str:
    return mask_identity(canonicalize(str(text)))


@dataclass
class MessageScore:
    harm: float
    threat: float
    evidence: list[str] = field(default_factory=list)
    obfuscation: dict[str, int] = field(default_factory=dict)


class MessageModel:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                    sublinear_tf=True, strip_accents="unicode")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    max_features=200_000, sublinear_tf=True)
        self.heads: dict[str, LogisticRegression] = {}
        self.thresholds: dict[str, float] = {"harm": 0.5, "threat": 0.5}
        self._names = None

    def _X(self, texts, fit: bool = False):
        clean = [prepare(t) for t in texts]
        if fit:
            return hstack([self.word.fit_transform(clean), self.char.fit_transform(clean)]).tocsr()
        return hstack([self.word.transform(clean), self.char.transform(clean)]).tocsr()

    def _X_raw(self, texts):
        """Features without canonicalising: only used to measure what it buys."""
        low = [mask_identity(str(t).lower()) for t in texts]
        return hstack([self.word.transform(low), self.char.transform(low)]).tocsr()

    def fit(self, texts, labels: dict[str, np.ndarray]) -> "MessageModel":
        X = self._X(texts, fit=True)
        for name, y in labels.items():
            self.heads[name] = LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced",
                                                  solver="liblinear").fit(X, y)
        self._names = None
        return self

    def proba(self, texts) -> dict[str, np.ndarray]:
        X = self._X(texts)
        return {name: h.predict_proba(X)[:, 1] for name, h in self.heads.items()}

    def _feature_names(self):
        if self._names is None:
            w = self.word.get_feature_names_out()
            self._names = (np.concatenate([w, self.char.get_feature_names_out()]), len(w))
        return self._names

    def score(self, text: str, top_k: int = 4) -> MessageScore:
        if not text or not text.strip():
            return MessageScore(0.0, 0.0)
        X = self._X([text])
        p = {name: float(h.predict_proba(X)[0, 1]) for name, h in self.heads.items()}
        names, n_word = self._feature_names()
        contrib = X.multiply(self.heads["harm"].coef_[0]).tocsr()
        order = np.argsort(-contrib.data)
        pos = [contrib.indices[j] for j in order[: top_k * 3] if contrib.data[j] > 0]
        words = [names[i] for i in pos if i < n_word]
        evidence = (words or [names[i].strip() for i in pos])[:top_k]
        return MessageScore(harm=p.get("harm", 0.0), threat=p.get("threat", 0.0), evidence=evidence,
                            obfuscation={k: v for k, v in obfuscation_signals(text).items() if v})
