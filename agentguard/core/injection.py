"""Guard 1: detect instructions that try to manipulate the assistant.

Word n-grams catch phrasing; character n-grams catch obfuscated spellings that
survive canonicalisation. Linear model, so every verdict can be explained by
the n-grams that pushed it over the line, and inference is sub-millisecond on
a single core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .normalize import canonicalize, obfuscation_signals

MAX_CHARS = 2000


@dataclass
class InjectionVerdict:
    score: float
    flagged: bool
    evidence: list[str]
    obfuscation: dict[str, int]


class InjectionModel:
    def __init__(self, threshold: float = 0.5):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000,
                                    sublinear_tf=True, strip_accents="unicode")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                    max_features=150_000, sublinear_tf=True)
        self.clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                                      solver="liblinear")
        self.threshold = threshold

    @staticmethod
    def prep(texts) -> list[str]:
        return [canonicalize(str(t)[:MAX_CHARS]) for t in texts]

    def _features(self, clean: list[str], fit: bool = False):
        if fit:
            return hstack([self.word.fit_transform(clean), self.char.fit_transform(clean)]).tocsr()
        return hstack([self.word.transform(clean), self.char.transform(clean)]).tocsr()

    def fit(self, texts, labels, sample_weight=None) -> "InjectionModel":
        X = self._features(self.prep(texts), fit=True)
        self.clf.fit(X, np.asarray(labels), sample_weight=sample_weight)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._features(self.prep(texts)))[:, 1]

    def _names(self) -> tuple[np.ndarray, int]:
        # Built once: rebuilding ~300K feature names per call dominated latency.
        if getattr(self, "_name_cache", None) is None:
            words = self.word.get_feature_names_out()
            self._name_cache = (np.concatenate([words, self.char.get_feature_names_out()]), len(words))
        return self._name_cache

    def verdict(self, text: str, top_k: int = 5) -> InjectionVerdict:
        clean = self.prep([text])
        X = self._features(clean)
        score = float(self.clf.predict_proba(X)[0, 1])
        names, n_word = self._names()
        contrib = X.multiply(self.clf.coef_[0]).tocsr()
        order = np.argsort(-contrib.data)
        pos = [(contrib.indices[j], contrib.data[j]) for j in order[:top_k * 3] if contrib.data[j] > 0]
        # Prefer readable word n-grams for the explanation; fall back to char grams.
        words = [names[i] for i, _ in pos if i < n_word]
        evidence = (words or [names[i].strip() for i, _ in pos])[:top_k]
        return InjectionVerdict(score=score, flagged=score >= self.threshold,
                                evidence=evidence, obfuscation=obfuscation_signals(text))
