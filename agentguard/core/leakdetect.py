"""Guard 2: stop sensitive data leaving in the assistant's reply.

Three independent signals, combined into one risk score:
  1. exact match against the bank's own secret strings (highest confidence),
  2. structural patterns (card numbers validated by Luhn, account IDs, keys),
  3. volume cues that suggest a bulk dump rather than one answer.

The exact-match signal is what makes this trustworthy: we are not guessing
whether something looks sensitive, we know the string came from our vault.
Data that belongs to the logged-in customer is allowed; anyone else's is not.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
ACCOUNT = re.compile(r"\bZB\d{2}-\d{4}-\d{4}\b")
NATIONAL_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
API_KEY = re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9\-_]{16,}\b", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
CSV_ROW = re.compile(r"^[^,\n]{2,40}(?:,[^,\n]{1,60}){3,}$", re.M)


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class LeakFeatures:
    exact_secret_hits: int
    distinct_secret_kinds: int
    valid_cards: int
    accounts: int
    national_ids: int
    api_keys: int
    emails: int
    csv_rows: int
    max_token_entropy: float
    length: int

    def vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def names() -> list[str]:
        return list(LeakFeatures.__annotations__.keys())


def extract(text: str, secrets: set[str] | None = None) -> LeakFeatures:
    secrets = secrets or set()
    hits = [s for s in secrets if s and s in text]
    kinds = set()
    for h in hits:
        if ACCOUNT.fullmatch(h):
            kinds.add("account")
        elif NATIONAL_ID.fullmatch(h):
            kinds.add("nid")
        elif EMAIL.fullmatch(h):
            kinds.add("email")
        elif luhn_ok(h):
            kinds.add("card")
        else:
            kinds.add("key")

    cards = [m.group(0) for m in CARD.finditer(text)]
    tokens = re.findall(r"\S{12,}", text)
    return LeakFeatures(
        exact_secret_hits=len(hits),
        distinct_secret_kinds=len(kinds),
        valid_cards=sum(1 for c in cards if luhn_ok(c)),
        accounts=len(ACCOUNT.findall(text)),
        national_ids=len(NATIONAL_ID.findall(text)),
        api_keys=len(API_KEY.findall(text)),
        emails=len(EMAIL.findall(text)),
        csv_rows=len(CSV_ROW.findall(text)),
        max_token_entropy=max((shannon_entropy(t) for t in tokens), default=0.0),
        length=len(text),
    )


@dataclass
class LeakVerdict:
    score: float
    flagged: bool
    kinds: list[str]
    foreign_hits: int


def check(text: str, secrets: set[str], own: set[str] | None = None) -> LeakVerdict:
    """Score a reply. Secrets in `own` belong to the current customer and are allowed."""
    foreign = secrets - (own or set())
    f = extract(text, foreign)
    kinds = []
    if f.exact_secret_hits:
        kinds.append(f"{f.exact_secret_hits} known secret value(s)")
    if f.api_keys:
        kinds.append("credential / API key")
    if f.csv_rows >= 3:
        kinds.append(f"bulk record dump ({f.csv_rows} rows)")
    # Pattern hits without a vault match are weaker evidence than exact hits.
    score = min(1.0, 0.6 * min(f.exact_secret_hits, 1) + 0.2 * min(f.exact_secret_hits / 5, 1)
                + 0.5 * min(f.api_keys, 1) + 0.2 * min(f.csv_rows / 3, 1))
    return LeakVerdict(score=round(score, 3), flagged=score >= 0.5, kinds=kinds,
                       foreign_hits=f.exact_secret_hits)


def redact(text: str, secrets: set[str] | None = None) -> str:
    """Mask sensitive spans so a blocked reply can still be shown safely."""
    out = text
    for s in sorted(secrets or set(), key=len, reverse=True):
        if s and s in out:
            out = out.replace(s, "[REDACTED]")
    out = API_KEY.sub("[REDACTED-KEY]", out)
    out = ACCOUNT.sub("[REDACTED-ACCOUNT]", out)
    out = NATIONAL_ID.sub("[REDACTED-ID]", out)
    out = CARD.sub(lambda m: "[REDACTED-CARD]" if luhn_ok(m.group(0)) else m.group(0), out)
    return EMAIL.sub("[REDACTED-EMAIL]", out)
