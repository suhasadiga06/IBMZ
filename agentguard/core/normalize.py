"""Canonicalise text before classification so obfuscation tricks stop working.

Attackers hide instructions with base64, zero-width characters, l33t-speak,
s p a c e d letters and look-alike Unicode. We undo those first, then classify.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
B64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
SPACED = re.compile(r"\b(?:[A-Za-z][ .\-_]){3,}[A-Za-z]\b")
LEET_WORD = re.compile(r"\b(?=\w*[a-z])(?=\w*[0-9@$])[\w@$]{4,}\b", re.I)


def _decode_b64(match: re.Match) -> str:
    blob = match.group(0)
    try:
        raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return blob
    printable = sum(ch.isprintable() for ch in text) / max(len(text), 1)
    return f"{blob} [decoded: {text}]" if printable > 0.95 and " " in text else blob


def canonicalize(text: str) -> str:
    """Return a lower-cased, de-obfuscated version of `text`.

    Decoded content is appended rather than replacing the original, so the
    classifier sees both the trick and what it hides.
    """
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    t = B64_RUN.sub(_decode_b64, t)
    t = SPACED.sub(lambda m: re.sub(r"[ .\-_]", "", m.group(0)), t)
    t = LEET_WORD.sub(lambda m: m.group(0).translate(LEET), t)
    return re.sub(r"\s+", " ", t).strip().lower()


def obfuscation_signals(text: str) -> dict[str, int]:
    """Counts of obfuscation tricks; their presence alone is suspicious."""
    return {
        "zero_width": sum(text.count(ch) for ch in "​‌‍⁠﻿"),
        "base64_runs": sum(1 for m in B64_RUN.finditer(text) if "[decoded:" in _decode_b64(m)),
        "spaced_words": len(SPACED.findall(text)),
        "non_ascii": sum(1 for ch in text if ord(ch) > 127),
    }
