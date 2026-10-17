"""Undo the tricks people use to slip abuse past payment-reference filters.

Banks report that keyword blocklists are easy to evade: "k.i.l.l", "d1e",
"y o u", Cyrillic look-alike letters, and emoji standing in for words. We
canonicalise first, then classify, so the model sees what a human would read.
"""
from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Common Cyrillic/Greek letters that look identical to Latin ones.
CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "ѕ": "s", "ԁ": "d", "ӏ": "l",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
})
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

# Emoji that carry a threat or insult when sent as a payment reference.
# Mapped to plain words the text model already knows.
EMOJI_WORDS = {
    "🔪": " knife kill ", "🗡": " knife kill ", "🔫": " gun shoot ", "💣": " bomb ", "🧨": " bomb ",
    "💀": " dead ", "☠": " dead ", "⚰": " coffin dead ", "🪦": " grave dead ", "🩸": " blood ",
    "🖕": " fuck you ", "🤬": " swearing angry ", "😡": " angry ", "👀": " watching you ",
    "🐷": " pig ", "🤡": " clown ", "🐍": " snake ", "💩": " shit ",
}
EMOJI_RE = re.compile("|".join(map(re.escape, EMOJI_WORDS)))

# "k.i.l.l", "d-i-e", "y_o_u": letters split by punctuation (3+ letters).
DOTTED = re.compile(r"\b(?:[A-Za-z][.\-_*~|]){2,}[A-Za-z]\b")
# "k i l l": letters split by single spaces (4+ letters, so "I am a" is untouched).
SPACED = re.compile(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b")
LEET_WORD = re.compile(r"\b(?=[\w@$!]*[a-z])(?=[\w@$!]*[0-9@$!])[\w@$!]{3,}\b", re.I)
REPEATS = re.compile(r"(.)\1{2,}")


def canonicalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).translate(CONFUSABLES)
    t = EMOJI_RE.sub(lambda m: EMOJI_WORDS[m.group(0)], t)
    t = DOTTED.sub(lambda m: re.sub(r"[.\-_*~|]", "", m.group(0)), t)
    t = SPACED.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = LEET_WORD.sub(lambda m: m.group(0).translate(LEET), t)
    t = REPEATS.sub(r"\1\1", t)                       # "diiiie" -> "diie"
    return re.sub(r"\s+", " ", t).strip().lower()


def obfuscation_signals(text: str) -> dict[str, int]:
    """How many evasion tricks a reference uses. Their presence is itself a signal."""
    nfkc = unicodedata.normalize("NFKC", text)
    return {
        "zero_width": sum(text.count(ch) for ch in "​‌‍⁠﻿"),
        "lookalike_letters": sum(1 for ch in nfkc if ch in "аеорсухіјкмтвнѕԁӏαορνικτυ"),
        "split_letters": len(DOTTED.findall(nfkc)) + len(SPACED.findall(nfkc)),
        "leet_words": sum(1 for m in LEET_WORD.finditer(nfkc) if not m.group(0).isdigit()),
        "threat_emoji": len(EMOJI_RE.findall(nfkc)),
    }
