"""Text normalization helpers for LLM output and reports."""
from __future__ import annotations

import re
import unicodedata

MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "鈮?": ">=",
    "鈮?.": ">=.",
    "鈥?": "-",
    "鈥檚": "'s",
    "鈥檛": "'t",
    "鈥檙": "'r",
    "鈥檝": "'v",
    "鈥檓": "'m",
    "鈥檒": "'l",
    "鈥檇": "'d",
    "鈥慳": "-a",
    "鈥慛": "-N",
    "鈥憇": "-s",
    "鈥憆": "-r",
    "鈥": "-",
    "檚": "'s",
    "檛": "'t",
    "檙": "'r",
    "檝": "'v",
    "檓": "'m",
    "檒": "'l",
    "檇": "'d",
    "慳": "a",
    "慛": "N",
    "憇": "s",
    "憆": "r",
}

ASCII_PUNCTUATION = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
}

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str | None) -> str:
    """Return report-safe UTF-8 text with stable ASCII punctuation.

    The goal is not to transliterate all Unicode. It repairs common UTF-8
    mojibake, normalizes smart punctuation that often breaks on Windows
    terminals, and removes invalid control characters.
    """
    if text is None:
        return ""
    cleaned = str(text)
    for _ in range(2):
        repaired = _best_mojibake_repair(cleaned)
        if repaired == cleaned:
            break
        cleaned = repaired
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, replacement)
    cleaned = cleaned.translate(str.maketrans(ASCII_PUNCTUATION))
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = CONTROL_CHARS.sub("", cleaned)
    return cleaned


def normalize_plan_text(text: str | None) -> str | None:
    """Normalize a single plan text value; returns None for None input."""
    if text is None:
        return None
    return normalize_text(text)


def normalize_text_list(items: list[str]) -> list[str]:
    """Normalize every string in a list through normalize_text."""
    return [normalize_text(item) for item in items]


def _best_mojibake_repair(text: str) -> str:
    """Choose the best common mojibake repair."""
    candidates = [text]
    for encoding in ("latin1", "cp1252", "gb18030"):
        try:
            candidates.append(text.encode(encoding, errors="strict").decode("utf-8", errors="strict"))
        except UnicodeError:
            pass
    return min(candidates, key=_mojibake_score)


def _mojibake_score(text: str) -> int:
    """Score text for likely mojibake artifacts."""
    markers = ("鈥", "檚", "慳", "慛", "憇", "憆", "â", "Ã", "�")
    return sum(text.count(marker) for marker in markers)
