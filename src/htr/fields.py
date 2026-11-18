"""Typed form fields: patterns for constrained decoding and value normalization.

Each field type maps to a regular expression over the recognizer's charset. Decoding is
restricted to strings that match (see decode.beam_search), and the raw text is then
normalized: dates to ISO 8601, amounts to a plain decimal string, and so on.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

import regex  # supports partial matching, needed to test whether a prefix can still match

PATTERNS: dict[str, str | None] = {
    "text": None,  # free text, but no letter-digit mixing inside a word (see WordsConstraint)
    "free": None,  # completely unconstrained
    "check": None,  # checkbox: ticked or not, decided from the amount of ink (no reading)
    "digits": r"[0-9](?: ?[0-9])*",  # digits written in boxes read with gaps
    "number": r"[0-9]+(?:[.,][0-9]+)?",
    # 1250 | 1250.00 | 1,250.00 | 1.250,00 | 1 250,00 | 35.5
    "amount": r"[0-9]{1,3}(?:[ .,]?[0-9]{3})*(?:[.,][0-9]{1,2})?",
    # 15/03/2024 | 5.3.24 | 15-03-2024 | 11 / 05 / 2003 (day first)
    "date": r"(?:0?[1-9]|[12][0-9]|3[01]) ?[/.-] ?(?:0?[1-9]|1[0-2]) ?[/.-] ?(?:[0-9]{2}|[0-9]{4})",
    "phone": r"\+?[0-9](?:[ .-]?[0-9]){5,14}",
    # invoice / reference / order numbers: INV-2024/0153, A12345
    "code": r"[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)*",
    "name": r"[A-Za-z][A-Za-z'.-]*(?: [A-Za-z][A-Za-z'.-]*)*",
}


class RegexConstraint:
    def __init__(self, pattern: str):
        self.rx = regex.compile(pattern)

    def viable(self, text: str) -> bool:
        return self.rx.fullmatch(text, partial=True) is not None

    def complete(self, text: str) -> bool:
        return self.rx.fullmatch(text) is not None


class WordsConstraint:
    """Prose: no digit right after a letter inside a word ("b4", "e1", "l0" are misreads).

    Handwritten 1/l, 0/o, 5/s look alike, so a model that has seen many digits slips them
    into words. A digit followed by letters stays allowed: ordinals and units ("1st",
    "2ème", "5kg", "15h30"). Codes that put digits after letters ("A4", "H2O", "DP36")
    should use the "code" type."""

    _mix = regex.compile(r"[^\W\d_][0-9]")
    _time = regex.compile(r"(?<=[0-9])h(?=[0-9])")  # "15h30" is a time, not a misread

    def viable(self, text: str) -> bool:
        return self._mix.search(self._time.sub("", text)) is None

    complete = viable


class LexiconConstraint:
    """Value must be one of a list (cities, product codes, customer names...)."""

    def __init__(self, words: list[str]):
        self.words = set(words)
        self.prefixes = {w[:i] for w in words for i in range(len(w) + 1)}

    def viable(self, text: str) -> bool:
        return text in self.prefixes

    def complete(self, text: str) -> bool:
        return text in self.words


def constraint_for(field_type: str, lexicon: list[str] | None = None):
    if lexicon:
        return LexiconConstraint(lexicon)
    if field_type not in PATTERNS:
        raise ValueError(f"unknown field type {field_type!r}; known: {sorted(PATTERNS)}")
    if field_type == "text":
        return WordsConstraint()
    if field_type == "check":
        return None
    pattern = PATTERNS[field_type]
    return RegexConstraint(pattern) if pattern else None


def normalize(field_type: str, text: str) -> str | None:
    """Canonical value, or None if the text cannot be interpreted as this type."""
    t = text.strip()
    if field_type == "date":
        m = re.fullmatch(r"(\d{1,2}) ?[/.-] ?(\d{1,2}) ?[/.-] ?(\d{2}|\d{4})", t)
        if not m:
            return None
        d, mo, y = int(m[1]), int(m[2]), m[3]
        year = int(y)
        if len(y) == 2:  # sliding window: from 79 years ago to 20 years ahead of today
            now = _dt.date.today().year
            year += (now // 100) * 100
            if year > now + 20:
                year -= 100
            elif year <= now - 80:
                year += 100
        try:
            return _dt.date(year, mo, d).isoformat()  # also rejects 31/02 etc.
        except ValueError:
            return None
    if field_type == "amount":
        digits = re.sub(r"[ .,]", "", t)
        if not digits.isdigit():
            return None
        # A final separator followed by 1-2 digits is the decimal mark ("1.250,00", "35.5").
        m = re.search(r"[.,](\d{1,2})$", t)
        if m:
            whole = re.sub(r"[ .,]", "", t[: m.start()])
            return f"{int(whole or '0')}.{m[1].ljust(2, '0')}"
        return f"{int(digits)}.00"
    if field_type in ("digits", "phone"):
        return re.sub(r"[^0-9+]", "", t) or None
    return t or None


@dataclass
class FieldSpec:
    name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in template pixel coordinates
    type: str = "text"
    script: str = "latin"
    multiline: bool = False
    lexicon: list[str] | None = None
