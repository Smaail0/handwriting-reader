"""Character and word error rates (corpus-level: total edits / total reference length)."""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein


def cer(preds: Sequence[str], refs: Sequence[str]) -> float:
    edits = sum(Levenshtein.distance(p, r) for p, r in zip(preds, refs))
    return edits / max(1, sum(len(r) for r in refs))


def wer(preds: Sequence[str], refs: Sequence[str]) -> float:
    edits = sum(Levenshtein.distance(p.split(), r.split()) for p, r in zip(preds, refs))
    return edits / max(1, sum(len(r.split()) for r in refs))
