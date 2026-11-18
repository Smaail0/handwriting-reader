"""Character n-gram language models for CTC beam search (shallow fusion), plus domain lists.

Why characters, not words: the recognizers emit characters, handwriting has many
out-of-vocabulary tokens (names, codes, IP addresses), and a character model still rewards
"access-list" over "auess lest" without rejecting words it has never seen. Interpolated
Witten-Bell smoothing, no dependencies, small enough to ship (and to port to the browser).

The model works in the recognizer's OUTPUT order (htr.text.to_target), so it can score
prefixes during the beam search as they are emitted.

Domain lists ("lexicons/cisco_ios.txt", a list of names, a field's allowed words) are plain
text, one phrase per line. They become a second model interpolated with the base one:
    P(c | h) = (1 - w) * P_base(c | h) + w * P_domain(c | h)
so the decoder prefers domain words where the image allows it, without being forced to them.

  python scripts/build_lm.py --data-dir data/raw/iam_line --script latin --out models/lm_en.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .text import normalize, to_target

BOS, EOS = "\x02", "\x03"


class CharNgramLM:
    def __init__(self, order: int, counts: list[dict[str, dict[str, int]]], vocab: str):
        self.order = order
        self.counts = counts  # counts[k][context of length k][next char] = n
        self.totals = [{ctx: sum(d.values()) for ctx, d in level.items()} for level in counts]
        self.vocab = vocab
        self.base = 1.0 / (len(vocab) + 1)  # uniform over vocab + EOS
        self._logp = lru_cache(maxsize=1 << 18)(self._logp_uncached)

    @classmethod
    def train(cls, texts, order: int = 6) -> CharNgramLM:
        """texts: strings already in output order (htr.text.to_target)."""
        counts: list[dict[str, dict[str, int]]] = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        vocab = set()
        for t in texts:
            if not t:
                continue
            vocab.update(t)
            s = BOS * (order - 1) + t + EOS
            for i in range(order - 1, len(s)):
                c = s[i]
                for k in range(order):
                    counts[k][s[i - k:i]][c] += 1
        plain = [{ctx: dict(d) for ctx, d in level.items()} for level in counts]
        return cls(order, plain, "".join(sorted(vocab)))

    def _logp_uncached(self, history: str, c: str) -> float:
        h = (BOS * (self.order - 1) + history)[-(self.order - 1):] if self.order > 1 else ""
        p = self.base
        for k in range(self.order):  # Witten-Bell, interpolated from short to long contexts
            ctx = h[len(h) - k:] if k else ""
            d = self.counts[k].get(ctx)
            if d is None:
                break  # longer contexts were not seen either
            tot, distinct = self.totals[k][ctx], len(d)
            p = (d.get(c, 0) + distinct * p) / (tot + distinct)
        return math.log(p)

    def logp(self, history: str, c: str) -> float:
        """log P(c | history); c may be EOS."""
        n = self.order - 1
        return self._logp(history[-n:] if n else "", c)

    def text_logp(self, text: str) -> float:
        return sum(self.logp(text[:i], c) for i, c in enumerate(text + EOS))

    # --------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"order": self.order, "vocab": self.vocab, "counts": self.counts},
                                         ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharNgramLM:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["order"], d["counts"], d["vocab"])


class MixLM:
    """Linear interpolation of character models: [(lm, weight), ...]; weights sum to 1."""

    def __init__(self, parts: list[tuple[CharNgramLM, float]]):
        total = sum(w for _, w in parts)
        self.parts = [(m, w / total) for m, w in parts]
        self.order = max(m.order for m, _ in parts)
        self._logp = lru_cache(maxsize=1 << 18)(self._logp_uncached)

    def _logp_uncached(self, history: str, c: str) -> float:
        return math.log(sum(w * math.exp(m.logp(history, c)) for m, w in self.parts))

    def logp(self, history: str, c: str) -> float:
        return self._logp(history[-(self.order - 1):], c)


def read_lexicon(path: Path, script: str) -> list[str]:
    """One phrase per line; '#' starts a comment line. Returned in output order."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(to_target(normalize(ln, script), script))
    return out


def domain_lm(lexicons: list[Path], script: str, order: int = 6, repeat: int = 3) -> CharNgramLM:
    """A model of the domain phrases, alone and inside short random sequences of each other
    (so word boundaries between domain words are modelled too)."""
    import random

    phrases = [p for f in lexicons for p in read_lexicon(f, script)]
    rng = random.Random(0)
    texts = phrases * repeat + [" ".join(rng.sample(phrases, min(len(phrases), rng.randint(2, 5))))
                                for _ in range(len(phrases) * repeat)]
    return CharNgramLM.train(texts, order)


@dataclass
class LMScorer:
    """What the beam search needs: a model, its weight (alpha) and a per-character bonus
    (beta) that offsets the model's preference for short outputs."""
    lm: CharNgramLM | MixLM
    alpha: float = 0.5
    beta: float = 1.0

    @classmethod
    def build(cls, base: Path | None, lexicons: list[Path] | None = None, script: str = "latin",
              lexicon_weight: float = 0.3, alpha: float = 0.5, beta: float = 1.0) -> LMScorer:
        parts = []
        if base:
            parts.append((CharNgramLM.load(base), 1.0 - (lexicon_weight if lexicons else 0.0)))
        if lexicons:
            parts.append((domain_lm(lexicons, script), lexicon_weight if base else 1.0))
        if not parts:
            raise ValueError("need a base model or a lexicon")
        return cls(parts[0][0] if len(parts) == 1 else MixLM(parts), alpha, beta)
