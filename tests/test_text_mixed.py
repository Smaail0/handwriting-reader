"""Regression tests for mixed-script (Arabic + Latin) ordering and synthetic line labels.

Runs with pytest, or directly:  .venv/bin/python tests/test_text_mixed.py
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np

from htr.text import from_visual, normalize, to_visual

ROOT = Path(__file__).resolve().parents[1]
AR_WORDS = ["المبلغ", "دينار", "بتاريخ", "اليوم", "شمشم", "بغـلس", "له", "و", "النص", "تو", "علاش", "ندكي", "هكّا"]
LA_WORDS = ["facture", "n°", "contrat", "d'assurance", "Rue", "Gambetta", "METZ", "résilier", "été", "OK", "Ahmed"]
NUMS = ["12", "1 250,500", "12/03/2024", "3000", "27 456 789", "15.5%", "DP36"]
PUNCT = ["", "", "", ",", ".", ":", "،", "؟", "!"]


def random_line(rng: random.Random) -> str:
    words = []
    for _ in range(rng.randint(1, 9)):
        pool = rng.choice([AR_WORDS, AR_WORDS, LA_WORDS, NUMS])
        words.append(rng.choice(pool) + rng.choice(PUNCT))
    return normalize(" ".join(words), "mixed")


def test_round_trip(n: int = 3000) -> None:
    rng = random.Random(0)
    bad = [s for s in (random_line(rng) for _ in range(n)) if from_visual(to_visual(s)) != s]
    assert not bad, f"{len(bad)} of {n} fail, e.g. {bad[:3]}"


def test_js_matches_python(n: int = 400) -> None:
    rng = random.Random(1)
    lines = [random_line(rng) for _ in range(n)]
    visuals = [to_visual(s) for s in lines]
    script = ("import('./web/src/text.js').then(m => { const v = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              " console.log(JSON.stringify(v.map(m.fromVisual))); })")
    out = subprocess.run(["node", "-e", script], input=json.dumps(visuals), capture_output=True, text=True,
                         cwd=ROOT, check=True)
    js = json.loads(out.stdout)
    bad = [(v, a, b) for v, a, b in zip(visuals, js, (from_visual(v) for v in visuals)) if a != b]
    assert not bad, f"{len(bad)} of {n} differ, e.g. {bad[:2]}"


def test_synthetic_labels_match_layout(n: int = 500) -> None:
    """The image is laid out token by token (compose_tokens); the target is to_visual(logical).
    They must agree, or the model learns wrong labels."""
    from htr.mixed import MixedLineGenerator, WordBank, compose_tokens
    from htr.synth import DigitBank

    gen = MixedLineGenerator(WordBank(ROOT / "data/words", "validation"),
                             DigitBank(ROOT / "data/raw/mnist/train.parquet", 50000, 52000))
    rng = np.random.default_rng(0)
    bad = []
    for _ in range(n):
        seq = gen.logical_tokens(rng)
        order = compose_tokens([t for t, _, _ in seq])
        if order is None:
            continue  # the generator redraws these
        layout = " ".join(to_visual(seq[k][0]) for k in order)
        target = to_visual(" ".join(t for t, _, _ in seq))
        if layout != target:
            bad.append((" ".join(t for t, _, _ in seq), layout, target))
    assert not bad, f"{len(bad)} of {n} mismatch, e.g. {bad[:2]}"


def test_layout_fuzz_with_punctuation_tokens(n: int = 20000) -> None:
    """Reviewer's fuzz: stand-alone punctuation tokens at script boundaries. Whatever
    compose_tokens accepts must reproduce the target exactly."""
    from htr.mixed import compose_tokens

    rng = random.Random(2)
    extra = [",", ".", "-", "(", ")", '"', "50%", "é", "كتاب", "سلام"]
    accepted = 0
    for _ in range(n):
        toks = [rng.choice(AR_WORDS + LA_WORDS + NUMS + extra) for _ in range(rng.randint(1, 8))]
        order = compose_tokens(toks)
        if order is None:
            continue
        accepted += 1
        assert " ".join(to_visual(toks[k]) for k in order) == to_visual(" ".join(toks)), toks
    assert accepted > 0.5 * n, f"only {accepted} of {n} accepted"


def test_layout_rejects_split_tokens() -> None:
    """Tokens that mix scripts inside one word cannot be drawn as one piece in that line:
    compose_tokens must refuse them (the generator then draws a new line)."""
    from htr.mixed import compose_tokens

    assert compose_tokens(["سلامabc", "سلامabc", "abcسلام", "é"]) is None
    assert compose_tokens(["facture", "سلام"]) is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
