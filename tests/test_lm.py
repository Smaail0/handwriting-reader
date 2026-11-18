"""Character LM and beam-search fusion."""

import math

import numpy as np

from htr.charset import Charset
from htr.decode import beam_search
from htr.lm import EOS, CharNgramLM, LMScorer, MixLM


def _lm():
    return CharNgramLM.train(["access-list 2", "access-group 2 out", "permit any", "deny any"] * 5, order=4)


def test_probabilities_sum_to_one():
    lm = _lm()
    for hist in ["", "acc", "permit a", "zzz"]:
        total = sum(math.exp(lm.logp(hist, c)) for c in list(lm.vocab) + [EOS])
        assert abs(total - 1.0) < 1e-9, (hist, total)


def test_mix_sums_to_one_and_save_load(tmp_path):
    a, b = _lm(), CharNgramLM.train(["interface g0/0/1"], order=3)
    mix = MixLM([(a, 0.7), (b, 0.3)])
    vocab = set(a.vocab) | set(b.vocab)
    # Each part only spreads mass over its own vocabulary + EOS, so the mix sums to 1 too.
    for hist in ["", "acc", "inter"]:
        total = sum(math.exp(m.logp(hist, c)) * w for m, w in mix.parts for c in list(m.vocab) + [EOS])
        assert abs(total - 1.0) < 1e-9
    a.save(tmp_path / "lm.json")
    a2 = CharNgramLM.load(tmp_path / "lm.json")
    assert abs(a2.logp("acc", "e") - a.logp("acc", "e")) < 1e-12
    assert vocab  # silence unused


def _logits(charset: Charset, frames: list[dict[str, float]]) -> np.ndarray:
    """Frames of {char: prob}; the rest of the mass goes to blank."""
    out = np.full((len(frames), len(charset.chars) + 1), 1e-6)
    for t, f in enumerate(frames):
        for ch, p in f.items():
            out[t, charset.chars.index(ch) + 1] = p
        out[t, 0] = max(1e-6, 1 - sum(f.values()))
    return np.log(out)


def test_no_lm_is_unchanged_and_lm_fixes_ambiguity():
    cs = Charset("adeinsy ")
    # "deny": the image is ambiguous between "y" and "s" at the end, slightly favouring "s".
    frames = [{"d": 0.9}, {}, {"e": 0.9}, {}, {"n": 0.9}, {}, {"s": 0.5, "y": 0.45}, {}]
    lg = _logits(cs, frames)
    assert beam_search(lg, cs, beam=8).text == "dens"
    lm = LMScorer(CharNgramLM.train(["deny", "deny any", "any"] * 3, order=4), alpha=1.0, beta=0.0)
    assert beam_search(lg, cs, beam=8, lm=lm).text == "deny"
