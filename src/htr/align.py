"""CTC forced alignment: where each character of a known transcript sits in a line image.

Used to cut labelled *word* images out of labelled *line* datasets (IAM, RIMES, KHATT),
so words from different scripts and writers can be recombined into synthetic lines
(e.g. Arabic with French insertions, which no public dataset contains).
"""

from __future__ import annotations

import unicodedata

import numpy as np
from PIL import Image

from .decode import log_softmax
from .inference import OnnxRecognizer
from .preprocess import DOWNSAMPLE
from .text import to_target


def viterbi(lp: np.ndarray, target: list[int]) -> np.ndarray | None:
    """Best CTC path for `target` through log-probs lp (T, C).

    Returns, for each target position, [first_frame, last_frame] (inclusive), or None if
    the line is too short for the label."""
    t_len, s = lp.shape[0], 2 * len(target) + 1
    if t_len < len(target):
        return None
    ext = np.zeros(s, dtype=np.int64)
    ext[1::2] = target
    # A label may skip the blank before it unless it repeats the previous label.
    skip = np.zeros(s, dtype=bool)
    skip[3::2] = np.asarray(target[1:]) != np.asarray(target[:-1])
    neg = -1e30
    score = np.full(s, neg)
    score[0] = lp[0, ext[0]]
    if s > 1:
        score[1] = lp[0, ext[1]]
    back = np.zeros((t_len, s), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, t_len):
        stay = score
        prev1 = np.concatenate(([neg], score[:-1]))
        prev2 = np.where(skip, np.concatenate(([neg, neg], score[:-2])), neg)
        cand = np.stack([stay, prev1, prev2])
        arg = cand.argmax(0)
        back[t] = arg
        score = cand[arg, np.arange(s)] + lp[t, ext]
    end = s - 1 if s == 1 or score[s - 1] >= score[s - 2] else s - 2
    if score[end] <= neg / 2:
        return None
    spans = np.full((len(target), 2), -1, dtype=np.int64)
    pos = end
    for t in range(t_len - 1, -1, -1):
        if pos % 2 == 1:
            k = pos // 2
            spans[k, 0] = t
            if spans[k, 1] < 0:
                spans[k, 1] = t
        pos -= int(back[t, pos])
    return spans if (spans >= 0).all() else None


def fold(text: str, chars: str) -> str:
    """Map each character to one the aligning model knows (é -> e, œ -> o, € -> E ...).

    Keeps the length, so positions in the folded string match the original."""
    known = set(chars)
    out = []
    for c in text:
        if c in known:
            out.append(c)
            continue
        base = unicodedata.normalize("NFD", c)[0]
        if base in known:
            out.append(base)
        elif c.lower() in known:
            out.append(c.lower())
        else:
            out.append({"œ": "o", "Œ": "O", "æ": "a", "€": "E", "°": "o", "«": '"', "»": '"', "’": "'"}.get(c, "."))
    return "".join(c if c in known else "." for c in out)


def _ink_columns(arr: np.ndarray) -> np.ndarray:
    """Per-column ink amount of a preprocessed line (ink ~1), lightly smoothed."""
    col = (arr > 0.35).sum(0).astype(np.float64)
    return np.convolve(col, np.ones(3) / 3, mode="same")


def _snap(col: np.ndarray, x: float, radius: int) -> int:
    """Move a cut to the centre of the emptiest stretch near x."""
    lo, hi = max(1, int(x) - radius), min(len(col) - 1, int(x) + radius + 1)
    if hi <= lo:
        return int(x)
    window = col[lo:hi]
    best = window.min()
    idx = np.nonzero(window <= best + 0.5)[0]
    # Longest run of near-minimal columns; ties go to the run closest to x.
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    run = max(runs, key=lambda r: (len(r), -abs(lo + (r[0] + r[-1]) / 2 - x)))
    return int(lo + (run[0] + run[-1]) // 2)


def cut_words(img: Image.Image, text: str, rec: OnnxRecognizer, script: str,
              verify_cer: float | None = 0.25) -> list[tuple[Image.Image, str]] | None:
    """Split a labelled line into (word image, word text) pairs via forced alignment.

    CTC models may emit a character a few frames before or after its ink, so emission
    positions are mapped linearly onto the line's ink extent, then each cut snaps to the
    nearest blank stretch. With `verify_cer`, every word crop is re-read and kept only if
    the reading is within that CER of its label. Word images keep the line's height and
    original orientation; returns None if alignment fails."""
    from .metrics import cer

    target_text = to_target(text, script)  # the order the model emits characters in
    folded = fold(target_text, rec.charset.chars)
    ids = rec.charset.encode(folded)
    if len(ids) != len(folded) or not folded.strip():
        return None
    x = rec.prepare(img)
    lp = log_softmax(rec.logits(x).astype(np.float64))
    spans = viterbi(lp, ids)
    if spans is None:
        return None
    words, start = [], None
    for i, c in enumerate(target_text + " "):
        if c != " " and start is None:
            start = i
        elif c == " " and start is not None:
            words.append((start, i - 1))
            start = None
    if not words:
        return None
    arr = x[0, 0]
    col = _ink_columns(arr)
    inked = np.nonzero(col > 0.5)[0]
    if len(inked) < 8:
        return None
    ink0, ink1 = inked[0], inked[-1]
    e0 = spans[words[0][0], 0] * DOWNSAMPLE
    e1 = (spans[words[-1][1], 1] + 1) * DOWNSAMPLE
    if e1 <= e0:
        return None
    to_col = lambda f: ink0 + (f * DOWNSAMPLE - e0) * (ink1 - ink0) / (e1 - e0)  # noqa: E731
    radius = max(6, int(0.4 * arr.shape[0]))
    starts = [to_col(spans[a, 0]) for a, _ in words]
    ends = [to_col(spans[b, 1] + 1) for _, b in words]
    cuts = [max(0, ink0 - 2)]
    for k in range(len(words) - 1):
        mid = (ends[k] + starts[k + 1]) / 2
        # Never snap further than ~40% into either neighbouring word.
        r = int(min(radius, 0.4 * (ends[k] - starts[k]), 0.4 * (ends[k + 1] - starts[k + 1])))
        snapped = _snap(col, mid, max(2, r))
        cuts.append(snapped if snapped > cuts[-1] else int(mid))
    cuts.append(min(arr.shape[1], ink1 + 3))
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        return None
    w_img = img.size[0]
    prep_w = min(2048, max(DOWNSAMPLE * 4, round(img.size[0] * arr.shape[0] / img.size[1])))  # as in preprocess()
    scale = w_img / prep_w
    out = []
    for (a, b), (c0, c1) in zip(words, zip(cuts, cuts[1:])):
        word = to_target(target_text[a: b + 1], script)  # its own inverse: back to logical order
        if verify_cer is not None:
            piece = np.ascontiguousarray(arr[:, c0:c1])
            piece = np.pad(piece, ((0, 0), (8, 8 + (-(c1 - c0 + 16)) % DOWNSAMPLE)))
            read = rec.run(piece[None, None].astype(np.float32))
            if cer([read], [fold(word, rec.charset.chars)]) > verify_cer:
                continue
        o0, o1 = int(c0 * scale), int(c1 * scale)
        if script == "arabic":  # the model read the mirrored line
            o0, o1 = w_img - o1, w_img - o0
        if o1 - o0 >= 4:
            out.append((img.crop((max(0, o0), 0, min(w_img, o1), img.size[1])), word))
    return out
