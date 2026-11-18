"""Synthetic code-switched handwritten lines (Arabic with French/English, and vice versa).

No public dataset has handwritten lines that mix Arabic and Latin script, which is common
in Maghreb documents ("المبلغ 1 250,500 د حسب facture n° 125"). This module recombines
real handwritten *words* cut from IAM, RIMES and KHATT (scripts/build_word_bank.py) and
handwritten MNIST numbers into such lines:

- runs of consecutive words are taken from the same source line, so short phrases keep
  natural word order and one writer's style;
- a base-language run gets 1-2 insertions from another language, and sometimes a number
  (date, amount, phone, reference);
- tokens are laid out in visual order with the same mini-bidi rules as htr.text, so the
  image and its visual-order target always agree.

Lines are rendered on the fly, so every epoch sees new combinations.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .charset import Charset
from .data import augment_geometric, augment_photometric
from .preprocess import DEFAULT_HEIGHT, preprocess
from .synth import DigitBank, random_field_text, render_field
from .text import _ARABIC_LETTER, base_is_rtl, normalize, to_visual, visual_permutation

STRIP = 96  # word strips are stored at this height

# (base source, inserted sources, probability). Weighted toward what Maghreb documents mix.
MIXES = [
    ("khatt", ("rimes",), 0.40),
    ("rimes", ("khatt",), 0.25),
    ("khatt", ("iam",), 0.10),
    ("rimes", ("iam",), 0.05),
    ("iam", ("khatt",), 0.05),
    ("rimes", ("rimes",), 0.05),  # plain French lines with numbers
    ("khatt", ("khatt",), 0.10),  # plain Arabic lines with numbers
]


class WordBank:
    """Word images grouped into phrases (consecutive words of one source line).

    Images stay in memory-mapped Arrow files (converted once from the parquet bank), so
    data-loader worker processes share one copy through the OS page cache instead of
    each unpickling ~1 GB of PNG bytes."""

    def __init__(self, words_dir: Path, split: str, sources=("iam", "rimes", "khatt")):
        self.words_dir, self.split, self.sources = Path(words_dir), split, tuple(sources)
        self._open()

    def _open(self) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        self.images, self.phrases, self.n_words = {}, {}, {}
        for src in self.sources:
            pq_path = self.words_dir / f"{src}_{self.split}.parquet"
            arrow_path = pq_path.with_suffix(".arrow")
            if not arrow_path.exists() or arrow_path.stat().st_mtime < pq_path.stat().st_mtime:
                table = pq.read_table(pq_path, columns=["image", "text", "line", "pos"])
                tmp = arrow_path.with_suffix(".arrow.tmp")
                with pa.OSFile(str(tmp), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
                tmp.rename(arrow_path)
            table = ipc.open_file(pa.memory_map(str(arrow_path), "r")).read_all()
            self.images[src] = table.column("image")
            texts = [normalize(t, "mixed") for t in table.column("text").to_pylist()]
            lines = table.column("line").to_pylist()
            pos = table.column("pos").to_pylist()
            by_line = defaultdict(list)
            for i, ln in enumerate(lines):
                by_line[ln].append(i)
            phrases = []
            for rows in by_line.values():
                rows.sort(key=lambda i: pos[i])
                cur = [rows[0]]
                for i in rows[1:]:  # split where verification dropped a word
                    if pos[i] == pos[cur[-1]] + 1:
                        cur.append(i)
                    else:
                        phrases.append(cur)
                        cur = [i]
                phrases.append(cur)
            self.phrases[src] = [[(i, texts[i]) for i in ph] for ph in phrases]
            self.n_words[src] = len(texts)

    def __getstate__(self):
        return {"words_dir": self.words_dir, "split": self.split, "sources": self.sources}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def run(self, src: str, n: int, rng: np.random.Generator) -> list[tuple[bytes, str]]:
        """Up to n consecutive words (image bytes, text) from one random phrase."""
        pool = self.phrases[src]
        for _ in range(8):  # prefer phrases long enough
            ph = pool[int(rng.integers(len(pool)))]
            if len(ph) >= n:
                break
        start = int(rng.integers(0, max(1, len(ph) - n + 1)))
        return [(self.images[src][i].as_py(), t) for i, t in ph[start: start + n]]


def _is_rtl_token(text: str) -> bool:
    return _ARABIC_LETTER.search(text) is not None


def _word_image(data: bytes) -> np.ndarray:
    """Word strip as float ink map (ink ~1), contrast-normalized."""
    img = ImageOps.autocontrast(Image.open(io.BytesIO(data)).convert("L"), cutoff=1)
    arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    bg = float(np.median(arr))  # paper level (IAM paper is grey): pasted words must not show a box
    return np.clip((arr - bg - 0.04) / max(0.2, 1.0 - bg - 0.04), 0.0, 1.0)


def _normalize_ink(arr: np.ndarray, body: float) -> tuple[np.ndarray, float]:
    """Scale a token so its core ink band is `body` px tall; crop far-away fragments.

    Words come from line strips of three datasets whose ink fills different fractions of
    the strip. The core band (10th-90th percentile of ink rows) is a robust stand-in for
    the x-height. Returns (array, row of the band centre)."""
    rows = np.nonzero(arr > 0.3)
    if len(rows[0]) < 10:
        return arr, arr.shape[0] / 2
    lo, hi = np.percentile(rows[0], [10, 90])
    band = max(4.0, hi - lo)
    y0 = int(max(0, lo - 1.3 * band))
    y1 = int(min(arr.shape[0], hi + 1.3 * band + 1))  # ascenders/descenders, not the next line
    cols = np.nonzero(arr[y0:y1].max(0) > 0.3)[0]
    x0, x1 = (cols[0], cols[-1] + 1) if len(cols) else (0, arr.shape[1])
    arr = arr[y0:y1, x0:x1]
    f = body / band
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img = img.resize((max(2, round(arr.shape[1] * f)), max(2, round(arr.shape[0] * f))), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0, ((lo + hi) / 2 - y0) * f


def _number_image(text: str, digits: DigitBank, rng: np.random.Generator) -> np.ndarray:
    img = render_field(text, digits, rng).convert("L")
    w, h = img.size
    img = img.resize((max(4, round(w * STRIP / h)), STRIP), Image.Resampling.BILINEAR)
    arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    cols = np.nonzero(arr.max(0) > 0.3)[0]
    return arr[:, cols[0]: cols[-1] + 1] if len(cols) else arr


def compose_tokens(tokens: list[str]) -> list[int] | None:
    """Visual left-to-right order of logical tokens, derived from htr.text.to_visual itself.

    The line's target is to_visual(" ".join(tokens)); laying the token images out in the order
    their characters appear in that string makes image and target agree by construction.
    Returns None when a token would not stay one contiguous, internally ordered piece (the
    caller then draws a new line), so a mismatch can never reach training."""
    logical = " ".join(tokens)
    perm = visual_permutation(logical)
    pos = [0] * len(logical)
    for v, i in enumerate(perm):
        pos[i] = v
    visual = to_visual(logical)
    starts, k = [], 0
    for t in tokens:
        span = [pos[i] for i in range(k, k + len(t))]
        lo, hi = min(span), max(span)
        if hi - lo + 1 != len(t) or visual[lo: hi + 1] != to_visual(t):
            return None
        starts.append(lo)
        k += len(t) + 1
    return sorted(range(len(tokens)), key=lambda j: starts[j])


def layout_line(pieces: list[tuple[np.ndarray, float]], body: float, rng: np.random.Generator) -> Image.Image:
    """Paste ink maps (each with the row of its core-band centre) left to right on one line."""
    above = max(m for _, m in pieces)
    below = max(a.shape[0] - m for a, m in pieces)
    height = int(above + below) + 12
    gaps = [int(body * rng.uniform(0.6, 1.6)) for _ in pieces]
    width = sum(a.shape[1] for a, _ in pieces) + sum(gaps) + 16
    canvas = np.zeros((height, width), np.float32)
    x = 8
    for (a, m), g in zip(pieces, gaps):
        y = int(np.clip(6 + above - m + rng.normal(0, body * 0.08), 0, height - a.shape[0]))
        region = canvas[y: y + a.shape[0], x: x + a.shape[1]]
        np.maximum(region, a, out=region)
        x += a.shape[1] + g
    canvas = canvas[:, : max(16, x - gaps[-1] + 8)]
    return Image.fromarray(((1.0 - canvas) * 255).astype(np.uint8))


class MixedLineGenerator:
    def __init__(self, bank: WordBank, digits: DigitBank | None):
        self.bank = bank
        self.digits = digits
        probs = np.array([m[2] for m in MIXES])
        self.probs = probs / probs.sum()

    def logical_tokens(self, rng: np.random.Generator):
        """[(text, image, is_rtl)] in logical (reading) order."""
        base, inserts, _ = MIXES[int(rng.choice(len(MIXES), p=self.probs))]
        n_base = int(rng.integers(2, 9))
        seq = [(t, b, _is_rtl_token(t)) for b, t in self.bank.run(base, n_base, rng)]
        n_ins = 1 if rng.random() < 0.7 else 2
        if inserts[0] != base:
            for _ in range(n_ins):
                src = inserts[int(rng.integers(len(inserts)))]
                piece = [(t, b, _is_rtl_token(t)) for b, t in self.bank.run(src, int(rng.integers(1, 5)), rng)]
                at = int(rng.integers(0, len(seq) + 1))
                seq[at:at] = piece
        if self.digits is not None and rng.random() < 0.45:
            kind = str(rng.choice(["date", "amount", "digits", "phone"], p=[0.3, 0.4, 0.2, 0.1]))
            num = random_field_text(kind, rng)
            at = int(rng.integers(0, len(seq) + 1))
            seq.insert(at, (num, ("num", num), False))
        return seq

    def __call__(self, rng: np.random.Generator) -> tuple[Image.Image, str]:
        """(line image, logical text)."""
        for _ in range(20):
            seq = self.logical_tokens(rng)
            order = compose_tokens([t for t, _, _ in seq])
            if order is not None:
                break
        else:
            raise RuntimeError("could not lay out a consistent mixed line")
        logical = " ".join(t for t, _, _ in seq)
        body = STRIP * rng.uniform(0.22, 0.32)  # one x-height per line, like one writer's hand
        pieces = []
        for k in order:
            _, src, _ = seq[k]
            arr = _number_image(src[1], self.digits, rng) if isinstance(src, tuple) else _word_image(src)
            arr, mid = _normalize_ink(arr, body * (1.5 if isinstance(src, tuple) else 1.0) * rng.uniform(0.9, 1.1))
            pieces.append((arr, mid))
        img = layout_line(pieces, body, rng)
        return img, logical


class SyntheticMixedDataset(Dataset):
    """On-the-fly code-switched lines for training (script "mixed", visual-order targets)."""

    def __init__(self, gen: MixedLineGenerator, charset: Charset, n: int, seed: int = 0, train: bool = True,
                 height: int = DEFAULT_HEIGHT):
        self.gen, self.charset, self.n, self.seed, self.train, self.height = gen, charset, n, seed, train, height
        self.epoch = 0

    def __len__(self) -> int:
        return self.n

    def widths(self) -> np.ndarray:
        return np.full(self.n, 900)  # rough; lines are generated on the fly

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i) if not self.train else None)
        img, logical = self.gen(rng)
        if self.train:
            arr = augment_photometric(preprocess(augment_geometric(img, rng), self.height), rng)
        else:
            arr = preprocess(img, self.height)
        target = self.charset.encode(to_visual(logical))
        return torch.from_numpy(arr), torch.tensor(target, dtype=torch.long), logical
