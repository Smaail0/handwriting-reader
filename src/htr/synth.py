"""Synthetic handwritten number fields (dates, amounts, IDs, phone numbers).

IAM has very few digits, but invoices and forms are mostly digits. This module composes
field images from real handwritten MNIST digits, with drawn punctuation (. , / - :) and
random size, spacing, baseline jitter, stroke width and slant. MNIST's train and test
writers are disjoint, so a test set built from the test split measures generalization
to unseen handwriting.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .charset import Charset
from .data import augment_geometric, augment_photometric
from .preprocess import DEFAULT_HEIGHT, DOWNSAMPLE, preprocess

FIELD_KINDS = ("date", "amount", "digits", "phone")


class DigitBank:
    """MNIST digits cropped to their ink, white ink on black, grouped by label."""

    def __init__(self, parquet: Path, start: int = 0, stop: int | None = None):
        rows = pq.read_table(parquet).to_pylist()[start:stop]
        self.by_label: list[list[np.ndarray]] = [[] for _ in range(10)]
        for r in rows:
            a = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])))
            ys, xs = np.nonzero(a > 40)
            if len(xs):
                self.by_label[r["label"]].append(a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1])

    def sample(self, digit: int, rng: np.random.Generator) -> np.ndarray:
        pool = self.by_label[digit]
        return pool[int(rng.integers(len(pool)))]


# ------------------------------------------------------------------ text generators


def random_field_text(kind: str, rng: np.random.Generator) -> str:
    if kind == "date":
        d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(1990, 2036))
        sep = str(rng.choice(["/", ".", "-"]))
        yy = f"{y % 100:02d}" if rng.random() < 0.35 else str(y)
        dd = f"{d:02d}" if rng.random() < 0.6 else str(d)
        mm = f"{m:02d}" if rng.random() < 0.6 else str(m)
        return f"{dd}{sep}{mm}{sep}{yy}"
    if kind == "amount":
        value = float(np.exp(rng.uniform(0, np.log(200000))))
        whole, cents = int(value), int(round((value % 1) * 100)) % 100
        style = int(rng.integers(6))
        grouped = f"{whole:,}"
        if style == 0:
            return f"{whole}.{cents:02d}"
        if style == 1:
            return f"{grouped}.{cents:02d}"
        if style == 2:
            return f"{grouped.replace(',', '.')},{cents:02d}"
        if style == 3:
            return f"{grouped.replace(',', ' ')},{cents:02d}"
        if style == 4:
            return f"{whole},{cents:02d}"
        return str(whole)
    if kind == "digits":
        n = int(rng.integers(3, 13))
        return "".join(str(int(d)) for d in rng.integers(0, 10, n))
    if kind == "phone":
        digits = "".join(str(int(d)) for d in rng.integers(0, 10, 10))
        style = int(rng.integers(4))
        if style == 0:
            return " ".join(digits[i: i + 2] for i in range(0, 10, 2))
        if style == 1:
            return ".".join(digits[i: i + 2] for i in range(0, 10, 2))
        if style == 2:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return digits
    raise ValueError(kind)


# ------------------------------------------------------------------ rendering


def render_field(text: str, bank: DigitBank, rng: np.random.Generator) -> Image.Image:
    """Dark ink on white, roughly IAM-like proportions (text ~ half the image height)."""
    h = int(rng.integers(26, 44))  # digit height in pixels
    t = max(2, int(round(h * rng.uniform(0.07, 0.12))))  # stroke width for drawn punctuation
    height = int(h * 2.0)
    base = int(h * 1.45)  # baseline
    width = int(len(text) * h * 1.1 + 4 * h)
    canvas = np.zeros((height, width), np.float32)
    x = h // 2
    for ch in text:
        gap = int(h * rng.uniform(-0.05, 0.22))
        if ch.isdigit():
            g = bank.sample(int(ch), rng)
            gh = max(8, int(h * rng.uniform(0.85, 1.1)))
            gw = max(3, int(g.shape[1] * gh / g.shape[0] * rng.uniform(0.85, 1.15)))
            g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            y0 = base - gh + int(h * rng.uniform(-0.06, 0.06))
            y0 = int(np.clip(y0, 0, height - gh))
            x = max(0, min(x, width - gw - 1))
            region = canvas[y0: y0 + gh, x: x + gw]
            np.maximum(region, g, out=region)
            x += gw + gap
        elif ch == " ":
            x += int(h * rng.uniform(0.35, 0.75))
        else:
            ink = 1.0
            if ch == ".":
                r = max(1, int(h * rng.uniform(0.05, 0.09)))
                cv2.circle(canvas, (x + r, base - r), r, ink, -1, cv2.LINE_AA)
                x += 2 * r + gap + int(h * 0.1)
            elif ch == ",":
                x0 = x + int(h * 0.1)
                cv2.line(canvas, (x0, base - int(h * 0.05)), (x0 - int(h * 0.12), base + int(h * 0.28)), ink, t,
                         cv2.LINE_AA)
                x += int(h * 0.25) + gap
            elif ch == "/":
                w = int(h * rng.uniform(0.08, 0.55))  # from upright to strongly slanted
                cv2.line(canvas, (x, base), (x + w, base - int(h * rng.uniform(0.9, 1.1))), ink, t, cv2.LINE_AA)
                x += w + gap + int(h * 0.1)
            elif ch == "-":
                w = int(h * rng.uniform(0.3, 0.5))
                y = base - int(h * rng.uniform(0.4, 0.55))
                cv2.line(canvas, (x, y), (x + w, y + int(h * rng.uniform(-0.05, 0.05))), ink, t, cv2.LINE_AA)
                x += w + gap + int(h * 0.1)
            elif ch == ":":
                r = max(1, int(h * 0.06))
                cv2.circle(canvas, (x + r, base - r), r, ink, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x + r, base - int(h * 0.6)), r, ink, -1, cv2.LINE_AA)
                x += 2 * r + gap + int(h * 0.1)
            else:
                raise ValueError(f"cannot render {ch!r}")
    cols = np.nonzero(canvas.max(0) > 0.1)[0]
    x0, x1 = (max(0, cols.min() - h // 3), min(width, cols.max() + h // 3)) if len(cols) else (0, h)
    canvas = canvas[:, x0:x1]
    img = Image.fromarray((255 - np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    shear = rng.uniform(-0.3, 0.3)  # handwriting slant
    w, hh = img.size
    nw = w + int(abs(shear) * hh)
    return img.transform((nw, hh), Image.Transform.AFFINE, (1, shear, -max(shear, 0.0) * hh, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=255)


class SyntheticFieldDataset(Dataset):
    """Fixed texts per index; images re-rendered every time (train) or seeded (eval)."""

    def __init__(self, bank: DigitBank, charset: Charset, n: int, seed: int, train: bool,
                 height: int = DEFAULT_HEIGHT, kinds: tuple[str, ...] = FIELD_KINDS):
        rng = np.random.default_rng(seed)
        self.items = [(k, random_field_text(k, rng)) for k in rng.choice(kinds, n)]
        self.bank, self.charset, self.seed, self.train, self.height = bank, charset, seed, train, height
        missing = {c for _, t in self.items for c in t} - set(charset.chars)
        if missing:
            raise ValueError(f"charset lacks {missing}")

    def __len__(self) -> int:
        return len(self.items)

    def widths(self) -> np.ndarray:
        # Rendered at ~2x digit height then resized to `height`: about 0.55*height per char.
        return np.asarray([int(len(t) * 0.55 * self.height + self.height) for _, t in self.items])

    def __getitem__(self, i: int):
        kind, text = self.items[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed * 100003 + i)
        img = render_field(text, self.bank, rng)
        if self.train:
            arr = augment_photometric(preprocess(augment_geometric(img, rng), self.height), rng)
        else:
            arr = preprocess(img, self.height)
        return torch.from_numpy(arr), torch.tensor(self.charset.encode(text), dtype=torch.long), text

    def ctc_feasible(self) -> np.ndarray:
        return np.ones(len(self.items), bool)


assert DOWNSAMPLE == 4
