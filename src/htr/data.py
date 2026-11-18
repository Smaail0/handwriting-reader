"""IAM line dataset, augmentation and width-bucketed batching."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler

from .charset import Charset
from .text import is_rtl, normalize, to_target
from .preprocess import DEFAULT_HEIGHT, DOWNSAMPLE, preprocess

Sample = tuple[bytes, str]


def load_split(data_dir: Path, split: str, script: str | None = None) -> list[Sample]:
    """Rows of (image bytes, text). With `script`, texts are normalized for that script."""
    rows = pq.read_table(Path(data_dir) / f"{split}.parquet", columns=["image", "text"]).to_pylist()
    return [(r["image"]["bytes"], normalize(r["text"], script) if script else r["text"]) for r in rows]


# ---------------------------------------------------------------- augmentation


def _shear(img: Image.Image, s: float) -> Image.Image:
    """Horizontal shear (slant) with white fill, widening the canvas to keep all ink."""
    w, h = img.size
    new_w = w + int(abs(s) * h)
    c = -max(s, 0.0) * h
    return img.transform((new_w, h), Image.Transform.AFFINE, (1, s, c, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=255)


def augment_geometric(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = img.convert("L")
    r = rng.random()
    if r < 0.2:
        img = img.filter(ImageFilter.MinFilter(3))  # thicker strokes
    elif r < 0.4:
        img = img.filter(ImageFilter.MaxFilter(3))  # thinner strokes
    if rng.random() < 0.7:
        img = _shear(img, rng.uniform(-0.5, 0.5))
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    w, h = img.size
    if rng.random() < 0.5:  # horizontal stretch / squeeze
        img = img.resize((max(8, int(w * rng.uniform(0.8, 1.25))), h), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:  # vertical margin jitter
        top, bottom = (int(h * rng.uniform(-0.08, 0.15)) for _ in range(2))
        w, h = img.size
        canvas = Image.new("L", (w, h + top + bottom), 255)
        canvas.paste(img, (0, top))
        img = canvas
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return img


def augment_photometric(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = arr * rng.uniform(0.6, 1.0)
    if rng.random() < 0.5:
        arr = arr + rng.normal(0.0, rng.uniform(0.01, 0.06), arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.0, out=arr)


# ---------------------------------------------------------------- dataset


class LineDataset(Dataset):
    def __init__(self, samples: list[Sample], charset: Charset, height: int = DEFAULT_HEIGHT,
                 max_width: int = 2048, train: bool = False, script: str = "latin"):
        self.samples = samples
        self.charset = charset
        self.height = height
        self.max_width = max_width
        self.train = train
        self.script = script
        self.rtl = is_rtl(script)

    def __len__(self) -> int:
        return len(self.samples)

    def widths(self) -> np.ndarray:
        """Approximate preprocessed widths (from image headers), for bucketing."""
        out = []
        for b, _ in self.samples:
            w, h = Image.open(io.BytesIO(b)).size
            out.append(min(self.max_width, round(w * self.height / h)))
        return np.asarray(out)

    def __getitem__(self, i: int):
        data, text = self.samples[i]
        img = Image.open(io.BytesIO(data))
        if self.train:
            rng = np.random.default_rng()
            arr = augment_photometric(preprocess(augment_geometric(img, rng), self.height, self.max_width,
                                                 self.rtl), rng)
        else:
            arr = preprocess(img, self.height, self.max_width, self.rtl)
        target = self.charset.encode(to_target(text, self.script))
        return torch.from_numpy(arr), torch.tensor(target, dtype=torch.long), text

    def ctc_feasible(self) -> np.ndarray:
        """Lines whose label fits in the frames CTC gets (len + repeated neighbours, with slack).

        Drops e.g. KHATT "lines" that are really diagrams or lists, which become far too
        narrow once squeezed to the input height.
        """
        frames = self.widths() // DOWNSAMPLE
        need = []
        for _, text in self.samples:
            t = to_target(text, self.script)
            need.append(len(t) + sum(a == b for a, b in zip(t, t[1:])))
        return frames >= 1.5 * np.asarray(need)


def collate(batch, pad_to: int = DOWNSAMPLE):
    """Pad a batch to a common width.

    pad_to rounds the batch width up to a multiple of this value. On GPU backends
    that compile per input shape (Apple MPS), a coarse value such as 256 keeps the
    number of distinct shapes small, which made training about 3x faster.
    Must be a multiple of DOWNSAMPLE.
    """
    imgs, targets, texts = zip(*batch)
    height = imgs[0].shape[0]
    max_w = max(im.shape[1] for im in imgs)
    max_w = -(-max_w // pad_to) * pad_to
    x = torch.zeros(len(imgs), 1, height, max_w)
    for k, im in enumerate(imgs):
        x[k, 0, :, : im.shape[1]] = im
    input_lengths = torch.tensor([im.shape[1] // DOWNSAMPLE for im in imgs], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return x, torch.cat(targets), input_lengths, target_lengths, list(texts)


class BucketBatchSampler(Sampler[list[int]]):
    """Groups lines of similar width so batches carry little padding."""

    def __init__(self, widths: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.widths = np.asarray(widths, dtype=np.float64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return -(-len(self.widths) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        keys = self.widths * (rng.uniform(0.85, 1.15, len(self.widths)) if self.shuffle else 1.0)
        order = np.argsort(keys)
        batches = [order[i : i + self.batch_size].tolist() for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
