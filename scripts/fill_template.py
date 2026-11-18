"""Synthetic filled-in scans of ANY template, with ground truth, for testing and calibration.

Handwriting comes only from held-out writers: words cut from the IAM/RIMES *test* lines
(data/words, built by scripts/build_word_bank.py) and MNIST *test* digits. Field types
decide what is written: name / text / digits / code / date / phone / amount / number / check.

  python scripts/fill_template.py templates/cnam_bulletin.json --n 40 --out data/forms_cnam \\
      --blank samples/cnam/blank.png      # optional colour version of the blank form
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from htr.forms import Template
from htr.mixed import _normalize_ink, _word_image
from htr.synth import DigitBank, random_field_text, render_field

_spec = importlib.util.spec_from_file_location("make_forms", Path(__file__).with_name("make_forms.py"))
_mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf)
paste, scan_effects = _mf.paste, _mf.scan_effects


class Words:
    """Held-out handwritten words, grouped into phrases (consecutive words of one line)."""

    def __init__(self, words_dir: Path, sources=("rimes", "iam")):
        self.phrases = []
        for src in sources:
            rows = pq.read_table(words_dir / f"{src}_test.parquet", columns=["image", "text", "line", "pos"]).to_pylist()
            by_line = defaultdict(list)
            for r in rows:
                by_line[r["line"]].append(r)
            for line_rows in by_line.values():
                line_rows.sort(key=lambda r: r["pos"])
                cur = [line_rows[0]]
                for r in line_rows[1:]:
                    if r["pos"] == cur[-1]["pos"] + 1:
                        cur.append(r)
                    else:
                        self.phrases.append(cur)
                        cur = [r]
                self.phrases.append(cur)
        # The recognizer's charset has no accents (IAM): keep plain-ASCII words for exact truth.
        self.names = [r for ph in self.phrases for r in ph if re.fullmatch(r"[A-Z][a-z]{2,}", r["text"])]

    @staticmethod
    def image(rows) -> np.ndarray:
        """Words side by side on one baseline band, paper removed, dark ink on white."""
        body = 24.0
        pieces = [_normalize_ink(_word_image(r["image"]), body) for r in rows]
        above = max(m for _, m in pieces)
        below = max(a.shape[0] - m for a, m in pieces)
        gap = int(body * 0.9)
        canvas = np.zeros((int(above + below) + 4, sum(a.shape[1] for a, _ in pieces) + gap * (len(pieces) - 1) + 4),
                          np.float32)
        x = 2
        for a, m in pieces:
            y = int(2 + above - m)
            region = canvas[y: y + a.shape[0], x: x + a.shape[1]]
            np.maximum(region, a, out=region)
            x += a.shape[1] + gap
        return (255 * (1 - canvas)).astype(np.uint8)

    def name(self, rng):
        r = self.names[int(rng.integers(len(self.names)))]
        return r["text"], self.image([r])

    def phrase(self, rng, lo=2, hi=5):
        for _ in range(200):
            ph = self.phrases[int(rng.integers(len(self.phrases)))]
            n = int(rng.integers(lo, hi + 1))
            if len(ph) < n:
                continue
            i = int(rng.integers(0, len(ph) - n + 1))
            rows = ph[i: i + n]
            text = " ".join(r["text"] for r in rows)
            if re.fullmatch(r"[A-Za-z0-9 ,.'-]+", text):
                return text, self.image(rows)
        raise RuntimeError("no plain phrase found")


def draw_tick(canvas, box, rng, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pts = np.array([[x0 + 0.2 * w, y0 + 0.55 * h], [x0 + 0.42 * w, y0 + 0.8 * h], [x0 + 0.85 * w, y0 + 0.2 * h]])
    pts += rng.normal(0, 0.05 * min(w, h), pts.shape)
    cv2.polylines(canvas, [pts.astype(np.int32)], False, color, max(2, int(min(w, h) * 0.08)), cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template", type=Path)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blank", type=Path, default=None, help="colour image of the blank form (same layout)")
    p.add_argument("--words-dir", type=Path, default=Path("data/words"))
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tmpl = Template.load(args.template)
    base = cv2.imread(str(args.blank)) if args.blank else cv2.cvtColor(tmpl.image, cv2.COLOR_GRAY2BGR)
    base = cv2.resize(base, (tmpl.image.shape[1], tmpl.image.shape[0]))
    rng = np.random.default_rng(args.seed)
    digits = DigitBank(Path("data/raw/mnist/test.parquet"))  # unseen writers
    words = Words(args.words_dir)
    truth = {}
    for k in range(args.n):
        canvas = base.copy()
        ink = (int(rng.uniform(90, 170)), int(rng.uniform(20, 60)), int(rng.uniform(10, 40))) if rng.random() < 0.7 \
            else (int(rng.uniform(20, 60)),) * 3
        t = {}
        for spec in tmpl.fields:
            if spec.type == "check":
                ticked = rng.random() < 0.35
                if ticked:
                    draw_tick(canvas, spec.box, rng, ink)
                t[spec.name] = "x" if ticked else ""
                continue
            if spec.type == "name":
                txt, im = words.name(rng)
            elif spec.type in ("text", "free"):
                txt, im = words.phrase(rng)
            else:
                if spec.type in ("digits", "code"):
                    txt = "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(4, 12))))
                else:
                    txt = random_field_text({"number": "amount"}.get(spec.type, spec.type), rng)
                im = np.asarray(render_field(txt, digits, rng))
            paste(canvas, im, spec.box, rng, ink)
            t[spec.name] = txt
        name = f"form_{k:03d}.jpg"
        cv2.imwrite(str(args.out / name), scan_effects(canvas, rng))
        truth[name] = t
    (args.out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{args.n} filled scans of {tmpl.name} ({len(tmpl.fields)} fields) -> {args.out}")


if __name__ == "__main__":
    main()
