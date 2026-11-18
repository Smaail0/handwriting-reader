"""Score a page photo against a list of its handwritten lines (order-free).

  python scripts/eval_photo.py samples/jobsheet.jpg samples/truth/jobsheet.txt \\
      --model runs/joint/model_int8_dynamic.onnx --lm models/lm_mixed.json

For each truth line, the best-matching recognized line (lowest CER against any window of the
same length, so a value read together with its printed label still counts) gives a line CER;
the score is the character-weighted mean, plus word recall (truth words found anywhere).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
from PIL import Image

from htr.decode import beam_search
from htr.inference import OnnxRecognizer
from htr.lm import LMScorer
from htr.metrics import cer
from htr.page import segment_page
from htr.text import detokenize, from_output


def _words(t: str) -> list[str]:
    return [w for w in re.split(r"[\s,]+", t.lower().replace("£", " ")) if w.strip(".")]


def best_window_cer(truth: str, hyps: list[str]) -> float:
    best = 1.0
    for h in hyps:
        n = len(truth)
        for i in range(0, max(1, len(h) - n + 1)):
            best = min(best, cer([h[i:i + n]], [truth]))
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("truth", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--lm", type=Path, default=None)
    p.add_argument("--method", default="db")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    rec = OnnxRecognizer(args.model)
    lm = LMScorer.build(args.lm, None, rec.script) if args.lm else None
    lines = segment_page(cv2.imread(str(args.image)), method=args.method, rtl=rec.rtl)
    hyps = []
    for ln in lines:
        x = rec.prepare(Image.fromarray(ln.image))
        t = from_output(beam_search(rec.logits(x), rec.charset, lm=lm).text, rec.script) if lm else rec.run(x)
        hyps.append(detokenize(t))
    if args.show:
        for i, h in enumerate(hyps, 1):
            print(f"  {i:2d}. {h}")
    truth = [t.strip() for t in args.truth.read_text(encoding="utf-8").splitlines() if t.strip()]
    errs = [best_window_cer(t, hyps) for t in truth]
    mean = sum(e * len(t) for e, t in zip(errs, truth)) / sum(len(t) for t in truth)
    found = [w for w in hyps for w in _words(w)]
    tw = [w for t in truth for w in _words(t)]
    pool = list(found)
    hit = 0
    for w in tw:
        if w in pool:
            pool.remove(w)
            hit += 1
    print(f"{args.image.name}: {len(lines)} lines | value CER {mean:.1%} | word recall {hit}/{len(tw)} ({hit / len(tw):.0%})")


if __name__ == "__main__":
    main()
