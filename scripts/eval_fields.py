"""Line-level benchmark on synthetic handwritten number fields from *unseen* MNIST writers.

Compares greedy decoding, unconstrained beam search and beam search constrained to the
field type, per field kind (date, amount, digits, phone).

  python scripts/eval_fields.py runs/lite_en/model_int8_static.onnx runs/lite_fields/model_int8_static.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from htr import fields as F
from htr.decode import beam_search, greedy
from htr.inference import OnnxRecognizer
from htr.metrics import cer
from htr.synth import FIELD_KINDS, DigitBank, random_field_text, render_field


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", type=Path, nargs="+")
    p.add_argument("--mnist", type=Path, default=Path("data/raw/mnist/test.parquet"))
    p.add_argument("--per-kind", type=int, default=250)
    p.add_argument("--log", type=Path, default=Path("runs/field_line_results.jsonl"))
    args = p.parse_args()

    bank = DigitBank(args.mnist)
    rng = np.random.default_rng(2024)
    samples = []
    for kind in FIELD_KINDS:
        for _ in range(args.per_kind):
            text = random_field_text(kind, rng)
            samples.append((kind, text, render_field(text, bank, rng)))

    with open(args.log, "a") as log:
        for path in args.models:
            rec = OnnxRecognizer(path)
            print(f"\n{path}")
            print(f"{'kind':8s} {'greedy':>8s} {'beam':>8s} {'constrained':>12s} {'CER(constr.)':>13s}")
            row = {"model": str(path)}
            for kind in FIELD_KINDS:
                sub = [s for s in samples if s[0] == kind]
                ftype = "phone" if kind == "phone" else kind
                cons = F.constraint_for(ftype)
                g, b, c, preds, refs = 0, 0, 0, [], []
                for _, text, img in sub:
                    lg = rec.logits(rec.prepare(img))
                    g += greedy(lg, rec.charset).text == text
                    b += beam_search(lg, rec.charset).text == text
                    d = beam_search(lg, rec.charset, constraint=cons)
                    c += d.text == text
                    preds.append(d.text)
                    refs.append(text)
                n = len(sub)
                row[kind] = {"greedy": g / n, "beam": b / n, "constrained": c / n, "cer": cer(preds, refs)}
                print(f"{kind:8s} {g / n:8.1%} {b / n:8.1%} {c / n:12.1%} {cer(preds, refs):13.3f}")
            log.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
