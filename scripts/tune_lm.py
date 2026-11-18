"""Tune language-model fusion weights (alpha, beta) on a VALIDATION split, then report test.

  python scripts/tune_lm.py runs/lite_en/model_int8_static.onnx --lm models/lm_en.json \\
      --data-dir data/raw/iam_line --val-lines 400

Prints CER/WER of greedy, plain beam and beam + LM for each (alpha, beta), picks the best
pair on validation, then evaluates greedy / beam / beam + LM with that pair on the test split
(--test-lines 0 = all). Results are appended to runs/lm_results.jsonl.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from htr.data import load_split
from htr.decode import beam_search
from htr.inference import OnnxRecognizer
from htr.lm import LMScorer
from htr.metrics import cer, wer
from htr.text import from_output, normalize


def decode_all(rec, logits, lm=None, beam=16):
    out, t0 = [], time.perf_counter()
    for lg in logits:
        if beam:
            out.append(from_output(beam_search(lg, rec.charset, beam=beam, lm=lm).text, rec.script))
        else:
            out.append(from_output(rec.charset.decode_greedy(lg.argmax(-1)), rec.script))
    return out, (time.perf_counter() - t0) * 1000 / max(1, len(logits))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("--lm", type=Path, default=None)
    p.add_argument("--lexicon", type=Path, nargs="*", default=None)
    p.add_argument("--lexicon-weight", type=float, default=0.3)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--val-lines", type=int, default=400)
    p.add_argument("--test-lines", type=int, default=0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--log", type=Path, default=Path("runs/lm_results.jsonl"))
    args = p.parse_args()
    rec = OnnxRecognizer(args.model)

    def load(split, n):
        rows = load_split(args.data_dir, split)
        rows = rows[:n] if n else rows
        logits = [rec.logits(rec.prepare(Image.open(io.BytesIO(b)))) for b, _ in rows]
        return logits, [normalize(t, rec.script) for _, t in rows]

    val_lg, val_ref = load("validation", args.val_lines)
    base = decode_all(rec, val_lg, beam=0)[0]
    plain = decode_all(rec, val_lg, beam=args.beam)[0]
    print(f"validation ({len(val_ref)} lines): greedy CER {cer(base, val_ref):.4f}  beam {cer(plain, val_ref):.4f}")
    best = None
    for a in args.alphas:
        for b in args.betas:
            lm = LMScorer.build(args.lm, args.lexicon, rec.script, args.lexicon_weight, a, b)
            hyp = decode_all(rec, val_lg, lm, args.beam)[0]
            c, w = cer(hyp, val_ref), wer(hyp, val_ref)
            print(f"  alpha {a:.1f} beta {b:.1f}: CER {c:.4f} WER {w:.4f}", flush=True)
            if best is None or c < best[0]:
                best = (c, a, b)
    _, a, b = best
    print(f"best on validation: alpha {a} beta {b}")

    test_lg, test_ref = load("test", args.test_lines)
    lm = LMScorer.build(args.lm, args.lexicon, rec.script, args.lexicon_weight, a, b)
    res = {"model": str(args.model), "lm": str(args.lm), "lexicon": [str(x) for x in args.lexicon or []],
           "dataset": args.data_dir.name, "split": "test", "lines": len(test_ref), "alpha": a, "beta": b}
    for name, kw in [("greedy", dict(beam=0)), ("beam", dict(beam=args.beam)), ("beam_lm", dict(beam=args.beam, lm=lm))]:
        hyp, ms = decode_all(rec, test_lg, **kw)
        res[name] = {"cer": round(cer(hyp, test_ref), 5), "wer": round(wer(hyp, test_ref), 5), "decode_ms": round(ms, 2)}
        print(f"test {name:8s} CER {res[name]['cer']:.4f}  WER {res[name]['wer']:.4f}  decode {ms:.1f} ms/line")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
