"""Evaluate ONNX recognizers on an IAM split: accuracy (CER/WER) and CPU latency.

Example:
  python scripts/evaluate.py runs/crnn_lstm/model_fp32.onnx runs/crnn_lstm/model_int8_*.onnx --threads 1
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from htr.data import load_split
from htr.inference import OnnxRecognizer
from htr.metrics import cer, wer
from htr.text import normalize


def evaluate(path: Path, images: list, refs: list[str], threads: int, warmup: int = 5) -> dict:
    rec = OnnxRecognizer(path, threads=threads)
    refs = [normalize(r, rec.script) for r in refs]
    inputs = [rec.prepare(img) for img in images]
    for x in inputs[:warmup]:
        rec.run(x)
    preds, times = [], []
    for x in inputs:
        t0 = time.perf_counter()
        preds.append(rec.run(x))
        times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return {
        "model": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 3),
        "threads": threads,
        "lines": len(refs),
        "cer": round(cer(preds, refs), 5),
        "wer": round(wer(preds, refs), 5),
        "ms_mean": round(float(t.mean()), 2),
        "ms_p50": round(float(np.percentile(t, 50)), 2),
        "ms_p95": round(float(np.percentile(t, 95)), 2),
        "lines_per_s": round(1000 / float(t.mean()), 1),
        "cpu": platform.processor() or platform.machine(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", type=Path, nargs="+")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/iam_line"),
                   help="line dataset matching the models' script (e.g. data/raw/khatt for Arabic)")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N lines")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--log", type=Path, default=Path("runs/results.jsonl"))
    args = p.parse_args()

    samples = load_split(args.data_dir, args.split)
    if args.limit:
        samples = samples[: args.limit]
    images = [Image.open(io.BytesIO(b)) for b, _ in samples]
    for img in images:
        img.load()
    refs = [t for _, t in samples]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'model':48s} {'MB':>6s} {'CER':>7s} {'WER':>7s} {'ms p50':>7s} {'ms p95':>7s} {'lines/s':>8s}")
    with open(args.log, "a") as log:
        for path in args.models:
            r = evaluate(path, images, refs, args.threads)
            r["split"] = args.split
            r["dataset"] = args.data_dir.name
            log.write(json.dumps(r) + "\n")
            print(f"{str(path)[-48:]:48s} {r['size_mb']:6.2f} {r['cer']:7.4f} {r['wer']:7.4f} "
                  f"{r['ms_p50']:7.2f} {r['ms_p95']:7.2f} {r['lines_per_s']:8.1f}")


if __name__ == "__main__":
    main()
