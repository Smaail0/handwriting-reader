"""Benchmark CPU latency and size of candidate architectures (untrained weights).

Latency does not depend on weight values, so this ranks architectures by speed
before spending hours training them. Each candidate is exported to fp32 ONNX and
quantized (dynamic + static int8), then timed on real IAM test lines.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from htr.charset import Charset
from htr.data import load_split
from htr.export import export_onnx, quantize
from htr.inference import OnnxRecognizer
from htr.model import CRNN, ModelConfig, count_params

CANDIDATES = {
    "vgg_lstm256 (current)": dict(arch="vgg", channels=(32, 64, 128, 256), hidden=256, seq="lstm"),
    "lite_lstm128": dict(arch="lite", channels=(16, 32, 64, 128), hidden=128, seq="lstm"),
    "lite_conv128": dict(arch="lite", channels=(16, 32, 64, 128), hidden=128, seq="conv"),
    "lite_sep_conv128": dict(arch="lite", channels=(16, 32, 64, 128), hidden=128, seq="conv", separable=True),
    "lite_sep_conv192": dict(arch="lite", channels=(24, 48, 96, 192), hidden=192, seq="conv", separable=True),
    "tiny_conv96": dict(arch="lite", channels=(16, 24, 48, 96), hidden=96, seq="conv"),
}


def time_model(path: Path, inputs: list[np.ndarray]) -> float:
    rec = OnnxRecognizer(path, threads=1)
    for x in inputs[:5]:
        rec.run(x)
    t = []
    for x in inputs:
        t0 = time.perf_counter()
        rec.run(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/iam_line"))
    p.add_argument("--out", type=Path, default=Path("runs/arch_bench"))
    p.add_argument("--lines", type=int, default=200)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data_dir, "train")
    charset = Charset.from_texts(t for _, t in train)
    test = load_split(args.data_dir, "test")[: args.lines]
    calib_imgs = [Image.open(io.BytesIO(b)) for b, _ in train[:64]]
    test_imgs = [Image.open(io.BytesIO(b)) for b, _ in test]

    results = []
    print(f"{'candidate':24s} {'params':>8s} {'fp32 MB':>8s} {'int8 MB':>8s} {'fp32 ms':>8s} {'dyn ms':>8s} {'static ms':>9s}")
    for name, kw in CANDIDATES.items():
        cfg = ModelConfig(num_classes=charset.num_classes, **kw)
        model = CRNN(cfg).eval()
        stem = name.split()[0]
        fp32 = export_onnx(model, args.out / f"{stem}_fp32.onnx", {"charset": charset.chars})
        rec = OnnxRecognizer(fp32, threads=1)
        inputs = [rec.prepare(im) for im in test_imgs]
        feeds = [{rec.input_name: rec.prepare(im)} for im in calib_imgs]
        dyn = quantize(fp32, args.out / f"{stem}_dynamic.onnx", "dynamic")
        sta = quantize(fp32, args.out / f"{stem}_static.onnx", "static", feeds)
        r = {"name": name, "params": count_params(model), "fp32_mb": fp32.stat().st_size / 1e6,
             "int8_mb": min(dyn.stat().st_size, sta.stat().st_size) / 1e6,
             "fp32_ms": time_model(fp32, inputs), "dynamic_ms": time_model(dyn, inputs),
             "static_ms": time_model(sta, inputs), "config": cfg.to_dict()}
        results.append(r)
        print(f"{name:24s} {r['params'] / 1e6:7.2f}M {r['fp32_mb']:8.2f} {r['int8_mb']:8.2f} "
              f"{r['fp32_ms']:8.2f} {r['dynamic_ms']:8.2f} {r['static_ms']:9.2f}", flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
