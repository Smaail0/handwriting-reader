"""Quantize an fp32 ONNX recognizer to int8 with ONNX Runtime.

Modes:
  dynamic  weights int8, activations quantized on the fly (covers Conv, MatMul, LSTM)
  static   weights + activations int8 with ranges calibrated on training lines (QDQ format);
           LSTM stays fp32 because ORT has no static int8 LSTM kernel
  hybrid   static QDQ for Conv/MatMul, then dynamic int8 for the LSTM
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

from htr.data import load_split
from htr.export import quantize
from htr.inference import OnnxRecognizer


def calibration_feeds(model: Path, data_dirs: list[Path], n: int, seed: int = 0) -> list[dict]:
    """n calibration lines split evenly across data_dirs (a joint model needs every script's ranges)."""
    rec = OnnxRecognizer(model, threads=4)
    rng = np.random.default_rng(seed)
    feeds = []
    for k, d in enumerate(data_dirs):
        samples = load_split(d, "train")
        if not samples:
            raise SystemExit(f"no train lines in {d}")
        m = n // len(data_dirs) + (k < n % len(data_dirs))
        idx = rng.choice(len(samples), size=min(m, len(samples)), replace=False)
        feeds += [{rec.input_name: rec.prepare(Image.open(io.BytesIO(samples[i][0])))} for i in idx]
    return feeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help="fp32 ONNX model")
    p.add_argument("--mode", choices=["dynamic", "static", "hybrid"], default="dynamic")
    p.add_argument("--out", type=Path, default=None, help="default: <model dir>/model_int8_<mode>.onnx")
    p.add_argument("--data-dir", type=Path, nargs="+", default=[Path("data/raw/iam_line")],
                   help="calibration lines come from these datasets' train splits (split evenly)")
    p.add_argument("--calib-lines", type=int, default=256)
    p.add_argument("--calib-method", choices=["minmax", "entropy", "percentile"], default="minmax")
    p.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    out = args.out or args.model.parent / f"model_int8_{args.mode}.onnx"

    feeds = None if args.mode == "dynamic" else calibration_feeds(args.model, args.data_dir, args.calib_lines)
    quantize(args.model, out, args.mode, feeds, per_channel=args.per_channel, calib_method=args.calib_method)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, fp32 was {args.model.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
