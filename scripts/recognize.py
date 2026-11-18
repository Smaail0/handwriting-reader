"""Recognize handwritten text in page photos/scans: segment lines, then read each line.

Example:
  python scripts/recognize.py samples/page.jpg --model runs/crnn_lstm/model_int8_dynamic.onnx --debug-dir runs/demo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from PIL import Image

from htr.decode import beam_search
from htr.inference import OnnxRecognizer
from htr.lm import LMScorer
from htr.page import segment_page
from htr.text import detokenize, from_output, is_rtl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, required=True, nargs="+", help="one or more ONNX models to compare")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--debug-dir", type=Path, default=None, help="save line crops and an overlay of detected lines")
    p.add_argument("--method", choices=["db", "projection", "blobs"], default="db",
                   help="line segmentation: db = learned text detector (default), projection/blobs = classic")
    p.add_argument("--raw", action="store_true", help="keep IAM-style spaced punctuation")
    p.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True,
                   help="find the sheet in a photo and flatten it first (default on)")
    p.add_argument("--lm", type=Path, default=None, help="character language model (scripts/build_lm.py)")
    p.add_argument("--lexicon", type=Path, nargs="*", default=None, help="domain phrase lists (lexicons/*.txt)")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=1.5)
    p.add_argument("--beam", type=int, default=16, help="beam width when a language model is used")
    args = p.parse_args()

    recognizers = {m: OnnxRecognizer(m, threads=args.threads) for m in args.model}
    scorers = {m: LMScorer.build(args.lm, args.lexicon, r.script, alpha=args.lm_alpha, beta=args.lm_beta)
               for m, r in recognizers.items()} if (args.lm or args.lexicon) else {}
    for path in args.images:
        page = cv2.imread(str(path))
        if page is None:
            raise SystemExit(f"cannot read {path} (convert HEIC to JPEG/PNG first, e.g. with `sips`)")
        rtl = any(r.rtl for r in recognizers.values())
        t0 = time.perf_counter()
        lines = segment_page(page, method=args.method, rtl=rtl, paper=args.paper)
        seg_ms = (time.perf_counter() - t0) * 1000
        print(f"\n=== {path.name}: {len(lines)} lines, segmentation {seg_ms:.1f} ms")

        if args.debug_dir:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            overlay = page.copy()
            for i, line in enumerate(lines):
                cv2.polylines(overlay, [line.polygon.astype("int32")], True, (0, 0, 255), 3)
                x, y = (int(v) for v in line.polygon[0])
                cv2.putText(overlay, str(i + 1), (x, max(30, y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imwrite(str(args.debug_dir / f"{path.stem}_line{i + 1}.png"), line.image)
            cv2.imwrite(str(args.debug_dir / f"{path.stem}_overlay.jpg"), overlay)

        for model, rec in recognizers.items():
            print(f"--- {model.name}")
            total = 0.0
            for i, line in enumerate(lines):
                x = rec.prepare(Image.fromarray(line.image))
                t0 = time.perf_counter()
                if model in scorers:
                    d = beam_search(rec.logits(x), rec.charset, beam=args.beam, lm=scorers[model])
                    text = from_output(d.text, rec.script)
                else:
                    text = rec.run(x)
                if not (args.raw or rec.rtl):
                    text = detokenize(text)
                ms = (time.perf_counter() - t0) * 1000
                total += ms
                print(f"  {i + 1}. {text!r:60s} {ms:6.1f} ms  ({line.angle:+.0f}°)")
            print(f"  total recognition {total:.1f} ms")


if __name__ == "__main__":
    main()
