"""Export a trained checkpoint to ONNX (fp32) with dynamic batch and width.

The charset, input height and text direction are stored in the ONNX metadata, so
the .onnx file is self-contained for inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from htr.export import export_onnx
from htr.model import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: <checkpoint dir>/model_fp32.onnx")
    args = p.parse_args()
    out = args.out or args.checkpoint.parent / "model_fp32.onnx"

    model, charset, ckpt = load_checkpoint(args.checkpoint)
    meta = {"charset": charset.chars, "rtl": str(int(ckpt.get("rtl", False))),
            "script": ckpt.get("script", "latin"), "val_cer": f"{ckpt.get('val_cer', float('nan')):.5f}"}
    export_onnx(model, out, meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
