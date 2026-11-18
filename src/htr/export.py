"""ONNX export and int8 quantization helpers."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import onnx
import torch
from torch import nn


def export_onnx(model: nn.Module, path: Path, metadata: dict[str, str]) -> Path:
    """Export (B, 1, H, W) -> (B, W/4, C) with dynamic batch and width; metadata goes into the file."""
    model = model.eval().cpu()
    height = model.cfg.height
    dims = {"x": {0: torch.export.Dim("batch", max=512), 3: torch.export.Dim("width", min=16, max=16384)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prog = torch.onnx.export(model, (torch.rand(1, 1, height, 800),), input_names=["image"],
                                 output_names=["logits"], dynamic_shapes=dims, dynamo=True, optimize=True,
                                 verbose=False)
    prog.save(str(path))
    m = onnx.load(str(path))
    for k, v in {"height": str(height), "downsample": "4", **metadata}.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(m, str(path))
    onnx.checker.check_model(str(path))
    return path


def copy_metadata(src: Path, dst: Path) -> None:
    meta = {p.key: p.value for p in onnx.load(str(src)).metadata_props}
    m = onnx.load(str(dst))
    existing = {p.key for p in m.metadata_props}
    for k, v in meta.items():
        if k not in existing:
            entry = m.metadata_props.add()
            entry.key, entry.value = k, v
    onnx.save(m, str(dst))


def _flush_chunk(n_samples: int, preferred: int = 64) -> int:
    """Calibration flush size that does not evenly divide n_samples.

    The calibrator flushes every `preferred` samples to bound memory, but (ORT 1.32)
    raises "No data is collected" if the last flush lands exactly on the final sample.
    """
    for c in range(preferred, 1, -1):
        if n_samples % c:
            return c
    return n_samples + 1


def quantize(fp32: Path, out: Path, mode: str, calib_inputs: list | None = None,
             per_channel: bool = True, calib_method: str = "minmax") -> Path:
    """mode: dynamic | static | hybrid. calib_inputs: list of feed dicts (static/hybrid only)."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, feeds):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=False)
        if mode == "dynamic":
            quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8, per_channel=per_channel)
        elif mode in ("static", "hybrid"):
            if not calib_inputs:
                raise ValueError("static/hybrid quantization needs calibration inputs")
            method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                      "percentile": CalibrationMethod.Percentile}[calib_method]
            static_out = Path(tmp) / "static.onnx" if mode == "hybrid" else out
            quantize_static(str(pre), str(static_out), Reader(calib_inputs), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=per_channel, calibrate_method=method,
                            extra_options={"CalibMaxIntermediateOutputs": _flush_chunk(len(calib_inputs))})
            if mode == "hybrid":
                quantize_dynamic(str(static_out), str(out), weight_type=QuantType.QInt8,
                                 op_types_to_quantize=["LSTM"])
        else:
            raise ValueError(f"unknown mode {mode!r}")
    copy_metadata(fp32, out)
    return out
