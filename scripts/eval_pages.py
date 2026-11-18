"""End-to-end page evaluation: line segmentation + recognition on whole pages.

Page sets:
  iam          IAM paragraphs whose lines all come from the IAM *test* split (held-out writers)
  khatt        KHATT paragraphs. These come from KHATT's *training* split, so use them for
               segmentation (line counts) only; recognition CER here is optimistic.
  khatt_synth  Synthetic pages stacked from KHATT *test* lines (held out), with tight,
               sometimes overlapping line spacing.

Without --model, only segmentation is evaluated (line-count accuracy and speed).

Example:
  python scripts/eval_pages.py --set iam --model runs/crnn_lstm/model_fp32.onnx --limit 50
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from htr.inference import OnnxRecognizer
from htr.metrics import cer, wer
from htr.page import segment_page
from htr.text import detokenize, is_rtl, normalize

_SQUASH = re.compile(r"\s+")


def _squash(s: str) -> str:
    return _SQUASH.sub("", s)


def _tiling_line_count(page_text: str, lines: list[str], max_unmatched: float = 0.05) -> int | None:
    """Number of dataset lines that tile the page text, or None if no good tiling exists.

    Several writers copied the same texts, so many lines match a page. Only this page's
    own lines tile it end to end, so a DP minimizing unmatched characters recovers
    them (and hence the true line count) without page IDs.
    """
    pg = _squash(page_text)
    cands = {s for s in (_squash(l) for l in lines) if len(s) >= 4 and s in pg}
    by_prefix: dict[str, list[str]] = {}
    for c in cands:
        by_prefix.setdefault(c[:4], []).append(c)
    inf = (10**9, 0)
    dp = [inf] * (len(pg) + 1)  # (unmatched chars, lines used)
    dp[0] = (0, 0)
    for i in range(len(pg)):
        if dp[i] == inf:
            continue
        u, n = dp[i]
        dp[i + 1] = min(dp[i + 1], (u + 1, n))
        for c in by_prefix.get(pg[i: i + 4], ()):
            if pg.startswith(c, i):
                dp[i + len(c)] = min(dp[i + len(c)], (u, n + 1))
    u, n = dp[len(pg)]
    return n if u <= max_unmatched * len(pg) else None


def _coverage(page_text: str, lines: list[str]) -> float:
    pg = _squash(page_text)
    return sum(len(_squash(l)) for l in lines if len(_squash(l)) >= 8 and _squash(l) in pg) / max(1, len(pg))


def load_pages(name: str, root: Path, limit: int, seed: int = 0) -> tuple[list[dict], str]:
    """Returns pages as dicts {image (BGR array), text, n_lines} and the script."""
    if name == "iam":
        test = [normalize(r["text"], "latin") for r in
                pq.read_table(root / "iam_line/test.parquet", columns=["text"]).to_pylist()]
        out = []
        for r in pq.read_table(root / "iam_pages/all.parquet").to_pylist():
            text = normalize(r["label"], "latin")
            n = _tiling_line_count(text, test) if _coverage(text, test) >= 0.8 else None
            if n:
                out.append({"image": r["image"]["bytes"], "text": text, "n_lines": n})
            if limit and len(out) >= limit:
                break
        return out, "latin"
    if name == "khatt":
        train = [normalize(r["text"], "arabic") for r in
                 pq.read_table(root / "khatt/train.parquet", columns=["text"]).to_pylist()]
        out = []
        for r in pq.read_table(root / "khatt_pages/all.parquet").to_pylist():
            text = normalize(r["answer"], "arabic")
            n = _tiling_line_count(text, train) if _coverage(text, train) >= 0.8 else None
            if n:
                out.append({"image": r["image"]["bytes"], "text": text, "n_lines": n})
            if limit and len(out) >= limit:
                break
        return out, "arabic"
    if name == "khatt_synth":
        rows = pq.read_table(root / "khatt/test.parquet").to_pylist()
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(limit or 50):
            k = int(rng.integers(4, 10))
            start = int(rng.integers(0, len(rows) - k))
            out.append(_synth_page(rows[start:start + k], rng))
        return out, "arabic"
    raise ValueError(name)


def _synth_page(rows: list[dict], rng: np.random.Generator) -> dict:
    """Stack line images right-aligned, with spacing from -10% (overlap) to +30% of line height."""
    imgs, texts = [], []
    for r in rows:
        im = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("L"))
        ys = np.where((im < 128).any(1))[0]
        if len(ys) == 0:
            continue
        im = im[ys.min(): ys.max() + 1]
        s = 110 / max(im.shape[0], 1) if im.shape[0] < 250 else 0.5  # lines ~110 px; tall "lines" just halved
        imgs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        texts.append(normalize(r["text"], "arabic"))
    width = max(i.shape[1] for i in imgs) + 120
    height = sum(i.shape[0] for i in imgs) + 60 * len(imgs) + 120
    canvas = np.full((height, width), 255, np.uint8)
    y = 60
    for im in imgs:
        x = width - 60 - im.shape[1] + int(rng.integers(-20, 20))
        x = max(0, min(x, width - im.shape[1]))
        region = canvas[y: y + im.shape[0], x: x + im.shape[1]]
        np.minimum(region, im, out=region)
        y += int(im.shape[0] * (1 + rng.uniform(-0.1, 0.3)))
    canvas = canvas[: y + 60]
    ok, buf = cv2.imencode(".png", canvas)
    return {"image": buf.tobytes(), "text": " ".join(texts), "n_lines": len(imgs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["iam", "khatt", "khatt_synth"], default="iam")
    p.add_argument("--model", type=Path, nargs="*", default=[])
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--debug-dir", type=Path, default=None, help="save overlays for the first 10 pages")
    p.add_argument("--method", choices=["db", "projection", "blobs"], default="db",
                   help="line segmentation: db = learned text detector (default), projection/blobs = classic")
    p.add_argument("--log", type=Path, default=Path("runs/page_results.jsonl"))
    p.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True,
                   help="find the sheet in a photo and flatten it first (default on)")
    args = p.parse_args()

    pages, script = load_pages(args.set, args.root, args.limit)
    print(f"{args.set}: {len(pages)} pages ({script})")

    # Segmentation (shared by all models)
    seg_ms, segmented = [], []
    for i, pg in enumerate(pages):
        img = cv2.imdecode(np.frombuffer(pg["image"], np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        lines = segment_page(img, method=args.method, rtl=is_rtl(script), paper=args.paper)
        seg_ms.append((time.perf_counter() - t0) * 1000)
        segmented.append(lines)
        if args.debug_dir and i < 10:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            ov = img.copy()
            for j, ln in enumerate(lines):
                cv2.polylines(ov, [ln.polygon.astype(np.int32)], True, (0, 0, 255), 3)
                cv2.putText(ov, str(j + 1), tuple(int(v) for v in ln.polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)
            cv2.imwrite(str(args.debug_dir / f"{args.set}_{args.method}_{i:03d}_{len(lines)}of{pg['n_lines']}.jpg"), ov)
    diff = np.array([len(s) - pg["n_lines"] for s, pg in zip(segmented, pages)])
    seg = {"set": args.set, "method": args.method, "pages": len(pages), "line_count_exact": float(np.mean(diff == 0)),
           "too_few": float(np.mean(diff < 0)), "too_many": float(np.mean(diff > 0)),
           "seg_ms_p50": float(np.median(seg_ms))}
    print(f"segmentation: exact line count {seg['line_count_exact']:.1%}  too few {seg['too_few']:.1%}  "
          f"too many {seg['too_many']:.1%}  {seg['seg_ms_p50']:.1f} ms/page (p50)")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as log:
        log.write(json.dumps({"type": "segmentation", **seg}) + "\n")
        for model in args.model:
            rec = OnnxRecognizer(model, threads=args.threads)
            if rec.script != script:
                print(f"skip {model}: {rec.script} model on a {script} page set")
                continue
            preds, refs, rec_ms = [], [], []
            for lines, pg in zip(segmented, pages):
                t0 = time.perf_counter()
                texts = [rec(Image.fromarray(ln.image)) for ln in lines]
                rec_ms.append((time.perf_counter() - t0) * 1000)
                text = " ".join(texts)
                preds.append(text if is_rtl(script) else detokenize(text))
                refs.append(pg["text"])
            r = {"type": "page", "set": args.set, "model": str(model), "page_cer": cer(preds, refs),
                 "page_wer": wer(preds, refs),
                 "page_cer_nospace": cer([_squash(x) for x in preds], [_squash(x) for x in refs]),
                 "rec_ms_p50": float(np.median(rec_ms)),
                 "total_ms_p50": float(np.median(np.array(rec_ms) + np.array(seg_ms)))}
            log.write(json.dumps(r) + "\n")
            print(f"{str(model)[-45:]:45s} page CER {r['page_cer']:.4f} (no-space {r['page_cer_nospace']:.4f})  "
                  f"WER {r['page_wer']:.4f}  rec {r['rec_ms_p50']:.0f} ms + seg = {r['total_ms_p50']:.0f} ms/page")
            if args.set == "khatt":
                print("  (KHATT paragraphs are from the training split: recognition CER is optimistic)")


if __name__ == "__main__":
    main()
