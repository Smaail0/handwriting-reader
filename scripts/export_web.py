"""Copy ONNX recognizers into web/models/ for the client-side (browser) build.

ONNX Runtime Web does not expose a model's custom metadata, so each model gets a sidecar
JSON (charset, height, script, direction). Also writes parity fixtures: line images and
the Python pipeline's outputs, which web/test/parity.mjs checks the JS port against.

  python scripts/export_web.py en=runs/lite_en/model_int8_static.onnx ar=runs/lite_ar/model_int8_static.onnx
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

from htr.data import load_split
from htr.inference import OnnxRecognizer
from htr.preprocess import preprocess
from htr.text import is_rtl

DATASETS = {"latin": "data/raw/iam_line", "arabic": "data/raw/khatt", "mixed": "data/raw/rimes"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="name=path.onnx")
    p.add_argument("--out", type=Path, default=Path("web/models"))
    p.add_argument("--fixtures", type=int, default=60, help="test lines per model for the parity check")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures_dir = Path("web/test/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.models:
        name, path = spec.split("=", 1)
        meta = {m.key: m.value for m in onnx.load(path, load_external_data=False).metadata_props}
        info = {"charset": meta["charset"], "height": int(meta.get("height", 64)),
                "downsample": int(meta.get("downsample", 4)), "script": meta.get("script", "latin"),
                "rtl": meta.get("rtl", "0") == "1", "source": path}
        # Default language model for this recognizer's script (scripts/build_lm.py); none for
        # Arabic, which has no text that is not shared with KHATT's validation/test lines.
        lm_file = {"latin": "lm_en.json", "mixed": "lm_mixed.json"}.get(info["script"])
        if lm_file and Path("models", lm_file).exists():
            shutil.copyfile(Path("models", lm_file), args.out / lm_file)
            info["lm"] = lm_file
            info["lm_alpha"], info["lm_beta"] = 0.5, 1.5  # tuned on IAM / RIMES validation (runs/lm_results.jsonl)
        shutil.copyfile(path, args.out / f"{name}.onnx")
        (args.out / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        size = (args.out / f"{name}.onnx").stat().st_size / 1e6
        print(f"{name}: {path} ({size:.2f} MB, script={info['script']}, {len(info['charset'])} chars)")

        rec = OnnxRecognizer(path)
        rows = load_split(Path(DATASETS[info["script"]]), "test", info["script"])[: args.fixtures]
        cases = []
        for data, text in rows:
            img = Image.open(io.BytesIO(data)).convert("L")
            arr = preprocess(img, info["height"], rtl=is_rtl(info["script"]))
            cases.append({
                "width": img.size[0], "height": img.size[1],
                "pixels": base64.b64encode(zlib.compress(np.asarray(img).tobytes(), 6)).decode(),
                "prep_width": int(arr.shape[1]), "prep_sum": float(arr.astype(np.float64).sum()),
                "greedy": rec(img), "reference": text,
            })
        (fixtures_dir / f"{name}.json").write_text(json.dumps({"model": f"{name}.onnx", "cases": cases},
                                                              ensure_ascii=False))
        print(f"  {len(cases)} parity fixtures -> {fixtures_dir / (name + '.json')}")
    shutil.copyfile(Path("models/ppocr_det_v4.onnx"), args.out / "ppocr_det_v4.onnx")  # text detector (page.js)
    write_page_fixtures(Path("web/test/fixtures_page.json"))
    lex_out = Path("web/lexicons")
    lex_out.mkdir(exist_ok=True)
    for f in Path("lexicons").glob("*.txt"):
        shutil.copyfile(f, lex_out / f.name)
    write_resize_fixtures(Path("web/test/resize_fixtures.json"))
    write_beam_fixtures(Path("web/test/beam_fixtures.json"))


def write_resize_fixtures(path: Path, n: int = 300, seed: int = 7) -> None:
    """Random grayscale images and Pillow's BILINEAR resize of them (pass order varies with shape)."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        w, h = int(rng.integers(8, 600)), int(rng.integers(8, 400))
        if i % 2:
            oh = 64
            ow = max(16, round(w * oh / h))
        else:
            ow, oh = int(rng.integers(8, 400)), int(rng.integers(8, 200))
        a = rng.integers(0, 256, (h, w), dtype=np.uint8)
        out = np.asarray(Image.fromarray(a).resize((ow, oh), Image.Resampling.BILINEAR))
        cases.append({"w": w, "h": h, "ow": ow, "oh": oh,
                      "px": base64.b64encode(a.tobytes()).decode(), "out": base64.b64encode(out.tobytes()).decode()})
    path.write_text(json.dumps(cases))


def write_page_fixtures(path: Path, n_iam: int = 15) -> None:
    """RGBA pages (IAM test scans + photos in web/test/samples) and Python's segment_page boxes."""
    import cv2

    from htr.page import segment_page

    sys_path = Path("scripts")
    import sys
    sys.path.insert(0, str(sys_path))
    from eval_pages import load_pages

    imgs = [(f"iam_{i:03d}", cv2.imdecode(np.frombuffer(pg["image"], np.uint8), cv2.IMREAD_COLOR))
            for i, pg in enumerate(load_pages("iam", Path("data/raw"), n_iam)[0])]
    imgs += [(f.stem, cv2.imread(str(f))) for f in sorted(Path("web/test/samples").glob("*.jpg"))]
    cases = []
    for name, bgr in imgs:
        f = min(1.0, 2000 / max(bgr.shape[:2]))  # the demo works at <= 2000 px
        if f < 1:
            bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        lines = segment_page(bgr)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        cases.append({"name": name, "width": int(bgr.shape[1]), "height": int(bgr.shape[0]),
                      "pixels": base64.b64encode(zlib.compress(rgba.tobytes(), 6)).decode(),
                      "boxes": [list(map(int, ln.box)) for ln in lines]})
    path.write_text(json.dumps({"cases": cases}))
    print(f"page fixtures: {len(cases)} pages -> {path}")


def write_beam_fixtures(path: Path, n: int = 200, seed: int = 3) -> None:
    """Logits (random, plus real model outputs on test lines) and decode.py's beam search results."""
    from htr.decode import beam_search
    from htr.charset import Charset
    from htr.lm import CharNgramLM, LMScorer, MixLM

    rng = np.random.default_rng(seed)
    cases = []
    charset = Charset("abcdefghij")
    for _ in range(n):
        t = int(rng.integers(5, 60))
        logits = (rng.normal(0, 3, (t, charset.num_classes)) + np.eye(charset.num_classes)[rng.integers(0, 11, t)] * 4)
        logits = logits.astype(np.float32)
        d = beam_search(logits, charset, beam=16)
        cases.append({"chars": charset.chars, "T": t, "C": charset.num_classes,
                      "logits": base64.b64encode(logits.tobytes()).decode(), "text": d.text, "confidence": d.confidence})
    rec = OnnxRecognizer("runs/lite_en/model_fp32.onnx")
    for data, _ in load_split(Path("data/raw/iam_line"), "test", "latin")[:40]:
        lg = rec.logits(rec.prepare(Image.open(io.BytesIO(data)).convert("L"))).astype(np.float32)
        d = beam_search(lg, rec.charset, beam=16)
        cases.append({"chars": rec.charset.chars, "T": int(lg.shape[0]), "C": int(lg.shape[1]),
                      "logits": base64.b64encode(lg.tobytes()).decode(), "text": d.text, "confidence": d.confidence})
        # Same logits with language-model fusion: one base model, then a 70/30 mix of two.
        en, fr = CharNgramLM.load(Path("models/lm_en.json")), CharNgramLM.load(Path("models/lm_fr.json"))
        for lm_files, scorer in [(["lm_en.json"], LMScorer(en, 0.3, 1.5)),
                                 (["lm_en.json", "lm_fr.json"], LMScorer(MixLM([(en, 0.7), (fr, 0.3)]), 0.3, 1.5))]:
            d = beam_search(lg, rec.charset, beam=16, lm=scorer)
            cases.append({"chars": rec.charset.chars, "T": int(lg.shape[0]), "C": int(lg.shape[1]),
                          "logits": base64.b64encode(lg.tobytes()).decode(), "text": d.text, "confidence": d.confidence,
                          "lm": lm_files, "alpha": 0.3, "beta": 1.5})
    path.write_text(json.dumps(cases, ensure_ascii=False))


if __name__ == "__main__":
    main()
