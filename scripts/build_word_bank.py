"""Cut labelled word images out of the line datasets, for synthesizing mixed-script lines.

Uses CTC forced alignment with an already-trained recognizer per script (see htr.align),
then re-reads every word crop and keeps it only if the reading matches its label.
Splits are kept apart: words from test lines are only ever used to build test lines.

  python scripts/build_word_bank.py --out data/words
"""

from __future__ import annotations

import argparse
import io
import os
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SOURCES = {  # name: (line dataset, script, aligning model)
    "iam": ("data/raw/iam_line", "latin", "runs/crnn_lstm/model_fp32.onnx"),
    "rimes": ("data/raw/rimes", "latin", "runs/crnn_lstm/model_fp32.onnx"),
    "khatt": ("data/raw/khatt", "arabic", "runs/lite_ar/model_fp32.onnx"),
}
STORE_HEIGHT = 96  # word strips are stored at this height (the recognizer uses 64)

_rec = None


def _init(model: str) -> None:
    global _rec
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from htr.inference import OnnxRecognizer
    _rec = OnnxRecognizer(model, threads=1)


def _work(args):
    from htr.align import cut_words
    idx, data, text, script = args
    img = Image.open(io.BytesIO(data)).convert("L")
    words = cut_words(img, text, _rec, script) or []
    out = []
    for k, (im, word) in enumerate(words):
        if not any(ch.isalnum() for ch in word):
            continue  # stand-alone punctuation (IAM tokenizes ", ." as words)
        w, h = im.size
        im = im.resize((max(4, round(w * STORE_HEIGHT / h)), STORE_HEIGHT), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        out.append({"image": buf.getvalue(), "text": word, "line": idx, "pos": k})
    return len(text.split()), out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/words"))
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None, help="lines per split (for a quick test)")
    args = p.parse_args()

    from htr.data import load_split

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sources:
        data_dir, script, model = SOURCES[name]
        with Pool(args.workers, initializer=_init, initargs=(model,)) as pool:
            for split in args.splits:
                rows = load_split(Path(data_dir), split, script)[: args.limit]
                t0 = time.time()
                total, kept = 0, []
                jobs = [(i, b, t, script) for i, (b, t) in enumerate(rows)]
                for n_words, words in pool.imap(_work, jobs, chunksize=8):
                    total += n_words
                    kept.extend(words)
                for w in kept:
                    w.update(source=name, script=script, split=split)
                path = args.out / f"{name}_{split}.parquet"
                pq.write_table(pa.Table.from_pylist(kept), path)
                print(f"{name:6s} {split:10s} {len(rows):6d} lines -> {len(kept):7d} words "
                      f"({len(kept) / max(1, total):.0%} of {total}) in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
