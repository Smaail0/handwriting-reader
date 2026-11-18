"""Build a character n-gram language model from TRAINING-split transcripts.

  python scripts/build_lm.py --data-dir data/raw/iam_line --script latin --out models/lm_en.json
  python scripts/build_lm.py --data-dir data/raw/iam_line data/raw/rimes data/raw/khatt --script mixed \\
      --out models/lm_mixed.json

Texts are normalized and put in the recognizer's output order (htr.text.to_target) for the
given script, so the model can score prefixes during beam search. Only the train split is
read, and by default any training line that shares a 20-character passage with a validation
or test line is dropped: KHATT and RIMES reuse the same texts across writers (97% / 37% of
KHATT / RIMES test lines share a 30-character passage with training text; 99% / 69% at 20), so a model of
the raw training text would partly memorize the test answers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from htr.lm import CharNgramLM
from htr.text import normalize, to_target


def _overlaps(t: str, held: set[str], held_text: list[str], span: int) -> bool:
    """Shares a span-long passage with held-out text, or (when shorter) occurs inside it."""
    if len(t) < span:
        return any(t in h for h in held_text)
    return any(t[i:i + span] in held for i in range(len(t) - span + 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, nargs="+", required=True)
    p.add_argument("--script", required=True, choices=["latin", "arabic", "mixed"])
    p.add_argument("--order", type=int, default=6)
    p.add_argument("--extra-text", type=Path, nargs="*", default=[], help="more plain-text files, one line each")
    p.add_argument("--span", type=int, default=20, help="overlap test: shared passage length (0 = keep all)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    texts = []
    # Passages of every given dataset's validation/test lines (across datasets, so a sentence
    # shared between two datasets cannot leak either).
    held, held_text = set(), []
    if args.span:
        for d in args.data_dir:
            for split in ("validation", "test"):
                for t in pq.read_table(d / f"{split}.parquet", columns=["text"]).column("text").to_pylist():
                    t = normalize(t, args.script)
                    held_text.append(t)
                    held.update(t[i:i + args.span] for i in range(max(1, len(t) - args.span + 1)))
    for d in args.data_dir:
        rows = [normalize(t, args.script) for t in pq.read_table(d / "train.parquet", columns=["text"]).column("text").to_pylist()]
        kept = rows
        if args.span:
            kept = [t for t in rows if not _overlaps(t, held, held_text, args.span)]
        texts += [to_target(t, args.script) for t in kept]
        print(f"{d}: {len(rows)} training lines, {len(kept)} kept (no passage shared with validation/test)")
    for f in args.extra_text:
        lines = [normalize(ln, args.script) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.span:
            lines = [t for t in lines if not _overlaps(t, held, held_text, args.span)]
        texts += [to_target(normalize(t, args.script), args.script) for t in lines]
        print(f"{f}: {len(lines)} lines")
    lm = CharNgramLM.train(texts, args.order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(args.out)
    n_ctx = sum(len(level) for level in lm.counts)
    print(f"order {args.order}, {len(lm.vocab)} characters, {n_ctx} contexts -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
