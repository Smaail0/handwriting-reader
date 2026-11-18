"""Download handwriting datasets from Hugging Face into data/raw/<name>/<split>.parquet.

Line-level (training):
  iam          English lines        Teklia/IAM-line
  khatt        Arabic lines         Teklia/KHATT_v1.0
  rimes        French lines         Teklia/RIMES-2011-line
Page/paragraph-level (end-to-end evaluation):
  iam_pages    English paragraphs   vklinhhh/iam_paragraph
  khatt_pages  Arabic paragraphs    ahmedheakl/arocrbench_khattparagraph
Digits (synthetic handwritten number fields for invoices/forms):
  mnist        handwritten digits   ylecun/mnist (train/test writers are disjoint)

Licences: IAM, RIMES and KHATT are for non-commercial research. Check each source
before publishing results or using them commercially.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
DATASETS: dict[str, tuple[str, dict[str, str]]] = {
    "iam": ("Teklia/IAM-line", {s: f"data/{s}.parquet" for s in ("train", "validation", "test")}),
    "khatt": ("Teklia/KHATT_v1.0", {s: f"data/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")}),
    "rimes": ("Teklia/RIMES-2011-line", {s: f"data/{s}.parquet" for s in ("train", "validation", "test")}),
    "iam_pages": ("vklinhhh/iam_paragraph", {"all": "data/train-00000-of-00001.parquet"}),
    "mnist": ("ylecun/mnist", {s: f"mnist/{s}-00000-of-00001.parquet" for s in ("train", "test")}),
    "khatt_pages": ("ahmedheakl/arocrbench_khattparagraph", {"all": "data/train-00000-of-00001.parquet"}),
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {dest.parent.name}/{dest.name}: {done / total:6.1%} of {total / 1e6:.0f} MB")
                sys.stdout.flush()
    tmp.rename(dest)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", default=["iam"], choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    for name in args.datasets:
        repo, files = DATASETS[name]
        out = args.root / ("iam_line" if name == "iam" else name)  # keep the original IAM path
        out.mkdir(parents=True, exist_ok=True)
        for split, path in files.items():
            dest = out / f"{split}.parquet"
            if dest.exists():
                print(f"  {dest} already exists, skipping")
                continue
            download(HF.format(repo=repo, path=path), dest)


if __name__ == "__main__":
    main()
