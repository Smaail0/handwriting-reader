"""Fine-tune a recognizer on a few corrected lines from your own documents, in minutes.

Input: one or more line datasets exported by the browser page's correction mode, or built
by hand: a folder (or .zip) with `labels.jsonl`, one JSON object per line:
    {"image": "lines/p001_l03.png", "text": "corrected text"}
(images: grayscale or RGB line crops, dark ink on light paper).

What it does:
1. Drops duplicate lines, then holds out ~20% of your lines (at least 6), whole pages at a time
   when the export has several pages. Half of them pick the best epoch, the other half decide
   acceptance, so the decision is not biased by the choice of epoch.
2. Multiplies the rest: cuts them into words (CTC forced alignment with the base model) and
   recombines those words into new lines of the same handwriting (10x more lines).
3. Mixes in "replay" lines from the base model's original training data, so it does not
   forget everything else.
4. Fine-tunes from the base checkpoint with a low learning rate (new characters, e.g. "é" or
   "°" for an English model, are added to the output layer).
5. Only epochs that lose at most --max-forgetting CER points on the replay validation lines can
   be picked; the picked model is accepted only if it beats the base model on the acceptance half.

  python scripts/finetune.py --base runs/lite_en/best.pt --data exports/my_forms.zip --out runs/ft_my_forms

Outputs in --out: best.pt, model_fp32.onnx (+ .json sidecar for the browser), report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import tempfile
import time
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from htr.align import cut_words
from htr.charset import Charset
from htr.data import BucketBatchSampler, LineDataset, augment_geometric, augment_photometric, collate, load_split
from htr.export import export_onnx
from htr.inference import OnnxRecognizer
from htr.metrics import cer
from htr.mixed import _normalize_ink, compose_tokens, layout_line
from htr.model import CRNN, ModelConfig, load_checkpoint
from htr.preprocess import preprocess
from htr.text import from_output, is_rtl, normalize, to_target

REPLAY = {"latin": ["data/raw/iam_line"], "arabic": ["data/raw/khatt"],
          "mixed": ["data/raw/iam_line", "data/raw/rimes", "data/raw/khatt"]}


# ------------------------------------------------------------------ data

def read_dataset(path: Path, script: str) -> list[tuple[bytes, str, str]]:
    """(PNG bytes, normalized text, page key) from a correction export (folder or zip).

    Browser exports name crops lines/pNNN_lNN.png, so lines of one page share a key; other
    files are their own page."""
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        names = set(z.namelist())
        label_name = next(n for n in names if n.endswith("labels.jsonl"))
        root = label_name[: -len("labels.jsonl")]
        read = lambda rel: z.read(root + rel)  # noqa: E731
        lines = z.read(label_name).decode("utf-8").splitlines()
    else:
        read = lambda rel: (path / rel).read_bytes()  # noqa: E731
        lines = (path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        r = json.loads(ln)
        text = normalize(r["text"], script)
        if text:
            img = Image.open(io.BytesIO(read(r["image"]))).convert("L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            m = re.search(r"(p\d+)_l\d+\.png$", r["image"])
            out.append((buf.getvalue(), text, f"{path}:{m.group(1) if m else r['image']}"))
    return out


def dedupe(lines: list[tuple[bytes, str, str]]) -> list[tuple[bytes, str, str]]:
    """Drop repeated (image, text) pairs, e.g. a page added twice or overlapping exports."""
    seen, out = set(), []
    for data, text, page in lines:
        key = (hashlib.sha1(data).hexdigest(), text)
        if key not in seen:
            seen.add(key)
            out.append((data, text, page))
    return out


def split_holdout(lines: list, frac: float, rng: random.Random, min_lines: int = 6) -> tuple[list, list, bool]:
    """(held, train, split_by_page). Whole pages go to the holdout when there are at least 3 pages (neighbouring
    lines share writer and paper, so a line-level split would be optimistic)."""
    n_hold = max(min_lines, int(round(frac * len(lines))))
    pages: dict[str, list] = {}
    for item in lines:
        pages.setdefault(item[2], []).append(item)
    if len(pages) >= 3:
        keys = sorted(pages)
        rng.shuffle(keys)
        held, train = [], []
        for k in keys:
            (held if len(held) < n_hold else train).extend(pages[k])
        if len(train) >= len(lines) // 2:
            return held, train, len(pages) < len(lines)  # False when every "page" is one line
    items = list(lines)
    rng.shuffle(items)
    return items[:n_hold], items[n_hold:], False


def split_select_accept(held: list) -> tuple[list, list]:
    """Halves of the holdout: whole pages alternate between them when there are 2+ held pages,
    so the acceptance half shares no page with the half that picks the epoch."""
    pages: dict[str, list] = {}
    for item in held:
        pages.setdefault(item[2], []).append(item)
    if len(pages) >= 2:
        select, accept = [], []
        for k in sorted(pages, key=lambda k: -len(pages[k])):  # biggest first, into the smaller half
            (select if len(select) <= len(accept) else accept).extend(pages[k])
        return select, accept
    return held[0::2], held[1::2]


class RecombinedLines(Dataset):
    """New lines made of the user's own handwritten words, in random order."""

    def __init__(self, words: list[tuple[np.ndarray, str]], charset: Charset, script: str, n: int, height: int):
        self.words, self.charset, self.script, self.n, self.height = words, charset, script, n, height

    def __len__(self) -> int:
        return self.n

    def widths(self) -> np.ndarray:
        return np.full(self.n, 700)

    def __getitem__(self, i: int):
        rng = np.random.default_rng()
        for _ in range(20):
            picks = [self.words[int(k)] for k in rng.integers(0, len(self.words), int(rng.integers(2, 8)))]
            order = compose_tokens([t for _, t in picks])
            if order is not None:
                break
        else:
            picks = [self.words[int(rng.integers(len(self.words)))]]
            order = [0]
        body = 24.0 * rng.uniform(0.9, 1.1)
        pieces = [_normalize_ink(picks[k][0], body * rng.uniform(0.95, 1.05)) for k in order]
        img = layout_line(pieces, body, rng)
        logical = " ".join(t for _, t in picks)
        arr = augment_photometric(preprocess(augment_geometric(img, rng), self.height, rtl=is_rtl(self.script)), rng)
        return torch.from_numpy(arr), torch.tensor(self.charset.encode(to_target(logical, self.script))), logical


def ink_map(img: Image.Image) -> np.ndarray:
    arr = 1.0 - np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    bg = float(np.median(arr))
    return np.clip((arr - bg - 0.04) / max(0.2, 1.0 - bg - 0.04), 0.0, 1.0)


# ------------------------------------------------------------------ model

def extend_charset(model: CRNN, charset: Charset, new_chars: str) -> tuple[CRNN, Charset]:
    """Add output classes for characters the base model has never seen."""
    if not new_chars:
        return model, charset
    old = model.head
    cfg = ModelConfig.from_dict({**model.cfg.to_dict(), "num_classes": charset.num_classes + len(new_chars)})
    grown = CRNN(cfg)
    grown.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("head.")}, strict=False)
    with torch.no_grad():
        grown.head.weight[: old.out_features] = old.weight
        grown.head.bias[: old.out_features] = old.bias
        grown.head.weight[old.out_features:].normal_(0, 0.01)
        grown.head.bias[old.out_features:] = old.bias[1:].mean()  # start as an average character
    return grown, Charset(charset.chars + new_chars)


@torch.no_grad()
def line_cer(model: CRNN, samples, charset: Charset, script: str, height: int, device) -> tuple[float, list[str]]:
    model.eval()
    preds, refs = [], []
    for data, text in samples:
        x = torch.from_numpy(preprocess(Image.open(io.BytesIO(data)), height, rtl=is_rtl(script)))[None, None]
        ids = model(x.to(device)).argmax(-1)[0].cpu().tolist()
        preds.append(from_output(charset.decode_greedy(ids), script))
        refs.append(text)
    return cer(preds, refs), preds


# ------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="base checkpoint (.pt)")
    p.add_argument("--data", type=Path, nargs="+", required=True, help="correction exports (folder or .zip)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--recombined", type=int, default=0, help="recombined lines per epoch (default 10x your lines)")
    p.add_argument("--replay", type=int, default=0, help="original training lines per epoch (default 15x your lines, max 3000)")
    p.add_argument("--replay-dir", type=Path, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train-bn", action="store_true",
                   help="also update BatchNorm statistics (default: frozen; a few hundred lines would drag them away)")
    p.add_argument("--max-forgetting", type=float, default=0.015, help="max CER increase on replay validation lines")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    base, charset, ckpt = load_checkpoint(args.base)
    script, height = ckpt.get("script", "latin"), base.cfg.height
    raw = [s for d in args.data for s in read_dataset(d, script)]
    user = dedupe(raw)
    if len(user) < 12:
        raise SystemExit(f"need at least 12 distinct labelled lines, got {len(user)}")
    held, train, by_page = split_holdout(user, args.holdout, random.Random(args.seed))
    # Two halves: one picks the epoch, the other (never used for any choice) decides acceptance.
    select, accept = split_select_accept(held) if by_page else (held[0::2], held[1::2])
    held, select, accept, train = ([(d, t) for d, t, _ in part] for part in (held, select, accept, train))
    new_chars = "".join(sorted({c for _, t, _ in user for c in to_target(t, script)} - set(charset.chars)))
    print(f"{len(user)} lines ({len(raw) - len(user)} duplicates dropped; {len(train)} train, {len(held)} held out"
          f"{' by page' if by_page else ''}: {len(select)} select + {len(accept)} accept); script={script}; "
          f"new characters: {new_chars or 'none'}; device={device}")

    # Words of the user's own handwriting, cut with the base model.
    with tempfile.TemporaryDirectory() as tmp:
        onnx = Path(tmp) / "base.onnx"
        export_onnx(base, onnx, {"charset": charset.chars, "rtl": str(int(is_rtl(script))), "script": script})
        rec = OnnxRecognizer(onnx)
        words = []
        for data, text in train:
            for im, w in cut_words(Image.open(io.BytesIO(data)).convert("L"), text, rec, script, verify_cer=0.5) or []:
                words.append((ink_map(im), w))
    print(f"cut {len(words)} words from {len(train)} training lines")

    model, charset = extend_charset(base, charset, new_chars)
    replay_dirs = args.replay_dir if args.replay_dir is not None else [Path(d) for d in REPLAY[script]]
    replay_pool = [s for d in replay_dirs for s in load_split(d, "train", script)]
    replay_val = [s for d in replay_dirs for s in load_split(d, "validation", script)]
    random.shuffle(replay_val)
    replay_val = replay_val[:300]
    n_replay = args.replay or min(3000, 15 * len(train))
    n_recomb = args.recombined or (10 * len(train) if len(words) >= 10 else 0)
    oversample = max(1, n_replay // (4 * len(train)))

    base_cs = Charset(ckpt["charset"])
    base_sel, _ = line_cer(base.to(device), select, base_cs, script, height, device)
    base_acc, _ = line_cer(base, accept, base_cs, script, height, device)
    base_held, _ = line_cer(base, held, base_cs, script, height, device)
    base_replay, _ = line_cer(base, replay_val, base_cs, script, height, device)
    print(f"base model: held-out CER {base_held:.4f} (select {base_sel:.4f}, accept {base_acc:.4f}), "
          f"replay-val CER {base_replay:.4f}")

    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    loader_kw = dict(collate_fn=partial(collate, pad_to=256), num_workers=args.workers,
                     persistent_workers=False)
    best = (math.inf, None)
    history = []
    for epoch in range(1, args.epochs + 1):
        replay = random.sample(replay_pool, min(n_replay, len(replay_pool)))
        parts = [LineDataset(train * oversample, charset, height, train=True, script=script),
                 LineDataset(replay, charset, height, train=True, script=script)]
        if n_recomb:
            parts.append(RecombinedLines(words, charset, script, n_recomb, height))
        ds = ConcatDataset(parts)
        widths = np.concatenate([p_.widths() for p_ in parts])
        loader = DataLoader(ds, batch_sampler=BucketBatchSampler(widths, 16, seed=args.seed + epoch), **loader_kw)
        for g in optim.param_groups:  # cosine decay
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        model.train()
        if not args.train_bn:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()  # keep the base model's normalization statistics
        t0, total, n = time.time(), 0.0, 0
        for x, targets, in_len, tgt_len, _ in loader:
            logits = model(x.to(device))
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.to(device), in_len, tgt_len)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item()
            n += 1
        sel_cer, _ = line_cer(model, select, charset, script, height, device)
        replay_cer, _ = line_cer(model, replay_val, charset, script, height, device)
        history.append({"epoch": epoch, "loss": total / max(1, n), "select_cer": sel_cer, "replay_cer": replay_cer,
                        "seconds": round(time.time() - t0, 1)})
        ok = replay_cer <= base_replay + args.max_forgetting
        print(f"epoch {epoch}  loss {total / max(1, n):.3f}  select CER {sel_cer:.4f}  replay CER {replay_cer:.4f}"
              f"{'' if ok else '  (forgetting too much)'}  ({time.time() - t0:.0f}s)", flush=True)
        if ok and sel_cer < best[0]:
            best = (sel_cer, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, replay_cer, epoch)

    acc_cer = held_cer = None
    if best[1] is not None:
        model.load_state_dict(best[1])
        acc_cer, _ = line_cer(model, accept, charset, script, height, device)
        held_cer, _ = line_cer(model, held, charset, script, height, device)
    accepted = acc_cer is not None and acc_cer < base_acc
    report = {"base": str(args.base), "script": script, "lines": len(user), "train_lines": len(train),
              "held_out_lines": len(held), "words_cut": len(words), "new_characters": new_chars,
              "held_out_by_page": by_page, "duplicates_dropped": len(raw) - len(user),
              "base_held_cer": base_held, "base_select_cer": base_sel, "base_accept_cer": base_acc,
              "base_replay_cer": base_replay, "history": history,
              "best_epoch": best[3] if best[1] is not None else None,
              "best_select_cer": best[0] if best[1] is not None else None,
              "best_accept_cer": acc_cer, "best_held_cer": held_cer,
              "best_replay_cer": best[2] if best[1] is not None else None,
              "accepted": accepted, "minutes": round((time.time() - t_start) / 60, 1)}
    if accepted:
        model = model.cpu().eval()
        torch.save({"model": best[1], "config": model.cfg.to_dict(), "charset": charset.chars, "script": script,
                    "rtl": is_rtl(script), "val_cer": held_cer, "finetuned_from": str(args.base)}, args.out / "best.pt")
        meta = {"charset": charset.chars, "rtl": str(int(is_rtl(script))), "script": script}
        export_onnx(model, args.out / "model_fp32.onnx", meta)
        (args.out / "model_fp32.json").write_text(json.dumps(
            {"charset": charset.chars, "height": height, "downsample": 4, "script": script, "rtl": is_rtl(script)},
            ensure_ascii=False))
        print(f"ACCEPTED (epoch {best[3]}): acceptance-half CER {base_acc:.4f} -> {acc_cer:.4f}, all held-out "
              f"{base_held:.4f} -> {held_cer:.4f} (replay {base_replay:.4f} -> {best[2]:.4f}); "
              f"wrote {args.out / 'best.pt'} and model_fp32.onnx")
    else:
        why = ("no epoch stayed within the forgetting limit" if best[1] is None else
               f"the picked epoch did not beat the base model on the acceptance half ({acc_cer:.4f} vs {base_acc:.4f})")
        print(f"REJECTED: {why}; keeping the base model")
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
