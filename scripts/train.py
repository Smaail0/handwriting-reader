"""Train a CRNN-CTC line recognizer (optionally distilled from a larger teacher).

Examples:
  # English baseline (BiLSTM)
  python scripts/train.py --out runs/crnn_lstm
  # Light English model distilled from the baseline
  python scripts/train.py --out runs/lite_en --arch lite --separable --seq conv --channels 16 32 64 128 \\
      --hidden 128 --teacher runs/crnn_lstm/best.pt
  # Arabic (KHATT)
  python scripts/train.py --out runs/lite_ar --script arabic --data-dir data/raw/khatt --arch lite ...
  # One model for English + French + Arabic, plus synthetic code-switched lines (visual order)
  python scripts/train.py --out runs/joint --script mixed \\
      --data-dir data/raw/iam_line data/raw/rimes data/raw/khatt --synth-mixed 20000 --synth-fields 6000 ...
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from htr.charset import Charset
from htr.data import BucketBatchSampler, LineDataset, collate, load_split
from htr.metrics import cer, wer
from htr.model import CRNN, ModelConfig, count_params, load_checkpoint
from htr.mixed import MixedLineGenerator, SyntheticMixedDataset, WordBank
from htr.synth import DigitBank, SyntheticFieldDataset
from htr.text import SCRIPTS, from_output, is_rtl, to_target


def _single_thread_worker(_: int) -> None:
    torch.set_num_threads(1)  # avoid CPU oversubscription across data-loader workers


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, charset: Charset, device: torch.device, script: str) -> dict:
    model.eval()
    preds, refs = [], []
    for x, _, input_lengths, _, texts in loader:
        ids = model(x.to(device)).argmax(-1).cpu()
        for row, n in zip(ids, input_lengths):
            preds.append(from_output(charset.decode_greedy(row[: int(n)].tolist()), script))
        refs.extend(texts)
    exact = sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs))
    return {"cer": cer(preds, refs), "wer": wer(preds, refs), "exact": exact, "examples": list(zip(refs[:3], preds[:3]))}


def distill_loss(student: torch.Tensor, teacher: torch.Tensor, lengths: torch.Tensor, temperature: float) -> torch.Tensor:
    """Frame-level KL(teacher || student); both models emit one frame per 4 input columns."""
    t = temperature
    kl = F.kl_div(F.log_softmax(student / t, -1), F.log_softmax(teacher / t, -1), log_target=True,
                  reduction="none").sum(-1)  # (B, T)
    mask = torch.arange(kl.shape[1], device=kl.device)[None, :] < lengths.to(kl.device)[:, None]
    return (kl * mask).sum() / mask.sum() * t * t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, nargs="+", default=[Path("data/raw/iam_line")],
                   help="one or more line datasets (several only make sense with --script mixed)")
    p.add_argument("--script", choices=SCRIPTS, default="latin", help="arabic = right-to-left + Arabic normalization")
    p.add_argument("--out", type=Path, default=Path("runs/crnn_lstm"))
    p.add_argument("--arch", choices=["vgg", "lite"], default="vgg")
    p.add_argument("--separable", action="store_true", help="lite: depthwise-separable convs")
    p.add_argument("--seq", choices=["lstm", "conv"], default="lstm")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seq-layers", type=int, default=2)
    p.add_argument("--channels", type=int, nargs=4, default=[32, 64, 128, 256])
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--teacher", type=Path, default=None, help="checkpoint to distill from (same charset/height)")
    p.add_argument("--distill-weight", type=float, default=0.5, help="loss = (1-w)*CTC + w*KL")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint's weights (fine-tuning)")
    p.add_argument("--synth-fields", type=int, default=0,
                   help="add N synthetic handwritten number fields (dates, amounts, IDs, phones) to training")
    p.add_argument("--mnist-dir", type=Path, default=Path("data/raw/mnist"))
    p.add_argument("--synth-mixed", type=int, default=0,
                   help="add N synthetic code-switched lines per epoch (needs --script mixed and --words-dir)")
    p.add_argument("--words-dir", type=Path, default=Path("data/words"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pad-to", type=int, default=256,
                   help="round batch widths up to a multiple of this (fewer shapes = faster on MPS)")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-train", type=int, default=0, help="use only N training lines (smoke tests)")
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    if len(args.data_dir) > 1 and args.script != "mixed":
        raise SystemExit("several --data-dir need --script mixed (one charset, visual order)")
    if args.synth_mixed and args.script != "mixed":
        raise SystemExit("--synth-mixed needs --script mixed")
    train_parts = [load_split(d, "train", args.script) for d in args.data_dir]
    val_parts = {d.name: load_split(d, "validation", args.script) for d in args.data_dir}
    train_samples = [s for part in train_parts for s in part]
    charset_texts = [to_target(t, args.script) for _, t in train_samples]
    if args.synth_fields or args.synth_mixed:
        charset_texts.append("0123456789./,- ")  # synthetic numbers (dates, amounts, phones)
    charset = Charset.from_texts(charset_texts)

    teacher = None
    if args.teacher:
        teacher, t_charset, _ = load_checkpoint(args.teacher)
        if t_charset.chars != charset.chars or teacher.cfg.height != args.height:
            raise SystemExit("teacher must share the student's charset and input height")
        teacher = teacher.to(device).eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)

    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        val_parts = {k: v[: args.limit_val] for k, v in val_parts.items()}

    train_ds = LineDataset(train_samples, charset, args.height, train=True, script=args.script)
    val_sets = {k: LineDataset(v, charset, args.height, script=args.script) for k, v in val_parts.items()}
    ok = train_ds.ctc_feasible()
    widths = train_ds.widths()[ok]
    train_ds = Subset(train_ds, [i for i in range(len(ok)) if ok[i]])
    print(f"dropped {int((~ok).sum())} training lines too narrow for their label (kept {len(train_ds)})")
    field_val_loader = None
    if args.synth_fields:
        # MNIST train digits 0-50k render training fields, 50k-60k validation fields.
        # The MNIST test split (unseen writers) is kept for scripts/eval_fields.py.
        synth = SyntheticFieldDataset(DigitBank(args.mnist_dir / "train.parquet", 0, 50000), charset,
                                      args.synth_fields, seed=args.seed + 1, train=True, height=args.height)
        train_ds = ConcatDataset([train_ds, synth])
        widths = np.concatenate([widths, synth.widths()])
        field_val = SyntheticFieldDataset(DigitBank(args.mnist_dir / "train.parquet", 50000), charset, 600,
                                          seed=4242, train=False, height=args.height)
        print(f"added {len(synth)} synthetic field samples (+{len(field_val)} for validation)")
    if args.synth_mixed:
        digits = DigitBank(args.mnist_dir / "train.parquet", 0, 50000)
        gen = MixedLineGenerator(WordBank(args.words_dir, "train"), digits)
        mixed = SyntheticMixedDataset(gen, charset, args.synth_mixed, seed=args.seed + 2, train=True, height=args.height)
        train_ds = ConcatDataset([train_ds, mixed])
        widths = np.concatenate([widths, mixed.widths()])
        val_gen = MixedLineGenerator(WordBank(args.words_dir, "validation"),
                                     DigitBank(args.mnist_dir / "train.parquet", 50000))
        val_sets["mixed_synth"] = SyntheticMixedDataset(val_gen, charset, 600, seed=777, train=False,
                                                        height=args.height)
        print(f"added {len(mixed)} synthetic code-switched lines per epoch "
              f"(words: {gen.bank.n_words}; +600 for validation)")

    loader_kw = dict(collate_fn=partial(collate, pad_to=args.pad_to), num_workers=args.workers,
                     persistent_workers=args.workers > 0, worker_init_fn=_single_thread_worker)
    train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(widths, args.batch_size, seed=args.seed),
                              **loader_kw)
    # Validation sets are small; 2 workers each keeps the process count (and RAM) down with several sets.
    val_kw = {**loader_kw, "num_workers": min(2, args.workers), "persistent_workers": args.workers > 0}
    val_loaders = {k: DataLoader(v, batch_sampler=BucketBatchSampler(v.widths(), 32, shuffle=False), **val_kw)
                   for k, v in val_sets.items()}
    if args.synth_fields:
        field_val_loader = DataLoader(field_val, batch_sampler=BucketBatchSampler(field_val.widths(), 32, shuffle=False),
                                      **val_kw)

    cfg = ModelConfig(num_classes=charset.num_classes, height=args.height, channels=tuple(args.channels),
                      hidden=args.hidden, seq=args.seq, seq_layers=args.seq_layers, arch=args.arch,
                      separable=args.separable)
    model = CRNN(cfg)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        if init["charset"] != charset.chars or ModelConfig.from_dict(init["config"]) != cfg:
            raise SystemExit("--init checkpoint must have the same charset and model config")
        model.load_state_dict(init["model"])
        print(f"initialized from {args.init} (val CER {init.get('val_cer', float('nan')):.4f})")
    model = model.to(device)
    print(f"device={device} params={count_params(model) / 1e6:.2f}M train={len(train_ds)} "
          f"val={ {k: len(v) for k, v in val_sets.items()} } "
          f"classes={charset.num_classes} script={args.script} teacher={args.teacher}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_loader), pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    (args.out / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                      "model": cfg.to_dict(), "charset": charset.chars}, indent=2))
    log_file = open(args.out / "metrics.jsonl", "a")
    best_cer = math.inf
    w = args.distill_weight if teacher is not None else 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, targets, input_lengths, target_lengths, _ in train_loader:
            x = x.to(device)
            logits = model(x)
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.to(device), input_lengths, target_lengths)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = (1 - w) * loss + w * distill_loss(logits, t_logits, input_lengths, args.temperature)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            sched.step()
            total += loss.item()
            n += 1

        train_time = time.time() - t0
        per_set = {k: evaluate(model, v, charset, device, args.script) for k, v in val_loaders.items()}
        # One dataset: its scores. Several: the unweighted mean, so each language counts equally.
        metrics = {"cer": float(np.mean([m["cer"] for m in per_set.values()])),
                   "wer": float(np.mean([m["wer"] for m in per_set.values()])),
                   "examples": [e for m in per_set.values() for e in m["examples"][:1]]}
        record = {"epoch": epoch, "loss": total / max(1, n), "val_cer": metrics["cer"], "val_wer": metrics["wer"],
                  "lr": sched.get_last_lr()[0], "seconds": round(train_time, 1)}
        if len(per_set) > 1:
            record["val_cer_by_set"] = {k: round(m["cer"], 4) for k, m in per_set.items()}
        select = metrics["cer"]
        if field_val_loader is not None:
            fm = evaluate(model, field_val_loader, charset, device, args.script)
            record["field_cer"] = fm["cer"]
            record["field_exact"] = fm["exact"]
            select = metrics["cer"] + fm["cer"]  # best checkpoint balances prose and number fields
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        ckpt = {"model": model.state_dict(), "config": cfg.to_dict(), "charset": charset.chars,
                "script": args.script, "rtl": is_rtl(args.script),
                "epoch": epoch, "val_cer": metrics["cer"], "val_wer": metrics["wer"]}
        torch.save(ckpt, args.out / "last.pt")
        marker = ""
        if select < best_cer:
            best_cer = select
            torch.save(ckpt, args.out / "best.pt")
            marker = " *"
        fields = (f"  fields CER {record['field_cer']:.4f} exact {record['field_exact']:.3f}"
                  if "field_cer" in record else "")
        by_set = ("  [" + " ".join(f"{k} {v:.3f}" for k, v in record["val_cer_by_set"].items()) + "]"
                  if "val_cer_by_set" in record else "")
        print(f"epoch {epoch:3d}  loss {record['loss']:.3f}  val CER {metrics['cer']:.4f}  WER {metrics['wer']:.4f}"
              f"{by_set}{fields}  ({train_time:.0f}s){marker}", flush=True)
        if epoch == 1 or epoch % 10 == 0:
            for ref, pred in metrics["examples"]:
                print(f"    ref : {ref}\n    pred: {pred}")

    label = "IAM val CER + field CER" if args.synth_fields else "val CER"
    print(f"best {label} {best_cer:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
