"""Extract typed fields from scanned forms with a template; optionally score against truth.

Examples:
  # Read fields from scans (prints JSON)
  python scripts/extract_fields.py scan1.jpg scan2.jpg --template templates/invoice_demo.json \\
      --model latin=runs/lite_fields/model_int8_static.onnx
  # Benchmark on the synthetic invoices
  python scripts/extract_fields.py data/forms_synth/*.jpg --template templates/invoice_demo.json \\
      --model latin=runs/lite_fields/model_int8_static.onnx --truth data/forms_synth/truth.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from htr import fields as F
from htr.forms import FormExtractor, RegistrationError, Template
from htr.inference import OnnxRecognizer
from htr.lm import LMScorer
from htr.metrics import cer
from htr.text import detokenize


def canonical(ftype: str, text: str) -> str:
    """Comparison key: normalized value for typed fields, whitespace-normalized text otherwise."""
    if ftype == "text":
        return re.sub(r"\s+", " ", detokenize(text)).strip()
    v = F.normalize(ftype, text)
    return v if v is not None else f"<invalid:{text}>"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--model", action="append", required=True, help="script=path.onnx, or script:type=path.onnx to use another model for one field type (repeatable)")
    p.add_argument("--truth", type=Path, default=None)
    p.add_argument("--review-below", type=float, default=0.5, help="flag fields under this confidence")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--lm-for", action="append", default=[],
                   help="soft language model for a field type or name: type=models/lm_fr.json,lexicons/tn_places.txt "
                        "(.json = base model, .txt = phrase list); prefix with the script for non-Latin fields, "
                        "e.g. arabic:name=lexicons/names_ar.txt")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=1.5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="write all results as JSON")
    p.add_argument("--log", type=Path, default=Path("runs/field_results.jsonl"))
    args = p.parse_args()

    recs = {}
    for spec in args.model:
        script, path = spec.split("=", 1)
        recs[script] = OnnxRecognizer(path, threads=args.threads)
    lms = {}
    for spec in args.lm_for:
        key, paths = spec.split("=", 1)
        script = key.split(":", 1)[0] if ":" in key else "latin"
        paths = [Path(x) for x in paths.split(",")]
        base = next((x for x in paths if x.suffix == ".json"), None)
        lex = [x for x in paths if x.suffix == ".txt"] or None
        lms[key] = LMScorer.build(base, lex, script, alpha=args.lm_alpha, beta=args.lm_beta)
    extractor = FormExtractor(Template.load(args.template), recs, args.review_below, args.beam, lms)
    truth = json.loads(args.truth.read_text()) if args.truth else None

    all_results = {}
    stats = defaultdict(lambda: defaultdict(list))  # type -> metric -> values
    ms = []
    for path in args.images:
        img = cv2.imread(str(path))
        try:
            res = extractor(img)
        except RegistrationError as e:
            print(f"{path.name}: registration failed ({e})")
            all_results[path.name] = {"error": str(e)}
            continue
        ms.append(sum(res.ms.values()))
        all_results[path.name] = res.as_dict()
        if truth is None:
            print(json.dumps({path.name: {k: {"value": v.value, "confidence": round(v.confidence, 3),
                                                "review": v.review} for k, v in res.fields.items()}},
                             ensure_ascii=False))
            continue
        for name, ref in truth[path.name].items():
            r = res.fields[name]
            ok = canonical(r.type, r.text) == canonical(r.type, ref)
            s = stats[r.type]
            s["correct"].append(ok)
            s["review"].append(r.review)
            s["auto_ok"].append(ok if not r.review else None)
            s["conf"].append(r.confidence if r.value is not None else 0.0)
            s["pred"].append(detokenize(r.text) if r.type == "text" else r.text)
            s["ref"].append(ref)

    if args.out:
        args.out.write_text(json.dumps(all_results, indent=1, ensure_ascii=False))
    if truth is None:
        return
    print(f"\n{len(ms)} forms, {np.median(ms):.0f} ms/form (p50, registration + all fields)")
    print(f"{'type':8s} {'fields':>6s} {'exact':>7s} {'CER':>7s} {'review':>7s} {'auto-accepted acc':>18s}")
    tot = defaultdict(list)
    for ftype, s in sorted(stats.items()):
        auto = [x for x in s["auto_ok"] if x is not None]
        row = {"type": ftype, "n": len(s["correct"]), "exact": float(np.mean(s["correct"])),
               "cer": cer(s["pred"], s["ref"]), "review_rate": float(np.mean(s["review"])),
               "auto_acc": float(np.mean(auto)) if auto else float("nan")}
        print(f"{ftype:8s} {row['n']:6d} {row['exact']:7.1%} {row['cer']:7.3f} {row['review_rate']:7.1%} "
              f"{row['auto_acc']:17.1%}")
        for k in ("correct", "review"):
            tot[k] += s[k]
        tot["auto"] += auto
    print(f"{'all':8s} {len(tot['correct']):6d} {np.mean(tot['correct']):7.1%} {'':>7s} "
          f"{np.mean(tot['review']):7.1%} {np.mean(tot['auto']) if tot['auto'] else float('nan'):17.1%}")
    # Calibration: what a given review threshold would auto-accept, and how accurate that is.
    conf = np.array([c for s in stats.values() for c in s["conf"]])
    correct = np.array([c for s in stats.values() for c in s["correct"]])
    print("\nreview threshold sweep (all fields): threshold -> auto-accepted share, accuracy of auto-accepted")
    for th in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        acc = conf >= th
        print(f"  {th:4.2f} -> {acc.mean():6.1%} accepted, {correct[acc].mean() if acc.any() else float('nan'):6.1%} correct")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps({"models": args.model, "forms": len(ms), "ms_p50": float(np.median(ms)),
                            "exact": float(np.mean(tot["correct"])), "review_rate": float(np.mean(tot["review"])),
                            "auto_acc": float(np.mean(tot["auto"])) if tot["auto"] else None,
                            "review_below": args.review_below}) + "\n")


if __name__ == "__main__":
    main()
