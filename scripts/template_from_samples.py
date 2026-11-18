"""Build a form template from a few FILLED samples, when no blank form is available.

The printed form is identical on every copy while the handwriting differs, so after
aligning the samples to one reference, the lightest value per pixel (ink survives only where
every copy has ink) recovers the blank form. Field boxes are then proposed on it as in propose_template.py.

  python scripts/template_from_samples.py scan1.jpg scan2.jpg scan3.jpg --name my_form --out templates/

3 samples work; 5+ are better (more copies, fewer pixels where every copy happens to have ink). Edit field names and types
in the JSON afterwards, as with propose_template.py.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from htr.fields import FieldSpec
from htr.forms import RegistrationError, Template, register
from htr.segment import estimate_skew, normalize_background

_spec = importlib.util.spec_from_file_location("propose_template", Path(__file__).with_name("propose_template.py"))
_pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pt)


def estimate_blank(scans: list[np.ndarray], max_side: int = 2000) -> tuple[np.ndarray, list[int]]:
    """Median of the samples aligned to the sharpest-registering reference. Returns (blank, used indices)."""
    grays = []
    for s in scans:
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if s.ndim == 3 else s
        f = min(1.0, max_side / max(g.shape))
        grays.append(cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else g)
    # Reference = the straightest sample (rule detection on the result needs straight lines);
    # fall back to the others if too few samples align to it.
    skews = [abs(estimate_skew(normalize_background(g) < 170)) for g in grays]
    best = None
    for r in sorted(range(len(grays)), key=lambda i: skews[i]):
        ref = normalize_background(grays[r])
        warped, used, score = [ref], [r], 0
        for i, g in enumerate(grays):
            if i == r:
                continue
            try:
                w, _, inliers = register(normalize_background(g), ref)
            except RegistrationError:
                continue
            warped.append(w)
            used.append(i)
            score += inliers
        if best is None or (len(used), score) > (len(best[1]), best[2]):
            best = (warped, used, score)
        if len(used) == len(grays):
            break
    warped, used, _ = best
    if len(warped) < 3:
        raise SystemExit(f"only {len(warped)} samples could be aligned; need at least 3 of the same form")
    # Print is on every copy; handwriting changes from copy to copy, even inside the same box.
    # Per pixel, keep the LIGHTEST value across copies: ink survives only where every copy has
    # ink. Each copy is first darkened by 1 px (min filter), so thin printed lines survive the
    # small residual misalignment of registration.
    k = np.ones((3, 3), np.uint8)
    stack = np.stack([cv2.erode(wp, k) for wp in warped])
    blank = stack.max(axis=0)
    return blank.astype(np.uint8), used


def score_against(template: Template, truth: Template) -> dict:
    """Map proposed boxes into the truth template's frame and match them (IoU >= 0.5)."""
    try:
        _, h, _ = register(template.image, truth.image)
    except RegistrationError as e:
        return {"found": 0, "of": len(truth.fields), "missed": [f.name for f in truth.fields], "extra": len(template.fields),
                "type_ok": 0, "error": f"could not register the proposal to the truth template: {e}"}

    def warp(b):
        pts = cv2.perspectiveTransform(np.float32([[[b[0], b[1]]], [[b[2], b[3]]]]), h).reshape(-1, 2)
        return [pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]]

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / union if union else 0.0

    props = [(warp(f.box), f.type) for f in template.fields]
    found, type_ok, matched = [], 0, set()
    for f in truth.fields:
        free = [i for i in range(len(props)) if i not in matched]  # one proposal per truth field
        best = max(free, key=lambda i: iou(f.box, props[i][0]), default=None)
        if best is not None and iou(f.box, props[best][0]) >= 0.5:
            found.append(f.name)
            matched.add(best)
            type_ok += props[best][1] == f.type or (props[best][1] == "text" and f.type in ("name", "date", "phone", "code"))
    return {"found": len(found), "of": len(truth.fields), "missed": [f.name for f in truth.fields if f.name not in found],
            "extra": len(props) - len(matched), "type_ok": type_ok}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", type=Path, nargs="+", help="3 or more filled scans/photos of the same form")
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=Path("templates"))
    p.add_argument("--truth-template", type=Path, default=None, help="score the proposals against this template")
    args = p.parse_args()
    scans = [cv2.imread(str(s)) for s in args.samples]
    if any(s is None for s in scans):
        raise SystemExit("could not read one of the images")
    blank, used = estimate_blank(scans)
    blank, angle = _pt.deskew(blank)
    print(f"aligned {len(used)} of {len(scans)} samples (reference: {args.samples[used[0]].name})"
          + (f", deskewed by {angle:.1f} degrees" if angle else ""))
    fields = [FieldSpec(f"field_{i + 1}", b, t) for i, (b, t) in enumerate(_pt.propose_fields(blank))]
    args.out.mkdir(parents=True, exist_ok=True)
    Template(args.name, blank, fields).save(args.out / f"{args.name}.json", f"{args.name}.png")
    preview = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
    for f in fields:
        x0, y0, x1, y1 = f.box
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(preview, f.name, (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(args.out / f"{args.name}_preview.jpg"), preview)
    print(f"{len(fields)} candidate fields -> {args.out / (args.name + '.json')} (edit names/types), "
          f"preview {args.out / (args.name + '_preview.jpg')}")
    if args.truth_template:
        r = score_against(Template.load(args.out / f"{args.name}.json"), Template.load(args.truth_template))
        if "error" in r:
            print(r["error"])
        print(f"vs {args.truth_template.name}: found {r['found']}/{r['of']} fields (IoU>=0.5), "
              f"{r['extra']} extra proposals, type right or generic for {r['type_ok']}; missed {r['missed']}")


if __name__ == "__main__":
    main()
