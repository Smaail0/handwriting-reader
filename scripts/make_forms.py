"""Build a demo invoice template and synthetic filled-in "scans" with ground truth.

Handwriting comes only from held-out sources: IAM *test* lines for text fields and
MNIST *test* digits for number fields (writers never seen in training). Scans get random
rotation, perspective, scale, paper tint, lighting gradient, blur, noise and JPEG artifacts.

Outputs:
  templates/invoice_demo.png + .json     blank form + field boxes/types
  data/forms_synth/form_XXX.jpg          filled scans
  data/forms_synth/truth.json            {form file: {field: text}}
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from htr.fields import FieldSpec
from htr.forms import Template
from htr.synth import DigitBank, random_field_text, render_field

W, H = 1240, 1754  # A4 at 150 dpi
FONT, BOLD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX


def put(img, text, x, y, scale=0.7, font=FONT, thick=1):
    cv2.putText(img, text, (x, y), font, scale, 0, thick, cv2.LINE_AA)


def build_template() -> Template:
    img = np.full((H, W), 255, np.uint8)
    put(img, "ACME TRADING SARL", 80, 95, 1.3, BOLD, 2)
    put(img, "12 Rue de la Liberte, 1002 Tunis", 80, 135)
    put(img, "contact@acme.example  |  +216 71 000 000", 80, 165)
    put(img, "VAT ID 1234567/A/M/000", 80, 195)
    put(img, "INVOICE", 900, 110, 1.6, BOLD, 3)
    cv2.line(img, (80, 230), (1160, 230), 0, 2)
    fields: list[FieldSpec] = []

    def box(name, label, lx, ly, rect, ftype, multiline=False):
        put(img, label, lx, ly, 0.75, FONT, 1)
        cv2.rectangle(img, rect[:2], rect[2:], 0, 1)
        fields.append(FieldSpec(name, tuple(rect), ftype, multiline=multiline))

    box("invoice_no", "Invoice No.", 80, 315, (300, 270, 760, 340), "digits")
    box("date", "Date", 800, 315, (880, 270, 1160, 340), "date")
    box("customer", "Customer", 80, 415, (300, 370, 1160, 440), "text")
    box("phone", "Phone", 80, 515, (300, 470, 760, 540), "phone")
    # item table
    cv2.rectangle(img, (80, 590), (1160, 800), 0, 1)
    cv2.line(img, (80, 640), (1160, 640), 0, 1)
    for x in (680, 820):
        cv2.line(img, (x, 590), (x, 800), 0, 1)
    cv2.line(img, (80, 720), (1160, 720), 0, 1)
    put(img, "Description", 95, 625, 0.75, BOLD, 1)
    put(img, "Qty", 720, 625, 0.75, BOLD, 1)
    put(img, "Amount", 940, 625, 0.75, BOLD, 1)
    for row, (y0, y1) in enumerate([(640, 720), (720, 800)], start=1):
        fields.append(FieldSpec(f"desc_{row}", (82, y0 + 2, 678, y1 - 2), "text"))
        fields.append(FieldSpec(f"qty_{row}", (682, y0 + 2, 818, y1 - 2), "digits"))
        fields.append(FieldSpec(f"amount_{row}", (822, y0 + 2, 1158, y1 - 2), "amount"))
    box("subtotal", "Subtotal", 650, 885, (820, 840, 1160, 905), "amount")
    box("tax", "Tax", 650, 965, (820, 920, 1160, 985), "amount")
    box("total", "Total", 650, 1045, (820, 1000, 1160, 1065), "amount")
    put(img, "Notes", 80, 1135, 0.75, FONT, 1)
    cv2.rectangle(img, (80, 1150), (1160, 1340), 0, 1)
    fields.append(FieldSpec("notes", (80, 1150, 1160, 1340), "text", multiline=True))
    put(img, "Payment due within 30 days. Thank you for your business.", 80, 1450)
    put(img, "Bank: Example Bank  IBAN TN59 1000 6035 1835 9847 8831", 80, 1485)
    put(img, "Signature", 820, 1600, 0.75, FONT, 1)
    cv2.line(img, (820, 1660), (1160, 1660), 0, 1)
    return Template("invoice_demo", img, fields)


def _ink_crop(gray: np.ndarray) -> np.ndarray:
    """Tight crop to the ink, ignoring isolated specks (they would shrink the pasted text)."""
    ink = gray < 150
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    ys, xs = np.nonzero(big[labels])
    return gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] if len(xs) else gray


def split_words(gray: np.ndarray, text: str) -> list[tuple[str, np.ndarray]] | None:
    """Cut an IAM line image into words at its widest column gaps. Accepted only when
    there are exactly as many clear gaps as spaces in the transcript."""
    words = text.split()
    g = _ink_crop(gray)
    cols = (g < 150).sum(0) > 0
    gaps, start = [], None
    for x, inked in enumerate(cols):
        if not inked and start is None:
            start = x
        elif inked and start is not None:
            gaps.append((x - start, start, x))
            start = None
    k = len(words) - 1
    if k == 0:
        return [(words[0], g)]
    if len(gaps) < k:
        return None
    gaps.sort(reverse=True)
    if len(gaps) > k and gaps[k - 1][0] < 1.6 * gaps[k][0]:
        return None  # word gaps don't stand out from letter gaps: ambiguous
    cuts = sorted((a + b) // 2 for _, a, b in gaps[:k])
    bounds = [0] + cuts + [g.shape[1]]
    return [(w, g[:, bounds[i]: bounds[i + 1]]) for i, w in enumerate(words)]


def paste(canvas: np.ndarray, gray: np.ndarray, rect, rng, color, height_frac=(0.45, 0.7), y_center=None):
    x0, y0, x1, y1 = rect
    g = _ink_crop(gray)
    bh, bw = y1 - y0, x1 - x0
    th = int(bh * rng.uniform(*height_frac)) if y_center is None else int(height_frac[1])
    s = th / g.shape[0]
    if g.shape[1] * s > bw * 0.92:
        s = bw * 0.92 / g.shape[1]
    g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    x = x0 + int(rng.uniform(0.02, max(0.03, (bw - g.shape[1]) / bw - 0.02)) * bw)
    cy = (y0 + y1) / 2 if y_center is None else y_center
    y = int(np.clip(cy - g.shape[0] / 2 + rng.uniform(-0.08, 0.08) * bh, 0, canvas.shape[0] - g.shape[0]))
    x = int(np.clip(x, 0, canvas.shape[1] - g.shape[1]))
    gf = g.astype(np.float32)  # paper level -> transparent, darkest ink -> opaque (no grey boxes)
    bg, lo = np.percentile(gf, 85), np.percentile(gf, 2)
    alpha = np.clip((bg - gf) / max(bg - lo, 1.0), 0, 1)
    region = canvas[y: y + g.shape[0], x: x + g.shape[1]].astype(np.float32)
    region = region * (1 - alpha[..., None]) + np.asarray(color, np.float32) * alpha[..., None]
    canvas[y: y + g.shape[0], x: x + g.shape[1]] = region.astype(np.uint8)


def scan_effects(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    out_w = int(w * rng.uniform(0.85, 1.25))
    out_h = int(out_w * h / w)
    j = 0.02 * np.array([w, h])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * j
    ang = np.deg2rad(rng.uniform(-3, 3))
    c = np.array([w / 2, h / 2], np.float32)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = (dst - c) @ rot.T + c
    dst = dst * (out_w / w)
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    paper = np.array([rng.uniform(225, 250), rng.uniform(228, 252), rng.uniform(230, 255)])
    out = cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(int(v) for v in paper * 0.9))
    out = out.astype(np.float32) * (paper / 255.0)
    gx = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.05), out_w)[None, :, None]
    out = out * gx
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))
    out = out + rng.normal(0, rng.uniform(2, 7), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 92))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", type=Path, default=Path("data/forms_synth"))
    p.add_argument("--templates", type=Path, default=Path("templates"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.templates.mkdir(parents=True, exist_ok=True)

    tmpl = build_template()
    tmpl.save(args.templates / "invoice_demo.json", "invoice_demo.png")
    rng = np.random.default_rng(args.seed)
    bank = DigitBank(Path("data/raw/mnist/test.parquet"))  # unseen writers
    iam = pq.read_table("data/raw/iam_line/test.parquet").to_pylist()  # unseen writers
    plain = [r for r in iam if re.fullmatch(r"[A-Za-z ]+", r["text"])]
    segmented = []
    for r in plain:
        ws = split_words(np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")), r["text"])
        if ws:
            segmented.append(ws)
    print(f"{len(segmented)} of {len(plain)} letter-only IAM test lines split cleanly into words")

    def iam_line(lo, hi):
        """A phrase of consecutive words (same writer) whose text length is in [lo, hi]."""
        for _ in range(1000):
            ws = segmented[int(rng.integers(len(segmented)))]
            i = int(rng.integers(len(ws)))
            for j in range(i + 1, len(ws) + 1):
                txt = " ".join(w for w, _ in ws[i:j])
                if len(txt) > hi:
                    break
                if len(txt) >= lo and rng.random() < 0.5:
                    return txt, _ink_crop(np.hstack([im for _, im in ws[i:j]]))
        raise RuntimeError("no phrase found")

    truth = {}
    for k in range(args.n):
        canvas = cv2.cvtColor(tmpl.image, cv2.COLOR_GRAY2BGR)
        ink = [(int(rng.uniform(90, 160)), int(rng.uniform(20, 60)), int(rng.uniform(10, 40)))  # blue (BGR)
               if rng.random() < 0.6 else (int(rng.uniform(20, 60)),) * 3][0]
        t: dict[str, str] = {}
        for spec in tmpl.fields:
            if spec.name == "notes":
                lines = [iam_line(15, 40) for _ in range(2)]
                x0, y0, x1, y1 = spec.box
                for i, (txt, im) in enumerate(lines):
                    paste(canvas, im, spec.box, rng, ink, (0, 45), y_center=y0 + (i + 0.5) * (y1 - y0) / 2)
                t["notes"] = " ".join(txt for txt, _ in lines)
                continue
            if spec.type == "text":
                txt, im = iam_line(6, 22) if spec.name.startswith("desc") else iam_line(8, 26)
            else:
                kind = {"digits": "digits", "date": "date", "phone": "phone", "amount": "amount"}[spec.type]
                txt = str(int(rng.integers(1, 50))) if spec.name.startswith("qty") else random_field_text(kind, rng)
                if spec.name == "invoice_no":
                    txt = "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(4, 9))))
                im = np.asarray(render_field(txt, bank, rng))
            paste(canvas, im, spec.box, rng, ink)
            t[spec.name] = txt
        name = f"form_{k:03d}.jpg"
        cv2.imwrite(str(args.out / name), scan_effects(canvas, rng))
        truth[name] = t
    (args.out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"template -> {args.templates / 'invoice_demo.json'} ({len(tmpl.fields)} fields); "
          f"{args.n} scans -> {args.out}")


if __name__ == "__main__":
    main()
