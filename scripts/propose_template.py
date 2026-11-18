"""Propose field boxes from a blank form scan and write a template skeleton.

Finds empty rectangles (boxes, table cells) and underlines with free space above them,
which is where handwriting goes on most forms. Writes <name>.json (edit field names and
types: text | digits | number | amount | date | phone | code | name) and <name>_preview.jpg.

  python scripts/propose_template.py blank_invoice.jpg --name my_invoice --out templates/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from htr.fields import FieldSpec
from htr.forms import Template
from htr.segment import estimate_skew, normalize_background


def _lines(binary: np.ndarray, horizontal: bool, min_len: int, max_thickness: int = 7, bridge_px: int = 15) -> np.ndarray:
    """Ruling lines: bridge small breaks along the direction (dotted lines too), keep long thin runs."""
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, 1) if horizontal else (1, bridge_px))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1) if horizontal else (1, min_len))
    runs = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, bridge), cv2.MORPH_OPEN, k)
    # Rules are thin; bold text bridged into long runs is thick. Keep only thin runs.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    length = stats[:, cv2.CC_STAT_WIDTH] if horizontal else stats[:, cv2.CC_STAT_HEIGHT]
    thickness = stats[:, cv2.CC_STAT_AREA] / np.maximum(length, 1)  # robust to slightly slanted rules
    keep = thickness <= max_thickness
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def propose_fields(gray: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
    """Candidate fields (box, type) on a blank form: empty cells, rows of digit cells, tick boxes,
    and solid or dotted write-on lines with free space above them."""
    norm = normalize_background(gray)
    binary = (norm < 170).astype(np.uint8) * 255  # printed text
    faint = (norm < 215).astype(np.uint8) * 255  # thin rules turn light grey once scanned
    h, w = binary.shape
    hl = _lines(faint, True, max(30, w // 25))
    vl = _lines(faint, False, max(20, h // 60))
    # Dotted write-on lines, from the dots alone: joining every faint mark would also merge the
    # printed labels on the same row ("Prénom : ......... : الإسم") into the line.
    # Dots are printed at full strength; a stricter threshold keeps them separate from pale
    # backgrounds (watermarks, shaded panels) that would otherwise swallow them.
    dark = (norm < 190).astype(np.uint8) * 255
    n, labels, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    dot_max = max(4, int(round(6 * w / 1450)))
    is_dot = (st[:, cv2.CC_STAT_WIDTH] <= dot_max) & (st[:, cv2.CC_STAT_HEIGHT] <= dot_max)
    is_dot[0] = False
    dots = (is_dot[labels] * 255).astype(np.uint8)
    dotted = _lines(dots, True, max(30, w // 25), bridge_px=max(15, w // 50))
    grid = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
    text_ink = cv2.bitwise_and(binary, cv2.bitwise_not(grid))

    cells, checks = [], []
    # 1. Regions enclosed by ruling lines.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(grid), connectivity=4)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # page background around the form, not a cell
        if area < 0.8 * bw * bh or bh > 0.25 * h:
            continue
        inner = text_ink[y + 3: y + bh - 3, x + 3: x + bw - 3]
        if inner.size and (inner > 0).mean() > 0.01:
            continue  # holds printed text (a label or header), not an empty field
        box = (int(x), int(y), int(x + bw), int(y + bh))
        if bw < 0.06 * w and bh < 0.06 * h and 0.5 <= bw / max(bh, 1) <= 2.5 and min(bw, bh) >= 0.02 * w:
            checks.append(box)  # small square-ish box: tick box (or one digit cell, merged below)
        elif bh >= 0.015 * h and bw >= 0.04 * w:
            cells.append(box)

    # Rows of 3+ equal cells side by side (identifier / account numbers): one digits field.
    small = sorted(checks + [c for c in cells if c[2] - c[0] < 0.08 * w], key=lambda b: (b[1], b[0]))
    used, fields = set(), []
    for a in small:
        if a in used:
            continue
        row = [a]
        for b in small:
            last = row[-1]
            if b in used or b in row or b[0] <= last[0]:
                continue
            same_row = abs(b[1] - last[1]) <= 4 and abs(b[3] - last[3]) <= 4
            if same_row and 0 <= b[0] - last[2] <= 0.012 * w:
                row.append(b)
        if len(row) >= 3:
            used.update(row)
            fields.append(((row[0][0], min(r[1] for r in row), row[-1][2], max(r[3] for r in row)), "digits"))
    fields += [(c, "check") for c in checks if c not in used]
    fields += [(c, "text") for c in cells if c not in used]

    # 2. Write-on lines (solid or dotted) not part of a box.
    boxes = [b for b, _ in fields]
    candidates = []
    for mask in (hl, dotted):  # separately: OR-ing them would re-attach dots to label-bridged rows
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates += [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)]
    for x, y, bw, bh in candidates:
        if bh > 6 or bw < 0.08 * w:
            continue
        top = max(0, y - int(0.035 * h))
        if any(bx0 <= x + bw / 2 <= bx1 and by0 - 5 <= y <= by1 + 5 for bx0, by0, bx1, by1 in boxes):
            continue  # inside a box already proposed (or the same line found by both masks)
        ends = [vl[max(0, y - 4): y + bh + 4, max(0, ex - 4): ex + 5] for ex in (x, x + bw - 1)]
        if all(e.size and e.any() for e in ends):
            continue  # meets vertical rules at both ends: a box or table border, not a write-on line
        if (text_ink[top: y - 2, x: x + bw] > 0).mean() > 0.01:
            continue  # line under printed text (a rule or heading), not a write-on line
        fields.append(((int(x), int(top), int(x + bw), int(y + 4)), "text"))
        boxes.append(fields[-1][0])
    return sorted(fields, key=lambda f: (f[0][1] // 20, f[0][0]))


def propose_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    return [b for b, _ in propose_fields(gray)]


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Straighten a crooked scan (rule detection needs horizontal/vertical lines). Returns (image, angle)."""
    angle = estimate_skew(normalize_background(gray) < 170)
    if abs(angle) < 0.25:
        return gray, 0.0
    hh, ww = gray.shape
    rot = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    return cv2.warpAffine(gray, rot, (ww, hh), flags=cv2.INTER_LINEAR, borderValue=255), float(angle)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blank", type=Path, help="scan or render of the blank form")
    p.add_argument("--name", default=None)
    p.add_argument("--out", type=Path, default=Path("templates"))
    args = p.parse_args()
    name = args.name or args.blank.stem
    gray = cv2.imread(str(args.blank), cv2.IMREAD_GRAYSCALE)
    gray, angle = deskew(gray)
    if angle:
        print(f"deskewed by {angle:.1f} degrees")
    fields = [FieldSpec(f"field_{i + 1}", b, t) for i, (b, t) in enumerate(propose_fields(gray))]
    args.out.mkdir(parents=True, exist_ok=True)
    Template(name, gray, fields).save(args.out / f"{name}.json", f"{name}.png")
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for f in fields:
        x0, y0, x1, y1 = f.box
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(preview, f.name, (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(args.out / f"{name}_preview.jpg"), preview)
    print(f"{len(fields)} candidate fields -> {args.out / (name + '.json')} (edit names/types), "
          f"preview {args.out / (name + '_preview.jpg')}")


if __name__ == "__main__":
    main()
