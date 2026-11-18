"""Template-based field extraction for scanned forms (invoices, delivery notes, claims...).

For a known form layout (a "template": the blank form image plus named field boxes):

1. Register the scan to the template: ORB keypoints + RANSAC homography, so rotation,
   scale, shift and mild perspective from scanning or photographing are undone.
2. Subtract the template: pixels that are printed on the blank form (labels, boxes,
   rules) are masked out of each field box, leaving only the handwriting.
3. Read each field with the line recognizer, using decoding constrained to the field's
   type (date, amount, digits, ...), and normalize the value (ISO dates, decimal amounts).
4. Flag fields for human review when the reading is invalid or low-confidence.

Everything is classical CV + the small CTC recognizer: no GPU, tens of ms per field.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import fields as F
from .fields import FieldSpec
from .inference import OnnxRecognizer
from .segment import normalize_background, segment_lines
from .text import detokenize


@dataclass
class Template:
    name: str
    image: np.ndarray  # blank form, grayscale uint8
    fields: list[FieldSpec]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"template {self.name!r}: duplicate field names {dupes}")

    @classmethod
    def load(cls, path: str | Path) -> Template:
        path = Path(path)
        spec = json.loads(path.read_text())
        img = cv2.imread(str(path.parent / spec["image"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path.parent / spec["image"])
        return cls(spec["name"], img, [FieldSpec(**{**f, "box": tuple(f["box"])}) for f in spec["fields"]])

    def save(self, path: str | Path, image_name: str) -> None:
        path = Path(path)
        cv2.imwrite(str(path.parent / image_name), self.image)
        path.write_text(json.dumps({"name": self.name, "image": image_name,
                                    "fields": [asdict(f) for f in self.fields]}, indent=2))


@dataclass
class FieldResult:
    name: str
    type: str
    text: str  # raw reading
    value: str | None  # normalized value (None if empty or not interpretable)
    confidence: float
    review: bool  # needs a human look
    empty: bool = False


@dataclass
class FormResult:
    fields: dict[str, FieldResult]
    inliers: int  # registration matches supporting the alignment
    ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"fields": {k: asdict(v) for k, v in self.fields.items()}, "inliers": self.inliers, "ms": self.ms}


class RegistrationError(RuntimeError):
    pass


def register(scan: np.ndarray, template: np.ndarray, work_side: int = 1200) -> tuple[np.ndarray, np.ndarray, int]:
    """Warp `scan` (gray) into the template's frame. Returns (warped, H scan->template, inliers)."""
    st = min(1.0, work_side / max(template.shape))
    ss = min(1.0, work_side / max(scan.shape))
    small_t = cv2.resize(template, None, fx=st, fy=st, interpolation=cv2.INTER_AREA)
    small_s = cv2.resize(scan, None, fx=ss, fy=ss, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=5000)
    kt, dt = orb.detectAndCompute(small_t, None)
    ks, ds = orb.detectAndCompute(small_s, None)
    if dt is None or ds is None:
        raise RegistrationError("no features found")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dt, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        raise RegistrationError(f"only {len(good)} feature matches")
    src = np.float32([ks[m.queryIdx].pt for m in good])
    dst = np.float32([kt[m.trainIdx].pt for m in good])
    hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if hs is None or int(mask.sum()) < 12:
        raise RegistrationError("no consistent alignment")
    h = np.diag([1 / st, 1 / st, 1.0]) @ hs @ np.diag([ss, ss, 1.0])
    warped = cv2.warpPerspective(scan, h, (template.shape[1], template.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderValue=255)
    return warped, h, int(mask.sum())


def field_ink(scan_norm: np.ndarray, tmpl_norm: np.ndarray, box, ink_thresh: int = 180,
              printed_thresh: int = 170, tolerance: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Clean crop of the handwriting in `box`: printed template content removed.

    Returns (crop with only handwriting on white, ink mask)."""
    x0, y0, x1, y1 = box
    s = scan_norm[y0:y1, x0:x1]
    t = tmpl_norm[y0:y1, x0:x1]
    printed = (t < printed_thresh).astype(np.uint8)
    printed = cv2.dilate(printed, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)) > 0
    ink = (s < ink_thresh) & ~printed
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6  # drop specks and registration residue
    hh, ww = ink.shape
    for i in range(1, n):  # slivers of the field's own box border left by small registration errors
        x, y, cw, ch, _ = stats[i]
        at_edge = x <= 1 or y <= 1 or x + cw >= ww - 1 or y + ch >= hh - 1
        if at_edge and ((ch >= 0.5 * hh and cw <= 5) or (cw >= 0.5 * ww and ch <= 5)):
            keep[i] = False
    ink = keep[labels]
    soft = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # keep anti-aliased edges
    clean = np.full_like(s, 255)
    clean[soft] = np.maximum(s[soft], 0)
    return clean, ink


class FormExtractor:
    def __init__(self, template: Template, recognizers: dict[str, OnnxRecognizer], review_below: float = 0.5,
                 beam: int = 16, lms: dict | None = None):
        self.template = template
        self.tmpl_norm = normalize_background(template.image)
        self.recognizers = recognizers
        self.review_below = review_below
        self.beam = beam
        # Soft language models (htr.lm.LMScorer) keyed by field name, "script:type" or (Latin
        # fields only) bare type ("name", "text"): a model's text order is script-specific.
        self.lms = lms or {}
        # Built once: a lexicon constraint precomputes all prefixes of its word list.
        self.constraints = {f.name: F.constraint_for(f.type, f.lexicon) for f in template.fields}

    def _constraint(self, spec: FieldSpec):
        """Cached constraint for a template field; built on the fly for any other spec."""
        if spec.name in self.constraints and any(f is spec for f in self.template.fields):
            return self.constraints[spec.name]
        return F.constraint_for(spec.type, spec.lexicon)

    @staticmethod
    def _value_key(field_type: str, text: str) -> str | None:
        if field_type in ("text", "free", "name"):
            return " ".join(detokenize(text).split()) or None
        return F.normalize(field_type, text)

    def read_field(self, clean: np.ndarray, ink: np.ndarray, spec: FieldSpec) -> FieldResult:
        # A recognizer keyed "script:type" (e.g. "latin:text") overrides the script's default for that type.
        if spec.type == "check":
            x0, y0, x1, y1 = spec.box
            inner = ink[3: max(4, y1 - y0 - 3), 3: max(4, x1 - x0 - 3)]  # ignore the box border
            frac = float(inner.mean()) if inner.size else 0.0
            ticked = frac > 0.04
            # Clear when far from the threshold; ambiguous marks (smudges, partial ticks) go to review.
            conf = float(min(1.0, abs(frac - 0.04) / 0.04))
            return FieldResult(spec.name, spec.type, "x" if ticked else "", "1" if ticked else "0", conf,
                               conf < self.review_below, empty=not ticked)
        rec = self.recognizers.get(f"{spec.script}:{spec.type}") or self.recognizers[spec.script]
        lm = (self.lms.get(spec.name) or self.lms.get(f"{spec.script}:{spec.type}")
              or (self.lms.get(spec.type) if spec.script == "latin" else None))
        if ink.sum() < 30:
            return FieldResult(spec.name, spec.type, "", None, 1.0, False, empty=True)
        if spec.multiline:
            lines = segment_lines(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), rtl=rec.rtl) or []
            constraint = self._constraint(spec)
            reads = [rec.read(Image.fromarray(ln.image), constraint, self.beam, lm) for ln in lines]
            text = " ".join(r.text for r in reads)
            conf = min((r.confidence for r in reads), default=0.0)
            valid = all(r.valid for r in reads)
        else:
            ys, xs = np.nonzero(ink)
            pad = max(4, int(0.15 * (ys.max() - ys.min() + 1)))
            y0, y1 = max(0, ys.min() - pad), min(ink.shape[0], ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(ink.shape[1], xs.max() + pad + 1)
            constraint = self._constraint(spec)
            d = rec.read(Image.fromarray(clean[y0:y1, x0:x1]), constraint, self.beam, lm)
            text, conf, valid = d.text, d.confidence, d.valid
            # Confidence in the value, not the exact string: sum every hypothesis that means the
            # same thing (spacing variants of a date, "3 000" vs "3000", ...).
            key = self._value_key(spec.type, text)
            if key is not None and d.alternatives:
                # Only readings that are themselves valid count: "1034l" must not lend its
                # probability to "1034" just because normalization strips the letter.
                conf = sum(q for t, q in d.alternatives
                           if (constraint is None or constraint.complete(t)) and self._value_key(spec.type, t) == key)
        if spec.type in ("text", "free") and not rec.rtl:
            text = detokenize(text)
        value = F.normalize(spec.type, text) if text else None
        review = (not valid) or conf < self.review_below or (value is None)
        return FieldResult(spec.name, spec.type, text, value, conf, review)

    def __call__(self, scan_bgr: np.ndarray) -> FormResult:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2GRAY) if scan_bgr.ndim == 3 else scan_bgr
        warped, _, inliers = register(gray, self.template.image)
        warped_norm = normalize_background(warped)
        t1 = time.perf_counter()
        results = {}
        for spec in self.template.fields:
            clean, ink = field_ink(warped_norm, self.tmpl_norm, spec.box)
            results[spec.name] = self.read_field(clean, ink, spec)
        t2 = time.perf_counter()
        return FormResult(results, inliers, {"register": (t1 - t0) * 1000, "read": (t2 - t1) * 1000})
