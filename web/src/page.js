// Page steps for phone photos, a port of src/htr/page.py:
//   findPaper      the sheet = large bright, low-saturation region; flattened (perspective),
//                  everything outside it painted white. Scans/tight crops are left alone.
//   TextDetector   PP-OCRv4 mobile DBNet (models/ppocr_det_v4.onnx, Apache-2.0): text probability map.
//   segmentLinesDB text regions -> lines; ink outside every region (drawings, desk) is ignored.
//   segmentPage    all of it; line polygons/boxes in the input image's coordinates.
// Not bit-exact with OpenCV (square instead of elliptic dilation, 8- instead of 4-connected
// background); test/page_parity.mjs measures the agreement with Python on line boxes.

import { toGray } from "./preprocess.js";
import { components, extract, gaussianBlur, morph, normalizeBackground, otsu, readingOrder, resizeArea, runFilter } from "./segment.js";

// ------------------------------------------------------------------ image helpers

/** RGBA ImageData-like {data, width, height} -> per-channel planes. */
function planes(img) {
  const n = img.width * img.height;
  const r = new Uint8Array(n); const g = new Uint8Array(n); const b = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { r[i] = img.data[j]; g[i] = img.data[j + 1]; b[i] = img.data[j + 2]; }
  const wrap = (d) => ({ data: d, width: img.width, height: img.height });
  return [wrap(r), wrap(g), wrap(b)];
}

function resizeRGBA(img, w, h) {
  const [r, g, b] = planes(img).map((p) => resizeArea(p, w, h));
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) { out[j] = r.data[i]; out[j + 1] = g.data[i]; out[j + 2] = b.data[i]; out[j + 3] = 255; }
  return { data: out, width: w, height: h };
}

/** Bilinear resize of a float map (OpenCV half-pixel convention). */
function resizeFloat(src, sw, sh, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(Math.max((y + 0.5) * (sh / h) - 0.5, 0), sh - 1);
    const y0 = Math.floor(fy); const y1 = Math.min(y0 + 1, sh - 1); const dy = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(Math.max((x + 0.5) * (sw / w) - 0.5, 0), sw - 1);
      const x0 = Math.floor(fx); const x1 = Math.min(x0 + 1, sw - 1); const dx = fx - x0;
      const top = src[y0 * sw + x0] * (1 - dx) + src[y0 * sw + x1] * dx;
      const bot = src[y1 * sw + x0] * (1 - dx) + src[y1 * sw + x1] * dx;
      out[y * w + x] = top * (1 - dy) + bot * dy;
    }
  }
  return out;
}

// ------------------------------------------------------------------ geometry

function convexHull(pts) { // [[x, y]] -> hull, counter-clockwise (monotone chain)
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = []; const upper = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
  for (const q of p.slice().reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

const polyArea = (q) => Math.abs(q.reduce((s, a, i) => { const b = q[(i + 1) % q.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;
const perimeter = (q) => q.reduce((s, a, i) => { const b = q[(i + 1) % q.length]; return s + Math.hypot(b[0] - a[0], b[1] - a[1]); }, 0);

function segDist(p, a, b) {
  const vx = b[0] - a[0]; const vy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (vx * vx + vy * vy || 1)));
  return Math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy);
}

/** Douglas-Peucker on a closed polygon (split at the two farthest-apart vertices). */
function approxClosed(poly, eps) {
  const n = poly.length;
  let i0 = 0; let i1 = 0; let best = -1;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d = Math.hypot(poly[i][0] - poly[j][0], poly[i][1] - poly[j][1]);
    if (d > best) { best = d; i0 = i; i1 = j; }
  }
  const dp = (pts) => {
    if (pts.length < 3) return [pts[0]];
    let idx = 0; let dmax = 0;
    for (let k = 1; k < pts.length - 1; k++) { const d = segDist(pts[k], pts[0], pts.at(-1)); if (d > dmax) { dmax = d; idx = k; } }
    if (dmax <= eps) return [pts[0]];
    return dp(pts.slice(0, idx + 1)).concat(dp(pts.slice(idx)));
  };
  const a = poly.slice(i0, i1 + 1);
  const b = poly.slice(i1).concat(poly.slice(0, i0 + 1));
  return dp(a).concat(dp(b));
}

function insideConvex(q, p) { // q counter-clockwise or clockwise; true if p inside
  let sign = 0;
  for (let i = 0; i < q.length; i++) {
    const a = q[i]; const b = q[(i + 1) % q.length];
    const c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (c !== 0) { if (sign && Math.sign(c) !== sign) return false; sign = Math.sign(c); }
  }
  return true;
}

/** Signed distance like cv2.pointPolygonTest(measureDist=True): >= 0 inside. */
function polyTest(q, p) {
  let d = Infinity;
  for (let i = 0; i < q.length; i++) d = Math.min(d, segDist(p, q[i], q[(i + 1) % q.length]));
  return insideConvex(q, p) ? d : -d;
}

function minAreaRect(hull) { // rotating calipers over hull edges -> 4 corners
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i]; const b = hull[(i + 1) % hull.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const ux = (b[0] - a[0]) / len; const uy = (b[1] - a[1]) / len;
    let u0 = Infinity; let u1 = -Infinity; let v0 = Infinity; let v1 = -Infinity;
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy; const v = -p[0] * uy + p[1] * ux;
      u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    const area = (u1 - u0) * (v1 - v0);
    if (!best || area < best.area) best = { area, ux, uy, u0, u1, v0, v1 };
  }
  const { ux, uy, u0, u1, v0, v1 } = best;
  const pt = (u, v) => [u * ux - v * uy, u * uy + v * ux];
  return [pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)];
}

function orderCorners(q) { // top-left, top-right, bottom-right, bottom-left
  const s = q.map((p) => p[0] + p[1]); const d = q.map((p) => p[1] - p[0]);
  const at = (arr, f) => q[arr.indexOf(f(...arr))];
  return [at(s, Math.min), at(d, Math.min), at(s, Math.max), at(d, Math.max)];
}

/** Homography mapping 4 src points to 4 dst points (8x8 Gaussian elimination). */
function homography(src, dst) {
  const A = []; const bv = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i]; const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); bv.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); bv.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let piv = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [bv[c], bv[piv]] = [bv[piv], bv[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      bv[r] -= f * bv[c];
    }
  }
  const h = bv.map((v, i) => v / A[i][i]);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

function invert3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h; const B = -(d * i - f * g); const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}

const applyH = (m, x, y) => {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w];
};

// ------------------------------------------------------------------ 1. paper

/**
 * RGBA page -> { image (flattened RGBA), H (3x3, photo -> flattened) } or null when the
 * image already is the page or no sheet stands out from the background.
 */
export function findPaper(img, { work = 640, minArea = 0.15, minBackground = 0.03, maxSat = 70 } = {}) {
  const { width: W, height: H } = img;
  const s = Math.min(1, work / Math.max(W, H));
  const sw = Math.max(1, Math.round(W * s)); const sh = Math.max(1, Math.round(H * s));
  const small = s < 1 ? resizeRGBA(img, sw, sh) : img;
  const [r, g, b] = planes(small).map((p) => gaussianBlur(p, 1.1)); // cv2.GaussianBlur((5, 5), 0)
  const n = sw * sh;
  const val = new Uint8Array(n); const sat = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const mx = Math.max(r.data[i], g.data[i], b.data[i]); const mn = Math.min(r.data[i], g.data[i], b.data[i]);
    val[i] = mx; sat[i] = mx ? Math.round((255 * (mx - mn)) / mx) : 0;
  }
  const thr = otsu(val);
  let mask = { data: new Uint8Array(n), width: sw, height: sh };
  for (let i = 0; i < n; i++) mask.data[i] = val[i] > 0.6 * thr && sat[i] < maxSat ? 1 : 0;
  const k = Math.max(5, Math.floor(0.03 * Math.max(sw, sh)) | 1);
  mask = morph(morph(mask, k, true), k, false); // close: fill ink
  mask = morph(morph(mask, k, false), k, true); // open: cut bridges
  // Background = non-paper regions touching the photo's edge.
  const inv = new Uint8Array(n);
  for (let i = 0; i < n; i++) inv[i] = mask.data[i] ? 0 : 1;
  const bg = components(inv, sw, sh);
  let edgeArea = 0;
  for (let l = 1; l < bg.n; l++) {
    const st = bg.stats[l];
    if (st.x0 === 0 || st.y0 === 0 || st.x1 === sw - 1 || st.y1 === sh - 1) edgeArea += st.area;
  }
  if (edgeArea < minBackground * n) return null;
  const pc = components(mask.data, sw, sh);
  let big = 0;
  for (let l = 1; l < pc.n; l++) if (!big || pc.stats[l].area > pc.stats[big].area) big = l;
  if (!big || pc.stats[big].area < minArea * n) return null;
  const pts = [];
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const i = y * sw + x;
      if (pc.labels[i] !== big) continue;
      if (x === 0 || y === 0 || x === sw - 1 || y === sh - 1 || pc.labels[i - 1] !== big || pc.labels[i + 1] !== big
        || pc.labels[i - sw] !== big || pc.labels[i + sw] !== big) pts.push([x, y]);
    }
  }
  const hull = convexHull(pts);
  if (hull.length < 3) return null;
  const peri = perimeter(hull);
  let quad = null;
  for (const eps of [0.01, 0.02, 0.03, 0.05, 0.08]) {
    const approx = approxClosed(hull, eps * peri);
    if (approx.length === 4) {
      if (hull.every((p) => polyTest(approx, p) >= -0.01 * peri)) quad = approx;
      break;
    }
  }
  if (!quad) quad = minAreaRect(hull);
  const [tl, tr, br, bl] = orderCorners(quad.map(([x, y]) => [x / s, y / s]));
  const ow = Math.floor(Math.max(Math.hypot(tr[0] - tl[0], tr[1] - tl[1]), Math.hypot(br[0] - bl[0], br[1] - bl[1])));
  const oh = Math.floor(Math.max(Math.hypot(bl[0] - tl[0], bl[1] - tl[1]), Math.hypot(br[0] - tr[0], br[1] - tr[1])));
  if (ow < 100 || oh < 100) return null;
  const Hm = homography([tl, tr, br, bl], [[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]]);
  const Hi = invert3(Hm);
  const hullFull = hull.map(([x, y]) => [x / s, y / s]);
  const out = new Uint8ClampedArray(ow * oh * 4);
  const src = img.data;
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const [fx, fy] = applyH(Hi, x, y);
      const j = (y * ow + x) * 4;
      if (!insideConvex(hullFull, [fx, fy])) { out[j] = out[j + 1] = out[j + 2] = out[j + 3] = 255; continue; }
      const cx = Math.min(Math.max(fx, 0), W - 1); const cy = Math.min(Math.max(fy, 0), H - 1); // replicate border
      const x0 = Math.floor(cx); const y0 = Math.floor(cy);
      const x1 = Math.min(x0 + 1, W - 1); const y1 = Math.min(y0 + 1, H - 1);
      const dx = cx - x0; const dy = cy - y0;
      for (let c = 0; c < 3; c++) {
        const v = (src[(y0 * W + x0) * 4 + c] * (1 - dx) + src[(y0 * W + x1) * 4 + c] * dx) * (1 - dy)
          + (src[(y1 * W + x0) * 4 + c] * (1 - dx) + src[(y1 * W + x1) * 4 + c] * dx) * dy;
        out[j + c] = Math.round(v);
      }
      out[j + 3] = 255;
    }
  }
  return { image: { data: out, width: ow, height: oh }, H: Hm };
}

// ------------------------------------------------------------------ 2. text detector

export class TextDetector {
  constructor(ort, session) { this.ort = ort; this.session = session; this.input = session.inputNames[0]; }

  /** @param model URL or bytes of ppocr_det_v4.onnx */
  static async load(ort, model = "models/ppocr_det_v4.onnx") {
    return new TextDetector(ort, await ort.InferenceSession.create(model, { graphOptimizationLevel: "all" }));
  }

  /** RGBA image -> Float32Array text probability at the image's resolution. */
  async probMap(img, side = 1280) {
    const { width: w, height: h } = img;
    const s = Math.min(1, side / Math.max(w, h));
    const nh = Math.max(32, Math.round((h * s) / 32) * 32); const nw = Math.max(32, Math.round((w * s) / 32) * 32);
    const x = resizeRGBA(img, nw, nh);
    const n = nw * nh;
    const t = new Float32Array(3 * n); // BGR planes, (v / 255 - 0.5) / 0.5
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      t[i] = x.data[j + 2] / 127.5 - 1; t[n + i] = x.data[j + 1] / 127.5 - 1; t[2 * n + i] = x.data[j] / 127.5 - 1;
    }
    const out = await this.session.run({ [this.input]: new this.ort.Tensor("float32", t, [1, 3, nh, nw]) });
    return resizeFloat(out[this.session.outputNames[0]].data, nw, nh, w, h);
  }
}

// ------------------------------------------------------------------ 3. lines from text regions

// ------------------------------------------------------------------ ruled lines (forms, tables)

/**
 * Binary opening along one axis with a line of k pixels (OpenCV MORPH_OPEN, rect k x 1 or 1 x k).
 * For a 1-pixel-thick line this keeps exactly the runs of at least k ink pixels: one pass.
 */
function openLine(mask, w, h, k, horizontal) {
  const out = new Uint8Array(w * h);
  const [outer, inner, step, stride] = horizontal ? [h, w, 1, w] : [w, h, w, 1];
  for (let o = 0; o < outer; o++) {
    const base = o * stride;
    let start = -1;
    for (let i = 0; i <= inner; i++) {
      const on = i < inner && mask[base + i * step];
      if (on && start < 0) start = i;
      else if (!on && start >= 0) {
        if (i - start >= k) for (let j = start; j < i; j++) out[base + j * step] = 1;
        start = -1;
      }
    }
  }
  return out;
}

const boxOf = (s) => [s.x0, s.y0, s.x1 + 1, s.y1 + 1];

/** Join pieces of one rule broken where strokes cross it or the print is faint. */
function mergeCollinear(bs, along, maxGap) {
  const key = along === 0 ? (b) => [b[1], b[0]] : (b) => [b[0], b[1]];
  const sorted = bs.slice().sort((a, b) => { const ka = key(a); const kb = key(b); return ka[0] - kb[0] || ka[1] - kb[1]; });
  const out = [];
  for (const b of sorted) {
    let joined = false;
    for (const m of out) {
      const same = along === 0
        ? Math.abs((b[1] + b[3]) / 2 - (m[1] + m[3]) / 2) <= 4 && b[0] - m[2] <= maxGap && m[0] - b[2] <= maxGap
        : Math.abs((b[0] + b[2]) / 2 - (m[0] + m[2]) / 2) <= 4 && b[1] - m[3] <= maxGap && m[1] - b[3] <= maxGap;
      if (same) {
        m[0] = Math.min(m[0], b[0]); m[1] = Math.min(m[1], b[1]); m[2] = Math.max(m[2], b[2]); m[3] = Math.max(m[3], b[3]);
        joined = true;
        break;
      }
    }
    if (!joined) out.push(b.slice());
  }
  return out;
}

/** Height of the component holding the median ink pixel (htr.segment._text_height). */
function textHeight(stats, ids) {
  const byH = ids.map((l) => stats[l]).sort((a, b) => a.h - b.h);
  if (!byH.length) return 0;
  const total = byH.reduce((s, c) => s + c.area, 0);
  let cum = 0;
  for (const c of byH) { cum += c.area; if (cum >= total / 2) return c.h; }
  return byH[byH.length - 1].h;
}

/**
 * Long straight strokes in a binary ink image: printed underlines, table and box borders
 * (port of htr.page.find_rules). Returns { removed (Uint8Array), hBoxes, vBoxes }.
 */
export function findRules(ink, w, h, textH = 0, { minLen = 0.05, cross = 5 } = {}) {
  const lh = Math.max(40, Math.floor(minLen * w), Math.floor(3.0 * textH));
  let lv = Math.max(40, Math.floor(minLen * h), Math.floor(2.5 * textH));
  // Horizontal: pieces from 5% of the width, joined when collinear, kept if long, thin and mostly present.
  const piece = Math.max(40, Math.floor(minLen * w));
  const hc = components(openLine(ink, w, h, piece, true), w, h);
  const maxThick = Math.max(5, 0.1 * textH);
  const pieces = hc.stats.map((s, l) => (l && s.h <= maxThick ? boxOf(s) : null));
  const joined = mergeCollinear(pieces.filter(Boolean), 0, 0.02 * w);
  const ok = new Uint8Array(hc.n);
  for (const j of joined) {
    if (j[2] - j[0] < lh) continue;
    const inside = [];
    pieces.forEach((q, l) => { if (q && j[0] <= q[0] && q[2] <= j[2] && j[1] <= q[1] && q[3] <= j[3]) inside.push(l); });
    if (inside.reduce((s, l) => s + pieces[l][2] - pieces[l][0], 0) >= 0.6 * (j[2] - j[0])) for (const l of inside) ok[l] = 1;
  }
  let hr = new Uint8Array(w * h);
  let anyH = false;
  for (let i = 0; i < hr.length; i++) if (ok[hc.labels[i]]) { hr[i] = 1; anyH = true; }
  let vr = openLine(ink, w, h, lv, false);
  const ls = Math.max(20, Math.floor(1.5 * textH));
  if (ls < lv && anyH) { // short vertical runs count when both ends meet a horizontal rule (box/cell sides)
    const nearH = morph({ data: hr, width: w, height: h }, 9, true).data;
    const sc = components(openLine(ink, w, h, ls, false), w, h);
    const accept = new Uint8Array(sc.n);
    for (let l = 1; l < sc.n; l++) {
      const s = sc.stats[l];
      const cx = s.x0 + Math.floor(s.w / 2);
      if (nearH[s.y0 * w + cx] && nearH[s.y1 * w + cx]) accept[l] = 1;
    }
    for (let i = 0; i < vr.length; i++) if (accept[sc.labels[i]]) vr[i] = 1;
    lv = ls;
  }
  // anti-aliased edges: grow 1 px across the rule, within the ink
  const hr2 = new Uint8Array(w * h); const vr2 = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!ink[i]) continue;
      hr2[i] = hr[i] || (y > 0 && hr[i - w]) || (y < h - 1 && hr[i + w]) ? 1 : 0;
      vr2[i] = vr[i] || (x > 0 && vr[i - 1]) || (x < w - 1 && vr[i + 1]) ? 1 : 0;
    }
  }
  hr = hr2; vr = vr2;
  const removed = new Uint8Array(w * h);
  const rest = (x, y) => x >= 0 && y >= 0 && x < w && y < h && ink[y * w + x] && !hr[y * w + x] && !vr[y * w + x];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!hr[i] && !vr[i]) continue;
      let kept = false;
      if (hr[i]) {
        let up = false; let down = false;
        for (let k = 1; k <= cross && !(up && down); k++) { up ||= rest(x, y - k); down ||= rest(x, y + k); }
        kept = up && down;
      }
      if (!kept && vr[i]) {
        let left = false; let right = false;
        for (let k = 1; k <= cross && !(left && right); k++) { left ||= rest(x - k, y); right ||= rest(x + k, y); }
        kept = left && right;
      }
      if (!kept) removed[i] = 1;
    }
  }
  const hcomp = components(hr, w, h); const vcomp = components(vr, w, h);
  const hBoxes = mergeCollinear(hcomp.stats.slice(1).map(boxOf), 0, 0.03 * w);
  const vBoxes = mergeCollinear(vcomp.stats.slice(1).filter((s) => s.h >= lv).map(boxOf), 1, 0.03 * h);
  return { removed, hBoxes, vBoxes };
}

/** Ruled-line boundary between side-by-side text regions a and b ([x0, y0, x1, y1])? */
function separated(a, b, rules) {
  if (!rules) return false;
  if (a[0] > b[0]) [a, b] = [b, a];
  const gx0 = a[2]; const gx1 = b[0];
  if (gx1 <= gx0) return false;
  const hh = Math.max(a[3] - a[1], b[3] - b[1], 1);
  const tol = 0.5 * hh;
  const ylo = Math.min(a[1], b[1]); const yhi = Math.max(a[3], b[3]);
  for (const [x0, y0, x1, y1] of rules.vBoxes) {
    if (gx0 - tol <= x0 && x1 <= gx1 + tol && y0 <= ylo + tol && y1 >= yhi - tol) return true;
  }
  for (const [x0, y0, x1] of rules.hBoxes) {
    if (!(Math.min(a[3], b[3]) - tol <= y0 && y0 <= Math.max(a[3], b[3]) + 1.2 * hh)) continue;
    if (gx0 - tol <= x1 && x1 <= gx1 + 3 * tol && x1 < b[2] - tol) return true;
    if (gx0 - 3 * tol <= x0 && x0 <= gx1 + tol && x0 > a[0] + tol) return true;
  }
  return false;
}

/** Split detector regions at an underline's start/end (or a vertical rule) with 2 text heights each side. */
function cutAtRules(text, w, h, rules) {
  const cc = components(text, w, h);
  const hs = cc.stats.slice(1).filter((s) => s.area >= 20).map((s) => s.h).sort((a, b) => a - b);
  if (!hs.length) return text;
  const mh = hs.length % 2 ? hs[hs.length >> 1] : (hs[hs.length / 2 - 1] + hs[hs.length / 2]) / 2;
  const out = text.slice();
  for (let r = 1; r < cc.n; r++) {
    const s = cc.stats[r];
    if (s.area < 20) continue;
    const x = s.x0; const y = s.y0; const rw = s.w; const rh = s.h;
    const cuts = [];
    for (const [x0, y0, x1] of rules.hBoxes) if (y + rh - 0.5 * mh <= y0 && y0 <= y + rh + 1.2 * mh) cuts.push(x0, x1);
    for (const [x0, y0, x1, y1] of rules.vBoxes) if (y0 <= y + rh && y1 >= y) cuts.push(Math.floor((x0 + x1) / 2));
    for (const cx of cuts) {
      if (!(x + 2 * mh <= cx && cx <= x + rw - 2 * mh)) continue;
      for (let yy = y; yy < y + rh; yy++) {
        for (let xx = Math.max(0, cx - 2); xx < Math.min(w, cx + 3); xx++) if (cc.labels[yy * w + xx] === r) out[yy * w + xx] = 0;
      }
    }
  }
  return out;
}

function groupRegions(boxes, rules = null, { minOverlap = 0.5, nearGap = 8, farGap = 30, farOverlap = 0.6, maxRatio = 2 } = {}) {
  const n = boxes.length;
  const parent = [...Array(n).keys()];
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const hs = boxes.map((b) => b[3] - b[1]);
  const cands = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const lo = Math.min(hs[i], hs[j]); const hi = Math.max(hs[i], hs[j]);
      const ov = (Math.min(boxes[i][3], boxes[j][3]) - Math.max(boxes[i][1], boxes[j][1])) / Math.max(lo, 1);
      if (ov < minOverlap) continue;
      const gap = (Math.max(boxes[i][0], boxes[j][0]) - Math.min(boxes[i][2], boxes[j][2])) / Math.max(hi, 1);
      if (gap < nearGap || (gap < farGap && ov >= farOverlap && hi <= maxRatio * lo)) cands.push([gap, i, j]);
    }
  }
  // closest pairs first; separator test between the two groups' horizontal extents
  cands.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const gx0 = boxes.map((b) => b[0]); const gx1 = boxes.map((b) => b[2]);
  for (const [, i, j] of cands) {
    const ri = find(i); const rj = find(j);
    if (ri === rj) continue;
    if (separated([gx0[ri], boxes[i][1], gx1[ri], boxes[i][3]], [gx0[rj], boxes[j][1], gx1[rj], boxes[j][3]], rules)) continue;
    parent[ri] = rj;
    gx0[rj] = Math.min(gx0[ri], gx0[rj]); gx1[rj] = Math.max(gx1[ri], gx1[rj]);
  }
  const groups = new Map();
  for (let i = 0; i < n; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
  return [...groups.values()];
}

/** Max filter of a label image with a (2r+1)^2 square (stands in for OpenCV's ellipse). */
function dilateLabels(src, w, h, r) {
  const k = 2 * r + 1;
  const tmp = new Float64Array(w * h); const out = new Int32Array(w * h);
  for (let y = 0; y < h; y++) runFilter(null, w, k, true, (x) => src[y * w + x], (x, v) => { tmp[y * w + x] = v; });
  for (let x = 0; x < w; x++) runFilter(null, h, k, true, (y) => tmp[y * w + x], (y, v) => { out[y * w + x] = v; });
  return out;
}

/** Port of htr.segment._refine_angle (np.cov uses n-1, np.var uses n). */
function refineAngle(xs, ys, a0, maxDelta) {
  const n = xs.length;
  if (n < 50) return a0;
  const a = (a0 * Math.PI) / 180;
  const u = new Float64Array(n); const v = new Float64Array(n);
  let mu = 0; let mv = 0; let umin = Infinity; let umax = -Infinity; let vmin = Infinity; let vmax = -Infinity;
  for (let i = 0; i < n; i++) {
    u[i] = xs[i] * Math.cos(a) - ys[i] * Math.sin(a); v[i] = xs[i] * Math.sin(a) + ys[i] * Math.cos(a);
    mu += u[i]; mv += v[i];
    umin = Math.min(umin, u[i]); umax = Math.max(umax, u[i]); vmin = Math.min(vmin, v[i]); vmax = Math.max(vmax, v[i]);
  }
  if (umax - umin < 3 * (vmax - vmin)) return a0;
  mu /= n; mv /= n;
  let cov = 0; let varu = 0;
  for (let i = 0; i < n; i++) { cov += (u[i] - mu) * (v[i] - mv); varu += (u[i] - mu) ** 2; }
  const slope = cov / (n - 1) / (varu / n);
  const d = (Math.atan(slope) * 180) / Math.PI;
  return a0 - Math.min(maxDelta, Math.max(-maxDelta, d));
}

/**
 * RGBA page + detector -> lines in reading order ({ image, box, polygon, angle }, page coordinates).
 */
export async function segmentLinesDB(img, det, { rtl = false, maxSide = 2000, minAreaFrac = 2e-6, margin = 0.12,
  detSide = 1280, thresh = 0.3, boxThresh = 0.5, removeRules = true } = {}) {
  let work = img; let scale = 1;
  if (Math.max(img.width, img.height) > maxSide) {
    scale = maxSide / Math.max(img.width, img.height);
    work = resizeRGBA(img, Math.round(img.width * scale), Math.round(img.height * scale));
  }
  const { width: w, height: h } = work;
  const norm = normalizeBackground(toGray(work.data, w, h));
  const t = otsu(norm.data);
  const ink = new Uint8Array(w * h);
  for (let i = 0; i < ink.length; i++) ink[i] = norm.data[i] <= t ? 1 : 0;
  const minArea = Math.max(4, Math.floor(minAreaFrac * h * w));
  const kept = ({ stats, n }) => {
    const out = [];
    for (let l = 1; l < n; l++) {
      const s = stats[l];
      if (s.area < minArea) continue;
      const touches = s.x0 === 0 || s.y0 === 0 || s.x1 === w - 1 || s.y1 === h - 1;
      if (touches && (s.w > 0.3 * w || s.h > 0.3 * h)) continue;
      if (s.w > 25 * s.h || s.h > 25 * s.w) continue;
      out.push(l);
    }
    return out;
  };
  let rules = null;
  if (removeRules) {
    // Underlines and table borders: out of the ink, whitened in the line crops (not in the
    // detector's input), then used as separators.
    const c0 = components(ink, w, h);
    const k0 = kept(c0).filter((l) => c0.stats[l].h >= 6 && c0.stats[l].h <= 0.1 * h);
    rules = findRules(ink, w, h, k0.length ? textHeight(c0.stats, k0) : 0);
    for (let i = 0; i < ink.length; i++) if (rules.removed[i]) ink[i] = 0;
    const wipe = morph({ data: rules.removed, width: w, height: h }, 3, true).data;
    for (let i = 0; i < ink.length; i++) if (wipe[i] && !ink[i]) norm.data[i] = 255;
  }
  const { labels, stats, n } = components(ink, w, h);
  const keep = kept({ stats, n });
  if (!keep.length) return [];
  const prob = await det.probMap(work, detSide);
  let text = new Uint8Array(w * h);
  for (let i = 0; i < text.length; i++) text[i] = prob[i] > thresh ? 1 : 0;
  if (rules) text = cutAtRules(text, w, h, rules);
  const reg = components(text, w, h);
  const probSum = new Float64Array(reg.n);
  for (let i = 0; i < text.length; i++) if (reg.labels[i]) probSum[reg.labels[i]] += prob[i];
  const regions = []; const boxes = [];
  for (let r = 1; r < reg.n; r++) {
    const s = reg.stats[r];
    if (s.area < 20 || s.h < 4 || probSum[r] / s.area < boxThresh) continue;
    regions.push(r); boxes.push([s.x0, s.y0, s.x1 + 1, s.y1 + 1]);
  }
  if (!regions.length) return [];
  const groups = groupRegions(boxes, rules);
  const lut = new Int32Array(reg.n);
  groups.forEach((g, li) => { for (const i of g) lut[regions[i]] = li + 1; });
  const core = new Int32Array(w * h);
  for (let i = 0; i < core.length; i++) core[i] = lut[reg.labels[i]];
  const hsSorted = boxes.map((b) => b[3] - b[1]).sort((a, b) => a - b);
  const medH = hsSorted.length % 2 ? hsSorted[hsSorted.length >> 1]
    : (hsSorted[hsSorted.length / 2 - 1] + hsSorted[hsSorted.length / 2]) / 2;
  const grown = dilateLabels(core, w, h, Math.floor(0.4 * medH));
  // Assign each kept ink component to the line region covering most of its pixels.
  const nLines = groups.length + 1;
  const keepIdx = new Int32Array(n).fill(-1);
  keep.forEach((l, k) => { keepIdx[l] = k; });
  const counts = new Int32Array(keep.length * nLines);
  for (let i = 0; i < labels.length; i++) {
    const k = keepIdx[labels[i]];
    if (k < 0) continue;
    counts[k * nLines + (core[i] || grown[i])]++;
  }
  const members = new Map();
  const orphans = [];
  keep.forEach((l, k) => {
    let best = 0; let bc = 0;
    for (let li = 1; li < nLines; li++) if (counts[k * nLines + li] > bc) { bc = counts[k * nLines + li]; best = li; }
    if (best && bc >= 0.3 * stats[l].area) { if (!members.has(best)) members.set(best, []); members.get(best).push(l); }
    else orphans.push(l);
  });
  // Tall first capitals and loops outside the detector's kernels: attach to the line beside them.
  const lineBox = groups.map((g) => {
    const bs = g.map((i) => boxes[i]);
    const hsg = bs.map((b) => b[3] - b[1]).sort((a, b) => a - b);
    const med = hsg.length % 2 ? hsg[hsg.length >> 1] : (hsg[hsg.length / 2 - 1] + hsg[hsg.length / 2]) / 2;
    return [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])),
      Math.max(...bs.map((b) => b[3])), med];
  });
  const minOrphan = Math.max(4, 0.5 * medH * medH);
  for (const l of orphans) {
    const s = stats[l];
    if (s.area < minOrphan) continue;
    let best = 0; let bestD = Infinity;
    lineBox.forEach(([x0, y0, x1, y1, hh], idx) => {
      const band = 1.5 * hh;
      if (!(y0 - band <= s.cy && s.cy <= y1 + band)) return;
      if (s.x1 + 1 < x0 - 2 * hh || s.x0 > x1 + 2 * hh) return;
      const d = Math.abs(s.cy - (y0 + y1) / 2);
      if (d < bestD) { bestD = d; best = idx + 1; }
    });
    if (best) { if (!members.has(best)) members.set(best, []); members.get(best).push(l); }
  }
  // Angles: per long line, else the page median.
  const angles = new Map();
  for (const [li, mem] of members) {
    const set = new Set(mem);
    let x0 = Infinity; let y0 = Infinity; let x1 = -1; let y1 = -1;
    for (const l of mem) { const s = stats[l]; x0 = Math.min(x0, s.x0); y0 = Math.min(y0, s.y0); x1 = Math.max(x1, s.x1); y1 = Math.max(y1, s.y1); }
    const xs = []; const ys = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (set.has(labels[y * w + x])) { xs.push(x - x0); ys.push(y - y0); }
    let xmn = Infinity; let xmx = -Infinity; let ymn = Infinity; let ymx = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      xmn = Math.min(xmn, xs[i]); xmx = Math.max(xmx, xs[i]); ymn = Math.min(ymn, ys[i]); ymx = Math.max(ymx, ys[i]);
    }
    const long = xs.length >= 50 && xmx - xmn >= 3 * Math.max(1, ymx - ymn);
    angles.set(li, long ? refineAngle(xs, ys, 0, 15) : null);
  }
  const known = [...angles.values()].filter((a) => a !== null).sort((a, b) => a - b);
  const pageAngle = known.length ? (known.length % 2 ? known[known.length >> 1]
    : (known[known.length / 2 - 1] + known[known.length / 2]) / 2) : 0;
  const lines = [];
  for (const [li, mem] of members) {
    const a = angles.get(li) ?? pageAngle;
    const line = extract(norm, labels, mem, stats, a, margin, scale);
    if (line && line.image.width >= 8) lines.push(line);
  }
  return readingOrder(lines, rtl);
}

/** Paper detection (optional) + detector-guided lines; polygons/boxes in the input's coordinates. */
export async function segmentPage(img, det, { paper = true, ...opts } = {}) {
  let work = img; let Hm = null;
  if (paper) {
    const found = findPaper(img);
    if (found) ({ image: work, H: Hm } = found);
  }
  const lines = await segmentLinesDB(work, det, opts);
  if (Hm) {
    const Hi = invert3(Hm);
    for (const l of lines) {
      l.polygon = l.polygon.map(([x, y]) => applyH(Hi, x, y));
      const xs = l.polygon.map((p) => p[0]); const ys = l.polygon.map((p) => p[1]);
      l.box = [Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys))];
    }
  }
  return lines;
}
