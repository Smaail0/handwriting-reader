// Classical page -> lines segmentation, a dependency-free port of the "projection"
// method in src/htr/segment.py (no OpenCV.js: that would add ~8 MB to the page).
//
// Images are { data: Uint8Array, width, height }, grayscale, dark ink on light paper.

// ------------------------------------------------------------------ image helpers

/** Area-average resize (cv2.INTER_AREA), for downscaling. */
export function resizeArea(img, outW, outH) {
  const { data, width: w, height: h } = img;
  const out = new Uint8Array(outW * outH);
  const sx = w / outW;
  const sy = h / outH;
  for (let oy = 0; oy < outH; oy++) {
    const y0 = oy * sy;
    const y1 = y0 + sy;
    for (let ox = 0; ox < outW; ox++) {
      const x0 = ox * sx;
      const x1 = x0 + sx;
      let acc = 0;
      let wsum = 0;
      for (let y = Math.floor(y0); y < Math.min(h, Math.ceil(y1)); y++) {
        const wy = Math.min(y + 1, y1) - Math.max(y, y0);
        for (let x = Math.floor(x0); x < Math.min(w, Math.ceil(x1)); x++) {
          const wx = Math.min(x + 1, x1) - Math.max(x, x0);
          acc += data[y * w + x] * wx * wy;
          wsum += wx * wy;
        }
      }
      out[oy * outW + ox] = Math.round(acc / wsum);
    }
  }
  return { data: out, width: outW, height: outH };
}

/** Bilinear upscale with OpenCV's half-pixel convention. */
function resizeLinear(img, outW, outH) {
  const { data, width: w, height: h } = img;
  const out = new Uint8Array(outW * outH);
  for (let oy = 0; oy < outH; oy++) {
    let fy = (oy + 0.5) * (h / outH) - 0.5;
    fy = Math.min(Math.max(fy, 0), h - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, h - 1);
    const dy = fy - y0;
    for (let ox = 0; ox < outW; ox++) {
      let fx = (ox + 0.5) * (w / outW) - 0.5;
      fx = Math.min(Math.max(fx, 0), w - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, w - 1);
      const dx = fx - x0;
      const top = data[y0 * w + x0] * (1 - dx) + data[y0 * w + x1] * dx;
      const bot = data[y1 * w + x0] * (1 - dx) + data[y1 * w + x1] * dx;
      out[oy * outW + ox] = Math.round(top * (1 - dy) + bot * dy);
    }
  }
  return { data: out, width: outW, height: outH };
}

/** 1-D running max/min over a window (van Herk / Gil-Werman, O(1) per sample). */
function runFilter(src, n, k, isMax, get, set) {
  const r = k >> 1;
  const g = new Float64Array(n + 2 * r);
  const hb = new Float64Array(n + 2 * r);
  const ext = n + 2 * r;
  const pad = isMax ? -Infinity : Infinity;
  const at = (i) => (i < r || i >= n + r ? pad : get(i - r));
  const op = isMax ? Math.max : Math.min;
  for (let i = 0; i < ext; i++) g[i] = i % k === 0 ? at(i) : op(g[i - 1], at(i));
  for (let i = ext - 1; i >= 0; i--) hb[i] = i % k === k - 1 || i === ext - 1 ? at(i) : op(hb[i + 1], at(i));
  for (let i = 0; i < n; i++) set(i, op(hb[i], g[i + 2 * r]));
}

/** Grayscale dilation (max) or erosion (min) with a k x k square, separable. */
function morph(img, k, isMax) {
  const { width: w, height: h } = img;
  const tmp = new Float64Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    runFilter(null, w, k, isMax, (x) => img.data[y * w + x], (x, v) => { tmp[y * w + x] = v; });
  }
  for (let x = 0; x < w; x++) {
    runFilter(null, h, k, isMax, (y) => tmp[y * w + x], (y, v) => { out[y * w + x] = v; });
  }
  return { data: out, width: w, height: h };
}

function gaussianKernel(sigma, size) {
  const k = new Float64Array(size);
  const c = (size - 1) / 2;
  let s = 0;
  for (let i = 0; i < size; i++) { k[i] = Math.exp(-((i - c) ** 2) / (2 * sigma * sigma)); s += k[i]; }
  for (let i = 0; i < size; i++) k[i] /= s;
  return k;
}

const reflect101 = (i, n) => {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
};

function blur1d(arr, sigma, size) {
  const k = gaussianKernel(sigma, size);
  const n = arr.length;
  const c = (size - 1) >> 1;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < size; j++) s += arr[reflect101(i + j - c, n)] * k[j];
    out[i] = s;
  }
  return out;
}

function gaussianBlur(img, sigma) {
  const size = (Math.round(sigma * 6 + 1) | 1);
  const { width: w, height: h } = img;
  const k = gaussianKernel(sigma, size);
  const c = (size - 1) >> 1;
  const tmp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = 0; j < size; j++) s += img.data[y * w + reflect101(x + j - c, w)] * k[j];
      tmp[y * w + x] = s;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = 0; j < size; j++) s += tmp[reflect101(y + j - c, h) * w + x] * k[j];
      out[y * w + x] = Math.round(s);
    }
  }
  return { data: out, width: w, height: h };
}

/** Divide out paper colour and lighting (background estimated at 1/scale resolution). */
export function normalizeBackground(gray, scale = 4) {
  const { width: w, height: h } = gray;
  const sw = Math.max(1, Math.floor(w / scale));
  const sh = Math.max(1, Math.floor(h / scale));
  const small = resizeArea(gray, sw, sh);
  const k = Math.max(5, Math.floor(Math.min(sw, sh) / 15) | 1);
  let bg = morph(morph(small, k, true), k, false); // closing removes the dark ink
  bg = gaussianBlur(bg, k / 4);
  bg = resizeLinear(bg, w, h);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) {
    const b = bg.data[i];
    out[i] = b === 0 ? 0 : Math.min(255, Math.round((gray.data[i] * 255) / b));
  }
  return { data: out, width: w, height: h };
}

function otsu(data) {
  const hist = new Float64Array(256);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;
  const n = data.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let best = 0;
  let thresh = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; thresh = t; }
  }
  return thresh;
}

/** 8-connected components of a binary mask: labels plus per-component stats. */
function components(mask, w, h) {
  const labels = new Int32Array(w * h);
  const parent = [0];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[Math.max(a, b)] = Math.min(a, b); };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      let l = 0;
      const nb = [];
      if (x > 0 && labels[i - 1]) nb.push(labels[i - 1]);
      if (y > 0) {
        if (x > 0 && labels[i - w - 1]) nb.push(labels[i - w - 1]);
        if (labels[i - w]) nb.push(labels[i - w]);
        if (x < w - 1 && labels[i - w + 1]) nb.push(labels[i - w + 1]);
      }
      if (nb.length === 0) { l = parent.length; parent.push(l); } else {
        l = Math.min(...nb);
        for (const m of nb) union(l, m);
      }
      labels[i] = l;
    }
  }
  const remap = new Int32Array(parent.length);
  let n = 1;
  for (let l = 1; l < parent.length; l++) {
    const r = find(l);
    if (r === l) remap[l] = n++;
  }
  for (let l = 1; l < parent.length; l++) remap[l] = remap[find(l)];
  const stats = Array.from({ length: n }, () => ({ x0: Infinity, y0: Infinity, x1: -1, y1: -1, area: 0, sx: 0, sy: 0 }));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!labels[i]) continue;
      const l = remap[labels[i]];
      labels[i] = l;
      const s = stats[l];
      if (x < s.x0) s.x0 = x;
      if (x > s.x1) s.x1 = x;
      if (y < s.y0) s.y0 = y;
      if (y > s.y1) s.y1 = y;
      s.area++;
      s.sx += x;
      s.sy += y;
    }
  }
  for (const s of stats) { s.w = s.x1 - s.x0 + 1; s.h = s.y1 - s.y0 + 1; s.cx = s.sx / s.area; s.cy = s.sy / s.area; }
  return { labels, stats, n };
}

/** cv2.getRotationMatrix2D */
function rotationMatrix(cx, cy, angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  const al = Math.cos(a);
  const be = Math.sin(a);
  return [[al, be, (1 - al) * cx - be * cy], [-be, al, be * cx + (1 - al) * cy]];
}

/** Row profile of ink points after rotating the page by `angle` (forward splatting). */
function rotatedProfile(points, weights, w, h, angle) {
  const m = rotationMatrix(w / 2, h / 2, angle);
  const prof = new Float64Array(h);
  for (let k = 0; k < points.length; k += 2) {
    const y = Math.round(m[1][0] * points[k] + m[1][1] * points[k + 1] + m[1][2]);
    if (y >= 0 && y < h) prof[y] += weights ? weights[k >> 1] : 1;
  }
  return prof;
}

/** Angle (degrees) that makes the horizontal ink profile sharpest. */
export function estimateSkew(mask, w, h, maxAngle = 10, step = 0.5) {
  const scale = Math.min(1, 600 / Math.max(w, h));
  const sw = Math.max(1, Math.round(w * scale));
  const sh = Math.max(1, Math.round(h * scale));
  const m8 = new Uint8Array(w * h);
  for (let i = 0; i < m8.length; i++) m8[i] = mask[i] ? 255 : 0;
  const small = scale < 1 ? resizeArea({ data: m8, width: w, height: h }, sw, sh) : { data: m8, width: w, height: h };
  const pts = [];
  const wts = [];
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const v = small.data[y * sw + x];
      if (v) { pts.push(x, y); wts.push(v); }
    }
  }
  let best = 0;
  let bestScore = -1;
  for (let a = -maxAngle; a <= maxAngle + 1e-6; a += step) {
    const prof = rotatedProfile(pts, wts, sw, sh, a);
    let s = 0;
    for (const v of prof) s += v * v;
    if (s > bestScore) { bestScore = s; best = a; }
  }
  return Math.round(best / step) * step;
}

function linePitch(profileRaw, h) {
  const n = profileRaw.length;
  let mean = 0;
  for (const v of profileRaw) mean += v;
  mean /= n;
  const p = profileRaw.map((v) => v - mean);
  const lo = Math.floor(0.8 * h);
  const hi = Math.min(n - 1, Math.floor(8 * h));
  let ac0 = 0;
  for (let i = 0; i < n; i++) ac0 += p[i] * p[i];
  if (ac0 <= 0 || hi <= lo + 2) return 2.5 * h;
  const seg = [];
  for (let lag = lo; lag < hi; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += p[i] * p[i + lag];
    seg.push(s / ac0);
  }
  for (let i = 1; i < seg.length - 1; i++) {
    if (seg[i] >= seg[i - 1] && seg[i] >= seg[i + 1] && seg[i] > 0.1) return lo + i;
  }
  return 2.5 * h;
}

function invertAffine(m) {
  const [[a, b, c], [d, e, f]] = m;
  const det = a * e - b * d;
  const ia = e / det;
  const ib = -b / det;
  const id = -d / det;
  const ie = a / det;
  return [[ia, ib, -(ia * c + ib * f)], [id, ie, -(id * c + ie * f)]];
}

/** Straighten and crop one line: bilinear warp of the normalized page, ink of other lines masked. */
function extract(norm, labels, members, stats, angle, margin, scale) {
  const { width: w, data } = norm;
  let sx = 0;
  let sy = 0;
  let cnt = 0;
  let bx0 = Infinity; let by0 = Infinity; let bx1 = -1; let by1 = -1;
  for (const l of members) {
    const s = stats[l];
    sx += s.sx; sy += s.sy; cnt += s.area;
    bx0 = Math.min(bx0, s.x0); by0 = Math.min(by0, s.y0); bx1 = Math.max(bx1, s.x1); by1 = Math.max(by1, s.y1);
  }
  if (!cnt) return null;
  const memberSet = new Uint8Array(stats.length);
  for (const l of members) memberSet[l] = 1;
  const m = rotationMatrix(sx / cnt, sy / cnt, -angle);
  let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity;
  for (let y = by0; y <= by1; y++) {
    for (let x = bx0; x <= bx1; x++) {
      if (!memberSet[labels[y * w + x]]) continue;
      const u = m[0][0] * x + m[0][1] * y + m[0][2];
      const v = m[1][0] * x + m[1][1] * y + m[1][2];
      if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
    }
  }
  const pad = (y1 - y0) * margin;
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  m[0][2] -= x0;
  m[1][2] -= y0;
  const ow = Math.ceil(x1 - x0) + 1;
  const oh = Math.ceil(y1 - y0) + 1;
  const inv = invertAffine(m);
  const crop = new Uint8Array(ow * oh);
  const inMask = new Uint8Array(ow * oh);
  const h = norm.height;
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const fx = inv[0][0] * x + inv[0][1] * y + inv[0][2];
      const fy = inv[1][0] * x + inv[1][1] * y + inv[1][2];
      const nx = Math.round(fx);
      const ny = Math.round(fy);
      if (nx >= 0 && ny >= 0 && nx < w && ny < h && memberSet[labels[ny * w + nx]]) inMask[y * ow + x] = 1;
      const xa = Math.floor(fx);
      const ya = Math.floor(fy);
      if (xa < 0 || ya < 0 || xa + 1 >= w || ya + 1 >= h) { crop[y * ow + x] = 255; continue; }
      const dx = fx - xa;
      const dy = fy - ya;
      const i = ya * w + xa;
      const v = (data[i] * (1 - dx) + data[i + 1] * dx) * (1 - dy) + (data[i + w] * (1 - dx) + data[i + w + 1] * dx) * dy;
      crop[y * ow + x] = Math.round(v);
    }
  }
  // dilate the mask 3x3 to keep anti-aliased stroke edges, then blank everything else
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && !on; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy;
          const xx = x + dx;
          if (yy >= 0 && xx >= 0 && yy < oh && xx < ow && inMask[yy * ow + xx]) { on = 1; break; }
        }
      }
      if (!on) crop[y * ow + x] = 255;
    }
  }
  const corners = [[0, 0], [ow, 0], [ow, oh], [0, oh]];
  const polygon = corners.map(([u, v]) => [
    (inv[0][0] * u + inv[0][1] * v + inv[0][2]) / scale,
    (inv[1][0] * u + inv[1][1] * v + inv[1][2]) / scale,
  ]);
  const xs = polygon.map((p) => p[0]);
  const ys = polygon.map((p) => p[1]);
  const box = [Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys))];
  return { image: { data: crop, width: ow, height: oh }, box, polygon, angle };
}

/** Columns (horizontally overlapping lines that sit side by side), then top to bottom. */
function readingOrder(lines, rtl) {
  const n = lines.length;
  const parent = [...Array(n).keys()];
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = lines[i].box;
      const b = lines[j].box;
      const overlap = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
      if (overlap > 0.3 * Math.min(a[2] - a[0], b[2] - b[0])) parent[find(i)] = find(j);
    }
  }
  const cols = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!cols.has(r)) cols.set(r, []);
    cols.get(r).push(lines[i]);
  }
  const groups = [...cols.values()];
  let merged = true;
  while (merged && groups.length > 1) {
    merged = false;
    outer: for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        const ya = [Math.min(...groups[a].map((l) => l.box[1])), Math.max(...groups[a].map((l) => l.box[3]))];
        const yb = [Math.min(...groups[b].map((l) => l.box[1])), Math.max(...groups[b].map((l) => l.box[3]))];
        const overlap = Math.min(ya[1], yb[1]) - Math.max(ya[0], yb[0]);
        if (overlap < 0.5 * Math.min(ya[1] - ya[0], yb[1] - yb[0])) {
          groups[a].push(...groups.splice(b, 1)[0]);
          merged = true;
          break outer;
        }
      }
    }
  }
  const mid = (c) => c.reduce((s, l) => s + (l.box[0] + l.box[2]) / 2, 0) / c.length;
  groups.sort((a, b) => (rtl ? mid(b) - mid(a) : mid(a) - mid(b)));
  const cy = (l) => (l.box[1] + l.box[3]) / 2;
  return groups.flatMap((c) => c.sort((a, b) => cy(a) - cy(b)));
}

/**
 * Page -> lines in reading order. Each line: { image, box: [x0,y0,x1,y1], polygon, angle }
 * in original page coordinates.
 */
export function segmentLines(page, { rtl = false, maxSide = 2000, minAreaFrac = 2e-5, margin = 0.12,
  peakThresh = 0.15 } = {}) {
  let gray = page;
  let scale = 1;
  if (Math.max(page.width, page.height) > maxSide) {
    scale = maxSide / Math.max(page.width, page.height);
    gray = resizeArea(page, Math.round(page.width * scale), Math.round(page.height * scale));
  }
  const { width: w, height: h } = gray;
  const norm = normalizeBackground(gray);
  const t = otsu(norm.data);
  const ink = new Uint8Array(w * h);
  for (let i = 0; i < ink.length; i++) ink[i] = norm.data[i] <= t ? 1 : 0;
  const { labels, stats, n } = components(ink, w, h);
  const minArea = Math.max(4, Math.floor(minAreaFrac * h * w));
  const keep = [];
  for (let l = 1; l < n; l++) {
    const s = stats[l];
    if (s.area < minArea) continue;
    const touches = s.x0 === 0 || s.y0 === 0 || s.x1 === w - 1 || s.y1 === h - 1;
    if (touches && (s.w > 0.3 * w || s.h > 0.3 * h)) continue;
    if (s.w > 25 * s.h || s.h > 25 * s.w) continue;
    keep.push(l);
  }
  if (!keep.length) return [];
  const keepSet = new Uint8Array(n);
  for (const l of keep) keepSet[l] = 1;
  const clean = new Uint8Array(w * h);
  const pts = [];
  let inkTotal = 0;
  for (let i = 0; i < clean.length; i++) {
    if (keepSet[labels[i]]) { clean[i] = 1; pts.push(i % w, (i / w) | 0); inkTotal++; }
  }
  const angle = estimateSkew(clean, w, h);
  const rot = rotationMatrix(w / 2, h / 2, angle);
  const rawProfile = rotatedProfile(pts, null, w, h, angle);

  // text height: height of the component holding the median ink pixel
  const byH = keep.map((l) => stats[l]).sort((a, b) => a.h - b.h);
  let half = byH.reduce((s, c) => s + c.area, 0) / 2;
  let textH = byH[byH.length - 1].h;
  for (const c of byH) { half -= c.area; if (half <= 0) { textH = c.h; break; } }

  const pitch = linePitch(rawProfile, textH);
  const sigma = Math.max(2.0, pitch * 0.2);
  const profile = blur1d(rawProfile, sigma, (Math.round(sigma * 8 + 1) | 1));
  const minGap = Math.max(3, Math.floor(pitch * 0.6));
  let pmax = 0;
  for (const v of profile) pmax = Math.max(pmax, v);
  const thresh = peakThresh * pmax;
  const order = [...profile.keys()].sort((a, b) => profile[b] - profile[a]);
  const peaks = [];
  for (const y of order) {
    if (profile[y] < thresh) break;
    if (peaks.every((p) => Math.abs(y - p) >= minGap)) peaks.push(y);
  }
  peaks.sort((a, b) => a - b);
  const bounds = [-Infinity];
  for (let k = 0; k + 1 < peaks.length; k++) {
    let arg = peaks[k];
    for (let y = peaks[k]; y < peaks[k + 1]; y++) if (profile[y] < profile[arg]) arg = y;
    bounds.push(arg);
  }
  bounds.push(Infinity);

  const lines = [];
  for (let k = 0; k < peaks.length; k++) {
    const members = keep.filter((l) => {
      const s = stats[l];
      const cy = s.cx * rot[1][0] + s.cy * rot[1][1] + rot[1][2];
      return bounds[k] <= cy && cy < bounds[k + 1];
    });
    const area = members.reduce((s, l) => s + stats[l].area, 0);
    if (!members.length || area < 0.02 * inkTotal) continue;
    const line = extract(norm, labels, members, stats, -angle, margin, scale);
    if (line) lines.push(line);
  }
  return readingOrder(lines, rtl);
}

// Shared with page.js (detector-guided segmentation).
export { components, extract, gaussianBlur, morph, otsu, readingOrder, runFilter };
