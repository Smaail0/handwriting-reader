// Line image preprocessing, a bit-exact port of src/htr/preprocess.py (Pillow semantics).
//
// Images are plain objects { data: Uint8Array|Uint8ClampedArray, width, height } holding
// one grayscale byte per pixel (dark ink on light paper).

export const DOWNSAMPLE = 4; // horizontal CNN stride: one output frame per 4 input columns

/** RGBA (canvas ImageData) -> grayscale, same fixed-point luma as Pillow's convert("L"). */
export function toGray(rgba, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >> 16;
  }
  return { data: out, width, height };
}

/** Pillow ImageOps.autocontrast(img, cutoff) for an 8-bit grayscale image. */
export function autocontrast(img, cutoff = 1) {
  const h = new Array(256).fill(0);
  for (let i = 0; i < img.data.length; i++) h[img.data[i]]++;
  const n = img.data.length;
  if (cutoff) {
    let cut = Math.floor((n * cutoff) / 100);
    for (let lo = 0; lo < 256; lo++) {
      if (cut > h[lo]) { cut -= h[lo]; h[lo] = 0; } else { h[lo] -= cut; cut = 0; }
      if (cut <= 0) break;
    }
    cut = Math.floor((n * cutoff) / 100);
    for (let hi = 255; hi >= 0; hi--) {
      if (cut > h[hi]) { cut -= h[hi]; h[hi] = 0; } else { h[hi] -= cut; cut = 0; }
      if (cut <= 0) break;
    }
  }
  let lo = 0;
  while (lo < 255 && !h[lo]) lo++;
  let hi = 255;
  while (hi > 0 && !h[hi]) hi--;
  const lut = new Uint8Array(256);
  if (hi <= lo) {
    for (let i = 0; i < 256; i++) lut[i] = i;
  } else {
    const scale = 255.0 / (hi - lo);
    const offset = -lo * scale;
    for (let i = 0; i < 256; i++) lut[i] = Math.min(255, Math.max(0, Math.trunc(i * scale + offset)));
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = lut[img.data[i]];
  return { data: out, width: img.width, height: img.height };
}

// ---- Pillow's separable resampling (Resample.c), bilinear filter, 8-bit fixed point.
const PRECISION_BITS = 32 - 8 - 2;

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 1.0 * filterscale;
  const ss = 1.0 / filterscale;
  const bounds = [];
  const weights = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k[x] = w;
      ww += w;
    }
    const kk = new Int32Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[x] = Math.trunc(v * (1 << PRECISION_BITS) + (v < 0 ? -0.5 : 0.5));
    }
    bounds.push(xmin);
    weights.push(kk);
  }
  return { bounds, weights };
}

/** Python's round(): halves go to the even neighbour. */
export function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const clip8 = (v) => {
  const x = Math.floor(v / (1 << PRECISION_BITS)); // arithmetic shift, safe beyond 2^31
  return x < 0 ? 0 : x > 255 ? 255 : x;
};

/** Pillow img.resize((w, h), Image.BILINEAR). */
export function resizeBilinear(img, outW, outH) {
  // Pillow (Resample.c) runs the vertical pass first when the height shrinks by more than
  // twice as many pixels as the width; the pass order changes the 8-bit rounding.
  const verticalFirst = img.height - outH > 0 && img.height - outH > (img.width - outW) * 2;
  if (verticalFirst && outW !== img.width) {
    return resizePasses(resizePasses(img, img.width, outH), outW, outH);
  }
  return resizePasses(img, outW, outH);
}

function resizePasses(img, outW, outH) {
  let { data, width, height } = img;
  if (outW !== width) {
    const { bounds, weights } = coeffs(width, outW);
    const out = new Uint8Array(outW * height);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let xx = 0; xx < outW; xx++) {
        const k = weights[xx];
        const xmin = bounds[xx];
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < k.length; x++) ss += data[row + xmin + x] * k[x];
        out[y * outW + xx] = clip8(ss);
      }
    }
    data = out;
    width = outW;
  }
  if (outH !== height) {
    const { bounds, weights } = coeffs(height, outH);
    const out = new Uint8Array(width * outH);
    for (let yy = 0; yy < outH; yy++) {
      const k = weights[yy];
      const ymin = bounds[yy];
      for (let x = 0; x < width; x++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < k.length; y++) ss += data[(ymin + y) * width + x] * k[y];
        out[yy * width + x] = clip8(ss);
      }
    }
    data = out;
    height = outH;
  }
  return { data, width, height };
}

/**
 * Line image -> model input: Float32Array (height x paddedWidth), ink ~1, background 0,
 * right-padded with zeros to a multiple of DOWNSAMPLE. Mirrors columns when rtl.
 */
export function preprocess(img, { height = 64, maxWidth = 2048, rtl = false } = {}) {
  const ac = autocontrast(img, 1);
  let newW = Math.max(DOWNSAMPLE * 4, pyRound((img.width * height) / img.height));
  if (maxWidth) newW = Math.min(newW, maxWidth);
  const r = resizeBilinear(ac, newW, height);
  const padded = Math.ceil(newW / DOWNSAMPLE) * DOWNSAMPLE;
  const out = new Float32Array(height * padded);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < newW; x++) {
      const src = rtl ? newW - 1 - x : x;
      out[y * padded + x] = 1.0 - r.data[y * newW + src] / 255.0;
    }
  }
  return { data: out, width: padded, height };
}
