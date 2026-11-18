// CTC decoding, a port of src/htr/decode.py (index 0 = blank).

/** Row-wise log-softmax of logits (Float32Array, T x C) -> Float64Array. */
export function logSoftmax(logits, T, C) {
  const out = new Float64Array(T * C);
  for (let t = 0; t < T; t++) {
    let m = -Infinity;
    for (let c = 0; c < C; c++) m = Math.max(m, logits[t * C + c]);
    let s = 0;
    for (let c = 0; c < C; c++) s += Math.exp(logits[t * C + c] - m);
    const lse = m + Math.log(s);
    for (let c = 0; c < C; c++) out[t * C + c] = logits[t * C + c] - lse;
  }
  return out;
}

/** Greedy decoding: best class per frame, collapse repeats, drop blanks. */
export function greedy(logits, T, C, chars) {
  let out = "";
  let prev = 0;
  for (let t = 0; t < T; t++) {
    let best = 0;
    let bv = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = logits[t * C + c];
      if (v > bv) { bv = v; best = c; }
    }
    if (best !== prev && best !== 0) out += chars[best - 1];
    prev = best;
  }
  return out;
}

const NEG = -1e30;
const lse2 = (a, b) => {
  if (a < b) [a, b] = [b, a];
  if (b <= NEG) return a;
  return a + Math.log1p(Math.exp(b - a));
};

/**
 * CTC prefix beam search. Returns { text, confidence } where confidence is the best
 * prefix's share of the probability mass found by the search (image only).
 * Optional lm = { lm, alpha, beta } (see lm.js): shallow fusion, as in src/htr/decode.py.
 * The language model helps choose the text; confidence stays image-only.
 */
export function beamSearch(logits, T, C, chars, { beam = 16, prune = 1e-3, lm = null } = {}) {
  const lp = logSoftmax(logits, T, C);
  const logPrune = Math.log(prune);
  // prefix key: string of class ids joined by ","; value [pBlank, pNonBlank, ids]
  let beams = new Map([["", [0.0, NEG, []]]]);
  const lmsc = new Map([["", 0.0]]);
  const n = lm ? lm.lm.order - 1 : 0;
  const hist = (ids) => ids.slice(Math.max(0, ids.length - n)).map((i) => chars[i - 1]).join("");
  const step = (ids, c) => lm.alpha * lm.lm.logp(n ? hist(ids) : "", chars[c - 1]) + lm.beta;
  for (let t = 0; t < T; t++) {
    const row = t * C;
    const cand = [];
    for (let c = 1; c < C; c++) if (lp[row + c] > logPrune) cand.push(c);
    const next = new Map();
    const get = (key, ids) => {
      let e = next.get(key);
      if (!e) { e = [NEG, NEG, ids]; next.set(key, e); }
      return e;
    };
    for (const [key, [pb, pnb, ids]] of beams) {
      const total = lse2(pb, pnb);
      const entry = get(key, ids);
      entry[0] = lse2(entry[0], total + lp[row]);
      const last = ids.length ? ids[ids.length - 1] : -1;
      for (const c of cand) {
        const p = lp[row + c];
        const nids = ids.concat(c);
        const nkey = key ? key + "," + c : String(c);
        if (c === last) {
          entry[1] = lse2(entry[1], pnb + p);
          const e = get(nkey, nids);
          e[1] = lse2(e[1], pb + p);
        } else {
          const e = get(nkey, nids);
          e[1] = lse2(e[1], total + p);
        }
        if (lm && !lmsc.has(nkey)) lmsc.set(nkey, lmsc.get(key) + step(ids, c));
      }
    }
    const rank = ([k, v]) => lse2(v[0], v[1]) + (lmsc.get(k) ?? 0);
    beams = new Map([...next.entries()].sort((a, b) => rank(b) - rank(a)).slice(0, beam));
  }
  let best = null;
  let bestRank = -Infinity;
  let bestAc = NEG;
  let mass = NEG;
  for (const [key, [pb, pnb, ids]] of beams) {
    const ac = lse2(pb, pnb);
    mass = lse2(mass, ac);
    const r = ac + (lm ? lmsc.get(key) + lm.alpha * lm.lm.logp(n ? hist(ids) : "", "\x03") : 0);
    if (r > bestRank) { bestRank = r; bestAc = ac; best = ids; }
  }
  const text = best.map((i) => chars[i - 1]).join("");
  // Top readings (ranked as searched), each with its image-only share of the mass found.
  const alternatives = [...beams.entries()]
    .map(([key, [pb, pnb, ids]]) => {
      const ac = lse2(pb, pnb);
      const r = ac + (lm ? lmsc.get(key) + lm.alpha * lm.lm.logp(n ? hist(ids) : "", "\x03") : 0);
      return { text: ids.map((i) => chars[i - 1]).join(""), p: Math.exp(ac - mass), r };
    })
    .sort((a, b) => b.r - a.r)
    .slice(0, 3)
    .map(({ text: t, p }) => ({ text: t, p }));
  return { text, confidence: Math.exp(bestAc - mass), alternatives };
}
