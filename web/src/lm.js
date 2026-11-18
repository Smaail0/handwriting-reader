// Character n-gram language model for beam search, a port of src/htr/lm.py.
// Loads the JSON written by scripts/build_lm.py (same interpolated Witten-Bell maths), so the
// browser and Python give the same scores. Domain phrase lists (lexicons/*.txt) can be mixed in.
//
//   const lm = await CharNgramLM.fromUrl("models/lm_mixed.json");
//   const scorer = { lm: new MixLM([[lm, 0.7], [domainLM(phrases), 0.3]]), alpha: 0.5, beta: 1.5 };
//   beamSearch(logits, T, C, chars, { beam: 16, lm: scorer });

export const BOS = "\x02";
export const EOS = "\x03";

export class CharNgramLM {
  constructor(order, counts, vocab) {
    this.order = order;
    // counts[k]: Map(context -> { n: Map(char -> count), total, distinct })
    this.levels = counts.map((level) => {
      const m = new Map();
      for (const [ctx, d] of Object.entries(level)) {
        const n = new Map(Object.entries(d));
        let total = 0;
        for (const v of n.values()) total += v;
        m.set(ctx, { n, total, distinct: n.size });
      }
      return m;
    });
    this.vocab = [...vocab];
    this.base = 1 / (this.vocab.length + 1);
    this.cache = new Map();
  }

  static async fromUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`language model ${url}: HTTP ${res.status}`);
    const d = await res.json();
    return new CharNgramLM(d.order, d.counts, d.vocab);
  }

  /** Train from texts already in output order (small domain lists; base models come from Python). */
  static train(texts, order = 6) {
    const counts = Array.from({ length: order }, () => ({}));
    const vocab = new Set();
    for (const t of texts) {
      if (!t) continue;
      const s = [...(BOS.repeat(order - 1) + t + EOS)];
      for (const ch of t) vocab.add(ch);
      for (let i = order - 1; i < s.length; i++) {
        for (let k = 0; k < order; k++) {
          const ctx = s.slice(i - k, i).join("");
          const d = (counts[k][ctx] ??= {});
          d[s[i]] = (d[s[i]] ?? 0) + 1;
        }
      }
    }
    return new CharNgramLM(order, counts, [...vocab].sort().join(""));
  }

  /** log P(c | history); c may be EOS. history: string (code points). */
  logp(history, c) {
    const n = this.order - 1;
    const hist = [...history];
    const h = [...BOS.repeat(n), ...hist].slice(hist.length).join(""); // last n code points, BOS-padded
    const key = h + "\x00" + c;
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit;
    const hc = [...h];
    let p = this.base;
    for (let k = 0; k < this.order; k++) {
      const ctx = k ? hc.slice(hc.length - k).join("") : "";
      const d = this.levels[k].get(ctx);
      if (!d) break;
      p = ((d.n.get(c) ?? 0) + d.distinct * p) / (d.total + d.distinct);
    }
    const lp = Math.log(p);
    if (this.cache.size > 1 << 18) this.cache.clear();
    this.cache.set(key, lp);
    return lp;
  }
}

/** Linear interpolation: [[lm, weight], ...]. */
export class MixLM {
  constructor(parts) {
    const total = parts.reduce((s, [, w]) => s + w, 0);
    this.parts = parts.map(([m, w]) => [m, w / total]);
    this.order = Math.max(...parts.map(([m]) => m.order));
  }

  logp(history, c) {
    let p = 0;
    for (const [m, w] of this.parts) p += w * Math.exp(m.logp(history, c));
    return Math.log(p);
  }
}

/** Domain model from phrases (in output order): each phrase alone and in short random runs. */
export function domainLM(phrases, order = 6, repeat = 3, seed = 1) {
  let s = seed >>> 0;
  const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  const texts = [];
  for (let r = 0; r < repeat; r++) texts.push(...phrases);
  for (let i = 0; i < phrases.length * repeat; i++) {
    const k = Math.min(phrases.length, 2 + Math.floor(rand() * 4));
    const pick = [];
    for (let j = 0; j < k; j++) pick.push(phrases[Math.floor(rand() * phrases.length)]);
    texts.push(pick.join(" "));
  }
  return CharNgramLM.train(texts, order);
}
