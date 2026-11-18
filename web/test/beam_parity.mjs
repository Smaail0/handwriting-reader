// JS prefix beam search (src/decode.js) vs Python (src/htr/decode.py) on shared logits:
// 200 random cases + 40 real model outputs on IAM test lines, each real line also decoded with
// a character language model (models/lm_en.json) and with a 70/30 mix of two (lm_en + lm_fr).
// Fixtures from scripts/export_web.py.   node test/beam_parity.mjs   (from web/)
import { readFileSync } from "node:fs";
import { beamSearch } from "../src/decode.js";
import { CharNgramLM, MixLM } from "../src/lm.js";

const cases = JSON.parse(readFileSync(new URL("beam_fixtures.json", import.meta.url), "utf8"));
const lms = {};
const loadLM = (f) => {
  if (!lms[f]) {
    const d = JSON.parse(readFileSync(new URL(`../../models/${f}`, import.meta.url), "utf8"));
    lms[f] = new CharNgramLM(d.order, d.counts, d.vocab);
  }
  return lms[f];
};
let textBad = 0;
let lmCases = 0;
let lmBad = 0;
let maxConfDiff = 0;
for (const c of cases) {
  const buf = Buffer.from(c.logits, "base64");
  const logits = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let lm = null;
  if (c.lm) {
    lmCases++;
    const model = c.lm.length === 1 ? loadLM(c.lm[0]) : new MixLM([[loadLM(c.lm[0]), 0.7], [loadLM(c.lm[1]), 0.3]]);
    lm = { lm: model, alpha: c.alpha, beta: c.beta };
  }
  const r = beamSearch(logits, c.T, c.C, [...c.chars], { beam: 16, lm });
  if (r.text !== c.text) { textBad++; if (c.lm) lmBad++; }
  maxConfDiff = Math.max(maxConfDiff, Math.abs(r.confidence - c.confidence));
}
console.log(`beam: ${cases.length - textBad} of ${cases.length} same text as Python (${lmCases - lmBad} of ${lmCases} with ` +
  `a language model), max confidence diff ${maxConfDiff.toExponential(1)}`);
process.exit(textBad || maxConfDiff > 1e-6 ? 1 : 0);
