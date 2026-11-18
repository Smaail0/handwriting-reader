// Checks the JS port against the Python pipeline on the fixtures written by
// scripts/export_web.py: same preprocessed input, same greedy reading.
//   node test/parity.mjs            (from web/)

import { readFileSync, readdirSync } from "node:fs";
import { inflateSync } from "node:zlib";
import * as ort from "onnxruntime-web";
import { greedy } from "../src/decode.js";
import { LineRecognizer } from "../src/htr.js";
import { fromOutput } from "../src/text.js";

const here = new URL(".", import.meta.url);
const lev = (a, b) => {
  a = [...a]; b = [...b];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length];
};
const cer = (preds, refs) => preds.reduce((s, p, i) => s + lev(p, refs[i]), 0) / refs.reduce((s, r) => s + [...r].length, 0);

let failed = false;
for (const file of readdirSync(new URL("fixtures/", here)).filter((f) => f.endsWith(".json"))) {
  const fx = JSON.parse(readFileSync(new URL(`fixtures/${file}`, here), "utf8"));
  const modelUrl = new URL(`../models/${fx.model}`, here);
  const meta = JSON.parse(readFileSync(new URL(`../models/${fx.model.replace(/\.onnx$/, ".json")}`, here), "utf8"));
  const rec = await LineRecognizer.load(ort, readFileSync(modelUrl), { meta });
  let prepExact = 0;
  let textExact = 0;
  let maxSumDiff = 0;
  const js = [];
  const py = [];
  const refs = [];
  let ms = 0;
  for (const c of fx.cases) {
    const gray = { data: new Uint8Array(inflateSync(Buffer.from(c.pixels, "base64"))), width: c.width, height: c.height };
    const t0 = performance.now();
    const prep = rec.prepare(gray);
    const { data, T, C } = await rec.logits(prep);
    const text = fromOutput(greedy(data, T, C, rec.chars), rec.script);
    ms += performance.now() - t0;
    let sum = 0;
    for (const v of prep.data) sum += v;
    const diff = Math.abs(sum - c.prep_sum);
    maxSumDiff = Math.max(maxSumDiff, diff);
    if (prep.width === Math.ceil(c.prep_width / 4) * 4 && diff < 1e-2) prepExact++;
    if (text === c.greedy) textExact++;
    js.push(text);
    py.push(c.greedy);
    refs.push(c.reference);
  }
  const n = fx.cases.length;
  const textRate = textExact / n;
  console.log(`${file.padEnd(12)} lines ${n}  preprocess identical ${(prepExact / n * 100).toFixed(1)}% `
    + `(max |sum diff| ${maxSumDiff.toExponential(1)})  same text as Python ${(textRate * 100).toFixed(1)}%  `
    + `CER vs truth: JS ${(cer(js, refs) * 100).toFixed(2)}%  Python ${(cer(py, refs) * 100).toFixed(2)}%  `
    + `${(ms / n).toFixed(1)} ms/line`);
  if (textRate < 0.95) failed = true;
}
process.exit(failed ? 1 : 0);
