// JS segmenter vs Python (src/htr/segment.py, projection method) on IAM pages.
//   node test/segment_parity.mjs        (from web/)

import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { segmentLines } from "../src/segment.js";

const fx = JSON.parse(readFileSync(new URL("fixtures_segment.json", import.meta.url), "utf8"));
const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
};

let sameCount = 0;
let matched = 0;
let total = 0;
const ious = [];
const times = [];
for (const c of fx.cases) {
  const page = { data: new Uint8Array(inflateSync(Buffer.from(c.pixels, "base64"))), width: c.width, height: c.height };
  const t0 = performance.now();
  const lines = segmentLines(page);
  times.push(performance.now() - t0);
  if (lines.length === c.boxes.length) sameCount++;
  for (const pb of c.boxes) {
    total++;
    const best = Math.max(0, ...lines.map((l) => iou(l.box, pb)));
    ious.push(best);
    if (best >= 0.7) matched++;
  }
}
ious.sort((a, b) => a - b);
times.sort((a, b) => a - b);
const n = fx.cases.length;
console.log(`pages ${n}  same line count ${(sameCount / n * 100).toFixed(0)}%  `
  + `Python lines matched (IoU>=0.7) ${(matched / total * 100).toFixed(1)}%  median IoU ${ious[ious.length >> 1].toFixed(3)}  `
  + `JS ${times[n >> 1].toFixed(0)} ms/page (Python ${fx.cases.map((c) => c.ms).sort((a, b) => a - b)[n >> 1].toFixed(0)} ms, OpenCV)`);
process.exit(sameCount / n >= 0.9 ? 0 : 1);
