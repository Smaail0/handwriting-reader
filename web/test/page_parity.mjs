// JS page pipeline (src/page.js: paper detection + DBNet text detector + line grouping) vs
// Python (src/htr/page.py segment_page) on IAM test scans and the photos in test/samples.
//   node test/page_parity.mjs        (from web/; fixtures from scripts/export_web.py)
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import * as ort from "onnxruntime-web";
import { segmentPage, TextDetector } from "../src/page.js";

const fx = JSON.parse(readFileSync(new URL("fixtures_page.json", import.meta.url), "utf8"));
const det = await TextDetector.load(ort, readFileSync(new URL("../models/ppocr_det_v4.onnx", import.meta.url)));
const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return (ix * iy) / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy);
};
let same = 0; let matched = 0; let total = 0;
const ious = []; const times = [];
for (const c of fx.cases) {
  const img = { data: new Uint8ClampedArray(inflateSync(Buffer.from(c.pixels, "base64"))), width: c.width, height: c.height };
  const t0 = performance.now();
  const lines = await segmentPage(img, det);
  times.push(performance.now() - t0);
  const ok = lines.length === c.boxes.length;
  same += ok;
  let m = 0;
  for (const pb of c.boxes) {
    const best = Math.max(0, ...lines.map((l) => iou(l.box, pb)));
    ious.push(best); total++;
    if (best >= 0.7) { matched++; m++; }
  }
  console.log(`  ${c.name.padEnd(12)} Python ${String(c.boxes.length).padStart(2)} lines, JS ${String(lines.length).padStart(2)}, `
    + `${m} matched (IoU>=0.7)  ${times.at(-1).toFixed(0)} ms`);
}
ious.sort((a, b) => a - b); times.sort((a, b) => a - b);
const n = fx.cases.length;
console.log(`pages ${n}  same line count ${((same / n) * 100).toFixed(0)}%  Python lines matched ${((matched / total) * 100).toFixed(1)}%  `
  + `median IoU ${ious[ious.length >> 1].toFixed(3)}  JS ${times[n >> 1].toFixed(0)} ms/page`);
process.exit(matched / total >= 0.9 ? 0 : 1);
