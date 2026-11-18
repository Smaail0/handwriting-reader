// JS bilinear resize vs Pillow on 300 random shapes (fixtures from scripts/export_web.py).
//   node test/resize_parity.mjs        (from web/)
import { readFileSync } from "node:fs";
import { resizeBilinear } from "../src/preprocess.js";

const cases = JSON.parse(readFileSync(new URL("resize_fixtures.json", import.meta.url), "utf8"));
let bad = 0;
for (const c of cases) {
  const px = Uint8Array.from(Buffer.from(c.px, "base64"));
  const want = Uint8Array.from(Buffer.from(c.out, "base64"));
  const got = resizeBilinear({ data: px, width: c.w, height: c.h }, c.ow, c.oh).data;
  if (got.length !== want.length || got.some((v, i) => v !== want[i])) bad++;
}
console.log(`resize: ${cases.length - bad} of ${cases.length} identical to Pillow`);
process.exit(bad ? 1 : 0);
