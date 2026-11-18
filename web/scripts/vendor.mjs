// Copy the ONNX Runtime Web files the page needs into web/vendor/, so the page works
// from any static host (or offline) with no CDN and no build step.
import { copyFileSync, mkdirSync, statSync } from "node:fs";

const src = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url);
const dst = new URL("../vendor/", import.meta.url);
mkdirSync(dst, { recursive: true });
for (const f of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(new URL(f, src), new URL(f, dst));
  console.log(`vendor/${f}  ${(statSync(new URL(f, dst)).size / 1e6).toFixed(2)} MB`);
}
