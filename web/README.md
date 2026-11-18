# Handwriting reader in the browser (no backend)

The same pipeline as the Python package, ported to plain JavaScript ES modules. Images never leave the device:

- **Recognition:** ONNX Runtime Web (WebAssembly) runs the exported `.onnx` models.
- **Preprocessing:** bit-exact port of `htr/preprocess.py`, including Pillow's autocontrast and bilinear resampling.
- **Decoding:** greedy decoding, and beam search with a confidence score (`src/decode.js`).
- **Page segmentation:** dependency-free port of the projection segmenter (`src/segment.js`). No OpenCV.js, which would add about 8 MB.
  It is close, not bit-identical: a square closing kernel instead of OpenCV's ellipse, and point-projection for the skew profile. The measured agreement is in the table below.
- **Text order:** Arabic run handling and visual-to-logical order for mixed-script models (`src/text.js`).

## Run the demo

```bash
cd web
npm install && npm run vendor                      # copies the WebAssembly runtime into vendor/
cd .. && .venv/bin/python scripts/export_web.py en=runs/lite_en/model_fp32.onnx \
    ar=runs/lite_ar/model_fp32.onnx fields=runs/lite_fields/model_fp32.onnx joint=runs/joint/model_fp32.onnx
cd web && npm run serve                            # http://localhost:8765
```

Any static host works (GitHub Pages, S3, a USB stick with a local server). ES modules don't load over `file://`.

## Correction mode (collect training data)

After a page is read, click any line's text to correct it (leave it empty to skip a badly cut line). Then click
**Add page to training set**. **Download training set** saves a `.zip` (`labels.jsonl` + `lines/*.png`, written in the
page by `src/dataset.js`, no upload) that `scripts/finetune.py --data` reads directly.

## Use it in your app

```js
import * as ort from "onnxruntime-web";
import { LineRecognizer, readPage } from "./src/htr.js";

const rec = await LineRecognizer.load(ort, "models/en.onnx");   // reads models/en.json too
const { lines } = await readPage(rec, imageData);              // canvas ImageData of a page photo
lines.forEach((l) => console.log(l.text, l.box));

// Photos: paper detection + learned text detector (src/page.js, models/ppocr_det_v4.onnx, 4.6 MB)
import { TextDetector } from "./src/page.js";
const detector = await TextDetector.load(ort, "models/ppocr_det_v4.onnx");
// Language model (+ optional domain word list), as tuned in Python
import { loadLanguageModel } from "./src/htr.js";
const lm = await loadLanguageModel({ base: `models/${rec.meta.lm}`, lexicons: ["lexicons/cisco_ios.txt"] });
const page = await readPage(rec, imageData, { detector, lm });
```

The demo has both on by default ("Learned line finder", "Language model"), plus a vocabulary picker. Each model's sidecar names its language model (`lm_en.json` for Latin models, `lm_mixed.json` for the joint model; none for Arabic). The phone photo of notes: 34 lines in about 1.1 s in Chromium (0.8 s finding lines).

React: `src/react.js` exports `useHandwriting(ort, modelUrl)`. It returns `{ ready, busy, result, readFile }`; the file header has an example.

## Checks (Node): `npm test` runs them all

| Test | Result |
|---|---|
| `npm run parity`: JS vs Python on 60 test lines per model | Preprocessing identical, **same text on 100% of lines** (en, ar, fields, joint) |
| `node test/resize_parity.mjs`: JS vs Pillow bilinear resize, 300 random shapes | 300 of 300 pixel-identical (follows Pillow's pass-order rule) |
| `node test/beam_parity.mjs`: JS vs Python prefix beam search on shared logits (200 random + 40 real, the real ones also with a language model and with a 70/30 mix of two) | 320/320 same text (80/80 with a language model), confidence within 1e-15 |
| `npm run test:browser`, then open `test/react_page.html`: real browser, React 18 + hook + ONNX Runtime Web | Photo read identical to the demo; transparent PNG read (white fill works) |
| `node test/hook_loader.mjs`: loader + React hook, fake runtime and local HTTP server | 7/7: sidecar errors, bytes without meta, model switching never reads with the old model, error cleared |
| `node test/segment_parity.mjs`: JS vs Python segmenter on 20 IAM pages | Same line count on 100% of pages, median box IoU 1.000 |
| `node test/page_parity.mjs`: JS vs Python paper detection + ruled-line handling + text detector + line grouping on 15 IAM pages and 5 phone photos (incl. a filled job sheet) | Same line count on 20/20 pages, 99.1% of Python lines matched (IoU >= 0.7), median IoU 1.000, about 0.9 s/page in Node |

## Size and speed (Apple M5)

| | |
|---|---|
| Runtime (`vendor/ort-wasm-simd-threaded.wasm`) | 14.9 MB, about 3–4 MB with gzip/brotli; cached after the first visit |
| Model | 2.5 MB fp32 per model (0.7 MB int8) |
| Recognition | about 5–15 ms per line in WebAssembly (single thread), against 3 ms native fp32 |
| Segmentation | about 120–190 ms per page |
| Phone photo, whole page | about 0.2 s |

In WebAssembly the **fp32 model is faster than int8**: 13 ms against 20 ms (static) and 43 ms (dynamic) on a long line, because ORT's WebAssembly int8 kernels are slow. Use int8 only to save bandwidth.

## Notes

- **HEIC** photos decode only in Safari. Other browsers need a JPEG.
- The models were trained on IAM, RIMES and KHATT, which are licensed for non-commercial research. Don't ship them in a commercial product.
- Not ported yet: constrained decoding for typed form fields (dates, amounts). That needs a JS regex engine with partial matching.
