import * as ort from "./vendor/ort.wasm.min.mjs";
import { CorrectionSet } from "./src/dataset.js";
import { buildExport, cropFiles, exportZip } from "./src/export.js";
import { LineRecognizer, loadLanguageModel, readPage } from "./src/htr.js";
import { TextDetector } from "./src/page.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };
const MAX_SIDE = 2000; // the segmenter works at this size anyway

let rec = null;
let detector = null; // learned text detector, loaded once (4.6 MB)
let lm = null; // language model for beam search (null = off)
let lmToken = 0;
let lastImage = null;
let lastLines = [];
let lastPage = null; // { width, height } of the image the line boxes refer to
const correctionSet = new CorrectionSet();

// "Auto" = the joint English/French/Arabic model: it reads any of the three, and mixed
// lines, with no choice to make (the single-language models are ~1 point better on their own).
const modelUrl = () => ($("model").value === "auto" ? "models/joint.onnx" : $("model").value);

async function loadModel(url = modelUrl()) {
  status("Getting ready… (the first visit downloads about 30 MB; later visits are instant)");
  rec = await LineRecognizer.load(ort, url);
  await loadLM();
}

// Language model = the recognizer's default (sidecar "lm") + an optional domain vocabulary.
async function loadLM() {
  const token = ++lmToken;
  const vocab = $("vocab").value;
  const base = rec?.meta.lm ? `models/${rec.meta.lm}` : null;
  const lexicons = vocab ? vocab.split(",").map((f) => `lexicons/${f}`) : [];
  let next = null;
  if ($("use-lm").checked && rec && (base || (lexicons.length && rec.meta.script !== "arabic"))) {
    next = await loadLanguageModel({ base, lexicons: rec.meta.script === "arabic" ? [] : lexicons,
      alpha: rec.meta.lm_alpha ?? 0.5, beta: rec.meta.lm_beta ?? 1.5 });
  }
  if (token !== lmToken) return; // a newer request superseded this one
  lm = next;
  status(lastImage ? "Reading…" : "Ready: take or choose a photo");
  if (lastImage) run(lastImage);
}

function toImageData(bitmap) {
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = $("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; // transparent PNGs: transparent = paper
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function drawLines(imageData, lines) {
  const ctx = $("canvas").getContext("2d");
  ctx.putImageData(imageData, 0, 0);
  ctx.lineWidth = Math.max(2, imageData.width / 600);
  ctx.font = `${Math.max(14, imageData.width / 60)}px system-ui`;
  lines.forEach((l, i) => {
    ctx.strokeStyle = l.confidence !== null && l.confidence < 0.5 ? "#ea580c" : "#dc2626";
    ctx.beginPath();
    l.polygon.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#dc2626";
    ctx.fillText(String(i + 1), l.polygon[0][0] + 4, l.polygon[0][1] + 18);
  });
}

function listLines(lines) {
  const ol = $("lines");
  ol.replaceChildren(...lines.map((l) => {
    const li = document.createElement("li");
    const text = document.createElement("div");
    text.className = "text";
    text.dir = "auto";
    text.textContent = l.text;
    text.contentEditable = "plaintext-only";
    text.spellcheck = false;
    text.title = "Click to correct; leave empty to skip this line";
    text.addEventListener("input", () => { l.corrected = text.textContent; });
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${l.ms.toFixed(1)} ms` + (l.confidence !== null ? ` · confidence ${(l.confidence * 100).toFixed(0)}%` : "");
    if (l.confidence !== null && l.confidence < 0.5) meta.classList.add("low");
    li.append(text, meta);
    return li;
  }));
}

async function run(bitmap) {
  if (!rec) return;
  lastImage = bitmap;
  $("hint").hidden = true;
  const imageData = toImageData(bitmap);
  status("Reading…");
  await new Promise((r) => setTimeout(r, 0)); // let the status paint
  const { lines, ms } = await readPage(rec, imageData, { beam: $("beam").checked ? 16 : 0, lm,
    detector: $("use-det").checked ? detector : null });
  drawLines(imageData, lines);
  listLines(lines);
  lastLines = lines;
  lastPage = { width: imageData.width, height: imageData.height };
  $("add-page").disabled = lines.length === 0;
  for (const id of ["llm-copy", "llm-share", "llm-zip"]) $(id).disabled = lines.length === 0;
  refreshExport();
  const rec_ms = lines.reduce((s, l) => s + l.ms, 0);
  status(lines.length ? `Done: ${lines.length} line${lines.length === 1 ? "" : "s"} read` : "No handwriting found: try a closer, straighter photo");
  $("llm").hidden = lines.length === 0;
  $("tip").hidden = lines.length === 0;
  $("timing").textContent = `segmentation ${ms.segment.toFixed(0)} ms · recognition ${rec_ms.toFixed(0)} ms `
    + `(${lines.length ? (rec_ms / lines.length).toFixed(1) : 0} ms/line) · total ${ms.total.toFixed(0)} ms · `
    + `${imageData.width}×${imageData.height}`;
}

async function openFile(file) {
  try {
    run(await createImageBitmap(file));
  } catch {
    status(file.type === "image/heic" || /\.heic$/i.test(file.name)
      ? "This browser can't decode HEIC. Use Safari, or convert to JPEG first."
      : "Could not read that image.");
  }
}

// ---- Send to an LLM: text + positions + confidence, crops only of uncertain lines
function currentExport() {
  return buildExport(lastLines, lastPage, { title: new Date().toISOString().slice(0, 16).replace("T", " ") });
}

function refreshExport() {
  if (!lastLines.length) return;
  const exp = currentExport();
  $("llm-text").textContent = exp.text;
  $("llm-preview").hidden = false;
  // A page photo costs about (width x height) / 750 tokens, after resizing to at most ~1.15 MP.
  const photo = Math.round(Math.min(lastPage.width * lastPage.height, 1.15e6) / 750);
  const total = exp.tokens.text + exp.tokens.images;
  $("llm-status").textContent = `≈${total} tokens (${exp.tokens.text} text + ${exp.crops.length} line image`
    + `${exp.crops.length === 1 ? "" : "s"}, ${exp.tokens.images}) vs ≈${photo} for the photo`;
}

$("lines").addEventListener("input", () => refreshExport());

$("llm-copy").addEventListener("click", async () => {
  const exp = currentExport();
  try {
    await navigator.clipboard.writeText(exp.text);
    $("llm-status").textContent = "Copied" + (exp.crops.length ? ` (the ${exp.crops.length} line images are in Share / Download)` : "");
  } catch {
    $("llm-status").textContent = "Copy blocked by the browser: use the preview below";
  }
});

$("llm-share").addEventListener("click", async () => {
  const exp = currentExport();
  const files = await cropFiles(exp.crops);
  const data = { title: "Handwritten page (OCR)", text: exp.text };
  try {
    if (files.length && navigator.canShare?.({ files })) await navigator.share({ ...data, files });
    else if (navigator.share) await navigator.share(data);
    else { await navigator.clipboard.writeText(exp.text); $("llm-status").textContent = "Sharing not supported here: copied instead"; }
  } catch (e) {
    if (e.name !== "AbortError") $("llm-status").textContent = `Share failed: ${e.message}`;
  }
});

$("llm-zip").addEventListener("click", async () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(await exportZip(currentExport()));
  a.download = "page_for_llm.zip";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$("file").addEventListener("change", (e) => e.target.files[0] && openFile(e.target.files[0]));
$("model").addEventListener("change", () => loadModel().catch((e) => status(`Failed to load the model: ${e.message}`)));
$("beam").addEventListener("change", () => lastImage && run(lastImage));
$("use-det").addEventListener("change", () => lastImage && run(lastImage));
$("use-lm").addEventListener("change", () => loadLM().catch((e) => status(`Language model: ${e.message}`)));
$("vocab").addEventListener("change", () => loadLM().catch((e) => status(`Language model: ${e.message}`)));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files[0]) openFile(e.dataTransfer.files[0]);
});

$("add-page").addEventListener("click", async () => {
  const n = await correctionSet.addPage(lastLines, { model: modelUrl(), script: rec?.meta.script });
  $("add-page").disabled = true; // once per page
  $("download").disabled = correctionSet.size === 0;
  $("set-status").textContent = `${correctionSet.size} lines collected from ${correctionSet.pages} pages (+${n})`;
});
$("download").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(correctionSet.build());
  a.download = `htr_corrections_${correctionSet.size}_lines.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

TextDetector.load(ort, "models/ppocr_det_v4.onnx")
  .then((d) => { detector = d; if (lastImage && $("use-det").checked) run(lastImage); })
  .catch((e) => { $("use-det").checked = false; $("use-det").disabled = true; console.warn("text detector:", e); });
loadModel().catch((e) => status(`Failed to load the model: ${e.message}`));
