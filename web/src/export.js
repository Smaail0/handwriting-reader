// "Export for LLM": the page as compact text an LLM can read reliably, instead of the photo.
//
// A blind test (a model given only OCR text, never the photos) recovered running text well but
// failed on forms in three ways, each addressed here:
//   1. lost layout (which value belongs to which label)  -> each line carries its position;
//   2. it "corrected" digits from context                 -> instructions: copy digits as read;
//   3. garbled numbers/codes cannot be recovered from text -> crops of uncertain lines and of
//      lines with digits are attached (a line crop costs ~50-150 image tokens vs ~1,500 for
//      a whole page photo).
// Nothing is uploaded by this module; it builds text, PNG crops and a .zip for the user to share.

import { grayToPng, makeZip } from "./dataset.js";

const HEADER = `Handwritten page read on-device by an OCR model (the photo was not kept).
Each line: [n] (x,y w*h) conf | text   (alt: other readings)
- Position and size are % of the page, origin top-left. Use them to pair labels with values
  (same row) and to rebuild tables (same column).
- conf is the model's confidence in that reading, from the image alone. Low-confidence lines may
  list alternative readings.
- Copy numbers, amounts, dates, codes and IDs exactly as read. Do not "correct" digits from
  context. If an attached line image shows them clearly, use the image. Mark any guessed
  character with [?].
- Lines marked [image] have a crop of that line attached as an image.
---`;

/**
 * @param lines  readPage().lines: { text, confidence, alternatives, image, box }
 * @param page   { width, height } of the page the boxes refer to
 * @param opts   cropBelow: attach a crop when confidence is under this; cropDigits: also when the
 *               line has digits and confidence is under this; maxCrops: cap on attached crops
 * @returns { text, crops: [{ name, line, image }], json }
 */
export function buildExport(lines, page, { cropBelow = 0.6, cropDigits = 0.95, maxCrops = 8, title = "" } = {}) {
  const pct = (v, total) => Math.max(0, Math.min(100, Math.round((100 * v) / total)));
  const rows = [];
  const crops = [];
  const json = [];
  // Which lines get a crop: uncertain ones, numbers first (they cannot be repaired from
  // context), then the least confident, up to maxCrops.
  const cand = lines.map((l, i) => {
    const text = (l.corrected ?? l.text ?? "").trim();
    const edited = l.corrected != null && l.corrected.trim() !== (l.text ?? "").trim();
    const conf = l.confidence ?? null;
    const digits = /[0-9]/.test(text);
    const unsure = conf !== null && !edited && !!l.image && (conf < cropBelow || (digits && conf < cropDigits));
    return { i, unsure, digits, conf };
  }).filter((c) => c.unsure);
  cand.sort((a, b) => (b.digits - a.digits) || (a.conf - b.conf));
  const cropSet = new Set(cand.slice(0, maxCrops).map((c) => c.i));
  lines.forEach((l, i) => {
    const n = i + 1;
    const [x0, y0, x1, y1] = l.box;
    const pos = `(${pct(x0, page.width)},${pct(y0, page.height)} ${pct(x1 - x0, page.width)}*${pct(y1 - y0, page.height)})`;
    const conf = l.confidence ?? null;
    const text = (l.corrected ?? l.text ?? "").trim();
    const edited = l.corrected != null && l.corrected.trim() !== (l.text ?? "").trim();
    const withImage = cropSet.has(i);
    if (withImage) crops.push({ name: `line_${String(n).padStart(2, "0")}.png`, line: n, image: l.image });
    const alts = conf !== null && conf < 0.8 && !edited
      ? [...new Set((l.alternatives ?? []).map((a) => a.text.trim()))].filter((t) => t && t !== text).slice(0, 2)
      : [];
    const confStr = edited ? "corrected" : conf === null ? "?" : `${Math.round(conf * 100)}%`;
    rows.push(`[${n}] ${pos} ${confStr} | ${text}${alts.length ? `   (alt: ${alts.join(" / ")})` : ""}${withImage ? "   [image]" : ""}`);
    json.push({ n, box: l.box, confidence: conf, text, edited, alternatives: alts, image: withImage ? crops.at(-1).name : null });
  });
  const head = title ? `${HEADER}\n# ${title}` : HEADER;
  const text = `${head}\n${rows.join("\n")}\n`;
  // Rough token cost: ~3.5 characters per text token; Claude counts an image as about
  // width x height / 750 tokens.
  const tokens = {
    text: Math.round(text.length / 3.5),
    images: crops.reduce((s, c) => s + Math.max(1, Math.round((c.image.width * c.image.height) / 750)), 0),
  };
  return { text, crops, json: { page, lines: json }, tokens };
}

/** Crops as PNG Files (for navigator.share / attaching to a chat). */
export async function cropFiles(crops) {
  return Promise.all(crops.map(async (c) => new File([await grayToPng(c.image)], c.name, { type: "image/png" })));
}

/** One .zip: page.md (the prompt), page.json (structured), crops/*.png. */
export async function exportZip(exp) {
  const enc = new TextEncoder();
  const files = [
    { name: "page.md", bytes: enc.encode(exp.text) },
    { name: "page.json", bytes: enc.encode(JSON.stringify(exp.json, null, 1)) },
  ];
  for (const c of exp.crops) files.push({ name: `crops/${c.name}`, bytes: await grayToPng(c.image) });
  return makeZip(files);
}
