// Client-side handwriting recognition: ONNX Runtime Web + the ports in this folder.
// Everything runs in the page (or in Node); no server is involved.
//
//   import * as ort from "onnxruntime-web";
//   import { LineRecognizer, readPage } from "./htr.js";
//   const rec = await LineRecognizer.load(ort, "models/en.onnx");
//   const { lines } = await readPage(rec, imageData);   // canvas ImageData of a page photo

import { beamSearch, greedy } from "./decode.js";
import { CharNgramLM, domainLM, MixLM } from "./lm.js";
import { preprocess, toGray } from "./preprocess.js";
import { segmentPage } from "./page.js";
import { segmentLines } from "./segment.js";
import { detokenize, fromOutput } from "./text.js";

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class LineRecognizer {
  constructor(ort, session, meta) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
    this.chars = [...meta.charset]; // code points, matches Python's str indexing
    this.inputName = session.inputNames[0];
  }

  /**
   * @param ort       the onnxruntime-web (or onnxruntime-node) module
   * @param model     URL of the .onnx file (its .json sidecar sits next to it), or bytes
   * @param meta      sidecar contents, if `model` is bytes
   */
  static async load(ort, model, { meta = null, threads = 1 } = {}) {
    if (!meta) {
      if (typeof model !== "string" && !(model instanceof URL)) {
        throw new Error("pass { meta } (the model's .json sidecar) when loading a model from bytes");
      }
      const url = String(model).replace(/\.onnx$/, ".json");
      const res = await fetch(url);
      if (!res.ok) throw new Error(`could not load ${url}: HTTP ${res.status}`);
      meta = await res.json();
    }
    if (ort.env?.wasm) ort.env.wasm.numThreads = threads;
    const session = await ort.InferenceSession.create(model, { graphOptimizationLevel: "all" });
    return new LineRecognizer(ort, session, meta);
  }

  get script() { return this.meta.script; }
  get rtl() { return this.meta.rtl; }

  /** Model input for a grayscale line image { data, width, height }. */
  prepare(gray) {
    return preprocess(gray, { height: this.meta.height, rtl: this.meta.rtl });
  }

  async logits(prep) {
    const x = new this.ort.Tensor("float32", prep.data, [1, 1, prep.height, prep.width]);
    const out = await this.session.run({ [this.inputName]: x });
    const y = out[this.session.outputNames[0]];
    const [, T, C] = y.dims;
    return { data: y.data, T, C };
  }

  /**
   * Read one line. beam > 0 uses prefix beam search (adds a confidence score); a language
   * model (`lm`, from loadLanguageModel) implies beam search with width 16 if beam is 0.
   */
  async read(gray, { beam = 0, lm = null } = {}) {
    const t0 = now();
    const { data, T, C } = await this.logits(this.prepare(gray));
    let raw;
    let confidence = null;
    let alts = [];
    if (lm && !beam) beam = 16;
    if (beam > 0) ({ text: raw, confidence, alternatives: alts } = beamSearch(data, T, C, this.chars, { beam, lm }));
    else raw = greedy(data, T, C, this.chars);
    const finish = (t) => {
      const out = fromOutput(t, this.meta.script);
      return this.meta.script === "latin" ? detokenize(out) : out;
    };
    const text = finish(raw);
    const alternatives = alts.map((a) => ({ text: finish(a.text), p: a.p }));
    return { text, confidence, alternatives, ms: now() - t0 };
  }
}

/**
 * Read a whole page, then each line. With `detector` (page.js TextDetector) and an RGBA
 * ImageData: paper detection + learned text detector (photos, diagrams, cluttered pages);
 * otherwise the classical projection segmenter.
 */
export async function readPage(rec, imageData, { beam = 0, lm = null, maxSide = 2000, detector = null } = {}) {
  const t0 = now();
  const isGray = imageData.data.length === imageData.width * imageData.height;
  const segs = detector && !isGray
    ? await segmentPage(imageData, detector, { rtl: rec.rtl, maxSide })
    : segmentLines(isGray ? imageData : toGray(imageData.data, imageData.width, imageData.height), { rtl: rec.rtl, maxSide });
  const tSeg = now() - t0;
  const lines = [];
  for (const s of segs) {
    const r = await rec.read(s.image, { beam, lm });
    lines.push({ ...r, image: s.image, box: s.box, polygon: s.polygon, angle: s.angle });
  }
  return { lines, ms: { segment: tSeg, total: now() - t0 } };
}

/**
 * Language model for beam search: a base character model (JSON from scripts/build_lm.py) and/or
 * domain phrase lists (plain text, one phrase per line, '#' comments), mixed with `weight`.
 * Phrase lists are used as written, so they must be Latin script (or already in the
 * recognizer's output order). Fusion weights default to the values tuned on validation sets.
 */
export async function loadLanguageModel({ base = null, lexicons = [], weight = 0.3, alpha = 0.5, beta = 1.5 } = {}) {
  const parts = [];
  if (base) parts.push([await CharNgramLM.fromUrl(base), 1 - (lexicons.length ? weight : 0)]);
  if (lexicons.length) {
    const phrases = [];
    for (const url of lexicons) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`lexicon ${url}: HTTP ${res.status}`);
      for (const ln of (await res.text()).split(/\r?\n/)) {
        const t = ln.normalize("NFC").replace(/\s+/g, " ").trim();
        if (t && !t.startsWith("#")) phrases.push(t);
      }
    }
    parts.push([domainLM(phrases), base ? weight : 1]);
  }
  if (!parts.length) return null;
  return { lm: parts.length === 1 ? parts[0][0] : new MixLM(parts), alpha, beta };
}
