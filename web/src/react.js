// React binding (no JSX, no build step needed for this file).
//
//   import * as ort from "onnxruntime-web";
//   import { useHandwriting } from "./htr/react.js";
//
//   function Reader() {
//     const { ready, error, result, busy, readFile } = useHandwriting(ort, "/models/en.onnx");
//     return (
//       <>
//         <input type="file" accept="image/*" disabled={!ready || busy}
//                onChange={(e) => readFile(e.target.files[0])} />
//         {result?.lines.map((l, i) => <p key={i} dir="auto">{l.text}</p>)}
//       </>
//     );
//   }

import { useCallback, useEffect, useRef, useState } from "react";
import { LineRecognizer, readPage } from "./htr.js";

const MAX_SIDE = 2000;

async function fileToImageData(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), { width: w, height: h });
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; // transparent PNGs: transparent = paper, not black ink
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Load a recognizer once and read page images with it, entirely in the browser.
 * Returns { ready, error, busy, result, readFile(file), readImageData(imageData) }.
 * result = { lines: [{ text, confidence, box, polygon, ms }], ms: { segment, total } }
 */
export function useHandwriting(ort, modelUrl, { beam = 0 } = {}) {
  const rec = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let cancelled = false;
    rec.current = null; // never read with the previous model while the new one loads
    setReady(false);
    setError(null);
    LineRecognizer.load(ort, modelUrl)
      .then((r) => { if (!cancelled) { rec.current = r; setReady(true); } })
      .catch((e) => !cancelled && setError(e));
    return () => { cancelled = true; };
  }, [ort, modelUrl]);

  const readImageData = useCallback(async (imageData) => {
    if (!rec.current) throw new Error("model not loaded yet");
    setBusy(true);
    setError(null);
    try {
      const r = await readPage(rec.current, imageData, { beam });
      setResult(r);
      return r;
    } catch (e) {
      setError(e);
      throw e;
    } finally {
      setBusy(false);
    }
  }, [beam]);

  const readFile = useCallback(async (file) => readImageData(await fileToImageData(file)), [readImageData]);

  return { ready, error, busy, result, readFile, readImageData };
}
