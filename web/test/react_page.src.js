// Real-browser check of src/react.js: React 18 + the hook + ONNX Runtime Web, bundled by esbuild.
import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import * as ort from "../vendor/ort.wasm.min.mjs";
import { useHandwriting } from "../src/react.js";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url).href;

async function transparentPng() {
  // Dark handwriting-like strokes on a fully transparent background.
  const c = new OffscreenCanvas(600, 120);
  const g = c.getContext("2d");
  g.font = "64px cursive";
  g.fillStyle = "#000";
  g.fillText("hello world", 20, 85);
  return new File([await c.convertToBlob({ type: "image/png" })], "t.png", { type: "image/png" });
}

function App() {
  const { ready, error, busy, result, readFile } = useHandwriting(ort, "/models/en.onnx");
  useEffect(() => {
    if (!ready) return;
    (async () => {
      const out = {};
      const photo = await (await fetch("/test/samples/IMG_2951.jpg")).blob();
      out.photo = (await readFile(new File([photo], "p.jpg", { type: photo.type }))).lines.map((l) => l.text);
      out.transparent = (await readFile(await transparentPng())).lines.map((l) => l.text);
      window.__result = out;
      document.title = "done";
    })().catch((e) => { window.__result = { error: String(e) }; document.title = "done"; });
  }, [ready]);
  return React.createElement("pre", null, JSON.stringify({ ready, busy, error: error && String(error), lines: result?.lines.map((l) => l.text) }, null, 1));
}

createRoot(document.getElementById("root")).render(React.createElement(App));
