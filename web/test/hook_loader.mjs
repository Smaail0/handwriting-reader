// End-to-end checks of the loader (src/htr.js) and the React hook (src/react.js) with a fake
// ONNX Runtime and a local HTTP server, no browser needed.
//   node test/hook_loader.mjs        (from web/)

import assert from "node:assert/strict";
import { createServer } from "node:http";
import React from "react";
import TestRenderer, { act } from "react-test-renderer";
import { LineRecognizer } from "../src/htr.js";
import { useHandwriting } from "../src/react.js";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

// Fake ORT: a session whose logits spell "ab" with chars "ab" (blank, a, blank, b, blank).
const fakeOrt = {
  env: { wasm: {} },
  Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } },
  InferenceSession: {
    async create(model) {
      return {
        inputNames: ["x"], outputNames: ["y"], model,
        async run() {
          const T = 5, C = 3, d = new Float32Array(T * C).fill(-10);
          [0, 1, 0, 2, 0].forEach((c, t) => { d[t * C + c] = 10; });
          return { y: { data: d, dims: [1, T, C] } };
        },
      };
    },
  },
};
const meta = { charset: "ab", height: 64, downsample: 4, script: "latin", rtl: false };
const server = createServer((req, res) => {
  if (req.url === "/m.json") { res.setHeader("content-type", "application/json"); res.end(JSON.stringify(meta)); return; }
  res.statusCode = 404; res.end("missing");
});
await new Promise((r) => server.listen(0, r));
const base = `http://127.0.0.1:${server.address().port}`;
const line = { data: new Uint8Array(64 * 40).fill(255), width: 40, height: 64 };
let passed = 0;
const ok = (name) => { passed++; console.log(`ok - ${name}`); };

// Loader
const rec = await LineRecognizer.load(fakeOrt, `${base}/m.onnx`);
assert.equal((await rec.read(line)).text, "ab");
ok("loads model + sidecar over HTTP and reads a line");
await assert.rejects(LineRecognizer.load(fakeOrt, `${base}/missing.onnx`), /HTTP 404/);
ok("missing sidecar fails with a clear HTTP error");
await assert.rejects(LineRecognizer.load(fakeOrt, new Uint8Array([1, 2, 3])), /pass \{ meta \}/);
ok("bytes without meta fail with a clear message");
assert.equal((await (await LineRecognizer.load(fakeOrt, new Uint8Array([1]), { meta })).read(line)).text, "ab");
ok("bytes with meta work");

// Hook
let state;
function Probe({ url }) { state = useHandwriting(fakeOrt, url); return null; }
const settle = () => act(() => new Promise((r) => setTimeout(r, 20)));
let renderer;
await act(async () => { renderer = TestRenderer.create(React.createElement(Probe, { url: `${base}/m.onnx` })); });
await settle();
assert.equal(state.ready, true);
assert.equal(state.error, null);
ok("hook becomes ready");

await act(async () => { renderer.update(React.createElement(Probe, { url: `${base}/missing.onnx` })); });
await settle();
assert.equal(state.ready, false);
assert.match(String(state.error), /HTTP 404/);
await assert.rejects(state.readImageData({ data: new Uint8ClampedArray(4 * 64 * 64), width: 64, height: 64 }), /not loaded/);
ok("switching to a broken model: not ready, error set, and the old model is NOT used");

await act(async () => { renderer.update(React.createElement(Probe, { url: `${base}/m.onnx` })); });
await settle();
assert.equal(state.ready, true);
assert.equal(state.error, null);
ok("switching back clears the error");

server.close();
console.log(`${passed} checks passed`);
