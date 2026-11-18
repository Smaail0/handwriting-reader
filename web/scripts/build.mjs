// Static site for hosting (Vercel or any static host): copies only what the page needs into
// dist/ (no tests, fixtures or node_modules). Run after `npm run vendor`.
import { cpSync, mkdirSync, rmSync, statSync, readdirSync } from "node:fs";

const root = new URL("../", import.meta.url);
const out = new URL("../dist/", import.meta.url);
rmSync(out, { recursive: true, force: true });
mkdirSync(out);
for (const f of ["index.html", "style.css", "demo.js", "src", "vendor", "models", "lexicons"]) {
  cpSync(new URL(f, root), new URL(f, out), { recursive: true });
}
const size = (u) => {
  const st = statSync(u);
  return st.isDirectory() ? readdirSync(u).reduce((s, n) => s + size(new URL(`${u.href.replace(/\/?$/, "/")}${n}`)), 0) : st.size;
};
console.log(`dist/ ready: ${(size(out) / 1e6).toFixed(1)} MB`);
