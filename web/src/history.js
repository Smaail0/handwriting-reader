// On-device history of readings (IndexedDB): text, line positions, confidence and alternative
// readings. Never the photo or line images. Nothing leaves the browser; export/import is a JSON file.
//
//   const id = await history.save({ page, lines, model });   // returns the new id
//   await history.update(id, { lines });                     // after the user fixes lines
//   const pages = await history.list();                      // newest first, without lines
//   const p = await history.get(id); await history.remove(id);

const DB = "handwriting-reader";
const STORE = "pages";

let dbPromise = null;
function db() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("createdAt", "createdAt");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let result;
    Promise.resolve(fn(store)).then((r) => { result = r; });
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

const req2p = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

/** Only what the history keeps from a line: no image, no timing. */
function slim(l) {
  return {
    text: (l.corrected ?? l.text ?? "").trim(),
    read: l.read ?? l.text ?? "",
    edited: l.corrected != null ? l.corrected.trim() !== (l.text ?? "").trim() : !!l.edited,
    confidence: l.confidence ?? null,
    alternatives: (l.alternatives ?? []).slice(0, 3).map((a) => ({ text: a.text, p: a.p })),
    box: l.box,
  };
}

// Title: the first line the user fixed, else the first confident one, else the first readable one.
const titleOf = (lines) => {
  const ok = (l) => l.text.replace(/[^\p{L}\p{N}]/gu, "").length >= 3;
  const pick = lines.find((l) => ok(l) && l.edited)
    ?? lines.find((l) => ok(l) && (l.confidence ?? 0) >= 0.5)
    ?? lines.find(ok);
  return (pick?.text ?? "Untitled").slice(0, 80);
};

/** Ask the browser not to evict the history under storage pressure (silently ignored if refused). */
export async function persist() {
  try { return (await navigator.storage?.persist?.()) ?? false; } catch { return false; }
}

export async function save({ page, lines, model = "" }) {
  const sl = lines.map(slim);
  const rec = { createdAt: new Date().toISOString(), title: titleOf(sl), page, model, lines: sl };
  return tx("readwrite", (s) => req2p(s.add(rec)));
}

export async function update(id, { lines }) {
  return tx("readwrite", async (s) => {
    const rec = await req2p(s.get(id));
    if (!rec) return;
    rec.lines = lines.map(slim);
    rec.title = titleOf(rec.lines);
    rec.updatedAt = new Date().toISOString();
    s.put(rec);
  });
}

/** Newest first; each entry without its lines (id, createdAt, title, count). */
export async function list() {
  const all = await tx("readonly", (s) => req2p(s.getAll()));
  return all.sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .map(({ id, createdAt, title, lines }) => ({ id, createdAt, title, count: lines.length }));
}

export const get = (id) => tx("readonly", (s) => req2p(s.get(id)));
export const remove = (id) => tx("readwrite", (s) => req2p(s.delete(id)));
export const clear = () => tx("readwrite", (s) => req2p(s.clear()));

/** Everything as a JSON Blob (a backup the user downloads). */
export async function exportAll() {
  const all = await tx("readonly", (s) => req2p(s.getAll()));
  return new Blob([JSON.stringify({ format: "handwriting-reader-history", version: 1, pages: all }, null, 1)],
    { type: "application/json" });
}

/** Import a backup made by exportAll; returns the number of pages added (ids are re-assigned). */
export async function importAll(file) {
  const data = JSON.parse(await file.text());
  if (data?.format !== "handwriting-reader-history" || !Array.isArray(data.pages)) throw new Error("not a history backup");
  let n = 0;
  await tx("readwrite", (s) => {
    for (const { id, ...rec } of data.pages) {
      if (!Array.isArray(rec.lines)) continue;
      s.add(rec);
      n++;
    }
  });
  return n;
}
