// Correction dataset export: line crops + corrected text -> a .zip that scripts/finetune.py reads
// (labels.jsonl + lines/*.png). Minimal store-only ZIP writer, no dependencies.

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** files: [{ name, bytes: Uint8Array }] -> Blob (application/zip), no compression. */
export function makeZip(files, date = new Date()) {
  const enc = new TextEncoder();
  // MS-DOS time/date fields (a zero date reads as 1980-00-00 in some tools).
  const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const dosDate = ((Math.max(1980, date.getFullYear()) - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  const chunks = [];
  const central = [];
  let offset = 0;
  for (const f of files) {
    const name = enc.encode(f.name);
    const crc = crc32(f.bytes);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); // UTF-8 names
    local.setUint16(10, dosTime, true);
    local.setUint16(12, dosDate, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, f.bytes.length, true);
    local.setUint32(22, f.bytes.length, true);
    local.setUint16(26, name.length, true);
    chunks.push(new Uint8Array(local.buffer), name, f.bytes);
    const cen = new DataView(new ArrayBuffer(46));
    cen.setUint32(0, 0x02014b50, true);
    cen.setUint16(4, 20, true);
    cen.setUint16(6, 20, true);
    cen.setUint16(8, 0x0800, true);
    cen.setUint16(12, dosTime, true);
    cen.setUint16(14, dosDate, true);
    cen.setUint32(16, crc, true);
    cen.setUint32(20, f.bytes.length, true);
    cen.setUint32(24, f.bytes.length, true);
    cen.setUint16(28, name.length, true);
    cen.setUint32(42, offset, true);
    central.push(new Uint8Array(cen.buffer), name);
    offset += 30 + name.length + f.bytes.length;
  }
  const size = central.reduce((s, c) => s + c.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, size, true);
  end.setUint32(16, offset, true);
  return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
}

/** Grayscale line image { data, width, height } -> PNG bytes (browser). */
export async function grayToPng(img) {
  const c = new OffscreenCanvas(img.width, img.height);
  const ctx = c.getContext("2d");
  const rgba = ctx.createImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i++) {
    rgba.data[4 * i] = rgba.data[4 * i + 1] = rgba.data[4 * i + 2] = img.data[i];
    rgba.data[4 * i + 3] = 255;
  }
  ctx.putImageData(rgba, 0, 0);
  return new Uint8Array(await (await c.convertToBlob({ type: "image/png" })).arrayBuffer());
}

/** Accumulates corrected lines across pages; build() returns the .zip Blob. */
export class CorrectionSet {
  constructor() { this.items = []; this.pages = 0; }

  async addPage(lines, { model = "", script = "" } = {}) {
    this.pages += 1;
    let added = 0;
    for (const [i, l] of lines.entries()) {
      const text = (l.corrected ?? l.text ?? "").trim();
      if (!text || !l.image) continue; // empty = skip this line (e.g. a segmentation mistake)
      const name = `lines/p${String(this.pages).padStart(3, "0")}_l${String(i + 1).padStart(2, "0")}.png`;
      this.items.push({ name, png: await grayToPng(l.image), text, edited: text !== (l.text ?? "").trim(), model, script });
      added++;
    }
    return added;
  }

  get size() { return this.items.length; }

  build() {
    const enc = new TextEncoder();
    const labels = this.items.map((it) => JSON.stringify({ image: it.name, text: it.text, edited: it.edited,
      model: it.model, script: it.script })).join("\n") + "\n";
    return makeZip([{ name: "labels.jsonl", bytes: enc.encode(labels) },
      ...this.items.map((it) => ({ name: it.name, bytes: it.png }))]);
  }
}
