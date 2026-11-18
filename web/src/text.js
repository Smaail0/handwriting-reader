// Script-aware text handling, a port of src/htr/text.py.

const LTR_CHAR = "0-9A-Za-z٠-٩۰-۹";
const LTR_RUN = new RegExp(`%?[${LTR_CHAR}]+(?:[.,:/][${LTR_CHAR}]+)*%?`, "g");

/** Reverse runs of left-to-right characters (digits, Latin) inside Arabic text. Self-inverse. */
export function swapLtrRuns(text) {
  return text.replace(LTR_RUN, (m) => [...m].reverse().join(""));
}

// ---- mixed script (visual order), see to_visual / from_visual in text.py
const LATIN = "0-9A-Za-zÀ-ÖØ-öø-ɏ٠-٩۰-۹";
const ARABIC = "\u0621-\u063A\u0640-\u064A\u0671-\u06D3"; // letters + tatweel (no digits)
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
const NEUTRAL = esc(" .,;:!?()\"'/&#*+=_-");
const ATTACHED = esc(".,;:!?()\"'/&#*+=_-") + "\u060C\u061B\u061F"; // punctuation glued to a word
const LTR_TOKEN = `[%°€]?[${LATIN}]+(?:[.,:/'’-][${LATIN}]+)*[%°€]?`;
// A run keeps the punctuation glued to its first/last word ("شمشم،", "Bonjour,"), as written by hand.
const JOIN = NEUTRAL + "\u060C\u061B\u061F"; // what may sit between two words of one run (same for both run types)
const LTR_PHRASE = new RegExp(`[${ATTACHED}]*${LTR_TOKEN}(?:[${JOIN}]*\\s[${JOIN}]*${LTR_TOKEN})*[${ATTACHED}]*`, "g");
const RTL_PHRASE = new RegExp(`[${ATTACHED}]*[${ARABIC}]+(?:[${JOIN}]*[${ARABIC}]+)*[${ATTACHED}]*`, "g");
const ARABIC_LETTER = new RegExp(`[${ARABIC}]`, "g");
const LATIN_LETTER = /[A-Za-zÀ-ÖØ-öø-ɏ]/g;

const rev = (s) => [...s].reverse().join("");

export function baseIsRtl(text) {
  const nAr = (text.match(ARABIC_LETTER) || []).length;
  return nAr > 0 && nAr >= (text.match(LATIN_LETTER) || []).length;
}

/** Left-to-right visual order (joint model output) -> logical reading order. */
export function fromVisual(text) {
  if (baseIsRtl(text)) return rev(text).replace(LTR_PHRASE, rev);
  return text.replace(RTL_PHRASE, rev);
}

/** Model output -> logical text, for the model's script (from its metadata). */
export function fromOutput(text, script) {
  if (script === "mixed") return fromVisual(text);
  if (script === "arabic") return swapLtrRuns(text);
  return text;
}

/** Undo IAM-style spaced punctuation for display: "great , let me" -> "great, let me". */
export function detokenize(text) {
  text = text.replace(/ ([,.;:!?)\]]|n't|'s|'re|'ll|'ve|'d|'m)(?=\s|$|[,.;:!?])/g, "$1").replace(/([(\[]) /g, "$1");
  const parts = text.split(' " ');
  if (parts.length > 1) {
    let out = parts[0];
    parts.slice(1).forEach((part, i) => { out += (i % 2 === 0 ? ' "' : '" ') + part; });
    text = out;
  }
  return text;
}
