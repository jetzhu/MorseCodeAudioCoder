// The decoded-text log: every character the decoder emitted, stamped with
// when it arrived. Mirror of morse/declog.py; exports produce identical text
// for identical input (clock times in ms here, seconds there).
//
// Two clocks appear in every line: `elapsedMs` is audio time since the stream
// started (blocks processed times the block length), `wallMs` is Date.now()
// when the text was decoded. Words are split on whitespace; a word's stamps
// are those of its first character and the audio time of its last.

/** @typedef {{text: string, elapsedMs: number, wallMs: number}} LogEntry */
/** @typedef {{text: string, elapsedMs: number, wallMs: number, endMs: number}} LogWord */

export class DecodedLog {
  constructor() {
    /** @type {LogEntry[]} */
    this._entries = [];
  }

  /** Record `text` decoded at audio time `elapsedMs` and clock time `wallMs`; "" is ignored. */
  add(text, elapsedMs, wallMs) {
    if (text) this._entries.push({ text: String(text), elapsedMs: Number(elapsedMs), wallMs: Number(wallMs) });
  }

  reset() {
    this._entries.length = 0;
  }

  /** @returns {LogEntry[]} a copy */
  get entries() {
    return this._entries.slice();
  }

  /** Everything logged, concatenated (the decoder's text before any Clear). */
  get text() {
    return this._entries.map((e) => e.text).join("");
  }

  get isEmpty() {
    return !this._entries.some((e) => e.text.trim());
  }

  get length() {
    return this._entries.length;
  }

  /** The log split on whitespace, each word stamped by its first character. @returns {LogWord[]} */
  words() {
    /** @type {LogWord[]} */
    const words = [];
    let chars = [];
    /** @type {LogEntry | null} */
    let start = null;
    let lastMs = 0;
    for (const entry of this._entries) {
      for (const ch of entry.text) {
        if (/\s/.test(ch)) {
          if (chars.length && start) words.push({ text: chars.join(""), elapsedMs: start.elapsedMs, wallMs: start.wallMs, endMs: lastMs });
          chars = [];
          start = null;
          continue;
        }
        if (!start) start = entry;
        chars.push(ch);
        lastMs = entry.elapsedMs;
      }
    }
    if (chars.length && start) words.push({ text: chars.join(""), elapsedMs: start.elapsedMs, wallMs: start.wallMs, endMs: lastMs });
    return words;
  }

  // ----------------------------------------------------------------- export

  /**
   * The log as a readable table, one word per line.
   * @param {Array<[string, string]>} [header] `(label, value)` lines after the title
   * @param {object} [options]
   * @param {number | null} [options.nowMs=null] export time; null leaves the Exported line out
   * @param {boolean} [options.utc=false] format clocks in UTC instead of local time
   */
  renderText(header = [], { nowMs = null, utc = false } = {}) {
    const words = this.words();
    const lines = ["Beeper Morse Console decoded log"];
    if (nowMs !== null) lines.push(`Exported: ${formatIso(nowMs, utc).replace("T", " ").slice(0, 19)}`);
    for (const [label, value] of header) lines.push(`${label}: ${value}`);
    lines.push(`Words: ${words.length}`);
    lines.push("Time is the computer clock when the word's first letter was decoded; elapsed is audio time since the stream started.");
    lines.push("");
    lines.push(`${"time".padEnd(12)}  ${"elapsed".padStart(8)}  word`);
    for (const w of words) lines.push(`${formatClock(w.wallMs, utc).padEnd(12)}  ${formatElapsed(w.elapsedMs).padStart(8)}  ${w.text}`);
    return lines.join("\n") + "\n";
  }

  /** The log as CSV: `time,elapsed_s,word` with a header line. @param {{utc?: boolean}} [options] */
  renderCsv({ utc = false } = {}) {
    const lines = ["time,elapsed_s,word"];
    for (const w of this.words()) lines.push(`${formatIso(w.wallMs, utc)},${(w.elapsedMs / 1000).toFixed(2)},${csvField(w.text)}`);
    return lines.join("\n") + "\n";
  }
}

// ------------------------------------------------------------------ formatting

function csvField(text) {
  return /[,"\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

const p2 = (n) => String(n).padStart(2, "0");
const p3 = (n) => String(n).padStart(3, "0");

function parts(wallMs, utc) {
  const d = new Date(wallMs);
  return utc
    ? { y: d.getUTCFullYear(), mo: d.getUTCMonth() + 1, d: d.getUTCDate(), h: d.getUTCHours(), mi: d.getUTCMinutes(), s: d.getUTCSeconds(), ms: d.getUTCMilliseconds() }
    : { y: d.getFullYear(), mo: d.getMonth() + 1, d: d.getDate(), h: d.getHours(), mi: d.getMinutes(), s: d.getSeconds(), ms: d.getMilliseconds() };
}

/** `HH:MM:SS.mmm` of a clock time (local unless `utc`). */
export function formatClock(wallMs, utc = false) {
  const t = parts(wallMs, utc);
  return `${p2(t.h)}:${p2(t.mi)}:${p2(t.s)}.${p3(t.ms)}`;
}

/** `YYYY-MM-DDTHH:MM:SS.mmm` of a clock time, without a zone designator. */
export function formatIso(wallMs, utc = false) {
  const t = parts(wallMs, utc);
  return `${t.y}-${p2(t.mo)}-${p2(t.d)}T${p2(t.h)}:${p2(t.mi)}:${p2(t.s)}.${p3(t.ms)}`;
}

/** Audio time as `+M:SS.d` (minutes unbounded, tenths of a second). */
export function formatElapsed(ms) {
  const tenths = Math.round(Math.max(0, Number(ms) || 0) / 100);
  const minutes = Math.floor(tenths / 600);
  const rest = tenths - minutes * 600;
  const seconds = Math.floor(rest / 10);
  return `+${minutes}:${p2(seconds)}.${rest - seconds * 10}`;
}

/** `morse_log_YYYYmmdd_HHMMSS.<ext>` for an export at `wallMs`. */
export function defaultFilename(wallMs, ext = "txt", utc = false) {
  const t = parts(wallMs, utc);
  return `morse_log_${t.y}${p2(t.mo)}${p2(t.d)}_${p2(t.h)}${p2(t.mi)}${p2(t.s)}.${ext.replace(/^\./, "")}`;
}
