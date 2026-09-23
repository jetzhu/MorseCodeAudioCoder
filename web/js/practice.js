/**
 * Practice mode: targets to key, scoring of the copy, and a rhythm report.
 *
 * Port of `morse/practice.py`: same word list, same generators (driven by a
 * seeded PRNG so tests are reproducible), same edit-distance scoring and the
 * same rhythm measure, so both apps grade the same way. No DOM, no audio.
 */

export const KINDS = ["words", "calls", "digits", "mixed"];

export const WORDS = [
  "THE", "AND", "FOR", "YOU", "ARE", "WITH", "THIS", "HAVE", "FROM", "THAT", "NOT", "BUT",
  "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
  "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT",
  "SAY", "SHE", "TOO", "USE", "GOOD", "TIME", "YEAR", "WORK", "BACK", "CALL", "COME", "EACH",
  "FIND", "GIVE", "HAND", "HELP", "HERE", "HOME", "JUST", "KEEP", "KIND", "KNOW", "LAST",
  "LIKE", "LINE", "LIVE", "LONG", "LOOK", "MAKE", "MANY", "MORE", "MOST", "MOVE", "MUCH",
  "MUST", "NAME", "NEED", "NEXT", "ONLY", "OPEN", "OVER", "PART", "PLAY", "READ", "REAL",
  "SAME", "SEND", "SHOW", "SIDE", "SOME", "TAKE", "TELL", "THAN", "THEM", "THEN", "THEY",
  "TURN", "VERY", "WANT", "WEEK", "WELL", "WHAT", "WHEN", "WILL", "WORD", "ABOUT", "AFTER",
  "AGAIN", "COULD", "EVERY", "FIRST", "GREAT", "HOUSE", "LARGE", "NEVER", "OTHER", "PLACE",
  "RADIO", "RIGHT", "SMALL", "SOUND", "STILL", "THEIR", "THERE", "THESE", "THING", "THINK",
  "THREE", "UNDER", "WATER", "WHERE", "WHICH", "WORLD", "WOULD", "WRITE", "MORSE", "SIGNAL",
  "BEEPER", "HELLO", "PARIS", "CODEX", "ANTENNA", "REPEAT", "READY", "AGAIN", "OVER", "OUT",
];

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const PREFIXES = ["K", "W", "N", "AA", "KB", "KC", "KD", "WA", "WB", "VE", "VK", "G", "M", "DL", "JA", "F", "EA", "PY"];

/**
 * Small seeded PRNG (mulberry32); `Math.random` when no seed is given.
 * @param {number | null | undefined} seed
 * @returns {() => number} uniform in [0, 1)
 */
export function makeRng(seed) {
  if (seed === null || seed === undefined) return Math.random;
  let a = (seed >>> 0) || 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (rng, items) => items[Math.floor(rng() * items.length)];
const randint = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));

/** A plausible amateur call sign: prefix, one digit, one to three letters. @param {() => number} rng */
export function callSign(rng) {
  const prefix = pick(rng, PREFIXES);
  const n = randint(rng, 1, 3);
  let suffix = "";
  for (let i = 0; i < n; i++) suffix += LETTERS[Math.floor(rng() * LETTERS.length)];
  return `${prefix}${randint(rng, 0, 9)}${suffix}`;
}

/** A group of digits, the classic numbers drill. @param {() => number} rng @param {number} [length=5] */
export function digitGroup(rng, length = 5) {
  let out = "";
  for (let i = 0; i < length; i++) out += String(randint(rng, 0, 9));
  return out;
}

export class Practice {
  /**
   * @param {"words" | "calls" | "digits" | "mixed"} [kind="words"]
   * @param {number | null} [seed=null]
   */
  constructor(kind = "words", seed = null) {
    if (!KINDS.includes(kind)) throw new RangeError(`kind must be one of ${KINDS.join(", ")}, got ${kind}`);
    this.kind = kind;
    this._rng = makeRng(seed);
    /** @type {string | null} */
    this.target = null;
    this.asked = 0;
    this.correct = 0;
  }

  /** @param {"words" | "calls" | "digits" | "mixed"} kind */
  setKind(kind) {
    if (!KINDS.includes(kind)) throw new RangeError(`kind must be one of ${KINDS.join(", ")}, got ${kind}`);
    this.kind = kind;
  }

  /** Draw the next target (never the same as the current one). @returns {string} */
  nextTarget() {
    let target = this.target ?? "";
    for (let i = 0; i < 20; i++) {
      let kind = this.kind;
      if (kind === "mixed") kind = pick(this._rng, ["words", "words", "calls", "digits"]);
      if (kind === "words") target = pick(this._rng, WORDS);
      else if (kind === "calls") target = callSign(this._rng);
      else target = digitGroup(this._rng);
      if (target !== this.target) break;
    }
    this.target = target;
    this.asked += 1;
    return target;
  }

  /** Count a checked target toward the session tally. @param {Score} result */
  record(result) {
    if (result.correct === result.total && result.errors === 0) this.correct += 1;
  }

  resetSession() {
    this.asked = 0;
    this.correct = 0;
    this.target = null;
  }
}

export class Score {
  constructor(target, sent, correct, errors, total) {
    this.target = target;
    this.sent = sent;
    /** Target characters reproduced in order (the alignment's matches). */
    this.correct = correct;
    /** Edit distance: substitutions, insertions and deletions. */
    this.errors = errors;
    /** Characters in the target. */
    this.total = total;
  }

  /** 1 for a perfect copy; errors beyond the target's length drive it to 0. @returns {number} */
  get accuracy() {
    if (this.total === 0) return this.sent ? 0 : 1;
    return Math.max(0, 1 - this.errors / this.total);
  }

  get perfect() {
    return this.errors === 0 && this.total > 0;
  }
}

const normalise = (text) => text.toUpperCase().split(/\s+/).filter(Boolean).join(" ");

/**
 * Grade `sent` against `target` (case and surplus whitespace ignored).
 * @param {string} target @param {string} sent @returns {Score}
 */
export function score(target, sent) {
  const a = normalise(target);
  const b = normalise(sent);
  const n = a.length;
  const m = b.length;
  const dist = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) dist[i][0] = i;
  for (let j = 1; j <= m; j++) dist[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dist[i][j] = Math.min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + cost);
    }
  }
  let i = n;
  let j = m;
  let matches = 0;
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1] && dist[i][j] === dist[i - 1][j - 1]) {
      matches += 1;
      i -= 1;
      j -= 1;
    } else if (dist[i][j] === dist[i - 1][j - 1] + 1) {
      i -= 1;
      j -= 1;
    } else if (dist[i][j] === dist[i - 1][j] + 1) {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  return new Score(a, b, matches, dist[n][m], n);
}

export class Rhythm {
  constructor(markErrorPct, gapErrorPct, marks, gaps) {
    this.markErrorPct = markErrorPct;
    this.gapErrorPct = gapErrorPct;
    this.marks = marks;
    this.gaps = gaps;
  }

  /** Marks and gaps together, weighted by count. @returns {number} */
  get errorPct() {
    const n = this.marks + this.gaps;
    if (n === 0) return 0;
    return (this.markErrorPct * this.marks + this.gapErrorPct * this.gaps) / n;
  }
}

/**
 * Compare each run with the nearest ideal element at `ditMs`: marks are dits
 * (1) or dahs (3) by the 2-dit split; gaps are element (1), letter (3) or
 * word (7) gaps by the 2- and 5-dit splits. Pass only the runs between the
 * first and last mark.
 * @param {Array<{on: boolean, ms: number}>} runs @param {number} ditMs @returns {Rhythm}
 */
export function rhythm(runs, ditMs) {
  if (!(ditMs > 0)) throw new RangeError(`ditMs must be positive, got ${ditMs}`);
  const markErrors = [];
  const gapErrors = [];
  for (const run of runs) {
    const units = run.ms / ditMs;
    if (run.on) {
      const ideal = units < 2 ? 1 : 3;
      markErrors.push(Math.abs(units - ideal) / ideal);
    } else {
      const ideal = units < 2 ? 1 : units < 5 ? 3 : 7;
      gapErrors.push(Math.abs(units - ideal) / ideal);
    }
  }
  const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
  return new Rhythm(100 * mean(markErrors), 100 * mean(gapErrors), markErrors.length, gapErrors.length);
}
