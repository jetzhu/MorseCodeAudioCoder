/**
 * International Morse code table and text encoder.
 *
 * Port of `morse/table.py`. `MORSE_TABLE` maps a character to its dot/dash
 * string and `INVERSE` maps the dot/dash string back to the character. Every
 * code is unique, so the two are exact inverses of each other.
 *
 * Both tables are frozen objects without a prototype, so `INVERSE["constructor"]`
 * is `undefined` rather than a function; use {@link lookup} rather than indexing
 * when the key comes from outside.
 *
 * ES module; runs in the browser and in Node without a DOM.
 */

/**
 * Character -> dot/dash code for A-Z, 0-9 and common punctuation.
 * @type {Readonly<Record<string, string>>}
 */
export const MORSE_TABLE = Object.freeze(Object.assign(Object.create(null), {
  // Letters
  A: ".-",
  B: "-...",
  C: "-.-.",
  D: "-..",
  E: ".",
  F: "..-.",
  G: "--.",
  H: "....",
  I: "..",
  J: ".---",
  K: "-.-",
  L: ".-..",
  M: "--",
  N: "-.",
  O: "---",
  P: ".--.",
  Q: "--.-",
  R: ".-.",
  S: "...",
  T: "-",
  U: "..-",
  V: "...-",
  W: ".--",
  X: "-..-",
  Y: "-.--",
  Z: "--..",
  // Digits
  "0": "-----",
  "1": ".----",
  "2": "..---",
  "3": "...--",
  "4": "....-",
  "5": ".....",
  "6": "-....",
  "7": "--...",
  "8": "---..",
  "9": "----.",
  // Punctuation
  ".": ".-.-.-",
  ",": "--..--",
  "?": "..--..",
  "/": "-..-.",
  "=": "-...-",
  "'": ".----.",
  "(": "-.--.",
  ")": "-.--.-",
  ":": "---...",
  ";": "-.-.-.",
  "+": ".-.-.",
  "-": "-....-",
  _: "..--.-",
  '"': ".-..-.",
  "@": ".--.-.",
  "!": "-.-.--",
}));

/**
 * Dot/dash code -> character; the exact inverse of {@link MORSE_TABLE}.
 * @type {Readonly<Record<string, string>>}
 */
export const INVERSE = Object.freeze(
  Object.entries(MORSE_TABLE).reduce((inverse, [char, code]) => {
    inverse[code] = char;
    return inverse;
  }, Object.create(null)),
);

if (Object.keys(INVERSE).length !== Object.keys(MORSE_TABLE).length) {
  throw new Error("MORSE_TABLE contains a duplicate code");
}

/** Separator between letters in {@link encode} output. */
export const LETTER_SEP = " ";

/** Separator between words in {@link encode} output. */
export const WORD_SEP = " / ";

/**
 * Return the character for a dot/dash string, or `null` when unknown.
 *
 * The lookup is exact: `"..."` gives `"S"`, while `""`, `".. ."` or any
 * sequence not in the table gives `null` (the port of Python's `None`).
 *
 * @param {string} symbols dot/dash string such as `".-"`
 * @returns {string | null}
 */
export function lookup(symbols) {
  if (typeof symbols !== "string") return null;
  return Object.hasOwn(INVERSE, symbols) ? INVERSE[symbols] : null;
}

/**
 * Encode text as Morse, letters separated by one space, words by `" / "`.
 *
 * `"SOS X"` becomes `"... --- ... / -..-"`. Input is upper-cased and any run
 * of whitespace is one word boundary. Characters not in the table are
 * replaced by `unknown`; with the default empty string they are dropped, and
 * a word that becomes empty as a result is dropped too.
 *
 * @param {string} text plain text
 * @param {string} [unknown=""] placeholder for characters without a code
 * @returns {string}
 */
export function encode(text, unknown = "") {
  const words = [];
  for (const word of String(text).toUpperCase().split(/\s+/)) {
    if (!word) continue;
    const codes = [];
    for (const ch of word) {
      const code = Object.hasOwn(MORSE_TABLE, ch) ? MORSE_TABLE[ch] : unknown;
      if (code) codes.push(code);
    }
    if (codes.length) words.push(codes.join(LETTER_SEP));
  }
  return words.join(WORD_SEP);
}
