// The Morse chart: the characters the shared table covers, in display order,
// and how a cell lights up. Mirror of morse/reference.py; the Reference strip
// (section G) is built from chartEntries() so it can never disagree with what
// the decoder and encoder use.
import { MORSE_TABLE } from "./table.js";

/**
 * `[character, code]` pairs: letters A to Z, digits 0 to 9, then punctuation
 * ordered by code length, then the code itself, then the character.
 * @param {Record<string, string>} [table]
 * @returns {Array<[string, string]>}
 */
export function chartEntries(table = MORSE_TABLE) {
  const pairs = Object.entries(table);
  const byChar = (a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  const letters = pairs.filter(([ch]) => /^[A-Z]$/.test(ch)).sort(byChar);
  const digits = pairs.filter(([ch]) => /^[0-9]$/.test(ch)).sort(byChar);
  const other = pairs
    .filter(([ch]) => !/^[A-Z0-9]$/.test(ch))
    .sort((a, b) => a[1].length - b[1].length || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0) || byChar(a, b));
  return [...letters, ...digits, ...other];
}

/**
 * How a chart cell shows against the symbols keyed so far: "match" when
 * `buffer` is exactly `code`, "prefix" when the code could still become
 * `buffer` plus more symbols, "" otherwise and always for an empty buffer.
 * @param {string} code @param {string} buffer
 * @returns {"match" | "prefix" | ""}
 */
export function cellState(code, buffer) {
  if (!buffer) return "";
  if (code === buffer) return "match";
  if (code.startsWith(buffer)) return "prefix";
  return "";
}
