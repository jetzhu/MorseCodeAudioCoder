// Tests for web/js/table.js: table integrity, lookup, encode.
// Mirrors tests/test_table.py so the two implementations can be compared.
import { test } from "node:test";
import assert from "node:assert/strict";

import { INVERSE, LETTER_SEP, MORSE_TABLE, WORD_SEP, encode, lookup } from "../js/table.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGITS = "0123456789";
const PUNCTUATION = ".,?/='():;+-_\"@!";

/** Inverse of `encode` for round-trip tests: " / " splits words, " " letters. */
function decode(morse) {
  return morse
    .split(WORD_SEP)
    .map((word) => word.split(LETTER_SEP).filter(Boolean).map((code) => lookup(code) ?? "?").join(""))
    .join(" ");
}

// --- table integrity -------------------------------------------------------

test("table covers letters, digits and punctuation", () => {
  for (const ch of LETTERS + DIGITS + PUNCTUATION) {
    assert.ok(Object.hasOwn(MORSE_TABLE, ch), ch);
  }
  assert.equal(Object.keys(MORSE_TABLE).length, 26 + 10 + PUNCTUATION.length);
});

test("codes use only dots and dashes", () => {
  for (const [ch, code] of Object.entries(MORSE_TABLE)) {
    assert.ok(code.length > 0, ch);
    assert.match(code, /^[.-]+$/, `${ch}: ${code}`);
  }
});

test("codes are unique and INVERSE matches", () => {
  assert.equal(Object.keys(INVERSE).length, Object.keys(MORSE_TABLE).length);
  for (const [ch, code] of Object.entries(MORSE_TABLE)) assert.equal(INVERSE[code], ch);
  for (const [code, ch] of Object.entries(INVERSE)) assert.equal(MORSE_TABLE[ch], code);
});

test("known codes match the Python table", () => {
  const known = [
    ["E", "."], ["T", "-"], ["S", "..."], ["O", "---"], ["A", ".-"], ["N", "-."],
    ["Q", "--.-"], ["Z", "--.."],
    ["0", "-----"], ["1", ".----"], ["5", "....."], ["9", "----."],
    [".", ".-.-.-"], [",", "--..--"], ["?", "..--.."], ["/", "-..-."],
    ["=", "-...-"], ["'", ".----."], ["(", "-.--."], [")", "-.--.-"],
    [":", "---..."], [";", "-.-.-."], ["+", ".-.-."], ["-", "-....-"],
    ["_", "..--.-"], ['"', ".-..-."], ["@", ".--.-."], ["!", "-.-.--"],
  ];
  for (const [ch, code] of known) assert.equal(MORSE_TABLE[ch], code, ch);
});

test("tables are frozen and have no prototype keys", () => {
  assert.ok(Object.isFrozen(MORSE_TABLE));
  assert.ok(Object.isFrozen(INVERSE));
  assert.throws(() => {
    MORSE_TABLE.A = "x";
  }, TypeError);
  assert.equal(INVERSE.constructor, undefined);
  assert.equal(MORSE_TABLE.toString, undefined);
});

// --- lookup ---------------------------------------------------------------

test("lookup known codes", () => {
  assert.equal(lookup("..."), "S");
  assert.equal(lookup("---"), "O");
  assert.equal(lookup(".-"), "A");
  assert.equal(lookup("-----"), "0");
  assert.equal(lookup("..--.."), "?");
});

test("lookup round-trips every entry", () => {
  for (const [ch, code] of Object.entries(MORSE_TABLE)) assert.equal(lookup(code), ch);
});

test("lookup of unknown symbols returns null", () => {
  for (const symbols of ["", ".......", "--------", "abc", ". .", " ...", "...-...-"]) {
    assert.equal(lookup(symbols), null, JSON.stringify(symbols));
  }
  // object-prototype names must not leak through
  for (const symbols of ["constructor", "__proto__", "toString", "hasOwnProperty"]) {
    assert.equal(lookup(symbols), null, symbols);
  }
  assert.equal(lookup(undefined), null);
  assert.equal(lookup(null), null);
  assert.equal(lookup(123), null);
});

// --- encode ---------------------------------------------------------------

test("encode contract example: 'SOS X' -> '... --- ... / -..-'", () => {
  assert.equal(encode("SOS X"), "... --- ... / -..-");
});

test("encode: single word, letters separated by one space", () => {
  assert.equal(encode("SOS"), "... --- ...");
  assert.equal(encode("E"), ".");
});

test("encode is case-insensitive", () => {
  assert.equal(encode("sos"), encode("SOS"));
  assert.equal(encode("Hello"), encode("HELLO"));
});

test("encode HELLO WORLD", () => {
  assert.equal(encode("HELLO WORLD"), ".... . .-.. .-.. --- / .-- --- .-. .-.. -..");
});

test("encode digits and punctuation", () => {
  assert.equal(encode("73!"), "--... ...-- -.-.--");
  assert.equal(encode("A@B.C"), ".- .--.-. -... .-.-.- -.-.");
});

test("encode collapses whitespace runs", () => {
  assert.equal(encode("  SOS   X  "), "... --- ... / -..-");
  assert.equal(encode("A\tB\nC"), ".- / -... / -.-.");
});

test("encode empty and blank", () => {
  assert.equal(encode(""), "");
  assert.equal(encode("   "), "");
});

test("encode drops unknown characters by default", () => {
  assert.equal(encode("A~B"), ".- -...");
  assert.equal(encode("A ~ B"), ".- / -..."); // an all-unknown word disappears
  assert.equal(encode("~"), "");
});

test("encode unknown placeholder", () => {
  assert.equal(encode("A~B", "?"), ".- ? -...");
  assert.equal(encode("A ~ B", "?"), ".- / ? / -...");
});

test("encode/decode round trip over the full table", () => {
  const text = "THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS 2490.7 HZ ?!";
  assert.equal(decode(encode(text)), text);
  const every = Object.keys(MORSE_TABLE).sort().join("");
  assert.equal(decode(encode(every)), every);
});
