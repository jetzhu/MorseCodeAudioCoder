// Tests for web/js/reference.js (mirrors tests/test_reference.py).
import { test } from "node:test";
import assert from "node:assert/strict";

import { cellState, chartEntries } from "../js/reference.js";
import { MORSE_TABLE } from "../js/table.js";

test("chart covers the table in display order", () => {
  const entries = chartEntries();
  assert.equal(entries.length, Object.keys(MORSE_TABLE).length);
  assert.equal(entries.length, 52);
  assert.deepEqual(Object.fromEntries(entries), { ...MORSE_TABLE });
  const chars = entries.map(([ch]) => ch);
  assert.deepEqual(chars.slice(0, 26), Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)));
  assert.deepEqual(chars.slice(26, 36), Array.from({ length: 10 }, (_, i) => String(i)));
  const punctuation = entries.slice(36);
  assert.ok(punctuation.every(([ch]) => !/^[A-Z0-9]$/.test(ch)));
  const keys = punctuation.map(([ch, code]) => `${String(code.length).padStart(2, "0")}|${code}|${ch}`);
  assert.deepEqual(keys, [...keys].sort());
  assert.deepEqual(entries[0], ["A", ".-"]);
  assert.deepEqual(entries[26], ["0", "-----"]);
  assert.deepEqual(chartEntries({ B: "-...", A: ".-", 1: ".----", "?": "..--.." }), [
    ["A", ".-"], ["B", "-..."], ["1", ".----"], ["?", "..--.."],
  ]);
});

test("cellState lights matches and prefixes", () => {
  assert.equal(cellState(".-", ""), "");
  assert.equal(cellState(".-", ".-"), "match");
  assert.equal(cellState(".--", ".-"), "prefix");
  assert.equal(cellState(".", ".-"), "");
  assert.equal(cellState("-", "."), "");
  const lit = Object.fromEntries(chartEntries().map(([ch, code]) => [ch, cellState(code, ".-")]));
  assert.equal(lit.A, "match");
  const expected = new Set(Object.entries(MORSE_TABLE).filter(([, code]) => code.startsWith(".-") && code !== ".-").map(([ch]) => ch));
  assert.deepEqual(new Set(Object.entries(lit).filter(([, s]) => s === "prefix").map(([ch]) => ch)), expected);
  assert.ok(expected.has("W") && expected.has("R") && expected.has("P"));
});
