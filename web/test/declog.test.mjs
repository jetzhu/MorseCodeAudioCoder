// Tests for web/js/declog.js (mirrors tests/test_declog.py; the rendered strings match).
import { test } from "node:test";
import assert from "node:assert/strict";

import { DecodedLog, defaultFilename, formatClock, formatElapsed, formatIso } from "../js/declog.js";

const WALL = 1700000000123; // 2023-11-14T22:13:20.123Z

function sample() {
  const log = new DecodedLog();
  log.add("H", 1000, WALL);
  log.add("E", 1400, WALL + 400);
  log.add("L", 1800, WALL + 800);
  log.add("L", 2200, WALL + 1200);
  log.add("O ", 3000, WALL + 2000);
  log.add("", 3100, WALL + 2100);
  log.add("7", 4000, WALL + 3000);
  log.add("3", 4400, WALL + 3400);
  return log;
}

test("entries, words and text", () => {
  const log = sample();
  assert.equal(log.length, 7);
  assert.equal(log.text, "HELLO 73");
  assert.ok(!log.isEmpty);
  assert.deepEqual(log.words(), [
    { text: "HELLO", elapsedMs: 1000, wallMs: WALL, endMs: 3000 },
    { text: "73", elapsedMs: 4000, wallMs: WALL + 3000, endMs: 4400 },
  ]);
  log.add(" ", 5000, WALL + 4000);
  assert.deepEqual(log.words().map((w) => w.text), ["HELLO", "73"]);
  log.add("  A", 6000, WALL + 5000);
  assert.deepEqual(log.words().map((w) => w.text), ["HELLO", "73", "A"]);
  log.reset();
  assert.ok(log.isEmpty && log.words().length === 0 && log.text === "");
  log.add("   ", 1, WALL);
  assert.ok(log.isEmpty);
  const copy = log.entries;
  copy.push({ text: "Z", elapsedMs: 0, wallMs: 0 });
  assert.equal(log.length, 1, "entries is a copy");
});

test("formatting helpers", () => {
  assert.equal(formatClock(WALL, true), "22:13:20.123");
  assert.equal(formatIso(WALL, true), "2023-11-14T22:13:20.123");
  assert.equal(formatElapsed(0), "+0:00.0");
  assert.equal(formatElapsed(1000), "+0:01.0");
  assert.equal(formatElapsed(61260), "+1:01.3");
  assert.equal(formatElapsed(3600000), "+60:00.0");
  assert.equal(formatElapsed(-5), "+0:00.0");
  assert.equal(formatElapsed(NaN), "+0:00.0");
  assert.equal(defaultFilename(WALL, "txt", true), "morse_log_20231114_221320.txt");
  assert.equal(defaultFilename(WALL, ".csv", true), "morse_log_20231114_221320.csv");
});

const EXPECTED_TEXT = `Beeper Morse Console decoded log
Exported: 2023-11-14 22:13:30
Source: Replay: sos.wav
Tone: 1000 Hz
Words: 2
Time is the computer clock when the word's first letter was decoded; elapsed is audio time since the stream started.

time           elapsed  word
22:13:20.123   +0:01.0  HELLO
22:13:23.123   +0:04.0  73
`;

const EXPECTED_CSV = `time,elapsed_s,word
2023-11-14T22:13:20.123,1.00,HELLO
2023-11-14T22:13:23.123,4.00,73
`;

test("renderText and renderCsv match the Python output", () => {
  const log = sample();
  const header = [["Source", "Replay: sos.wav"], ["Tone", "1000 Hz"]];
  assert.equal(log.renderText(header, { nowMs: WALL + 10000, utc: true }), EXPECTED_TEXT);
  assert.equal(log.renderCsv({ utc: true }), EXPECTED_CSV);
  const bare = log.renderText([], { utc: true });
  assert.ok(!bare.includes("Exported:") && bare.startsWith("Beeper Morse Console decoded log\nWords: 2\n"));
  const empty = new DecodedLog();
  assert.ok(empty.renderText([], { utc: true }).endsWith("word\n"));
  assert.equal(empty.renderCsv(), "time,elapsed_s,word\n");
});

test("CSV quotes awkward words", () => {
  const log = new DecodedLog();
  log.add('A,B"C"', 0, WALL); // no space: one word with a comma and quotes
  assert.equal(log.renderCsv({ utc: true }).split("\n")[1], '2023-11-14T22:13:20.123,0.00,"A,B""C"""');
});
