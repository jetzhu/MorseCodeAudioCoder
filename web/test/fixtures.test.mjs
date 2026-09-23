/**
 * End-to-end decode of both recorded WAV fixtures through the JavaScript
 * chain, the counterpart of tests/test_fixtures.py:
 *
 *     WAV file -> 480-sample blocks -> Goertzel -> ToneDetector -> MorseDecoder
 *
 * `decodeSamples` below follows `morse.pipeline.decode_samples` stage for
 * stage (Goertzel dB per block, detector update, decoder feed for each final
 * run, `idle(current OFF run)` every block, detector flush and
 * `idle(Infinity)` at the end), so the text and run list it produces are what
 * the browser app produces from the same audio.  The expected run lengths are
 * the contract's figures (docs/INTERFACES.md, "Conventions"), allowed 100 ms.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { Goertzel } from "../js/dsp.js";
import { ToneDetector, isTonal } from "../js/detector.js";
import { MorseDecoder } from "../js/decoder.js";
import { Run } from "../js/runs.js";
import { readWav } from "./wav.mjs";

const ROOT = new URL("../../", import.meta.url);
const LOOPBACK = new URL("tests/fixtures/loopback_sos_1khz_15wpm.wav", ROOT);
const BEEPER = new URL("tests/fixtures/beeper_long_2491hz_1m.wav", ROOT);
const vectors = JSON.parse(readFileSync(new URL("./vectors.json", import.meta.url), "utf8"));

const FS = 48000;
const BLOCK_SIZE = 480; // 10 ms at 48 kHz
const TOLERANCE_MS = 100.0;

// ------------------------------------------------------------------ helpers

/**
 * Decode a whole signal the way `morse.pipeline.decode_samples` does.
 *
 * @param {Float32Array} samples mono samples in -1..1
 * @param {number} fs sample rate in Hz
 * @param {number} f0 tone frequency in Hz
 * @param {{wpm?: number | null, blockSize?: number}} [options]
 * @returns {{text: string, streamed: string, runs: Run[], blocks: number,
 *           detector: ToneDetector, decoder: MorseDecoder}}
 *   `text` is everything decoded including the end-of-stream flush;
 *   `streamed` is what had been emitted block by block before the flush.
 */
function decodeSamples(samples, fs, f0, { wpm = null, blockSize = BLOCK_SIZE } = {}) {
  const blockMs = (1000.0 * blockSize) / fs;
  const goertzel = new Goertzel(f0, fs, blockSize);
  const detector = new ToneDetector({ blockMs }); // contract defaults, like Pipeline
  const decoder = new MorseDecoder({ wpm });
  /** @type {Run[]} */
  const runs = [];
  let text = "";
  let blocks = 0;

  const feedBlock = (block) => {
    const powerDb = goertzel.powerDb(block);
    let sumSq = 0;
    for (let k = 0; k < block.length; k++) sumSq += block[k] * block[k];
    const levelDb = 10 * Math.log10(sumSq / block.length + 1e-12); // the pipeline's level_dbfs
    for (const run of detector.update(powerDb, isTonal(powerDb, levelDb))) {
      runs.push(run);
      text += decoder.feed(run);
    }
    const current = detector.currentRun;
    if (!current.on) text += decoder.idle(current.ms);
    blocks += 1;
  };

  const nFull = Math.floor(samples.length / blockSize);
  for (let i = 0; i < nFull; i++) {
    feedBlock(samples.subarray(i * blockSize, (i + 1) * blockSize));
  }
  const remainder = samples.length - nFull * blockSize;
  if (remainder > 0) {
    const tail = new Float32Array(blockSize); // a short tail is zero-padded to a full block
    tail.set(samples.subarray(nFull * blockSize));
    feedBlock(tail);
  }

  const streamed = text;
  for (const run of detector.flush()) {
    runs.push(run);
    text += decoder.feed(run);
  }
  text += decoder.idle(Infinity);
  return { text, streamed, runs, blocks, detector, decoder };
}

/** @param {Run[]} runs @returns {Run[]} */
function onRuns(runs) {
  return runs.filter((r) => r.on);
}

/**
 * OFF run lengths strictly between the first and the last ON run.
 * @param {Run[]} runs
 * @returns {number[]}
 */
function gapsBetweenOnRuns(runs) {
  const onIdx = runs.map((r, i) => (r.on ? i : -1)).filter((i) => i >= 0);
  const out = [];
  for (let i = onIdx[0] + 1; i < onIdx[onIdx.length - 1]; i++) {
    if (!runs[i].on) out.push(runs[i].ms);
  }
  return out;
}

/** `[on, blocks]` pairs, the form used by vectors.json. */
function pairs(runs) {
  return runs.map((r) => [r.on, r.blocks]);
}

function assertWithin(got, want, tolerance, message) {
  assert.ok(
    Math.abs(got - want) <= tolerance,
    `${message}: ${got} ms is not within ${tolerance} ms of ${want} ms`,
  );
}

/** Structural checks every decoded run list must satisfy. */
function assertWellFormed(runs, blocks, blockMs = 10.0) {
  assert.ok(runs.length >= 2, "at least one mark and one gap");
  for (const run of runs) {
    assert.ok(run instanceof Run, "the chain passes Run instances");
    assert.equal(run.blockMs, blockMs);
    assert.ok(Number.isInteger(run.blocks) && run.blocks >= 1, `run of ${run.blocks} blocks`);
  }
  for (let i = 1; i < runs.length; i++) {
    assert.notEqual(runs[i].on, runs[i - 1].on, `runs ${i - 1} and ${i} have the same state`);
  }
  assert.equal(runs[0].on, false, "the stream opens with silence");
  assert.equal(runs[runs.length - 1].on, false, "the stream ends with silence");
  assert.equal(
    runs.reduce((sum, r) => sum + r.blocks, 0),
    blocks,
    "the runs account for every block, warm-up included",
  );
}

// ------------------------------------------------------------ the fixtures

test("both fixtures are 48 kHz 16-bit mono PCM in whole 10 ms blocks", () => {
  const expected = [
    [LOOPBACK, 4.4],
    [BEEPER, 12.0],
  ];
  for (const [path, seconds] of expected) {
    const wav = readWav(path);
    assert.equal(wav.sampleRate, FS, path.pathname);
    assert.equal(wav.channels, 1, path.pathname);
    assert.equal(wav.bitsPerSample, 16, path.pathname);
    assert.equal(wav.format, "pcm", path.pathname);
    assert.equal(wav.samples.length, wav.frames);
    assert.equal(wav.samples.length / FS, seconds, path.pathname);
    assert.equal(wav.samples.length % BLOCK_SIZE, 0, "whole blocks");
    let maxAbs = 0;
    for (const v of wav.samples) maxAbs = Math.max(maxAbs, Math.abs(v));
    assert.ok(maxAbs > 0.0 && maxAbs < 1.0, `peak ${maxAbs}`);
  }
});

test("the loopback recording decodes to SOS", (t) => {
  const wav = readWav(LOOPBACK);
  const result = decodeSamples(wav.samples, wav.sampleRate, 1000.0);
  const { text, streamed, runs, blocks, decoder } = result;

  assert.equal(text.trim(), "SOS");
  assert.equal(text, decoder.text, "feed() and idle() return exactly what they append");
  assert.equal(decoder.letterCount, 3);
  assert.equal(decoder.unknownCount, 0);
  assert.equal(decoder.buffer, "", "nothing left pending");

  // The last letter must appear during the trailing silence, not only at the
  // end-of-stream flush: 7 dits of silence (about 560 ms) fit inside the
  // 860 ms of quiet room after the last dit.
  assert.equal(streamed.trim(), "SOS", "idle() flushed the last letter while streaming");
  assert.equal(text, streamed, "the flush had nothing left to add");

  assert.equal(blocks, wav.samples.length / BLOCK_SIZE);
  assertWellFormed(runs, blocks);

  // 9 marks (three letters of three symbols) keyed 80 / 240 ms at 15 WPM.
  const marks = onRuns(runs).map((r) => r.ms);
  assert.equal(marks.length, 9, `marks ${marks}`);
  const dits = [...marks.slice(0, 3), ...marks.slice(6, 9)];
  const dahs = marks.slice(3, 6);
  for (const ms of dits) assertWithin(ms, 80.0, TOLERANCE_MS, "dit");
  for (const ms of dahs) assertWithin(ms, 240.0, TOLERANCE_MS, "dah");
  for (const dit of dits) for (const dah of dahs) assert.ok(dah > dit, `dah ${dah} vs dit ${dit}`);

  // 8 gaps: six inside letters (keyed 80 ms) and two between letters (keyed 240 ms).
  const gaps = gapsBetweenOnRuns(runs);
  assert.equal(gaps.length, 8, `gaps ${gaps}`);
  const inner = [gaps[0], gaps[1], gaps[3], gaps[4], gaps[6], gaps[7]];
  const letter = [gaps[2], gaps[5]];
  for (const ms of inner) assertWithin(ms, 80.0, TOLERANCE_MS, "intra-letter gap");
  for (const ms of letter) assertWithin(ms, 240.0, TOLERANCE_MS, "letter gap");
  assert.equal(runs.length, 19, "lead-in, 9 marks, 8 gaps, trailing silence: no chatter");

  // The timing estimate recovered the keying speed.
  assertWithin(decoder.ditMs, 80.0, 20.0, "dit estimate");
  assert.ok(Math.abs(decoder.wpm - 15.0) <= 3.0, `wpm ${decoder.wpm}`);
  assert.ok(decoder.offsetMs >= 0.0 && decoder.offsetMs <= 50.0, `offset ${decoder.offsetMs} ms`);
  t.diagnostic(
    `loopback: "${text}", dit ${decoder.ditMs.toFixed(1)} ms, offset ${decoder.offsetMs.toFixed(1)} ms, ` +
      `marks ${marks.join("/")} ms, gaps ${gaps.join("/")} ms`,
  );
});

test("the beeper recording gives exactly four tones with the measured gaps", (t) => {
  const wav = readWav(BEEPER);
  const result = decodeSamples(wav.samples, wav.sampleRate, 2491.0);
  const { text, streamed, runs, blocks, decoder } = result;

  assert.equal(blocks, wav.samples.length / BLOCK_SIZE);
  assertWellFormed(runs, blocks);

  // Four ON runs of about 3700, 490, 570 and 2020 ms (docs/INTERFACES.md).
  const on = onRuns(runs);
  const lengths = on.map((r) => r.ms);
  assert.equal(on.length, 4, `expected 4 ON runs, got ${on.length}: ${lengths}`);
  const expectedOn = [3700.0, 490.0, 570.0, 2020.0];
  expectedOn.forEach((want, i) => assertWithin(lengths[i], want, TOLERANCE_MS, `ON run ${i + 1}`));

  // ... separated by gaps of about 370, 250 and 300 ms.
  const gaps = gapsBetweenOnRuns(runs);
  const expectedOff = [370.0, 250.0, 300.0];
  assert.equal(gaps.length, 3, `gaps ${gaps}`);
  expectedOff.forEach((want, i) => assertWithin(gaps[i], want, TOLERANCE_MS, `gap ${i + 1}`));

  // The first tone starts near 0.69 s and the leading silence is one OFF run.
  const firstOn = runs.findIndex((r) => r.on);
  assert.equal(firstOn, 1, "no chatter before the first tone");
  assertWithin(runs[0].ms, 690.0, TOLERANCE_MS, "start of the first tone");

  // Nothing but silence after the last tone (about 3.7 s of quiet room).
  const lastOn = runs.length - 1 - [...runs].reverse().findIndex((r) => r.on);
  const trailing = runs.slice(lastOn + 1);
  assert.ok(trailing.every((r) => !r.on));
  assert.ok(trailing.reduce((sum, r) => sum + r.ms, 0) > 3000.0);
  assert.equal(runs.length, 9);

  // Not Morse, but the decoder must take it in its stride: two dahs and a
  // dit-dah, no unknown-symbol marker, everything emitted while streaming.
  assert.equal(typeof text, "string");
  assert.ok(!text.includes("?"), `text "${text}"`);
  assert.equal(decoder.unknownCount, 0);
  assert.equal(text, decoder.text);
  assert.equal(text, streamed, "the flush had nothing left to add");
  assert.equal(decoder.buffer, "");
  t.diagnostic(`beeper: "${text}", ON ${lengths.join("/")} ms, gaps ${gaps.join("/")} ms`);
});

// -------------------------------------------- parity with the Python pipeline

for (const [name, fixture] of Object.entries(vectors.fixtures)) {
  test(`end-to-end result on ${name} equals the Python pipeline's (vectors.json)`, () => {
    const wav = readWav(new URL(fixture.file, ROOT));
    assert.equal(wav.sampleRate, fixture.fs);
    const result = decodeSamples(wav.samples, wav.sampleRate, fixture.f0, { blockSize: fixture.block_size });
    assert.equal(result.blocks, fixture.blocks);
    assert.deepEqual(pairs(result.runs), fixture.runs);
    assert.equal(result.text, fixture.text);
  });
}
