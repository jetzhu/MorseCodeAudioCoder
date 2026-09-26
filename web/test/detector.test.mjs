/**
 * Tests for web/js/detector.js (ToneDetector version 2), mirrored from
 * tests/test_tone_detector.py, plus run-list parity against the Python
 * reference on both recorded fixtures (web/test/vectors.json).
 *
 * As in the Python suite, "noise" is never added in the dB domain: every
 * noise series is Gaussian audio at a stated dBFS level run through the JS
 * Goertzel in 480-sample blocks, so the single-bin statistics are the real
 * ones.  The random source is a seeded generator local to this file, so the
 * exact streams differ from numpy's; the assertions are the statistical
 * acceptance criteria from the contract, which do not depend on the stream.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { DETECTOR_DEFAULTS, ToneDetector } from "../js/detector.js";
import { Goertzel } from "../js/dsp.js";
import { Run } from "../js/runs.js";
import { readWav } from "./wav.mjs";

const FS = 48000;
const BLOCK = 480;
const ROOT = new URL("../../", import.meta.url);
const vectors = JSON.parse(readFileSync(new URL("./vectors.json", import.meta.url), "utf8"));

// Clean levels for the structural tests: 70 dB apart, both well above the
// -100 dB silence clamp.
const OFF_DB = -90.0;
const ON_DB = -20.0;
// Structural patterns open with this many OFF blocks so the 30-block warm-up
// is over before the first mark.
const LEAD = 40;

// ------------------------------------------------------------------ helpers

/** Feed every value through update() and collect all emitted runs in order. */
function feed(det, values) {
  const out = [];
  for (const v of values) out.push(...det.update(v));
  return out;
}

/** Expand [[on, blocks], ...] into one clean dB value per block. */
function levels(pattern, onDb = ON_DB, offDb = OFF_DB) {
  const out = [];
  for (const [on, blocks] of pattern) for (let i = 0; i < blocks; i++) out.push(on ? onDb : offDb);
  return out;
}

/** Repeat a value. */
function times(value, count) {
  return new Array(count).fill(value);
}

/** `[on, blocks, blockMs]` triples, the comparable form of a run list. */
function plain(runs) {
  return runs.map((r) => [r.on, r.blocks, r.blockMs]);
}

/** Expected triples for a pattern. */
function runsOf(pattern, blockMs = 10.0) {
  return pattern.map(([on, blocks]) => [on, blocks, blockMs]);
}

function assertRuns(runs, pattern, blockMs = 10.0) {
  assert.deepEqual(plain(runs), runsOf(pattern, blockMs));
}

const toDb = (power) => 10.0 * Math.log10(power + 1e-12);
const fromDb = (powerDb) => Math.pow(10.0, powerDb / 10.0);

/**
 * What floorDb/peakDb report for a level tracked exactly at `levelDb`: the
 * properties use 10*log10(x + 1e-12), so -90 dB reads -89.9957 dB.
 */
const reads = (levelDb) => toDb(fromDb(levelDb));

/** pytest.approx: |actual - expected| <= max(rel * |expected|, abs). */
function approx(actual, expected, { rel = 1e-6, abs = 1e-12 } = {}, message = "") {
  const tol = Math.max(rel * Math.abs(expected), abs);
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${message ? message + ": " : ""}${actual} is not within ${tol} of ${expected}`,
  );
}

/** Seeded uniform generator (mulberry32) with a Box-Muller Gaussian. */
function rng(seed) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u1 = uniform();
    while (u1 <= 1e-300) u1 = uniform();
    const u2 = uniform();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
  const integer = (lo, hi) => lo + Math.floor(uniform() * (hi - lo)); // [lo, hi)
  return { uniform, normal, integer };
}

/** Goertzel dB per block of a mono signal (whole blocks only). */
function goertzelSeries(x, f0, fs = FS, block = BLOCK) {
  const g = new Goertzel(f0, fs, block);
  const n = Math.floor(x.length / block);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = g.powerDb(x.subarray(i * block, (i + 1) * block));
  return out;
}

/** White Gaussian noise with the given rms level in dBFS. */
function gaussian(dbfs, samples, r) {
  const sigma = Math.pow(10.0, dbfs / 20.0);
  const out = new Float32Array(samples);
  for (let i = 0; i < samples; i++) out[i] = sigma * r.normal();
  return out;
}

function noiseSeries(dbfs, seconds, seed, f0 = 2491.0) {
  return goertzelSeries(gaussian(dbfs, Math.round(seconds * FS), rng(seed)), f0);
}

/**
 * Keyed sine for tests, like morse.dsp.synth_keyed_tone: pattern is
 * [[on, ms], ...]; reverbMs > 0 gives the envelope an exponential release.
 */
function synthKeyedTone(pattern, f0, fs, amplitude = 0.3, reverbMs = 0.0) {
  const lengths = pattern.map(([, ms]) => Math.max(0, Math.round((ms * fs) / 1000.0)));
  const n = lengths.reduce((a, b) => a + b, 0);
  const env = new Float64Array(n);
  let pos = 0;
  pattern.forEach(([on], k) => {
    if (on) env.fill(1.0, pos, pos + lengths[k]);
    pos += lengths[k];
  });
  if (reverbMs > 0.0 && n > 0) {
    const tau = (reverbMs * fs) / 1000.0;
    let lastOn = -1;
    for (let i = 0; i < n; i++) {
      if (env[i] > 0.0) lastOn = i;
      env[i] = lastOn >= 0 ? Math.exp(-(i - lastOn) / tau) : 0.0;
    }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = amplitude * env[i] * Math.sin((2.0 * Math.PI * f0 * i) / fs);
    out[i] = Math.max(-1.0, Math.min(1.0, v));
  }
  return out;
}

/** (on, ms) sequence for a dot/dash code such as "... --- ...": dit T, dah 3T, gaps 1T / 3T / 7T. */
function keyedPattern(code, wpm, leadMs, tailMs) {
  const dit = 1200.0 / wpm;
  const pat = [[false, leadMs]];
  for (const ch of code) {
    if (ch === ".") pat.push([true, dit], [false, dit]);
    else if (ch === "-") pat.push([true, 3 * dit], [false, dit]);
    else if (ch === " ") pat[pat.length - 1] = [false, 3 * dit];
    else if (ch === "/") pat[pat.length - 1] = [false, 7 * dit];
  }
  pat[pat.length - 1] = [false, tailMs];
  return pat;
}

const SOS = "... --- ...";

/**
 * Goertzel series of SOS at 15 WPM, snrDb above Gaussian noise; the tone's rms
 * is snrDb above the noise rms.  silenceS seconds of digital zeros are prepended.
 */
function sosInNoise(noiseDbfs, seed, { snrDb = 20.0, leadMs = 1000.0, tailMs = 1000.0, silenceS = 0.0, reverbMs = 0.0, f0 = 1000.0 } = {}) {
  const r = rng(seed);
  const amplitude = Math.pow(10.0, (noiseDbfs + snrDb) / 20.0) * Math.SQRT2;
  const tone = synthKeyedTone(keyedPattern(SOS, 15.0, leadMs, tailMs), f0, FS, amplitude, reverbMs);
  const noise = gaussian(noiseDbfs, tone.length, r);
  const silence = Math.round(silenceS * FS);
  const x = new Float32Array(silence + tone.length);
  for (let i = 0; i < tone.length; i++) x[silence + i] = tone[i] + noise[i];
  return goertzelSeries(x, f0);
}

/** (startBlock, blocks) of every ON run. */
function onStarts(runs) {
  const out = [];
  let t = 0;
  for (const r of runs) {
    if (r.on) out.push([t, r.blocks]);
    t += r.blocks;
  }
  return out;
}

/** Nine ON runs of dit, dit, dit, dah, dah, dah, dit, dit, dit. */
function assertSosRuns(runs, ditMs = 80.0, tolMs = 20.0) {
  const marks = runs.filter((r) => r.on).map((r) => r.ms);
  assert.equal(marks.length, 9, `marks ${JSON.stringify(marks)}`);
  marks.forEach((ms, i) => {
    const want = i >= 3 && i < 6 ? 3 * ditMs : ditMs;
    assert.ok(Math.abs(ms - want) <= tolMs, `mark ${i}: ${ms} ms, want ${want}: ${JSON.stringify(marks)}`);
  });
}

/**
 * (hi, lo) offsets above the noise level for a tracked SNR, per the contract:
 * noise-referenced lifts at low SNR; at high SNR the OFF threshold follows the
 * signal (S - releaseDb) and hi sits hysteresisDb above it.
 */
function expectedLifts(snr) {
  const liftOn = Math.min(30.0, Math.max(12.0, 0.6 * snr));
  const liftOff = Math.min(24.0, Math.max(6.0, 0.4 * snr));
  const lo = Math.max(liftOff, snr - 12.0);
  const hi = Math.max(liftOn, lo + 3.0);
  return [hi, lo];
}

// ----------------------------------------------------------------- defaults

test("defaults match the contract and vectors.json", () => {
  const det = new ToneDetector();
  assert.equal(det.blockMs, 10.0);
  assert.deepEqual([det.onFrac, det.offFrac], [0.6, 0.4]);
  assert.equal(det.minRunBlocks, 2);
  assert.deepEqual([det.minLiftDb, det.maxLiftDb, det.minOffLiftDb], [12.0, 30.0, 6.0]);
  // Equal alphas keep N an unbiased linear mean (Python module docstring).
  assert.equal(det.noiseAlphaUp, 0.1);
  assert.equal(det.noiseAlphaDown, 0.1);
  assert.equal(det.signalAlpha, 0.1);
  assert.equal(det.signalDecayDb, 0.1);
  assert.equal(det.silenceFloorDb, -100.0);
  assert.equal(det.warmupBlocks, 30);
  assert.equal(det.maxOnBlocks, 500);
  assert.equal(det.releaseDb, 12.0);
  assert.equal(det.hysteresisDb, 3.0);
  assert.equal(det.warmingUp, true);
  assert.equal(det.state, false);
  assert.ok(det.currentRun instanceof Run);
  assert.deepEqual(plain([det.currentRun]), [[false, 0, 10.0]]);

  // The same numbers the Python constructor exported, under camelCase names.
  const snakeToCamel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
  const fromPython = Object.fromEntries(Object.entries(vectors.params.detector).map(([k, v]) => [snakeToCamel(k), v]));
  assert.deepEqual({ ...DETECTOR_DEFAULTS }, fromPython);
  for (const [key, value] of Object.entries(fromPython)) assert.equal(det[key], value, key);
  assert.equal(vectors.params.detector_constants.warmup_alpha, 0.3);
});

test("invalid parameters are rejected", () => {
  const bad = [
    { blockMs: 0.0 },
    { onFrac: 0.5, offFrac: 0.6 },
    { onFrac: 1.2 },
    { offFrac: -0.1 },
    { minRunBlocks: 0 },
    { minLiftDb: -1.0 },
    { maxLiftDb: 10.0 }, // below minLiftDb
    { minOffLiftDb: -1.0 },
    { noiseAlphaUp: 0.0 },
    { noiseAlphaDown: 1.5 },
    { signalAlpha: -0.1 },
    { signalDecayDb: -0.1 },
    { warmupBlocks: -1 },
    { maxOnBlocks: 0 },
    { releaseDb: -1.0 },
    { hysteresisDb: -0.5 },
  ];
  for (const kwargs of bad) {
    assert.throws(() => new ToneDetector(kwargs), RangeError, JSON.stringify(kwargs));
  }
  // Python-style names and typos are caught rather than silently ignored.
  assert.throws(() => new ToneDetector({ min_run_blocks: 3 }), TypeError);
  assert.throws(() => new ToneDetector({ blockMs: "10" }), TypeError);
  assert.throws(() => new ToneDetector({ blockMs: NaN }), TypeError);
  // Valid custom values are stored.
  const det = new ToneDetector({ minRunBlocks: 3, warmupBlocks: 0, blockMs: 5.0 });
  assert.equal(det.minRunBlocks, 3);
  assert.equal(det.warmupBlocks, 0);
  assert.equal(det.blockMs, 5.0);
  assert.equal(det.warmingUp, false);
});

// ----------------------------------------------------------- level tracking

test("the first block primes the noise and signal levels", () => {
  const det = new ToneDetector();
  assert.deepEqual(det.update(-50.0), []);
  approx(det.floorDb, -50.0, { abs: 1e-6 });
  approx(det.peakDb, -50.0, { abs: 1e-6 });
  // snr 0: lifts at their minimums
  approx(det.thresholdHiDb, -38.0, { abs: 1e-6 });
  approx(det.thresholdLoDb, -44.0, { abs: 1e-6 });
  assert.equal(det.state, false);
  assert.equal(det.warmingUp, true);
  assert.deepEqual(plain([det.currentRun]), [[false, 1, 10.0]]);
});

test("warm-up forces OFF and smooths the noise level with alpha 0.3", () => {
  const det = new ToneDetector();
  det.update(-90.0);
  let n = fromDb(-90.0);
  // every warm-up block, above or below N, uses alpha 0.3
  det.update(-80.0);
  n += 0.3 * (fromDb(-80.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });
  det.update(-100.0);
  n += 0.3 * (fromDb(-100.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });
  // a loud tone from the very start is learned as noise while warming up
  for (let i = 3; i < 30; i++) {
    assert.equal(det.warmingUp, true);
    assert.deepEqual(det.update(ON_DB), []);
    assert.equal(det.state, false);
  }
  assert.equal(det.warmingUp, false);
  approx(det.floorDb, ON_DB, { abs: 0.1 });
  approx(det.peakDb, det.floorDb);
  assert.deepEqual(det.update(ON_DB), []);
  assert.equal(det.state, false); // not above N + 12 dB
  assertRuns(det.flush(), [[false, 31]]);
});

test("warm-up settles on a fade-in before any verdict", () => {
  // A stream opening with a fade-in from digital silence (both fixtures do)
  // must land on the real level within the warm-up and not trip ON after it.
  const fade = [];
  for (let i = 0; i < 25; i++) fade.push(-119.0 + ((-82.0 + 119.0) * i) / 24);
  const det = new ToneDetector();
  feed(det, fade);
  feed(det, times(-82.0, 5));
  assert.equal(det.warmingUp, false);
  approx(det.floorDb, -82.0, { abs: 1.0 });
  for (let k = 0; k < 4; k++) {
    for (const v of [-80.0, -84.0, -78.0, -86.0, -79.0]) {
      det.update(v);
      assert.equal(det.state, false);
    }
  }
  assertRuns(det.flush(), [[false, 50]]);
});

test("the noise level never drops below the silence floor", () => {
  let det = new ToneDetector();
  feed(det, times(-120.0, 60)); // digital silence, well past the warm-up
  approx(det.floorDb, reads(-100.0), { abs: 1e-9 });
  assert.ok(-100.0 <= det.floorDb && det.floorDb <= -99.9);
  approx(det.thresholdHiDb, det.floorDb + 12.0);
  approx(det.thresholdLoDb, det.floorDb + 6.0);
  // a tone just above the clamped ON threshold is heard at once ...
  det.update(-87.0);
  assert.equal(det.state, true);
  // ... one just below it is not (and, being OFF, it lifts the noise level)
  det = new ToneDetector();
  feed(det, times(-120.0, 60));
  det.update(-89.0);
  assert.equal(det.state, false);
  assert.ok(det.floorDb > reads(-100.0));
});

test("a custom silence floor is honoured", () => {
  const det = new ToneDetector({ silenceFloorDb: -80.0 });
  feed(det, times(-120.0, 40));
  approx(det.floorDb, -80.0, { abs: 0.05 });
  approx(det.thresholdHiDb, det.floorDb + 12.0);
});

test("the noise level is updated only by OFF blocks", () => {
  const det = new ToneDetector();
  feed(det, times(OFF_DB, LEAD));
  const floor = det.floorDb;
  feed(det, times(ON_DB, 50));
  assert.equal(det.state, true);
  assert.equal(det.floorDb, floor); // ON blocks leave N alone
  det.update(OFF_DB);
  assert.equal(det.floorDb, floor); // p == N: no change either way
  det.update(-85.0);
  assert.ok(det.floorDb > floor);
});

test("the signal level follows ON blocks with signalAlpha", () => {
  const det = new ToneDetector();
  feed(det, times(OFF_DB, LEAD));
  let s = fromDb(OFF_DB); // S starts at N, which sits exactly at the constant input
  for (let i = 0; i < 3; i++) {
    det.update(ON_DB);
    s += 0.1 * (fromDb(ON_DB) - s);
    assert.equal(det.state, true);
    approx(det.peakDb, toDb(s), { abs: 1e-6 });
  }
  // first ON block puts S 10 dB under a tone that starts at N + 70 dB
  assert.ok(det.peakDb < ON_DB);
});

test("the signal level decays toward the noise level while OFF and never below", () => {
  const det = new ToneDetector();
  feed(det, times(OFF_DB, LEAD));
  feed(det, times(ON_DB, 100));
  const peak = det.peakDb;
  approx(peak, ON_DB, { abs: 0.01 });
  feed(det, times(OFF_DB, 100));
  assert.equal(det.state, false);
  approx(det.peakDb, peak - 100 * det.signalDecayDb, { abs: 0.01 });
  feed(det, times(OFF_DB, 3000));
  approx(det.peakDb, det.floorDb, { abs: 1e-9 });
  assert.ok(det.peakDb >= det.floorDb);
});

test("lifts follow the tracked SNR with min and max, then the signal-referenced release", () => {
  const cases = [
    [-77.0, 12.0, 6.0], // snr 13: noise-referenced minimums (a tone at exactly N + 12 dB is not *above* the threshold)
    [-75.0, 12.0, 6.0], // snr 15: 0.4*snr == 6, still the minimum
    [-60.0, 21.0, 18.0], // snr 30: lo = S - 12 = N + 18 beats 0.4*snr; hi = lo + 3
    [-45.0, 36.0, 33.0], // snr 45: signal-referenced, lo = N + 33
    [-30.0, 51.0, 48.0], // snr 60
    [-10.0, 71.0, 68.0], // snr 80: thresholds keep following the signal
  ];
  for (const [toneDb, liftOn, liftOff] of cases) {
    const det = new ToneDetector();
    feed(det, times(-90.0, LEAD));
    feed(det, times(toneDb, 150)); // S converges to the tone (0.9**150 ~ 1e-7)
    assert.equal(det.state, true, `tone ${toneDb}`);
    approx(det.peakDb, toneDb, { abs: 0.01 }, `tone ${toneDb} peak`);
    approx(det.floorDb, reads(-90.0), { abs: 1e-9 }, `tone ${toneDb} floor`);
    approx(det.thresholdHiDb, det.floorDb + liftOn, { abs: 0.02 }, `tone ${toneDb} hi`);
    approx(det.thresholdLoDb, det.floorDb + liftOff, { abs: 0.02 }, `tone ${toneDb} lo`);
  }
});

test("noise tracking is a linear EMA in both directions", () => {
  let det = new ToneDetector();
  feed(det, times(-90.0, LEAD));
  let n = fromDb(-90.0); // a constant input leaves the EMA exactly at that power
  approx(det.floorDb, toDb(n), { abs: 1e-9 });
  det.update(-80.0); // above N: noiseAlphaUp
  n += 0.1 * (fromDb(-80.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });
  det.update(-100.0); // below N: noiseAlphaDown (same value by default)
  n += 0.1 * (fromDb(-100.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });

  // The direction rule is honoured when the alphas differ.
  det = new ToneDetector({ noiseAlphaUp: 0.02, noiseAlphaDown: 0.2 });
  feed(det, times(-90.0, LEAD));
  n = fromDb(-90.0);
  det.update(-80.0);
  n += 0.02 * (fromDb(-80.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });
  det.update(-100.0);
  n += 0.2 * (fromDb(-100.0) - n);
  approx(det.floorDb, toDb(n), { abs: 1e-6 });
});

test("thresholds are always consistent with the levels", () => {
  const r = rng(3);
  const det = new ToneDetector();
  const values = [];
  for (let i = 0; i < 20; i++) values.push(-119.0 + ((-80.0 + 119.0) * i) / 19);
  for (let i = 0; i < 200; i++) values.push(-80.0 + 5.0 * r.normal());
  values.push(...times(-30.0, 40), ...times(-80.0, 40), ...times(-50.0, 20), ...times(-95.0, 40), ...times(-20.0, 600));
  for (const v of values) {
    det.update(v);
    const snr = det.peakDb - det.floorDb;
    assert.ok(snr >= 0.0);
    const [liftOn, liftOff] = expectedLifts(snr);
    approx(det.thresholdHiDb, det.floorDb + liftOn, { abs: 1e-6 });
    approx(det.thresholdLoDb, det.floorDb + liftOff, { abs: 1e-6 });
    assert.ok(det.floorDb <= det.thresholdLoDb && det.thresholdLoDb < det.thresholdHiDb && det.thresholdHiDb <= det.peakDb + 30.0);
  }
});

test("the stuck-ON timeout ends the run and adopts the level", () => {
  const det = new ToneDetector();
  let got = feed(det, times(OFF_DB, LEAD));
  // 499 ON blocks: nothing final yet beyond the lead
  got.push(...feed(det, times(ON_DB, 499)));
  assertRuns(got, [[false, LEAD]]);
  assert.equal(det.state, true);
  // block 500 is still ON (the run reaches maxOnBlocks) ...
  assert.deepEqual(det.update(ON_DB), []);
  assert.equal(det.state, true);
  assert.deepEqual(plain([det.currentRun]), [[true, 500, 10.0]]);
  // ... block 501 is forced OFF and N jumps to S
  assert.deepEqual(det.update(ON_DB), []);
  assert.equal(det.state, false);
  approx(det.floorDb, det.peakDb);
  approx(det.floorDb, ON_DB, { abs: 0.01 });
  // the ON run is final once the OFF run after it reaches minRunBlocks,
  // i.e. maxOnBlocks + minRunBlocks blocks after it began
  assertRuns(det.update(ON_DB), [[true, 500]]);
  // the same level from now on is noise, never ON again
  got = feed(det, times(ON_DB, 200));
  assert.deepEqual(got, []);
  assert.equal(det.state, false);
  assertRuns(det.flush(), [[false, 202]]);
});

test("the stuck-ON timeout recovers when the level drops again", () => {
  const det = new ToneDetector({ maxOnBlocks: 100 });
  feed(det, times(OFF_DB, LEAD));
  feed(det, times(ON_DB, 105));
  assert.equal(det.state, false);
  approx(det.floorDb, ON_DB, { abs: 0.01 });
  // back to quiet: N follows down (0.1 per block in the linear domain, so
  // roughly 0.46 dB per block) and a tone is heard again once it is 12 dB up
  feed(det, times(OFF_DB, 200));
  approx(det.floorDb, OFF_DB, { abs: 0.5 });
  det.update(ON_DB);
  assert.equal(det.state, true);
});

test("reset restarts the warm-up but flush does not", () => {
  const det = new ToneDetector();
  feed(det, levels([[false, LEAD], [true, 5]]));
  assert.equal(det.warmingUp, false);
  det.flush();
  assert.equal(det.warmingUp, false);
  approx(det.floorDb, reads(OFF_DB), { abs: 1e-9 }); // levels survive a flush
  det.update(ON_DB);
  assert.equal(det.state, true);
  det.reset();
  assert.equal(det.warmingUp, true);
  assert.equal(det.state, false);
  assert.equal(det.currentRun.blocks, 0);
  assert.deepEqual(det.flush(), []);
  det.update(-30.0);
  approx(det.floorDb, reads(-30.0), { abs: 1e-9 });
  approx(det.peakDb, reads(-30.0), { abs: 1e-9 });
  assert.equal(det.state, false); // warming up again
});

test("non-finite power is treated as silence", () => {
  const det = new ToneDetector();
  feed(det, times(OFF_DB, LEAD));
  det.update(NaN);
  det.update(Infinity);
  det.update(-Infinity);
  assert.ok(Number.isFinite(det.floorDb) && Number.isFinite(det.thresholdHiDb));
  assert.equal(det.state, false);
  // Non-numbers are a programming error, not silence (Python's float() raises too).
  for (const bad of ["-50", null, undefined, {}, [-50]]) {
    assert.throws(() => det.update(bad), TypeError, String(bad));
  }
  assertRuns(det.flush(), [[false, LEAD + 3]]);
});

// ---------------------------------------------------------- run generation

test("a clean step sequence gives exact runs", () => {
  const pattern = [[false, LEAD], [true, 5], [false, 3], [true, 8], [false, 10]];
  const det = new ToneDetector();
  const got = feed(det, levels(pattern));
  got.push(...det.flush());
  assertRuns(got, pattern);
  assert.ok(got.every((r) => r instanceof Run && r.blockMs === 10.0));
  assert.equal(got[1].ms, 50.0);
});

test("blockMs is propagated into runs", () => {
  const det = new ToneDetector({ blockMs: 5.0 });
  const pattern = [[false, LEAD], [true, 6], [false, 4]];
  const got = feed(det, levels(pattern));
  got.push(...det.flush());
  assertRuns(got, pattern, 5.0);
  assert.equal(got[1].ms, 30.0);
  assert.equal(det.currentRun.blockMs, 5.0);
});

test("a one-block dropout inside an ON run is debounced", () => {
  const values = levels([[false, LEAD], [true, 5], [false, 1], [true, 5], [false, 10]]);
  const det = new ToneDetector({ minRunBlocks: 2 });
  const got = feed(det, values);
  got.push(...det.flush());
  assertRuns(got, [[false, LEAD], [true, 11], [false, 10]]);
});

test("a one-block spike during OFF does not create a run", () => {
  const values = levels([[false, LEAD], [true, 1], [false, 10]]);
  const det = new ToneDetector({ minRunBlocks: 2 });
  feed(det, values.slice(0, LEAD));
  assert.equal(det.state, false);
  // The spike does flip the raw verdict for one block...
  det.update(values[LEAD]);
  assert.equal(det.state, true);
  // ...but debouncing folds it back into one OFF run.
  const got = feed(det, values.slice(LEAD + 1));
  got.push(...det.flush());
  assertRuns(got, [[false, LEAD + 11]]);
});

test("currentRun tracks the in-progress run and merges", () => {
  const det = new ToneDetector({ minRunBlocks: 2 });
  assert.equal(det.currentRun.blocks, 0);
  feed(det, times(OFF_DB, LEAD));
  assert.deepEqual(plain([det.currentRun]), [[false, LEAD, 10.0]]);
  feed(det, times(ON_DB, 3));
  assert.deepEqual(plain([det.currentRun]), [[true, 3, 10.0]]);
  det.update(OFF_DB);
  assert.deepEqual(plain([det.currentRun]), [[false, 1, 10.0]]);
  det.update(ON_DB); // the 1-block OFF merges back into the ON run
  assert.deepEqual(plain([det.currentRun]), [[true, 5, 10.0]]);
});

test("runs are not returned before they are final", () => {
  for (const minRunBlocks of [2, 3, 4]) {
    const det = new ToneDetector({ minRunBlocks });
    assert.deepEqual(feed(det, times(OFF_DB, LEAD)), []);
    const got = feed(det, times(ON_DB, 50));
    // the leading OFF run comes out once the ON run is long enough
    assertRuns(got, [[false, LEAD]]);
    // a long ON run is still not final while the following OFF is too short
    for (let i = 0; i < minRunBlocks - 1; i++) {
      assert.deepEqual(det.update(OFF_DB), []);
      assert.equal(det.currentRun.on, false);
    }
    assertRuns(det.update(OFF_DB), [[true, 50]]);
    assert.deepEqual(plain([det.currentRun]), [[false, minRunBlocks, 10.0]]);
  }
});

test("flush emits the tail and clears pending runs", () => {
  let det = new ToneDetector();
  assert.deepEqual(det.flush(), []);
  let got = feed(det, levels([[false, LEAD], [true, 5]]));
  assertRuns(got, [[false, LEAD]]);
  assertRuns(det.flush(), [[true, 5]]);
  assert.deepEqual(det.flush(), []);
  assert.equal(det.currentRun.blocks, 0);

  // A short current run behind a pending run: both come out, oldest first.
  det = new ToneDetector();
  feed(det, levels([[false, LEAD], [true, 5], [false, 1]]));
  assertRuns(det.flush(), [[true, 5], [false, 1]]);

  // Levels survive a flush, so the stream can continue.
  approx(det.floorDb, reads(OFF_DB), { abs: 1e-9 });
  got = feed(det, levels([[false, 4], [true, 3], [false, 2]]));
  got.push(...det.flush());
  assertRuns(got, [[false, 4], [true, 3], [false, 2]]);
});

test("power between lo and hi does not toggle the verdict (hysteresis)", () => {
  const det = new ToneDetector();
  const got = feed(det, levels([[false, LEAD], [true, 20], [false, 20]]));
  assert.equal(det.state, false);
  // While OFF, a block inside the band leaves the verdict OFF.  (It counts
  // as noise, so the thresholds move; check against the band of each block.)
  for (let i = 0; i < 5; i++) {
    const v = 0.5 * (det.thresholdLoDb + det.thresholdHiDb);
    got.push(...det.update(v));
    assert.equal(det.state, false);
  }
  // Switch ON, then hover inside the band: it must stay ON.
  got.push(...feed(det, times(ON_DB, 5)));
  assert.equal(det.state, true);
  for (let i = 0; i < 15; i++) {
    const v = 0.5 * (det.thresholdLoDb + det.thresholdHiDb);
    got.push(...det.update(v));
    assert.equal(det.state, true);
  }
  got.push(...det.flush());
  assertRuns(got, [[false, LEAD], [true, 20], [false, 25], [true, 20]]);
});

test("minRunBlocks 1 disables the debounce", () => {
  const pattern = [[false, LEAD], [true, 1], [false, 1], [true, 2], [false, 3]];
  const det = new ToneDetector({ minRunBlocks: 1 });
  const got = feed(det, levels(pattern));
  got.push(...det.flush());
  assertRuns(got, pattern);
});

test("a leading short run joins the run that follows it", () => {
  // One quiet block, then tone: the 1-block OFF has no earlier neighbour,
  // so it is absorbed into the ON run rather than emitted on its own.
  // (No warm-up, so the very first blocks are judged.)
  const det = new ToneDetector({ minRunBlocks: 2, warmupBlocks: 0 });
  const got = feed(det, levels([[false, 1], [true, 10], [false, 10]]));
  got.push(...det.flush());
  assertRuns(got, [[true, 11], [false, 10]]);
});

test("emitted runs alternate, conserve blocks and respect the minimum length", () => {
  // Random keying only 10 dB above Gaussian noise, through the real
  // Goertzel: glitchy enough to exercise every merge path.
  const r = rng(1234);
  const pattern = [[false, 500.0]];
  for (let i = 0; i < 40; i++) {
    pattern.push([true, r.integer(1, 12) * 10.0]);
    pattern.push([false, r.integer(1, 12) * 10.0]);
  }
  const amplitude = Math.pow(10.0, (-60.0 + 10.0) / 20.0) * Math.SQRT2;
  const tone = synthKeyedTone(pattern, 1000.0, FS, amplitude);
  const noise = gaussian(-60.0, tone.length, r);
  const x = new Float32Array(tone.length);
  for (let i = 0; i < x.length; i++) x[i] = tone[i] + noise[i];
  const values = goertzelSeries(x, 1000.0);
  for (const minRunBlocks of [2, 3]) {
    const det = new ToneDetector({ minRunBlocks });
    const got = feed(det, values);
    const tail = det.flush();
    assert.ok(got.every((run) => run.blocks >= minRunBlocks), "update() emitted a short run");
    assert.ok(tail.every((run) => run.blocks >= 1));
    const everything = got.concat(tail);
    assert.equal(everything.reduce((a, run) => a + run.blocks, 0), values.length);
    for (let i = 1; i < everything.length; i++) assert.notEqual(everything[i - 1].on, everything[i].on);
    assert.ok(everything.filter((run) => run.on).length >= 10); // the keying was heard
  }
});

// ------------------------------------------------------ noise, no chatter

test("a minute of tone-free noise never switches ON", () => {
  // Acceptance: 60 s of Gaussian noise at any fixed level from -110 to
  // -40 dBFS produces zero ON runs with the defaults.
  for (const dbfs of [-110.0, -90.0, -70.0, -50.0, -40.0]) {
    const values = noiseSeries(dbfs, 60.0, 7);
    assert.equal(values.length, 6000);
    const det = new ToneDetector();
    for (const v of values) {
      det.update(v);
      assert.equal(det.state, false, `${dbfs} dBFS: switched ON at ${v.toFixed(1)} dB, hi ${det.thresholdHiDb.toFixed(1)}`);
    }
    assertRuns(det.flush(), [[false, 6000]]);
    const settled = values.slice(30);
    assert.ok(det.thresholdHiDb > Math.max(...settled));
    const linearMeanDb = toDb(settled.reduce((a, v) => a + Math.pow(10.0, v / 10.0), 0) / settled.length);
    if (linearMeanDb > -98.0) {
      // above the silence clamp N is the unbiased linear mean of the noise
      approx(det.floorDb, linearMeanDb, { abs: 1.5 }, `${dbfs} dBFS floor`);
    } else {
      approx(det.floorDb, -100.0, { abs: 0.1 }, `${dbfs} dBFS floor`);
    }
  }
});

// -------------------------------------------------------- keyed tone in noise

test("a tone 20 dB above the noise gives the nine SOS marks", () => {
  for (const dbfs of [-110.0, -90.0, -70.0, -50.0, -40.0]) {
    for (const seed of [0, 1]) {
      const det = new ToneDetector();
      const runs = feed(det, sosInNoise(dbfs, seed, { snrDb: 20.0 }));
      runs.push(...det.flush());
      assert.ok(runs[0].on === false && runs[0].ms >= 900.0, `${dbfs} dBFS seed ${seed}: ON during the lead`);
      assertSosRuns(runs);
    }
  }
});

test("a reverberant tone 20 dB above the noise gives the nine SOS marks", () => {
  for (const reverbMs of [5.0, 10.0]) {
    const det = new ToneDetector();
    const runs = feed(det, sosInNoise(-60.0, 2, { snrDb: 20.0, reverbMs }));
    runs.push(...det.flush());
    assert.equal(runs.filter((r) => r.on).length, 9, `reverb ${reverbMs} ms`);
  }
});

test("silence, then noise, then the tone: no early runs", () => {
  // Acceptance: 2 s of digital silence, then noise, then the keyed tone;
  // no ON run during the silence or the noise.  Levels up to -80 dBFS put
  // the single-bin noise at or under the -100 dB silence clamp.
  for (const dbfs of [-110.0, -90.0, -80.0]) {
    const values = sosInNoise(dbfs, 0, { snrDb: 20.0, leadMs: 2000.0, silenceS: 2.0 });
    const det = new ToneDetector();
    values.slice(0, 400).forEach((v, i) => {
      det.update(v);
      assert.equal(det.state, false, `${dbfs} dBFS: ON at block ${i} (${v.toFixed(1)} dB)`);
    });
    assert.equal(det.warmingUp, false);
    const runs = feed(det, values.slice(400));
    runs.push(...det.flush());
    assert.ok(runs[0].on === false && runs[0].blocks >= 400);
    assertSosRuns(runs);
  }
});

test("loud noise after a long silence recovers within two timeouts", () => {
  // Noise 40 dB above the silence clamp arriving 2 s in cannot be told from
  // a tone at its onset; the detector must nevertheless be OFF again within
  // 2 * maxOnBlocks and then hear the tone normally.
  for (const seed of [0, 1]) {
    const values = sosInNoise(-50.0, seed, { snrDb: 20.0, leadMs: 20000.0, silenceS: 2.0 });
    const noiseStart = 200;
    const toneStart = 2200;
    const det = new ToneDetector();
    const runs = feed(det, values);
    runs.push(...det.flush());
    const starts = onStarts(runs);
    assert.ok(starts.every(([s]) => s >= noiseStart));
    const early = starts.filter(([s]) => s < toneStart);
    assert.ok(early.every(([s, b]) => s + b <= noiseStart + 2 * det.maxOnBlocks), JSON.stringify(early));
    // the ON phase may fragment into a few runs; in total it stays under two timeouts
    assert.ok(early.reduce((a, [, b]) => a + b, 0) <= 2 * det.maxOnBlocks, JSON.stringify(early));
    const onRuns = runs.filter((r) => r.on);
    const toneRuns = onRuns.filter((_, i) => starts[i][0] >= toneStart);
    assert.equal(toneRuns.length, 9);
    assertSosRuns([new Run(false, 100)].concat(toneRuns));
  }
});

// ------------------------------------------------------------ noise jumps

test("a clean 25 dB step gives exactly one timed-out run", () => {
  // The contract's statement on a clean series: one ON run of maxOnBlocks,
  // OFF from then on, and the floor adopts the new level.
  const det = new ToneDetector();
  const got = feed(det, times(-80.0, 200));
  got.push(...feed(det, times(-55.0, 2000)));
  got.push(...det.flush());
  const on = got.filter((r) => r.on);
  assert.equal(on.length, 1);
  assert.equal(on[0].blocks, det.maxOnBlocks);
  assertRuns(got, [[false, 200], [true, 500], [false, 1500]]);
  approx(det.floorDb, -55.0, { abs: 0.01 });
});

test("a 25 dB noise jump recovers and stays OFF", () => {
  // Real Goertzel noise jumping 25 dB dips below the OFF threshold now and
  // then, so the ON phase may fragment into a few runs; the total ON time
  // stays under two timeouts and nothing is ON after them.
  for (const seed of [0, 1, 2]) {
    const r = rng(seed);
    const quiet = gaussian(-70.0, 10 * FS, r);
    const loud = gaussian(-45.0, 30 * FS, r);
    const x = new Float32Array(quiet.length + loud.length);
    x.set(quiet, 0);
    x.set(loud, quiet.length);
    const values = goertzelSeries(x, 2491.0);
    const det = new ToneDetector();
    const runs = feed(det, values);
    runs.push(...det.flush());
    const starts = onStarts(runs);
    assert.ok(starts.length > 0, "the jump must at least be noticed");
    assert.ok(starts.every(([s]) => s >= 1000), JSON.stringify(starts)); // nothing before the jump
    assert.ok(Math.max(...starts.map(([s, b]) => s + b)) <= 1000 + 2 * det.maxOnBlocks, JSON.stringify(starts));
    assert.ok(starts.reduce((a, [, b]) => a + b, 0) <= 2 * det.maxOnBlocks, JSON.stringify(starts));
    const tail = values.slice(3000);
    const linearMeanDb = toDb(tail.reduce((a, v) => a + Math.pow(10.0, v / 10.0), 0) / tail.length);
    approx(det.floorDb, linearMeanDb, { abs: 3.0 }, `seed ${seed}`);
  }
});

// ------------------------------------------- recorded fixtures, defaults only

/**
 * Default detector over a series (with per-block tonal flags, as the pipeline
 * feeds them), then flush; returns `[on, blocks]` pairs like vectors.json.
 */
function detectorPairs(series, tonal = null) {
  const det = new ToneDetector();
  const runs = [];
  series.forEach((v, i) => runs.push(...det.update(v, tonal ? Boolean(tonal[i]) : true)));
  runs.push(...det.flush());
  assert.ok(runs.every((r) => r instanceof Run && r.blockMs === 10.0));
  return runs.map((r) => [r.on, r.blocks]);
}

/** `10*log10(mean square)` per block, the pipeline's level measure. */
function levelSeries(samples, block) {
  const n = Math.floor(samples.length / block);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = i * block; k < (i + 1) * block; k++) sum += samples[k] * samples[k];
    out[i] = 10 * Math.log10(sum / block + 1e-12);
  }
  return out;
}

for (const [name, fixture] of Object.entries(vectors.fixtures)) {
  test(`default detector reproduces the Python run list on ${name} (rounded series)`, () => {
    assert.equal(fixture.runs_from_rounded_power_db_identical, true);
    assert.equal(fixture.tonal.length, fixture.blocks);
    const pairs = detectorPairs(fixture.power_db, fixture.tonal);
    assert.deepEqual(pairs, fixture.runs);
    assert.equal(pairs.reduce((a, [, b]) => a + b, 0), fixture.blocks);
  });

  test(`default detector reproduces the Python run list on ${name} (JS Goertzel series)`, async () => {
    const { isTonal } = await import("../js/detector.js");
    const wav = readWav(new URL(fixture.file, ROOT));
    const series = goertzelSeries(wav.samples, fixture.f0, wav.sampleRate, fixture.block_size);
    const levels = levelSeries(wav.samples, fixture.block_size);
    assert.equal(series.length, fixture.blocks);
    const { combineDb, neighborFrequencies } = await import("../js/detector.js");
    const nbSeries = neighborFrequencies(fixture.f0, wav.sampleRate).map((f) => goertzelSeries(wav.samples, f, wav.sampleRate, fixture.block_size));
    const neighborDb = series.map((_, i) => combineDb(nbSeries.map((s) => s[i])));
    const tonal = series.map((p, i) => isTonal(p, levels[i], undefined, neighborDb[i]));
    assert.deepEqual(tonal.map(Number), fixture.tonal, "tonal flags agree with Python");
    assert.deepEqual(detectorPairs(series, tonal), fixture.runs);
  });
}

test("the loopback fixture gives nine marks with the keyed proportions", () => {
  const fixture = vectors.fixtures.loopback_sos_1khz_15wpm;
  const det = new ToneDetector();
  const got = [];
  fixture.power_db.forEach((v, i) => got.push(...det.update(v, Boolean(fixture.tonal[i]))));
  got.push(...det.flush());
  const marks = got.filter((r) => r.on).map((r) => r.ms);
  assert.equal(marks.length, 9);
  const dits = marks.slice(0, 3).concat(marks.slice(6, 9));
  const dahs = marks.slice(3, 6);
  assert.ok(dits.every((ms) => Math.abs(ms - 100.0) <= 20.0), JSON.stringify(dits));
  assert.ok(dahs.every((ms) => Math.abs(ms - 260.0) <= 20.0), JSON.stringify(dahs));
  const gaps = got.slice(1, -1).filter((r) => !r.on).map((r) => r.ms);
  assert.equal(gaps.length, 8);
  const intra = [gaps[0], gaps[1], gaps[3], gaps[4], gaps[6], gaps[7]];
  const letter = [gaps[2], gaps[5]];
  assert.ok(intra.every((ms) => Math.abs(ms - 55.0) <= 20.0), JSON.stringify(intra));
  assert.ok(letter.every((ms) => Math.abs(ms - 220.0) <= 20.0), JSON.stringify(letter));
  // no chatter during the fade-in and room noise before the message
  assert.equal(got[0].on, false);
  assert.ok(Math.abs(got[0].ms - 1370.0) <= 60.0, String(got[0]));
  assert.equal(got[got.length - 1].on, false);
});

test("the beeper fixture gives four tones with the measured gaps", () => {
  const fixture = vectors.fixtures.beeper_long_2491hz_1m;
  const det = new ToneDetector();
  const got = [];
  fixture.power_db.forEach((v, i) => got.push(...det.update(v, Boolean(fixture.tonal[i]))));
  got.push(...det.flush());
  const on = got.filter((r) => r.on).map((r) => r.ms);
  assert.equal(on.length, 4, JSON.stringify(on));
  [3700.0, 490.0, 570.0, 2020.0].forEach((want, i) => assert.ok(Math.abs(on[i] - want) <= 100.0, JSON.stringify(on)));
  const gaps = got.slice(1, -1).filter((r) => !r.on).map((r) => r.ms);
  assert.equal(gaps.length, 3, JSON.stringify(gaps));
  [370.0, 250.0, 300.0].forEach((want, i) => assert.ok(Math.abs(gaps[i] - want) <= 100.0, JSON.stringify(gaps)));
  // the first ON edge is near 0.69 s, not at the fade-in from digital silence
  assert.equal(got[0].on, false);
  assert.ok(Math.abs(got[0].ms - 690.0) <= 60.0, String(got[0]));
  assert.equal(got[got.length - 1].on, false);
});

// ---------------------------------------------------------- broadband gate

test("isTonal and the tonal flag: a click teaches the noise level but never switches ON", async () => {
  const { isTonal, tonalityDb, TONALITY_MIN_DB } = await import("../js/detector.js");
  assert.equal(TONALITY_MIN_DB, -15);
  assert.ok(Math.abs(tonalityDb(-41, -44.0103)) < 1e-9, "pure tone: 0 dB");
  assert.ok(tonalityDb(-64, -40) < -20, "white noise: about -24 dB");
  assert.equal(isTonal(-41, -44), true);
  assert.equal(isTonal(-64, -40), false);

  const det = new ToneDetector();
  for (let i = 0; i < 40; i++) det.update(-90);
  assert.equal(det.state, false);
  const floorBefore = det.floorDb;
  for (let i = 0; i < 5; i++) det.update(-40, false); // loud click blocks, not tonal
  assert.equal(det.state, false, "never ON");
  assert.ok(det.floorDb > floorBefore + 20, "but the noise level followed them");
  // a tonal block of the same power does switch ON once the level has settled again
  for (let i = 0; i < 200; i++) det.update(-90);
  assert.equal(det.state, false);
  det.update(-40, true);
  assert.equal(det.state, true);
  // while ON the flag is ignored: a click during a mark does not chop it
  det.update(-40, false);
  assert.equal(det.state, true);
});

test("neighbour-band filter: helpers and rule mirror the Python module", async () => {
  const { NEIGHBOR_OFFSETS_HZ, PEAKINESS_MIN_DB, combineDb, isTonal: tonal, neighborFrequencies } = await import("../js/detector.js");
  assert.deepEqual([...NEIGHBOR_OFFSETS_HZ], [-600, -300, 300, 600]);
  assert.equal(PEAKINESS_MIN_DB, 8);
  assert.deepEqual(neighborFrequencies(2491, 48000), [1891, 2191, 2791, 3091]);
  assert.deepEqual(neighborFrequencies(400, 48000), [100, 700, 1000]);
  assert.ok(Math.abs(combineDb([-10, -10]) + 10) < 1e-9);
  assert.equal(combineDb([]), -Infinity);
  assert.equal(tonal(-10, -13), true);
  assert.equal(tonal(-10, -13, undefined, -40), true);
  assert.equal(tonal(-10, -13, undefined, -15), false);
  assert.equal(tonal(-40, -13, undefined, -90), false);
});
