/**
 * Tests for web/js/dsp.js: the Goertzel port, mirrored from tests/test_dsp.py,
 * plus parity of the per-block dB series against the Python reference on both
 * recorded fixtures (web/test/vectors.json).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { DB_FLOOR, Goertzel, goertzelPower } from "../js/dsp.js";
import { readWav } from "./wav.mjs";

const FS = 48000;
const N = 480; // 10 ms
const BEEPER = 2491.0; // measured PC-speaker tone, off-bin for N = 480
const ROOT = new URL("../../", import.meta.url);
const vectors = JSON.parse(readFileSync(new URL("./vectors.json", import.meta.url), "utf8"));

// ------------------------------------------------------------------ helpers

/**
 * A float32 sine, like the Python test helper.
 * @param {number} f
 * @param {{n?: number, fs?: number, amp?: number, phase?: number}} [opts]
 * @returns {Float32Array}
 */
function sine(f, { n = N, fs = FS, amp = 1.0, phase = 0.0 } = {}) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * f * i) / fs + phase);
  return out;
}

/** @param {Float32Array} a @param {Float32Array} b @returns {Float32Array} */
function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/** Deterministic pseudo-random Gaussian-ish samples (sum of uniforms), seeded. */
function noise(n, sigma, seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = 0; k < 12; k++) acc += next();
    out[i] = sigma * (acc - 6); // variance 12 * 1/12 = 1
  }
  return out;
}

const db = (power) => 10 * Math.log10(power + 1e-30);

/**
 * pytest.approx semantics: |actual - expected| <= max(rel * |expected|, abs).
 */
function approx(actual, expected, { rel = 1e-6, abs = 1e-12 } = {}, message = "") {
  const tol = Math.max(rel * Math.abs(expected), abs);
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${message ? message + ": " : ""}${actual} is not within ${tol} of ${expected}`,
  );
}

/**
 * Reference: the single DTFT bin at f0, 4*|X(f0)|**2/N**2, by direct summation.
 * @param {ArrayLike<number>} x
 * @param {number} f0
 * @param {number} [fs]
 */
function dtftPower(x, f0, fs = FS) {
  const n = x.length;
  let re = 0;
  let im = 0;
  for (let i = 0; i < n; i++) {
    const w = (-2 * Math.PI * f0 * i) / fs;
    re += x[i] * Math.cos(w);
    im += x[i] * Math.sin(w);
  }
  return (4 * (re * re + im * im)) / (n * n);
}

// ------------------------------------------------------------ goertzelPower

test("full-scale sine on a DFT bin gives exactly 1.0", () => {
  // 1000 Hz is bin 10 of a 480-sample block at 48 kHz: the image term vanishes.
  for (const phase of [0.0, 0.7, 1.9, 3.1]) {
    approx(goertzelPower(sine(1000.0, { phase }), 1000.0, FS), 1.0, { rel: 1e-6 }, `phase ${phase}`);
  }
});

test("full-scale sine off-bin gives close to 1.0", () => {
  for (const phase of [0.0, 0.7, 1.9, 3.1]) {
    approx(goertzelPower(sine(BEEPER, { phase }), BEEPER, FS), 1.0, { abs: 0.05 }, `phase ${phase}`);
  }
});

test("power scales with amplitude squared", () => {
  approx(goertzelPower(sine(1000.0, { amp: 0.3 }), 1000.0, FS), 0.09, { rel: 1e-6 });
  approx(goertzelPower(sine(1000.0, { amp: 0.01 }), 1000.0, FS), 1e-4, { rel: 1e-6 });
});

test("normalisation is independent of block length", () => {
  for (const n of [240, 480, 960, 2048, 4800]) {
    approx(goertzelPower(sine(1000.0, { n }), 1000.0, FS), 1.0, { abs: 0.05 }, `n=${n} on-bin`);
    approx(goertzelPower(sine(BEEPER, { n }), BEEPER, FS), 1.0, { abs: 0.05 }, `n=${n} off-bin`);
  }
});

test("a tone 500 Hz away is rejected by at least 40 dB", () => {
  for (const f0 of [1000.0, BEEPER]) {
    for (const offset of [500.0, -500.0]) {
      const pOn = goertzelPower(sine(f0), f0, FS);
      const pOff = goertzelPower(sine(f0 + offset), f0, FS);
      assert.ok(db(pOn) - db(pOff) >= 40.0, `f0=${f0} offset=${offset}: ${db(pOn) - db(pOff)} dB`);
    }
  }
});

test("silence gives zero power", () => {
  const p = goertzelPower(new Float32Array(N), 1000.0, FS);
  assert.equal(p, 0.0);
  assert.ok(10 * Math.log10(p + DB_FLOOR) <= -90.0);
});

test("plain arrays and Float64Array give the same answer as Float32Array", () => {
  const ref = goertzelPower(sine(1000.0), 1000.0, FS);
  approx(goertzelPower(Array.from(sine(1000.0)), 1000.0, FS), ref, { rel: 1e-12 });
  approx(goertzelPower(Float64Array.from(sine(1000.0)), 1000.0, FS), ref, { rel: 1e-12 });
});

test("degenerate blocks return zero", () => {
  assert.equal(goertzelPower(new Float32Array(0), 1000.0, FS), 0.0);
  assert.equal(goertzelPower(Float32Array.of(1.0), 1000.0, FS), 0.0);
  assert.equal(goertzelPower([], 1000.0, FS), 0.0);
});

test("goertzelPower is exactly the single DTFT bin at f0", () => {
  // Pins the design: one bin at f0, no neighbouring bins summed in.  Any
  // contribution from f0 +- 50 Hz would show up as a mismatch here.
  for (const f0 of [1000.0, BEEPER, 2510.0]) {
    for (const n of [240, 480, 960, 2048]) {
      const blocks = [
        sine(f0, { n, phase: 0.7 }),
        sine(f0 + 37.0, { n }),
        add(sine(1000.0, { n }), sine(BEEPER, { n, amp: 0.5 })),
        noise(n, 0.1, 5),
      ];
      for (const b of blocks) {
        const want = dtftPower(b, f0);
        approx(goertzelPower(b, f0, FS), want, { rel: 1e-9, abs: 1e-15 }, `f0=${f0} n=${n}`);
        approx(new Goertzel(f0, FS, n).power(b), want, { rel: 1e-9, abs: 1e-15 }, `class f0=${f0} n=${n}`);
      }
    }
  }
});

test("the neighbour-bin sum would break the 0 dB normalisation (why it is not implemented)", () => {
  for (const f0 of [1000.0, BEEPER]) {
    const threeBin = (n) => {
      const x = sine(f0, { n, phase: 0.7 });
      return [-50.0, 0.0, 50.0].reduce((acc, d) => acc + goertzelPower(x, f0 + d, FS), 0);
    };
    for (const n of [240, 480, 960]) {
      approx(goertzelPower(sine(f0, { n, phase: 0.7 }), f0, FS), 1.0, { abs: 0.05 });
    }
    assert.ok(db(threeBin(240)) > 4.0); // about +4.2 dB
    assert.ok(db(threeBin(480)) > 2.4); // about +2.6 dB
    approx(db(threeBin(960)), 0.0, { abs: 0.1 });
  }
});

// ----------------------------------------------------------- Goertzel class

test("Goertzel class matches the function", () => {
  const g = new Goertzel(BEEPER, FS, N);
  const blocks = [sine(BEEPER), sine(1000.0), sine(BEEPER, { amp: 0.2, phase: 1.0 })];
  for (let i = 0; i < 5; i++) blocks.push(noise(N, 0.1, 42 + i));
  for (const b of blocks) {
    approx(g.power(b), goertzelPower(b, BEEPER, FS), { rel: 1e-9, abs: 1e-15 });
  }
});

test("Goertzel.powerDb is 10*log10(power + 1e-12)", () => {
  const g = new Goertzel(1000.0, FS, N);
  approx(g.powerDb(sine(1000.0)), 0.0, { abs: 1e-6 });
  approx(g.powerDb(sine(1000.0, { amp: 0.1 })), -20.0, { abs: 1e-6 });
  approx(g.powerDb(new Float32Array(N)), -120.0, { abs: 1e-9 });
  assert.equal(DB_FLOOR, 1e-12);
  assert.equal(DB_FLOOR, vectors.params.detector_constants.db_floor);
});

test("setFrequency retunes and f0 is read-only", () => {
  const g = new Goertzel(1000.0, FS, N);
  assert.equal(g.f0, 1000.0);
  const strongAt1k = g.power(sine(1000.0));
  g.setFrequency(BEEPER);
  assert.equal(g.f0, BEEPER);
  assert.ok(g.power(sine(1000.0)) < 1e-4 * strongAt1k);
  approx(g.power(sine(BEEPER)), 1.0, { abs: 0.05 });
  assert.throws(() => {
    g.f0 = 500.0;
  }, TypeError);
  assert.equal(g.f0, BEEPER);
});

test("Goertzel class handles other block lengths", () => {
  const g = new Goertzel(1000.0, FS, N);
  approx(g.power(sine(1000.0, { n: 960 })), 1.0, { abs: 0.05 });
  approx(g.power(sine(1000.0, { n: 200 })), goertzelPower(sine(1000.0, { n: 200 }), 1000.0, FS), { rel: 1e-12 });
  assert.equal(g.power(new Float32Array(0)), 0.0);
  assert.equal(g.power(Float32Array.of(0.5)), 0.0);
});

test("Goertzel constructor validates fs and blockSize", () => {
  assert.throws(() => new Goertzel(1000.0, 0, N), RangeError);
  assert.throws(() => new Goertzel(1000.0, -48000, N), RangeError);
  assert.throws(() => new Goertzel(1000.0, FS, 1), RangeError);
  assert.throws(() => new Goertzel(1000.0, FS, 0), RangeError);
  assert.throws(() => new Goertzel(1000.0, NaN, N), RangeError);
  const g = new Goertzel(1000.0, FS, N);
  assert.equal(g.fs, FS);
  assert.equal(g.blockSize, N);
});

// ------------------------------------------------- parity with the Python side

for (const [name, fixture] of Object.entries(vectors.fixtures)) {
  test(`Goertzel dB series matches Python within 0.05 dB on ${name}`, (t) => {
    const wav = readWav(new URL(fixture.file, ROOT));
    assert.equal(wav.sampleRate, fixture.fs);
    assert.equal(wav.channels, 1);
    assert.equal(wav.bitsPerSample, 16);
    assert.equal(wav.samples.length, fixture.samples);
    assert.equal(fixture.blocks * fixture.block_size, fixture.samples, "fixture is whole blocks");
    assert.equal(fixture.power_db.length, fixture.blocks);

    const g = new Goertzel(fixture.f0, wav.sampleRate, fixture.block_size);
    const tolerance = 0.05 + 1e-9; // the series is rounded to 0.1 dB; the guard covers float round-off
    let maxDiff = 0;
    let worst = -1;
    for (let i = 0; i < fixture.blocks; i++) {
      const block = wav.samples.subarray(i * fixture.block_size, (i + 1) * fixture.block_size);
      const got = g.powerDb(block);
      const want = fixture.power_db[i];
      const diff = Math.abs(got - want);
      if (diff > maxDiff) {
        maxDiff = diff;
        worst = i;
      }
      assert.ok(diff <= tolerance, `block ${i}: JS ${got} vs Python ${want} (diff ${diff} dB)`);
      // The function and the class measure the same quantity.
      approx(10 * Math.log10(goertzelPower(block, fixture.f0, wav.sampleRate) + DB_FLOOR), got, { abs: 1e-9 });
    }
    t.diagnostic(`${name}: ${fixture.blocks} blocks, largest deviation ${maxDiff.toFixed(5)} dB at block ${worst}`);
  });
}

test("the recorded fixtures span silence, room noise and tone", () => {
  // Sanity of the WAV reader on real data: digital silence at the start,
  // the tone well above the floor later on.
  const wav = readWav(new URL("tests/fixtures/loopback_sos_1khz_15wpm.wav", ROOT));
  const g = new Goertzel(1000.0, FS, N);
  const first = g.powerDb(wav.samples.subarray(0, N));
  assert.ok(first < -110, `first block ${first} dB`);
  let peak = -Infinity;
  for (let i = 0; i + N <= wav.samples.length; i += N) peak = Math.max(peak, g.powerDb(wav.samples.subarray(i, i + N)));
  assert.ok(peak > -40, `peak ${peak} dB`);
  let maxAbs = 0;
  for (const v of wav.samples) maxAbs = Math.max(maxAbs, Math.abs(v));
  assert.ok(maxAbs <= 1.0 && maxAbs > 0.001);
});
