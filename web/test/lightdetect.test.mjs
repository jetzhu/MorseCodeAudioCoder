// Synthetic-frame tests for web/js/lightdetect.js: a camera watching a
// flashing screen, simulated frame by frame, decoded by the real MorseDecoder.
import { test } from "node:test";
import assert from "node:assert/strict";

import { LightDetector, SourceArbiter } from "../js/lightdetect.js";
import { MorseDecoder } from "../js/decoder.js";
import { buildTiming } from "../js/player.js";
import { timingStateAt } from "../js/channels.js";

/** Deterministic pseudo-random numbers (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Frames of a camera watching `text` keyed as light.
 * Each frame integrates the light over its exposure (so an edge inside a
 * frame gives an intermediate value), adds sensor noise, a slow exposure
 * drift, and frame-interval jitter.
 */
function frames(text, wpm, { fps = 30, dark = 40, bright = 200, noise = 4, jitterMs = 3, driftAmp = 0, lead = 1500, tail = 3000, seed = 1 } = {}) {
  const timing = buildTiming(text, wpm);
  const total = timing.reduce((s, x) => s + x.ms, 0);
  const r = rng(seed);
  const gauss = () => (r() + r() + r() + r() - 2) * 1.2;
  const out = [];
  const period = 1000 / fps;
  const exposure = period * 0.8;
  let t = 0;
  while (t < lead + total + tail) {
    let lit = 0;
    const n = 8;
    for (let k = 0; k < n; k++) lit += timingStateAt(timing, t - exposure + (k + 0.5) * (exposure / n) - lead) ? 1 : 0;
    const level = dark + (bright - dark) * (lit / n) + driftAmp * Math.sin(t / 2500) + noise * gauss();
    out.push({ t, value: Math.max(0, Math.min(255, level)) });
    t += period + (r() - 0.5) * 2 * jitterMs;
  }
  return out;
}

function decode(fr, { wpm } = {}) {
  const det = new LightDetector();
  const dec = new MorseDecoder(wpm ? { wpm, adaptive: false } : undefined);
  let text = "";
  for (const f of fr) {
    for (const run of det.update(f.value, f.t)) text += dec.feed(run);
    const off = det.currentOffMs(f.t);
    if (off > 0) text += dec.idle(off);
  }
  const last = fr[fr.length - 1].t;
  for (const run of det.flush(last)) text += dec.feed(run);
  text += dec.idle(Infinity);
  return { text: text.trim(), det };
}

for (const wpm of [5, 8, 12]) {
  test(`a flashing screen at ${wpm} WPM, 30 fps, decodes`, () => {
    const { text, det } = decode(frames("HELLO WORLD", wpm, { seed: wpm }));
    assert.equal(text, "HELLO WORLD");
    assert.ok(Math.abs(det.fps - 30) < 2, `fps ${det.fps}`);
  });
}

test("low contrast, heavier noise and a slow exposure drift still decode at 8 WPM", () => {
  const { text } = decode(frames("PARIS 73", 8, { dark: 90, bright: 135, noise: 3, driftAmp: 6, seed: 7 }));
  assert.equal(text, "PARIS 73");
});

test("a 60 fps camera reaches 16 WPM", () => {
  const { text } = decode(frames("CQ CQ DE K1ABC", 16, { fps: 60, seed: 3 }));
  assert.equal(text, "CQ CQ DE K1ABC");
});

test("a steady or noisy scene never produces marks", () => {
  const det = new LightDetector();
  const r = rng(5);
  let runs = 0;
  for (let i = 0; i < 900; i++) runs += det.update(120 + (r() - 0.5) * 10, i * 33.3).length;
  assert.equal(runs, 0);
  assert.equal(det.state, false);
});

test("a single-frame flash (a reflection) is not a mark", () => {
  const det = new LightDetector();
  let on = 0;
  for (let i = 0; i < 300; i++) {
    const v = i === 150 ? 250 : 40;
    det.update(v, i * 33.3);
    if (det.state) on += 1;
  }
  assert.equal(on, 0);
});

test("edges are placed where the brightness crossed, not at the frame", () => {
  const det = new LightDetector();
  let t = 0;
  const period = 33.3;
  for (let i = 0; i < 60; i++, t += period) det.update(40, t); // 2 s of dark
  for (let i = 0; i < 10; i++, t += period) det.update(200, t); // bright
  // A frame half-way through the fall: 120 is the mid level, so the edge
  // lands on this frame exactly; then dark again.
  det.update(120, t);
  t += period;
  const runs = det.update(40, t);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].on, true);
  const expected = 10 * period; // from the first bright frame's crossing to the mid frame
  assert.ok(Math.abs(runs[0].ms - expected) <= 20, `mark ${runs[0].ms} ms vs ${expected}`);
});

test("a light left on fades out of the window instead of sticking", () => {
  const det = new LightDetector({ windowMs: 3000 });
  let t = 0;
  for (let i = 0; i < 60; i++, t += 33) det.update(40, t);
  for (let i = 0; i < 40; i++, t += 33) det.update(200, t);
  assert.equal(det.state, true);
  for (let i = 0; i < 200; i++, t += 33) det.update(200, t);
  assert.equal(det.state, false, "contrast collapses once the dark frames leave the window");
});

test("SourceArbiter: the first source to start a mark owns the message", () => {
  const a = new SourceArbiter();
  const off = { on: false };
  const on = { on: true };
  assert.equal(a.mayIdle("mic"), true);
  assert.equal(a.offer("camera", off), true, "camera starts a mark first");
  assert.equal(a.owner, "camera");
  assert.equal(a.offer("mic", off), false, "the microphone's copy is ignored");
  assert.equal(a.offer("mic", on), false);
  assert.equal(a.offer("camera", on), true);
  assert.equal(a.mayIdle("mic"), false);
  a.tick("mic", false, 5000, 2000);
  assert.equal(a.owner, "camera", "only the owner's silence releases it");
  a.tick("camera", false, 1500, 2000);
  assert.equal(a.owner, "camera");
  a.tick("camera", true, 9999, 2000);
  assert.equal(a.owner, "camera", "never released while ON");
  a.tick("camera", false, 2500, 2000);
  assert.equal(a.owner, null);
  assert.equal(a.offer("mic", on), false, "an ON run cannot claim: it belongs to a mark already missed");
  assert.equal(a.offer("mic", off), true);
  a.release("camera");
  assert.equal(a.owner, "mic");
  a.release("mic");
  assert.equal(a.owner, null);
});

test("two sources with the same message, 180 ms apart, decode once", () => {
  const cam = new LightDetector();
  const camFrames = frames("SOS", 8, { seed: 11 });
  // The microphone path is simulated with a second light detector on a
  // delayed copy; the arbiter must keep exactly one copy.
  const late = new LightDetector();
  const lateFrames = camFrames.map((f) => ({ t: f.t + 180, value: f.value }));
  const arb = new SourceArbiter();
  const dec = new MorseDecoder();
  let text = "";
  const events = [...camFrames.map((f) => ["camera", cam, f]), ...lateFrames.map((f) => ["mic", late, f])].sort((x, y) => x[2].t - y[2].t);
  for (const [src, det, f] of events) {
    for (const run of det.update(f.value, f.t)) if (arb.offer(src, run)) text += dec.feed(run);
    const off = det.currentOffMs(f.t);
    arb.tick(src, det.state, off, Math.max(2000, 8 * dec.ditMs));
    if (off > 0 && arb.mayIdle(src)) text += dec.idle(off);
  }
  text += dec.idle(Infinity);
  assert.equal(text.trim(), "SOS");
});

test("robust across noise patterns: 20 seeds at 5, 8 and 12 WPM", () => {
  const failures = [];
  for (const wpm of [5, 8, 12]) {
    for (let seed = 1; seed <= 20; seed++) {
      for (const msg of ["SOS", "HELLO WORLD"]) {
        const { text } = decode(frames(msg, wpm, { seed: seed * 31 + wpm }));
        if (text !== msg) failures.push(`${wpm} WPM seed ${seed} ${msg}: ${JSON.stringify(text)}`);
      }
    }
  }
  assert.deepEqual(failures, []);
});

test("frameTime prefers the capture time over the callback time", async () => {
  const { frameTime } = await import("../js/camera.js");
  assert.equal(frameTime(500, { captureTime: 480, expectedDisplayTime: 510 }), 480);
  assert.equal(frameTime(500, { expectedDisplayTime: 510 }), 510);
  assert.equal(frameTime(500, { captureTime: 0 }), 500);
  assert.equal(frameTime(500, undefined), 500);
});
