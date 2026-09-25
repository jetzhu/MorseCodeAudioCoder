// Tests for web/js/player.js: buildTiming (mirrors tests/test_beep_sender.py)
// and TonePlayer scheduling against a fake AudioContext, so the Web Audio
// wiring is exercised in Node without a browser.
import { test } from "node:test";
import assert from "node:assert/strict";

import { TonePlayer, buildGuide, buildTiming, ditMs, farnsworthGaps, layoutGuideLabels, roundHalfEven, timingDurationMs } from "../js/player.js";

const seg = (on, ms) => ({ on, ms });

// SOS at 10 WPM: dit 120, dah 360, letter gap 360.
const SOS_10WPM = [
  seg(true, 120), seg(false, 120), seg(true, 120), seg(false, 120), seg(true, 120),
  seg(false, 360),
  seg(true, 360), seg(false, 120), seg(true, 360), seg(false, 120), seg(true, 360),
  seg(false, 360),
  seg(true, 120), seg(false, 120), seg(true, 120), seg(false, 120), seg(true, 120),
];

function assertClose(actual, expected, tol = 1e-9, message = "") {
  assert.ok(Math.abs(actual - expected) <= tol, `${message} expected ${expected} +- ${tol}, got ${actual}`);
}

// -- buildTiming --------------------------------------------------------------

test("ditMs is 1200 / wpm and rejects non-positive speeds", () => {
  assert.equal(ditMs(10), 120);
  assert.equal(ditMs(15), 80);
  for (const bad of [0, -1, NaN, undefined]) assert.throws(() => ditMs(bad), RangeError);
});

test("SOS at 10 WPM exact", () => {
  assert.deepEqual(buildTiming("SOS", 10), SOS_10WPM);
  assert.equal(timingDurationMs(SOS_10WPM), 3240);
});

test("word gap is seven dits", () => {
  assert.deepEqual(buildTiming("E E", 10), [seg(true, 120), seg(false, 840), seg(true, 120)]);
});

test("multiple spaces and surrounding whitespace make one word gap", () => {
  assert.deepEqual(buildTiming("  E \t\n  E  ", 10), [seg(true, 120), seg(false, 840), seg(true, 120)]);
});

test("two words share one word gap", () => {
  const timing = buildTiming("SOS SOS", 10);
  assert.equal(timing.filter((s) => !s.on && s.ms === 840).length, 1);
  assert.equal(timing.length, 2 * SOS_10WPM.length + 1);
  assert.deepEqual(timing.slice(0, SOS_10WPM.length), SOS_10WPM);
  assert.deepEqual(timing.slice(SOS_10WPM.length + 1), SOS_10WPM);
});

test("case-insensitive and unknown characters skipped", () => {
  assert.deepEqual(buildTiming("sos", 10), SOS_10WPM);
  assert.deepEqual(buildTiming("S#O~S", 10), SOS_10WPM);
  assert.deepEqual(buildTiming("##", 10), []);
  assert.deepEqual(buildTiming("", 10), []);
  // an all-unknown word does not produce a gap either
  assert.deepEqual(buildTiming("E ## E", 10), [seg(true, 120), seg(false, 840), seg(true, 120)]);
});

test("no leading, trailing or adjacent gaps; marks always separated", () => {
  const timing = buildTiming("HELLO WORLD 73", 12);
  assert.ok(timing.length > 0);
  assert.equal(timing[0].on, true);
  assert.equal(timing[timing.length - 1].on, true);
  for (let i = 1; i < timing.length; i++) {
    assert.notEqual(timing[i].on, timing[i - 1].on, `segments ${i - 1} and ${i} have the same state`);
  }
  for (const s of timing) assert.ok(s.ms > 0);
});

test("roundHalfEven is Python's round()", () => {
  assert.equal(roundHalfEven(0.5), 0);
  assert.equal(roundHalfEven(1.5), 2);
  assert.equal(roundHalfEven(2.5), 2);
  assert.equal(roundHalfEven(112.5), 112); // Math.round gives 113
  assert.equal(roundHalfEven(2.51), 3);
  assert.equal(roundHalfEven(2.49), 2);
  assert.equal(roundHalfEven(171.42857142857142), 171);
  assert.equal(roundHalfEven(514.2857142857143), 514);
  assert.equal(roundHalfEven(120), 120);
});

test("dit length scales with WPM and is rounded to whole ms like morse/player.py", () => {
  assert.deepEqual(buildTiming("E", 20), [seg(true, 60)]);
  assert.deepEqual(buildTiming("E", 8), [seg(true, 150)]);
  assert.deepEqual(buildTiming("T", 8), [seg(true, 450)]);
  assert.deepEqual(buildTiming("E", 7), [seg(true, 171)]); // 171.43 rounded
  assert.deepEqual(buildTiming("E E", 7), [seg(true, 171), seg(false, 1200), seg(true, 171)]);
  assert.deepEqual(buildTiming("E", 12.5), [seg(true, 96)]);
  // Each duration is rounded from the unrounded dit: round(3 * 171.43) = 514, not 3 * 171 = 513.
  assert.deepEqual(buildTiming("T", 7), [seg(true, 514)]);
  // 32 WPM: dit 37.5 -> 38, dah 112.5 -> 112 and word gap 262.5 -> 262 (ties to even, as Python).
  assert.deepEqual(buildTiming("E T", 32), [seg(true, 38), seg(false, 262), seg(true, 112)]);
  for (const s of buildTiming("HELLO WORLD 73", 7)) assert.equal(s.ms, Math.trunc(s.ms), `${s.ms} is whole`);
});

// Generated with morse.player.build_timing (the Python reference) for the
// same inputs; [on, ms] pairs. Regenerate from Python if the rule ever changes.
const PYTHON_BUILD_TIMING = [
  ["HELLO WORLD", 7, [[true, 171], [false, 171], [true, 171], [false, 171], [true, 171], [false, 171], [true, 171], [false, 514], [true, 171], [false, 514], [true, 171], [false, 171], [true, 514], [false, 171], [true, 171], [false, 171], [true, 171], [false, 514], [true, 171], [false, 171], [true, 514], [false, 171], [true, 171], [false, 171], [true, 171], [false, 514], [true, 514], [false, 171], [true, 514], [false, 171], [true, 514], [false, 1200], [true, 171], [false, 171], [true, 514], [false, 171], [true, 514], [false, 514], [true, 514], [false, 171], [true, 514], [false, 171], [true, 514], [false, 514], [true, 171], [false, 171], [true, 514], [false, 171], [true, 171], [false, 514], [true, 171], [false, 171], [true, 514], [false, 171], [true, 171], [false, 171], [true, 171], [false, 514], [true, 514], [false, 171], [true, 171], [false, 171], [true, 171]]],
  ["PARIS", 32, [[true, 38], [false, 38], [true, 112], [false, 38], [true, 112], [false, 38], [true, 38], [false, 112], [true, 38], [false, 38], [true, 112], [false, 112], [true, 38], [false, 38], [true, 112], [false, 38], [true, 38], [false, 112], [true, 38], [false, 38], [true, 38], [false, 112], [true, 38], [false, 38], [true, 38], [false, 38], [true, 38]]],
  ["S#O~S 73", 12.5, [[true, 96], [false, 96], [true, 96], [false, 96], [true, 96], [false, 288], [true, 288], [false, 96], [true, 288], [false, 96], [true, 288], [false, 288], [true, 96], [false, 96], [true, 96], [false, 96], [true, 96], [false, 672], [true, 288], [false, 96], [true, 288], [false, 96], [true, 96], [false, 96], [true, 96], [false, 96], [true, 96], [false, 288], [true, 96], [false, 96], [true, 96], [false, 96], [true, 96], [false, 96], [true, 288], [false, 96], [true, 288]]],
  ["E E", 33, [[true, 36], [false, 255], [true, 36]]],
];

test("buildTiming matches morse.player.build_timing on the Python reference outputs", () => {
  for (const [text, wpm, expected] of PYTHON_BUILD_TIMING) {
    const ours = buildTiming(text, wpm).map((s) => [s.on, s.ms]);
    assert.deepEqual(ours, expected, `${JSON.stringify(text)} at ${wpm} WPM`);
  }
});

test("buildTiming rejects non-positive speeds", () => {
  assert.throws(() => buildTiming("SOS", 0), RangeError);
  assert.throws(() => buildTiming("SOS", -3), RangeError);
});

test("buildGuide places letters on the timeline", () => {
  const { timing, letters, totalMs } = buildGuide("SOS", 10);
  assert.deepEqual(timing, SOS_10WPM);
  assert.equal(totalMs, 3240);
  assert.deepEqual(letters, [
    { ch: "S", code: "...", startMs: 0, endMs: 600 },
    { ch: "O", code: "---", startMs: 960, endMs: 2280 },
    { ch: "S", code: "...", startMs: 2640, endMs: 3240 },
  ]);
  const ee = buildGuide("e ~ E", 10);
  assert.deepEqual(ee.letters, [
    { ch: "E", code: ".", startMs: 0, endMs: 120 },
    { ch: "E", code: ".", startMs: 960, endMs: 1080 },
  ]);
  assert.equal(ee.totalMs, 1080);
  assert.deepEqual(buildGuide("", 10), { timing: [], letters: [], totalMs: 0 });
});

// -- TonePlayer with a fake AudioContext --------------------------------------

class FakeParam {
  constructor(value) {
    this.value = value;
    this.events = [];
  }
  setValueAtTime(value, time) {
    this.events.push(["set", value, time]);
  }
  linearRampToValueAtTime(value, time) {
    this.events.push(["ramp", value, time]);
  }
}

class FakeNode {
  constructor() {
    this.connectedTo = [];
    this.disconnected = false;
  }
  connect(node) {
    this.connectedTo.push(node);
    return node;
  }
  disconnect() {
    this.disconnected = true;
  }
}

class FakeOscillator extends FakeNode {
  constructor() {
    super();
    this.type = "square";
    this.frequency = new FakeParam(440);
    this.startedAt = null;
    this.stopCalls = [];
    this.onended = null;
  }
  start(when) {
    this.startedAt = when;
  }
  stop(when) {
    this.stopCalls.push(when);
  }
}

class FakeGain extends FakeNode {
  constructor() {
    super();
    this.gain = new FakeParam(1);
  }
}

class FakeAudioContext {
  constructor({ state = "suspended", sampleRate = 48000 } = {}) {
    this.currentTime = 1.0;
    this.sampleRate = sampleRate;
    this.state = state;
    this.destination = { name: "destination" };
    this.resumeCalls = 0;
    this.oscillators = [];
    this.gains = [];
  }
  resume() {
    this.resumeCalls += 1;
    this.state = "running";
    return Promise.resolve();
  }
  createOscillator() {
    const osc = new FakeOscillator();
    this.oscillators.push(osc);
    return osc;
  }
  createGain() {
    const gain = new FakeGain();
    this.gains.push(gain);
    return gain;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test("module imports in Node and a player can be constructed without Web Audio", () => {
  const player = new TonePlayer(null);
  assert.equal(player.playing, false);
  assert.equal(player.elapsedMs, 0);
  assert.throws(() => player.play(SOS_10WPM, 1000), /AudioContext/);
  player.stop(); // no-op
  assert.equal(player.playing, false);
});

test("play schedules a sine oscillator with 3 ms ramps at every edge", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  try {
    const total = player.play(buildTiming("E", 10), 2491);
    assert.equal(total, 120);
    assert.equal(player.playing, true);
    assert.equal(player.durationMs, 120);
    assertClose(player.startTime, 1.08);
    assert.equal(ac.resumeCalls, 1, "a suspended context is resumed");

    assert.equal(ac.oscillators.length, 1);
    const osc = ac.oscillators[0];
    assert.equal(osc.type, "sine");
    assert.equal(osc.frequency.value, 2491);
    assertClose(osc.startedAt, 1.08);
    assert.equal(osc.stopCalls.length, 1);
    assertClose(osc.stopCalls[0], 1.08 + 0.12 + 0.1);

    const gain = ac.gains[0];
    const speakers = ac.gains[1]; // the speaker branch, muted by setSpeakers(false)
    assert.deepEqual(osc.connectedTo, [gain]);
    assert.deepEqual(gain.connectedTo, [speakers]);
    assert.deepEqual(speakers.connectedTo, [ac.destination]);
    assert.equal(speakers.gain.value, 1);

    const ev = gain.gain.events;
    assert.equal(ev.length, 5);
    assert.deepEqual(ev[0], ["set", 0, 1.0]);
    const expected = [
      ["set", 0, 1.08],
      ["ramp", 0.15, 1.083],
      ["set", 0.15, 1.197],
      ["ramp", 0, 1.2],
    ];
    expected.forEach(([kind, value, time], i) => {
      const [k, v, t] = ev[i + 1];
      assert.equal(k, kind, `event ${i + 1}`);
      assertClose(v, value, 1e-12, `event ${i + 1} value`);
      assertClose(t, time, 1e-9, `event ${i + 1} time`);
    });
  } finally {
    player.stop();
  }
});

test("gaps produce no automation events and marks are placed end to end", () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  try {
    player.play(buildTiming("I", 10), 1000, 0.3); // dit, gap, dit
    assert.equal(ac.resumeCalls, 0);
    const ev = ac.gains[0].gain.events.slice(1);
    assert.equal(ev.length, 8, "two marks, four events each");
    const times = ev.map((e) => e[2]);
    const t0 = 1.08;
    const want = [t0, t0 + 0.003, t0 + 0.117, t0 + 0.12, t0 + 0.24, t0 + 0.243, t0 + 0.357, t0 + 0.36];
    want.forEach((t, i) => assertClose(times[i], t, 1e-9, `event ${i}`));
    assert.equal(ev[1][1], 0.3, "the requested gain");
  } finally {
    player.stop();
  }
});

test("ramps shrink to half the mark on very short marks", () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  try {
    player.play([seg(true, 4)], 1000);
    const ev = ac.gains[0].gain.events.slice(1);
    assertClose(ev[1][2], 1.08 + 0.002);
    assertClose(ev[2][2], 1.08 + 0.002);
    assertClose(ev[3][2], 1.08 + 0.004);
  } finally {
    player.stop();
  }
});

test("stop cuts the oscillator, fires onEnd once, and is a no-op afterwards", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  let ended = 0;
  player.onEnd = () => {
    ended += 1;
  };
  player.play(SOS_10WPM, 1000);
  const osc = ac.oscillators[0];
  assert.equal(player.playing, true);
  player.stop();
  assert.equal(player.playing, false);
  assert.equal(ended, 1);
  assert.equal(osc.stopCalls.length, 2, "scheduled stop plus the immediate one");
  assert.equal(osc.stopCalls[1], undefined);
  assert.ok(osc.disconnected);
  assert.ok(ac.gains[0].disconnected);
  assert.equal(osc.onended, null);
  player.stop();
  assert.equal(ended, 1);
  assert.equal(player.elapsedMs, 0);
});

test("play while playing restarts silently (no onEnd)", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  let ended = 0;
  player.onEnd = () => {
    ended += 1;
  };
  try {
    player.play(SOS_10WPM, 1000);
    const first = ac.oscillators[0];
    ac.currentTime = 2.0;
    player.play(buildTiming("E", 10), 1000);
    assert.equal(ended, 0);
    assert.equal(first.stopCalls.length, 2);
    assert.ok(first.disconnected);
    assert.equal(ac.oscillators.length, 2);
    assertClose(player.startTime, 2.08);
    assert.equal(player.durationMs, 120);
    assert.equal(player.playing, true);
  } finally {
    player.stop();
  }
  assert.equal(ended, 1);
});

test("empty timing is a no-op", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  let ended = 0;
  player.onEnd = () => {
    ended += 1;
  };
  assert.equal(player.play([], 1000), 0);
  assert.equal(player.playing, false);
  assert.equal(ac.oscillators.length, 0);
  assert.equal(player.play(buildTiming("##", 10), 1000), 0);
  assert.equal(ac.oscillators.length, 0);
  assert.equal(ended, 0, "nothing was playing, so nothing ended");
});

test("empty timing while playing only stops what is playing (with onEnd), like the Python player", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  let ended = 0;
  player.onEnd = () => {
    ended += 1;
  };
  player.play(SOS_10WPM, 1000);
  const osc = ac.oscillators[0];
  assert.equal(player.play([], 1000), 0);
  assert.equal(player.playing, false);
  assert.equal(ended, 1);
  assert.ok(osc.disconnected);
  assert.equal(ac.oscillators.length, 1, "no new oscillator");
  player.stop();
  assert.equal(ended, 1);
});

test("f0 and gain are validated", () => {
  const ac = new FakeAudioContext();
  const player = new TonePlayer(ac);
  for (const f0 of [0, -100, NaN, 24000, 30000]) {
    assert.throws(() => player.play(SOS_10WPM, f0), RangeError, `f0 ${f0}`);
  }
  for (const gain of [-0.1, 1.5, NaN]) {
    assert.throws(() => player.play(SOS_10WPM, 1000, gain), RangeError, `gain ${gain}`);
  }
  assert.equal(player.playing, false);
  assert.equal(ac.oscillators.length, 0);
  // just under Nyquist is fine
  player.play(SOS_10WPM, 23999);
  assert.equal(player.playing, true);
  player.stop();
});

test("onProgress reports the playhead and onEnd fires when the sequence is over", async () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  const progress = [];
  player.onProgress = (ms) => progress.push(ms);
  const ended = new Promise((resolve) => {
    player.onEnd = resolve;
  });
  player.play(buildTiming("E", 10), 1000); // 120 ms
  // before the first mark the playhead is clamped to 0
  await sleep(40);
  assert.ok(progress.length >= 1, "progress ticks while playing");
  assert.ok(progress.every((ms) => ms === 0), `lead time reports 0, got ${progress}`);
  assert.equal(player.elapsedMs, 0);

  ac.currentTime = player.startTime + 0.05; // 50 ms into the mark
  await sleep(40);
  assertClose(progress[progress.length - 1], 50, 1e-6);
  assertClose(player.elapsedMs, 50, 1e-6);
  assert.equal(player.playing, true);

  ac.currentTime = player.startTime + 0.2; // past the end
  await ended;
  assert.equal(player.playing, false);
  assert.equal(progress[progress.length - 1], 120, "the last progress call is clamped to the duration");
  assert.equal(player.elapsedMs, 0);
  assert.ok(ac.oscillators[0].disconnected);
});

test("the oscillator's onended also ends playback", () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  let ended = 0;
  player.onEnd = () => {
    ended += 1;
  };
  player.play(SOS_10WPM, 1000);
  const osc = ac.oscillators[0];
  assert.equal(typeof osc.onended, "function");
  osc.onended();
  assert.equal(player.playing, false);
  assert.equal(ended, 1);
  player.stop();
  assert.equal(ended, 1);
});

// ------------------------------------------------ keying guide labels, outputs

test("layoutGuideLabels labels every letter of HELLO WORLD! including the E", () => {
  // At 8 WPM in an 800 px guide the E is a single 150 ms dit about 6 px wide;
  // a minimum-width gate used to drop its label.
  const { letters, totalMs } = buildGuide("HELLO WORLD!", 8);
  const scale = 800 / totalMs;
  const placed = layoutGuideLabels(letters, (ms) => ms * scale, () => 7);
  assert.deepEqual(placed.map((p) => p.ch), [..."HELLOWORLD!"]);
  const xs = placed.map((p) => p.x);
  for (let i = 1; i < xs.length; i++) assert.ok(xs[i] - xs[i - 1] >= 7 + 3, "labels never overlap");
});

test("layoutGuideLabels skips only colliding labels when squeezed", () => {
  const { letters, totalMs } = buildGuide("HELLO WORLD!", 8);
  const scale = 60 / totalMs;
  const placed = layoutGuideLabels(letters, (ms) => ms * scale, () => 7);
  assert.ok(placed.length > 0 && placed.length < letters.length);
  assert.equal(placed[0].ch, "H");
  const xs = placed.map((p) => p.x);
  for (let i = 1; i < xs.length; i++) assert.ok(xs[i] - xs[i - 1] >= 10);
});

test("TonePlayer extra outputs: registered before play, connected live, removable", () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  const bus = new FakeNode();
  player.addOutput(bus);
  player.addOutput(bus); // idempotent
  assert.deepEqual(player.outputs, [bus]);
  player.play(SOS_10WPM, 2491, 0.15);
  const gain = ac.gains[0];
  assert.ok(ac.gains[1].connectedTo.includes(ac.destination), "speakers still get the tone, through the speaker gain");
  assert.ok(gain.connectedTo.includes(bus), "the bus gets the tone");
  const late = new FakeNode();
  player.addOutput(late);
  assert.ok(gain.connectedTo.includes(late), "an output added while playing is connected at once");
  player.removeOutput(late);
  assert.deepEqual(player.outputs, [bus]);
  player.clearOutputs();
  assert.deepEqual(player.outputs, []);
  player.stop();
  player.addOutput(bus);
  player.play(SOS_10WPM, 2491, 0.15);
  assert.ok(ac.gains[2].connectedTo.includes(bus), "outputs persist across plays (the second play made gains 2 and 3)");
  player.stop();
});


// -------------------------------------------------------------- Farnsworth

test("farnsworthGaps follow the ARRL rule", () => {
  assert.deepEqual(farnsworthGaps(18, 5), { letterGap: 1568, wordGap: 3660 });
  assert.deepEqual(farnsworthGaps(18), { letterGap: 200, wordGap: 467 });
  assert.deepEqual(farnsworthGaps(18, 18), { letterGap: 200, wordGap: 467 });
  assert.deepEqual(farnsworthGaps(18, 25), { letterGap: 200, wordGap: 467 });
  assert.throws(() => farnsworthGaps(18, 0), RangeError);
});

test("buildTiming with Farnsworth stretches only the gaps between letters", () => {
  const std = buildTiming("SOS E", 18);
  const farns = buildTiming("SOS E", 18, 5);
  assert.equal(std.length, farns.length);
  std.forEach((s, i) => {
    assert.equal(s.on, farns[i].on);
    if (s.on || s.ms === 67) assert.equal(farns[i].ms, s.ms);
  });
  const letterGaps = farns.filter((s, i) => !s.on && std[i].ms === 200).map((s) => s.ms);
  const wordGaps = farns.filter((s, i) => !s.on && std[i].ms === 467).map((s) => s.ms);
  assert.deepEqual(letterGaps, [1568, 1568]);
  assert.deepEqual(wordGaps, [3660]);
  assert.deepEqual(buildTiming("SOS E", 18, 20), std);
  const guide = buildGuide("SOS E", 18, 5);
  assert.equal(guide.totalMs, timingDurationMs(farns));
});

test("TonePlayer speakers off and stateAtNow", () => {
  const ac = new FakeAudioContext({ state: "running" });
  const player = new TonePlayer(ac);
  const bus = new FakeNode();
  player.addOutput(bus);
  player.setSpeakers(false);
  player.play(buildTiming("E", 10), 2491); // one 120 ms dit
  assert.equal(ac.gains[1].gain.value, 0, "the speaker branch starts muted");
  assert.ok(ac.gains[0].connectedTo.includes(bus), "the feed still gets the tone");
  ac.currentTime = player.startTime + 0.05;
  assert.equal(player.stateAtNow(), true);
  ac.currentTime = player.startTime + 0.13;
  assert.equal(player.stateAtNow(), false);
  player.stop();
  assert.equal(player.stateAtNow(), false);
});
