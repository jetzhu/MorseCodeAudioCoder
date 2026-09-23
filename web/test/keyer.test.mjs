// Tests for web/js/keyer.js: the pure Keyer state machine (mirrors
// tests/test_keyer.py) and LiveKey against a fake AudioContext.
import { test } from "node:test";
import assert from "node:assert/strict";

import { Keyer, LiveKey } from "../js/keyer.js";
import { MorseDecoder } from "../js/decoder.js";
import { Run } from "../js/runs.js";

const T = 100; // dit length used below (12 WPM)

function runsFrom(transitions, endMs) {
  const out = [];
  for (let i = 0; i < transitions.length; i++) {
    const [t0, on] = transitions[i];
    const t1 = i + 1 < transitions.length ? transitions[i + 1][0] : endMs;
    out.push(new Run(on, Math.max(1, Math.round((t1 - t0) / 10))));
  }
  return out;
}

function decode(transitions, endMs) {
  const dec = new MorseDecoder({ wpm: 1200 / T });
  let text = "";
  for (const r of runsFrom(transitions, endMs)) text += dec.feed(r);
  return (text + dec.idle(Infinity)).trim();
}

// ------------------------------------------------------------ straight key

test("straight key follows the hand", () => {
  const k = new Keyer(T);
  assert.deepEqual(k.keyDown(10), [[10, true]]);
  assert.deepEqual(k.keyDown(20), []);
  assert.equal(k.stateAt(15), true);
  assert.deepEqual(k.keyUp(250), [[250, false]]);
  assert.deepEqual(k.keyUp(260), []);
  assert.equal(k.stateAt(300), false);
  assert.equal(k.nextWakeupMs, null);
  assert.deepEqual(k.tick(1000), []);
});

test("straight key ignores paddles and vice versa", () => {
  assert.deepEqual(new Keyer(T).paddleDown("dit", 0), []);
  assert.deepEqual(new Keyer(T, "paddle").keyDown(0), []);
});

test("hand-keyed SOS on the straight key decodes", () => {
  const k = new Keyer(T);
  let t = 0;
  const trans = [];
  for (const symbol of "... --- ...") {
    if (symbol === " ") {
      t += 2 * T;
      continue;
    }
    trans.push(...k.keyDown(t));
    t += symbol === "." ? T : 3 * T;
    trans.push(...k.keyUp(t));
    t += T;
  }
  assert.equal(decode(trans, t + 1000), "SOS");
});

// ----------------------------------------------------------------- paddles

test("a tap sends one full element", () => {
  const k = new Keyer(T, "paddle");
  assert.deepEqual(k.paddleDown("dit", 0), [[0, true], [T, false]]);
  assert.deepEqual(k.paddleUp("dit", 30), []);
  assert.equal(k.nextWakeupMs, T);
  assert.deepEqual(k.tick(T), []);
  assert.equal(k.nextWakeupMs, 2 * T);
  assert.deepEqual(k.tick(2 * T), []);
  assert.equal(k.nextWakeupMs, null);
  assert.equal(k.stateAt(50), true);
  assert.equal(k.stateAt(150), false);
});

test("holding a paddle repeats with one-dit gaps", () => {
  const k = new Keyer(T, "paddle");
  const trans = k.paddleDown("dah", 0);
  while (trans.length < 8) trans.push(...k.tick(k.nextWakeupMs)); // a tick at an element's end only opens the gap
  assert.deepEqual(trans, [
    [0, true], [3 * T, false],
    [4 * T, true], [7 * T, false],
    [8 * T, true], [11 * T, false],
    [12 * T, true], [15 * T, false],
  ]);
  k.paddleUp("dah", 12.5 * T);
  assert.deepEqual(k.tick(15 * T), []);
  assert.deepEqual(k.tick(16 * T), []);
  assert.equal(k.nextWakeupMs, null);
});

test("a late tick catches up element by element", () => {
  const k = new Keyer(T, "paddle");
  k.paddleDown("dit", 0);
  assert.deepEqual(k.tick(4.5 * T), [[2 * T, true], [3 * T, false], [4 * T, true], [5 * T, false]]);
});

test("paddle memory makes a clean A", () => {
  const k = new Keyer(T, "paddle");
  const trans = k.paddleDown("dit", 0);
  trans.push(...k.paddleDown("dah", 40));
  k.paddleUp("dah", 60);
  k.paddleUp("dit", 70);
  trans.push(...k.tick(2 * T));
  assert.deepEqual(trans, [[0, true], [T, false], [2 * T, true], [5 * T, false]]);
  assert.deepEqual(k.tick(6 * T), []);
  assert.equal(decode(trans, 10 * T), "A");
});

test("holding both paddles alternates", () => {
  const k = new Keyer(T, "paddle");
  const trans = k.paddleDown("dit", 0);
  k.paddleDown("dah", 10);
  while (trans.length < 8) trans.push(...k.tick(k.nextWakeupMs));
  const lengths = [];
  for (let i = 0; i < trans.length; i += 2) lengths.push(trans[i + 1][0] - trans[i][0]);
  assert.deepEqual(lengths, [T, 3 * T, T, 3 * T]);
});

test("releaseAll and setMode end a sounding tone", () => {
  const k = new Keyer(T, "paddle");
  k.paddleDown("dah", 0); // logs the ON edge and the OFF edge at 3T
  assert.deepEqual(k.releaseAll(50), [[50, false]]); // the future OFF edge is dropped, the tone ends now
  assert.equal(k.stateAt(60), false);
  assert.deepEqual(k.transitionsSince(0).items, [[0, true], [50, false]]);
  assert.deepEqual(k.tick(1000), []);
  const s = new Keyer(T);
  s.keyDown(0);
  assert.deepEqual(s.setMode("paddle", 30), [[30, false]]);
  assert.equal(s.mode, "paddle");
  assert.deepEqual(s.keyDown(40), []);
});

test("setSpeed applies to the next element", () => {
  const k = new Keyer(T, "paddle");
  k.paddleDown("dit", 0);
  k.setSpeed(60);
  assert.deepEqual(k.tick(2 * T), [[2 * T, true], [2 * T + 60, false]]);
});

test("the log is monotonic and readable incrementally", () => {
  const k = new Keyer(T);
  k.keyDown(100);
  k.keyUp(50); // out of order: clamped
  const first = k.transitionsSince(0);
  assert.deepEqual(first.items, [[100, true], [100, false]]);
  assert.equal(first.index, 2);
  assert.deepEqual(k.transitionsSince(first.index).items, []);
});

test("validation", () => {
  assert.throws(() => new Keyer(0), RangeError);
  assert.throws(() => new Keyer(T, "iambic"), RangeError);
  assert.throws(() => new Keyer(T, "paddle").paddleDown("squeeze", 0), RangeError);
});

// ----------------------------------------------------------------- LiveKey

class FakeParam {
  constructor(v) {
    this.value = v;
    this.events = [];
  }
  setTargetAtTime(target, at, tau) {
    this.events.push({ target, at, tau });
  }
  setValueAtTime(v, at) {
    this.events.push({ value: v, at });
  }
}
class FakeNode {
  constructor() {
    this.connectedTo = [];
    this.disconnected = false;
  }
  connect(n) {
    this.connectedTo.push(n);
    return n;
  }
  disconnect() {
    this.disconnected = true;
  }
}
class FakeOsc extends FakeNode {
  constructor() {
    super();
    this.type = "square";
    this.frequency = new FakeParam(440);
    this.started = false;
    this.stopped = false;
  }
  start() {
    this.started = true;
  }
  stop() {
    this.stopped = true;
  }
}
class FakeGain extends FakeNode {
  constructor() {
    super();
    this.gain = new FakeParam(1);
  }
}
class FakeAC {
  constructor() {
    this.currentTime = 2.0;
    this.destination = { name: "destination" };
    this.oscs = [];
    this.gains = [];
  }
  createOscillator() {
    const o = new FakeOsc();
    this.oscs.push(o);
    return o;
  }
  createGain() {
    const g = new FakeGain();
    this.gains.push(g);
    return g;
  }
}

test("LiveKey schedules gain edges on the AudioContext clock and feeds extra outputs", () => {
  const ac = new FakeAC();
  const keyer = new Keyer(T);
  const lk = new LiveKey(keyer, { gain: 0.2 });
  const bus = new FakeNode();
  lk.addOutput(bus);
  assert.equal(lk.running, false);
  assert.equal(lk.isOn(), false);
  lk.start(ac, 2491);
  assert.equal(lk.running, true);
  const [osc] = ac.oscs;
  const [g] = ac.gains;
  assert.equal(osc.type, "sine");
  assert.equal(osc.frequency.value, 2491);
  assert.ok(osc.started);
  assert.ok(g.connectedTo.includes(ac.destination));
  assert.ok(g.connectedTo.includes(bus));
  lk.keyDown();
  assert.equal(lk.isOn(), false, "the edge is 5 ms ahead of now");
  ac.currentTime += 0.010;
  assert.equal(lk.isOn(), true);
  assert.equal(g.gain.events.length, 1);
  assert.equal(g.gain.events[0].target, 0.2);
  assert.ok(g.gain.events[0].at >= 2.0);
  lk.keyUp();
  assert.equal(g.gain.events[1].target, 0);
  lk.setFrequency(1000);
  assert.equal(osc.frequency.events.at(-1).value, 1000);
  lk.stop();
  assert.ok(osc.stopped && osc.disconnected && g.disconnected);
  assert.equal(lk.running, false);
});

test("LiveKey paddle: the element end is scheduled at once and the timer arms for the gap", async () => {
  const ac = new FakeAC();
  const keyer = new Keyer(50, "paddle"); // 24 WPM keeps the test quick
  const lk = new LiveKey(keyer);
  lk.start(ac, 2491);
  let changes = 0;
  lk.onChange = () => {
    changes += 1;
  };
  lk.paddleDown("dit");
  const [g] = ac.gains;
  assert.equal(g.gain.events.length, 2, "on and off edges scheduled together");
  assert.ok(g.gain.events[1].at - g.gain.events[0].at >= 0.05 - 1e-9);
  assert.ok(lk._timer !== null, "a wake-up is armed for the element end");
  lk.paddleUp("dit");
  // Let the armed timers run: the fake clock does not advance, so tick sees
  // the element still sounding and simply re-arms; then stop cancels it.
  await new Promise((r) => setTimeout(r, 80));
  assert.ok(changes >= 2);
  lk.stop();
  assert.equal(lk._timer, null);
});
