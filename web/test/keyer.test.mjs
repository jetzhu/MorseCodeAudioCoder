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
  assert.throws(() => new Keyer(T, "straight", { iambic: "C" }), RangeError);
  assert.throws(() => new Keyer(T, "straight", { dahRatio: 1.5 }), RangeError);
  assert.throws(() => new Keyer(T, "straight", { weight: 80 }), RangeError);
  const k = new Keyer(T);
  assert.throws(() => k.setIambic("ab"), RangeError);
  assert.throws(() => k.setWeighting(null, NaN), RangeError);
  k.setWeighting();
  assert.deepEqual([k.dahRatio, k.weight], [3, 50]);
});

// ---------------------------------------------------------- iambic A and B

function squeezeReleaseDuringDah(iambic) {
  const k = new Keyer(T, "paddle", { iambic });
  const trans = k.paddleDown("dit", 0);
  trans.push(...k.paddleDown("dah", 30)); // squeezed during the dit: dah remembered
  trans.push(...k.tick(2 * T)); // gap over: the dah starts at 2T and ends at 5T
  k.paddleUp("dit", 3 * T);
  k.paddleUp("dah", 3.5 * T); // both released during the dah
  for (let i = 0; i < 4; i++) {
    const w = k.nextWakeupMs;
    if (w === null) break;
    trans.push(...k.tick(w));
  }
  return trans;
}

test("iambic A ends with the element in progress", () => {
  const trans = squeezeReleaseDuringDah("A");
  assert.deepEqual(trans, [[0, true], [T, false], [2 * T, true], [5 * T, false]]);
  assert.equal(decode(trans, 12 * T), "A");
});

test("iambic B adds one opposite element after a squeeze", () => {
  const trans = squeezeReleaseDuringDah("B");
  assert.deepEqual(trans, [[0, true], [T, false], [2 * T, true], [5 * T, false], [6 * T, true], [7 * T, false]]);
  assert.equal(decode(trans, 12 * T), "R");
});

test("iambic B: a squeeze released during the first element gives a plain A", () => {
  const k = new Keyer(T, "paddle", { iambic: "B" });
  const trans = k.paddleDown("dit", 0);
  k.paddleDown("dah", 20);
  k.paddleUp("dit", 40);
  k.paddleUp("dah", 50);
  while (k.nextWakeupMs !== null) trans.push(...k.tick(k.nextWakeupMs));
  assert.deepEqual(trans, [[0, true], [T, false], [2 * T, true], [5 * T, false]]);
});

test("iambic B without a squeeze behaves like A", () => {
  const k = new Keyer(T, "paddle", { iambic: "B" });
  const trans = k.paddleDown("dah", 0);
  k.paddleUp("dah", 50);
  while (k.nextWakeupMs !== null) trans.push(...k.tick(k.nextWakeupMs));
  assert.deepEqual(trans, [[0, true], [3 * T, false]]);
  k.setIambic("A");
  assert.equal(k.iambic, "A");
});

// ----------------------------------------------------------------- bug mode

test("bug: dits are automatic and dahs by hand", () => {
  const k = new Keyer(T, "bug");
  assert.deepEqual(k.keyDown(0), []);
  const trans = k.paddleDown("dit", 0);
  while (trans.length < 4) trans.push(...k.tick(k.nextWakeupMs));
  assert.deepEqual(trans, [[0, true], [T, false], [2 * T, true], [3 * T, false]]);
  k.paddleUp("dit", 2.5 * T);
  assert.deepEqual(k.tick(4 * T), []);
  assert.equal(k.nextWakeupMs, null);
  assert.deepEqual(k.paddleDown("dah", 5 * T), [[5 * T, true]]);
  assert.equal(k.nextWakeupMs, null);
  assert.deepEqual(k.tick(6 * T), []);
  assert.deepEqual(k.paddleDown("dah", 6 * T), []);
  assert.deepEqual(k.paddleUp("dah", 8.7 * T), [[8.7 * T, false]]);
  assert.deepEqual(k.paddleUp("dah", 9 * T), []);
  assert.equal(k.stateAt(7 * T), true);
  assert.equal(k.stateAt(9 * T), false);
});

test("bug: the lever cuts a dit short and dits resume after it", () => {
  const k = new Keyer(T, "bug");
  const trans = k.paddleDown("dit", 0);
  trans.push(...k.paddleDown("dah", 0.5 * T));
  assert.deepEqual(trans, [[0, true], [T, false]]); // what the calls returned; the log dropped the OFF edge
  assert.deepEqual(k.transitionsSince(0).items, [[0, true]]);
  assert.deepEqual(k.paddleDown("dit", 1.5 * T), []);
  trans.push(...k.paddleUp("dah", 3 * T));
  assert.deepEqual(trans.at(-1), [3 * T, false]);
  assert.equal(k.nextWakeupMs, 4 * T);
  trans.push(...k.tick(4 * T));
  assert.deepEqual(trans.slice(-2), [[4 * T, true], [5 * T, false]]);
  k.paddleUp("dit", 4.5 * T);
  assert.deepEqual(k.releaseAll(4.6 * T), [[4.6 * T, false]]);
  assert.deepEqual(k.tick(10 * T), []);
});

test("bug: memory is not used", () => {
  const k = new Keyer(T, "bug");
  k.paddleDown("dit", 0);
  k.paddleDown("dah", 0.3 * T);
  k.paddleUp("dit", 0.4 * T);
  assert.deepEqual(k.paddleUp("dah", 2 * T), [[2 * T, false]]);
  assert.equal(k.nextWakeupMs, null);
});

// ---------------------------------------------------------------- weighting

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} vs ${b}`);

test("weighting shapes marks and spaces but keeps the period", () => {
  const k = new Keyer(T, "paddle", { dahRatio: 4, weight: 60 });
  near(k.markMs("dit"), 1.2 * T);
  near(k.markMs("dah"), 4.2 * T);
  near(k.gapMs, 0.8 * T);
  let trans = k.paddleDown("dit", 0);
  while (trans.length < 4) trans.push(...k.tick(k.nextWakeupMs));
  const expect1 = [[0, true], [1.2 * T, false], [2 * T, true], [3.2 * T, false]];
  trans.forEach(([t, on], i) => {
    near(t, expect1[i][0]);
    assert.equal(on, expect1[i][1]);
  });
  k.releaseAll(3.5 * T);
  k.setWeighting(null, 40);
  trans = k.paddleDown("dah", 10 * T);
  while (trans.length < 4) trans.push(...k.tick(k.nextWakeupMs));
  const expect2 = [[10 * T, true], [13.8 * T, false], [15 * T, true], [18.8 * T, false]];
  trans.forEach(([t, on], i) => {
    near(t, expect2[i][0]);
    assert.equal(on, expect2[i][1]);
  });
  k.setWeighting(3, 50);
  assert.deepEqual([k.markMs("dit"), k.markMs("dah"), k.gapMs], [T, 3 * T, T]);
});

test("LiveKey passes keyer variants through", () => {
  const keyer = new Keyer(T, "paddle");
  const lk = new LiveKey(keyer);
  lk.setIambic("B");
  lk.setWeighting(3.5, 55);
  assert.deepEqual([keyer.iambic, keyer.dahRatio, keyer.weight], ["B", 3.5, 55]);
});

// The Sent line's rule: runs from the keyer's transitions, then the growing
// silence reported through idle(); the last letter closes after seven dit
// lengths with no further input (the page must keep calling idle for that).
test("a hand-keyed letter is committed by idle() after seven dits of silence", () => {
  const k = new Keyer(T);
  const trans = [...k.keyDown(0), ...k.keyUp(T), ...k.keyDown(2 * T), ...k.keyUp(5 * T)]; // dit dah: A
  const dec = new MorseDecoder({ wpm: 1200 / T });
  let text = "";
  for (let i = 1; i < trans.length; i++) {
    const [t0, on] = trans[i - 1];
    text += dec.feed(new Run(on, Math.round((trans[i][0] - t0) / 10)));
  }
  assert.equal(text, "");
  assert.equal(dec.buffer, ".-", "the letter is still pending");
  assert.equal(dec.idle(6 * T), "", "not yet");
  assert.equal(dec.idle(7.5 * T), "A ", "closed by the pause alone");
  assert.equal(dec.buffer, "");
  assert.equal(dec.idle(20 * T), "", "flushes once");
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
  const [osc, oscFeed] = ac.oscs;
  const [g, gFeed] = ac.gains;
  assert.equal(osc.type, "sine");
  assert.equal(osc.frequency.value, 2491, "no sidetone: the speakers follow f0");
  assert.equal(oscFeed.frequency.value, 2491);
  assert.ok(osc.started && oscFeed.started);
  assert.ok(g.connectedTo.includes(ac.destination));
  assert.ok(!g.connectedTo.includes(bus), "the speaker chain does not reach the decoder bus");
  assert.ok(gFeed.connectedTo.includes(bus));
  assert.ok(!gFeed.connectedTo.includes(ac.destination));
  lk.keyDown();
  assert.equal(lk.isOn(), false, "the edge is 5 ms ahead of now");
  ac.currentTime += 0.010;
  assert.equal(lk.isOn(), true);
  assert.equal(g.gain.events.length, 1);
  assert.equal(g.gain.events[0].target, 0.2);
  assert.ok(g.gain.events[0].at >= 2.0);
  lk.keyUp();
  assert.equal(g.gain.events[1].target, 0);
  assert.equal(gFeed.gain.events.length, 2, "the feed chain gets every edge too");
  lk.setFrequency(1000);
  assert.equal(osc.frequency.events.at(-1).value, 1000);
  assert.equal(oscFeed.frequency.events.at(-1).value, 1000);
  lk.stop();
  assert.ok(osc.stopped && osc.disconnected && g.disconnected);
  assert.ok(oscFeed.stopped && oscFeed.disconnected && gFeed.disconnected);
  assert.equal(lk.running, false);
});

test("LiveKey sidetone pitches the speakers while the feed stays on f0", () => {
  const ac = new FakeAC();
  const lk = new LiveKey(new Keyer(T));
  lk.setSidetone(600);
  assert.equal(lk.speakerHz, 600);
  lk.start(ac, 2491);
  const [osc, oscFeed] = ac.oscs;
  assert.equal(osc.frequency.value, 600);
  assert.equal(oscFeed.frequency.value, 2491);
  lk.setFrequency(1000);
  assert.equal(oscFeed.frequency.events.at(-1).value, 1000);
  assert.equal(osc.frequency.events.length, 0, "a sidetone does not follow f0");
  lk.setSidetone(null);
  assert.equal(lk.speakerHz, 1000);
  assert.equal(osc.frequency.events.at(-1).value, 1000);
  lk.setSidetone(0);
  assert.equal(lk.sidetoneHz, null);
  lk.setSidetone(NaN);
  assert.equal(lk.sidetoneHz, null);
  lk.setSidetone(750);
  assert.equal(osc.frequency.events.at(-1).value, 750);
  const bus = { connectedTo: [], connect() {}, disconnect() {} };
  lk.addOutput(bus);
  assert.ok(ac.gains[1].connectedTo.includes(bus));
  lk.stop();
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

test("LiveKey setSpeakers mutes the speaker chain only", () => {
  const ac = new FakeAC();
  const lk = new LiveKey(new Keyer(T), { gain: 0.2 });
  lk.setSpeakers(false);
  lk.start(ac, 2491);
  const [g, gFeed] = ac.gains;
  lk.keyDown();
  assert.equal(g.gain.events.at(-1).target, 0, "speakers muted");
  assert.equal(gFeed.gain.events.at(-1).target, 0.2, "the feed still gets the tone");
  ac.currentTime += 0.01;
  lk.setSpeakers(true);
  assert.equal(g.gain.events.at(-1).target, 0.2, "unmuted while the key is down");
  lk.keyUp();
  lk.stop();
});
