// Tests for web/js/channels.js and web/js/light.js (mirrors tests/test_channels.py).
import { test } from "node:test";
import assert from "node:assert/strict";

import { DEFAULTS, LISTEN, SEND, effective, sanitize, timingStateAt, vibrationPattern } from "../js/channels.js";
import { Lamp } from "../js/light.js";
import { buildTiming } from "../js/player.js";

test("defaults select everything and sanitize repairs input", () => {
  assert.deepEqual(LISTEN, ["mic", "camera"]);
  assert.deepEqual(SEND, ["audio", "light", "torch", "vibrate"]);
  assert.deepEqual(sanitize(null), { listen: { mic: true, camera: true }, send: { audio: true, light: true, torch: true, vibrate: true } });
  assert.deepEqual(sanitize("junk"), sanitize(undefined));
  const s = sanitize({ listen: { mic: false, camera: "yes" }, send: { audio: 0, light: false, extra: true } });
  assert.deepEqual(s, { listen: { mic: false, camera: true }, send: { audio: true, light: false, torch: true, vibrate: true } });
  assert.equal(Object.isFrozen(DEFAULTS), true);
});

test("effective is selected and available", () => {
  const sel = sanitize({ send: { audio: false } });
  const avail = { listen: { mic: true, camera: false }, send: { audio: true, light: true, torch: false, vibrate: true } };
  assert.deepEqual(effective(sel, avail), { listen: { mic: true, camera: false }, send: { audio: false, light: true, torch: false, vibrate: true } });
});

test("timingStateAt walks the keying sequence", () => {
  const t = buildTiming("A", 12); // dit 100, gap 100, dah 300
  assert.equal(timingStateAt(t, -1), false);
  assert.equal(timingStateAt(t, 0), true);
  assert.equal(timingStateAt(t, 99.9), true);
  assert.equal(timingStateAt(t, 100), false);
  assert.equal(timingStateAt(t, 199.9), false);
  assert.equal(timingStateAt(t, 200), true);
  assert.equal(timingStateAt(t, 499.9), true);
  assert.equal(timingStateAt(t, 500), false);
  assert.equal(timingStateAt([], 0), false);
});

test("vibrationPattern alternates vibrate and pause, starting with a vibration", () => {
  assert.deepEqual(vibrationPattern(buildTiming("A", 12)), [100, 100, 300]);
  assert.deepEqual(vibrationPattern(buildTiming("E E", 12)), [100, 700, 100]);
  assert.deepEqual(vibrationPattern([{ on: false, ms: 50 }, { on: true, ms: 20 }]), [0, 50, 20]);
  assert.deepEqual(vibrationPattern([{ on: true, ms: 10 }, { on: true, ms: 15 }, { on: false, ms: 0 }, { on: false, ms: 5 }]), [25, 5]);
  assert.deepEqual(vibrationPattern([{ on: true, ms: 10 }, { on: false, ms: 30 }]), [10, 30]);
  assert.deepEqual(vibrationPattern([]), []);
});

class FakeEl {
  constructor() {
    this.classes = new Set();
    this.hidden = true;
    this.classList = { toggle: (n, f) => (f ? this.classes.add(n) : this.classes.delete(n)) };
  }
}

test("Lamp drives every element and the full-screen overlay", () => {
  const a = new FakeEl();
  const b = new FakeEl();
  const full = new FakeEl();
  const lamp = new Lamp([a, b], full);
  lamp.setState(true, false);
  assert.ok(a.classes.has("rx") && b.classes.has("rx") && full.classes.has("rx"));
  assert.ok(!a.classes.has("tx"));
  lamp.setState(true, true);
  assert.ok(a.classes.has("tx") && full.classes.has("tx"));
  lamp.setState(false, false);
  assert.equal(a.classes.size, 0);
  assert.equal(full.hidden, true);
  lamp.enterFull();
  assert.equal(full.hidden, false);
  assert.equal(lamp.fullActive, true);
  lamp.exitFull();
  assert.equal(full.hidden, true);
  const bare = new Lamp([a]);
  bare.enterFull(); // no overlay: harmless
  assert.equal(bare.fullActive, false);
});
