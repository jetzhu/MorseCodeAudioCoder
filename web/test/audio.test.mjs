// Tests for the parts of web/js/audio.js that run without a browser: the
// getUserMedia constraints for both microphone modes and the settings readout.
import { test } from "node:test";
import assert from "node:assert/strict";

import { MIC_GATE_GAIN, MIC_GATE_TAIL_MS, MicGate, PROCESSING_MODES, captureConstraints, describeProcessing } from "../js/audio.js";

test("raw mode asks for mono with every processing stage off", () => {
  assert.deepEqual(PROCESSING_MODES, ["raw", "browser"]);
  assert.deepEqual(captureConstraints(), {
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  assert.deepEqual(captureConstraints("abc", "raw").audio.deviceId, "abc");
  assert.equal("deviceId" in captureConstraints("").audio, false);
});

test("browser mode leaves the processing constraints to the browser", () => {
  const { audio } = captureConstraints("mic-1", "browser");
  assert.deepEqual(audio, { channelCount: 1, deviceId: "mic-1" });
  assert.throws(() => captureConstraints("", "loud"), RangeError);
});

test("describeProcessing reads what the browser reports", () => {
  assert.equal(describeProcessing(null), "unknown");
  assert.equal(describeProcessing({ deviceId: "x", sampleRate: 48000 }), "unknown");
  assert.equal(describeProcessing({ echoCancellation: false, noiseSuppression: false, autoGainControl: false }), "raw");
  assert.equal(describeProcessing({ echoCancellation: false, noiseSuppression: true }), "processed");
  assert.equal(describeProcessing({ autoGainControl: true }), "processed");
});

test("MicGate attenuates while sounding and for the tail (mirror of tests/test_player.py)", () => {
  assert.equal(MIC_GATE_GAIN, 0.01);
  assert.equal(MIC_GATE_TAIL_MS, 400);
  const g = new MicGate();
  assert.equal(g.active, false);
  assert.equal(g.update(false, 0), false);
  assert.equal(g.update(true, 1000), true);
  assert.equal(g.active, true);
  assert.equal(g.update(false, 1399), true);
  assert.equal(g.update(false, 1400), false);
  assert.equal(g.active, false);
  g.update(true, 2000);
  g.update(false, 2300);
  assert.equal(g.update(true, 2350), true);
  assert.equal(g.update(false, 2740), true);
  assert.equal(g.update(false, 2750), false);
  g.update(true, 3000);
  g.reset();
  assert.equal(g.active, false);
  assert.equal(g.update(false, 3001), false);
  const short = new MicGate(0);
  assert.equal(short.update(true, 0), true);
  assert.equal(short.update(false, 0), false);
  assert.throws(() => new MicGate(-1), RangeError);
});
