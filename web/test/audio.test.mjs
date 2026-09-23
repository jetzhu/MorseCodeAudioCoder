// Tests for the parts of web/js/audio.js that run without a browser: the
// getUserMedia constraints for both microphone modes and the settings readout.
import { test } from "node:test";
import assert from "node:assert/strict";

import { PROCESSING_MODES, captureConstraints, describeProcessing } from "../js/audio.js";

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
