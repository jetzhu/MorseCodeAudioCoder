// Tests for web/js/torch.js against fake camera tracks.
import { test } from "node:test";
import assert from "node:assert/strict";

import { TORCH_MAX_WPM, Torch } from "../js/torch.js";

class FakeTrack {
  constructor({ torch = true, delayMs = 5 } = {}) {
    this.readyState = "live";
    this.torch = torch;
    this.delayMs = delayMs;
    this.calls = [];
    this.stopped = false;
  }
  getCapabilities() {
    return this.torch ? { torch: true } : {};
  }
  applyConstraints(c) {
    this.calls.push(c.advanced[0].torch);
    return new Promise((r) => setTimeout(r, this.delayMs));
  }
  stop() {
    this.stopped = true;
    this.readyState = "ended";
  }
}
const streamOf = (track) => ({ getVideoTracks: () => [track], getTracks: () => [track] });
const settle = (ms = 60) => new Promise((r) => setTimeout(r, ms));

test("opens its own camera on first use and switches the torch", async () => {
  const track = new FakeTrack();
  let opened = 0;
  const torch = new Torch({ getUserMedia: async () => (opened++, streamOf(track)) });
  assert.equal(TORCH_MAX_WPM, 8);
  assert.equal(torch.ready, false);
  assert.equal(await torch.ensure(), true);
  assert.equal(opened, 1);
  assert.equal(torch.ready, true);
  torch.set(true);
  await settle();
  assert.deepEqual(track.calls, [true]);
  torch.set(false);
  await settle();
  assert.deepEqual(track.calls, [true, false]);
  assert.equal(await torch.ensure(), true);
  assert.equal(opened, 1, "the open stream is reused");
  torch.releaseOwn();
  assert.equal(track.stopped, true);
});

test("a burst of edges is coalesced to the latest state", async () => {
  const track = new FakeTrack({ delayMs: 30 });
  const torch = new Torch({ getUserMedia: async () => streamOf(track) });
  await torch.ensure();
  torch.set(true); // in flight
  torch.set(false);
  torch.set(true);
  torch.set(false); // only this one matters once the first finishes
  await settle(120);
  assert.deepEqual(track.calls, [true, false]);
  assert.equal(torch.applied, false);
});

test("uses the listening camera's track instead of opening another", async () => {
  const cam = new FakeTrack();
  let opened = 0;
  const torch = new Torch({ external: () => cam, getUserMedia: async () => (opened++, streamOf(new FakeTrack())) });
  assert.equal(await torch.ensure(), true);
  assert.equal(opened, 0);
  torch.set(true);
  await settle();
  assert.deepEqual(cam.calls, [true]);
  torch.releaseOwn();
  assert.equal(cam.stopped, false, "the listening camera is left alone");
});

test("no torch capability (iPhone, laptops) or no permission: unavailable with a reason", async () => {
  const noTorch = new FakeTrack({ torch: false });
  const t1 = new Torch({ getUserMedia: async () => streamOf(noTorch) });
  assert.equal(await t1.ensure(), false);
  assert.equal(t1.supported, false);
  assert.match(t1.reason, /Chrome on Android/);
  assert.equal(noTorch.stopped, true, "the stream opened just to check is closed");
  t1.set(true);
  await settle();
  assert.deepEqual(noTorch.calls, []);
  const denied = Object.assign(new Error("no"), { name: "NotAllowedError" });
  const t2 = new Torch({ getUserMedia: async () => { throw denied; } });
  assert.equal(await t2.ensure(), false);
  assert.match(t2.reason, /permission/);
});

test("a driver error marks the torch unavailable instead of retrying forever", async () => {
  const track = new FakeTrack();
  track.applyConstraints = () => Promise.reject(new Error("busy"));
  const torch = new Torch({ getUserMedia: async () => streamOf(track) });
  await torch.ensure();
  torch.set(true);
  await settle();
  assert.equal(torch.supported, false);
  assert.match(torch.reason, /stopped responding/);
});
