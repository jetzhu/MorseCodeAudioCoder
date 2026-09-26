// Light detection: turn a camera's brightness readings into Morse runs, and
// arbitrate between listen sources. Pure logic, tested in Node on synthetic
// frame sequences (web/test/lightdetect.test.mjs).
//
// LightDetector
//   Frames arrive at the camera's rate (typically 30 per second) with a
//   brightness value (mean luma 0..255 of the tracked spot) and a timestamp.
//   Over a sliding window it tracks a dark level and a bright level, each
//   robust to one-frame spikes (the min of neighbouring pairs for bright, the
//   max for dark), and switches with hysteresis at 60 % / 40 % of the way
//   from dark to bright. It stays OFF unless the contrast clears both an
//   absolute floor and a multiple of the frame-to-frame noise, so a steady
//   scene or sensor noise never produces marks. Each edge is placed where the
//   brightness crossed the mid level between the two frames around it
//   (exposure blur gives intermediate frames), not at the frame time, which
//   halves the timing error. Runs are emitted in 10 ms blocks, the unit the
//   decoder uses for audio, so the same MorseDecoder reads both.
//
// SourceArbiter
//   With the microphone and the camera both listening, one transmission can
//   arrive through both at different delays. The first source to start a
//   mark owns the message and only its runs reach the decoder; ownership is
//   released once the owner has been quiet for the release time (the caller
//   passes at least 2 s, or 8 dits at slow speeds), after which any source
//   may claim it again.

import { Run } from "./runs.js";

const BLOCK_MS = 10;

export class LightDetector {
  /**
   * @param {object} [options]
   * @param {number} [options.windowMs=6000] history used for the dark and bright levels
   * @param {number} [options.minContrast=12] luma units the bright level must clear the dark one by
   * @param {number} [options.noiseFactor=8] ... and this many times the median frame-to-frame change
   * @param {number} [options.hiFrac=0.6] switch ON above dark + hiFrac * contrast
   * @param {number} [options.loFrac=0.4] switch OFF below dark + loFrac * contrast
   */
  constructor({ windowMs = 6000, minContrast = 12, noiseFactor = 8, hiFrac = 0.6, loFrac = 0.4 } = {}) {
    this.windowMs = windowMs;
    this.minContrast = minContrast;
    this.noiseFactor = noiseFactor;
    this.hiFrac = hiFrac;
    this.loFrac = loFrac;
    this.reset();
  }

  reset() {
    /** @type {Array<{t: number, v: number}>} */
    this._samples = [];
    this.state = false;
    this.dark = 0;
    this.bright = 0;
    this.noise = 0;
    this.contrastOk = false;
    this._runStart = null;
    this._prev = null;
    this.frames = 0;
    this._firstT = null;
  }

  /** Frames per second over the window (0 until two frames have arrived). */
  get fps() {
    const s = this._samples;
    if (s.length < 2) return 0;
    const span = s[s.length - 1].t - s[0].t;
    return span > 0 ? ((s.length - 1) * 1000) / span : 0;
  }

  /**
   * One frame. Returns the runs completed by it (zero, or one at an edge).
   * @param {number} value brightness 0..255
   * @param {number} tMs frame time in ms (monotonic)
   * @returns {Run[]}
   */
  update(value, tMs) {
    const v = Number(value);
    const t = Number(tMs);
    if (!Number.isFinite(v) || !Number.isFinite(t)) return [];
    const prev = this._prev;
    if (prev && t <= prev.t) return []; // out of order or duplicate frame
    this.frames += 1;
    if (this._firstT === null) this._firstT = t;
    if (this._runStart === null) this._runStart = t;
    const s = this._samples;
    s.push({ t, v });
    while (s.length > 2 && t - s[0].t > this.windowMs) s.shift();
    this._levels();

    const c = this.bright - this.dark;
    const hi = this.dark + this.hiFrac * c;
    const lo = this.dark + this.loFrac * c;
    let next = this.state;
    if (!this.contrastOk) next = false;
    else if (!this.state && v > hi) next = true;
    else if (this.state && v < lo) next = false;
    this._prev = { t, v };
    if (next === this.state) return [];

    // The edge is where the brightness crossed the mid level. The switch can
    // come a frame or two after that (the first mark of a message is only
    // recognised once the bright level is known), so walk back over the
    // frames already on the new side, then interpolate between the last
    // frame on the old side and the first on the new one.
    const mid = this.dark + 0.5 * c;
    const onNewSide = (x) => (next ? x > mid : x < mid);
    let k = s.length - 1;
    for (let steps = 0; k > 0 && steps < 4 && onNewSide(s[k - 1].v); steps++) k--;
    let edge = s[k].t;
    if (k > 0) {
      const a = s[k - 1];
      const b = s[k];
      const dv = b.v - a.v;
      const frac = dv !== 0 ? Math.min(1, Math.max(0, (mid - a.v) / dv)) : 0.5;
      edge = a.t + frac * (b.t - a.t);
    }
    edge = Math.max(edge, this._runStart);
    const blocks = Math.max(1, Math.round((edge - this._runStart) / BLOCK_MS));
    const run = new Run(this.state, blocks, BLOCK_MS);
    this.state = next;
    this._runStart = edge;
    return [run];
  }

  /** Length of the current OFF run at `tMs` (0 while ON or before any frame). */
  currentOffMs(tMs) {
    if (this.state || this._runStart === null) return 0;
    return Math.max(0, tMs - this._runStart);
  }

  /** End of stream: the run in progress, closed at `tMs`. */
  flush(tMs) {
    if (this._runStart === null) return [];
    const blocks = Math.round((tMs - this._runStart) / BLOCK_MS);
    const out = blocks > 0 ? [new Run(this.state, blocks, BLOCK_MS)] : [];
    this.state = false;
    this._runStart = tMs;
    return out;
  }

  _levels() {
    const s = this._samples;
    if (s.length < 3) {
      this.contrastOk = false;
      return;
    }
    let bright = -Infinity;
    let dark = Infinity;
    const diffs = [];
    for (let i = 1; i < s.length; i++) {
      const a = s[i - 1].v;
      const b = s[i].v;
      bright = Math.max(bright, Math.min(a, b));
      dark = Math.min(dark, Math.max(a, b));
      diffs.push(Math.abs(b - a));
    }
    diffs.sort((x, y) => x - y);
    this.noise = diffs[Math.floor(diffs.length / 2)];
    this.bright = bright;
    this.dark = dark;
    const c = bright - dark;
    this.contrastOk = c >= this.minContrast && c >= this.noiseFactor * this.noise;
  }
}

export class SourceArbiter {
  constructor() {
    /** @type {string | null} the source whose runs reach the decoder */
    this.owner = null;
  }

  /**
   * A source completed `run`. An OFF run ends because the source has just
   * started a mark, so with no owner it claims the message. Returns whether
   * the run should reach the decoder.
   * @param {string} src @param {{on: boolean}} run
   */
  offer(src, run) {
    if (this.owner === null && !run.on) this.owner = src;
    return this.owner === src;
  }

  /**
   * Called regularly for each source: releases ownership once the owner has
   * been OFF for longer than `releaseMs`.
   * @param {string} src @param {boolean} on @param {number} offMs @param {number} releaseMs
   */
  tick(src, on, offMs, releaseMs) {
    if (this.owner === src && !on && offMs > releaseMs) this.owner = null;
  }

  /** Whether `src` may drive the decoder's idle flush now. @param {string} src */
  mayIdle(src) {
    return this.owner === null || this.owner === src;
  }

  /** Forget the owner (a source stopped). @param {string} [src] only if it is this one */
  release(src) {
    if (src === undefined || this.owner === src) this.owner = null;
  }
}
