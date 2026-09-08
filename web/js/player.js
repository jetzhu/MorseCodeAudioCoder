/**
 * Morse keying timing and Web Audio tone playback for the encoder strip.
 *
 * `buildTiming` is the port of `morse/player.py::build_timing`, which follows
 * `tools/beep_sender.py::build_timing` exactly: dit `1200 / wpm` ms, dah 3
 * dits, intra-letter gap 1, letter gap 3, word gap 7; unknown characters
 * skipped; no leading or trailing gap and never two adjacent gaps. Every
 * duration is `round(k * 1200 / wpm)` to whole milliseconds with Python's
 * half-to-even rounding (see {@link roundHalfEven}), so what Play keys through
 * the speakers is, millisecond for millisecond, what the desktop beeper sends
 * and what the Python player renders; the decoder is exercised on the same
 * signal either way.
 *
 * `TonePlayer` keys a sine `OscillatorNode` through a `GainNode` with 3 ms
 * linear ramps at every edge (no clicks) and reports a playhead position
 * through `onProgress`. It touches Web Audio only inside its methods, so this
 * module imports in Node; the constructor just stores the `AudioContext`.
 *
 * ES module. `buildTiming` and `buildGuide` run anywhere; `TonePlayer.play`
 * needs a Web Audio `AudioContext` (or a test double with the same shape).
 */

import { MORSE_TABLE } from "./table.js";

/** @typedef {{on: boolean, ms: number}} Segment one keyed mark or gap */

/** @typedef {{ch: string, code: string, startMs: number, endMs: number}} LetterSpan */

/**
 * Dit length in milliseconds for a keying speed.
 * @param {number} wpm words per minute, positive
 * @returns {number}
 * @throws {RangeError} when `wpm` is not positive
 */
export function ditMs(wpm) {
  if (!(wpm > 0) || !Number.isFinite(wpm)) throw new RangeError(`wpm must be positive, got ${wpm}`);
  return 1200.0 / wpm;
}

/**
 * Round to the nearest integer, ties to even: Python's built-in `round()`.
 *
 * `Math.round` rounds ties up, so `Math.round(112.5)` is 113 where Python
 * gives 112; the dah at 32 WPM is exactly that case. Using this keeps
 * {@link buildTiming} identical to `morse/player.py` at every speed.
 *
 * @param {number} x finite number
 * @returns {number}
 */
export function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Keying sequence and letter spans for `text` at `wpm`.
 *
 * The letter spans place each encoded character on the timeline (start of its
 * first mark to end of its last mark) so a keying guide can label them, as the
 * mock's encoder strip does. Characters without a Morse code are skipped and
 * get no span. `totalMs` is the sum of every segment. Durations follow
 * {@link buildTiming}: whole milliseconds, each rounded from the unrounded dit
 * (`round(3 * dit)`, not `3 * round(dit)`), exactly as the Python player does.
 *
 * @param {string} text plain text; whitespace of any kind separates words
 * @param {number} wpm words per minute, positive
 * @returns {{timing: Segment[], letters: LetterSpan[], totalMs: number}}
 * @throws {RangeError} when `wpm` is not positive
 */
export function buildGuide(text, wpm) {
  const rawDit = ditMs(wpm);
  const dit = roundHalfEven(rawDit);
  const dah = roundHalfEven(3 * rawDit);
  const letterGap = roundHalfEven(3 * rawDit);
  const wordGap = roundHalfEven(7 * rawDit);

  /** @type {{ch: string, code: string}[][]} */
  const words = [];
  for (const word of String(text).toUpperCase().split(/\s+/)) {
    if (!word) continue;
    const codes = [];
    for (const ch of word) {
      if (Object.hasOwn(MORSE_TABLE, ch)) codes.push({ ch, code: MORSE_TABLE[ch] });
    }
    if (codes.length) words.push(codes);
  }

  /** @type {Segment[]} */
  const timing = [];
  /** @type {LetterSpan[]} */
  const letters = [];
  let t = 0;
  const push = (on, ms) => {
    timing.push({ on, ms });
    t += ms;
  };
  words.forEach((codes, wi) => {
    if (wi > 0) push(false, wordGap);
    codes.forEach(({ ch, code }, li) => {
      if (li > 0) push(false, letterGap);
      const startMs = t;
      [...code].forEach((symbol, si) => {
        if (si > 0) push(false, dit);
        push(true, symbol === "-" ? dah : dit);
      });
      letters.push({ ch, code, startMs, endMs: t });
    });
  });
  return { timing, letters, totalMs: t };
}

/**
 * Return the keying sequence for `text` as `[{on, ms}, ...]`.
 *
 * Same rule as `build_timing` in `morse/player.py` and `tools/beep_sender.py`:
 * case-insensitive; characters without a Morse code are skipped (a word left
 * empty by that is skipped too); whitespace of any kind and length separates
 * words. Durations are `round(k * 1200 / wpm)` ms for `k` in 1 (dit,
 * intra-letter gap), 3 (dah, letter gap) and 7 (word gap), rounded half to
 * even like Python, so at 10 WPM a dit is 120 ms, a dah 360 ms and a word gap
 * 840 ms, and at 7 WPM a dit is 171 ms (171.43) and a dah 514 ms (514.29).
 * The sequence starts with the first mark and ends with the last mark: no
 * leading or trailing gap, never two adjacent gaps, and `[]` when nothing in
 * `text` is encodable. `"E E"` at 10 WPM gives `[on 120, off 840, on 120]`.
 *
 * @param {string} text plain text
 * @param {number} wpm words per minute, positive
 * @returns {Segment[]}
 * @throws {RangeError} when `wpm` is not positive
 */
export function buildTiming(text, wpm) {
  return buildGuide(text, wpm).timing;
}

/**
 * Total length of a keying sequence in milliseconds.
 * @param {Iterable<Segment>} timing
 * @returns {number}
 */
export function timingDurationMs(timing) {
  let total = 0;
  for (const seg of timing) total += seg.ms;
  return total;
}

/**
 * Keys a sine tone through the speakers following a `buildTiming` sequence.
 *
 * Usage: `const p = new TonePlayer(new AudioContext()); p.onProgress = (ms) =>
 * drawPlayhead(ms); p.onEnd = () => resetButton(); p.play(timing, f0);`.
 *
 * - `play(timing, f0, gain)` schedules every edge on the AudioContext clock
 *   (`startTime` is the first mark's context time, `leadMs` after the call)
 *   and starts a progress loop (`requestAnimationFrame` when present, else
 *   `setTimeout`) that calls `onProgress(elapsedMs)` with the playhead
 *   position clamped to `0..durationMs`, then `onEnd()` once the sequence is
 *   over. Calling `play` while playing restarts silently (no `onEnd`); calling
 *   it with an empty sequence only stops what is playing (with `onEnd`).
 * - `stop()` cuts the tone at once and calls `onEnd()`; it is a no-op when
 *   nothing is playing.
 */
export class TonePlayer {
  /**
   * @param {AudioContext | null} audioContext created by the caller from a
   *   user gesture; may be `null` until the first `play` if set later via
   *   {@link TonePlayer#audioContext}
   * @param {object} [options]
   * @param {number} [options.rampMs=3] linear ramp length at each edge
   * @param {number} [options.leadMs=80] scheduling lead before the first mark
   */
  constructor(audioContext, { rampMs = 3, leadMs = 80 } = {}) {
    /** @type {AudioContext | null} */
    this.audioContext = audioContext ?? null;
    /** @type {number} */
    this.rampMs = rampMs;
    /** @type {number} */
    this.leadMs = leadMs;
    /** @type {((elapsedMs: number) => void) | null} playhead callback */
    this.onProgress = null;
    /** @type {(() => void) | null} called once when playback ends or is stopped */
    this.onEnd = null;
    /** @type {number} AudioContext time (s) of the first mark of the current playback */
    this.startTime = 0;
    /** @type {number} total length (ms) of the current playback */
    this.durationMs = 0;

    /** @type {OscillatorNode | null} */
    this._osc = null;
    /** @type {GainNode | null} */
    this._gain = null;
    /** @type {boolean} */
    this._playing = false;
    /** @type {number | null} */
    this._tickId = null;
    /** @type {"raf" | "timeout" | null} */
    this._tickKind = null;
  }

  /** `true` between `play` and the end of the sequence or `stop`. @returns {boolean} */
  get playing() {
    return this._playing;
  }

  /** Playhead position in ms, clamped to `0..durationMs`; 0 when idle. @returns {number} */
  get elapsedMs() {
    if (!this._playing || !this.audioContext) return 0;
    const elapsed = (this.audioContext.currentTime - this.startTime) * 1000;
    return Math.min(this.durationMs, Math.max(0, elapsed));
  }

  /**
   * Key `timing` at frequency `f0`.
   *
   * @param {Iterable<Segment>} timing from {@link buildTiming}
   * @param {number} f0 tone frequency in Hz, `0 < f0 < sampleRate / 2`
   * @param {number} [gain=0.15] peak gain, `0..1`
   * @returns {number} the scheduled length in ms (0 when `timing` is empty)
   * @throws {Error} without an AudioContext; {@link RangeError} on a bad `f0` or `gain`
   */
  play(timing, f0, gain = 0.15) {
    const ac = this.audioContext;
    if (!ac) throw new Error("TonePlayer needs an AudioContext");
    if (!(f0 > 0) || (Number.isFinite(ac.sampleRate) && f0 >= ac.sampleRate / 2)) {
      throw new RangeError(`f0 must be between 0 and sampleRate/2, got ${f0}`);
    }
    if (!(gain >= 0 && gain <= 1)) throw new RangeError(`gain must be in 0..1, got ${gain}`);

    const segments = Array.from(timing);
    const totalMs = timingDurationMs(segments);
    if (!segments.length || !(totalMs > 0)) {
      this.stop(); // like the Python player: an empty sequence only stops what is playing
      return 0;
    }
    if (this._playing) this._teardown(false);

    if (ac.state === "suspended" && typeof ac.resume === "function") {
      const resumed = ac.resume();
      if (resumed && typeof resumed.catch === "function") resumed.catch(() => {});
    }

    const osc = ac.createOscillator();
    osc.type = "sine";
    osc.frequency.value = f0;
    const gainNode = ac.createGain();
    gainNode.gain.setValueAtTime(0, ac.currentTime);
    osc.connect(gainNode).connect(ac.destination);

    const t0 = ac.currentTime + this.leadMs / 1000;
    let t = t0;
    for (const seg of segments) {
      const d = seg.ms / 1000;
      if (seg.on) {
        const r = Math.min(this.rampMs / 1000, d / 2);
        gainNode.gain.setValueAtTime(0, t);
        gainNode.gain.linearRampToValueAtTime(gain, t + r);
        gainNode.gain.setValueAtTime(gain, t + d - r);
        gainNode.gain.linearRampToValueAtTime(0, t + d);
      }
      t += d;
    }
    osc.start(t0);
    osc.stop(t + 0.1);
    osc.onended = () => {
      if (this._osc === osc) this._teardown(true);
    };

    this._osc = osc;
    this._gain = gainNode;
    this.startTime = t0;
    this.durationMs = totalMs;
    this._playing = true;
    this._scheduleTick();
    return totalMs;
  }

  /** Stop at once; calls `onEnd` if something was playing. */
  stop() {
    if (this._playing) this._teardown(true);
  }

  // --------------------------------------------------------------- private

  /** @param {boolean} fireEnd */
  _teardown(fireEnd) {
    this._cancelTick();
    const osc = this._osc;
    const gainNode = this._gain;
    this._osc = null;
    this._gain = null;
    this._playing = false;
    if (osc) {
      osc.onended = null;
      try {
        osc.stop();
      } catch {
        /* already stopped */
      }
      try {
        osc.disconnect();
      } catch {
        /* never connected */
      }
    }
    if (gainNode) {
      try {
        gainNode.disconnect();
      } catch {
        /* never connected */
      }
    }
    if (fireEnd && typeof this.onEnd === "function") this.onEnd();
  }

  _scheduleTick() {
    const tick = () => {
      this._tickId = null;
      this._tick();
    };
    if (typeof globalThis.requestAnimationFrame === "function") {
      this._tickKind = "raf";
      this._tickId = globalThis.requestAnimationFrame(tick);
    } else {
      this._tickKind = "timeout";
      this._tickId = setTimeout(tick, 16);
    }
  }

  _cancelTick() {
    if (this._tickId === null) return;
    if (this._tickKind === "raf" && typeof globalThis.cancelAnimationFrame === "function") {
      globalThis.cancelAnimationFrame(this._tickId);
    } else {
      clearTimeout(this._tickId);
    }
    this._tickId = null;
    this._tickKind = null;
  }

  _tick() {
    if (!this._playing || !this.audioContext) return;
    const raw = (this.audioContext.currentTime - this.startTime) * 1000;
    const elapsed = Math.min(this.durationMs, Math.max(0, raw));
    if (typeof this.onProgress === "function") this.onProgress(elapsed);
    if (raw >= this.durationMs) {
      this._teardown(true);
      return;
    }
    this._scheduleTick();
  }
}
