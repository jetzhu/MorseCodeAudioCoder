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
 * Letter and word gap (whole ms) for characters at `wpm` and an overall
 * Farnsworth speed. Standard spacing (3 and 7 dits) without a Farnsworth
 * speed or when it is not below `wpm`; otherwise the ARRL rule: with
 * character speed c and overall speed s, `ta = (60c - 37.2s) / (s c)` seconds
 * and the gaps are `3 ta / 19` (letter) and `7 ta / 19` (word). At 18/5 that
 * is a 1568 ms letter gap and a 3660 ms word gap. Mirror of
 * `morse.player.farnsworth_gaps`.
 *
 * @param {number} wpm character speed, positive
 * @param {number | null} [farnsworthWpm=null] overall speed
 * @returns {{letterGap: number, wordGap: number}}
 * @throws {RangeError} when a speed is not positive
 */
export function farnsworthGaps(wpm, farnsworthWpm = null) {
  const rawDit = ditMs(wpm);
  if (farnsworthWpm === null || farnsworthWpm === undefined) {
    return { letterGap: roundHalfEven(3 * rawDit), wordGap: roundHalfEven(7 * rawDit) };
  }
  const s = Number(farnsworthWpm);
  if (!(s > 0) || !Number.isFinite(s)) throw new RangeError(`farnsworthWpm must be positive, got ${farnsworthWpm}`);
  const c = Number(wpm);
  if (s >= c) return { letterGap: roundHalfEven(3 * rawDit), wordGap: roundHalfEven(7 * rawDit) };
  const ta = (60 * c - 37.2 * s) / (s * c);
  return { letterGap: roundHalfEven((1000 * 3 * ta) / 19), wordGap: roundHalfEven((1000 * 7 * ta) / 19) };
}

/**
 * Keying sequence and letter spans for `text` at `wpm`, optionally with
 * Farnsworth spacing (`farnsworthWpm` below `wpm` stretches the letter and
 * word gaps, see {@link farnsworthGaps}).
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
 * @param {number | null} [farnsworthWpm=null] overall speed for Farnsworth spacing
 * @returns {{timing: Segment[], letters: LetterSpan[], totalMs: number}}
 * @throws {RangeError} when `wpm` is not positive
 */
export function buildGuide(text, wpm, farnsworthWpm = null) {
  const rawDit = ditMs(wpm);
  const dit = roundHalfEven(rawDit);
  const dah = roundHalfEven(3 * rawDit);
  const { letterGap, wordGap } = farnsworthGaps(wpm, farnsworthWpm);

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
export function buildTiming(text, wpm, farnsworthWpm = null) {
  return buildGuide(text, wpm, farnsworthWpm).timing;
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
/**
 * Connect `from` to `to`, ignoring nodes that refuse (already connected,
 * different context, torn down).
 * @param {AudioNode} from @param {AudioNode} to
 */
function safeConnect(from, to) {
  try {
    from.connect(to);
  } catch {
    /* leave the speakers path alone */
  }
}

/**
 * Place the letter labels of a keying guide so none overlap.
 *
 * Every letter gets a label centred on its span when there is room; a label
 * that would collide with the one drawn before it is skipped. Narrow letters
 * (a lone dit such as E) are placed like any other, since the neighbouring
 * letters sit at least a letter gap away. Pure function: the caller supplies
 * the horizontal scale (`xOf(ms)`) and the text measurer (`widthOf(ch)`).
 *
 * @param {Array<{ch: string, startMs: number, endMs: number}>} letters from {@link buildGuide}
 * @param {(ms: number) => number} xOf pixel position of a time
 * @param {(ch: string) => number} widthOf rendered width of a label
 * @param {number} [pad=3] minimum pixels between neighbouring labels
 * @returns {Array<{ch: string, x: number}>} label text and centre x, in order
 */
export function layoutGuideLabels(letters, xOf, widthOf, pad = 3) {
  const placed = [];
  let lastRight = -Infinity;
  for (const L of letters) {
    const cx = xOf((L.startMs + L.endMs) / 2);
    const half = widthOf(L.ch) / 2;
    if (cx - half < lastRight + pad) continue;
    placed.push({ ch: L.ch, x: cx });
    lastRight = cx + half;
  }
  return placed;
}

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

    /**
     * Extra destinations for the keyed tone besides the speakers: the
     * decoder's input bus, so Play can be decoded without going through the
     * room, the microphone and whatever the OS does to it.
     * @type {Set<AudioNode>}
     */
    this._outputs = new Set();
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
    osc.connect(gainNode);
    gainNode.connect(ac.destination);
    for (const node of this._outputs) safeConnect(gainNode, node);

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

  /**
   * Also send the keyed tone to `node` (for example the decoder's input bus).
   * Takes effect at once when something is playing and for every later `play`.
   *
   * @param {AudioNode} node
   */
  addOutput(node) {
    if (!node || this._outputs.has(node)) return;
    this._outputs.add(node);
    if (this._gain) safeConnect(this._gain, node);
  }

  /** Stop sending the tone to `node`. @param {AudioNode} node */
  removeOutput(node) {
    if (!this._outputs.delete(node)) return;
    if (this._gain) {
      try {
        this._gain.disconnect(node);
      } catch {
        /* not connected */
      }
    }
  }

  /** Drop every extra output; the speakers keep playing. */
  clearOutputs() {
    for (const node of Array.from(this._outputs)) this.removeOutput(node);
  }

  /** The extra outputs currently registered. @returns {AudioNode[]} */
  get outputs() {
    return Array.from(this._outputs);
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
