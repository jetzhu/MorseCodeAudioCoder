/**
 * A Morse key the operator works by hand: straight key or two-paddle keyer,
 * and the Web Audio tone it drives.
 *
 * `Keyer` is the pure timing state machine, a line-by-line port of
 * `morse/keyer.py` and tested in Node. Every method takes the current time
 * in milliseconds and returns the tone transitions `[tMs, on]` it decided on;
 * some lie in the future (the end of a paddle element is known the moment it
 * starts). A bounded log of every transition lets callers read the tone state
 * at any time or decode what was sent.
 *
 * - `straight`: the tone follows the key; `keyDown` on, `keyUp` off.
 * - `paddle`: an electronic keyer. The dit paddle sends one dit of the
 *   configured length and keeps sending dits, one dit gap apart, while held;
 *   the dah paddle likewise with dahs. Holding both alternates (iambic). A
 *   paddle tapped during an element is remembered and sent after it, so a
 *   quick "dit, dah" gives a clean A. Releasing during an element never cuts
 *   it short.
 *
 * `LiveKey` plays the keyer's tone through a persistent oscillator whose gain
 * is automated on the AudioContext clock (3 ms exponential edges, no clicks);
 * extra outputs (the decoder's input bus) receive the same signal, exactly
 * like `TonePlayer`. Web Audio is touched only inside `LiveKey` methods, so
 * the module imports in Node.
 */

const MODES = new Set(["straight", "paddle"]);
const PADDLES = new Set(["dit", "dah"]);
const LOG_MAX = 4000;
const MAX_ELEMENTS_PER_TICK = 64;

export class Keyer {
  /**
   * @param {number} [ditMs=150] dit length for paddle elements (`1200 / wpm`)
   * @param {"straight" | "paddle"} [mode="straight"]
   */
  constructor(ditMs = 150, mode = "straight") {
    if (!(ditMs > 0)) throw new RangeError(`ditMs must be positive, got ${ditMs}`);
    if (!MODES.has(mode)) throw new RangeError(`mode must be 'straight' or 'paddle', got ${mode}`);
    this.ditMs = ditMs;
    this.mode = mode;
    this._held = { dit: false, dah: false };
    /** @type {string[]} */
    this._memory = [];
    this._phase = "idle";
    /** @type {string | null} */
    this._current = null;
    this._elementEnd = 0;
    this._gapEnd = 0;
    /** @type {Array<[number, boolean]>} */
    this._log = [];
    this._count = 0;
    this._lastT = -Infinity;
  }

  // ---------------------------------------------------------------- settings

  /** @param {number} ditMs */
  setSpeed(ditMs) {
    if (!(ditMs > 0)) throw new RangeError(`ditMs must be positive, got ${ditMs}`);
    this.ditMs = ditMs;
  }

  /**
   * Switch mode at time `t`; releases everything and ends a sounding tone.
   * @param {"straight" | "paddle"} mode @param {number} t @returns {Array<[number, boolean]>}
   */
  setMode(mode, t) {
    if (!MODES.has(mode)) throw new RangeError(`mode must be 'straight' or 'paddle', got ${mode}`);
    const out = this.releaseAll(t);
    this.mode = mode;
    return out;
  }

  /** Let go of every key and paddle at `t`: the tone stops at once. @param {number} t */
  releaseAll(t) {
    this._held = { dit: false, dah: false };
    this._memory.length = 0;
    // An element's OFF edge is logged the moment it starts; releasing earlier
    // must cut it short, so drop the edges that lie ahead of t.
    while (this._log.length && this._log[this._log.length - 1][0] > t) {
      this._log.pop();
      this._count -= 1;
    }
    this._lastT = this._log.length ? this._log[this._log.length - 1][0] : -Infinity;
    const out = [];
    if (this.stateAt(t)) out.push(this._emit(t, false));
    this._phase = "idle";
    this._current = null;
    return out;
  }

  // ------------------------------------------------------------ straight key

  /** @param {number} t @returns {Array<[number, boolean]>} */
  keyDown(t) {
    if (this.mode !== "straight" || this._phase === "mark") return [];
    this._phase = "mark";
    return [this._emit(t, true)];
  }

  /** @param {number} t @returns {Array<[number, boolean]>} */
  keyUp(t) {
    if (this.mode !== "straight" || this._phase !== "mark") return [];
    this._phase = "idle";
    return [this._emit(t, false)];
  }

  // ----------------------------------------------------------------- paddles

  /** @param {"dit" | "dah"} which @param {number} t @returns {Array<[number, boolean]>} */
  paddleDown(which, t) {
    if (!PADDLES.has(which)) throw new RangeError(`which must be 'dit' or 'dah', got ${which}`);
    if (this.mode !== "paddle") return [];
    this._held[which] = true;
    if (this._phase === "idle") return this._startElement(which, t);
    if (which !== this._current && !this._memory.includes(which)) this._memory.push(which);
    return [];
  }

  /** @param {"dit" | "dah"} which @param {number} t @returns {Array<[number, boolean]>} */
  paddleUp(which, t) {
    if (!PADDLES.has(which)) throw new RangeError(`which must be 'dit' or 'dah', got ${which}`);
    void t;
    this._held[which] = false;
    return [];
  }

  /**
   * Advance the paddle keyer to time `t`; returns the transitions of any new
   * elements. Call it at `nextWakeupMs` or more often.
   * @param {number} t @returns {Array<[number, boolean]>}
   */
  tick(t) {
    const out = [];
    if (this.mode !== "paddle") return out;
    for (let i = 0; i < MAX_ELEMENTS_PER_TICK; i++) {
      if (this._phase === "mark" && t >= this._elementEnd) this._phase = "gap";
      if (this._phase === "gap" && t >= this._gapEnd) {
        const next = this._nextElement();
        if (next === null) {
          this._phase = "idle";
          this._current = null;
          return out;
        }
        out.push(...this._startElement(next, this._gapEnd));
        continue;
      }
      return out;
    }
    return out;
  }

  /** When `tick` next has a decision to make; `null` while idle or in straight mode. @returns {number | null} */
  get nextWakeupMs() {
    if (this.mode !== "paddle") return null;
    if (this._phase === "mark") return this._elementEnd;
    if (this._phase === "gap") return this._gapEnd;
    return null;
  }

  // --------------------------------------------------------------------- log

  /** Tone state at time `t` according to the transitions emitted so far. @param {number} t */
  stateAt(t) {
    let state = false;
    for (const [tt, on] of this._log) {
      if (tt > t) break;
      state = on;
    }
    return state;
  }

  /** Transitions ever emitted. @returns {number} */
  get transitionCount() {
    return this._count;
  }

  /**
   * Transitions emitted after the `index`-th; returns them and the new index.
   * @param {number} index @returns {{items: Array<[number, boolean]>, index: number}}
   */
  transitionsSince(index) {
    const dropped = this._count - this._log.length;
    const start = Math.max(index - dropped, 0);
    return { items: this._log.slice(start), index: this._count };
  }

  // --------------------------------------------------------------- internals

  _emit(t, on) {
    const tt = Math.max(Number(t), this._lastT); // the log is never out of order
    this._lastT = tt;
    const tr = /** @type {[number, boolean]} */ ([tt, on]);
    this._log.push(tr);
    if (this._log.length > LOG_MAX) this._log.shift();
    this._count += 1;
    return tr;
  }

  _startElement(which, t) {
    const length = which === "dit" ? this.ditMs : 3 * this.ditMs;
    this._current = which;
    this._phase = "mark";
    this._elementEnd = t + length;
    this._gapEnd = this._elementEnd + this.ditMs;
    const i = this._memory.indexOf(which);
    if (i >= 0) this._memory.splice(i, 1);
    return [this._emit(t, true), this._emit(this._elementEnd, false)];
  }

  _nextElement() {
    if (this._memory.length) return this._memory.shift();
    const { dit, dah } = this._held;
    if (dit && dah) return this._current === "dit" ? "dah" : "dit";
    if (dit) return "dit";
    if (dah) return "dah";
    return null;
  }
}

/** Scheduling lead so a transition is never in the past for the audio thread. */
const LEAD_MS = 5;
/** Time constant of the exponential edges (about 3 ms to settle). */
const EDGE_TAU_S = 0.0012;

/**
 * Connect `from` to `to`, ignoring nodes that refuse.
 * @param {AudioNode} from @param {AudioNode} to
 */
function safeConnect(from, to) {
  try {
    from.connect(to);
  } catch {
    /* leave the rest of the graph alone */
  }
}

export class LiveKey {
  /**
   * @param {Keyer} keyer
   * @param {object} [options]
   * @param {number} [options.gain=0.15] peak gain, 0..1
   */
  constructor(keyer, { gain = 0.15 } = {}) {
    this.keyer = keyer;
    this.gain = gain;
    /** @type {AudioContext | null} */
    this.audioContext = null;
    this.f0 = 2491;
    /** Pitch the speakers play; null follows `f0`. The decoder feed is always at `f0`. @type {number | null} */
    this.sidetoneHz = null;
    /** Speaker chain (sidetone). @type {OscillatorNode | null} */
    this._osc = null;
    /** @type {GainNode | null} */
    this._gain = null;
    /** Feed chain (f0), for the extra outputs. @type {OscillatorNode | null} */
    this._oscFeed = null;
    /** @type {GainNode | null} */
    this._gainFeed = null;
    /** @type {Set<AudioNode>} */
    this._outputs = new Set();
    /** @type {ReturnType<typeof setTimeout> | null} */
    this._timer = null;
    /** @type {(() => void) | null} called after every scheduled change, for the UI */
    this.onChange = null;
  }

  /** `true` between `start` and `stop`. */
  get running() {
    return this._osc !== null;
  }

  /** The keyer clock: AudioContext time in milliseconds. */
  get nowMs() {
    return this.audioContext ? this.audioContext.currentTime * 1000 : 0;
  }

  /** The pitch the speakers play now: the sidetone, or `f0` without one. */
  get speakerHz() {
    return this.sidetoneHz === null ? this.f0 : this.sidetoneHz;
  }

  /**
   * Create the silent oscillators; the tone sounds only while keyed. The
   * first chain plays the sidetone through the speakers, the second renders
   * `f0` for the extra outputs (the decoder feed).
   * @param {AudioContext} audioContext @param {number} f0
   */
  start(audioContext, f0) {
    if (this._osc) return;
    this.audioContext = audioContext;
    this.f0 = f0;
    const make = (hz) => {
      const osc = audioContext.createOscillator();
      osc.type = "sine";
      osc.frequency.value = hz;
      const g = audioContext.createGain();
      g.gain.value = 0;
      osc.connect(g);
      osc.start();
      return [osc, g];
    };
    const [osc, g] = make(this.speakerHz);
    g.connect(audioContext.destination);
    const [oscFeed, gFeed] = make(f0);
    for (const node of this._outputs) safeConnect(gFeed, node);
    this._osc = osc;
    this._gain = g;
    this._oscFeed = oscFeed;
    this._gainFeed = gFeed;
  }

  /** Silence and tear down; the keyer forgets everything held. */
  stop() {
    this._cancelTimer();
    if (this.audioContext) this.keyer.releaseAll(this.nowMs);
    const nodes = [this._osc, this._gain, this._oscFeed, this._gainFeed];
    this._osc = null;
    this._gain = null;
    this._oscFeed = null;
    this._gainFeed = null;
    for (const node of nodes) {
      if (!node) continue;
      if (typeof node.stop === "function") {
        try {
          node.stop();
        } catch {
          /* already stopped */
        }
      }
      try {
        node.disconnect();
      } catch {
        /* never connected */
      }
    }
  }

  // ---------------------------------------------------------------- controls

  keyDown() {
    this._apply(this.keyer.keyDown(this.nowMs + LEAD_MS));
  }

  keyUp() {
    this._apply(this.keyer.keyUp(this.nowMs + LEAD_MS));
  }

  /** @param {"dit" | "dah"} which */
  paddleDown(which) {
    this._apply(this.keyer.paddleDown(which, this.nowMs + LEAD_MS));
  }

  /** @param {"dit" | "dah"} which */
  paddleUp(which) {
    this._apply(this.keyer.paddleUp(which, this.nowMs + LEAD_MS));
  }

  releaseAll() {
    this._apply(this.keyer.releaseAll(this.nowMs + LEAD_MS));
  }

  /** @param {"straight" | "paddle"} mode */
  setMode(mode) {
    this._apply(this.keyer.setMode(mode, this.nowMs + LEAD_MS));
  }

  /** @param {number} ditMs */
  setSpeed(ditMs) {
    this.keyer.setSpeed(ditMs);
  }

  /** Retune the decoder feed (and the speakers when no sidetone is set). @param {number} f0 */
  setFrequency(f0) {
    this.f0 = f0;
    if (!this.audioContext) return;
    const now = this.audioContext.currentTime;
    if (this._oscFeed) this._oscFeed.frequency.setValueAtTime(f0, now);
    if (this._osc && this.sidetoneHz === null) this._osc.frequency.setValueAtTime(f0, now);
  }

  /**
   * Pitch the speakers play; null, 0 or a non-finite value follows `f0`.
   * @param {number | null} hz
   */
  setSidetone(hz) {
    this.sidetoneHz = Number.isFinite(hz) && hz > 0 ? Number(hz) : null;
    if (this._osc && this.audioContext) this._osc.frequency.setValueAtTime(this.speakerHz, this.audioContext.currentTime);
  }

  /** Whether the tone is sounding now. */
  isOn() {
    return this.audioContext ? this.keyer.stateAt(this.nowMs) : false;
  }

  // ----------------------------------------------------------------- outputs

  /** Also send the tone, at `f0`, to `node` (the decoder's input bus). @param {AudioNode} node */
  addOutput(node) {
    if (!node || this._outputs.has(node)) return;
    this._outputs.add(node);
    if (this._gainFeed) safeConnect(this._gainFeed, node);
  }

  /** @param {AudioNode} node */
  removeOutput(node) {
    if (!this._outputs.delete(node)) return;
    if (this._gainFeed) {
      try {
        this._gainFeed.disconnect(node);
      } catch {
        /* not connected */
      }
    }
  }

  /** @returns {AudioNode[]} */
  get outputs() {
    return Array.from(this._outputs);
  }

  // --------------------------------------------------------------- internals

  /** Schedule transitions on the gain and arm the next wake-up. @param {Array<[number, boolean]>} transitions */
  _apply(transitions) {
    if (this._gain && this.audioContext) {
      const ac = this.audioContext;
      for (const [tMs, on] of transitions) {
        const at = Math.max(tMs / 1000, ac.currentTime);
        this._gain.gain.setTargetAtTime(on ? this.gain : 0, at, EDGE_TAU_S);
        if (this._gainFeed) this._gainFeed.gain.setTargetAtTime(on ? this.gain : 0, at, EDGE_TAU_S);
      }
    }
    this._arm();
    if (typeof this.onChange === "function") this.onChange();
  }

  _arm() {
    this._cancelTimer();
    const wake = this.keyer.nextWakeupMs;
    if (wake === null) return;
    const delay = Math.max(0, wake - this.nowMs) + 2;
    this._timer = setTimeout(() => {
      this._timer = null;
      this._apply(this.keyer.tick(this.nowMs + LEAD_MS));
    }, delay);
  }

  _cancelTimer() {
    if (this._timer !== null) {
      clearTimeout(this._timer);
      this._timer = null;
    }
  }
}
