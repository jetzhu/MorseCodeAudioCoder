/**
 * Adaptive ON/OFF tone detector, version 2: a port of `morse/tone_detector.py`.
 *
 * Feed one Goertzel power value (dB) per block.  The detector keeps two level
 * estimates in the *linear* power domain, a noise level `N` fed by OFF blocks
 * and a signal level `S` fed by ON blocks, places a hysteresis band above `N`
 * whose height is sized to the tracked signal-to-noise ratio, and turns the
 * block verdicts into runs of consecutive ON or OFF blocks.  Runs shorter than
 * `minRunBlocks` are folded into their neighbours, and a run is handed out
 * only once the run after it is long enough that no merge can change it.
 *
 * The rule, in full, is in `docs/INTERFACES.md` ("morse/tone_detector.py");
 * the Python module docstring explains why the levels are linear means and
 * why the noise alphas are equal.  This file follows the Python code
 * statement for statement so the two can be compared side by side.
 *
 * @module detector
 */

import { Run } from "./runs.js";

/** Added to linear power before `log10` (matches `Goertzel.powerDb`). */
const DB_FLOOR = 1e-12;

/** Noise-level smoothing used for every block during the warm-up. */
const WARMUP_ALPHA = 0.3;

/**
 * Constructor defaults, in the contract's order.  Exported so tests and the
 * UI can show them; do not mutate.
 * @type {Readonly<Record<string, number>>}
 */
export const DETECTOR_DEFAULTS = Object.freeze({
  blockMs: 10.0,
  onFrac: 0.6,
  offFrac: 0.4,
  minRunBlocks: 2,
  minLiftDb: 12.0,
  maxLiftDb: 30.0,
  minOffLiftDb: 6.0,
  noiseAlphaUp: 0.1,
  noiseAlphaDown: 0.1,
  signalAlpha: 0.1,
  signalDecayDb: 0.1,
  silenceFloorDb: -100.0,
  warmupBlocks: 30,
  maxOnBlocks: 500,
  releaseDb: 12.0,
  hysteresisDb: 3.0,
});

/**
 * Linear power to dB, `10*log10(x + 1e-12)`.
 * @param {number} power
 * @returns {number}
 */
function toDb(power) {
  return 10.0 * Math.log10(power + DB_FLOOR);
}

/**
 * dB to linear power, `10**(x/10)`.
 * @param {number} powerDb
 * @returns {number}
 */
function fromDb(powerDb) {
  return Math.pow(10.0, powerDb / 10.0);
}

/**
 * A mutable run under construction; converted to a `Run` when emitted.
 * @typedef {{on: boolean, blocks: number}} Segment
 */

/**
 * Turn a stream of per-block tone powers into debounced ON/OFF runs.
 *
 * Options (all optional, camelCase versions of the Python keywords):
 *
 * - `blockMs` duration of one block; copied into every emitted `Run`.
 * - `onFrac`, `offFrac` fractions of the tracked SNR that set the ON and OFF
 *   lifts above the noise level (`offFrac <= onFrac`).
 * - `minRunBlocks` a completed run shorter than this is merged into the runs
 *   on either side; a run becomes final once the run after it has reached
 *   this length.
 * - `minLiftDb`, `maxLiftDb` bounds of the ON lift.
 * - `minOffLiftDb` lower bound of the OFF lift (upper bound `maxLiftDb - 6`).
 * - `releaseDb`, `hysteresisDb` the OFF threshold is `max(N + liftOff,
 *   S - releaseDb)` and the ON threshold `max(N + liftOn, lo + hysteresisDb)`.
 * - `noiseAlphaUp`, `noiseAlphaDown` EMA coefficients for the noise level,
 *   applied to OFF blocks above and below the current level.
 * - `signalAlpha` EMA coefficient for the signal level, applied to ON blocks.
 * - `signalDecayDb` how far the signal level sinks toward the noise level per
 *   OFF block.
 * - `silenceFloorDb` the noise level is never tracked below this.
 * - `warmupBlocks` blocks after construction or `reset()` during which every
 *   verdict is OFF and the noise level is smoothed with alpha 0.3.
 * - `maxOnBlocks` an ON run that reaches this length is ended and the noise
 *   level is set to the signal level.
 *
 * Invalid values throw `RangeError` (the Python code raises `ValueError`);
 * an unknown option name or a non-numeric value throws `TypeError`.
 */
export class ToneDetector {
  /** @type {number} */ blockMs;
  /** @type {number} */ onFrac;
  /** @type {number} */ offFrac;
  /** @type {number} */ minRunBlocks;
  /** @type {number} */ minLiftDb;
  /** @type {number} */ maxLiftDb;
  /** @type {number} */ minOffLiftDb;
  /** @type {number} */ noiseAlphaUp;
  /** @type {number} */ noiseAlphaDown;
  /** @type {number} */ signalAlpha;
  /** @type {number} */ signalDecayDb;
  /** @type {number} */ silenceFloorDb;
  /** @type {number} */ warmupBlocks;
  /** @type {number} */ maxOnBlocks;
  /** @type {number} */ releaseDb;
  /** @type {number} */ hysteresisDb;

  #silenceFloor;
  #signalDecay;
  #primed = false;
  #state = false;
  #blocksSeen = 0;
  #n;
  #s;
  #hi = 0.0;
  #lo = 0.0;
  /**
   * Completed-but-not-yet-final runs followed by the in-progress run.  By
   * construction this never holds more than two segments.
   * @type {Segment[]}
   */
  #segments = [];

  /**
   * @param {Partial<typeof DETECTOR_DEFAULTS>} [options]
   */
  constructor(options = {}) {
    for (const key of Object.keys(options)) {
      if (!(key in DETECTOR_DEFAULTS)) {
        throw new TypeError("ToneDetector: unknown option " + key);
      }
    }
    const opt = { ...DETECTOR_DEFAULTS, ...options };
    for (const [key, value] of Object.entries(opt)) {
      if (typeof value !== "number" || Number.isNaN(value)) {
        throw new TypeError("ToneDetector: option " + key + " must be a number");
      }
    }

    if (opt.releaseDb < 0 || opt.hysteresisDb < 0) {
      throw new RangeError("releaseDb and hysteresisDb must be non-negative");
    }
    if (opt.blockMs <= 0) throw new RangeError("blockMs must be positive");
    if (!(0.0 <= opt.offFrac && opt.offFrac <= opt.onFrac && opt.onFrac <= 1.0)) {
      throw new RangeError("need 0 <= offFrac <= onFrac <= 1");
    }
    if (opt.minRunBlocks < 1) throw new RangeError("minRunBlocks must be at least 1");
    if (opt.minLiftDb < 0 || opt.minOffLiftDb < 0) {
      throw new RangeError("minLiftDb and minOffLiftDb must be non-negative");
    }
    if (opt.maxLiftDb < opt.minLiftDb) throw new RangeError("maxLiftDb must be at least minLiftDb");
    for (const name of ["noiseAlphaUp", "noiseAlphaDown", "signalAlpha"]) {
      const alpha = opt[name];
      if (!(0.0 < alpha && alpha <= 1.0)) throw new RangeError(name + " must be in (0, 1]");
    }
    if (opt.signalDecayDb < 0) throw new RangeError("signalDecayDb must be non-negative");
    if (opt.warmupBlocks < 0) throw new RangeError("warmupBlocks must be non-negative");
    if (opt.maxOnBlocks < 1) throw new RangeError("maxOnBlocks must be at least 1");

    this.blockMs = opt.blockMs;
    this.onFrac = opt.onFrac;
    this.offFrac = opt.offFrac;
    this.minRunBlocks = Math.trunc(opt.minRunBlocks);
    this.minLiftDb = opt.minLiftDb;
    this.maxLiftDb = opt.maxLiftDb;
    this.minOffLiftDb = opt.minOffLiftDb;
    this.noiseAlphaUp = opt.noiseAlphaUp;
    this.noiseAlphaDown = opt.noiseAlphaDown;
    this.signalAlpha = opt.signalAlpha;
    this.signalDecayDb = opt.signalDecayDb;
    this.silenceFloorDb = opt.silenceFloorDb;
    this.warmupBlocks = Math.trunc(opt.warmupBlocks);
    this.maxOnBlocks = Math.trunc(opt.maxOnBlocks);
    this.releaseDb = opt.releaseDb;
    this.hysteresisDb = opt.hysteresisDb;

    this.#silenceFloor = fromDb(this.silenceFloorDb);
    this.#signalDecay = fromDb(-this.signalDecayDb);
    this.#n = this.#silenceFloor;
    this.#s = this.#silenceFloor;
    this.#updateThresholds();
  }

  // ---------------------------------------------------------------- state

  /** Current raw ON/OFF verdict (before debouncing). */
  get state() {
    return this.#state;
  }

  /** True during the first `warmupBlocks` after construction or `reset()`. */
  get warmingUp() {
    return this.#blocksSeen < this.warmupBlocks;
  }

  /** Tracked noise level `N` (linear mean of OFF-block power) in dB. */
  get floorDb() {
    return toDb(this.#n);
  }

  /** Tracked signal level `S` (linear mean of ON-block power) in dB. */
  get peakDb() {
    return toDb(this.#s);
  }

  /** `max(N + liftOn, lo + hysteresisDb)`: power above which an OFF detector switches ON. */
  get thresholdHiDb() {
    return this.#hi;
  }

  /** `max(N + liftOff, S - releaseDb)`: power below which an ON detector switches OFF. */
  get thresholdLoDb() {
    return this.#lo;
  }

  /** The in-progress run (not yet final).  Zero blocks before any update. */
  get currentRun() {
    const segs = this.#segments;
    if (segs.length === 0) return new Run(this.#state, 0, this.blockMs);
    const last = segs[segs.length - 1];
    return new Run(last.on, last.blocks, this.blockMs);
  }

  // -------------------------------------------------------------- feeding

  /**
   * Feed one block's tone power in dB.
   *
   * Returns the runs that became final during this block, oldest first.
   * Usually empty; occasionally one run (or more if the debounce settings
   * allow several to complete at once).  A non-finite value is treated as
   * digital silence; anything that is not a number throws `TypeError`
   * (Python's `float()` would), so a `null` or `undefined` from a wiring
   * mistake cannot masquerade as a 0 dB tone.
   *
   * @param {number} powerDb
   * @returns {Run[]}
   */
  update(powerDb, tonal = true) {
    if (typeof powerDb !== "number") {
      throw new TypeError("ToneDetector.update: powerDb must be a number, got " + typeof powerDb);
    }
    let pDb = powerDb;
    if (Number.isNaN(pDb) || pDb === Infinity) pDb = -Infinity;
    const p = fromDb(pDb);

    if (!this.#primed) {
      this.#n = Math.max(p, this.#silenceFloor);
      this.#s = this.#n;
      this.#primed = true;
      this.#updateThresholds();
    }
    const warm = this.#blocksSeen < this.warmupBlocks;
    this.#blocksSeen += 1;

    // Verdict against the thresholds of the levels as they stand.
    let on;
    if (warm) {
      on = false;
    } else {
      on = this.#state;
      if (on) {
        if (pDb < this.#lo) on = false;
      } else if (pDb > this.#hi && tonal) {
        // A block that is not tonal (a click, speech) never switches an OFF
        // detector ON, but it still teaches the noise level below.
        on = true;
      }
      if (on && this.#segments.length > 0) {
        const current = this.#segments[this.#segments.length - 1];
        if (current.on && current.blocks >= this.maxOnBlocks) {
          // Stuck ON: no Morse mark lasts this long, so what the signal
          // tracker learned was the noise level.
          on = false;
          this.#n = this.#s;
        }
      }
    }

    // Level update: ON blocks feed S, OFF blocks feed N and let S sink.
    if (on) {
      this.#s += this.signalAlpha * (p - this.#s);
    } else {
      let alpha;
      if (warm) alpha = WARMUP_ALPHA;
      else if (p < this.#n) alpha = this.noiseAlphaDown;
      else alpha = this.noiseAlphaUp;
      this.#n = Math.max(this.#n + alpha * (p - this.#n), this.#silenceFloor);
      this.#s = Math.max(this.#s * this.#signalDecay, this.#n);
    }
    this.#state = on;
    this.#updateThresholds();

    const segs = this.#segments;
    if (segs.length === 0) {
      segs.push({ on, blocks: 1 });
    } else if (segs[segs.length - 1].on === on) {
      segs[segs.length - 1].blocks += 1;
    } else {
      const done = segs[segs.length - 1];
      if (done.blocks < this.minRunBlocks) {
        // A glitch: fold it, together with this block, into the run before
        // it (which has the same state as this block because runs
        // alternate).  With no run before it, it simply joins the run that
        // is starting now.
        if (segs.length >= 2) {
          segs.pop();
          segs[segs.length - 1].blocks += done.blocks + 1;
        } else {
          done.on = on;
          done.blocks += 1;
        }
      } else {
        segs.push({ on, blocks: 1 });
      }
    }

    if (segs.length > 1 && segs[segs.length - 1].blocks >= this.minRunBlocks) {
      // The current run can no longer be merged away, so everything before
      // it has its final length.
      const out = segs.slice(0, -1).map((seg) => this.#toRun(seg));
      segs.splice(0, segs.length - 1);
      return out;
    }
    return [];
  }

  /**
   * Emit everything pending, including the in-progress run.
   *
   * The tracked levels, the verdict and the warm-up state are kept so that a
   * stream can continue; only the run bookkeeping starts over.
   *
   * @returns {Run[]}
   */
  flush() {
    const out = this.#segments.map((seg) => this.#toRun(seg));
    this.#segments.length = 0;
    return out;
  }

  /** Forget everything: levels, verdict, pending runs; restart the warm-up. */
  reset() {
    this.#primed = false;
    this.#state = false;
    this.#blocksSeen = 0;
    this.#n = this.#silenceFloor;
    this.#s = this.#silenceFloor;
    this.#segments.length = 0;
    this.#updateThresholds();
  }

  // ------------------------------------------------------------ internals

  /**
   * @param {Segment} seg
   * @returns {Run}
   */
  #toRun(seg) {
    return new Run(seg.on, seg.blocks, this.blockMs);
  }

  #updateThresholds() {
    // Noise-referenced lifts protect against noise spikes at low SNR; the
    // signal-referenced release (S - releaseDb) lets go of a loud tone as
    // soon as its room tail has dropped releaseDb, instead of waiting for
    // the tail to fall all the way to the noise-referenced lift.  hi always
    // sits hysteresisDb above lo.
    const floor = toDb(this.#n);
    const peak = toDb(this.#s);
    const snr = peak - floor;
    const liftOn = Math.min(this.maxLiftDb, Math.max(this.minLiftDb, this.onFrac * snr));
    const liftOff = Math.min(this.maxLiftDb - 6.0, Math.max(this.minOffLiftDb, this.offFrac * snr));
    const lo = Math.max(floor + liftOff, peak - this.releaseDb);
    this.#lo = lo;
    this.#hi = Math.max(floor + liftOn, lo + this.hysteresisDb);
  }

  /** @returns {string} */
  toString() {
    const run = this.currentRun;
    return (
      "ToneDetector(state=" + (this.#state ? "ON" : "off") +
      ", floor=" + this.floorDb.toFixed(1) + " dB, peak=" + this.peakDb.toFixed(1) + " dB" +
      ", lo=" + this.#lo.toFixed(1) + " dB, hi=" + this.#hi.toFixed(1) + " dB" +
      ", warmingUp=" + this.warmingUp +
      ", current=" + (run.on ? "ON" : "off") + " " + run.blocks + " blocks)"
    );
  }
}

/** Blocks whose tone power sits further below the block's total power than this are broadband. */
export const TONALITY_MIN_DB = -15.0;

/** A pure sine's normalised Goertzel power is 3 dB above its mean square (A^2 vs A^2/2). */
const PURE_TONE_OFFSET_DB = 3.0103;

/**
 * How much of a block's energy sits in the tone bin, in dB; 0 for a pure tone at `f0`.
 *
 * `powerDb` is the normalised Goertzel power (0 dB = full-scale sine) and
 * `rmsDb` is `10*log10(mean square)` of the same block, as the worklet posts
 * them. White noise or a click spreads its energy over the whole band, so its
 * 100 Hz bin holds about 1/240 of it: around -24 dB here. A beeper
 * concentrates its energy in the bin: around 0 dB.
 *
 * @param {number} powerDb @param {number} rmsDb @returns {number}
 */
export function tonalityDb(powerDb, rmsDb) {
  return powerDb - rmsDb - PURE_TONE_OFFSET_DB;
}

/**
 * Whether a block is tone rather than noise: the `tonal` flag for
 * `ToneDetector.update`. Tonality: the tone bin must hold a fair share of the
 * block's energy (beeper near 0 dB, full-band noise near -24 dB, threshold
 * -15 dB). Peakiness, when `neighborDb` (mean power at f0 +-300 and +-600 Hz,
 * see `combineDb`) is given: the bin must stand `minPeakinessDb` above its
 * neighbours. Tonality alone fails on phones, whose band-limited microphones
 * concentrate a click's energy; noise of any bandwidth is about as strong
 * beside f0 as at it, a beeper is a sharp peak. A block that fails may teach
 * the detector the noise level but never switches it ON. Mirror of
 * `morse.tone_detector.is_tonal`.
 *
 * @param {number} powerDb @param {number} rmsDb
 * @param {number} [minTonalityDb=TONALITY_MIN_DB]
 * @param {number | null} [neighborDb=null] @param {number} [minPeakinessDb=PEAKINESS_MIN_DB]
 * @returns {boolean}
 */
export function isTonal(powerDb, rmsDb, minTonalityDb = TONALITY_MIN_DB, neighborDb = null, minPeakinessDb = PEAKINESS_MIN_DB) {
  if (tonalityDb(powerDb, rmsDb) < minTonalityDb) return false;
  if (neighborDb === null || neighborDb === undefined) return true;
  return powerDb - neighborDb >= minPeakinessDb;
}

/** Where the neighbour bands sit relative to `f0`: beyond the 10 ms block's main lobe (100 Hz). */
export const NEIGHBOR_OFFSETS_HZ = Object.freeze([-600, -300, 300, 600]);

/** A tone block must stand this far above the mean of its neighbour bands. */
export const PEAKINESS_MIN_DB = 8.0;

/** Neighbour-band frequencies for `f0`, leaving out any outside `0 < f < fs/2`. @param {number} f0 @param {number} fs */
export function neighborFrequencies(f0, fs) {
  return NEIGHBOR_OFFSETS_HZ.map((d) => f0 + d).filter((f) => f > 0 && f < fs / 2);
}

/** Mean power of several bands in dB (averaged in linear power). @param {number[]} valuesDb */
export function combineDb(valuesDb) {
  if (!valuesDb.length) return -Infinity;
  let sum = 0;
  for (const v of valuesDb) sum += 10 ** (v / 10);
  return 10 * Math.log10(sum / valuesDb.length);
}
