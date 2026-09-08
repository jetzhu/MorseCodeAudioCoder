/**
 * Signal-processing building blocks for the browser decoder.
 *
 * A line-by-line port of the Goertzel part of `morse/dsp.py`.  Pure
 * JavaScript, no DOM, no Web Audio: it runs in the browser (main thread or
 * AudioWorklet) and in Node so the parity tests can drive it.
 *
 * Power normalisation
 * -------------------
 * `goertzelPower` returns `4 * |X(f0)|**2 / N**2` where `X(f0)` is the DTFT of
 * the `N`-sample block evaluated at `f0` (what the Goertzel recursion
 * computes).  For a sine `A sin(w0 n + phi)` the image term vanishes exactly
 * on a DFT bin and is small off-bin, so `|X| ~= A N / 2` and the normalised
 * power is `A**2`: a full-scale sine gives 1.0 regardless of block length,
 * exactly on-bin and within a few percent off-bin.  Silence gives 0.0, which
 * `powerDb` reports as -120 dB.
 *
 * Single bin, deliberately: the neighbouring `f0 +- 50 Hz` bins are not
 * summed (see the Python module docstring for the block-length argument), so
 * this port and `tools/export_vectors.py` measure the same quantity.
 *
 * @module dsp
 */

/** Added to linear power before `log10` so silence reads -120 dB, not -Infinity. */
export const DB_FLOOR = 1e-12;

/**
 * `|X(w0)|**2` of `x` via the Goertzel recursion, `coeff = 2 cos(w0)`.
 *
 * The recursion `s[n] = x[n] + coeff*s[n-1] - s[n-2]` is evaluated in the
 * same operation order as scipy's transposed direct form II `lfilter`, so the
 * float64 results agree with the Python reference to the last bit for the
 * same input.  After the last sample `|X|**2 = s1**2 + s2**2 - coeff*s1*s2`.
 *
 * @param {ArrayLike<number>} x samples (any numeric array-like)
 * @param {number} coeff `2 * cos(2 * pi * f0 / fs)`
 * @returns {number}
 */
function goertzelMag2(x, coeff) {
  const n = x.length;
  if (n < 2) return 0.0;
  let s1 = 0.0;
  let s2 = 0.0;
  for (let i = 0; i < n; i++) {
    const s0 = x[i] + (coeff * s1 - s2);
    s2 = s1;
    s1 = s0;
  }
  return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

/**
 * Linear tone power at `f0` in `block`.
 *
 * A single Goertzel bin, `4 * |X(f0)|**2 / N**2`, normalised so that a
 * full-scale sine at `f0` returns 1.0 regardless of block length.  Blocks of
 * fewer than two samples return 0.0, as does silence.
 *
 * @param {ArrayLike<number>} block mono samples in -1..1 (Float32Array, Float64Array or plain array)
 * @param {number} f0 tone frequency in Hz
 * @param {number} fs sample rate in Hz
 * @returns {number} linear power
 */
export function goertzelPower(block, f0, fs) {
  const n = block.length;
  if (n < 2) return 0.0;
  const coeff = 2.0 * Math.cos((2.0 * Math.PI * f0) / fs);
  return (4.0 * goertzelMag2(block, coeff)) / (n * n);
}

/**
 * Single-frequency tone-power detector for fixed-size blocks.
 *
 * `power(block)` equals `goertzelPower(block, f0, fs)`; the class caches the
 * recursion coefficient and normalisation between calls.  Blocks of a length
 * other than `blockSize` are accepted and normalised by their actual length.
 */
export class Goertzel {
  /** @type {number} */
  fs;
  /** @type {number} */
  blockSize;
  #norm;
  #f0 = 0.0;
  #coeff = 0.0;

  /**
   * @param {number} f0 tone frequency in Hz
   * @param {number} fs sample rate in Hz (positive)
   * @param {number} blockSize nominal block length in samples (at least 2)
   */
  constructor(f0, fs, blockSize) {
    if (!(fs > 0)) throw new RangeError("fs must be positive");
    if (!(blockSize >= 2)) throw new RangeError("blockSize must be at least 2");
    this.fs = Math.trunc(fs);
    this.blockSize = Math.trunc(blockSize);
    this.#norm = 4.0 / (this.blockSize * this.blockSize);
    this.setFrequency(f0);
  }

  /** Tone frequency in Hz (change it with `setFrequency`). */
  get f0() {
    return this.#f0;
  }

  /**
   * Retune the detector to `f0` Hz.
   * @param {number} f0
   */
  setFrequency(f0) {
    this.#f0 = Number(f0);
    this.#coeff = 2.0 * Math.cos((2.0 * Math.PI * this.#f0) / this.fs);
  }

  /**
   * Linear power at `f0`, normalised so a full-scale sine gives 1.0.
   * @param {ArrayLike<number>} block
   * @returns {number}
   */
  power(block) {
    const n = block.length;
    if (n < 2) return 0.0;
    const norm = n === this.blockSize ? this.#norm : 4.0 / (n * n);
    return norm * goertzelMag2(block, this.#coeff);
  }

  /**
   * `10*log10(power + 1e-12)`: 0 dB for full scale, -120 dB for silence.
   * @param {ArrayLike<number>} block
   * @returns {number}
   */
  powerDb(block) {
    return 10.0 * Math.log10(this.power(block) + DB_FLOOR);
  }
}
