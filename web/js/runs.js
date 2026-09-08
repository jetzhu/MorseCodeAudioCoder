/**
 * Shared data type: one run of tone ON or OFF, measured in 10 ms blocks.
 *
 * Port of `morse/runs.py` (a frozen dataclass): instances are frozen after
 * construction. ES module; runs in the browser and in Node.
 */

/** A stretch of consecutive blocks with the same detector verdict. */
export class Run {
  /**
   * @param {boolean} on `true` when the tone is present
   * @param {number} blocks length in blocks
   * @param {number} [blockMs=10] block length in milliseconds
   */
  constructor(on, blocks, blockMs = 10) {
    /** @type {boolean} */
    this.on = Boolean(on);
    /** @type {number} */
    this.blocks = blocks;
    /** @type {number} */
    this.blockMs = blockMs;
    Object.freeze(this);
  }

  /** Length in milliseconds: `blocks * blockMs`. @returns {number} */
  get ms() {
    return this.blocks * this.blockMs;
  }

  /**
   * Build a `Run` from the `[on, blocks]` pair used by `web/test/vectors.json`.
   * @param {[boolean, number]} pair
   * @param {number} [blockMs=10]
   * @returns {Run}
   */
  static fromPair(pair, blockMs = 10) {
    return new Run(pair[0], pair[1], blockMs);
  }

  /** `"ON 110 ms"` / `"off 50 ms"`, like the Python `__str__`. @returns {string} */
  toString() {
    return `${this.on ? "ON" : "off"} ${Math.round(this.ms)} ms`;
  }
}
