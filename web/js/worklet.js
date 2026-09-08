/**
 * AudioWorklet processor `block-meter`: per-block tone power and level.
 *
 * Runs on the audio rendering thread.  Render quanta (128 samples in current
 * browsers) are accumulated into blocks of `round(sampleRate / 100)` samples
 * (480 at 48 kHz = 10 ms, 441 at 44.1 kHz) and for every full block one
 * message is posted to the main thread:
 *
 *     { powerDb, rmsDb, blockIndex, skippedFrames, badBlocks }
 *
 * - `powerDb`: single-bin Goertzel power at `f0`,
 *   `10*log10(4*|X(f0)|^2 / N^2 + 1e-12)`, the normalisation of
 *   `morse/dsp.py` and `web/js/dsp.js`: a full-scale sine at `f0` reads 0 dB,
 *   digital silence -120 dB.  The recursion is inlined (same operation order
 *   as `goertzelMag2` in `dsp.js`, so the two agree bit for bit) because an
 *   AudioWorklet module cannot `import` other modules in every browser.
 * - `rmsDb`: `10*log10(mean(x^2) + 1e-12)`, the block level in dBFS
 *   (`Pipeline.level_dbfs` in Python).
 * - `blockIndex`: 0-based block counter, so the main thread can spot a lost
 *   message.
 * - `skippedFrames`: cumulative frames the render graph jumped over (a
 *   `currentFrame` discontinuity between two `process` calls); normally 0.
 * - `badBlocks`: cumulative blocks that contained NaN or inf and were
 *   replaced by zeros, like `Pipeline.bad_blocks`.
 *
 * Messages accepted on the port:
 * - `{ f0 }` retunes the Goertzel bin from the next block on.
 * - `{ dump: true }` replies with `{ dump: Float32Array, sampleRate }`: the
 *   last `ringSeconds` (default 30) of raw input in chronological order, for
 *   the Save 30 s download.  The reply buffer is transferred, not copied.
 *
 * Only channel 0 of the input is used, like `indata[:, 0]` in the Python
 * capture.  A quantum containing NaN or inf is replaced by zeros and the
 * block it lands in is measured as silence, which is what `Pipeline` does.
 * Nothing is written to the output; the node is connected to a muted gain in
 * `audio.js` so every browser keeps rendering it.
 */

/** Added to linear power before `log10` so silence reads -120 dB, not -Infinity. */
const DB_FLOOR = 1e-12;

/** Tone frequency used when the node is created without `processorOptions.f0`. */
const DEFAULT_F0 = 2491;

/** Seconds of raw input kept for the Save 30 s download. */
const DEFAULT_RING_SECONDS = 30;

class BlockMeter extends AudioWorkletProcessor {
  /**
   * @param {{processorOptions?: {f0?: number, ringSeconds?: number}}} [options]
   */
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    const fs = sampleRate;

    this.blockSize = Math.max(2, Math.round(fs / 100));
    this.acc = new Float32Array(this.blockSize);
    this.fill = 0;
    this.blockIndex = 0;
    this.badBlocks = 0;
    this.blockBad = false;

    this.f0 = 0;
    this.coeff = 0;
    this.setFrequency(Number(opts.f0) > 0 ? Number(opts.f0) : DEFAULT_F0);

    const ringSeconds = Number(opts.ringSeconds) > 0 ? Number(opts.ringSeconds) : DEFAULT_RING_SECONDS;
    this.ring = new Float32Array(Math.max(this.blockSize, Math.round(fs * ringSeconds)));
    this.ringPos = 0;
    this.ringFilled = 0;

    this.lastFrame = -1;
    this.lastLen = 0;
    this.skippedFrames = 0;

    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  /**
   * Retune the Goertzel bin.
   * @param {number} f0 tone frequency in Hz
   */
  setFrequency(f0) {
    this.f0 = f0;
    this.coeff = 2.0 * Math.cos((2.0 * Math.PI * f0) / sampleRate);
  }

  /** @param {unknown} msg */
  onMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    const m = /** @type {{f0?: unknown, dump?: unknown}} */ (msg);
    if (typeof m.f0 === "number" && Number.isFinite(m.f0) && m.f0 > 0 && m.f0 < sampleRate / 2) {
      this.setFrequency(m.f0);
    }
    if (m.dump) this.postDump();
  }

  /** Post the ring buffer, oldest sample first, as a transferred Float32Array. */
  postDump() {
    const n = this.ringFilled;
    const out = new Float32Array(n);
    if (n < this.ring.length) {
      out.set(this.ring.subarray(0, n));
    } else {
      const tail = this.ring.length - this.ringPos;
      out.set(this.ring.subarray(this.ringPos), 0);
      out.set(this.ring.subarray(0, this.ringPos), tail);
    }
    this.port.postMessage({ dump: out, sampleRate }, [out.buffer]);
  }

  /**
   * Append one quantum to the ring buffer.
   * @param {Float32Array} x
   */
  writeRing(x) {
    const ring = this.ring;
    const len = ring.length;
    let pos = this.ringPos;
    let i = 0;
    while (i < x.length) {
      const take = Math.min(len - pos, x.length - i);
      ring.set(x.subarray(i, i + take), pos);
      pos += take;
      i += take;
      if (pos === len) pos = 0;
    }
    this.ringPos = pos;
    this.ringFilled = Math.min(len, this.ringFilled + x.length);
  }

  /** Measure the accumulated block and post it. */
  emitBlock() {
    const x = this.acc;
    const n = x.length;
    const coeff = this.coeff;
    let s1 = 0.0;
    let s2 = 0.0;
    let sumSq = 0.0;
    if (this.blockBad) {
      this.badBlocks += 1;
    } else {
      for (let i = 0; i < n; i++) {
        const v = x[i];
        const s0 = v + (coeff * s1 - s2);
        s2 = s1;
        s1 = s0;
        sumSq += v * v;
      }
    }
    const mag2 = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    const power = (4.0 * mag2) / (n * n);
    this.port.postMessage({
      powerDb: 10.0 * Math.log10(power + DB_FLOOR),
      rmsDb: 10.0 * Math.log10(sumSq / n + DB_FLOOR),
      blockIndex: this.blockIndex,
      skippedFrames: this.skippedFrames,
      badBlocks: this.badBlocks,
    });
    this.blockIndex += 1;
    this.blockBad = false;
  }

  /**
   * @param {Float32Array[][]} inputs
   * @returns {boolean} always true: keep the processor alive while the node exists
   */
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const x = input[0];
    const len = x.length;
    if (len === 0) return true;

    if (this.lastFrame >= 0) {
      const expected = this.lastFrame + this.lastLen;
      if (currentFrame > expected) this.skippedFrames += currentFrame - expected;
    }
    this.lastFrame = currentFrame;
    this.lastLen = len;

    // Sanitise like Pipeline.process_block: a quantum with NaN/inf becomes zeros.
    let bad = false;
    for (let i = 0; i < len; i++) {
      if (!Number.isFinite(x[i])) {
        bad = true;
        break;
      }
    }
    if (bad) {
      x.fill(0);
      this.blockBad = true;
    }

    this.writeRing(x);

    let i = 0;
    while (i < len) {
      const take = Math.min(this.blockSize - this.fill, len - i);
      this.acc.set(x.subarray(i, i + take), this.fill);
      this.fill += take;
      i += take;
      if (this.fill === this.blockSize) {
        this.emitBlock();
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("block-meter", BlockMeter);
