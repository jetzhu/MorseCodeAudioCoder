/**
 * Microphone capture for the browser decoder.
 *
 * `MicInput` opens the microphone with `getUserMedia`, runs it through the
 * `block-meter` AudioWorklet (`worklet.js`), which measures every 10 ms block
 * and keeps the last 30 s of raw samples, and exposes two analysis taps for
 * the display: `analyser` (an `AnalyserNode` with a 2048-point FFT for the
 * spectrum) and `filtered` (a `BiquadFilterNode` band-pass at `f0` with
 * `Q = f0 / 200`, i.e. a 200 Hz wide band like the Python `BandPass`) followed
 * by `waveAnalyser` for the 6 ms waveform.  The worklet output is muted into
 * the destination so every browser keeps the graph rendering.
 *
 * The `AudioContext` is shared with the encoder's `TonePlayer`: create it from
 * a user gesture in `app.js` and pass it in.  The Python counterpart is
 * `morse/audio_input.py`; like it, only the first channel is used and blocks
 * that arrive faster than they are consumed are counted, not queued forever.
 *
 * `encodeWav` writes 16-bit PCM WAV for the Save 30 s download and
 * `support()` reports what the browser can do so the page degrades with a
 * message instead of a blank window.  Nothing in this module touches the DOM
 * or Web Audio at import time, so it can be imported in Node for tests of
 * `encodeWav`; `MicInput` itself needs a browser.
 *
 * @module audio
 */

/** Tone frequency used until `setFrequency` is called. */
export const DEFAULT_F0 = 2491;

/** Seconds of raw input the worklet keeps for `dump30s`. */
export const RING_SECONDS = 30;

/** Q of the display band-pass: `f0 / Q` = 200 Hz, so f0 +- 100 Hz like the Python filter. */
const BANDWIDTH_HZ = 200;

/**
 * @typedef {object} BlockMessage
 * @property {number} powerDb Goertzel tone power at `f0`, 0 dB = full-scale sine
 * @property {number} rmsDb block level in dBFS (`10*log10(mean(x^2))`)
 * @property {number} blockIndex 0-based block counter from the worklet
 * @property {number} skippedFrames cumulative frames the render graph jumped over
 * @property {number} badBlocks cumulative blocks that contained NaN/inf and were zeroed
 */

/**
 * @typedef {object} InputDevice
 * @property {string} deviceId `""` for the browser default
 * @property {string} label empty until the user has granted permission
 * @property {string} groupId
 */

/**
 * What this browser offers.  `ok` is true when the page can capture and
 * analyse audio; otherwise `reason` explains what is missing in a sentence.
 *
 * @returns {{ok: boolean, getUserMedia: boolean, audioWorklet: boolean, secure: boolean, reason: string}}
 */
export function support() {
  const g = /** @type {any} */ (globalThis);
  const nav = g.navigator;
  const secure = typeof g.isSecureContext === "boolean" ? g.isSecureContext : true;
  const getUserMedia = Boolean(nav && nav.mediaDevices && typeof nav.mediaDevices.getUserMedia === "function");
  const AC = g.AudioContext || g.webkitAudioContext;
  const audioWorklet = Boolean(AC && typeof g.AudioWorkletNode === "function" &&
    (typeof g.AudioWorklet === "function" || "audioWorklet" in (AC.prototype || {})));
  let reason = "";
  if (!secure) {
    reason = "Microphone capture needs a secure page: open this app over https or from localhost.";
  } else if (!getUserMedia) {
    reason = "This browser does not offer microphone capture (getUserMedia). Use a current Chrome or Edge.";
  } else if (!audioWorklet) {
    reason = "This browser has no AudioWorklet, which the decoder needs for its 10 ms blocks. Use a current Chrome or Edge.";
  }
  return { ok: secure && getUserMedia && audioWorklet, getUserMedia, audioWorklet, secure, reason };
}

/**
 * The `getUserMedia` constraints the contract prescribes: mono, with the
 * browser's echo cancellation, noise suppression and automatic gain control
 * all off so the tone reaches the decoder as the driver delivers it.
 *
 * @param {string} [deviceId] `""`/undefined for the browser's default input
 * @returns {MediaStreamConstraints}
 */
export function captureConstraints(deviceId) {
  /** @type {MediaTrackConstraints} */
  const audio = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  };
  if (deviceId) audio.deviceId = deviceId;
  return { audio };
}

/**
 * Encode mono samples as a 16-bit PCM WAV file.
 *
 * Samples are clipped to -1..1 and scaled by 32767 (positive) / 32768
 * (negative) with rounding, the inverse of the `/ 32768` used when the
 * fixtures are read.  Returns a `Blob` in a browser and a `Uint8Array` of the
 * same bytes where `Blob` does not exist (Node tests).
 *
 * @param {ArrayLike<number>} samples mono samples in -1..1
 * @param {number} sampleRate frames per second (integer)
 * @returns {Blob | Uint8Array}
 */
export function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const bytes = new ArrayBuffer(44 + n * 2);
  const view = new DataView(bytes);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  const fs = Math.max(1, Math.round(sampleRate));
  ascii(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, fs, true);
  view.setUint32(28, fs * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++, o += 2) {
    let v = Number(samples[i]);
    if (!Number.isFinite(v)) v = 0;
    if (v > 1) v = 1;
    else if (v < -1) v = -1;
    view.setInt16(o, Math.round(v < 0 ? v * 32768 : v * 32767), true);
  }
  if (typeof Blob === "function") return new Blob([bytes], { type: "audio/wav" });
  return new Uint8Array(bytes);
}

/**
 * Turn a `getUserMedia` failure into a sentence for the message bar.
 * @param {unknown} err
 * @returns {string}
 */
export function describeCaptureError(err) {
  const e = /** @type {{name?: string, message?: string}} */ (err || {});
  switch (e.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Microphone access was denied. Allow the microphone for this site (the icon in the address bar) and press Start again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone was found. Plug one in or pick another input, then press Start again.";
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return "The microphone could not be opened. Another application may be holding it; close it and press Start again.";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "The selected microphone is not available any more. Choose another input and press Start again.";
    case "SecurityError":
      return "Microphone capture is blocked on this page. It needs https (or localhost) and the site's permission.";
    default:
      return `The microphone could not be started${e.message ? `: ${e.message}` : "."}`;
  }
}

/**
 * Microphone -> AudioWorklet block meter, with analysis taps for the display.
 *
 * Usage:
 *
 *     const mic = new MicInput(audioContext);
 *     await mic.start(deviceId, (block) => { ... });   // block: BlockMessage
 *     mic.setFrequency(1000);
 *     const raw = await mic.dump30s();                 // Float32Array, oldest first
 *     await mic.stop();
 */
export class MicInput {
  /**
   * @param {AudioContext} audioContext shared context, created from a user gesture
   * @param {object} [options]
   * @param {number} [options.f0=2491] initial tone frequency in Hz
   * @param {number} [options.ringSeconds=30] seconds of raw input kept for `dump30s`
   * @param {string|URL} [options.workletUrl] location of `worklet.js` (default: next to this module)
   */
  constructor(audioContext, { f0 = DEFAULT_F0, ringSeconds = RING_SECONDS, workletUrl } = {}) {
    if (!audioContext) throw new Error("MicInput needs an AudioContext");
    /** @type {AudioContext} */
    this.context = audioContext;
    /** @type {number} */
    this.f0 = f0;
    /** @type {number} */
    this.ringSeconds = ringSeconds;
    /** @type {string|URL} */
    this.workletUrl = workletUrl || new URL("./worklet.js", import.meta.url);

    /** @type {AnalyserNode | null} 2048-point FFT of the raw input, for the spectrum */
    this.analyser = null;
    /** @type {BiquadFilterNode | null} band-pass at `f0`, Q = f0/200, for the waveform */
    this.filtered = null;
    /** @type {AnalyserNode | null} time-domain tap after `filtered` */
    this.waveAnalyser = null;
    /** @type {AudioWorkletNode | null} */
    this.node = null;
    /** @type {MediaStream | null} */
    this.stream = null;
    /** @type {MediaStreamAudioSourceNode | null} */
    this.source = null;
    /** @type {GainNode | null} */
    this.mute = null;
    /** @type {string} label of the track actually opened ("" until started) */
    this.deviceLabel = "";
    /** @type {string} deviceId of the track actually opened ("" until started) */
    this.deviceId = "";
    /** @type {Float32Array | null} the ring buffer as it was when `stop()` ran */
    this.lastDump = null;
    /** @type {number} sample rate of `lastDump` */
    this.lastDumpRate = 0;
    /** @type {((block: BlockMessage) => void) | null} */
    this.onBlock = null;
    /** @type {((err: Error) => void) | null} called when the worklet processor throws */
    this.onError = null;
    /** @type {(() => void) | null} called when the track ends on its own (device unplugged) */
    this.onEnded = null;

    this._moduleLoaded = false;
    /** @type {Array<{resolve: (x: Float32Array) => void, reject: (e: Error) => void, timer: number}>} */
    this._dumpWaiters = [];
    this._running = false;
  }

  /**
   * Input-capable devices from `enumerateDevices()`; labels are empty until
   * the user has granted microphone permission once.
   *
   * @returns {Promise<InputDevice[]>}
   */
  static async listDevices() {
    const nav = /** @type {any} */ (globalThis).navigator;
    if (!nav || !nav.mediaDevices || typeof nav.mediaDevices.enumerateDevices !== "function") return [];
    let all;
    try {
      all = await nav.mediaDevices.enumerateDevices();
    } catch {
      return [];
    }
    return all
      .filter((d) => d.kind === "audioinput")
      .map((d) => ({ deviceId: d.deviceId || "", label: d.label || "", groupId: d.groupId || "" }));
  }

  /** True between a successful `start` and `stop`. */
  get running() {
    return this._running;
  }

  /** Sample rate of the shared AudioContext (what the worklet sees). */
  get sampleRate() {
    return this.context.sampleRate;
  }

  /** Block length in samples, `round(sampleRate / 100)`, as the worklet uses it. */
  get blockSize() {
    return Math.max(2, Math.round(this.context.sampleRate / 100));
  }

  /** Block length in milliseconds (10 ms up to rounding). */
  get blockMs() {
    return (1000 * this.blockSize) / this.context.sampleRate;
  }

  /**
   * Open the microphone and start posting blocks.
   *
   * Resolves once the graph is running.  Rejects with the `getUserMedia`
   * error (see {@link describeCaptureError}) or with the worklet load error;
   * nothing is left half-open in that case.  Calling `start` while running
   * stops the current stream first.
   *
   * @param {string} deviceId `""` for the default input
   * @param {(block: BlockMessage) => void} onBlock called once per 10 ms block
   * @returns {Promise<void>}
   */
  async start(deviceId, onBlock) {
    if (this._running) await this.stop();
    const ac = this.context;
    if (ac.state === "suspended" && typeof ac.resume === "function") {
      try {
        await ac.resume();
      } catch {
        /* resumes on the next gesture */
      }
    }
    this.onBlock = onBlock || null;

    const nav = /** @type {any} */ (globalThis).navigator;
    const stream = await nav.mediaDevices.getUserMedia(captureConstraints(deviceId));
    try {
      if (!this._moduleLoaded) {
        await ac.audioWorklet.addModule(String(this.workletUrl));
        this._moduleLoaded = true;
      }
      const track = stream.getAudioTracks()[0];
      this.deviceLabel = track ? track.label || "" : "";
      const settings = track && typeof track.getSettings === "function" ? track.getSettings() : {};
      this.deviceId = (settings && settings.deviceId) || deviceId || "";
      if (track) {
        track.onended = () => {
          if (this._running && typeof this.onEnded === "function") this.onEnded();
        };
      }

      const source = ac.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ac, "block-meter", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        processorOptions: { f0: this.f0, ringSeconds: this.ringSeconds },
      });
      node.port.onmessage = (event) => this._onMessage(event.data);
      node.onprocessorerror = () => {
        if (typeof this.onError === "function") this.onError(new Error("the audio worklet processor failed"));
      };

      const analyser = ac.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.5;

      const filtered = ac.createBiquadFilter();
      filtered.type = "bandpass";
      filtered.frequency.value = this.f0;
      filtered.Q.value = this.f0 / BANDWIDTH_HZ;

      const waveAnalyser = ac.createAnalyser();
      waveAnalyser.fftSize = 1024; // 21 ms at 48 kHz; the display takes the last 6 ms
      waveAnalyser.smoothingTimeConstant = 0;

      // A muted path to the destination keeps every branch rendering in every browser.
      const mute = ac.createGain();
      mute.gain.value = 0;

      source.connect(node);
      node.connect(mute);
      source.connect(analyser);
      analyser.connect(mute);
      source.connect(filtered);
      filtered.connect(waveAnalyser);
      waveAnalyser.connect(mute);
      mute.connect(ac.destination);

      this.stream = stream;
      this.source = source;
      this.node = node;
      this.analyser = analyser;
      this.filtered = filtered;
      this.waveAnalyser = waveAnalyser;
      this.mute = mute;
      this._running = true;
    } catch (err) {
      for (const t of stream.getTracks()) t.stop();
      throw err;
    }
  }

  /**
   * Close the microphone and tear the graph down.  The last 30 s of raw input
   * are kept in `lastDump` so Save 30 s still works after stopping.  The
   * shared AudioContext stays open for the encoder.
   *
   * @returns {Promise<void>}
   */
  async stop() {
    if (!this._running) return;
    this._running = false;
    try {
      const dump = await this.dump30s(1500);
      this.lastDump = dump;
      this.lastDumpRate = this.context.sampleRate;
    } catch {
      /* the worklet did not answer; keep whatever we had */
    }
    const { stream, source, node, analyser, filtered, waveAnalyser, mute } = this;
    this.stream = null;
    this.source = null;
    this.node = null;
    this.analyser = null;
    this.filtered = null;
    this.waveAnalyser = null;
    this.mute = null;
    if (node) {
      node.port.onmessage = null;
      node.onprocessorerror = null;
    }
    for (const n of [source, node, analyser, filtered, waveAnalyser, mute]) {
      if (!n) continue;
      try {
        n.disconnect();
      } catch {
        /* never connected */
      }
    }
    if (stream) for (const t of stream.getTracks()) t.stop();
    for (const w of this._dumpWaiters.splice(0)) {
      clearTimeout(w.timer);
      w.reject(new Error("microphone stopped"));
    }
  }

  /**
   * Retune the worklet's Goertzel bin and the display band-pass to `f0`.
   * Takes effect from the next block; the detector should be reset by the
   * caller because its level statistics belong to the old frequency.
   *
   * @param {number} f0 tone frequency in Hz, `0 < f0 < sampleRate / 2`
   */
  setFrequency(f0) {
    const value = Number(f0);
    if (!(value > 0) || value >= this.context.sampleRate / 2) {
      throw new RangeError(`f0 must be between 0 and sampleRate/2, got ${f0}`);
    }
    this.f0 = value;
    if (this.node) this.node.port.postMessage({ f0: value });
    if (this.filtered) {
      const t = this.context.currentTime;
      this.filtered.frequency.setValueAtTime(value, t);
      this.filtered.Q.setValueAtTime(value / BANDWIDTH_HZ, t);
    }
  }

  /**
   * The last 30 s (or less, right after start) of raw input, oldest first.
   *
   * While running the worklet answers within a few milliseconds; after
   * `stop()` the copy taken at that moment is returned.  Rejects when nothing
   * has been recorded yet or the worklet does not answer in `timeoutMs`.
   *
   * @param {number} [timeoutMs=2000]
   * @returns {Promise<Float32Array>}
   */
  dump30s(timeoutMs = 2000) {
    if (!this.node) {
      if (this.lastDump) return Promise.resolve(this.lastDump);
      return Promise.reject(new Error("nothing recorded yet: press Start first"));
    }
    const node = this.node;
    return new Promise((resolve, reject) => {
      const waiter = { resolve, reject, timer: 0 };
      waiter.timer = setTimeout(() => {
        const i = this._dumpWaiters.indexOf(waiter);
        if (i >= 0) this._dumpWaiters.splice(i, 1);
        reject(new Error("the audio worklet did not answer"));
      }, timeoutMs);
      this._dumpWaiters.push(waiter);
      node.port.postMessage({ dump: true });
    });
  }

  /**
   * Spectrum of the raw input in dB, one value per FFT bin from 0 to fs/2,
   * or `null` when not running.  The array is reused between calls.
   *
   * @returns {Float32Array | null}
   */
  spectrumDb() {
    const a = this.analyser;
    if (!a) return null;
    if (!this._specBuf || this._specBuf.length !== a.frequencyBinCount) {
      this._specBuf = new Float32Array(a.frequencyBinCount);
    }
    a.getFloatFrequencyData(this._specBuf);
    return this._specBuf;
  }

  /** Hz per bin of {@link spectrumDb}. */
  get binHz() {
    const a = this.analyser;
    return a ? this.context.sampleRate / a.fftSize : 0;
  }

  /**
   * The most recent band-passed samples (time domain), `ms` milliseconds of
   * them, or `null` when not running.  The array is reused between calls.
   *
   * @param {number} [ms=6]
   * @returns {Float32Array | null}
   */
  filteredWave(ms = 6) {
    const a = this.waveAnalyser;
    if (!a) return null;
    if (!this._waveBuf || this._waveBuf.length !== a.fftSize) this._waveBuf = new Float32Array(a.fftSize);
    a.getFloatTimeDomainData(this._waveBuf);
    const n = Math.min(a.fftSize, Math.max(2, Math.round((ms / 1000) * this.context.sampleRate)));
    return this._waveBuf.subarray(a.fftSize - n);
  }

  // --------------------------------------------------------------- private

  /** @param {any} msg */
  _onMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.dump) {
      const data = msg.dump instanceof Float32Array ? msg.dump : new Float32Array(msg.dump);
      for (const w of this._dumpWaiters.splice(0)) {
        clearTimeout(w.timer);
        w.resolve(data);
      }
      return;
    }
    if (typeof msg.powerDb === "number" && typeof this.onBlock === "function") {
      this.onBlock(/** @type {BlockMessage} */ (msg));
    }
  }
}
