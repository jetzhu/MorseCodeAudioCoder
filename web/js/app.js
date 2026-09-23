/**
 * Beeper Morse Console: page wiring.
 *
 * Everything that touches the DOM lives here.  The signal chain is the same
 * as `morse/pipeline.py`: the AudioWorklet (`worklet.js`, driven by
 * `audio.js`) posts one `{powerDb, rmsDb}` message per 10 ms block, which is
 * fed to `ToneDetector` -> `MorseDecoder` exactly as `Pipeline.process_block`
 * does (detector update, decoder feed for each final run, `idle()` while the
 * current run is OFF).  A `requestAnimationFrame` loop capped at 30 fps draws
 * the canvases ported from `web/mock.html` from ring buffers filled by that
 * per-block step; the encoder strip uses `table.js` and `player.js`.
 *
 * This is the one module that keeps page-level state; everything it imports
 * is a pure ES module.  Loaded from `index.html` as `<script type="module">`.
 *
 * @module app
 */

import { encode, lookup } from "./table.js";
import { ToneDetector, isTonal } from "./detector.js";
import { MorseDecoder } from "./decoder.js";
import { TonePlayer, buildGuide, layoutGuideLabels, roundHalfEven } from "./player.js";
import { MicInput, describeCaptureError, encodeWav, support } from "./audio.js";

// ------------------------------------------------------------------ constants

/** Measured PC-beeper tone (docs/PLAN.md section 8). */
const DEFAULT_F0 = 2491;
/** Blocks of history in the tone-power plot: 1000 x 10 ms = 10 s. */
const HIST_N = 1000;
/** Blocks in the chopped-signal window: 300 x 10 ms = 3 s. */
const CHOP_WINDOW = 300;
/** More than 4 fragments per second over the 3 s window shows the hint. */
const CHOP_FRAGMENTS = 4 * 3;
/** Blocks without the symptom after which a shown hint hides itself (15 s). */
const CHOP_CLEAR_BLOCKS = 1500;
/** Mark lengths kept for the histogram. */
const HIST_MARKS = 30;
/** Auto-detect accepts a peak only this far above the band median. */
const AUTO_MIN_PROMINENCE_DB = 10;
const RELEASES_API = "https://api.github.com/repos/jetzhu/MorseCodeAudioCoder/releases/latest";
const RELEASES_PAGE = "https://github.com/jetzhu/MorseCodeAudioCoder/releases";
const REPO_GIT = "https://github.com/jetzhu/MorseCodeAudioCoder.git";
const STORAGE = { device: "morse.deviceId", f0: "morse.f0" };

// -------------------------------------------------------------------- helpers

const $ = (id) => document.getElementById(id);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const font = (px, mono) => `${px}px ${mono ? '"IBM Plex Mono", Consolas, monospace' : '"IBM Plex Sans", system-ui, sans-serif'}`;

/** localStorage wrapped in try/catch: private windows and blocked storage must not break the page. */
const store = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* storage unavailable */
    }
  },
};

// ---------------------------------------------------------------------- state

const el = {
  start: $("startBtn"), dev: $("dev"), f0: $("f0"), auto: $("autoBtn"),
  wpmAuto: $("wpmAuto"), wpmManual: $("wpmManual"), wpmVal: $("wpmVal"),
  pause: $("pauseBtn"), save: $("saveBtn"), clear: $("clearBtn"),
  msgBar: $("msgBar"), msgText: $("msgText"), msgDismiss: $("msgDismiss"),
  chopHint: $("chopHint"), chopDismiss: $("chopDismiss"),
  noteBar: $("noteBar"), noteText: $("noteText"), noteDismiss: $("noteDismiss"),
  f0Label: $("f0Label"), specInfo: $("specInfo"),
  sym: $("symBuf"), hint: $("symHint"), text: $("textOut"),
  dit: $("ditV"), dah: $("dahV"), lgap: $("lgapV"), off: $("offV"),
  pill: $("statePill"), lvl: $("lvlV"), snr: $("snrV"), wpm: $("wpmV"), cnt: $("cntV"), unk: $("unkV"),
  encIn: $("encIn"), encWpm: $("encWpm"), encPlay: $("encPlay"), encCopy: $("encCopy"), encOut: $("encOut"),
  encFeed: $("encFeed"),
  encDur: $("encDur"), encDit: $("encDit"), encGap: $("encGap"), encTone: $("encTone"),
  stateDot: $("stateDot"), stateV: $("stateV"), blockV: $("blockV"), dropV: $("dropV"), clockV: $("clockV"), rateV: $("rateV"),
  titleDev: $("titleDev"), factTone: $("factTone"), factRate: $("factRate"), factBlock: $("factBlock"),
  dlAssets: $("dlAssets"), dlNote: $("dlNote"), win: $("win"),
};

const state = {
  f0: DEFAULT_F0,
  supported: true,
  running: false,
  paused: false,
  starting: false,
  everRan: false,
  manual: false,
  wpmManual: 8,
  fs: 0,
  blockSize: 480,
  blockMs: 10,
  blocks: 0, // blocks processed since Start (drives the clock)
  dropped: 0, // worklet messages that never arrived
  lastBlockIndex: -1,
  skippedFrames: 0,
  badBlocks: 0,
  lvlDb: -120,
  peakHold: -120,
  peakHoldAge: 0,
  deviceLabel: "",
  /** @type {number[]} */
  recentMarks: [],
  // raw-verdict bookkeeping for the chopped-signal hint
  rawSeen: false,
  rawOn: false,
  rawLen: 0,
  fragRing: new Uint8Array(CHOP_WINDOW),
  onRing: new Uint8Array(CHOP_WINDOW),
  fragCount: 0,
  onCount: 0,
  chopClearBlocks: 0,
  chopDismissed: false,
};

/** Tone-power history for plot B; NaN marks "no data yet". */
const hist = {
  power: new Float32Array(HIST_N).fill(NaN),
  on: new Uint8Array(HIST_N),
  hi: new Float32Array(HIST_N).fill(NaN),
  lo: new Float32Array(HIST_N).fill(NaN),
  idx: 0,
};

let detector = new ToneDetector();
let decoder = new MorseDecoder();
/** @type {AudioContext | null} shared by the microphone graph and the encoder's player */
let audioContext = null;
/** @type {MicInput | null} */
let mic = null;
const player = new TonePlayer(null);
/** @type {Float32Array | null} last spectrum frame, held while paused */
let specHold = null;
/** @type {Float32Array | null} last 6 ms of band-passed input, held while paused */
let waveHold = null;
let dirty = true;

// ------------------------------------------------------------- shared audio

/** Create (on the first gesture) and resume the shared AudioContext. */
function ensureContext() {
  if (!audioContext) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error("Web Audio is not available in this browser");
    audioContext = new AC({ latencyHint: "interactive" });
  }
  if (audioContext.state === "suspended") audioContext.resume().catch(() => {});
  return audioContext;
}

// --------------------------------------------------------------- messages

function showMessage(text) {
  el.msgText.textContent = text;
  el.msgBar.hidden = false;
}
function hideMessage() {
  el.msgBar.hidden = true;
}
let noteTimer = 0;
function showNote(text, ms = 6000) {
  el.noteText.textContent = text;
  el.noteBar.hidden = false;
  clearTimeout(noteTimer);
  if (ms > 0) noteTimer = setTimeout(() => { el.noteBar.hidden = true; }, ms);
}

// ------------------------------------------------------------- per block

/**
 * One block from the worklet.  Bookkeeping runs even while paused so the
 * dropped count stays truthful; the decode step is skipped.
 * @param {import("./audio.js").BlockMessage} m
 */
function onBlock(m) {
  if (!state.running) return;
  if (state.lastBlockIndex >= 0 && m.blockIndex > state.lastBlockIndex + 1) {
    state.dropped += m.blockIndex - state.lastBlockIndex - 1;
    dirty = true;
  }
  state.lastBlockIndex = m.blockIndex;
  state.skippedFrames = m.skippedFrames || 0;
  state.badBlocks = m.badBlocks || 0;
  if (state.paused) return;
  processBlock(m.powerDb, m.rmsDb);
}

/** The `Pipeline.process_block` order: detector, decoder feed, idle, then the display buffers. */
function processBlock(powerDb, rmsDb) {
  // A loud but broadband block (click, speech) never switches the detector ON.
  for (const run of detector.update(powerDb, isTonal(powerDb, rmsDb))) feedRun(run);
  const current = detector.currentRun;
  if (!current.on) decoder.idle(current.ms);
  trackFragments(detector.state);

  const i = hist.idx;
  hist.power[i] = powerDb;
  hist.on[i] = detector.state ? 1 : 0;
  hist.hi[i] = detector.thresholdHiDb;
  hist.lo[i] = detector.thresholdLoDb;
  hist.idx = (i + 1) % HIST_N;

  state.lvlDb = rmsDb;
  if (rmsDb > state.peakHold) {
    state.peakHold = rmsDb;
    state.peakHoldAge = 0;
  } else if (++state.peakHoldAge > 150) {
    state.peakHold -= 0.3;
  }
  state.blocks += 1;
}

/** Hand one final run to the decoder and remember marks for the histogram. */
function feedRun(run) {
  decoder.feed(run);
  if (run.on && run.ms < decoder.maxMarkMs) {
    state.recentMarks.push(run.ms);
    if (state.recentMarks.length > HIST_MARKS) state.recentMarks.shift();
  }
}

/**
 * Count raw verdict runs the debounce will remove (shorter than
 * `minRunBlocks`) over the last 3 s; with ON runs present, more than 4 per
 * second means the driver is chopping the tone.
 * @param {boolean} on the detector's raw verdict for this block
 */
function trackFragments(on) {
  let fragment = 0;
  if (!state.rawSeen) {
    state.rawSeen = true;
    state.rawOn = on;
    state.rawLen = 1;
  } else if (on === state.rawOn) {
    state.rawLen += 1;
  } else {
    if (state.rawLen < detector.minRunBlocks) fragment = 1;
    state.rawOn = on;
    state.rawLen = 1;
  }
  const slot = state.blocks % CHOP_WINDOW;
  state.fragCount += fragment - state.fragRing[slot];
  state.fragRing[slot] = fragment;
  const onv = on ? 1 : 0;
  state.onCount += onv - state.onRing[slot];
  state.onRing[slot] = onv;

  if (state.fragCount > CHOP_FRAGMENTS && state.onCount > 0) {
    state.chopClearBlocks = 0;
    if (!state.chopDismissed && el.chopHint.hidden) el.chopHint.hidden = false;
  } else if (!el.chopHint.hidden && ++state.chopClearBlocks > CHOP_CLEAR_BLOCKS) {
    el.chopHint.hidden = true;
  }
}

/** Forget the stream-dependent display state (a new Start begins with a clean plot). */
function resetStreamState() {
  state.blocks = 0;
  state.dropped = 0;
  state.lastBlockIndex = -1;
  state.skippedFrames = 0;
  state.badBlocks = 0;
  state.lvlDb = -120;
  state.peakHold = -120;
  state.peakHoldAge = 0;
  state.rawSeen = false;
  state.rawLen = 0;
  state.fragRing.fill(0);
  state.onRing.fill(0);
  state.fragCount = 0;
  state.onCount = 0;
  state.chopClearBlocks = 0;
  hist.power.fill(NaN);
  hist.hi.fill(NaN);
  hist.lo.fill(NaN);
  hist.on.fill(0);
  hist.idx = 0;
  specHold = null;
  waveHold = null;
}

// -------------------------------------------------------------- lifecycle

async function startListening() {
  if (state.starting || state.running || !state.supported) return;
  state.starting = true;
  el.start.disabled = true;
  el.start.textContent = "Starting…";
  hideMessage();
  updateStatus();
  try {
    const ac = ensureContext();
    if (!mic) mic = new MicInput(ac, { f0: state.f0 });
    mic.onError = (err) => {
      showMessage(`${err.message}. Press Start to reopen the microphone.`);
      stopListening();
    };
    mic.onEnded = () => {
      showMessage("The microphone stream ended (device unplugged or taken by another application). Press Start to reopen it.");
      stopListening();
    };
    mic.setFrequency(state.f0);
    await mic.start(el.dev.value, onBlock);
  } catch (err) {
    state.starting = false;
    showMessage(describeCaptureError(err));
    updateControls();
    updateStatus();
    return;
  }
  state.fs = mic.sampleRate;
  state.blockSize = mic.blockSize;
  state.blockMs = mic.blockMs;
  state.deviceLabel = mic.deviceLabel;
  detector = new ToneDetector({ blockMs: state.blockMs });
  resetStreamState();
  state.running = true;
  state.paused = false;
  state.starting = false;
  state.everRan = true;
  el.pause.textContent = "Pause";
  syncPlayerFeed(); // a tone already playing reaches the decoder from now on
  // Permission is granted now, so the device list has labels.
  await refreshDevices(el.dev.value === "" ? mic.deviceId : el.dev.value);
  if (state.f0 >= state.fs / 2) applyFrequency(DEFAULT_F0);
  updateControls();
  updateStatus();
  dirty = true;
}

async function stopListening() {
  if (!mic || !state.running) return;
  state.running = false;
  state.paused = false;
  syncPlayerFeed(); // detach the tone from the bus before the graph is torn down
  for (const run of detector.flush()) feedRun(run);
  decoder.idle(Infinity);
  finishAuto(false);
  await mic.stop();
  el.pause.textContent = "Pause";
  updateControls();
  updateStatus();
  dirty = true;
}

async function restartListening() {
  await stopListening();
  await startListening();
}

// ---------------------------------------------------------------- controls

/**
 * Apply a new tone frequency: retune the worklet and the display filter and,
 * like `Pipeline.set_frequency`, finalise what the detector holds and reset
 * its level statistics; the decoder keeps text and timing.
 * @param {number} f0
 * @returns {boolean} false when the value was rejected
 */
function applyFrequency(f0) {
  const value = Math.round(Number(f0));
  const nyquist = state.fs ? state.fs / 2 : 24000;
  if (!Number.isFinite(value) || value < 100 || value >= nyquist) {
    el.f0.value = String(state.f0);
    showNote(`The tone frequency must be between 100 and ${Math.floor(nyquist) - 1} Hz.`);
    return false;
  }
  const changed = value !== state.f0;
  state.f0 = value;
  el.f0.value = String(value);
  el.f0Label.textContent = `${value} Hz`;
  el.encTone.innerHTML = `${value}<small>Hz</small>`;
  el.factTone.textContent = `${value} Hz`;
  store.set(STORAGE.f0, String(value));
  if (mic) {
    try {
      mic.setFrequency(value);
    } catch {
      /* rejected by the context's rate; the check above already covers it */
    }
  }
  if (changed && state.running) {
    for (const run of detector.flush()) feedRun(run);
    detector.reset();
  }
  dirty = true;
  return true;
}

/** Speed Auto/Manual -> MorseDecoder options; the text and counters carry over. */
function setSpeedMode(manual) {
  state.manual = manual;
  el.wpmAuto.setAttribute("aria-pressed", String(!manual));
  el.wpmManual.setAttribute("aria-pressed", String(manual));
  el.wpmVal.disabled = !manual;
  rebuildDecoder();
}

function rebuildDecoder() {
  const wpm = clamp(parseFloat(el.wpmVal.value) || 8, 2, 40);
  state.wpmManual = wpm;
  el.wpmVal.value = String(wpm);
  const next = state.manual ? new MorseDecoder({ wpm, adaptive: false }) : new MorseDecoder();
  decoder.idle(Infinity); // commit the pending letter under the old rule
  next.adoptTiming(decoder); // keep the measured speed and reverb offset (same as the desktop app)
  next.text = decoder.text;
  next.letterCount = decoder.letterCount;
  next.unknownCount = decoder.unknownCount;
  decoder = next;
  dirty = true;
}

function togglePause() {
  if (!state.running) return;
  state.paused = !state.paused;
  el.pause.textContent = state.paused ? "Resume" : "Pause";
  updateStatus();
  dirty = true;
}

function clearText() {
  decoder.reset(true);
  dirty = true;
}

function timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function save30s() {
  if (!mic) return;
  el.save.disabled = true;
  el.save.textContent = "Saving…";
  try {
    const samples = await mic.dump30s();
    const fs = mic.running ? mic.sampleRate : mic.lastDumpRate || mic.sampleRate;
    if (!samples.length) throw new Error("no audio captured yet");
    const blob = encodeWav(samples, fs);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `morse_${timestamp()}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    el.save.textContent = `Saved ${(samples.length / fs).toFixed(0)} s`;
  } catch (err) {
    showNote(`Nothing saved: ${err.message}.`);
    el.save.textContent = "Save 30 s";
  }
  setTimeout(() => {
    el.save.textContent = "Save 30 s";
    updateControls();
  }, 1800);
}

/** Fill the device picker; labels are empty until permission has been granted once. */
async function refreshDevices(selectId) {
  const devices = await MicInput.listDevices();
  const wanted = selectId !== undefined && selectId !== null ? selectId : store.get(STORAGE.device) || "";
  el.dev.replaceChildren();
  const add = (value, label) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    el.dev.appendChild(o);
  };
  add("", "Default microphone");
  let unnamed = 0;
  for (const d of devices) {
    if (!d.deviceId) continue;
    const label = d.label || `Microphone ${++unnamed}${d.deviceId === "default" ? " (default)" : ""}`;
    add(d.deviceId, label);
  }
  const ids = Array.from(el.dev.options).map((o) => o.value);
  el.dev.value = ids.includes(wanted) ? wanted : "";
}

function updateControls() {
  const running = state.running;
  el.start.textContent = running ? "Stop" : "Start";
  el.start.classList.toggle("running", running);
  el.start.disabled = !state.supported || state.starting;
  el.pause.disabled = !running;
  el.auto.disabled = !running;
  el.save.disabled = !(running || (mic && mic.lastDump));
  el.encPlay.disabled = !encoder.guide.totalMs;
}

function updateStatus() {
  let dot = "○";
  let cls = "dot";
  let text;
  if (!state.supported) text = "Unavailable in this browser";
  else if (state.starting) text = "Opening microphone…";
  else if (state.running && state.paused) {
    dot = "●";
    cls = "dot paused";
    text = "Paused";
  } else if (state.running) {
    dot = "●";
    cls = "dot live";
    text = "Listening";
  } else if (state.everRan) text = "Stopped — press Start";
  else text = "Ready — press Start";
  el.stateDot.textContent = dot;
  el.stateDot.className = cls;
  el.stateV.textContent = text;
  if (state.fs) {
    const khz = (state.fs / 1000).toFixed(state.fs % 1000 ? 1 : 0);
    el.blockV.textContent = `${state.blockSize}-sample blocks · ${state.blockMs.toFixed(state.blockMs % 1 ? 2 : 0)} ms`;
    el.rateV.textContent = `${khz} kHz`;
    el.factRate.textContent = `${khz} kHz`;
    el.factBlock.textContent = `${state.blockSize} samples`;
    el.specInfo.textContent = `0–8 kHz · 2048-point FFT · ${((2048 / state.fs) * 1000).toFixed(0)} ms`;
    el.titleDev.textContent = state.running || state.everRan
      ? `${state.deviceLabel || "Default microphone"} · ${khz} kHz${state.running ? "" : " · closed"}`
      : "microphone closed";
  } else {
    el.blockV.textContent = "10 ms blocks";
    el.titleDev.textContent = "microphone closed";
  }
}

// ----------------------------------------------------------- auto-detect

const auto = { active: false, acc: null, frames: 0, timer: 0, started: 0 };

function startAutoDetect() {
  if (auto.active || !state.running || !mic) return;
  const spec = mic.spectrumDb();
  if (!spec) return;
  auto.acc = new Float64Array(spec.length);
  auto.frames = 0;
  auto.active = true;
  auto.started = performance.now();
  el.auto.disabled = true;
  el.auto.textContent = "Listening…";
  auto.timer = setInterval(sampleAuto, 40);
}

function sampleAuto() {
  const spec = mic ? mic.spectrumDb() : null;
  if (!spec || spec.length !== auto.acc.length) {
    finishAuto(false);
    return;
  }
  for (let i = 0; i < spec.length; i++) {
    const v = spec[i];
    auto.acc[i] += Number.isFinite(v) ? Math.pow(10, v / 10) : 0;
  }
  auto.frames += 1;
  if (performance.now() - auto.started >= 1000) finishAuto(true);
}

function finishAuto(ok) {
  if (!auto.active) return;
  clearInterval(auto.timer);
  auto.active = false;
  el.auto.textContent = "Auto-detect";
  el.auto.disabled = !state.running;
  if (!ok || !auto.frames || !mic) return;
  const n = auto.acc.length;
  const db = new Float64Array(n);
  for (let i = 0; i < n; i++) db[i] = 10 * Math.log10(auto.acc[i] / auto.frames + 1e-12);
  const { peakHz, prominenceDb } = findToneFrequency(db, mic.binHz, 300, 8000);
  if (!(peakHz > 0)) {
    showNote("No spectrum bins between 300 and 8000 Hz at this sample rate.");
    return;
  }
  if (prominenceDb < AUTO_MIN_PROMINENCE_DB) {
    showNote(`No clear tone: the strongest peak (${Math.round(peakHz)} Hz) is only ${prominenceDb.toFixed(0)} dB above the band. Hold a beep for a second and try again.`);
    return;
  }
  if (applyFrequency(Math.round(peakHz))) {
    showNote(`Tone set to ${Math.round(peakHz)} Hz, ${prominenceDb.toFixed(0)} dB above the band median.`);
  }
}

/**
 * Port of `morse.dsp.find_tone_frequency` for an evenly spaced spectrum:
 * strongest bin within `[fmin, fmax]`, refined by a parabola through its
 * neighbours; prominence is the peak minus the band median.
 *
 * @param {ArrayLike<number>} db power per bin in dB, bin `i` at `i * binHz`
 * @param {number} binHz
 * @param {number} fmin
 * @param {number} fmax
 * @returns {{peakHz: number, prominenceDb: number}}
 */
function findToneFrequency(db, binHz, fmin, fmax) {
  const lo = Math.max(0, Math.ceil(fmin / binHz));
  const hi = Math.min(db.length - 1, Math.floor(fmax / binHz));
  if (!(binHz > 0) || hi < lo) return { peakHz: 0, prominenceDb: 0 };
  let j = lo;
  for (let i = lo + 1; i <= hi; i++) if (db[i] > db[j]) j = i;
  let peakHz = j * binHz;
  if (j > 0 && j < db.length - 1) {
    const a = db[j - 1];
    const b = db[j];
    const c = db[j + 1];
    const denom = a - 2 * b + c;
    if (Number.isFinite(denom) && denom < 0) {
      const delta = clamp((0.5 * (a - c)) / denom, -0.5, 0.5);
      peakHz = (j + delta) * binHz;
    }
  }
  const band = Array.from({ length: hi - lo + 1 }, (_, k) => db[lo + k]).sort((x, y) => x - y);
  const m = band.length;
  const median = m % 2 ? band[(m - 1) / 2] : (band[m / 2 - 1] + band[m / 2]) / 2;
  return { peakHz, prominenceDb: db[j] - median };
}

// --------------------------------------------------------------- canvases

const canvases = {};
function setup(id) {
  const c = $(id);
  canvases[id] = { c, ctx: c.getContext("2d"), w: 0, h: 0 };
  return canvases[id];
}
function fit(cv) {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = cv.c.clientWidth;
  const h = cv.c.clientHeight;
  if (cv.w !== w || cv.h !== h) {
    cv.w = w;
    cv.h = h;
    cv.c.width = Math.round(w * dpr);
    cv.c.height = Math.round(h * dpr);
  }
  cv.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return cv;
}
["specCanvas", "powerCanvas", "stripCanvas", "meterCanvas", "waveCanvas", "histCanvas", "encCanvas"].forEach(setup);

/** Centered two-line instruction on an empty plot (the resting state before Start). */
function restingText(ctx, w, h, line1, line2) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = cssVar("--ink-2");
  ctx.font = font(13, false);
  ctx.fillText(line1, w / 2, h / 2 - (line2 ? 9 : 0));
  if (line2) {
    ctx.fillStyle = cssVar("--ink-3");
    ctx.font = font(11.5, false);
    ctx.fillText(line2, w / 2, h / 2 + 10);
  }
  ctx.restore();
}

let hoverX = null;
$("powerCanvas").addEventListener("mousemove", (e) => {
  hoverX = e.offsetX;
  dirty = true;
});
$("powerCanvas").addEventListener("mouseleave", () => {
  hoverX = null;
  dirty = true;
});

function drawSpectrum() {
  const cv = fit(canvases.specCanvas);
  const { ctx, w, h } = cv;
  const th = { grid: cssVar("--grid"), ink3: cssVar("--ink-3"), spec: cssVar("--spec"), fill: cssVar("--spec-fill"), trace: cssVar("--trace") };
  const padL = 40;
  const padR = 12;
  const padT = 10;
  const padB = 22;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const fmax = 8000;
  const dbMin = -110;
  const dbMax = -10;
  ctx.clearRect(0, 0, w, h);
  const x = (f) => padL + (f / fmax) * pw;
  const y = (db) => padT + ((dbMax - clamp(db, dbMin, dbMax)) / (dbMax - dbMin)) * ph;
  ctx.strokeStyle = th.grid;
  ctx.lineWidth = 1;
  ctx.font = font(10, true);
  ctx.fillStyle = th.ink3;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let db = dbMax; db >= dbMin; db -= 20) {
    ctx.beginPath();
    ctx.moveTo(padL, y(db) + 0.5);
    ctx.lineTo(w - padR, y(db) + 0.5);
    ctx.stroke();
    ctx.fillText(db, padL - 6, y(db));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let f = 0; f <= fmax; f += 1000) {
    ctx.beginPath();
    ctx.moveTo(x(f) + 0.5, padT);
    ctx.lineTo(x(f) + 0.5, padT + ph);
    ctx.stroke();
    ctx.fillText(f >= 1000 ? f / 1000 + "k" : "0", x(f), padT + ph + 6);
  }

  if (state.running && mic) {
    if (!state.paused || !specHold) {
      const s = mic.spectrumDb();
      if (s) {
        if (!specHold || specHold.length !== s.length) specHold = new Float32Array(s.length);
        specHold.set(s);
      }
    }
    const binHz = mic.binHz;
    if (specHold && binHz > 0) {
      const n = Math.min(specHold.length, Math.floor(fmax / binHz) + 1);
      const v = (i) => (Number.isFinite(specHold[i]) ? specHold[i] : dbMin);
      ctx.beginPath();
      ctx.moveTo(x(0), y(dbMin));
      for (let i = 0; i < n; i++) ctx.lineTo(x(i * binHz), y(v(i)));
      ctx.lineTo(x(Math.min(fmax, (n - 1) * binHz)), y(dbMin));
      ctx.closePath();
      ctx.fillStyle = th.fill;
      ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const px = x(i * binHz);
        const py = y(v(i));
        if (i) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
      }
      ctx.strokeStyle = th.spec;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = "round";
      ctx.stroke();
    }
  } else if (!state.everRan) {
    restingText(ctx, w, h, state.supported ? "Press Start to open the microphone" : "Microphone capture is not available here",
      "What it hears appears here, 0 to 8 kHz; the marker sits on the tone frequency");
  }

  const mark = (f, label, strong) => {
    ctx.save();
    ctx.setLineDash(strong ? [] : [3, 4]);
    ctx.strokeStyle = th.trace;
    ctx.globalAlpha = strong ? 1 : 0.55;
    ctx.lineWidth = strong ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(x(f) + 0.5, padT);
    ctx.lineTo(x(f) + 0.5, padT + ph);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = th.trace;
    ctx.font = font(11, true);
    ctx.textAlign = f > 7000 ? "right" : "left";
    ctx.textBaseline = "top";
    ctx.fillText(label, x(f) + (f > 7000 ? -5 : 5), padT + 2);
    ctx.restore();
  };
  if (state.f0 <= fmax) mark(state.f0, `${state.f0} Hz`, true);
  if (3 * state.f0 <= fmax) mark(3 * state.f0, "3f", false);
}

function drawPower() {
  const cv = fit(canvases.powerCanvas);
  const { ctx, w, h } = cv;
  const th = { grid: cssVar("--grid"), ink2: cssVar("--ink-2"), ink3: cssVar("--ink-3"), trace: cssVar("--trace"), fill: cssVar("--trace-fill"), thr: cssVar("--thr"), ink: cssVar("--ink") };
  const padL = 40;
  const padR = 12;
  const padT = 8;
  const padB = 4;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  const dbMin = -110;
  const dbMax = 0;
  const N = HIST_N;
  const y = (db) => padT + ((dbMax - clamp(db, dbMin, dbMax)) / (dbMax - dbMin)) * ph;
  const x = (i) => padL + (i / (N - 1)) * pw;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = th.grid;
  ctx.lineWidth = 1;
  ctx.font = font(10, true);
  ctx.fillStyle = th.ink3;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let db = dbMax; db >= dbMin; db -= 10) {
    ctx.beginPath();
    ctx.moveTo(padL, y(db) + 0.5);
    ctx.lineTo(w - padR, y(db) + 0.5);
    ctx.stroke();
    if (db % 20 === 0) ctx.fillText(db, padL - 6, y(db));
  }
  for (let s = 0; s <= 10; s += 2) {
    const px = padL + (1 - s / 10) * pw;
    ctx.beginPath();
    ctx.moveTo(px + 0.5, padT);
    ctx.lineTo(px + 0.5, padT + ph);
    ctx.stroke();
  }
  const at = (arr, i) => arr[(hist.idx + i) % N];
  let first = 0;
  while (first < N && !Number.isFinite(at(hist.power, first))) first++;
  const span = (N - 1) * state.blockMs / 1000;

  if (first < N) {
    // threshold band
    ctx.beginPath();
    for (let i = first; i < N; i += 4) ctx.lineTo(x(i), y(at(hist.hi, i)));
    for (let i = N - 1; i >= first; i -= 4) ctx.lineTo(x(i), y(at(hist.lo, i)));
    ctx.closePath();
    ctx.fillStyle = th.thr;
    ctx.globalAlpha = 0.1;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = th.thr;
    ctx.lineWidth = 1;
    for (const arr of [hist.hi, hist.lo]) {
      ctx.beginPath();
      for (let i = first; i < N; i += 4) {
        const px = x(i);
        const py = y(at(arr, i));
        if (i === first) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.restore();
    // trace area + line
    ctx.beginPath();
    ctx.moveTo(x(first), y(dbMin));
    for (let i = first; i < N; i++) ctx.lineTo(x(i), y(at(hist.power, i)));
    ctx.lineTo(x(N - 1), y(dbMin));
    ctx.closePath();
    ctx.fillStyle = th.fill;
    ctx.fill();
    ctx.beginPath();
    for (let i = first; i < N; i++) {
      const px = x(i);
      const py = y(at(hist.power, i));
      if (i === first) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = th.trace;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.stroke();
    const lastDb = at(hist.power, N - 1);
    ctx.beginPath();
    ctx.arc(x(N - 1), y(lastDb), 3, 0, Math.PI * 2);
    ctx.fillStyle = th.trace;
    ctx.fill();
    ctx.fillStyle = th.thr;
    ctx.font = font(10, true);
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("threshold", x(first) + 4, y(at(hist.hi, first)) - 3);
  } else if (!state.everRan) {
    restingText(ctx, w, h, "Tone power at the marker frequency, one point per 10 ms",
      "The dashed band is the detector's threshold; the strip below is its ON/OFF verdict");
  }
  ctx.fillStyle = th.ink3;
  ctx.font = font(10, true);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`−${span.toFixed(0)} s`, padL + 2, padT + 2);
  ctx.textAlign = "right";
  ctx.fillText("now", w - padR - 2, padT + 2);

  if (hoverX !== null && hoverX >= padL && hoverX <= w - padR && first < N) {
    const i = Math.max(first, Math.round(((hoverX - padL) / pw) * (N - 1)));
    const v = at(hist.power, i);
    const ago = (((N - 1 - i) * state.blockMs) / 1000).toFixed(2);
    ctx.save();
    ctx.strokeStyle = th.ink2;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x(i) + 0.5, padT);
    ctx.lineTo(x(i) + 0.5, padT + ph);
    ctx.stroke();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(x(i), y(v), 4, 0, Math.PI * 2);
    ctx.fillStyle = th.trace;
    ctx.fill();
    ctx.strokeStyle = cssVar("--panel");
    ctx.lineWidth = 2;
    ctx.stroke();
    const label = `${v.toFixed(1)} dB · ${ago} s ago${at(hist.on, i) ? " · ON" : ""}`;
    ctx.font = font(11, true);
    const tw = ctx.measureText(label).width + 12;
    let bx = x(i) + 8;
    if (bx + tw > w - padR) bx = x(i) - 8 - tw;
    const by = Math.max(padT, Math.min(padT + ph - 22, y(v) - 26));
    ctx.fillStyle = th.ink;
    ctx.beginPath();
    ctx.roundRect(bx, by, tw, 20, 4);
    ctx.fill();
    ctx.fillStyle = cssVar("--panel");
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, bx + 6, by + 10);
  }
}

function drawStrip() {
  const cv = fit(canvases.stripCanvas);
  const { ctx, w, h } = cv;
  const on = cssVar("--on");
  const off = cssVar("--off");
  const ink3 = cssVar("--ink-3");
  const padL = 40;
  const padR = 12;
  const pw = w - padL - padR;
  const N = HIST_N;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = off;
  ctx.fillRect(padL, 6, pw, h - 12);
  ctx.fillStyle = on;
  let i = 0;
  while (i < N) {
    const v = hist.on[(hist.idx + i) % N];
    if (!v) {
      i++;
      continue;
    }
    let j = i;
    while (j < N && hist.on[(hist.idx + j) % N]) j++;
    ctx.fillRect(padL + (i / N) * pw, 6, Math.max(1, ((j - i) / N) * pw), h - 12);
    i = j;
  }
  ctx.fillStyle = ink3;
  ctx.font = font(10, false);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("ON", padL - 6, h / 2);
}

function drawMeter() {
  const cv = fit(canvases.meterCanvas);
  const { ctx, w, h } = cv;
  const grid = cssVar("--grid");
  const trace = cssVar("--trace");
  const ink = cssVar("--ink");
  const ink3 = cssVar("--ink-3");
  ctx.clearRect(0, 0, w, h);
  const y = (db) => ((0 - db) / 90) * h;
  ctx.fillStyle = grid;
  ctx.fillRect(0, 0, w, h);
  if (state.everRan) {
    const lvl = clamp(state.lvlDb, -90, 0);
    ctx.fillStyle = trace;
    ctx.fillRect(0, y(lvl), w, h - y(lvl));
    if (state.peakHold > -90) {
      ctx.fillStyle = ink;
      ctx.fillRect(0, y(Math.min(0, state.peakHold)) - 1, w, 2);
    }
  }
  for (let db = -20; db > -90; db -= 20) {
    ctx.fillStyle = ink3;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(0, y(db), w, 1);
    ctx.globalAlpha = 1;
  }
}

function drawWave() {
  const cv = fit(canvases.waveCanvas);
  const { ctx, w, h } = cv;
  const spec = cssVar("--spec");
  const grid = cssVar("--grid");
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = grid;
  ctx.beginPath();
  ctx.moveTo(0, h / 2 + 0.5);
  ctx.lineTo(w, h / 2 + 0.5);
  ctx.stroke();
  if (state.running && mic && (!state.paused || !waveHold)) {
    const s = mic.filteredWave(6);
    if (s) {
      if (!waveHold || waveHold.length !== s.length) waveHold = new Float32Array(s.length);
      waveHold.set(s);
    }
  }
  if (!waveHold || !state.everRan) return;
  // Scale to the recent loudest block (the meter's peak hold), so a tone fills
  // the plot, the room between marks stays small and a dead input is flat.
  const ref = Math.max(0.002, Math.pow(10, state.peakHold / 20) * 1.5);
  const n = waveHold.length;
  const amp = h / 2 - 3;
  ctx.beginPath();
  for (let px = 0; px <= w; px++) {
    const pos = (px / w) * (n - 1);
    const i = Math.floor(pos);
    const frac = pos - i;
    const v = i + 1 < n ? waveHold[i] * (1 - frac) + waveHold[i + 1] * frac : waveHold[n - 1];
    const py = h / 2 - clamp(v / ref, -1, 1) * amp;
    if (px) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
  }
  ctx.strokeStyle = spec;
  ctx.lineWidth = 1.25;
  ctx.lineJoin = "round";
  ctx.stroke();
}

function drawHist() {
  const cv = fit(canvases.histCanvas);
  const { ctx, w, h } = cv;
  const trace = cssVar("--trace");
  const grid = cssVar("--grid");
  const ink3 = cssVar("--ink-3");
  ctx.clearRect(0, 0, w, h);
  const bins = 16;
  const maxMs = Math.max(600, decoder.ditMs * 5);
  const counts = new Array(bins).fill(0);
  for (const m of state.recentMarks) counts[Math.min(bins - 1, Math.floor((m / maxMs) * bins))]++;
  const mx = Math.max(1, ...counts);
  const bw = w / bins;
  ctx.fillStyle = grid;
  ctx.fillRect(0, h - 1, w, 1);
  for (let i = 0; i < bins; i++) {
    const bh = (counts[i] / mx) * (h - 12);
    if (counts[i]) {
      ctx.fillStyle = trace;
      ctx.beginPath();
      ctx.roundRect(i * bw + 1, h - 1 - bh, bw - 2, bh, [2, 2, 0, 0]);
      ctx.fill();
    }
  }
  ctx.fillStyle = ink3;
  ctx.font = font(9, true);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("0", 0, 0);
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(maxMs)} ms`, w, 0);
}

// ---------------------------------------------------------------- encoder

const encoder = { text: "", wpm: 8, guide: { timing: [], letters: [], totalMs: 0 }, playheadMs: null };

function buildEncoding() {
  const wpm = clamp(parseFloat(el.encWpm.value) || 8, 2, 40);
  encoder.wpm = wpm;
  encoder.text = el.encIn.value;
  encoder.guide = buildGuide(encoder.text, wpm);
  const morse = encode(encoder.text);
  el.encOut.innerHTML = morse
    ? escapeHtml(morse).split(" / ").join(' <span class="sep">/</span> ')
    : '<span class="sep">Type a message above</span>';
  const T = 1200 / wpm;
  el.encDur.innerHTML = `${(encoder.guide.totalMs / 1000).toFixed(1)}<small>s</small>`;
  el.encDit.textContent = `${roundHalfEven(T)} · ${roundHalfEven(3 * T)} ms`;
  el.encGap.textContent = `${roundHalfEven(3 * T)} · ${roundHalfEven(7 * T)} ms`;
  el.encPlay.disabled = !encoder.guide.totalMs;
  dirty = true;
}

function drawEncGuide() {
  const cv = fit(canvases.encCanvas);
  const { ctx, w, h } = cv;
  const on = cssVar("--on");
  const off = cssVar("--off");
  const ink2 = cssVar("--ink-2");
  const ink = cssVar("--ink");
  ctx.clearRect(0, 0, w, h);
  const { timing, letters, totalMs } = encoder.guide;
  if (!totalMs) return;
  const top = 16;
  const bh = h - top - 6;
  const x = (ms) => (ms / totalMs) * w;
  ctx.fillStyle = off;
  ctx.fillRect(0, top, w, bh);
  let t = 0;
  ctx.fillStyle = on;
  for (const seg of timing) {
    if (seg.on) ctx.fillRect(x(t), top, Math.max(1, x(seg.ms) - 1), bh);
    t += seg.ms;
  }
  ctx.fillStyle = ink2;
  ctx.font = font(11, true);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  // Every letter is labelled, however narrow (a lone dit like E); only a
  // label that would overlap its predecessor is skipped.
  for (const label of layoutGuideLabels(letters, x, (ch) => ctx.measureText(ch).width)) {
    ctx.fillText(label.ch, label.x, 0);
  }
  if (player.playing && encoder.playheadMs !== null) {
    ctx.fillStyle = ink;
    ctx.fillRect(x(encoder.playheadMs) - 1, top - 4, 2, bh + 8);
  }
}

function togglePlay() {
  if (player.playing) {
    player.stop();
    return;
  }
  if (!encoder.guide.totalMs) return;
  try {
    player.audioContext = ensureContext();
    syncPlayerFeed();
    player.play(encoder.guide.timing, state.f0, 0.15);
    encoder.playheadMs = 0;
    el.encPlay.textContent = "Stop";
  } catch {
    el.encPlay.textContent = "Audio unavailable";
    setTimeout(() => {
      el.encPlay.textContent = "Play tone";
    }, 1800);
  }
  dirty = true;
}
/**
 * Mix the encoder's tone into the decoder's input bus while listening and
 * "Feed the decoder" is on. This is the path that works everywhere: the
 * acoustic route (speakers, room, microphone) is broken wherever the OS
 * removes the machine's own output from the microphone, which is what
 * Firefox on Windows gets. Safe to call at any time; it only adds or removes
 * a Web Audio connection.
 */
function syncPlayerFeed() {
  const bus = mic && state.running ? mic.bus : null;
  const want = Boolean(bus) && el.encFeed.checked;
  for (const node of player.outputs) if (node !== bus || !want) player.removeOutput(node);
  if (want) player.addOutput(bus);
}

player.onProgress = (ms) => {
  encoder.playheadMs = ms;
  dirty = true;
};
player.onEnd = () => {
  encoder.playheadMs = null;
  el.encPlay.textContent = "Play tone";
  dirty = true;
};

// ------------------------------------------------------------ DOM readouts

function updateDom() {
  const cur = detector.currentRun;
  const live = state.running && cur.on ? (cur.ms - decoder.offsetMs < 2 * decoder.ditMs ? "·" : "−") : "";
  // While the speed estimate settles the held-back runs are shown dimmed, as
  // they read under the current estimate; the final reading replaces them.
  const pending = decoder.pendingCount > 0;
  const shown = (pending ? decoder.provisional : decoder.buffer).replace(/\./g, "·").replace(/-/g, "−");
  const symbolHtml = pending ? `<span class="prov">${shown}</span>` : shown;
  el.sym.innerHTML = symbolHtml + (live ? `<span class="cur">${live}</span>` : "") || "&nbsp;";
  const guess = decoder.buffer ? lookup(decoder.buffer) : null;
  el.hint.textContent = pending ? "estimating speed…" : decoder.buffer ? `→ ${guess || "?"}` : "";

  if (!decoder.text && !state.everRan) {
    el.text.innerHTML = `<span class="placeholder">Decoded letters appear here. Press Start, then key on the beeper; or press Play tone below to hear the encoder and watch this decoder read it back.</span>`;
  } else {
    const text = decoder.text.length > 2000 ? decoder.text.slice(-2000) : decoder.text;
    el.text.innerHTML = escapeHtml(text) + (state.running && !state.paused ? '<span class="caret"></span>' : "");
    el.text.scrollTop = el.text.scrollHeight;
  }

  const have = state.manual || state.recentMarks.length >= 2;
  el.dit.textContent = have ? `${Math.round(decoder.ditMs)} ms` : "— ms";
  el.dah.textContent = have ? `${Math.round(decoder.ditMs * 3)} ms` : "— ms";
  el.lgap.textContent = have ? `${Math.round(decoder.ditMs * 3)} ms` : "— ms";
  el.off.textContent = have ? `${Math.round(decoder.offsetMs)} ms` : "— ms";
  const on = state.running && detector.state;
  el.pill.className = "pill" + (on ? " on" : "");
  el.pill.innerHTML = `<i></i>${on ? "ON" : "OFF"}`;
  el.lvl.innerHTML = `${state.everRan ? Math.max(-120, state.lvlDb).toFixed(0) : "—"}<small>dBFS</small>`;
  const snr = state.running && !detector.warmingUp ? (detector.peakDb - detector.floorDb).toFixed(0) : "—";
  el.snr.innerHTML = `${snr}<small>dB</small>`;
  el.wpm.innerHTML = `${have ? decoder.wpm.toFixed(1) : "—"}<small>WPM</small>`;
  // In Auto mode the (disabled) speed field follows the live estimate, so a
  // switch to Manual starts from the measured speed.
  if (!state.manual) {
    el.wpmVal.value = decoder.timingReady ? String(Math.round(clamp(decoder.wpm, 2, 40))) : String(state.wpmManual);
  }
  el.cnt.textContent = String(decoder.letterCount);
  el.unk.textContent = String(decoder.unknownCount);
  const s = Math.floor((state.blocks * state.blockMs) / 1000);
  el.clockV.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  const dropped = state.dropped + Math.floor(state.skippedFrames / state.blockSize);
  el.dropV.textContent = `${dropped} dropped${state.badBlocks ? ` · ${state.badBlocks} bad` : ""}`;
}

function drawAll() {
  drawSpectrum();
  drawPower();
  drawStrip();
  drawMeter();
  drawWave();
  drawHist();
  drawEncGuide();
  updateDom();
}

// -------------------------------------------------------- download panel

async function loadReleases() {
  const fallback = () => {
    el.dlAssets.innerHTML =
      '<span class="k">Downloads</span>' +
      `<div>No packaged release is listed yet. See the <a href="${RELEASES_PAGE}">releases page</a> on GitHub.</div>` +
      `<div class="howto">With Python 3.13 installed: <code>pip install git+${escapeHtml(REPO_GIT)}</code> then run <code>morse-console</code>.</div>`;
    el.dlNote.textContent = "Windows build: unzip, run morse-console.exe, no internet needed.";
  };
  try {
    const r = await fetch(RELEASES_API, { headers: { Accept: "application/vnd.github+json" } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const rel = await r.json();
    const assets = Array.isArray(rel.assets) ? rel.assets : [];
    if (!assets.length) throw new Error("no assets");
    const tag = String(rel.tag_name || rel.name || "latest");
    let html = `<span class="k">Downloads · ${escapeHtml(tag)}</span>`;
    for (const a of assets) {
      const url = String(a.browser_download_url || "");
      if (!url.startsWith("https://")) continue;
      html += `<div class="asset"><a href="${escapeHtml(url)}">${escapeHtml(a.name || "download")}</a><span class="size mono">${(Number(a.size) / 1048576).toFixed(1)} MB</span></div>`;
    }
    html += '<div class="howto">Unzip, run <code>morse-console.exe</code>, no internet needed.</div>';
    html += `<div class="muted">Other platforms: the source zip and <code>pip install git+${escapeHtml(REPO_GIT)}</code>; <a href="${RELEASES_PAGE}">all releases</a>.</div>`;
    el.dlAssets.innerHTML = html;
    const when = rel.published_at ? ` · ${new Date(rel.published_at).toLocaleDateString()}` : "";
    el.dlNote.textContent = `Latest release ${tag}${when}.`;
  } catch {
    fallback();
  }
}

// ------------------------------------------------------------------ wiring

function wire() {
  el.start.addEventListener("click", () => {
    if (state.running) stopListening();
    else startListening();
  });
  el.dev.addEventListener("change", () => {
    store.set(STORAGE.device, el.dev.value);
    state.chopDismissed = false;
    if (state.running) restartListening();
  });
  el.f0.addEventListener("change", () => applyFrequency(el.f0.value));
  el.f0.addEventListener("keydown", (e) => {
    if (e.key === "Enter") applyFrequency(el.f0.value);
  });
  el.auto.addEventListener("click", startAutoDetect);
  el.wpmAuto.addEventListener("click", () => setSpeedMode(false));
  el.wpmManual.addEventListener("click", () => setSpeedMode(true));
  el.wpmVal.addEventListener("change", () => {
    if (state.manual) rebuildDecoder();
  });
  el.pause.addEventListener("click", togglePause);
  el.save.addEventListener("click", save30s);
  el.clear.addEventListener("click", clearText);
  el.msgDismiss.addEventListener("click", hideMessage);
  el.chopDismiss.addEventListener("click", () => {
    el.chopHint.hidden = true;
    state.chopDismissed = true;
  });
  el.noteDismiss.addEventListener("click", () => {
    el.noteBar.hidden = true;
  });

  el.encIn.addEventListener("input", () => {
    player.stop();
    buildEncoding();
  });
  el.encWpm.addEventListener("input", () => {
    player.stop();
    buildEncoding();
  });
  el.encPlay.addEventListener("click", togglePlay);
  el.encFeed.addEventListener("change", syncPlayerFeed);
  el.encCopy.addEventListener("click", () => {
    const done = () => {
      el.encCopy.textContent = "Copied";
      setTimeout(() => {
        el.encCopy.textContent = "Copy";
      }, 1500);
    };
    const morse = encode(encoder.text);
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(morse).then(done, done);
    else done();
  });

  if (navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === "function") {
    navigator.mediaDevices.addEventListener("devicechange", () => refreshDevices(el.dev.value));
  }
  new ResizeObserver(() => {
    dirty = true;
  }).observe(el.win);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    dirty = true;
  });
  new MutationObserver(() => {
    dirty = true;
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  document.addEventListener("visibilitychange", () => {
    dirty = true;
  });
  window.addEventListener("pagehide", () => {
    if (mic && state.running) mic.stop();
  });
}

function frameLoop() {
  const frameMs = reduceMotion ? 100 : 1000 / 30;
  let last = 0;
  const frame = (now) => {
    if (now - last >= frameMs - 1) {
      if ((state.running && !state.paused) || player.playing || dirty) {
        drawAll();
        dirty = false;
        last = now;
      }
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function init() {
  const sup = support();
  state.supported = sup.ok;
  if (!sup.ok) {
    showMessage(`${sup.reason} The encoder below still works, and the desktop app decodes offline.`);
  }
  const storedF0 = parseInt(store.get(STORAGE.f0) || "", 10);
  applyFrequency(Number.isFinite(storedF0) && storedF0 >= 100 && storedF0 < 20000 ? storedF0 : DEFAULT_F0);
  el.noteBar.hidden = true; // applyFrequency never fails here, but keep the bar quiet on load
  setSpeedMode(false);
  buildEncoding();
  wire();
  updateControls();
  updateStatus();
  refreshDevices();
  loadReleases();
  frameLoop();
}

init();
