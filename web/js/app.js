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
import { TonePlayer, buildGuide, farnsworthGaps, layoutGuideLabels, roundHalfEven } from "./player.js";
import { MIC_GATE_TAIL_MS, MicGate, MicInput, describeCaptureError, encodeWav, support } from "./audio.js";
import { Keyer, LiveKey } from "./keyer.js";
import { DecodedLog, defaultFilename } from "./declog.js";
import { DEFAULTS as BINDING_DEFAULTS, RESERVED_CODES, actionFor, isDefault as bindingsAreDefault, keyLabel, rebind, sanitize as sanitizeBindings } from "./bindings.js";
import { Practice, rhythm, score } from "./practice.js";
import { cellState, chartEntries } from "./reference.js";
import { sanitize as sanitizeChannels, vibrationPattern } from "./channels.js";
import { Lamp } from "./light.js";
import { CameraInput } from "./camera.js";
import { LightDetector, SourceArbiter } from "./lightdetect.js";
import { Run } from "./runs.js";

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
const REPO_API = "https://api.github.com/repos/jetzhu/MorseCodeAudioCoder";
const STORE_STAR_NUDGE = "morse.star.nudged";
const STORE_CHANNELS = "morse.channels";
const STORE_VIEW = "morse.view";
const STORE_FLASH_NOTICE = "morse.light.notice";
const RELEASES_PAGE = "https://github.com/jetzhu/MorseCodeAudioCoder/releases";
const REPO_GIT = "https://github.com/jetzhu/MorseCodeAudioCoder.git";
const STORAGE = { device: "morse.deviceId", f0: "morse.f0", micProc: "morse.micProcessing" };

// ---------------------------------------------------------------- channels

/** What the page listens with and sends with; any combination, remembered. */
let channels = sanitizeChannels(null);
/** What this device can do; filled in by initChannels(). */
const avail = { listen: { mic: false, camera: false }, send: { audio: true, light: true, torch: false, vibrate: false } };
const listenOn = (k) => Boolean(channels.listen[k] && avail.listen[k]);
/** The microphone is open and delivering blocks. */
const micLive = () => Boolean(mic && mic.running);

// Camera listening: the camera's brightness goes through its own detector;
// the arbiter lets only one source (the first to start a mark) feed the decoder.
/** @type {CameraInput | null} */
let camera = null;
const lightDet = new LightDetector();
const arbiter = new SourceArbiter();
/** Brightness history for the trace: {t, v, on}. */
const camTrace = [];
const camLive = () => Boolean(camera && camera.running);
/** Quiet time after which a source gives up the message: 2 s, or 8 dits at slow speeds. */
const releaseMs = () => Math.max(2000, 8 * decoder.ditMs);
const sendOn = (k) => Boolean(channels.send[k] && avail.send[k]);

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
  start: $("startBtn"), dev: $("dev"), micProc: $("micProc"), f0: $("f0"), auto: $("autoBtn"),
  wpmAuto: $("wpmAuto"), wpmManual: $("wpmManual"), wpmVal: $("wpmVal"),
  pause: $("pauseBtn"), save: $("saveBtn"), log: $("logBtn"), clear: $("clearBtn"),
  msgBar: $("msgBar"), msgText: $("msgText"), msgDismiss: $("msgDismiss"),
  chopHint: $("chopHint"), chopDismiss: $("chopDismiss"),
  noteBar: $("noteBar"), noteText: $("noteText"), noteDismiss: $("noteDismiss"),
  starNudge: $("starNudge"), starNudgeLink: $("starNudgeLink"), starNudgeDismiss: $("starNudgeDismiss"),
  chipMic: $("chipMic"), chipCamera: $("chipCamera"), chipAudio: $("chipAudio"), chipLight: $("chipLight"),
  chipTorch: $("chipTorch"), chipVibrate: $("chipVibrate"), lightFullBtn: $("lightFullBtn"), lightFull: $("lightFull"),
  flashNotice: $("flashNotice"), flashOk: $("flashOk"), flashCancel: $("flashCancel"),
  viewDesktop: $("viewDesktop"), viewHandset: $("viewHandset"), moreBtn: $("moreBtn"),
  camStrip: $("camStrip"), camView: $("camView"), camVideo: $("camVideo"), camSpot: $("camSpot"),
  camPill: $("camPill"), camLevel: $("camLevel"), camRange: $("camRange"), camFps: $("camFps"), camMax: $("camMax"),
  f0Label: $("f0Label"), specInfo: $("specInfo"),
  sym: $("symBuf"), hint: $("symHint"), text: $("textOut"),
  dit: $("ditV"), dah: $("dahV"), lgap: $("lgapV"), off: $("offV"),
  pill: $("statePill"), lvl: $("lvlV"), snr: $("snrV"), wpm: $("wpmV"), cnt: $("cntV"), unk: $("unkV"),
  encIn: $("encIn"), encWpm: $("encWpm"), encFarns: $("encFarns"), encPlay: $("encPlay"), encCopy: $("encCopy"), encOut: $("encOut"),
  encFeed: $("encFeed"),
  keyStraight: $("keyStraight"), keyPaddle: $("keyPaddle"), keyBug: $("keyBug"), keyPill: $("keyPill"), keyHelp: $("keyHelp"),
  iambicA: $("iambicA"), iambicB: $("iambicB"), keyRatio: $("keyRatio"), keyWeight: $("keyWeight"),
  keypadStraight: $("keypadStraight"), keypadPaddle: $("keypadPaddle"),
  keyBtn: $("keyBtn"), ditBtn: $("ditBtn"), dahBtn: $("dahBtn"),
  keyRebind: $("keyRebind"), ditRebind: $("ditRebind"), dahRebind: $("dahRebind"), keyReset: $("keyReset"),
  keySidetone: $("keySidetone"), keySide: $("keySide"),
  refToggle: $("refToggle"), refBody: $("refBody"), refGrid: $("refGrid"),
  sentOut: $("sentOut"), sentClear: $("sentClear"), keySpeed: $("keySpeed"), keyDit: $("keyDit"),
  prWords: $("prWords"), prCalls: $("prCalls"), prDigits: $("prDigits"), prMixed: $("prMixed"),
  prNext: $("prNext"), prCheck: $("prCheck"), prShowCode: $("prShowCode"), prTarget: $("prTarget"), prCode: $("prCode"),
  prResult: $("prResult"), prAcc: $("prAcc"), prWpm: $("prWpm"), prRhythm: $("prRhythm"), prSession: $("prSession"),
  encDur: $("encDur"), encDit: $("encDit"), encGap: $("encGap"), encTone: $("encTone"),
  stateDot: $("stateDot"), stateV: $("stateV"), blockV: $("blockV"), dropV: $("dropV"), clockV: $("clockV"), rateV: $("rateV"),
  titleDev: $("titleDev"), factTone: $("factTone"), factRate: $("factRate"), factBlock: $("factBlock"),
  dlAssets: $("dlAssets"), dlNote: $("dlNote"), win: $("win"),
};

const state = {
  f0: DEFAULT_F0,
  supported: true,
  running: false,
  startedAt: 0, // performance.now() at Start, the clock for camera-only logs
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
  micProcessing: "unknown", // what the browser reported applying to the open microphone
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
  if (!state.running || !listenOn("mic")) return;
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

/** Every character the decoder emitted, with when: exported by Download log. */
const log = new DecodedLog();

/** Stamp newly decoded text with the audio clock and the computer clock. */
function logEmit(text) {
  if (!text) return;
  const audioMs = micLive() ? state.blocks * state.blockMs : performance.now() - (state.startedAt || performance.now());
  log.add(text, audioMs, Date.now());
  if (text.includes(" ") && log.text.trim()) maybeStarNudge(); // a whole word decoded
}

// ------------------------------------------------------------------- stars

/**
 * One quiet request to star the repository, at a moment the page has just
 * done something for the person (first decoded word, first correct practice
 * target). Shown once per browser; dismissing or following the link ends it.
 */
function maybeStarNudge() {
  if (store.get(STORE_STAR_NUDGE) === "1" || !el.starNudge.hidden) return;
  el.starNudge.hidden = false;
}

function endStarNudge() {
  store.set(STORE_STAR_NUDGE, "1");
  el.starNudge.hidden = true;
}

/** Read the repository's star count from GitHub and show it next to the star links. */
async function loadStarCount() {
  try {
    const r = await fetch(REPO_API, { headers: { Accept: "application/vnd.github+json" } });
    if (!r.ok) return;
    const info = await r.json();
    const n = Number(info.stargazers_count);
    if (!Number.isFinite(n)) return;
    for (const span of document.querySelectorAll("[data-starcount]")) span.textContent = `${n} ${n === 1 ? "star" : "stars"}`;
  } catch {
    /* offline or rate-limited: the link stands without a count */
  }
}

/** The `Pipeline.process_block` order: detector, decoder feed, idle, then the display buffers. */
function processBlock(powerDb, rmsDb) {
  // A loud but broadband block (click, speech) never switches the detector ON.
  for (const run of detector.update(powerDb, isTonal(powerDb, rmsDb))) feedFrom("mic", run);
  const current = detector.currentRun;
  arbiter.tick("mic", current.on, current.on ? 0 : current.ms, releaseMs());
  if (!current.on && arbiter.mayIdle("mic")) logEmit(decoder.idle(current.ms));
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

/** A run from one listen source: it reaches the decoder only if that source owns the message. */
function feedFrom(src, run) {
  if (arbiter.offer(src, run)) feedRun(run);
}

/** Hand one final run to the decoder and remember marks for the histogram. */
function feedRun(run) {
  logEmit(decoder.feed(run));
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
  log.reset();
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
  if (state.starting || state.running) return;
  const wantMic = listenOn("mic");
  const wantCam = listenOn("camera");
  if (!wantMic && !wantCam) {
    showNote("Select a source to listen with (Listen with: Microphone or Camera).");
    return;
  }
  state.starting = true;
  el.start.disabled = true;
  el.start.textContent = "Starting…";
  hideMessage();
  updateStatus();
  const errors = [];
  let micOk = false;
  if (wantMic) {
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
      await mic.start(el.dev.value, onBlock, { processing: el.micProc.value === "browser" ? "browser" : "raw" });
      micOk = true;
    } catch (err) {
      errors.push(describeCaptureError(err));
    }
  }
  let camOk = false;
  if (wantCam) {
    try {
      await startCamera();
      camOk = true;
    } catch (err) {
      errors.push(describeCameraError(err));
    }
  }
  if (!micOk && !camOk) {
    state.starting = false;
    showMessage(errors.join(" "));
    updateControls();
    updateStatus();
    return;
  }
  if (errors.length) showNote(`${errors.join(" ")} Listening with the ${micOk ? "microphone" : "camera"} only.`, 9000);
  if (micOk) {
    state.fs = mic.sampleRate;
    state.blockSize = mic.blockSize;
    state.blockMs = mic.blockMs;
    state.deviceLabel = mic.deviceLabel;
    state.micProcessing = mic.appliedProcessing;
    detector = new ToneDetector({ blockMs: state.blockMs });
  }
  resetStreamState();
  arbiter.release();
  state.startedAt = performance.now();
  state.running = true;
  state.paused = false;
  state.starting = false;
  state.everRan = true;
  el.pause.textContent = "Pause";
  if (micOk) {
    syncPlayerFeed(); // a tone already playing reaches the decoder from now on
    // Permission is granted now, so the device list has labels.
    await refreshDevices(el.dev.value === "" ? mic.deviceId : el.dev.value);
    if (state.f0 >= state.fs / 2) applyFrequency(DEFAULT_F0);
  }
  updateControls();
  updateStatus();
  dirty = true;
}

// ------------------------------------------------------------------ camera

function describeCameraError(err) {
  const name = err && err.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return "Camera access was denied; allow it for this site to decode light.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found.";
  if (name === "NotReadableError") return "The camera is in use by another application.";
  return `The camera could not be opened: ${(err && err.message) || err}.`;
}

async function startCamera() {
  if (!camera) {
    camera = new CameraInput(el.camVideo);
    camera.onFrame = onCameraFrame;
    camera.onEnded = () => {
      showNote("The camera stream ended.");
      stopCamera();
      updateStatus();
    };
    el.camVideo.addEventListener("loadedmetadata", sizeCamView);
  }
  lightDet.reset();
  camTrace.length = 0;
  await camera.start();
  el.camStrip.hidden = false;
  sizeCamView();
  dirty = true;
}

function stopCamera() {
  if (!camLive()) return;
  for (const run of lightDet.flush(performance.now())) feedFrom("camera", run);
  camera.stop();
  arbiter.release("camera");
  el.camStrip.hidden = true;
  el.camPill.className = "pill";
  el.camPill.innerHTML = "<i></i>OFF";
  dirty = true;
}

/** Match the preview box to the video's shape and size the spot marker like the sampled spot. */
function sizeCamView() {
  const v = el.camVideo;
  if (!(v.videoWidth > 0 && v.videoHeight > 0)) return;
  el.camView.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
  const frac = camera ? camera.spotFrac : 0.06;
  const d = (frac * Math.min(v.videoWidth, v.videoHeight)) / v.videoWidth;
  el.camSpot.style.width = `${Math.max(4, d * 100 * 1.6)}%`;
}

function onCameraFrame(f) {
  if (!state.running || state.paused || !listenOn("camera")) return;
  for (const run of lightDet.update(f.value, f.t)) feedFrom("camera", run);
  const off = lightDet.currentOffMs(f.t);
  arbiter.tick("camera", lightDet.state, off, releaseMs());
  if (off > 0 && arbiter.mayIdle("camera")) logEmit(decoder.idle(off));
  camTrace.push({ t: f.t, v: f.value, on: lightDet.state });
  while (camTrace.length && f.t - camTrace[0].t > 10000) camTrace.shift();
  dirty = true;
}

function setCamSpot(e) {
  if (!camera) return;
  const r = el.camView.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = (e.clientY - r.top) / r.height;
  camera.setSpot(x, y);
  el.camSpot.style.left = `${Math.min(1, Math.max(0, x)) * 100}%`;
  el.camSpot.style.top = `${Math.min(1, Math.max(0, y)) * 100}%`;
  lightDet.reset(); // learn the new spot's dark and bright levels afresh
  camTrace.length = 0;
}

function drawCamera() {
  if (el.camStrip.hidden) return;
  const on = lightDet.state;
  el.camPill.className = "pill" + (on ? " on" : "");
  el.camPill.innerHTML = `<i></i>${on ? "ON" : "OFF"}`;
  el.camSpot.classList.toggle("on", on);
  const last = camTrace[camTrace.length - 1];
  el.camLevel.textContent = last ? `${Math.round(last.v)}` : "—";
  el.camRange.textContent = lightDet.contrastOk ? `${Math.round(lightDet.dark)} · ${Math.round(lightDet.bright)}` : "—";
  const fps = lightDet.fps;
  el.camFps.innerHTML = `${fps ? fps.toFixed(0) : "—"}<small>fps</small>`;
  el.camMax.innerHTML = `${fps ? Math.floor(0.4 * fps) : "—"}<small>WPM</small>`;
  const cv = fit(canvases.camCanvas);
  const { ctx, w, h } = cv;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = cssVar("--panel-2");
  ctx.fillRect(0, 0, w, h);
  if (!camTrace.length) return;
  const tEnd = camTrace[camTrace.length - 1].t;
  const x = (t) => w - ((tEnd - t) / 10000) * w;
  const stripH = 8;
  const y = (v) => h - stripH - 4 - (v / 255) * (h - stripH - 10);
  ctx.fillStyle = cssVar("--on");
  for (let i = 1; i < camTrace.length; i++) {
    if (camTrace[i - 1].on) ctx.fillRect(x(camTrace[i - 1].t), h - stripH, Math.max(1, x(camTrace[i].t) - x(camTrace[i - 1].t)), stripH);
  }
  if (lightDet.contrastOk) {
    const c = lightDet.bright - lightDet.dark;
    ctx.strokeStyle = cssVar("--thr");
    ctx.setLineDash([4, 3]);
    for (const level of [lightDet.dark + lightDet.hiFrac * c, lightDet.dark + lightDet.loFrac * c]) {
      ctx.beginPath();
      ctx.moveTo(0, y(level) + 0.5);
      ctx.lineTo(w, y(level) + 0.5);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = cssVar("--trace");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  camTrace.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.v)) : ctx.moveTo(x(p.t), y(p.v))));
  ctx.stroke();
  ctx.lineWidth = 1;
}

async function stopListening() {
  if (!state.running) return;
  state.running = false;
  state.paused = false;
  syncPlayerFeed(); // detach the tone from the bus before the graph is torn down
  if (micLive()) for (const run of detector.flush()) feedFrom("mic", run);
  stopCamera();
  logEmit(decoder.idle(Infinity));
  arbiter.release();
  finishAuto(false);
  if (micLive()) await mic.stop();
  el.pause.textContent = "Pause";
  updateControls();
  updateStatus();
  dirty = true;
}

async function restartListening() {
  await stopListening();
  await startListening();
}

/** " · raw" or " · processed" once the browser has told us, else nothing. */
function micProcessingText() {
  if (state.micProcessing === "raw") return " · raw";
  if (state.micProcessing === "processed") return " · processed";
  return "";
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
  if (liveKey.running) liveKey.setFrequency(value);
  if (el.keySide) applySidetone(false);
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
  logEmit(decoder.idle(Infinity)); // commit the pending letter under the old rule
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

/** Hand `blob` to the browser as a file download named `name`. */
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function logHeader() {
  const dev = el.dev && el.dev.selectedOptions && el.dev.selectedOptions[0];
  const source = dev && dev.textContent ? dev.textContent.trim() : "microphone";
  return [["Source", source], ["Tone", `${state.f0} Hz`], ["App", "Beeper Morse Console (web)"]];
}

function downloadLog() {
  if (log.isEmpty) {
    showNote("Nothing decoded yet.");
    return;
  }
  const now = Date.now();
  downloadBlob(new Blob([log.renderText(logHeader(), { nowMs: now })], { type: "text/plain;charset=utf-8" }), defaultFilename(now));
  el.log.textContent = "Saved";
  setTimeout(() => {
    el.log.textContent = "Download log";
  }, 1800);
}

async function save30s() {
  if (!mic) return;
  el.save.disabled = true;
  el.save.textContent = "Saving…";
  try {
    const samples = await mic.dump30s();
    const fs = mic.running ? mic.sampleRate : mic.lastDumpRate || mic.sampleRate;
    if (!samples.length) throw new Error("no audio captured yet");
    downloadBlob(encodeWav(samples, fs), `morse_${timestamp()}.wav`);
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
  const canListen = listenOn("mic") || listenOn("camera");
  el.start.disabled = state.starting || (!running && !canListen);
  el.start.title = !running && !canListen ? "Select a source to listen with first" : "Open the selected sources and start decoding";
  el.pause.disabled = !running;
  el.auto.disabled = !micLive();
  el.save.disabled = !(running || (mic && mic.lastDump));
  el.encPlay.disabled = !encoder.guide.totalMs;
}

function updateStatus() {
  let dot = "○";
  let cls = "dot";
  let text;
  if (!state.supported) text = "Unavailable in this browser";
  else if (state.starting) text = "Opening…";
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
  el.stateV.textContent = state.running && camLive() ? `${text} · ${micLive() ? "microphone + camera" : "camera"}` : text;
  if (state.fs) {
    const khz = (state.fs / 1000).toFixed(state.fs % 1000 ? 1 : 0);
    el.blockV.textContent = `${state.blockSize}-sample blocks · ${state.blockMs.toFixed(state.blockMs % 1 ? 2 : 0)} ms`;
    el.rateV.textContent = `${khz} kHz`;
    el.factRate.textContent = `${khz} kHz`;
    el.factBlock.textContent = `${state.blockSize} samples`;
    el.specInfo.textContent = `0–8 kHz · 2048-point FFT · ${((2048 / state.fs) * 1000).toFixed(0)} ms`;
    el.titleDev.textContent = state.running || state.everRan
      ? `${state.deviceLabel || "Default microphone"} · ${khz} kHz${micProcessingText()}${state.running ? "" : " · closed"}`
      : "microphone closed";
  } else {
    el.blockV.textContent = "10 ms blocks";
    el.titleDev.textContent = "microphone closed";
  }
}

// ----------------------------------------------------------- auto-detect

const auto = { active: false, acc: null, frames: 0, timer: 0, started: 0 };

function startAutoDetect() {
  if (auto.active || !micLive()) return;
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
["specCanvas", "powerCanvas", "stripCanvas", "meterCanvas", "waveCanvas", "histCanvas", "encCanvas", "camCanvas"].forEach(setup);

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

  if (micLive()) {
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
  if (micLive() && (!state.paused || !waveHold)) {
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

const encoder = { text: "", wpm: 8, farnsworth: null, guide: { timing: [], letters: [], totalMs: 0 }, playheadMs: null };

function buildEncoding() {
  const wpm = clamp(parseFloat(el.encWpm.value) || 8, 2, 40);
  encoder.wpm = wpm;
  keyer.setSpeed(1200 / wpm);
  updateKeyReadouts();
  const farnsRaw = parseFloat(el.encFarns.value);
  encoder.farnsworth = Number.isFinite(farnsRaw) && farnsRaw >= 2 && farnsRaw < wpm ? farnsRaw : null;
  encoder.text = el.encIn.value;
  encoder.guide = buildGuide(encoder.text, wpm, encoder.farnsworth);
  const morse = encode(encoder.text);
  el.encOut.innerHTML = morse
    ? escapeHtml(morse).split(" / ").join(' <span class="sep">/</span> ')
    : '<span class="sep">Type a message above</span>';
  const T = 1200 / wpm;
  el.encDur.innerHTML = `${(encoder.guide.totalMs / 1000).toFixed(1)}<small>s</small>`;
  el.encDit.textContent = `${roundHalfEven(T)} · ${roundHalfEven(3 * T)} ms`;
  const gaps = farnsworthGaps(wpm, encoder.farnsworth);
  el.encGap.textContent = `${gaps.letterGap} · ${gaps.wordGap} ms`;
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
    vibrateTiming(encoder.guide.timing);
    encoder.playheadMs = 0;
    el.encPlay.textContent = "Stop";
    updateMicGate();
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
  const bus = micLive() && state.running ? mic.bus : null;
  const want = Boolean(bus) && el.encFeed.checked;
  for (const source of [player, liveKey]) {
    for (const node of source.outputs) if (node !== bus || !want) source.removeOutput(node);
    if (want) source.addOutput(bus);
  }
  updateMicGate();
}

/**
 * The microphone is turned down in the decoder's input while the page's own
 * sound is fed to the decoder, and for a short tail after it stops, so the
 * speaker echo a raw microphone hears cannot fill the gaps between marks.
 * Re-evaluated on every player and key event and, while active, on a timer,
 * so the tail always starts from the true end of the sound.
 */
const micGate = new MicGate();
/** @type {ReturnType<typeof setTimeout> | null} */
let micGateTimer = null;

function pageSounding() {
  return player.playing || (liveKey.running && liveKey.isOn());
}

function updateMicGate() {
  if (micGateTimer !== null) {
    clearTimeout(micGateTimer);
    micGateTimer = null;
  }
  if (!micLive() || !state.running || !el.encFeed.checked || !sendOn("audio") || !listenOn("mic")) {
    micGate.reset();
    if (mic) mic.setGate(false);
    return;
  }
  const sounding = pageSounding();
  const attenuate = micGate.update(sounding, mic.context.currentTime * 1000);
  mic.setGate(attenuate);
  if (attenuate) micGateTimer = setTimeout(updateMicGate, sounding ? 100 : MIC_GATE_TAIL_MS + 20);
}

player.onProgress = (ms) => {
  if (!ref.playing) encoder.playheadMs = ms; // a chart character has no place on the keying guide
  dirty = true;
};
player.onEnd = () => {
  encoder.playheadMs = null;
  ref.playing = false;
  el.encPlay.textContent = "Play tone";
  updateMicGate();
  stopVibration();
  dirty = true;
};

// -------------------------------------------------------------------- key

/** The hand key: a straight key on Space or the button, or two paddles on the arrow keys. */
const keyer = new Keyer(1200 / 8, "straight");
const liveKey = new LiveKey(keyer, { gain: 0.15 });
liveKey.onChange = () => {
  updateMicGate();
  updateKeyVibration();
  dirty = true;
};

// ------------------------------------------------------- light and vibration

/** The signal lamp on every layout, and the full-screen overlay. */
const lamp = new Lamp(document.querySelectorAll(".lamp"), el.lightFull);

/** Whether the page is sending a mark right now (Play or the key). */
function sendingNow() {
  return (player.playing && player.stateAtNow()) || (liveKey.running && liveKey.isOn());
}

function updateLamp() {
  const rx = state.running && !state.paused && ((listenOn("mic") && micLive() && detector.state) || (listenOn("camera") && camLive() && lightDet.state));
  const tx = sendOn("light") && sendingNow();
  lamp.setState(rx, tx);
}

function vibrateTiming(timing) {
  if (!sendOn("vibrate")) return;
  try {
    navigator.vibrate(vibrationPattern(timing));
  } catch {
    /* not allowed without a gesture, or unsupported */
  }
}

function stopVibration() {
  if (!avail.send.vibrate) return;
  try {
    navigator.vibrate(0);
  } catch {
    /* ignore */
  }
}

/** Follow the hand key: vibrate for the element in progress, stop when it ends. */
function updateKeyVibration() {
  if (!sendOn("vibrate") || !liveKey.running) return;
  const on = liveKey.isOn();
  try {
    if (!on) navigator.vibrate(0);
    else if (keyer.mode === "straight") navigator.vibrate(30000); // until the key is let go
    else navigator.vibrate(Math.max(10, Math.round((keyer.nextWakeupMs ?? liveKey.nowMs) - liveKey.nowMs)));
  } catch {
    /* ignore */
  }
}

function requestFullLight() {
  if (store.get(STORE_FLASH_NOTICE) === "1") {
    enterFullLight();
    return;
  }
  el.flashNotice.hidden = false;
  el.flashOk.focus();
}

function enterFullLight() {
  el.flashNotice.hidden = true;
  lamp.enterFull();
  updateLamp();
  const root = document.documentElement;
  if (typeof root.requestFullscreen === "function") root.requestFullscreen().catch(() => {});
}

function exitFullLight() {
  lamp.exitFull();
  if (document.fullscreenElement && typeof document.exitFullscreen === "function") document.exitFullscreen().catch(() => {});
}

// ----------------------------------------------------------------- channels

function initChannels() {
  avail.listen.mic = state.supported;
  avail.listen.camera = CameraInput.supported() && window.isSecureContext !== false;
  avail.send.vibrate = typeof navigator.vibrate === "function";
  try {
    channels = sanitizeChannels(JSON.parse(store.get(STORE_CHANNELS) || "null"));
  } catch {
    channels = sanitizeChannels(null);
  }
  const chips = [
    ["listen", "mic", el.chipMic], ["listen", "camera", el.chipCamera], ["send", "audio", el.chipAudio],
    ["send", "light", el.chipLight], ["send", "torch", el.chipTorch], ["send", "vibrate", el.chipVibrate],
  ];
  for (const [group, key, chip] of chips) chip.addEventListener("click", () => toggleChannel(group, key));
  el.chipVibrate.title = avail.send.vibrate ? el.chipVibrate.title : "Vibration: not available on this device";
  applyChannels(false);
}

function toggleChannel(group, key) {
  if (!avail[group][key]) return;
  channels[group][key] = !channels[group][key];
  applyChannels(true);
  // While listening, a listen source can be added or dropped without a restart.
  if (state.running && group === "listen" && key === "camera") {
    if (listenOn("camera")) startCamera().catch((err) => showNote(describeCameraError(err)));
    else stopCamera();
  }
  if (group === "listen" && key === "mic" && !listenOn("mic")) arbiter.release("mic");
  updateStatus();
}

function applyChannels(persist) {
  if (persist) store.set(STORE_CHANNELS, JSON.stringify(channels));
  const chips = { mic: el.chipMic, camera: el.chipCamera, audio: el.chipAudio, light: el.chipLight, torch: el.chipTorch, vibrate: el.chipVibrate };
  for (const [key, chip] of Object.entries(chips)) {
    const group = key === "mic" || key === "camera" ? "listen" : "send";
    chip.disabled = !avail[group][key];
    chip.setAttribute("aria-pressed", String(Boolean(channels[group][key] && avail[group][key])));
  }
  player.setSpeakers(sendOn("audio"));
  liveKey.setSpeakers(sendOn("audio"));
  if (!sendOn("vibrate")) stopVibration();
  el.lightFullBtn.disabled = !sendOn("light");
  if (!sendOn("light") && lamp.fullActive) exitFullLight();
  updateMicGate();
  updateControls();
  updateLamp();
  dirty = true;
}

// --------------------------------------------------------------------- view

/** "desktop" or "handset": the full console or the compact phone layout. */
function setView(view, persist = true) {
  const handset = view === "handset";
  document.body.classList.toggle("view-handset", handset);
  if (!handset) document.body.classList.remove("more-open");
  el.viewDesktop.setAttribute("aria-pressed", String(!handset));
  el.viewHandset.setAttribute("aria-pressed", String(handset));
  if (persist) store.set(STORE_VIEW, handset ? "handset" : "desktop");
  ref.lastBuffer = null;
  dirty = true;
}

function initialView() {
  const fromUrl = new URLSearchParams(location.search).get("view");
  if (fromUrl === "handset" || fromUrl === "desktop") return fromUrl;
  const stored = store.get(STORE_VIEW);
  if (stored === "handset" || stored === "desktop") return stored;
  const narrow = matchMedia("(max-width: 700px)").matches;
  const touch = matchMedia("(pointer: coarse)").matches;
  return narrow && touch ? "handset" : "desktop";
}
/** Local reading of what the operator keyed, independent of the decoder above. */
const sent = { dec: new MorseDecoder(), index: 0, lastT: null, lastOn: false, /** @type {Run[]} */ runs: [] };
/** Which keyboard key works the straight key and each paddle; kept in this browser. */
const STORE_BINDINGS = "morse.key.bindings";
const STORE_SIDETONE = "morse.key.sidetone";
const STORE_IAMBIC = "morse.key.iambic";
const STORE_RATIO = "morse.key.ratio";
const STORE_WEIGHT = "morse.key.weight";
const STORE_REF_OPEN = "morse.reference.open";
const DEFAULT_SIDETONE_HZ = 600;
let bindings = loadBindings();
/** The action whose key is being chosen after Change key, else null. @type {"key" | "dit" | "dah" | null} */
let capture = null;
/** Code whose release must not key anything (it was just captured). @type {string | null} */
let swallowRelease = null;
const ACTION_TITLES = { key: "the key", dit: "Dit", dah: "Dah" };
const ACTION_BUTTONS = () => ({ key: el.keyBtn, dit: el.ditBtn, dah: el.dahBtn });

function loadBindings() {
  try {
    return sanitizeBindings(JSON.parse(store.get(STORE_BINDINGS) || "null"));
  } catch {
    return { ...BINDING_DEFAULTS };
  }
}

function saveBindings() {
  store.set(STORE_BINDINGS, JSON.stringify(bindings));
}

function keyHelpText(mode) {
  const b = (a) => `<b>${escapeHtml(keyLabel(bindings[a]))}</b>`;
  if (mode === "paddle") return `${b("dit")} sends dits and ${b("dah")} sends dahs at the encoder speed; hold to repeat, hold both to alternate.`;
  if (mode === "bug") return `${b("dit")} sends dits while held; ${b("dah")} keys the tone by hand, like a bug's dah lever.`;
  return `Hold ${b("key")} or the button: the tone sounds while it is held.`;
}

/** Iambic A or B for the two-key keyer; kept in this browser. */
function setIambic(which, persist = true) {
  liveKey.setIambic(which);
  el.iambicA.setAttribute("aria-pressed", String(which === "A"));
  el.iambicB.setAttribute("aria-pressed", String(which === "B"));
  if (persist) store.set(STORE_IAMBIC, which);
}

/** Dah ratio and weight from their fields (out-of-range values fall back to the standard 3 and 50). */
function applyWeighting(persist = true) {
  let ratio = parseFloat(el.keyRatio.value);
  if (!(Number.isFinite(ratio) && ratio >= 2 && ratio <= 5)) ratio = 3;
  ratio = Math.round(ratio * 10) / 10;
  let weight = parseFloat(el.keyWeight.value);
  if (!(Number.isFinite(weight) && weight >= 25 && weight <= 75)) weight = 50;
  weight = Math.round(weight);
  liveKey.setWeighting(ratio, weight);
  if (persist) {
    store.set(STORE_RATIO, String(ratio));
    store.set(STORE_WEIGHT, String(weight));
  }
  updateKeyReadouts();
}

/** Button captions, help line and Reset state after a binding change. */
function renderBindings() {
  el.keyBtn.innerHTML = `Key<small>${escapeHtml(keyLabel(bindings.key))}</small>`;
  el.ditBtn.innerHTML = `Dit<small>${escapeHtml(keyLabel(bindings.dit))}</small>`;
  el.dahBtn.innerHTML = `Dah<small>${escapeHtml(keyLabel(bindings.dah))}</small>`;
  el.keyReset.disabled = bindingsAreDefault(bindings);
  for (const [action, btn] of Object.entries({ key: el.keyRebind, dit: el.ditRebind, dah: el.dahRebind })) {
    btn.classList.toggle("armed", capture === action);
    btn.textContent = capture === action ? "Press a key\u2026" : "Change key";
  }
  el.keyHelp.classList.toggle("capture", capture !== null);
  el.keyHelp.innerHTML = capture
    ? `Press the key that should work <b>${ACTION_TITLES[capture]}</b> (Esc keeps the current one)`
    : keyHelpText(keyer.mode);
}

function startCapture(action) {
  capture = capture === action ? null : action;
  renderBindings();
}

function finishCapture(code) {
  const action = capture;
  capture = null;
  if (action && code) {
    bindings = rebind(bindings, action, code);
    saveBindings();
  }
  renderBindings();
}

function resetBindings() {
  capture = null;
  bindings = { ...BINDING_DEFAULTS };
  saveBindings();
  renderBindings();
}

/** Sidetone pitch from the field: a number in range, or null to follow the beeper frequency. */
function readSidetone() {
  const v = parseFloat(el.keySidetone.value);
  return Number.isFinite(v) && v >= 100 && v <= 4000 ? v : null;
}

function applySidetone(persist) {
  const hz = readSidetone();
  liveKey.setSidetone(hz);
  el.keySide.innerHTML = hz === null ? `${Math.round(state.f0)}<small>Hz (tone)</small>` : `${Math.round(hz)}<small>Hz</small>`;
  if (persist) store.set(STORE_SIDETONE, hz === null ? "" : String(hz));
}

function ensureLiveKey() {
  try {
    const ac = ensureContext();
    if (!liveKey.running) liveKey.start(ac, state.f0);
    syncPlayerFeed();
    return true;
  } catch (err) {
    showMessage(`The key needs Web Audio: ${err.message}`);
    return false;
  }
}

function setKeyMode(mode) {
  liveKey.setMode(mode);
  const paddles = mode === "paddle" || mode === "bug";
  el.keyStraight.setAttribute("aria-pressed", String(mode === "straight"));
  el.keyPaddle.setAttribute("aria-pressed", String(mode === "paddle"));
  el.keyBug.setAttribute("aria-pressed", String(mode === "bug"));
  el.keypadStraight.hidden = paddles;
  el.keypadPaddle.hidden = !paddles;
  el.iambicA.disabled = mode !== "paddle";
  el.iambicB.disabled = mode !== "paddle";
  capture = null;
  renderBindings();
  for (const b of [el.keyBtn, el.ditBtn, el.dahBtn]) b.classList.remove("down");
  dirty = true;
}

function updateKeyReadouts() {
  el.keySpeed.innerHTML = `${encoder.wpm}<small>WPM</small>`;
  el.keyDit.textContent = `${Math.round(keyer.markMs("dit"))} · ${Math.round(keyer.markMs("dah"))} ms`;
}

function isTypingTarget(target) {
  if (!target || !target.tagName) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || Boolean(target.isContentEditable);
}

/**
 * Keyboard: the bound keys work the straight key (single-key mode) or the
 * paddles (two-key mode). While Change key is armed the next press is captured
 * instead: Esc cancels, modifiers are ignored, and the captured key's release
 * is swallowed so it does not key anything.
 */
function onKeyboard(e, down) {
  if (isTypingTarget(e.target)) return;
  if (e.code === "Escape" && lamp.fullActive) {
    if (down) exitFullLight();
    e.preventDefault();
    return;
  }
  if (capture !== null) {
    e.preventDefault();
    if (!down || e.repeat) return;
    if (e.code === "Escape") finishCapture(null);
    else if (e.code && !RESERVED_CODES.has(e.code)) {
      swallowRelease = e.code;
      finishCapture(e.code);
    }
    return;
  }
  if (swallowRelease !== null && e.code === swallowRelease) {
    e.preventDefault();
    if (!down) swallowRelease = null;
    return;
  }
  const action = actionFor(bindings, e.code);
  if (!action) return;
  if (action === "key" ? keyer.mode !== "straight" : keyer.mode === "straight") return;
  e.preventDefault();
  if (down && e.repeat) return;
  if (down && !ensureLiveKey()) return;
  if (!liveKey.running) return;
  if (action === "key") {
    if (down) liveKey.keyDown();
    else liveKey.keyUp();
    el.keyBtn.classList.toggle("down", down);
  } else {
    if (down) liveKey.paddleDown(action);
    else liveKey.paddleUp(action);
    ACTION_BUTTONS()[action].classList.toggle("down", down);
  }
  dirty = true;
}

/** Mouse and touch on a key button: press and hold. */
function bindKeyButton(btn, onDown, onUp) {
  const release = () => {
    if (!btn.classList.contains("down")) return;
    btn.classList.remove("down");
    onUp();
    dirty = true;
  };
  btn.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    if (!ensureLiveKey()) return;
    btn.classList.add("down");
    try {
      btn.setPointerCapture(e.pointerId);
    } catch {
      /* capture is a nicety */
    }
    onDown();
    dirty = true;
  });
  btn.addEventListener("pointerup", release);
  btn.addEventListener("pointercancel", release);
  btn.addEventListener("lostpointercapture", release);
  btn.addEventListener("contextmenu", (e) => e.preventDefault());
}

function releaseKeys() {
  if (liveKey.running) liveKey.releaseAll();
  for (const b of [el.keyBtn, el.ditBtn, el.dahBtn]) b.classList.remove("down");
  dirty = true;
}

/** Turn new keyer transitions into runs for the local Sent decoder. */
function trackSent(nowMs) {
  const { items, index } = keyer.transitionsSince(sent.index);
  sent.index = index;
  for (const [t, on] of items) {
    if (sent.lastT !== null && t > sent.lastT) {
      const run = new Run(sent.lastOn, Math.max(1, Math.round((t - sent.lastT) / 10)));
      sent.runs.push(run);
      sent.dec.feed(run);
    }
    sent.lastT = t;
    sent.lastOn = on;
  }
  if (sent.lastT !== null && !sent.lastOn) sent.dec.idle(nowMs - sent.lastT);
}

function clearSent() {
  sent.dec.reset();
  sent.lastT = null;
  sent.lastOn = false;
  sent.runs = [];
  sent.index = keyer.transitionCount;
  dirty = true;
}

// -------------------------------------------------------------- reference

/** The Morse chart (section G): built from the shared table, lit by the letter in progress. */
const ref = { cells: /** @type {HTMLElement[]} */ ([]), lastBuffer: /** @type {string | null} */ (null), playing: false };

function buildReference() {
  // The cells are in the HTML (so crawlers and readers without scripts see the
  // chart); they are rebuilt only if the markup and the table disagree.
  const entries = chartEntries();
  const present = Array.from(el.refGrid.querySelectorAll(".refcell"));
  const same = present.length === entries.length
    && present.every((cell, i) => cell.dataset.code === entries[i][1] && cell.querySelector(".ch")?.textContent === entries[i][0]);
  if (!same) {
    el.refGrid.textContent = "";
    for (const [ch, code] of entries) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "refcell";
      cell.dataset.code = code;
      cell.title = `Play ${ch}`;
      cell.innerHTML = `<span class="ch">${escapeHtml(ch)}</span><span class="code mono">${code.replace(/\./g, "·").replace(/-/g, "−")}</span>`;
      el.refGrid.appendChild(cell);
    }
  }
  ref.cells = Array.from(el.refGrid.querySelectorAll(".refcell"));
  ref.cells.forEach((cell, i) => cell.addEventListener("click", () => playReference(entries[i][0])));
}

function setReferenceOpen(open, persist = true) {
  el.refBody.hidden = !open;
  el.refToggle.textContent = open ? "Hide chart" : "Show chart";
  el.refToggle.setAttribute("aria-expanded", String(open));
  if (persist) store.set(STORE_REF_OPEN, open ? "1" : "0");
  ref.lastBuffer = null;
  dirty = true;
}

/** Light the cells the letter in progress could still become; the exact match stands out. */
function updateReferenceDom() {
  if (el.refBody.hidden) return;
  const keyed = liveKey.running ? sent.dec.buffer : "";
  const buffer = keyed || (state.running ? decoder.buffer : "");
  if (buffer === ref.lastBuffer) return;
  ref.lastBuffer = buffer;
  for (const cell of ref.cells) {
    const s = cellState(cell.dataset.code || "", buffer);
    cell.classList.toggle("match", s === "match");
    cell.classList.toggle("prefix", s === "prefix");
  }
}

/** Play one character at the encoder speed through the speakers (and the feed while listening). */
function playReference(ch) {
  try {
    player.audioContext = ensureContext();
    syncPlayerFeed();
    ref.playing = true;
    encoder.playheadMs = null;
    const timing = buildGuide(ch, encoder.wpm).timing;
    player.play(timing, state.f0, 0.15);
    vibrateTiming(timing);
    el.encPlay.textContent = "Play tone";
    updateMicGate();
  } catch {
    ref.playing = false;
  }
  dirty = true;
}

// --------------------------------------------------------------- practice

const practice = new Practice("words");
const pr = { checked: false, kindButtons: null };
const KIND_BUTTON = { words: "prWords", calls: "prCalls", digits: "prDigits", mixed: "prMixed" };

function setPracticeKind(kind) {
  practice.setKind(kind);
  for (const [k, id] of Object.entries(KIND_BUTTON)) el[id].setAttribute("aria-pressed", String(k === kind));
}

function practiceNext() {
  const target = practice.nextTarget();
  clearSent();
  pr.checked = false;
  el.prTarget.textContent = target;
  el.prTarget.classList.remove("done");
  el.prCode.textContent = encode(target).replace(/\./g, "·").replace(/-/g, "−");
  el.prResult.textContent = "Key it with the Key strip above.";
  el.prResult.className = "pr-result";
  el.prAcc.innerHTML = "—<small>%</small>";
  el.prWpm.innerHTML = "—<small>WPM</small>";
  el.prRhythm.innerHTML = "—<small>%</small>";
  updatePracticeSession();
  dirty = true;
}

/** Runs between the operator's first and last mark, for the rhythm report. */
function keyedRuns() {
  const runs = sent.runs;
  let a = 0;
  let b = runs.length;
  while (a < b && !runs[a].on) a++;
  while (b > a && !runs[b - 1].on) b--;
  return runs.slice(a, b);
}

function practiceCheck(auto = false) {
  if (!practice.target || pr.checked) return;
  const copy = sent.dec.text + (sent.dec.pendingCount ? "" : "");
  const s = score(practice.target, copy);
  if (auto && !s.perfect) return;
  pr.checked = true;
  practice.record(s);
  if (s.perfect) maybeStarNudge();
  const r = rhythm(keyedRuns(), sent.dec.ditMs);
  el.prAcc.innerHTML = `${Math.round(100 * s.accuracy)}<small>%</small>`;
  el.prWpm.innerHTML = `${sent.dec.timingReady || sent.runs.length >= 3 ? sent.dec.wpm.toFixed(1) : "—"}<small>WPM</small>`;
  el.prRhythm.innerHTML = `${r.marks + r.gaps ? Math.round(r.errorPct) : "—"}<small>%</small>`;
  if (s.perfect) {
    el.prResult.textContent = "Correct. Press Next target to continue.";
    el.prResult.className = "pr-result good";
    el.prTarget.classList.add("done");
  } else {
    const sentText = s.sent || "(nothing)";
    el.prResult.textContent = `You sent ${sentText}: ${s.errors} error${s.errors === 1 ? "" : "s"} against ${s.target}.`;
    el.prResult.className = "pr-result bad";
  }
  updatePracticeSession();
  dirty = true;
}

function updatePracticeSession() {
  el.prSession.innerHTML = `${practice.correct}<small>/ ${practice.asked}</small>`;
}

/** Called every frame from updateKeyDom: grade automatically once the copy matches. */
function updatePracticeDom() {
  if (!practice.target || pr.checked) return;
  if (sent.dec.text.trim() && !sent.dec.buffer && !sent.dec.pendingCount) practiceCheck(true);
}

function updateKeyDom() {
  const on = liveKey.running && liveKey.isOn();
  el.keyPill.className = "pill" + (on ? " on" : "");
  el.keyPill.innerHTML = `<i></i>${on ? "SENDING" : "SILENT"}`;
  if (liveKey.running) trackSent(liveKey.nowMs);
  const pendingSymbols = sent.dec.pendingCount ? sent.dec.provisional : sent.dec.buffer;
  const shown = pendingSymbols.replace(/\./g, "·").replace(/-/g, "−");
  const text = sent.dec.text.length > 60 ? `…${sent.dec.text.slice(-60)}` : sent.dec.text;
  el.sentOut.innerHTML = escapeHtml(text) + (shown ? `<span class="sep"> ${shown}</span>` : "") || "&nbsp;";
  updatePracticeDom();
  updateReferenceDom();
  // Keep redrawing while the tone sounds and while a letter is still pending, so
  // trackSent keeps reporting the growing silence and the decoder can close the
  // letter (seven dit lengths) with no further input, even when the main decoder
  // is not listening and the frame loop would otherwise go to sleep.
  if (on || pendingSymbols) dirty = true;
}

// ------------------------------------------------------------ DOM readouts

function updateDom() {
  updateLamp();
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
  el.log.disabled = log.isEmpty;

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
  updateKeyDom();
}

function drawAll() {
  drawSpectrum();
  drawPower();
  drawStrip();
  drawMeter();
  drawWave();
  drawHist();
  drawEncGuide();
  drawCamera();
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
  const storedProc = store.get(STORAGE.micProc);
  if (storedProc === "browser" || storedProc === "raw") el.micProc.value = storedProc;
  el.micProc.addEventListener("change", () => {
    store.set(STORAGE.micProc, el.micProc.value);
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
  el.log.addEventListener("click", downloadLog);
  el.msgDismiss.addEventListener("click", hideMessage);
  el.chopDismiss.addEventListener("click", () => {
    el.chopHint.hidden = true;
    state.chopDismissed = true;
  });
  el.camView.addEventListener("click", setCamSpot);
  el.lightFullBtn.addEventListener("click", requestFullLight);
  el.lightFull.addEventListener("click", exitFullLight);
  el.flashOk.addEventListener("click", () => {
    store.set(STORE_FLASH_NOTICE, "1");
    enterFullLight();
  });
  el.flashCancel.addEventListener("click", () => {
    el.flashNotice.hidden = true;
  });
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && lamp.fullActive) lamp.exitFull();
  });
  el.viewDesktop.addEventListener("click", () => setView("desktop"));
  el.viewHandset.addEventListener("click", () => setView("handset"));
  el.moreBtn.addEventListener("click", () => {
    const open = document.body.classList.toggle("more-open");
    el.moreBtn.setAttribute("aria-expanded", String(open));
    el.moreBtn.textContent = open ? "Less" : "More";
    ref.lastBuffer = null;
    dirty = true;
  });
  el.starNudgeDismiss.addEventListener("click", endStarNudge);
  el.starNudgeLink.addEventListener("click", endStarNudge);
  loadStarCount();
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
  el.encFarns.addEventListener("input", () => {
    player.stop();
    buildEncoding();
  });
  el.encPlay.addEventListener("click", togglePlay);
  el.encFeed.addEventListener("change", syncPlayerFeed);

  el.keyStraight.addEventListener("click", () => setKeyMode("straight"));
  el.keyPaddle.addEventListener("click", () => setKeyMode("paddle"));
  el.keyBug.addEventListener("click", () => setKeyMode("bug"));
  el.iambicA.addEventListener("click", () => setIambic("A"));
  el.iambicB.addEventListener("click", () => setIambic("B"));
  setIambic(store.get(STORE_IAMBIC) === "B" ? "B" : "A", false);
  const storedRatio = store.get(STORE_RATIO);
  if (storedRatio !== null) el.keyRatio.value = storedRatio;
  const storedWeight = store.get(STORE_WEIGHT);
  if (storedWeight !== null) el.keyWeight.value = storedWeight;
  applyWeighting(false);
  el.keyRatio.addEventListener("input", () => applyWeighting(true));
  el.keyWeight.addEventListener("input", () => applyWeighting(true));
  el.iambicA.disabled = keyer.mode !== "paddle";
  el.iambicB.disabled = keyer.mode !== "paddle";
  buildReference();
  setReferenceOpen(store.get(STORE_REF_OPEN) === "1", false);
  el.refToggle.addEventListener("click", () => setReferenceOpen(el.refBody.hidden));
  bindKeyButton(el.keyBtn, () => liveKey.keyDown(), () => liveKey.keyUp());
  bindKeyButton(el.ditBtn, () => liveKey.paddleDown("dit"), () => liveKey.paddleUp("dit"));
  bindKeyButton(el.dahBtn, () => liveKey.paddleDown("dah"), () => liveKey.paddleUp("dah"));
  window.addEventListener("keydown", (e) => onKeyboard(e, true));
  window.addEventListener("keyup", (e) => onKeyboard(e, false));
  el.keyRebind.addEventListener("click", () => startCapture("key"));
  el.ditRebind.addEventListener("click", () => startCapture("dit"));
  el.dahRebind.addEventListener("click", () => startCapture("dah"));
  el.keyReset.addEventListener("click", resetBindings);
  const storedSidetone = store.get(STORE_SIDETONE);
  if (storedSidetone !== null) el.keySidetone.value = storedSidetone;
  else el.keySidetone.value = String(DEFAULT_SIDETONE_HZ);
  applySidetone(false);
  el.keySidetone.addEventListener("input", () => applySidetone(true));
  renderBindings();
  window.addEventListener("blur", releaseKeys);
  el.sentClear.addEventListener("click", clearSent);
  for (const [kind, id] of Object.entries(KIND_BUTTON)) el[id].addEventListener("click", () => setPracticeKind(kind));
  el.prNext.addEventListener("click", practiceNext);
  el.prCheck.addEventListener("click", () => practiceCheck(false));
  el.prShowCode.addEventListener("change", () => {
    el.prCode.hidden = !el.prShowCode.checked;
  });
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
        // Cleared before drawing: a draw that asks for another frame (the Sent
        // line with a letter still pending) must not have its request wiped.
        dirty = false;
        drawAll();
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
  setKeyMode("straight");
  setPracticeKind("words");
  wire();
  initChannels();
  setView(initialView(), false);
  updateControls();
  updateStatus();
  refreshDevices();
  loadReleases();
  frameLoop();
}

init();
