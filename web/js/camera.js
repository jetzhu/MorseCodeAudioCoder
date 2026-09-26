// CameraInput: open a camera, show its preview, and report the brightness of
// the spot the person taps, once per video frame. The numbers go to
// LightDetector (lightdetect.js); this module only touches the browser.

/**
 * @typedef {{t: number, value: number}} Frame brightness 0..255 at time t (ms, performance.now clock)
 */

/**
 * When the frame was captured, on the performance.now() clock. The time the
 * callback runs bunches up whenever rendering stalls, which would squeeze and
 * stretch marks; camera streams report the capture time, so use it.
 * @param {number} now @param {{captureTime?: number, expectedDisplayTime?: number} | undefined} meta
 */
export function frameTime(now, meta) {
  if (meta && Number.isFinite(meta.captureTime) && meta.captureTime > 0) return meta.captureTime;
  if (meta && Number.isFinite(meta.expectedDisplayTime) && meta.expectedDisplayTime > 0) return meta.expectedDisplayTime;
  return now;
}

export class CameraInput {
  /**
   * @param {HTMLVideoElement} video element that shows the preview
   * @param {object} [options]
   * @param {number} [options.sampleWidth=160] the frame is scaled to this width before sampling
   * @param {number} [options.spotFrac=0.06] spot diameter as a fraction of the frame's shorter side
   */
  constructor(video, { sampleWidth = 160, spotFrac = 0.06 } = {}) {
    this.video = video;
    this.sampleWidth = sampleWidth;
    this.spotFrac = spotFrac;
    /** Tracked spot, as fractions of the frame (0..1). */
    this.spot = { x: 0.5, y: 0.5 };
    /** @type {MediaStream | null} */
    this.stream = null;
    /** @type {((frame: Frame) => void) | null} */
    this.onFrame = null;
    /** @type {((err: Error) => void) | null} */
    this.onEnded = null;
    this.label = "";
    /** "locked", "settling" or "auto" (the camera does not let the page hold it). */
    this.exposure = "";
    this._canvas = null;
    this._ctx = null;
    this._handle = null;
    this._running = false;
  }

  get running() {
    return this._running;
  }

  /** Whether this browser can open a camera at all. */
  static supported() {
    const md = globalThis.navigator && globalThis.navigator.mediaDevices;
    return Boolean(md && typeof md.getUserMedia === "function");
  }

  /** Open the camera (the rear one on phones) and start sampling. */
  async start() {
    if (this._running) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } },
    });
    const track = stream.getVideoTracks()[0];
    this.label = track ? track.label || "camera" : "camera";
    if (track) {
      track.onended = () => {
        if (this._running && typeof this.onEnded === "function") this.onEnded(new Error("the camera stream ended"));
      };
    }
    this.stream = stream;
    this.video.srcObject = stream;
    this.video.muted = true;
    this.video.playsInline = true;
    try {
      await this.video.play();
    } catch {
      /* autoplay rules; frames still arrive once the element plays */
    }
    this._canvas = document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    this._running = true;
    this._schedule();
    this.lockExposure(1500);
  }

  /**
   * Automatic exposure darkens the whole picture each time a bright lamp
   * lights, which cuts marks short and blurs gaps. After `settleMs` of
   * automatic adjustment, hold exposure, white balance and focus still where
   * the camera allows it (Chrome on Android does; many laptops and iPhones do
   * not). Resolves to "locked", "auto" (not supported) or "" when stopped.
   * @param {number} [settleMs=1500]
   */
  async lockExposure(settleMs = 1500) {
    const track = this.track;
    if (!track || typeof track.applyConstraints !== "function") {
      this.exposure = "auto";
      return this.exposure;
    }
    const caps = typeof track.getCapabilities === "function" ? track.getCapabilities() : {};
    const modes = (name) => (Array.isArray(caps[name]) ? caps[name] : []);
    const lockable = ["exposureMode", "whiteBalanceMode", "focusMode"].filter((k) => modes(k).includes("manual"));
    if (!lockable.length) {
      this.exposure = "auto";
      return this.exposure;
    }
    this.exposure = "settling";
    try {
      await track.applyConstraints({ advanced: lockable.map((k) => ({ [k]: modes(k).includes("continuous") ? "continuous" : "manual" })) });
    } catch {
      /* ignore: we lock below anyway */
    }
    await new Promise((r) => setTimeout(r, settleMs));
    if (!this._running || this.track !== track) return "";
    try {
      await track.applyConstraints({ advanced: lockable.map((k) => ({ [k]: "manual" })) });
      this.exposure = "locked";
    } catch {
      this.exposure = "auto";
    }
    return this.exposure;
  }

  stop() {
    this._running = false;
    if (this._handle !== null) {
      if (typeof this.video.cancelVideoFrameCallback === "function" && this._handleKind === "rvfc") this.video.cancelVideoFrameCallback(this._handle);
      else cancelAnimationFrame(this._handle);
      this._handle = null;
    }
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
    this.video.srcObject = null;
  }

  /** Track another spot; `x`, `y` are fractions of the preview (0..1). */
  setSpot(x, y) {
    this.spot = { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
  }

  /** The camera's video track, for the torch (step 3). */
  get track() {
    return this.stream ? this.stream.getVideoTracks()[0] || null : null;
  }

  _schedule() {
    if (!this._running) return;
    // A hidden preview (Show: Light off) gets no frame callbacks, but its stream
    // keeps playing, so sample it on animation frames instead.
    const visible = this.video.isConnected && this.video.getClientRects().length > 0;
    if (visible && typeof this.video.requestVideoFrameCallback === "function") {
      this._handleKind = "rvfc";
      this._handle = this.video.requestVideoFrameCallback((now, meta) => this._sample(frameTime(now, meta)));
    } else {
      this._handleKind = "raf";
      this._handle = requestAnimationFrame((now) => this._sample(now));
    }
  }

  _sample(now) {
    this._handle = null;
    if (!this._running) return;
    const v = this.video;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    if (vw > 0 && vh > 0 && this._ctx) {
      const w = this.sampleWidth;
      const h = Math.max(1, Math.round((vh / vw) * w));
      if (this._canvas.width !== w || this._canvas.height !== h) {
        this._canvas.width = w;
        this._canvas.height = h;
      }
      this._ctx.drawImage(v, 0, 0, w, h);
      const d = Math.max(2, Math.round(Math.min(w, h) * this.spotFrac));
      const x0 = Math.min(w - d, Math.max(0, Math.round(this.spot.x * w - d / 2)));
      const y0 = Math.min(h - d, Math.max(0, Math.round(this.spot.y * h - d / 2)));
      const px = this._ctx.getImageData(x0, y0, d, d).data;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      const value = sum / (px.length / 4);
      if (typeof this.onFrame === "function") this.onFrame({ t: now, value });
    }
    this._schedule();
  }
}
