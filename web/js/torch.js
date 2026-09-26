// The phone's torch (camera flash) as a Morse light. Browsers expose it only
// as a constraint on a camera video track (`torch`), which Chrome on Android
// supports and iPhone Safari does not, so the torch needs an open rear
// camera: the camera the page is already listening with when there is one,
// otherwise a stream of its own opened on first use.
//
// Switching is asynchronous and takes tens of milliseconds, so requests are
// coalesced: only the latest wanted state is applied once the one in flight
// finishes, and a burst of edges can never queue up behind a slow driver.

/** Top speed the torch is keyed at; faster sending is slowed to this for every channel. */
export const TORCH_MAX_WPM = 8;

export class Torch {
  /**
   * @param {object} [options]
   * @param {() => (MediaStreamTrack | null)} [options.external] a camera track already open (the listening camera)
   * @param {(constraints: MediaStreamConstraints) => Promise<MediaStream>} [options.getUserMedia]
   */
  constructor({ external = () => null, getUserMedia = null } = {}) {
    this.external = external;
    this.getUserMedia = getUserMedia || ((c) => navigator.mediaDevices.getUserMedia(c));
    /** @type {boolean | null} null until a track has been checked */
    this.supported = null;
    this.wanted = false;
    this.applied = false;
    this._own = null;
    this._track = null;
    this._busy = false;
    /** @type {string} why the torch is unavailable, after a failed ensure() */
    this.reason = "";
  }

  /** A track with a usable torch is at hand. */
  get ready() {
    return Boolean(this.supported && this._currentTrack());
  }

  _currentTrack() {
    const ext = this.external();
    if (ext && ext.readyState !== "ended") return ext;
    if (this._own) {
      const t = this._own.getVideoTracks()[0];
      if (t && t.readyState !== "ended") return t;
    }
    return null;
  }

  /**
   * Make sure a camera track with a torch is open. Resolves true when the
   * torch can be switched, false (with `reason`) when this device or browser
   * cannot, or permission was refused.
   */
  async ensure() {
    let track = this._currentTrack();
    if (!track) {
      try {
        this._own = await this.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" } } });
      } catch (err) {
        this.supported = false;
        this.reason = err && err.name === "NotAllowedError" ? "the torch needs camera permission" : "no camera with a torch";
        return false;
      }
      track = this._currentTrack();
    }
    const caps = track && typeof track.getCapabilities === "function" ? track.getCapabilities() : null;
    this.supported = Boolean(caps && caps.torch);
    if (!this.supported) {
      this.reason = "this browser cannot switch the torch (Chrome on Android can)";
      this.releaseOwn();
      return false;
    }
    this._track = track;
    this.applied = false;
    return true;
  }

  /** Ask for the torch on or off; applied as soon as the driver allows. @param {boolean} on */
  set(on) {
    this.wanted = Boolean(on);
    if (!this.supported) return;
    this._pump();
  }

  async _pump() {
    if (this._busy) return;
    this._busy = true;
    try {
      while (this.wanted !== this.applied) {
        const track = this._currentTrack();
        if (!track) break;
        const target = this.wanted;
        try {
          await track.applyConstraints({ advanced: [{ torch: target }] });
          this.applied = target;
        } catch {
          this.supported = false;
          this.reason = "the torch stopped responding";
          break;
        }
      }
    } finally {
      this._busy = false;
    }
  }

  /** Close the stream opened for the torch (the listening camera is left alone). */
  releaseOwn() {
    if (this._own) for (const t of this._own.getTracks()) t.stop();
    this._own = null;
    this.applied = false;
  }
}
