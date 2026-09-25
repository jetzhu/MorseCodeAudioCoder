// The signal lamp: shows Morse as light. An amber ring while a mark is being
// received, a white fill while one is being sent, on every lamp element on
// the page and, when asked, on a full-screen overlay for sending across a
// room. Pure DOM class work, so it runs in Node against fake elements.

/**
 * @typedef {{classList: {toggle(name: string, force?: boolean): unknown}, hidden?: boolean}} LampElement
 */

export class Lamp {
  /**
   * @param {Iterable<LampElement>} elements the `.lamp` elements to drive
   * @param {LampElement | null} [full] the full-screen overlay, hidden until `enterFull`
   */
  constructor(elements, full = null) {
    this.elements = Array.from(elements);
    this.full = full;
    this.rx = false;
    this.tx = false;
    this.fullActive = false;
  }

  /**
   * Set both directions at once; touches the DOM only when something changed.
   * @param {boolean} rx a mark is being received
   * @param {boolean} tx a mark is being sent
   */
  setState(rx, tx) {
    rx = Boolean(rx);
    tx = Boolean(tx);
    if (rx === this.rx && tx === this.tx) return;
    this.rx = rx;
    this.tx = tx;
    for (const e of this.elements) {
      e.classList.toggle("rx", rx);
      e.classList.toggle("tx", tx);
    }
    if (this.full) {
      this.full.classList.toggle("rx", rx);
      this.full.classList.toggle("tx", tx);
    }
  }

  /** Show the full-screen overlay (the caller has shown the photosensitivity notice). */
  enterFull() {
    if (!this.full) return;
    this.fullActive = true;
    this.full.hidden = false;
  }

  exitFull() {
    if (!this.full) return;
    this.fullActive = false;
    this.full.hidden = true;
  }
}
