// Link check between two devices: one sends a fixed test phrase on its
// selected channels, the other listens and grades what arrives, with the
// practice strip's scoring. "VVV" is the traditional Morse test signal.

import { score } from "./practice.js";

export const LINK_TEST = "VVV PARIS 73";

/**
 * The part of the decoded text that belongs to the test: from the last
 * "VVV" (or whatever arrived if the Vs were lost), upper-cased and trimmed.
 * @param {string} text decoded since the check was armed
 */
export function extractTest(text) {
  const t = String(text).toUpperCase().replace(/\s+/g, " ").trim();
  const i = t.lastIndexOf("VVV");
  return i >= 0 ? t.slice(i) : t;
}

/** Whether the test has fully arrived (the closing 73, or as many letters as the phrase). @param {string} text */
export function testComplete(text) {
  const t = extractTest(text);
  return /73$/.test(t) || t.replace(/ /g, "").length >= LINK_TEST.replace(/ /g, "").length;
}

/**
 * Grade what arrived.
 * @param {string} text decoded since the check was armed
 * @param {{wpm?: number, fps?: number}} [ctx] speed and camera frame rate, for the advice
 * @returns {{received: string, accuracy: number, correct: number, total: number, perfect: boolean, advice: string}}
 */
export function gradeLink(text, { wpm = 0, fps = 0 } = {}) {
  const received = extractTest(text);
  const s = score(LINK_TEST, received);
  let advice;
  if (s.perfect) advice = "Link good.";
  else if (!received) advice = "Nothing arrived: check that the other lamp or speaker is in view (tap it in the camera preview) and that this device is listening.";
  else if (s.accuracy >= 0.75) advice = `Nearly there: hold both devices steadier, or slow down${wpm ? ` from ${Math.round(wpm)} WPM` : ""}.`;
  // The speed the decoder measured is only trustworthy when much of the text came through.
  else if (s.accuracy >= 0.3 && fps && wpm > 0.4 * fps) advice = `Too fast for a ${Math.round(fps)} fps camera: send at ${Math.floor(0.4 * fps)} WPM or less.`;
  else if (fps) advice = `Many errors: aim at the lamp and press Re-lock exposure, tap the middle of the lamp in the preview, keep both phones still, and send at ${Math.min(8, Math.floor(0.4 * fps))} WPM or less.`;
  else advice = "Many errors: move closer, dim the room, turn the sending screen's brightness up, or slow down.";
  return { received, accuracy: s.accuracy, correct: s.correct, total: s.total, perfect: s.perfect, advice };
}
