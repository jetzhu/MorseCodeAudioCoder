// Channels: what the page listens with and what it sends with, in any
// combination. Mirror of morse/channels.py.
//
// Listen sources: mic (microphone), camera (light through the camera).
// Send channels: audio (speakers), light (the lamp or the full screen),
// torch (the phone's flash), vibrate (the phone's vibration motor).
// Everything a device supports starts selected; a selection is remembered.

/** @typedef {{listen: {mic: boolean, camera: boolean}, send: {audio: boolean, light: boolean, torch: boolean, vibrate: boolean}}} Channels */

export const LISTEN = ["mic", "camera"];
export const SEND = ["audio", "light", "torch", "vibrate"];

/** Everything on. @type {Channels} */
export const DEFAULTS = Object.freeze({
  listen: Object.freeze({ mic: true, camera: true }),
  send: Object.freeze({ audio: true, light: true, torch: true, vibrate: true }),
});

/**
 * A complete selection from untrusted input (anything `JSON.parse` may give):
 * every key present and boolean, missing or odd values from the defaults.
 * @param {unknown} raw @param {Channels} [defaults]
 * @returns {Channels}
 */
export function sanitize(raw, defaults = DEFAULTS) {
  const src = raw && typeof raw === "object" ? /** @type {any} */ (raw) : {};
  const pick = (group, keys) => {
    const g = src[group] && typeof src[group] === "object" ? src[group] : {};
    /** @type {Record<string, boolean>} */
    const out = {};
    for (const k of keys) out[k] = typeof g[k] === "boolean" ? g[k] : defaults[group][k];
    return out;
  };
  return /** @type {Channels} */ ({ listen: pick("listen", LISTEN), send: pick("send", SEND) });
}

/**
 * What is actually in use: selected and available on this device.
 * @param {Channels} selection @param {Channels} available
 * @returns {Channels}
 */
export function effective(selection, available) {
  const both = (group, keys) => {
    /** @type {Record<string, boolean>} */
    const out = {};
    for (const k of keys) out[k] = Boolean(selection[group][k] && available[group][k]);
    return out;
  };
  return /** @type {Channels} */ ({ listen: both("listen", LISTEN), send: both("send", SEND) });
}

/**
 * Whether a keying sequence (`[{on, ms}, ...]`, as `buildTiming` makes) is
 * sounding `ms` after its start; false before the start and after the end.
 * @param {Array<{on: boolean, ms: number}>} timing @param {number} ms
 */
export function timingStateAt(timing, ms) {
  if (!(ms >= 0)) return false;
  let t = 0;
  for (const seg of timing) {
    const end = t + seg.ms;
    if (ms < end) return Boolean(seg.on);
    t = end;
  }
  return false;
}

/**
 * The Vibration API pattern for a keying sequence: alternating vibrate and
 * pause lengths in whole milliseconds, starting with a vibration (a leading
 * gap becomes a zero-length vibration first). Adjacent segments of the same
 * state are merged; zero-length segments are dropped.
 * @param {Array<{on: boolean, ms: number}>} timing
 * @returns {number[]}
 */
export function vibrationPattern(timing) {
  /** @type {number[]} */
  const out = [];
  let on = true; // the pattern's first entry is a vibration
  for (const seg of timing) {
    const ms = Math.max(0, Math.round(Number(seg.ms) || 0));
    if (ms === 0) continue;
    if (Boolean(seg.on) === on) {
      if (out.length === 0) out.push(0);
      out[out.length - 1] += ms;
    } else {
      if (out.length === 0) out.push(0); // a leading gap: the pattern must start with a (zero-length) vibration
      out.push(ms);
      on = Boolean(seg.on);
    }
  }
  while (out.length && out[out.length - 1] === 0) out.pop();
  return out;
}
