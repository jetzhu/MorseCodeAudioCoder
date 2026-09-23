// Key bindings for the hand key: which keyboard key works the straight key
// and each paddle. Mirror of morse/bindings.py; key names here are
// KeyboardEvent.code values ("Space", "ArrowLeft", "KeyJ").
//
// Every action always has exactly one key, no key serves two actions, and
// assigning a key another action holds swaps the two. Tables read back from
// storage go through sanitize(), which repairs anything broken with the
// defaults instead of throwing.

/** @typedef {"key" | "dit" | "dah"} Action */
/** @typedef {Record<Action, string>} Bindings */

/** The bindable actions, in display order. @type {Action[]} */
export const ACTIONS = ["key", "dit", "dah"];

/** Space for the straight key, the arrows for the paddles. @type {Bindings} */
export const DEFAULTS = Object.freeze({ key: "Space", dit: "ArrowLeft", dah: "ArrowRight" });

/** Codes that never make a binding: modifiers, and Escape which cancels a capture. */
export const RESERVED_CODES = new Set([
  "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight",
  "MetaLeft", "MetaRight", "OSLeft", "OSRight", "CapsLock", "NumLock", "ScrollLock",
  "Escape", "ContextMenu", "Fn", "FnLock",
]);

/**
 * A complete, duplicate-free binding table from untrusted input (anything
 * `JSON.parse` may give). Missing or empty entries take their default; a key
 * used twice stays with the first action in ACTIONS order and the later one
 * falls back to its default, or to any free default.
 * @param {unknown} mapping @param {Bindings} [defaults]
 * @returns {Bindings}
 */
export function sanitize(mapping, defaults = DEFAULTS) {
  const src = mapping && typeof mapping === "object" ? /** @type {Record<string, unknown>} */ (mapping) : {};
  /** @type {Record<string, string>} */
  const out = {};
  for (const action of ACTIONS) {
    const value = src[action];
    out[action] = typeof value === "string" && value.trim() ? value.trim() : defaults[action];
  }
  const used = new Set();
  for (const action of ACTIONS) {
    let key = out[action];
    if (used.has(key)) {
      key = defaults[action];
      if (used.has(key)) key = Object.values(defaults).find((d) => !used.has(d)) ?? key;
      out[action] = key;
    }
    used.add(key);
  }
  return /** @type {Bindings} */ (out);
}

/**
 * `bindings` with `action` on `key`; an action already holding `key` takes
 * the old key of `action`.
 * @param {Bindings} bindings @param {Action} action @param {string} key
 * @returns {Bindings}
 * @throws {RangeError} for an unknown action or an empty key
 */
export function rebind(bindings, action, key) {
  if (!ACTIONS.includes(action)) throw new RangeError(`unknown action ${action}`);
  if (typeof key !== "string" || !key.trim()) throw new RangeError("key must be a non-empty string");
  key = key.trim();
  const out = { ...bindings };
  const old = out[action];
  for (const other of ACTIONS) {
    if (other !== action && out[other] === key) out[other] = old || key;
  }
  out[action] = key;
  return out;
}

/** The action bound to `key`, or null. @param {Bindings} bindings @param {string} key @returns {Action | null} */
export function actionFor(bindings, key) {
  for (const action of ACTIONS) if (bindings[action] === key) return action;
  return null;
}

/** Whether every action is on its default key. @param {Bindings} bindings @param {Bindings} [defaults] */
export function isDefault(bindings, defaults = DEFAULTS) {
  return ACTIONS.every((a) => bindings[a] === defaults[a]);
}

const LABELS = {
  Space: "Space", ArrowLeft: "← left arrow", ArrowRight: "→ right arrow",
  ArrowUp: "↑ up arrow", ArrowDown: "↓ down arrow", Enter: "Enter", NumpadEnter: "Enter (numpad)",
  Backspace: "Backspace", Tab: "Tab", Backquote: "`", Minus: "-", Equal: "=", BracketLeft: "[",
  BracketRight: "]", Backslash: "\\", Semicolon: ";", Quote: "'", Comma: ",", Period: ".", Slash: "/",
  IntlBackslash: "\\", NumpadAdd: "+ (numpad)", NumpadSubtract: "- (numpad)", NumpadMultiply: "* (numpad)",
  NumpadDivide: "/ (numpad)", NumpadDecimal: ". (numpad)",
};

/**
 * A short human label for a KeyboardEvent.code: "KeyJ" is "J", "Digit5" is
 * "5", "Numpad5" is "5 (numpad)", "F3" stays, arrows get their glyph, and an
 * unknown code is split at its capitals ("PageDown" is "Page Down").
 * @param {string} code
 * @returns {string}
 */
export function keyLabel(code) {
  if (typeof code !== "string" || !code) return "?";
  if (code in LABELS) return LABELS[code];
  let m = /^Key([A-Z])$/.exec(code);
  if (m) return m[1];
  m = /^Digit([0-9])$/.exec(code);
  if (m) return m[1];
  m = /^Numpad([0-9])$/.exec(code);
  if (m) return `${m[1]} (numpad)`;
  if (/^F[0-9]{1,2}$/.test(code)) return code;
  return code.replace(/([a-z])([A-Z])/g, "$1 $2");
}
