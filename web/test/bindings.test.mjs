// Tests for web/js/bindings.js (mirrors tests/test_bindings.py).
import { test } from "node:test";
import assert from "node:assert/strict";

import { ACTIONS, DEFAULTS, RESERVED_CODES, actionFor, isDefault, keyLabel, rebind, sanitize } from "../js/bindings.js";

test("defaults are complete and distinct", () => {
  assert.deepEqual(Object.keys(DEFAULTS), ACTIONS);
  assert.equal(new Set(Object.values(DEFAULTS)).size, 3);
  assert.ok(isDefault(DEFAULTS));
  assert.deepEqual(sanitize(DEFAULTS), { ...DEFAULTS });
});

test("sanitize repairs garbage with the defaults", () => {
  for (const garbage of [null, undefined, 42, "Space", [], { key: 1 }, { dit: "" }, { nothing: "here" }]) {
    assert.deepEqual(sanitize(garbage), { ...DEFAULTS });
  }
  assert.deepEqual(sanitize({ key: "KeyK", dit: " KeyJ ", dah: "KeyL", extra: "X" }), { key: "KeyK", dit: "KeyJ", dah: "KeyL" });
  assert.deepEqual(sanitize({ key: "KeyK" }), { key: "KeyK", dit: "ArrowLeft", dah: "ArrowRight" });
});

test("sanitize resolves duplicates in action order", () => {
  assert.deepEqual(sanitize({ key: "KeyJ", dit: "KeyJ", dah: "KeyL" }), { key: "KeyJ", dit: "ArrowLeft", dah: "KeyL" });
  assert.deepEqual(sanitize({ key: "ArrowRight", dit: "ArrowLeft", dah: "ArrowLeft" }), { key: "ArrowRight", dit: "ArrowLeft", dah: "Space" });
});

test("rebind swaps when the key is taken", () => {
  const b = rebind(DEFAULTS, "dit", "KeyJ");
  assert.deepEqual(b, { key: "Space", dit: "KeyJ", dah: "ArrowRight" });
  assert.equal(DEFAULTS.dit, "ArrowLeft");
  const swapped = rebind(b, "dah", "KeyJ");
  assert.deepEqual(swapped, { key: "Space", dit: "ArrowRight", dah: "KeyJ" });
  assert.deepEqual(rebind(b, "dit", "KeyJ"), b);
  assert.equal(new Set(Object.values(rebind(swapped, "key", "ArrowRight"))).size, 3);
  assert.throws(() => rebind(DEFAULTS, "paddle", "KeyJ"), RangeError);
  assert.throws(() => rebind(DEFAULTS, "key", "  "), RangeError);
});

test("actionFor, isDefault and reserved codes", () => {
  assert.equal(actionFor(DEFAULTS, "Space"), "key");
  assert.equal(actionFor(DEFAULTS, "ArrowLeft"), "dit");
  assert.equal(actionFor(DEFAULTS, "ArrowRight"), "dah");
  assert.equal(actionFor(DEFAULTS, "KeyJ"), null);
  assert.ok(!isDefault(rebind(DEFAULTS, "key", "KeyK")));
  assert.ok(isDefault(rebind(rebind(DEFAULTS, "key", "KeyK"), "key", "Space")));
  for (const code of ["ShiftLeft", "ControlRight", "AltLeft", "MetaLeft", "Escape", "CapsLock"]) assert.ok(RESERVED_CODES.has(code), code);
  assert.ok(!RESERVED_CODES.has("Space") && !RESERVED_CODES.has("KeyJ"));
});

test("keyLabel gives short human names", () => {
  assert.equal(keyLabel("Space"), "Space");
  assert.equal(keyLabel("ArrowLeft"), "\u2190 left arrow");
  assert.equal(keyLabel("ArrowRight"), "\u2192 right arrow");
  assert.equal(keyLabel("KeyJ"), "J");
  assert.equal(keyLabel("Digit5"), "5");
  assert.equal(keyLabel("Numpad5"), "5 (numpad)");
  assert.equal(keyLabel("F3"), "F3");
  assert.equal(keyLabel("Period"), ".");
  assert.equal(keyLabel("PageDown"), "Page Down");
  assert.equal(keyLabel(""), "?");
});
