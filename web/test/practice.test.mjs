// Tests for web/js/practice.js (mirrors tests/test_practice.py).
import { test } from "node:test";
import assert from "node:assert/strict";

import { KINDS, Practice, WORDS, callSign, digitGroup, makeRng, rhythm, score } from "../js/practice.js";
import { MORSE_TABLE } from "../js/table.js";
import { Run } from "../js/runs.js";

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test("word list is upper case and encodable", () => {
  assert.ok(WORDS.length > 100);
  for (const w of WORDS) {
    assert.equal(w, w.toUpperCase());
    for (const ch of w) assert.ok(ch in MORSE_TABLE, w);
  }
});

test("generators are reproducible and well formed", () => {
  const a = makeRng(7);
  const b = makeRng(7);
  assert.deepEqual([1, 2, 3, 4, 5].map(() => callSign(a)), [1, 2, 3, 4, 5].map(() => callSign(b)));
  const rng = makeRng(11);
  for (let i = 0; i < 50; i++) {
    const cs = callSign(rng);
    assert.match(cs, /^[A-Z]{1,2}[0-9][A-Z]{1,3}$/, cs);
  }
  assert.equal(digitGroup(makeRng(1)), digitGroup(makeRng(1)));
  assert.match(digitGroup(makeRng(2)), /^[0-9]{5}$/);
  assert.equal(digitGroup(makeRng(2), 3).length, 3);
});

for (const kind of KINDS) {
  test(`practice hands out targets of kind ${kind}`, () => {
    const p = new Practice(kind, 3);
    const seen = Array.from({ length: 30 }, () => p.nextTarget());
    assert.equal(p.asked, 30);
    for (let i = 1; i < seen.length; i++) assert.notEqual(seen[i], seen[i - 1]);
    if (kind === "words") assert.ok(seen.every((t) => WORDS.includes(t)));
    else if (kind === "digits") assert.ok(seen.every((t) => /^[0-9]{5}$/.test(t)));
    else if (kind === "calls") assert.ok(seen.every((t) => /[0-9]/.test(t) && !/^[0-9]+$/.test(t)));
    else {
      const kinds = new Set(seen.map((t) => (WORDS.includes(t) ? "word" : /^[0-9]+$/.test(t) ? "digits" : "call")));
      assert.ok(kinds.size >= 2);
    }
  });
}

test("practice validation and session", () => {
  assert.throws(() => new Practice("letters"), RangeError);
  const p = new Practice("words", 1);
  p.nextTarget();
  p.record(score(p.target, p.target));
  p.nextTarget();
  p.record(score(p.target, "XX"));
  assert.deepEqual([p.asked, p.correct], [2, 1]);
  p.resetSession();
  assert.deepEqual([p.asked, p.correct, p.target], [0, 0, null]);
});

test("score counts matches and edit distance", () => {
  let s = score("HELLO", "HELLO");
  assert.deepEqual([s.correct, s.errors, s.total, s.perfect, s.accuracy], [5, 0, 5, true, 1]);
  s = score("HELLO", "HELO");
  assert.deepEqual([s.correct, s.errors], [4, 1]);
  close(s.accuracy, 0.8);
  s = score("HELLO", "HELLP");
  assert.deepEqual([s.correct, s.errors], [4, 1]);
  s = score("HELLO", "HEELLO");
  assert.deepEqual([s.correct, s.errors], [5, 1]);
  s = score("HELLO", "");
  assert.deepEqual([s.correct, s.errors, s.accuracy], [0, 5, 0]);
  assert.equal(score("HELLO", "XXXXXXXXXX").accuracy, 0);
  s = score("hello world", "  HELLO   WORLD ");
  assert.ok(s.perfect);
  assert.equal(s.target, "HELLO WORLD");
  assert.equal(score("", "").accuracy, 1);
  assert.equal(score("", "A").accuracy, 0);
});

test("rhythm is zero for ideal timing and grows with error", () => {
  const T = 100;
  const R = (on, blocks) => new Run(on, blocks);
  const ideal = [R(true, 10), R(false, 10), R(true, 30), R(false, 30), R(true, 10), R(false, 70), R(true, 30)];
  let r = rhythm(ideal, T);
  assert.deepEqual([r.markErrorPct, r.gapErrorPct, r.marks, r.gaps, r.errorPct], [0, 0, 4, 3, 0]);
  r = rhythm([R(true, 12), R(false, 8), R(true, 36), R(false, 24), R(true, 10)], T);
  close(r.markErrorPct, (100 * (0.2 + 0.2 + 0)) / 3);
  close(r.gapErrorPct, (100 * (0.2 + 0.2)) / 2);
  close(r.errorPct, (r.markErrorPct * 3 + r.gapErrorPct * 2) / 5);
  assert.equal(rhythm([], T).errorPct, 0);
  assert.throws(() => rhythm(ideal, 0), RangeError);
});
