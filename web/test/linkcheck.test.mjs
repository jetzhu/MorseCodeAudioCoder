// Tests for web/js/linkcheck.js.
import { test } from "node:test";
import assert from "node:assert/strict";

import { LINK_TEST, extractTest, gradeLink, testComplete } from "../js/linkcheck.js";

test("the test phrase is extracted from the last VVV", () => {
  assert.equal(LINK_TEST, "VVV PARIS 73");
  assert.equal(extractTest("  hello  vvv paris   73 "), "VVV PARIS 73");
  assert.equal(extractTest("VVV PAR VVV PARIS"), "VVV PARIS");
  assert.equal(extractTest("PARIS 73"), "PARIS 73", "the Vs lost: grade what came");
  assert.equal(extractTest(""), "");
});

test("the test is complete at the closing 73 or at full length", () => {
  assert.equal(testComplete("VVV PARIS"), false);
  assert.equal(testComplete("VVV PARIS 73"), true);
  assert.equal(testComplete("VVV PARIX 7E"), true, "as many letters as the phrase");
  assert.equal(testComplete("TE 73"), true);
});

test("grades and advises", () => {
  const good = gradeLink("VVV PARIS 73");
  assert.equal(good.perfect, true);
  assert.equal(good.accuracy, 1);
  assert.equal(good.advice, "Link good.");
  const near = gradeLink("VVV PARIS 7E", { wpm: 12 });
  assert.equal(near.perfect, false);
  assert.match(near.advice, /Nearly there.*12 WPM/);
  assert.match(gradeLink("").advice, /Nothing arrived/);
  assert.match(gradeLink("EE TTE", { wpm: 16, fps: 30 }).advice, /Too fast for a 30 fps camera: send at 12 WPM/);
  assert.match(gradeLink("EE TTE", { wpm: 8, fps: 30 }).advice, /Many errors/);
});
