"""Tests for morse.practice: targets, scoring and the rhythm report."""
from __future__ import annotations

import random

import pytest

from morse.practice import KINDS, WORDS, Practice, call_sign, digit_group, rhythm, score
from morse.runs import Run
from morse.table import MORSE_TABLE


def test_word_list_is_upper_case_and_encodable():
    assert len(WORDS) > 100
    for w in WORDS:
        assert w == w.upper() and all(ch in MORSE_TABLE for ch in w), w


def test_generators_are_reproducible_and_well_formed():
    rng1, rng2 = random.Random(7), random.Random(7)
    assert [call_sign(rng1) for _ in range(5)] == [call_sign(rng2) for _ in range(5)]
    for _ in range(50):
        cs = call_sign(random.Random())
        assert cs.isalnum() and cs == cs.upper() and any(ch.isdigit() for ch in cs), cs
        assert 3 <= len(cs) <= 6
    assert digit_group(random.Random(1)) == digit_group(random.Random(1))
    assert len(digit_group(random.Random(2))) == 5 and digit_group(random.Random(2)).isdigit()
    assert len(digit_group(random.Random(2), 3)) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_practice_hands_out_targets_of_the_kind(kind):
    p = Practice(kind, seed=3)
    seen = [p.next_target() for _ in range(30)]
    assert p.asked == 30
    assert all(a != b for a, b in zip(seen, seen[1:])), "never the same target twice in a row"
    if kind == "words":
        assert all(t in WORDS for t in seen)
    elif kind == "digits":
        assert all(t.isdigit() and len(t) == 5 for t in seen)
    elif kind == "calls":
        assert all(any(ch.isdigit() for ch in t) and not t.isdigit() for t in seen)
    else:
        kinds = {"word" if t in WORDS else "digits" if t.isdigit() else "call" for t in seen}
        assert len(kinds) >= 2


def test_practice_validation_and_session():
    with pytest.raises(ValueError):
        Practice("letters")  # type: ignore[arg-type]
    p = Practice(seed=1)
    p.next_target()
    p.record(score(p.target, p.target))
    p.next_target()
    p.record(score(p.target, "XX"))
    assert (p.asked, p.correct) == (2, 1)
    p.reset_session()
    assert (p.asked, p.correct, p.target) == (0, 0, None)


def test_score_counts_matches_and_edit_distance():
    s = score("HELLO", "HELLO")
    assert (s.correct, s.errors, s.total, s.perfect) == (5, 0, 5, True) and s.accuracy == 1.0
    s = score("HELLO", "HELO")  # one deletion
    assert (s.correct, s.errors) == (4, 1) and s.accuracy == pytest.approx(0.8)
    s = score("HELLO", "HELLP")  # one substitution
    assert (s.correct, s.errors) == (4, 1)
    s = score("HELLO", "HEELLO")  # one insertion
    assert (s.correct, s.errors) == (5, 1)
    s = score("HELLO", "")
    assert (s.correct, s.errors, s.accuracy) == (0, 5, 0.0)
    s = score("HELLO", "XXXXXXXXXX")
    assert s.accuracy == 0.0  # clamped
    s = score("hello world", "  HELLO   WORLD ")
    assert s.perfect and s.target == "HELLO WORLD"
    assert score("", "").accuracy == 1.0 and score("", "A").accuracy == 0.0


def test_rhythm_is_zero_for_ideal_timing_and_grows_with_error():
    T = 100.0
    ideal = [Run(True, 10), Run(False, 10), Run(True, 30), Run(False, 30), Run(True, 10), Run(False, 70), Run(True, 30)]
    r = rhythm(ideal, T)
    assert (r.mark_error_pct, r.gap_error_pct, r.marks, r.gaps) == (0.0, 0.0, 4, 3)
    assert r.error_pct == 0.0
    sloppy = [Run(True, 12), Run(False, 8), Run(True, 36), Run(False, 24), Run(True, 10)]
    r = rhythm(sloppy, T)
    assert r.mark_error_pct == pytest.approx(100 * (0.2 + 0.2 + 0.0) / 3)
    assert r.gap_error_pct == pytest.approx(100 * (0.2 + 0.2) / 2)
    assert r.error_pct == pytest.approx((r.mark_error_pct * 3 + r.gap_error_pct * 2) / 5)
    assert rhythm([], T).error_pct == 0.0
    with pytest.raises(ValueError):
        rhythm(ideal, 0.0)
