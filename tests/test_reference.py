"""Tests for morse.reference (mirrored by web/test/reference.test.mjs)."""
from __future__ import annotations

from morse.reference import cell_state, chart_entries
from morse.table import MORSE_TABLE


def test_chart_covers_the_table_in_display_order() -> None:
    entries = chart_entries()
    assert len(entries) == len(MORSE_TABLE) == 52
    assert dict(entries) == dict(MORSE_TABLE)
    chars = [ch for ch, _ in entries]
    assert chars[:26] == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert chars[26:36] == [str(d) for d in range(10)]
    punctuation = entries[36:]
    assert all(not ch.isalnum() for ch, _ in punctuation)
    keys = [(len(code), code, ch) for ch, code in punctuation]
    assert keys == sorted(keys), "punctuation by code length, then code, then character"
    assert entries[0] == ("A", ".-") and entries[26] == ("0", "-----")
    assert chart_entries({"B": "-...", "A": ".-", "1": ".----", "?": "..--.."}) == [
        ("A", ".-"), ("B", "-..."), ("1", ".----"), ("?", "..--.."),
    ]


def test_cell_state_lights_matches_and_prefixes() -> None:
    assert cell_state(".-", "") == ""
    assert cell_state(".-", ".-") == "match"
    assert cell_state(".--", ".-") == "prefix"  # W could still come
    assert cell_state(".", ".-") == ""  # E is already past
    assert cell_state("-", ".") == ""
    assert cell_state("...", "..") == "prefix" and cell_state("..", "..") == "match"
    lit = {ch: cell_state(code, ".-") for ch, code in chart_entries()}
    assert lit["A"] == "match"
    expected = {ch for ch, code in MORSE_TABLE.items() if code.startswith(".-") and code != ".-"}
    assert {ch for ch, s in lit.items() if s == "prefix"} == expected
    assert {"W", "R", "P"} <= expected
