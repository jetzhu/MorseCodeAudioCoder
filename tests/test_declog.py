"""Tests for morse.declog (mirrored by web/test/declog.test.mjs; the rendered strings match)."""
from __future__ import annotations

from datetime import timezone

import pytest

from morse.declog import DecodedLog, LogWord, default_filename, format_clock, format_elapsed, format_iso

WALL = 1_700_000_000.123  # 2023-11-14T22:13:20.123Z
UTC = timezone.utc


def _sample() -> DecodedLog:
    log = DecodedLog()
    log.add("H", 1000.0, WALL)
    log.add("E", 1400.0, WALL + 0.4)
    log.add("L", 1800.0, WALL + 0.8)
    log.add("L", 2200.0, WALL + 1.2)
    log.add("O ", 3000.0, WALL + 2.0)  # a letter and the word space arrive together
    log.add("", 3100.0, WALL + 2.1)  # ignored
    log.add("7", 4000.0, WALL + 3.0)
    log.add("3", 4400.0, WALL + 3.4)
    return log


def test_entries_words_and_text() -> None:
    log = _sample()
    assert len(log) == 7
    assert log.text == "HELLO 73"
    assert not log.is_empty
    assert log.words() == [
        LogWord("HELLO", 1000.0, WALL, 3000.0),
        LogWord("73", 4000.0, WALL + 3.0, 4400.0),
    ]
    log.add(" ", 5000.0, WALL + 4.0)
    assert [w.text for w in log.words()] == ["HELLO", "73"], "a trailing space closes nothing new"
    log.add("  A", 6000.0, WALL + 5.0)
    assert [w.text for w in log.words()] == ["HELLO", "73", "A"]
    log.reset()
    assert log.is_empty and log.words() == [] and log.text == ""
    log.add("   ", 1.0, WALL)
    assert log.is_empty, "only whitespace counts as empty"


def test_formatting_helpers() -> None:
    assert format_clock(WALL, UTC) == "22:13:20.123"
    assert format_iso(WALL, UTC) == "2023-11-14T22:13:20.123"
    assert format_elapsed(0) == "+0:00.0"
    assert format_elapsed(1000) == "+0:01.0"
    assert format_elapsed(61_250) == "+1:01.2"  # 61.25 s rounds half-even to 61.2
    assert format_elapsed(61_260) == "+1:01.3"
    assert format_elapsed(3_600_000) == "+60:00.0"
    assert format_elapsed(-5) == "+0:00.0"
    assert default_filename(WALL, tz=UTC) == "morse_log_20231114_221320.txt"
    assert default_filename(WALL, ".csv", tz=UTC) == "morse_log_20231114_221320.csv"


EXPECTED_TEXT = """Beeper Morse Console decoded log
Exported: 2023-11-14 22:13:30
Source: Replay: sos.wav
Tone: 1000 Hz
Words: 2
Time is the computer clock when the word's first letter was decoded; elapsed is audio time since the stream started.

time           elapsed  word
22:13:20.123   +0:01.0  HELLO
22:13:23.123   +0:04.0  73
"""

EXPECTED_CSV = """time,elapsed_s,word
2023-11-14T22:13:20.123,1.00,HELLO
2023-11-14T22:13:23.123,4.00,73
"""


def test_render_text_and_csv() -> None:
    log = _sample()
    header = [("Source", "Replay: sos.wav"), ("Tone", "1000 Hz")]
    assert log.render_text(header, now_s=WALL + 10.0, tz=UTC) == EXPECTED_TEXT
    assert log.render_csv(tz=UTC) == EXPECTED_CSV
    bare = log.render_text(tz=UTC)
    assert "Exported:" not in bare and bare.startswith("Beeper Morse Console decoded log\nWords: 2\n")
    empty = DecodedLog()
    assert empty.render_text(tz=UTC).endswith("word\n")
    assert empty.render_csv() == "time,elapsed_s,word\n"


def test_csv_quotes_awkward_words() -> None:
    log = DecodedLog()
    log.add('A,B"C"', 0.0, WALL)  # no space: one word with a comma and quotes
    assert log.render_csv(tz=UTC).splitlines()[1] == '2023-11-14T22:13:20.123,0.00,"A,B""C"""'


def test_add_validates_types_loosely() -> None:
    log = DecodedLog()
    log.add("X", 5, 6)
    entry = log.entries[0]
    assert (entry.text, entry.elapsed_ms, entry.wall_s) == ("X", 5.0, 6.0)
    with pytest.raises(AttributeError):
        log.entries.append(entry)  # type: ignore[attr-defined]  # a copy, not the store
