"""Tests for morse.channels (mirrored by web/test/channels.test.mjs)."""
from __future__ import annotations

from morse.channels import DEFAULTS, LISTEN, SEND, effective, sanitize, timing_state_at, vibration_pattern
from morse.player import build_timing


def test_defaults_select_everything_and_sanitize_repairs_input() -> None:
    assert LISTEN == ("mic", "camera") and SEND == ("audio", "light", "torch", "vibrate")
    assert sanitize(None) == DEFAULTS
    assert sanitize("junk") == sanitize(42) == DEFAULTS
    s = sanitize({"listen": {"mic": False, "camera": "yes"}, "send": {"audio": 0, "light": False, "extra": True}})
    assert s == {"listen": {"mic": False, "camera": True}, "send": {"audio": True, "light": False, "torch": True, "vibrate": True}}
    assert sanitize(None) is not DEFAULTS, "a fresh dict every time"


def test_effective_is_selected_and_available() -> None:
    sel = sanitize({"send": {"audio": False}})
    avail = {"listen": {"mic": True, "camera": False}, "send": {"audio": True, "light": True, "torch": False, "vibrate": True}}
    assert effective(sel, avail) == {"listen": {"mic": True, "camera": False},
                                     "send": {"audio": False, "light": True, "torch": False, "vibrate": True}}


def test_timing_state_at_walks_the_keying_sequence() -> None:
    t = build_timing("A", 12)  # dit 100, gap 100, dah 300
    assert timing_state_at(t, -1) is False
    assert timing_state_at(t, 0) is True
    assert timing_state_at(t, 99.9) is True
    assert timing_state_at(t, 100) is False
    assert timing_state_at(t, 200) is True
    assert timing_state_at(t, 499.9) is True
    assert timing_state_at(t, 500) is False
    assert timing_state_at([], 0) is False


def test_vibration_pattern_matches_the_page() -> None:
    assert vibration_pattern(build_timing("A", 12)) == [100, 100, 300]
    assert vibration_pattern(build_timing("E E", 12)) == [100, 700, 100]
    assert vibration_pattern([(False, 50), (True, 20)]) == [0, 50, 20]
    assert vibration_pattern([(True, 10), (True, 15), (False, 0), (False, 5)]) == [25, 5]
    assert vibration_pattern([(True, 10), (False, 30)]) == [10, 30]
    assert vibration_pattern([]) == []
