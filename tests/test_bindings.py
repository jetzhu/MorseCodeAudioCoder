"""Tests for morse.bindings (mirrored by web/test/bindings.test.mjs)."""
from __future__ import annotations

import pytest

from morse.bindings import ACTIONS, DEFAULTS, action_for, is_default, rebind, sanitize


def test_defaults_are_complete_and_distinct() -> None:
    assert tuple(DEFAULTS) == ACTIONS
    assert len(set(DEFAULTS.values())) == 3
    assert is_default(DEFAULTS)
    assert sanitize(DEFAULTS) == DEFAULTS


@pytest.mark.parametrize("garbage", [None, 42, "Space", [], {"key": 1}, {"dit": ""}, {"nothing": "here"}])
def test_sanitize_repairs_garbage_with_defaults(garbage) -> None:
    assert sanitize(garbage) == DEFAULTS


def test_sanitize_keeps_valid_entries_and_drops_extras() -> None:
    assert sanitize({"key": "K", "dit": " J ", "dah": "L", "extra": "X"}) == {"key": "K", "dit": "J", "dah": "L"}
    assert sanitize({"key": "K"}) == {"key": "K", "dit": "Left", "dah": "Right"}


def test_sanitize_resolves_duplicates_in_action_order() -> None:
    # dit copies key: dit falls back to its default
    assert sanitize({"key": "J", "dit": "J", "dah": "L"}) == {"key": "J", "dit": "Left", "dah": "L"}
    # dah's own default is taken by dit, so dah takes the first free default
    assert sanitize({"key": "Right", "dit": "Left", "dah": "Left"}) == {"key": "Right", "dit": "Left", "dah": "Space"}


def test_rebind_swaps_when_the_key_is_taken() -> None:
    b = rebind(DEFAULTS, "dit", "J")
    assert b == {"key": "Space", "dit": "J", "dah": "Right"}
    assert DEFAULTS["dit"] == "Left", "the input is not mutated"
    swapped = rebind(b, "dah", "J")
    assert swapped == {"key": "Space", "dit": "Right", "dah": "J"}
    assert rebind(b, "dit", "J") == b, "rebinding to the same key changes nothing"
    assert len(set(rebind(swapped, "key", "Right").values())) == 3


def test_rebind_validation() -> None:
    with pytest.raises(ValueError):
        rebind(DEFAULTS, "paddle", "J")
    with pytest.raises(ValueError):
        rebind(DEFAULTS, "key", "  ")


def test_action_for_and_is_default() -> None:
    assert action_for(DEFAULTS, "Space") == "key"
    assert action_for(DEFAULTS, "Left") == "dit"
    assert action_for(DEFAULTS, "Right") == "dah"
    assert action_for(DEFAULTS, "J") is None
    assert not is_default(rebind(DEFAULTS, "key", "K"))
    assert is_default(rebind(rebind(DEFAULTS, "key", "K"), "key", "Space"))
