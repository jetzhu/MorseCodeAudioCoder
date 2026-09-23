"""Key bindings for the hand key: which keyboard key works the straight key and each paddle.

A binding table maps the three actions, ``"key"`` (straight key), ``"dit"``
and ``"dah"``, to key names. The names are opaque strings: the desktop UI
uses Qt's portable key names (``"Space"``, ``"Left"``, ``"J"``), the web app
``KeyboardEvent.code`` values (``"Space"``, ``"ArrowLeft"``, ``"KeyJ"``);
``web/js/bindings.js`` mirrors this module for the web app.

The rules are the same everywhere: every action always has exactly one key,
no key serves two actions, and assigning a key that another action holds
swaps the two, so a rebind never leaves an action unbound. Tables read back
from storage go through :func:`sanitize`, which repairs anything broken with
the defaults instead of raising.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["ACTIONS", "DEFAULTS", "action_for", "is_default", "rebind", "sanitize"]

ACTIONS: tuple[str, ...] = ("key", "dit", "dah")
"""The bindable actions, in display order."""

DEFAULTS: dict[str, str] = {"key": "Space", "dit": "Left", "dah": "Right"}
"""Qt portable key names for the desktop app: Space, the left arrow, the right arrow."""


def sanitize(mapping: Any, defaults: Mapping[str, str] = DEFAULTS) -> dict[str, str]:
    """A complete, duplicate-free binding table from untrusted input.

    Missing, empty or non-string entries take their default. A key used by
    two actions stays with the first action in :data:`ACTIONS` order; the
    later one falls back to its default, or to any default not in use if its
    own default is taken too. Anything that is not a mapping yields the
    defaults. Extra keys in ``mapping`` are dropped.
    """
    src: Mapping[str, Any] = mapping if isinstance(mapping, Mapping) else {}
    out: dict[str, str] = {}
    for action in ACTIONS:
        value = src.get(action)
        out[action] = value.strip() if isinstance(value, str) and value.strip() else defaults[action]
    used: set[str] = set()
    for action in ACTIONS:
        key = out[action]
        if key in used:
            key = defaults[action]
            if key in used:
                key = next((d for d in defaults.values() if d not in used), key)
            out[action] = key
        used.add(key)
    return out


def rebind(bindings: Mapping[str, str], action: str, key: str) -> dict[str, str]:
    """``bindings`` with ``action`` on ``key``; an action already holding ``key`` takes the old key.

    Raises ``ValueError`` for an unknown action or an empty key.
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string")
    key = key.strip()
    out = dict(bindings)
    old = out.get(action)
    for other, other_key in out.items():
        if other != action and other_key == key:
            out[other] = old if old else key
    out[action] = key
    return out


def action_for(bindings: Mapping[str, str], key: str) -> str | None:
    """The action bound to ``key``, or None."""
    for action in ACTIONS:
        if bindings.get(action) == key:
            return action
    return None


def is_default(bindings: Mapping[str, str], defaults: Mapping[str, str] = DEFAULTS) -> bool:
    """Whether every action is on its default key."""
    return all(bindings.get(a) == defaults[a] for a in ACTIONS)
