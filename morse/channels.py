"""Channels: what the app listens with and what it sends with, in any combination.

Listen sources: ``mic`` (microphone) and ``camera`` (light through a camera);
send channels: ``audio`` (speakers), ``light`` (the lamp or the full screen),
``torch`` (a phone's flash) and ``vibrate`` (a phone's motor). Everything a
device supports starts selected and the selection is remembered. The
desktop app offers the microphone, audio and light; ``web/js/channels.js``
mirrors this module for the page, which adds the rest.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["DEFAULTS", "LISTEN", "SEND", "effective", "sanitize", "timing_state_at", "vibration_pattern"]

LISTEN: tuple[str, ...] = ("mic", "camera")
SEND: tuple[str, ...] = ("audio", "light", "torch", "vibrate")

DEFAULTS: dict[str, dict[str, bool]] = {
    "listen": {"mic": True, "camera": True},
    "send": {"audio": True, "light": True, "torch": True, "vibrate": True},
}
"""Everything on."""


def sanitize(raw: Any, defaults: Mapping[str, Mapping[str, bool]] = DEFAULTS) -> dict[str, dict[str, bool]]:
    """A complete selection from untrusted input: every key present and boolean."""
    src: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    def pick(group: str, keys: Iterable[str]) -> dict[str, bool]:
        g = src.get(group)
        g = g if isinstance(g, Mapping) else {}
        return {k: (g[k] if isinstance(g.get(k), bool) else bool(defaults[group][k])) for k in keys}

    return {"listen": pick("listen", LISTEN), "send": pick("send", SEND)}


def effective(selection: Mapping[str, Mapping[str, bool]],
              available: Mapping[str, Mapping[str, bool]]) -> dict[str, dict[str, bool]]:
    """What is actually in use: selected and available on this device."""
    return {
        "listen": {k: bool(selection["listen"].get(k) and available["listen"].get(k)) for k in LISTEN},
        "send": {k: bool(selection["send"].get(k) and available["send"].get(k)) for k in SEND},
    }


def timing_state_at(timing: Iterable[tuple[bool, float]], ms: float) -> bool:
    """Whether a keying sequence ``[(on, ms), ...]`` is sounding ``ms`` after its start.

    False before the start and after the end.
    """
    if not ms >= 0:
        return False
    t = 0.0
    for on, dur in timing:
        end = t + float(dur)
        if ms < end:
            return bool(on)
        t = end
    return False


def vibration_pattern(timing: Iterable[tuple[bool, float]]) -> list[int]:
    """The Vibration API pattern for a keying sequence (see ``vibrationPattern`` in the page).

    Alternating vibrate and pause lengths in whole milliseconds, starting
    with a vibration (a leading gap becomes a zero-length vibration first);
    adjacent segments of one state are merged and zero-length ones dropped.
    """
    out: list[int] = []
    on = True
    for seg_on, dur in timing:
        ms = max(0, int(round(float(dur))))
        if ms == 0:
            continue
        if bool(seg_on) == on:
            if not out:
                out.append(0)
            out[-1] += ms
        else:
            if not out:
                out.append(0)  # a leading gap: the pattern must start with a (zero-length) vibration
            out.append(ms)
            on = bool(seg_on)
    while out and out[-1] == 0:
        out.pop()
    return out
