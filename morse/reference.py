"""The Morse chart: the characters the shared table covers, in display order, and how a cell lights up.

Both apps draw their Reference strip (section G) from :func:`chart_entries`,
so the chart can never disagree with what the decoder and encoder use, and
light the cells with :func:`cell_state` as a letter takes shape under the
key or in the decoder. ``web/js/reference.js`` mirrors this module.
"""
from __future__ import annotations

from collections.abc import Mapping

from morse.table import MORSE_TABLE

__all__ = ["cell_state", "chart_entries"]


def chart_entries(table: Mapping[str, str] = MORSE_TABLE) -> list[tuple[str, str]]:
    """``(character, code)`` pairs: letters A to Z, digits 0 to 9, then punctuation.

    Punctuation is ordered by code length, then by the code itself, then by
    the character, so the order is the same however the table is written.
    """
    letters = sorted((ch, code) for ch, code in table.items() if ch.isalpha())
    digits = sorted((ch, code) for ch, code in table.items() if ch.isdigit())
    other = sorted(((ch, code) for ch, code in table.items() if not ch.isalnum()),
                   key=lambda pair: (len(pair[1]), pair[1], pair[0]))
    return letters + digits + other


def cell_state(code: str, buffer: str) -> str:
    """How a chart cell shows against the symbols keyed so far.

    ``"match"`` when ``buffer`` is exactly ``code``, ``"prefix"`` when the
    code could still become ``buffer`` plus more symbols, ``""`` otherwise
    and always for an empty buffer.
    """
    if not buffer:
        return ""
    if code == buffer:
        return "match"
    if code.startswith(buffer):
        return "prefix"
    return ""
