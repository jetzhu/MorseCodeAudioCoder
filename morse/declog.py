"""The decoded-text log: every character the decoder emitted, stamped with when it arrived.

Both apps keep one of these beside the decoder (``web/js/declog.js`` mirrors
this module) and export it on request as a plain-text table or CSV, one
line per word, with the clock time and the audio time at which the word's
first character was decoded. The log survives Clear text, which only wipes
the display, and is reset when a new stream starts.

Two clocks appear in every line: ``elapsed_ms`` is audio time since the
stream started (blocks processed times the block length, what the status bar
counts), ``wall_s`` is the computer's clock (``time.time()``) when the text
was decoded. Words are split on whitespace; a word's stamps are those of its
first character and the audio time of its last.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo

__all__ = [
    "DecodedLog",
    "LogEntry",
    "LogWord",
    "default_filename",
    "format_clock",
    "format_elapsed",
    "format_iso",
]


@dataclass(frozen=True)
class LogEntry:
    """Text the decoder emitted in one go (a letter, a letter plus a word space, a flush)."""

    text: str
    elapsed_ms: float
    wall_s: float


@dataclass(frozen=True)
class LogWord:
    """One word of the log with the stamps of its first character and the audio time of its last."""

    text: str
    elapsed_ms: float
    wall_s: float
    end_ms: float


class DecodedLog:
    """Ordered record of emitted text; see the module docstring."""

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def add(self, text: str, elapsed_ms: float, wall_s: float) -> None:
        """Record ``text`` decoded at audio time ``elapsed_ms`` and clock time ``wall_s``; '' is ignored."""
        if text:
            self._entries.append(LogEntry(str(text), float(elapsed_ms), float(wall_s)))

    def reset(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        return tuple(self._entries)

    @property
    def text(self) -> str:
        """Everything logged, concatenated (the decoder's text before any Clear)."""
        return "".join(e.text for e in self._entries)

    @property
    def is_empty(self) -> bool:
        return not any(e.text.strip() for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def words(self) -> list[LogWord]:
        """The log split on whitespace, each word stamped by its first character."""
        words: list[LogWord] = []
        chars: list[str] = []
        start: LogEntry | None = None
        last_ms = 0.0
        for entry in self._entries:
            for ch in entry.text:
                if ch.isspace():
                    if chars and start is not None:
                        words.append(LogWord("".join(chars), start.elapsed_ms, start.wall_s, last_ms))
                    chars, start = [], None
                    continue
                if start is None:
                    start = entry
                chars.append(ch)
                last_ms = entry.elapsed_ms
        if chars and start is not None:
            words.append(LogWord("".join(chars), start.elapsed_ms, start.wall_s, last_ms))
        return words

    # ----------------------------------------------------------------- export

    def render_text(self, header: Sequence[tuple[str, str]] = (), now_s: float | None = None,
                    tz: tzinfo | None = None) -> str:
        """The log as a readable table, one word per line.

        ``header`` is a sequence of ``(label, value)`` lines shown after the
        title; ``now_s`` is the export time (``None`` leaves the Exported
        line out); ``tz`` is the time zone for every clock (local when None).
        """
        words = self.words()
        lines = ["Beeper Morse Console decoded log"]
        if now_s is not None:
            lines.append(f"Exported: {format_iso(now_s, tz).replace('T', ' ')[:19]}")
        for label, value in header:
            lines.append(f"{label}: {value}")
        lines.append(f"Words: {len(words)}")
        lines.append("Time is the computer clock when the word's first letter was decoded; "
                     "elapsed is audio time since the stream started.")
        lines.append("")
        lines.append(f"{'time':<12}  {'elapsed':>8}  word")
        for w in words:
            lines.append(f"{format_clock(w.wall_s, tz):<12}  {format_elapsed(w.elapsed_ms):>8}  {w.text}")
        return "\n".join(lines) + "\n"

    def render_csv(self, tz: tzinfo | None = None) -> str:
        """The log as CSV: ``time,elapsed_s,word`` with a header line, ISO local time, seconds to 2 places."""
        lines = ["time,elapsed_s,word"]
        for w in self.words():
            lines.append(f"{format_iso(w.wall_s, tz)},{w.elapsed_ms / 1000.0:.2f},{_csv_field(w.text)}")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ formatting


def _csv_field(text: str) -> str:
    if any(c in text for c in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _dt(wall_s: float, tz: tzinfo | None) -> datetime:
    return datetime.fromtimestamp(float(wall_s), tz)


def format_clock(wall_s: float, tz: tzinfo | None = None) -> str:
    """``HH:MM:SS.mmm`` of a clock time (local unless ``tz`` is given)."""
    d = _dt(wall_s, tz)
    return f"{d:%H:%M:%S}.{d.microsecond // 1000:03d}"


def format_iso(wall_s: float, tz: tzinfo | None = None) -> str:
    """``YYYY-MM-DDTHH:MM:SS.mmm`` of a clock time, without a zone designator."""
    d = _dt(wall_s, tz)
    return f"{d:%Y-%m-%dT%H:%M:%S}.{d.microsecond // 1000:03d}"


def format_elapsed(ms: float) -> str:
    """Audio time as ``+M:SS.d`` (minutes unbounded, tenths of a second)."""
    tenths = int(round(max(0.0, float(ms)) / 100.0))
    minutes, rest = divmod(tenths, 600)
    seconds, tenth = divmod(rest, 10)
    return f"+{minutes}:{seconds:02d}.{tenth}"


def default_filename(wall_s: float, ext: str = "txt", tz: tzinfo | None = None) -> str:
    """``morse_log_YYYYmmdd_HHMMSS.<ext>`` for an export at ``wall_s``."""
    return f"morse_log_{_dt(wall_s, tz):%Y%m%d_%H%M%S}.{ext.lstrip('.')}"
