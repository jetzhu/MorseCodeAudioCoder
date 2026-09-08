"""Morse timing decoder: turns final ON/OFF runs into text.

The decoder consumes :class:`morse.runs.Run` objects one at a time (see
``docs/INTERFACES.md``) and keeps an adaptive estimate of the dit length ``T``
(``dit_ms``) and of the reverb offset ``d`` (``offset_ms``) that lengthens
every mark and shortens every gap by the same amount.

Timing rule
-----------
Keep the last ``window`` raw mark lengths and gap lengths. ``M`` is the 10th
percentile of the marks and ``G`` the 10th percentile of the gaps after
capping each gap at ``20 * M``. Percentiles use the nearest-rank method:
sort ascending and take the element at 1-based rank ``ceil(q / 100 * n)``
(so the minimum for ``n <= 10``, the 2nd smallest for ``11 <= n <= 20``, the
3rd smallest for ``n = 30``); never interpolate between order statistics.
Every port (``web/js/decoder.js``, ``tools/export_vectors.py``) must use the
same rule or ``dit_ms``/``offset_ms`` will not reproduce
``web/test/vectors.json``; see :func:`nearest_rank_percentile`.

Then ``T = (M + G) / 2`` and ``d = (M - G) / 2``; if ``d < 0`` use ``T = M``
and ``d = 0``. With fewer than two marks or one gap the estimate falls back to
``T = 1200 / wpm`` when a speed was given, else ``T = 150 ms``, and ``d = 0``.
When ``adaptive`` is False and ``wpm`` was given, ``T`` stays fixed and only
``d`` adapts.

Marks are corrected as ``ms - d`` and gaps as ``ms + d``. A corrected mark
``< 2T`` is a dit, otherwise a dah. A corrected gap ``< 2T`` is inside a
letter, ``2T..5T`` ends the letter and ``>= 5T`` ends the word (one space).
Unknown symbol sequences emit ``?``. A leading gap before any mark emits
nothing and is not used for timing.

A mark of ``max_mark_ms`` or longer (default 5000 ms; a constructor keyword and
a public attribute) is not a symbol: it is ignored, the pending buffer is
cleared and nothing is emitted, so a held button or a detector timeout never
produces ``?`` or disturbs the letter that follows. The comparison uses the
measured length before the offset correction and is strict, so a mark of
exactly ``max_mark_ms`` is still a dah. An ignored mark is not used for timing
and increments neither ``letter_count`` nor ``unknown_count``.

This module is pure Python: no numpy, no Qt, no audio.
"""
from __future__ import annotations

from collections import deque
from math import ceil
from collections.abc import Iterable

from morse.runs import Run
from morse.table import lookup

__all__ = ["MorseDecoder", "DEFAULT_DIT_MS", "DEFAULT_MAX_MARK_MS", "nearest_rank_percentile"]

DEFAULT_DIT_MS: float = 150.0
"""Dit length used before enough runs have arrived and no WPM was given."""

DEFAULT_MAX_MARK_MS: float = 5000.0
"""Measured marks longer than this (ms) are not symbols and are ignored."""

_PERCENTILE = 10.0
_GAP_CAP_FACTOR = 20.0
_DIT_DAH_SPLIT = 2.0  # corrected mark >= this * T is a dah
_LETTER_GAP = 2.0  # corrected gap >= this * T ends the letter
_WORD_GAP = 5.0  # corrected gap >= this * T ends the word
_IDLE_FLUSH = 7.0  # idle OFF (corrected) > this * T flushes the pending letter


def nearest_rank_percentile(values: Iterable[float], q: float) -> float:
    """Return the q-th percentile of ``values`` by the nearest-rank method.

    This is the reference definition of "percentile" for the timing rule in
    ``docs/INTERFACES.md`` and for every port of it (``web/js/decoder.js``,
    ``tools/export_vectors.py``): sort ascending and take the element at
    1-based rank ``ceil(q / 100 * n)``, clamped to ``1..n``. The result is
    always one of the observed values; for ``q = 10`` it is the minimum when
    ``n <= 10``, the 2nd smallest for ``11 <= n <= 20`` and the 3rd smallest
    for ``n = 30``. Interpolating between order statistics, as
    ``numpy.percentile`` does by default (3.9 for ``range(1, 31)`` where this
    gives 3), can blend a dit with a dah when only a few marks have been seen;
    an order statistic cannot. ``values`` must be non-empty.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile of an empty sequence")
    rank = ceil(q / 100.0 * n)
    index = min(max(rank - 1, 0), n - 1)
    return ordered[index]


_percentile = nearest_rank_percentile  # short internal name, kept for existing imports


class MorseDecoder:
    """Timing state machine that decodes final runs into text.

    Attributes:
        dit_ms: current dit-length estimate ``T`` in ms.
        offset_ms: current reverb correction ``d`` in ms (marks measured
            ``d`` too long, gaps ``d`` too short).
        max_mark_ms: a mark measured longer than this is ignored (buffer
            cleared, nothing emitted) instead of being decoded as a dah.
        buffer: pending symbols of the letter in progress, e.g. ``'.-'``.
        text: everything emitted so far.
        letter_count: number of letters emitted (unknown ``?`` included).
        unknown_count: number of unknown symbol sequences emitted as ``?``.
    """

    def __init__(
        self,
        wpm: float | None = None,
        adaptive: bool = True,
        window: int = 30,
        max_mark_ms: float = DEFAULT_MAX_MARK_MS,
    ) -> None:
        """Create a decoder.

        Args:
            wpm: optional keying speed; seeds ``T = 1200 / wpm`` until enough
                runs have arrived (and fixes it when ``adaptive`` is False).
            adaptive: when False and ``wpm`` is given, ``T`` never changes and
                only the offset ``d`` adapts to the measured runs.
            window: number of recent marks and of recent gaps kept for the
                timing estimate.
            max_mark_ms: marks measured longer than this are not symbols and
                are ignored (see :meth:`feed`); ``float('inf')`` disables the
                rule. Must be positive.
        """
        if wpm is not None and not wpm > 0:
            raise ValueError(f"wpm must be positive, got {wpm!r}")
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window!r}")
        if not max_mark_ms > 0:
            raise ValueError(f"max_mark_ms must be positive, got {max_mark_ms!r}")
        self._wpm: float | None = float(wpm) if wpm is not None else None
        self._adaptive: bool = bool(adaptive)
        self._window: int = int(window)
        self.max_mark_ms: float = float(max_mark_ms)
        self._marks: deque[float] = deque(maxlen=self._window)
        self._gaps: deque[float] = deque(maxlen=self._window)
        self._seen_mark: bool = False

        self.dit_ms: float = self._seed_dit_ms()
        self.offset_ms: float = 0.0
        self.buffer: str = ""
        self.text: str = ""
        self.letter_count: int = 0
        self.unknown_count: int = 0

    # ------------------------------------------------------------------ public

    @property
    def wpm(self) -> float:
        """Current speed estimate, ``1200 / dit_ms``."""
        return 1200.0 / self.dit_ms

    def feed(self, run: Run) -> str:
        """Consume one final run and return the newly emitted text.

        Marks append a dit or dah to :attr:`buffer` and return ``''``. Gaps
        return the letter they close (``''`` for an intra-letter gap), plus a
        space when they close a word. Everything returned is also appended to
        :attr:`text`.

        A mark measured longer than :attr:`max_mark_ms` is not a symbol (a
        held button, or the detector's stuck-ON timeout): it clears
        :attr:`buffer`, returns ``''``, leaves the timing estimate and the
        counters alone, and the gap after it is handled as usual, so the next
        letter decodes cleanly.
        """
        ms = float(run.ms)
        if run.on:
            if ms >= self.max_mark_ms:
                self.buffer = ""
                return ""
            self._marks.append(ms)
            self._seen_mark = True
            self._update_timing()
            corrected = ms - self.offset_ms
            self.buffer += "." if corrected < _DIT_DAH_SPLIT * self.dit_ms else "-"
            return ""

        if not self._seen_mark:
            return ""  # leading silence carries no information
        self._gaps.append(ms)
        self._update_timing()
        corrected = ms + self.offset_ms
        if corrected < _LETTER_GAP * self.dit_ms:
            return ""
        emitted = self._flush_letter()
        if corrected >= _WORD_GAP * self.dit_ms:
            emitted += self._end_word()
        return emitted

    def idle(self, off_ms: float) -> str:
        """Report the length of the current, unfinished OFF run.

        Once ``off_ms + offset_ms > 7 * dit_ms`` the pending letter is decoded
        and a space appended, so the last letter of a message appears without
        waiting for the next tone. Returns the newly emitted text; with an
        empty :attr:`buffer` nothing happens, so the flush occurs exactly once.
        """
        if not self.buffer:
            return ""
        if off_ms + self.offset_ms > _IDLE_FLUSH * self.dit_ms:
            return self._flush_letter() + self._end_word()
        return ""

    def reset(self, keep_timing: bool = False) -> None:
        """Clear the text, the pending letter and the counters.

        With ``keep_timing`` the recent-run windows and the current ``T`` and
        ``d`` survive; otherwise timing returns to the seed values.
        """
        self.buffer = ""
        self.text = ""
        self.letter_count = 0
        self.unknown_count = 0
        self._seen_mark = False
        if not keep_timing:
            self._marks.clear()
            self._gaps.clear()
            self._update_timing()

    # ----------------------------------------------------------------- private

    def _seed_dit_ms(self) -> float:
        return 1200.0 / self._wpm if self._wpm is not None else DEFAULT_DIT_MS

    def _update_timing(self) -> None:
        """Recompute ``dit_ms`` and ``offset_ms`` from the recent runs."""
        seed = self._seed_dit_ms()
        if len(self._marks) < 2 or len(self._gaps) < 1:
            self.dit_ms = seed
            self.offset_ms = 0.0
            return
        m = _percentile(self._marks, _PERCENTILE)
        cap = _GAP_CAP_FACTOR * m
        g = _percentile((min(gap, cap) for gap in self._gaps), _PERCENTILE)
        d = (m - g) / 2.0
        if not self._adaptive and self._wpm is not None:
            self.dit_ms = seed
            self.offset_ms = max(d, 0.0)
            return
        if d < 0.0:
            self.dit_ms = m
            self.offset_ms = 0.0
        else:
            self.dit_ms = (m + g) / 2.0
            self.offset_ms = d

    def _flush_letter(self) -> str:
        """Decode :attr:`buffer` into one character, append it and return it."""
        if not self.buffer:
            return ""
        char = lookup(self.buffer)
        if char is None:
            char = "?"
            self.unknown_count += 1
        self.letter_count += 1
        self.buffer = ""
        self.text += char
        return char

    def _end_word(self) -> str:
        """Append exactly one space after the last word, if not already there."""
        if not self.text or self.text.endswith(" "):
            return ""
        self.text += " "
        return " "
