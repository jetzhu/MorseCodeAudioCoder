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
and ``d = 0``. When only one mark class has been seen (longest mark below
twice the shortest) and ``M >= 2.5 * G``, the marks are dahs rather than
dits: ``T = (M + G) / 4`` and ``d = (M - 3G) / 4`` (a second gap class, when
present, arbitrates: see :meth:`MorseDecoder._marks_are_dahs`). With both
classes present but dits rare, ``M`` is taken from the dit class itself
(marks of 20 ms or less never anchor that class). With fewer than two marks
or one gap the estimate falls back to ``T = 1200 / wpm`` when a speed was
given, else ``T = 150 ms``, and ``d = 0``. When ``adaptive`` is False and
``wpm`` was given, ``T`` stays fixed and only ``d`` adapts.

Hold-back
---------
Without a WPM seed a mark alone cannot tell a slow dit from a fast dah, so
runs are held back (``timing_ready`` False, ``pending_count`` > 0) until the
estimate is trusted: both mark classes seen (longest mark >= 2 x shortest)
or three marks (five, or a letter gap, when three marks of one class look
like dahs: reverberant dits with jitter look the same). Once trusted, a mark
shorter than ``0.4 T`` is a click and ignored, unless four arrive in a row
with Morse-like spacing, which is a faster speed and opens a valve until the
next long gap. The held runs are then classified in one go with that
estimate, so a message may open with T, M, O or a digit and may be keyed at
2 WPM. While holding, ``idle()`` flushes after the longer of ``7T`` and
``3.5 x`` the longest held mark, using the best estimate available (a lone
mark falls back to the 150 ms seed: shorter than 300 ms is E, else T).

Marks are corrected as ``ms - d`` and gaps as ``ms + d``. A corrected mark
``< 2T`` is a dit, otherwise a dah. A corrected gap ``< 2T`` is inside a
letter, ``2T..5T`` ends the letter and ``>= 5T`` ends the word (one space).
Unknown symbol sequences emit ``?``. A leading gap before any mark emits
nothing and is not used for timing.

A mark of ``max_mark_ms`` or longer (default 5000 ms; a constructor keyword and
a public attribute) is not a symbol: it is ignored, the pending buffer is
cleared and nothing is emitted, so a held button or a detector timeout never
produces ``?`` or disturbs the letter that follows. The comparison uses the
measured length before the offset correction and is inclusive (``>=``), so the
detector's 5000 ms stuck-ON timeout is dropped too. An ignored mark is not
used for timing and increments neither ``letter_count`` nor ``unknown_count``.

This module is pure Python: no numpy, no Qt, no audio.
"""
from __future__ import annotations

from collections import deque
from math import ceil, log
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
_TRUST_RATIO = 2.0  # longest/shortest mark >= this: both mark classes have been seen
_TRUST_MARKS = 3  # ... or this many marks: the estimate is trusted and classification starts
_GLITCH_FRACTION = 0.4  # trusted: a measured run shorter than this * T is a glitch, not a symbol
_GLITCH_STREAK_MAX = 4  # ... unless this many arrive in a row: then the keying got faster
_GLITCH_STREAK_GAP_FACTOR = 8.0  # a short mark after a gap longer than this * its length is a click
_TRUST_MARKS_AMBIGUOUS = 5  # single-class, dah-like marks with no letter gap yet: wait for this many
_DAH_RULE_RATIO = 2.5  # single-class marks >= this * shortest gap are dahs, not dits
_PENDING_IDLE_FACTOR = 3.5  # untrusted: idle flush waits this * longest pending mark
_FRAGMENT_MS = 20.0  # marks this short never anchor the dit class (debounce leftovers)


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
        # Runs held back until the timing estimate is trusted (Auto mode only:
        # a WPM seed makes the estimate trusted from the start).
        self._pending: list[Run] = []
        self._trusted: bool = self._wpm is not None
        self._glitch_streak: int = 0
        self._last_gap_ms: float | None = None

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

    @property
    def timing_ready(self) -> bool:
        """True once the estimate is trusted and runs are classified as they arrive.

        Without a WPM seed the first runs are held back: a mark alone cannot
        tell a slow dit from a fast dah. The estimate is trusted once both
        mark classes have been seen (longest mark >= 2 x shortest) or three
        marks have arrived; the held runs are then decoded in one go.
        """
        return self._trusted

    @property
    def pending_count(self) -> int:
        """Number of final runs held back while the estimate is not yet trusted."""
        return len(self._pending)

    @property
    def provisional(self) -> str:
        """What the held-back runs read as under the current estimate, e.g. ``'... -'``.

        Dits and dahs of the runs in :attr:`pending_count`, with a space at
        each gap that would end a letter; empty when nothing is held. Shown
        by the UIs while the speed estimate settles, and replaced by the
        final reading when the runs are replayed.
        """
        out = ""
        for run in self._pending:
            if run.on:
                out += "." if run.ms - self.offset_ms < _DIT_DAH_SPLIT * self.dit_ms else "-"
            elif run.ms + self.offset_ms >= _LETTER_GAP * self.dit_ms and out and not out.endswith(" "):
                out += " "
        return out

    def adopt_timing(self, other: MorseDecoder) -> None:
        """Take over ``other``'s recent-run windows and trust state, then recompute.

        Used when the UI swaps decoders (Auto/Manual speed): the new decoder
        starts with the measured speed and reverb offset instead of
        re-learning them. Text, pending letter, counters and held-back runs
        are not copied; flush ``other`` first (``idle(float('inf'))``) if a
        letter is in progress. With a fixed ``wpm`` only the windows are
        taken, so ``T`` stays at the seed and ``d`` comes from the history.
        """
        self._marks.clear()
        self._marks.extend(other._marks)
        self._gaps.clear()
        self._gaps.extend(other._gaps)
        self._seen_mark = other._seen_mark
        if self._wpm is None:
            self._trusted = other._trusted
        self._update_timing()

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
        settled = self._trusted and not self._pending
        if run.on:
            if ms >= self.max_mark_ms:
                self.buffer = ""
                self._pending.clear()
                return ""
            if settled and ms < _GLITCH_FRACTION * self.dit_ms:
                # A click, far shorter than a dit: not a symbol, not timing
                # evidence. Four in a row with Morse spacing are a faster speed
                # instead: the valve opens and stays open until a long gap, so
                # the window adapts. A click after silence always resets it.
                if self._last_gap_ms is not None and self._last_gap_ms > _GLITCH_STREAK_GAP_FACTOR * ms:
                    self._glitch_streak = 0
                self._glitch_streak += 1
                if self._glitch_streak < _GLITCH_STREAK_MAX:
                    return ""
                self._glitch_streak = _GLITCH_STREAK_MAX
            self._marks.append(ms)
            self._seen_mark = True
            self._update_timing()
            if settled:
                return self._classify_mark(ms)
            self._pending.append(run)
            return self._replay() if self._trusted else ""

        self._last_gap_ms = ms
        if not self._seen_mark:
            return ""  # leading silence carries no information
        if settled and ms < _GLITCH_FRACTION * self.dit_ms:
            return ""  # the gap around a glitch or a dropout: inside the letter, not timing evidence
        self._gaps.append(ms)
        self._update_timing()
        if settled:
            return self._classify_gap(ms)
        if not self._pending:
            return ""  # untrusted with nothing held: an idle flush already closed the letter
        self._pending.append(run)
        return self._replay() if self._trusted else ""

    def idle(self, off_ms: float) -> str:
        """Report the length of the current, unfinished OFF run.

        Once ``off_ms + offset_ms > 7 * dit_ms`` the pending letter is decoded
        and a space appended, so the last letter of a message appears without
        waiting for the next tone. Returns the newly emitted text; with an
        empty :attr:`buffer` nothing happens, so the flush occurs exactly once.

        While runs are held back (see :attr:`timing_ready`) the flush waits
        for the longer of ``7 * dit_ms`` and ``3.5 x`` the longest held mark,
        because that mark may be a dah whose letter gap is as long as itself;
        it then decodes the held runs with the best estimate available.
        """
        if self._pending:
            longest = max(r.ms for r in self._pending if r.on)
            limit = max(_IDLE_FLUSH * self.dit_ms, _PENDING_IDLE_FACTOR * longest)
            if off_ms + self.offset_ms > limit:
                return self._replay() + self._flush_letter() + self._end_word()
            return ""
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
        self._pending.clear()
        self._glitch_streak = 0
        self._last_gap_ms = None
        if not keep_timing:
            self._marks.clear()
            self._gaps.clear()
            self._trusted = self._wpm is not None
            self._update_timing()

    # ----------------------------------------------------------------- private

    def _seed_dit_ms(self) -> float:
        return 1200.0 / self._wpm if self._wpm is not None else DEFAULT_DIT_MS

    def _classify_mark(self, ms: float) -> str:
        corrected = ms - self.offset_ms
        self.buffer += "." if corrected < _DIT_DAH_SPLIT * self.dit_ms else "-"
        return ""

    def _classify_gap(self, ms: float) -> str:
        corrected = ms + self.offset_ms
        if corrected < _LETTER_GAP * self.dit_ms:
            return ""
        emitted = self._flush_letter()
        if corrected >= _WORD_GAP * self.dit_ms:
            emitted += self._end_word()
        return emitted

    def _replay(self) -> str:
        """Classify every held-back run with the current estimate, oldest first."""
        emitted = ""
        for run in self._pending:
            emitted += self._classify_mark(run.ms) if run.on else self._classify_gap(run.ms)
        self._pending.clear()
        return emitted

    def _update_timing(self) -> None:
        """Recompute ``dit_ms`` and ``offset_ms`` from the recent runs; refresh trust.

        ``M`` and ``G`` are the nearest-rank 10th percentiles of the recent marks
        and (capped) gaps. Normally the shortest mark is a dit (``T + d``) and
        the shortest gap a dit gap (``T - d``), so ``T = (M + G) / 2`` and
        ``d = (M - G) / 2``. When only one mark class has been seen (longest
        mark below twice the shortest) and the marks are at least 2.5 x the
        shortest gap, they are dahs (``3T + d``): ``T = (M + G) / 4`` and
        ``d = (M - 3G) / 4``. That keeps a message opening with T, M, O or a
        digit like 0 from being read as dits.
        """
        seed = self._seed_dit_ms()
        if len(self._marks) < 2 or len(self._gaps) < 1:
            self.dit_ms = seed
            self.offset_ms = 0.0
            return
        m = _percentile(self._marks, _PERCENTILE)
        cap = _GAP_CAP_FACTOR * m
        g = _percentile((min(gap, cap) for gap in self._gaps), _PERCENTILE)
        if not self._adaptive and self._wpm is not None:
            self.dit_ms = seed
            self.offset_ms = max((m - g) / 2.0, 0.0)
            return
        single_class = max(self._marks) < _TRUST_RATIO * min(self._marks)
        if not single_class:
            # Both classes present but dits may be rare (digits, "MOM"): when
            # the 10th percentile lands in the dah class, take the dit class
            # itself. Marks of two blocks or less are debounce leftovers and
            # never anchor the class.
            anchor = min((x for x in self._marks if x > _FRAGMENT_MS), default=min(self._marks))
            if m >= _DAH_RULE_RATIO * g and m >= _TRUST_RATIO * anchor:
                dits = [x for x in self._marks if anchor <= x < _TRUST_RATIO * anchor]
                m = _percentile(dits, 50.0)
        if single_class and m >= _DAH_RULE_RATIO * g and self._marks_are_dahs(m, g):
            self.dit_ms = (m + g) / 4.0
            self.offset_ms = max((m - 3.0 * g) / 4.0, 0.0)
        else:
            d = (m - g) / 2.0
            if d < 0.0:
                self.dit_ms = m
                self.offset_ms = 0.0
            else:
                self.dit_ms = (m + g) / 2.0
                self.offset_ms = d
        if not self._trusted:
            # Three marks of one class are enough, unless they look like dahs
            # (>= 2.5 x the shortest gap) with no letter gap yet to arbitrate:
            # reverberant dits with jitter look the same, so wait for a letter
            # gap or five marks.
            n = len(self._marks)
            ambiguous = single_class and m >= _DAH_RULE_RATIO * g
            if (not single_class or n >= _TRUST_MARKS_AMBIGUOUS
                    or (n >= _TRUST_MARKS and (not ambiguous or self._second_gap_class(m, g) is not None))):
                self._trusted = True

    def _second_gap_class(self, m: float, g: float) -> float | None:
        """The shortest recent gap at least twice ``g`` (a letter or word gap), capped at ``20 m``."""
        cap = _GAP_CAP_FACTOR * m
        longer = [min(x, cap) for x in self._gaps if min(x, cap) >= _TRUST_RATIO * g]
        return min(longer) if longer else None

    def _marks_are_dahs(self, m: float, g: float) -> bool:
        """Single-class window with ``m >= 2.5 g``: dahs (``3T + d``) or reverberant dits?

        Without a second gap class the ratio alone decides. With one (the
        shortest gap at least twice ``g``: a letter or word gap), compare it
        with the letter and word gaps each hypothesis predicts and keep the
        closer in log distance. Dahs: letter ``(m + 3g) / 2``, word
        ``(3m + 5g) / 2``. Dits: letter ``m + 2g``, word ``3m + 4g``.
        """
        g2 = self._second_gap_class(m, g)
        if g2 is None:
            return True

        def score(letter: float, word: float) -> float:
            return min(abs(log(g2 / letter)), abs(log(g2 / word)))

        return score((m + 3.0 * g) / 2.0, (3.0 * m + 5.0 * g) / 2.0) <= score(m + 2.0 * g, 3.0 * m + 4.0 * g)

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
