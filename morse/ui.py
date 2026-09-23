"""Desktop UI for the beeper Morse decoder: pyqtgraph + PySide6.

The window is laid out exactly like the approved mock (``web/mock.html``, the
"Beeper Morse Console" artifact) and uses its light-theme tokens (the dark
palette is applied when the operating system reports a dark colour scheme):

* a toolbar with the input device, the tone frequency with Auto-detect, the
  speed mode (Auto / Manual), Pause, Save 30 s and Clear text;
* the spectrum from 0 to 8 kHz with markers at ``f0`` and ``3 f0``;
* the tone power over the last 10 s with the hysteresis band and the ON/OFF
  strip underneath;
* the decoded text with the pending letter, the dit / dah / gap / offset
  readouts and a histogram of the last 30 mark lengths;
* a right rail with the input level in dBFS (peak hold), a 6 ms slice of the
  band-passed input, the detector pill and the key numbers;
* a full-width encode strip: message, speed, Play tone, Copy, the Morse
  output, a keying guide drawn to scale with a playhead, and timing readouts;
* a status bar with the listening state, block size, dropped blocks and the
  elapsed time.

Audio comes from :class:`morse.audio_input.AudioInput`, or from
:class:`WavReplay` when ``--wav`` is given, and every block goes through
:class:`morse.pipeline.Pipeline` from a 30 Hz ``QTimer``.  The replay source
exists so the whole window can be exercised offscreen (``QT_QPA_PLATFORM=
offscreen``) by calling the timer slot :meth:`MainWindow.on_tick` in a loop;
see ``tests/test_ui_offscreen.py``.

Nothing that happens in the audio path may kill the window: every tick is
wrapped, and errors are shown in the status bar instead of raised.

Public entry points: :func:`run_ui`, :func:`make_window`, :func:`ensure_app`,
:func:`build_encoding`, :class:`WavReplay`, :class:`MainWindow`.
"""
from __future__ import annotations

import html
import math
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from morse import table
from morse.decoder import MorseDecoder
from morse.dsp import find_tone_frequency, spectrum
from morse.pipeline import BlockResult, Pipeline, load_wav
from morse.player import TonePlayer, build_timing, render_tone

__all__ = [
    "DARK",
    "LIGHT",
    "Encoding",
    "Fonts",
    "MainWindow",
    "RingBuffer",
    "Theme",
    "WavReplay",
    "build_encoding",
    "ensure_app",
    "make_window",
    "pick_fonts",
    "pick_theme",
    "run_ui",
]

# ------------------------------------------------------------------ constants

DEFAULT_FS = 48000
"""Sample rate used for the microphone and for Play tone."""
DEFAULT_BLOCK_SIZE = 480
"""Samples per block at 48 kHz: 10 ms."""
DEFAULT_FREQ_HZ = 2491.0
"""Fallback tone frequency when ``args`` has no ``freq`` (the measured beeper)."""
REFRESH_HZ = 30
"""Timer rate: drains the audio queue and redraws the plots."""
HISTORY_S = 10.0
"""Seconds of tone power kept for the power plot and the ON/OFF strip."""
SAVE_S = 30.0
"""Seconds of raw input kept for Save 30 s."""
SPECTRUM_N = 2048
"""FFT length for the spectrum (43 ms at 48 kHz)."""
SPECTRUM_FMAX_HZ = 8000.0
"""Upper edge of the spectrum plot: shows the fundamental and the third harmonic."""
SPECTRUM_DB_MIN = -110.0
"""Bottom of the spectrum plot; the top rises with the signal and rests at -30 dB."""
SPECTRUM_DB_TOP_REST = -30.0
SPECTRUM_SMOOTHING = 0.55
"""Persistence of the previous spectrum frame (the mock uses 0.55 / 0.45)."""
WAVE_MS = 6.0
"""Length of the band-passed waveform slice in the rail."""
WAVE_REF_DBFS = -41.0
"""A sine at this rms level fills the waveform display (the beeper at 1 m)."""
AUTO_DETECT_BLOCKS = 100
"""Auto-detect averages spectra over this many blocks (1 s at 10 ms)."""
AUTO_DETECT_MIN_PROMINENCE_DB = 6.0
"""Auto-detect refuses to retune to a peak less prominent than this."""
AUTO_DETECT_FMIN_HZ = 300.0
AUTO_DETECT_TIMEOUT_S = 3.0
"""Wall-clock limit for Auto-detect: it finishes with what it has (or reports
no audio) when the source is silent, stopped or a replay has ended."""
HIST_MARKS = 30
"""Mark lengths kept for the histogram."""
PEAK_HOLD_BLOCKS = 150
"""Blocks the level meter holds its peak before decaying."""
PEAK_HOLD_DECAY_DB = 0.3
"""Peak-hold decay per block once the hold has expired."""
PLAY_LATENCY_MS = 80.0
"""Assumed output latency: the playhead starts this long after Play is pressed."""
FREQ_MIN_HZ = 100
FREQ_MAX_HZ = 8000
WPM_MIN = 2
WPM_MAX = 40
MAIN_ROW_HEIGHTS = (212, 232, 168)
"""Minimum heights of the spectrum, power and decoded panels (from the mock)."""
RAIL_WIDTH = 232
TIMING_WIDTH = 210
STRIP_HEIGHT = 26

_DB_FLOOR = 1e-12


# --------------------------------------------------------------------- theme


@dataclass(frozen=True)
class Theme:
    """Colour tokens of the mock (``web/mock.html``), one set per scheme."""

    bg: str
    panel: str
    panel2: str
    chrome: str
    line: str
    grid: str
    ink: str
    ink2: str
    ink3: str
    trace: str
    trace_fill: tuple[int, int, int, int]
    spec: str
    spec_fill: tuple[int, int, int, int]
    on: str
    off: str
    thr: str
    good: str
    focus: str
    dark: bool = False


LIGHT = Theme(
    bg="#E4E7EA", panel="#F6F7F8", panel2="#EBEEF0", chrome="#D6DADF", line="#C3C9CF",
    grid="#DCE0E4", ink="#1B2126", ink2="#4A545D", ink3="#7B858E",
    trace="#B8621A", trace_fill=(184, 98, 26, 41), spec="#276A85", spec_fill=(39, 106, 133, 36),
    on="#B8621A", off="#CBD0D5", thr="#6D5AA6", good="#2E7D4F", focus="#276A85",
)
"""The mock's light theme: the default."""

DARK = Theme(
    bg="#141719", panel="#1D2124", panel2="#242A2E", chrome="#2A3035", line="#353C42",
    grid="#282E33", ink="#E3E7EA", ink2="#A5ADB4", ink3="#6E777F",
    trace="#EFA24E", trace_fill=(239, 162, 78, 46), spec="#62B3D2", spec_fill=(98, 179, 210, 41),
    on="#EFA24E", off="#343B41", thr="#A796DB", good="#5CB77F", focus="#62B3D2", dark=True,
)
"""The mock's dark theme, used when the OS reports a dark colour scheme."""


def pick_theme() -> Theme:
    """``DARK`` when the platform reports a dark colour scheme, else ``LIGHT``."""
    app = QtGui.QGuiApplication.instance()
    scheme_dark = getattr(getattr(Qt, "ColorScheme", None), "Dark", None)
    if app is not None and scheme_dark is not None:
        try:
            if app.styleHints().colorScheme() == scheme_dark:
                return DARK
        except Exception:  # very old Qt without colorScheme()
            pass
    return LIGHT


def _qcolor(spec: str | tuple[int, int, int, int]) -> QtGui.QColor:
    if isinstance(spec, tuple):
        return QtGui.QColor(*spec)
    return QtGui.QColor(spec)


# --------------------------------------------------------------------- fonts


@dataclass(frozen=True)
class Fonts:
    """Font families in use: the mock's IBM Plex faces when installed, else fallbacks."""

    sans: str
    mono: str


def pick_fonts() -> Fonts:
    """IBM Plex Sans / Mono if installed, else Segoe UI / Consolas, else the system fonts."""
    try:
        families = set(QtGui.QFontDatabase.families())
    except Exception:
        families = set()

    def first(candidates: Sequence[str], fallback: str) -> str:
        for name in candidates:
            if name in families:
                return name
        return fallback

    app = QtWidgets.QApplication.instance()
    default_sans = app.font().family() if app is not None else "Sans Serif"
    try:
        default_mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont).family()
    except Exception:
        default_mono = "monospace"
    return Fonts(
        sans=first(("IBM Plex Sans", "Segoe UI"), default_sans),
        mono=first(("IBM Plex Mono", "Consolas", "Courier New"), default_mono),
    )


def make_font(
    fonts: Fonts,
    px: int,
    mono: bool = False,
    weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal,
    spacing_em: float = 0.0,
) -> QtGui.QFont:
    """A font of ``px`` pixels; ``spacing_em`` is extra letter spacing in em (CSS style)."""
    font = QtGui.QFont(fonts.mono if mono else fonts.sans)
    font.setPixelSize(px)
    font.setWeight(weight)
    if spacing_em:
        font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spacing_em * px)
    return font


# --------------------------------------------------------------- data helpers


class RingBuffer:
    """Fixed-length ``float32`` history with an ordered (oldest to newest) view."""

    def __init__(self, size: int, fill: float = 0.0) -> None:
        if size < 1:
            raise ValueError("size must be at least 1")
        self.size = int(size)
        self._buf = np.full(self.size, fill, dtype=np.float32)
        self._pos = 0
        self.count = 0
        """Total values pushed so far (not capped at ``size``)."""

    def push(self, value: float) -> None:
        """Append one value."""
        self._buf[self._pos] = value
        self._pos = (self._pos + 1) % self.size
        self.count += 1

    def extend(self, values: np.ndarray) -> None:
        """Append many values; only the last ``size`` of them can survive."""
        x = np.asarray(values, dtype=np.float32).ravel()
        n = x.size
        if n == 0:
            return
        if n >= self.size:
            self._buf[:] = x[-self.size:]
            self._pos = 0
        else:
            end = self._pos + n
            if end <= self.size:
                self._buf[self._pos:end] = x
            else:
                k = self.size - self._pos
                self._buf[self._pos:] = x[:k]
                self._buf[: n - k] = x[k:]
            self._pos = end % self.size
        self.count += n

    def view(self) -> np.ndarray:
        """Copy of the whole buffer, oldest first."""
        return np.concatenate((self._buf[self._pos:], self._buf[: self._pos]))

    def last(self, n: int) -> np.ndarray:
        """Copy of the newest ``n`` values (``n`` capped at ``size``), oldest first."""
        n = max(0, min(int(n), self.size))
        if n == 0:
            return np.empty(0, dtype=np.float32)
        start = self._pos - n
        if start >= 0:
            return self._buf[start:self._pos].copy()
        return np.concatenate((self._buf[start:], self._buf[: self._pos]))

    def fill(self, value: float) -> None:
        """Overwrite everything with ``value`` (the count is kept)."""
        self._buf[:] = value


class BlockSource(Protocol):
    """What the window needs from an audio source (``AudioInput`` or ``WavReplay``)."""

    fs: int
    block_size: int
    dropped: int
    device_index: int
    device_name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_blocks(self) -> list[np.ndarray]: ...


class WavReplay:
    """Block source that replays a WAV file through the microphone code path.

    Blocks are ``round(fs / 100)`` samples (10 ms) of the file's own sample
    rate, the tail zero-padded to a full block like ``decode_samples``.  With
    ``realtime`` the file is paced against the wall clock so the plots move
    as they would live, but every ``read_blocks`` call hands out at least
    ``min_blocks`` so a caller that drives the timer slot in a tight loop
    (the offscreen smoke test) finishes in a bounded number of calls.
    """

    def __init__(self, path: str | Path, block_size: int | None = None,
                 realtime: bool = True, min_blocks: int = 3) -> None:
        self.path = Path(path)
        self.fs, samples = load_wav(self.path)
        self.block_size = int(block_size) if block_size else max(2, round(self.fs / 100))
        n_full = samples.size // self.block_size
        remainder = samples.size - n_full * self.block_size
        n_blocks = n_full + (1 if remainder else 0)
        padded = np.zeros(n_blocks * self.block_size, dtype=np.float32)
        padded[: samples.size] = samples
        self._blocks = padded.reshape(n_blocks, self.block_size)
        self.realtime = bool(realtime)
        self.min_blocks = max(1, int(min_blocks))
        self.dropped = 0
        self.device_index = -1
        self.device_name = f"Replay: {self.path.name}"
        self._next = 0
        self._started_at: float | None = None

    @property
    def total_blocks(self) -> int:
        """Number of blocks in the file."""
        return int(self._blocks.shape[0])

    @property
    def running(self) -> bool:
        return self._started_at is not None

    @property
    def finished(self) -> bool:
        """True once every block has been handed out."""
        return self._next >= self.total_blocks

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = time.monotonic()

    def stop(self) -> None:
        self._started_at = None

    def read_blocks(self) -> list[np.ndarray]:
        """Next blocks (at least ``min_blocks``, more if the wall clock is ahead)."""
        if self._started_at is None:
            return []
        due = self.min_blocks
        if self.realtime:
            elapsed = time.monotonic() - self._started_at
            due = max(due, int(elapsed * self.fs / self.block_size) - self._next)
        end = min(self._next + due, self.total_blocks)
        out = [self._blocks[i] for i in range(self._next, end)]
        self._next = end
        return out


# -------------------------------------------------------------------- encoder


@dataclass(frozen=True)
class Encoding:
    """A message rendered as Morse for the encode strip."""

    text: str
    """The message as typed."""
    morse: str
    """``morse.table.encode(text)``: letters separated by a space, words by ``' / '``."""
    timing: list[tuple[bool, float]]
    """``morse.player.build_timing`` output: ``(on, ms)`` marks and gaps."""
    letters: list[tuple[str, float, float]]
    """``(character, start_ms, end_ms)`` for every encoded letter, for the guide labels."""
    total_ms: float
    """Total duration of ``timing``."""
    dit_ms: float
    """``1200 / wpm``."""

    @property
    def mark_count(self) -> int:
        return sum(1 for on, _ in self.timing if on)


def build_encoding(text: str, wpm: float) -> Encoding:
    """Encode ``text`` at ``wpm`` for display, with letter spans for the keying guide.

    Walks the words and characters exactly as :func:`morse.player.build_timing`
    does (characters without a code are skipped, empty words dropped) while
    consuming its output, so the letter spans line up with the timing to the
    millisecond.
    """
    timing = build_timing(text, wpm)
    letters: list[tuple[str, float, float]] = []
    words: list[list[tuple[str, str]]] = []
    for word in text.upper().split():
        chars = [(ch, table.MORSE_TABLE[ch]) for ch in word if ch in table.MORSE_TABLE]
        if chars:
            words.append(chars)
    idx = 0
    t = 0.0
    for wi, chars in enumerate(words):
        if wi > 0:
            t += timing[idx][1]  # word gap
            idx += 1
        for li, (ch, code) in enumerate(chars):
            if li > 0:
                t += timing[idx][1]  # letter gap
                idx += 1
            start = t
            for si in range(len(code)):
                if si > 0:
                    t += timing[idx][1]  # intra-letter gap
                    idx += 1
                t += timing[idx][1]  # the mark
                idx += 1
            letters.append((ch, start, t))
    return Encoding(text=text, morse=table.encode(text), timing=timing, letters=letters,
                    total_ms=t, dit_ms=1200.0 / float(wpm))


# ------------------------------------------------------------ small widgets


class _Painted(QtWidgets.QWidget):
    """Base for the custom-painted displays: keeps the theme and fonts."""

    def __init__(self, theme: Theme, fonts: Fonts, height: int | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.fonts = fonts
        if height is not None:
            self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed if height else QtWidgets.QSizePolicy.Policy.Expanding)

    def _painter(self) -> QtGui.QPainter:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        return p


class LevelMeter(_Painted):
    """Vertical input-level bar in dBFS (0 at the top, -90 at the bottom) with peak hold."""

    SCALE_W = 26
    GAP = 10
    DB_RANGE = 90.0

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(theme, fonts, parent=parent)
        self.setFixedHeight(150)
        self.level_db = -120.0
        self.peak_db = -120.0

    def set_level(self, level_db: float, peak_db: float) -> None:
        self.level_db = float(level_db)
        self.peak_db = float(peak_db)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt API)
        t = self.theme
        p = self._painter()
        w, h = self.width(), self.height()
        x0 = self.SCALE_W + self.GAP
        bw = max(1, w - x0)

        def y(db: float) -> float:
            return (0.0 - db) / self.DB_RANGE * h

        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QtCore.QRectF(x0, 0, bw, h), _qcolor(t.grid))
        lvl = max(-self.DB_RANGE, min(0.0, self.level_db))
        p.fillRect(QtCore.QRectF(x0, y(lvl), bw, h - y(lvl)), _qcolor(t.trace))
        peak = max(-self.DB_RANGE, min(0.0, self.peak_db))
        p.fillRect(QtCore.QRectF(x0, y(peak) - 1, bw, 2), _qcolor(t.ink))
        tick = _qcolor(t.ink3)
        tick.setAlphaF(0.5)
        for db in (-20, -40, -60, -80):
            p.fillRect(QtCore.QRectF(x0, y(db), bw, 1), tick)
        p.setPen(_qcolor(t.ink3))
        p.setFont(make_font(self.fonts, 10, mono=True))
        for db, frac in ((0, 0.0), (-20, 0.22), (-40, 0.44), (-60, 0.66), (-80, 0.88)):
            rect = QtCore.QRectF(0, frac * h - 7, self.SCALE_W, 14)
            p.drawText(rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(db))
        p.end()


class Waveform(_Painted):
    """A few milliseconds of the band-passed input, scaled so ``WAVE_REF_DBFS`` fills it."""

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(theme, fonts, height=54, parent=parent)
        self.samples = np.zeros(0, dtype=np.float32)
        self._gain = 1.0 / (10.0 ** (WAVE_REF_DBFS / 20.0) * math.sqrt(2.0))

    def set_samples(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=np.float32).ravel()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = self._painter()
        w, h = self.width(), self.height()
        mid = h / 2.0
        p.setPen(QtGui.QPen(_qcolor(t.grid), 1))
        p.drawLine(QtCore.QPointF(0, mid + 0.5), QtCore.QPointF(w, mid + 0.5))
        n = self.samples.size
        if n >= 2 and w > 1:
            amp = mid - 3.0
            ys = np.clip(self.samples * self._gain, -1.0, 1.0) * amp
            xs = np.linspace(0.0, float(w), n)
            path = QtGui.QPainterPath()
            path.moveTo(xs[0], mid - float(ys[0]))
            for x, yv in zip(xs[1:], ys[1:]):
                path.lineTo(float(x), mid - float(yv))
            p.setPen(QtGui.QPen(_qcolor(t.spec), 1.25))
            p.drawPath(path)
        p.end()


class OnOffStrip(_Painted):
    """The detector's verdict over the power plot's time span, drawn as a tape."""

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(theme, fonts, height=STRIP_HEIGHT, parent=parent)
        self.states = np.zeros(0, dtype=np.uint8)
        self.left = 40.0
        self.right = 40.0

    def set_states(self, states: np.ndarray, left: float, right: float) -> None:
        """``states`` oldest first (non-zero = ON); ``left``/``right`` are the plot's x extent."""
        self.states = np.asarray(states) != 0
        self.left = float(left)
        self.right = float(right)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = self._painter()
        h = self.height()
        left = self.left
        pw = max(1.0, self.right - left)
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QtCore.QRectF(left, 6, pw, h - 12), _qcolor(t.off))
        n = self.states.size
        if n:
            on = self.states
            edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
            starts, stops = edges[0::2], edges[1::2]
            brush = _qcolor(t.on)
            for a, b in zip(starts, stops):
                x = left + a / n * pw
                width = max(1.0, (b - a) / n * pw)
                p.fillRect(QtCore.QRectF(x, 6, width, h - 12), brush)
        p.setPen(_qcolor(t.ink3))
        p.setFont(make_font(self.fonts, 10))
        p.drawText(QtCore.QRectF(0, 0, left - 6, h),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "ON")
        p.end()


class MarkHistogram(_Painted):
    """Histogram of recent mark lengths; two clusters mean dits and dahs are separable."""

    BINS = 16

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(theme, fonts, height=42, parent=parent)
        self.marks: list[float] = []
        self.dit_ms = 150.0

    def set_marks(self, marks: Sequence[float], dit_ms: float) -> None:
        self.marks = [float(m) for m in marks]
        self.dit_ms = float(dit_ms)
        self.update()

    @property
    def max_ms(self) -> float:
        return max(600.0, self.dit_ms * 5.0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        p = self._painter()
        w, h = self.width(), self.height()
        bins = self.BINS
        max_ms = self.max_ms
        counts = [0] * bins
        for m in self.marks:
            counts[min(bins - 1, max(0, int(m / max_ms * bins)))] += 1
        mx = max(1, *counts)
        bw = w / bins
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QtCore.QRectF(0, h - 1, w, 1), _qcolor(t.grid))
        p.setBrush(_qcolor(t.trace))
        for i, c in enumerate(counts):
            if not c:
                continue
            bh = c / mx * (h - 12)
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(i * bw + 1, h - 1 - bh, bw - 2, bh + 2), 2, 2)
            p.drawPath(path)
        p.setPen(_qcolor(t.ink3))
        p.setFont(make_font(self.fonts, 9, mono=True))
        p.drawText(QtCore.QRectF(0, 0, w, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "0")
        p.drawText(QtCore.QRectF(0, 0, w, 12), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                   f"{round(max_ms)} ms")
        p.end()


def layout_guide_labels(
    letters: Sequence[tuple[str, float, float]],
    x_of: Callable[[float], float],
    width_of: Callable[[str], float],
    pad: float = 3.0,
) -> list[tuple[str, float]]:
    """Place the keying guide's letter labels so none overlap.

    Every letter gets a label centred on its span when there is room; a label
    that would collide with the one placed before it is skipped. Narrow letters
    (a lone dit such as E) are placed like any other, since neighbouring
    letters sit at least a letter gap away. Pure function, mirrored by
    ``layoutGuideLabels`` in ``web/js/player.js``. Returns ``(char, centre_x)``
    pairs in message order.
    """
    placed: list[tuple[str, float]] = []
    last_right = -math.inf
    for ch, start, end in letters:
        cx = x_of((start + end) / 2.0)
        half = width_of(ch) / 2.0
        if cx - half < last_right + pad:
            continue
        placed.append((ch, cx))
        last_right = cx + half
    return placed


class KeyingGuide(_Painted):
    """Marks and gaps of the encoded message drawn to scale, letters labelled above."""

    TOP = 16

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(theme, fonts, height=46, parent=parent)
        self.encoding: Encoding | None = None
        self.playhead_ms: float | None = None

    def set_encoding(self, encoding: Encoding | None) -> None:
        self.encoding = encoding
        self.update()

    def set_playhead(self, ms: float | None) -> None:
        """Position of the playhead in ms from the start, or ``None`` to hide it."""
        self.playhead_ms = ms
        self.update()

    @property
    def mark_count(self) -> int:
        """Number of marks (ON segments) drawn."""
        return self.encoding.mark_count if self.encoding else 0

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self.theme
        enc = self.encoding
        p = self._painter()
        w, h = self.width(), self.height()
        if enc is None or enc.total_ms <= 0:
            p.end()
            return
        top = self.TOP
        bh = h - top - 6
        scale = w / enc.total_ms

        def x(ms: float) -> float:
            return ms * scale

        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QtCore.QRectF(0, top, w, bh), _qcolor(t.off))
        cursor = 0.0
        on_brush = _qcolor(t.on)
        for on, ms in enc.timing:
            if on:
                p.fillRect(QtCore.QRectF(x(cursor), top, max(1.0, x(ms) - 1), bh), on_brush)
            cursor += ms
        p.setPen(_qcolor(t.ink2))
        font = make_font(self.fonts, 11, mono=True)
        p.setFont(font)
        metrics = QtGui.QFontMetricsF(font)
        # Every letter is labelled, however narrow (a lone dit like E); only a
        # label that would overlap its predecessor is skipped.
        for ch, cx in layout_guide_labels(enc.letters, x, metrics.horizontalAdvance):
            half = metrics.horizontalAdvance(ch) / 2
            rect = QtCore.QRectF(cx - half - 1, 0, 2 * half + 2, top - 2)
            p.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, ch)
        if self.playhead_ms is not None and 0.0 <= self.playhead_ms <= enc.total_ms:
            p.setPen(Qt.PenStyle.NoPen)
            p.fillRect(QtCore.QRectF(x(self.playhead_ms) - 1, top - 4, 2, bh + 8), _qcolor(t.ink))
        p.end()


class Pill(QtWidgets.QLabel):
    """The detector state pill: amber ``ON`` or quiet ``OFF``."""

    def __init__(self, theme: Theme, fonts: Fonts, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setFont(make_font(fonts, 12, weight=QtGui.QFont.Weight.DemiBold, spacing_em=0.05))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.on = None
        self.set_on(False)

    def set_on(self, on: bool) -> None:
        if on == self.on:
            return
        self.on = on
        t = self.theme
        colour = t.on if on else t.ink2
        border = t.on if on else t.line
        dot = t.on if on else t.off
        self.setStyleSheet(
            f"QLabel {{ color: {colour}; border: 1px solid {border}; border-radius: 10px;"
            f" background: {t.panel}; padding: 1px 9px 1px 7px; }}"
        )
        self.setText(f'<span style="color:{dot}">●</span>&nbsp;{"ON" if on else "OFF"}')


class Readout(QtWidgets.QWidget):
    """A ``key ..... value unit`` row from the rail and the encode strip."""

    def __init__(self, theme: Theme, fonts: Fonts, key: str, value_px: int = 15,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.fonts = fonts
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.key = QtWidgets.QLabel(key)
        self.key.setFont(make_font(fonts, 12))
        self.key.setStyleSheet(f"color: {theme.ink3};")
        self.value = QtWidgets.QLabel("—")
        self.value.setFont(make_font(fonts, value_px, mono=True))
        self.value.setStyleSheet(f"color: {theme.ink};")
        self.value.setTextFormat(Qt.TextFormat.RichText)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.key)
        lay.addStretch(1)
        lay.addWidget(self.value)
        self._last: tuple[str, str] | None = None

    def set_widget(self, widget: QtWidgets.QWidget) -> None:
        """Replace the value label with ``widget`` (used for the detector pill)."""
        lay = self.layout()
        lay.removeWidget(self.value)
        self.value.hide()
        lay.addWidget(widget)

    def set_value(self, text: str, unit: str = "") -> None:
        if (text, unit) == self._last:
            return
        self._last = (text, unit)
        small = (f'<span style="font-size:11px;color:{self.theme.ink3}">&nbsp;{html.escape(unit)}</span>'
                 if unit else "")
        self.value.setText(html.escape(text) + small)


# --------------------------------------------------------------- main window


class MainWindow(QtWidgets.QMainWindow):
    """The Beeper Morse Console window.

    Construct it through :func:`make_window`; call :meth:`start` to open the
    audio source and start the 30 Hz timer.  :meth:`on_tick` is the timer
    slot and may be called directly (that is how the offscreen test drives a
    :class:`WavReplay` to completion).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        source: BlockSource | None,
        theme: Theme | None = None,
        fonts: Fonts | None = None,
        *,
        manual_wpm: float | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.theme = theme or pick_theme()
        self.fonts = fonts or pick_fonts()
        self.pipeline = pipeline
        self._source: BlockSource | None = source
        self.replay = isinstance(source, WavReplay)
        self.replay_finished = False
        self.paused = False

        fs, block = pipeline.fs, pipeline.block_size
        history_blocks = int(round(HISTORY_S * fs / block))
        self._power = RingBuffer(history_blocks, fill=-100.0)
        self._thr_hi = RingBuffer(history_blocks, fill=-88.0)
        self._thr_lo = RingBuffer(history_blocks, fill=-94.0)
        self._on = RingBuffer(history_blocks, fill=0.0)
        self._raw = RingBuffer(int(SAVE_S * fs), fill=0.0)
        self._spec_smooth: np.ndarray | None = None
        self._spec_freqs: np.ndarray | None = None
        self._spec_mask: np.ndarray | None = None
        self._spec_top = SPECTRUM_DB_TOP_REST
        self._last: BlockResult | None = None
        self._filtered = np.zeros(0, dtype=np.float32)
        self._level_db = -120.0
        self._peak_db = -120.0
        self._peak_age = 0
        self._mark_history: deque[float] = deque(maxlen=HIST_MARKS)
        self._marks_seen = 0
        self._gaps_seen = 0
        self._auto_blocks_left = 0
        self._auto_sum: np.ndarray | None = None
        self._auto_count = 0
        self._auto_deadline: float | None = None
        self._error = ""
        self._text_shown: str | None = None
        self._player: TonePlayer | None = None
        self._play_started: float | None = None
        # Rendered tone being mixed into the pipeline's input while Play runs and
        # "Feed the decoder" is on (a software loopback, independent of speakers
        # and microphone); ``_inject_pos`` is the next sample to mix.
        self._inject: np.ndarray | None = None
        self._inject_pos: int = 0
        self._encoding: Encoding = build_encoding("", 8)

        self.setWindowTitle("Beeper Morse Console")
        self._build_ui(manual_wpm)
        self._apply_stylesheet()
        self.resize(1120, 900)
        self.setMinimumSize(900, 700)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(round(1000 / REFRESH_HZ)))
        self.timer.timeout.connect(self.on_tick)

        self._set_device_title()
        self._rebuild_encoding()
        self._refresh()

    # ------------------------------------------------------------ building

    def _font(self, px: int, mono: bool = False,
              weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal,
              spacing_em: float = 0.0) -> QtGui.QFont:
        return make_font(self.fonts, px, mono=mono, weight=weight, spacing_em=spacing_em)

    def _label(self, text: str, px: int = 13, colour: str | None = None, mono: bool = False,
               weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal, spacing_em: float = 0.0,
               upper: bool = False) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text.upper() if upper else text)
        lab.setFont(self._font(px, mono=mono, weight=weight, spacing_em=spacing_em))
        lab.setStyleSheet(f"color: {colour or self.theme.ink};")
        return lab

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        return self._label(text, 12, self.theme.ink2, spacing_em=0.04, upper=True)

    def _tag(self, letter: str) -> QtWidgets.QLabel:
        t = self.theme
        lab = QtWidgets.QLabel(letter)
        lab.setFixedSize(18, 18)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setFont(self._font(11, weight=QtGui.QFont.Weight.DemiBold))
        lab.setStyleSheet(f"QLabel {{ background: {t.ink}; color: {t.panel}; border-radius: 9px; }}")
        return lab

    def _header(self, title: str, note: str, tag: str,
                extra: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(14, 8, 14, 0)
        lay.setSpacing(8)
        lay.addWidget(self._label(title, 12, self.theme.ink2, weight=QtGui.QFont.Weight.Medium,
                                  spacing_em=0.06, upper=True))
        if extra is not None:
            lay.addWidget(extra)
        note_label = self._label(note, 12, self.theme.ink3)
        note_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        lay.addWidget(note_label, 1)
        lay.addWidget(self._tag(tag))
        return w

    def _spin(self, lo: int, hi: int, value: int, width: int = 88) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        spin.setFixedWidth(width)
        spin.setFont(self._font(13, mono=True))
        spin.setKeyboardTracking(False)
        return spin

    def _button(self, text: str, primary: bool = False) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setFont(self._font(13, weight=QtGui.QFont.Weight.Medium if primary else QtGui.QFont.Weight.Normal))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if primary:
            btn.setObjectName("primary")
        return btn

    def _build_ui(self, manual_wpm: float | None) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar(manual_wpm))
        root.addWidget(self._build_body(), 1)
        root.addWidget(self._build_encode())
        root.addWidget(self._build_statusbar())
        self.setCentralWidget(central)

    def _build_toolbar(self, manual_wpm: float | None) -> QtWidgets.QWidget:
        t = self.theme
        bar = QtWidgets.QFrame()
        bar.setObjectName("toolbar")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(14)

        # Input device
        lay.addWidget(self._field_label("Input"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.device_combo.setMaximumWidth(460)
        self.device_combo.setFont(self._font(13))
        self.device_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(28)
        self._populate_devices()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        lay.addWidget(self.device_combo)

        # Tone frequency
        lay.addSpacing(4)
        lay.addWidget(self._field_label("Tone"))
        self.freq_spin = self._spin(FREQ_MIN_HZ, FREQ_MAX_HZ, int(round(self.pipeline.f0)))
        self.freq_spin.valueChanged.connect(self._on_freq_changed)
        lay.addWidget(self.freq_spin)
        lay.addWidget(self._label("Hz", 13, t.ink3, mono=True))
        self.auto_button = self._button("Auto-detect")
        self.auto_button.setToolTip("Hold a beep for one second")
        self.auto_button.clicked.connect(self._on_auto_detect)
        lay.addWidget(self.auto_button)

        # Speed
        lay.addSpacing(4)
        lay.addWidget(self._field_label("Speed"))
        seg = QtWidgets.QFrame()
        seg.setObjectName("segFrame")
        seg_lay = QtWidgets.QHBoxLayout(seg)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        seg_lay.setSpacing(0)
        self.speed_auto = QtWidgets.QPushButton("Auto")
        self.speed_auto.setObjectName("segL")
        self.speed_manual = QtWidgets.QPushButton("Manual")
        self.speed_manual.setObjectName("segR")
        for b in (self.speed_auto, self.speed_manual):
            b.setCheckable(True)
            b.setFont(self._font(12))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            seg_lay.addWidget(b)
        self.speed_group = QtWidgets.QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_group.addButton(self.speed_auto)
        self.speed_group.addButton(self.speed_manual)
        lay.addWidget(seg)
        wpm0 = int(round(manual_wpm)) if manual_wpm else 8
        self.wpm_spin = self._spin(WPM_MIN, WPM_MAX, min(WPM_MAX, max(WPM_MIN, wpm0)))
        self.wpm_spin.valueChanged.connect(self._on_wpm_changed)
        lay.addWidget(self.wpm_spin)
        lay.addWidget(self._label("WPM", 13, t.ink3, mono=True))
        manual = manual_wpm is not None
        self.speed_manual.setChecked(manual)
        self.speed_auto.setChecked(not manual)
        self.wpm_spin.setEnabled(manual)
        self.speed_auto.clicked.connect(lambda: self._on_speed_mode(False))
        self.speed_manual.clicked.connect(lambda: self._on_speed_mode(True))

        lay.addStretch(1)
        self.pause_button = self._button("Pause")
        self.pause_button.clicked.connect(self._on_pause)
        lay.addWidget(self.pause_button)
        self.save_button = self._button("Save 30 s")
        self.save_button.setToolTip("Write the last 30 seconds to a WAV file")
        self.save_button.clicked.connect(self._on_save)
        lay.addWidget(self.save_button)
        self.clear_button = self._button("Clear text", primary=True)
        self.clear_button.clicked.connect(self._on_clear)
        lay.addWidget(self.clear_button)
        return bar

    def _build_body(self) -> QtWidgets.QWidget:
        body = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        main = QtWidgets.QFrame()
        main.setObjectName("main")
        main_lay = QtWidgets.QVBoxLayout(main)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        panels = (self._build_spectrum_panel(), self._build_power_panel(), self._build_decoded_panel())
        for panel, height in zip(panels, MAIN_ROW_HEIGHTS):
            panel.setMinimumHeight(height)
            main_lay.addWidget(panel, height)
        panels[-1].setObjectName("panelLast")
        lay.addWidget(main, 1)
        lay.addWidget(self._build_rail())
        return body

    def _panel(self, header: QtWidgets.QWidget, content: QtWidgets.QWidget) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(header)
        lay.addWidget(content, 1)
        return panel

    def _style_plot(self, plot: pg.PlotWidget, margins: tuple[int, int, int, int]) -> None:
        t = self.theme
        plot.setBackground(t.panel)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setAntialiasing(True)
        vb = plot.getViewBox()
        vb.setDefaultPadding(0.0)
        vb.setBorder(None)
        item = plot.getPlotItem()
        item.layout.setContentsMargins(*margins)
        tick_font = self._font(10, mono=True)
        for name in ("left", "bottom"):
            ax = plot.getAxis(name)
            ax.setPen(pg.mkPen(t.grid))
            ax.setTextPen(pg.mkPen(t.ink3))
            ax.setTickFont(tick_font)
            ax.setStyle(tickLength=0)
        plot.getAxis("left").setWidth(40)
        plot.showGrid(x=True, y=True, alpha=1.0)

    def _build_spectrum_panel(self) -> QtWidgets.QFrame:
        t = self.theme
        header = self._header("Spectrum", "0–8 kHz · 2048-point FFT · 43 ms", "A")
        self.spectrum_plot = pg.PlotWidget()
        self._style_plot(self.spectrum_plot, (0, 10, 12, 0))
        plot = self.spectrum_plot
        plot.setXRange(0.0, SPECTRUM_FMAX_HZ, padding=0)
        plot.setYRange(SPECTRUM_DB_MIN, SPECTRUM_DB_TOP_REST, padding=0)
        plot.getAxis("bottom").setTicks([[(0.0, "0")] + [(f, f"{f // 1000}k") for f in range(1000, 8001, 1000)]])
        plot.getAxis("left").setTicks([[(db, str(db)) for db in range(-110, 11, 20)]])
        self.spectrum_curve = pg.PlotCurveItem(
            pen=pg.mkPen(t.spec, width=1.5), brush=pg.mkBrush(t.spec_fill), fillLevel=SPECTRUM_DB_MIN)
        plot.addItem(self.spectrum_curve)
        f0 = self.pipeline.f0
        self.f0_marker = pg.InfiniteLine(
            pos=f0, angle=90, pen=pg.mkPen(t.trace, width=1.5), movable=False,
            label=f" {f0:.0f} Hz", labelOpts={"position": 0.97, "color": t.trace, "anchors": [(0, 0), (1, 0)],
                                              "movable": False})
        harm_pen = pg.mkPen(_qcolor(t.trace), width=1, style=Qt.PenStyle.DashLine)
        harm_colour = _qcolor(t.trace)
        harm_colour.setAlphaF(0.55)
        harm_pen.setColor(harm_colour)
        self.f3_marker = pg.InfiniteLine(
            pos=3 * f0, angle=90, pen=harm_pen, movable=False,
            label=" 3f", labelOpts={"position": 0.97, "color": t.trace, "anchors": [(0, 0), (1, 0)],
                                    "movable": False})
        for marker in (self.f0_marker, self.f3_marker):
            marker.label.setFont(self._font(11, mono=True))
            plot.addItem(marker)
        return self._panel(header, plot)

    def _build_power_panel(self) -> QtWidgets.QFrame:
        t = self.theme
        self.f0_header_label = self._label(f"{self.pipeline.f0:.0f} Hz", 12, t.ink2, mono=True)
        header = self._header("Tone power at", "· last 10 s · threshold with hysteresis", "B",
                              extra=self.f0_header_label)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.power_plot = pg.PlotWidget()
        self._style_plot(self.power_plot, (0, 8, 12, 0))
        plot = self.power_plot
        plot.setXRange(-HISTORY_S, 0.0, padding=0)
        plot.setYRange(-130.0, -70.0, padding=0)
        bottom = plot.getAxis("bottom")
        bottom.setTicks([[(-s, "") for s in range(0, int(HISTORY_S) + 1, 2)]])
        bottom.setStyle(showValues=False, tickLength=0)
        bottom.setHeight(4)
        n = self._power.size
        self._power_x = np.linspace(-HISTORY_S, 0.0, n)
        thr_pen = pg.mkPen(t.thr, width=1, style=Qt.PenStyle.DashLine)
        self.thr_hi_curve = pg.PlotCurveItem(pen=thr_pen)
        self.thr_lo_curve = pg.PlotCurveItem(pen=thr_pen)
        band_colour = _qcolor(t.thr)
        band_colour.setAlphaF(0.10)
        self.thr_band = pg.FillBetweenItem(self.thr_hi_curve, self.thr_lo_curve, brush=pg.mkBrush(band_colour))
        self.power_curve = pg.PlotCurveItem(pen=pg.mkPen(t.trace, width=1.5), brush=pg.mkBrush(t.trace_fill),
                                            fillLevel=-130.0)
        self.power_dot = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(t.trace))
        for item in (self.thr_band, self.thr_hi_curve, self.thr_lo_curve, self.power_curve, self.power_dot):
            plot.addItem(item)
        small = self._font(10, mono=True)
        self.power_label_left = pg.TextItem("−10 s", color=t.ink3, anchor=(0, 0))
        self.power_label_right = pg.TextItem("now", color=t.ink3, anchor=(1, 0))
        self.power_label_thr = pg.TextItem("threshold", color=t.thr, anchor=(0, 1))
        for label in (self.power_label_left, self.power_label_right, self.power_label_thr):
            label.setFont(small)
            plot.addItem(label)
        lay.addWidget(plot, 1)
        self.strip = OnOffStrip(self.theme, self.fonts)
        lay.addWidget(self.strip)
        return self._panel(header, content)

    def _build_decoded_panel(self) -> QtWidgets.QFrame:
        t = self.theme
        header = self._header("Decoded", "· letters appear at each gap, a long gap adds a space", "C")
        content = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        tape = QtWidgets.QWidget()
        tape_lay = QtWidgets.QVBoxLayout(tape)
        tape_lay.setContentsMargins(14, 6, 14, 12)
        tape_lay.setSpacing(6)
        live = QtWidgets.QHBoxLayout()
        live.setSpacing(10)
        live.addWidget(self._label("Current letter", 12, t.ink2))
        self.symbol_label = self._label(" ", 18, t.trace, mono=True, spacing_em=0.15)
        self.symbol_label.setTextFormat(Qt.TextFormat.RichText)
        self.symbol_label.setMinimumWidth(90)
        live.addWidget(self.symbol_label)
        self.symbol_hint = self._label("", 12, t.ink3)
        live.addWidget(self.symbol_hint)
        live.addStretch(1)
        tape_lay.addLayout(live)
        self.text_view = QtWidgets.QPlainTextEdit()
        self.text_view.setObjectName("textOut")
        self.text_view.setReadOnly(True)
        self.text_view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.text_view.setFont(self._font(24, mono=True, spacing_em=0.06))
        self.text_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text_view.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.text_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                               | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.text_view.setCursorWidth(8)
        self.text_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tape_lay.addWidget(self.text_view, 1)
        lay.addWidget(tape, 1)

        timing = QtWidgets.QFrame()
        timing.setObjectName("timing")
        timing.setFixedWidth(TIMING_WIDTH)
        grid = QtWidgets.QGridLayout(timing)
        grid.setContentsMargins(14, 8, 14, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.timing_values: dict[str, QtWidgets.QLabel] = {}
        for row, key in enumerate(("Dit", "Dah", "Letter gap", "Reverb offset")):
            grid.addWidget(self._label(key, 12, t.ink3, mono=True), row, 0)
            value = self._label("— ms", 12, t.ink, mono=True)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(value, row, 1)
            self.timing_values[key] = value
        hist_label = self._label("Mark lengths, last 30", 12, t.ink3, mono=True)
        grid.addWidget(hist_label, 4, 0, 1, 2)
        self.histogram = MarkHistogram(self.theme, self.fonts)
        grid.addWidget(self.histogram, 5, 0, 1, 2)
        grid.setRowStretch(6, 1)
        lay.addWidget(timing)
        return self._panel(header, content)

    def _rail_block(self, last: bool = False) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        block = QtWidgets.QFrame()
        block.setObjectName("railBlockLast" if last else "railBlock")
        lay = QtWidgets.QVBoxLayout(block)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        return block, lay

    def _build_rail(self) -> QtWidgets.QWidget:
        t = self.theme
        rail = QtWidgets.QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(RAIL_WIDTH)
        lay = QtWidgets.QVBoxLayout(rail)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        block, blay = self._rail_block()
        blay.setSpacing(8)
        blay.addWidget(self._label("Input level", 12, t.ink2, weight=QtGui.QFont.Weight.Medium,
                                   spacing_em=0.06, upper=True))
        self.meter = LevelMeter(self.theme, self.fonts)
        blay.addWidget(self.meter)
        self.waveform = Waveform(self.theme, self.fonts)
        blay.addWidget(self.waveform)
        blay.addWidget(self._label("Band-passed input · 6 ms", 12, t.ink3, weight=QtGui.QFont.Weight.Medium,
                                   spacing_em=0.08, upper=True))
        lay.addWidget(block)

        block, blay = self._rail_block()
        self.pill = Pill(self.theme, self.fonts)
        detector = Readout(self.theme, self.fonts, "Detector")
        detector.set_widget(self.pill)
        blay.addWidget(detector)
        self.readouts: dict[str, Readout] = {}
        for key in ("Level", "Signal / floor", "Speed", "Letters", "Unknown"):
            ro = Readout(self.theme, self.fonts, key)
            self.readouts[key] = ro
            blay.addWidget(ro)
        lay.addWidget(block)

        block, blay = self._rail_block(last=True)
        note = QtWidgets.QLabel(
            f'Threshold and speed adapt on their own. Turn <b style="color:{t.ink2};font-weight:500">Manual</b> '
            f"on if the keying is very irregular.")
        note.setFont(self._font(12))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {t.ink3};")
        blay.addWidget(note)
        lay.addWidget(block)
        lay.addStretch(1)
        return rail

    def _build_encode(self) -> QtWidgets.QWidget:
        t = self.theme
        frame = QtWidgets.QFrame()
        frame.setObjectName("encode")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._header("Encode", "· type a message, read it as Morse, follow the keying guide or play it",
                                   "D"))
        body = QtWidgets.QWidget()
        body_lay = QtWidgets.QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(14, 8, 14, 12)
        left_lay.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.message_edit = QtWidgets.QLineEdit("HELLO WORLD")
        self.message_edit.setPlaceholderText("Type a message")
        self.message_edit.setFont(self._font(13, mono=True, spacing_em=0.04))
        self.message_edit.setMinimumWidth(240)
        self.message_edit.textChanged.connect(self._on_message_changed)
        row.addWidget(self.message_edit, 1)
        row.addWidget(self._field_label("Speed"))
        self.enc_wpm_spin = self._spin(WPM_MIN, WPM_MAX, 8)
        self.enc_wpm_spin.valueChanged.connect(self._on_message_changed)
        row.addWidget(self.enc_wpm_spin)
        row.addWidget(self._label("WPM", 13, t.ink3, mono=True))
        self.play_button = self._button("Play tone")
        self.play_button.clicked.connect(self._on_play)
        row.addWidget(self.play_button)
        self.copy_button = self._button("Copy")
        self.copy_button.clicked.connect(self._on_copy)
        row.addWidget(self.copy_button)
        self.feed_check = QtWidgets.QCheckBox("Feed the decoder")
        self.feed_check.setChecked(True)
        self.feed_check.setFont(self._font(12))
        self.feed_check.setToolTip("While listening, Play is also mixed straight into the decoder, "
                                   "independent of speakers, microphone and audio processing.")
        row.addWidget(self.feed_check)
        left_lay.addLayout(row)
        out_frame = QtWidgets.QFrame()
        out_frame.setObjectName("encOut")
        out_lay = QtWidgets.QVBoxLayout(out_frame)
        out_lay.setContentsMargins(10, 6, 10, 6)
        out_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.morse_output = QtWidgets.QLabel()
        self.morse_output.setFont(self._font(17, mono=True, spacing_em=0.12))
        self.morse_output.setTextFormat(Qt.TextFormat.RichText)
        self.morse_output.setWordWrap(True)
        self.morse_output.setMinimumHeight(26)
        self.morse_output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        out_lay.addWidget(self.morse_output)
        left_lay.addWidget(out_frame)
        self.keying_guide = KeyingGuide(self.theme, self.fonts)
        left_lay.addWidget(self.keying_guide)
        left_lay.addStretch(1)  # the output box stays content-sized; spare height goes below
        body_lay.addWidget(left, 1)

        right = QtWidgets.QFrame()
        right.setObjectName("encRight")
        right.setFixedWidth(RAIL_WIDTH)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(14, 12, 14, 12)
        right_lay.setSpacing(10)
        self.enc_readouts: dict[str, Readout] = {}
        for key in ("Duration", "Dit · dah", "Letter gap · word gap", "Tone"):
            ro = Readout(self.theme, self.fonts, key)
            self.enc_readouts[key] = ro
            right_lay.addWidget(ro)
        note = QtWidgets.QLabel("Play keys the tone through this machine's speakers, so the decoder above "
                                "can be tested without the desktop.")
        note.setFont(self._font(12))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {t.ink3};")
        right_lay.addWidget(note)
        right_lay.addStretch(1)
        body_lay.addWidget(right)
        lay.addWidget(body)
        return frame

    def _build_statusbar(self) -> QtWidgets.QWidget:
        t = self.theme
        bar = QtWidgets.QFrame()
        bar.setObjectName("statusbar")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(18)
        font = self._font(12, mono=True)
        self.status_state = QtWidgets.QLabel()
        self.status_state.setTextFormat(Qt.TextFormat.RichText)
        self.status_blocks = QtWidgets.QLabel(f"{self.pipeline.block_size}-sample blocks")
        self.status_dropped = QtWidgets.QLabel("0 dropped")
        self.status_clock = QtWidgets.QLabel("00:00")
        self.status_error = QtWidgets.QLabel("")
        self.status_error.setStyleSheet(f"color: {t.on};")
        for lab in (self.status_state, self.status_blocks, self.status_dropped, self.status_clock):
            lab.setFont(font)
            lab.setStyleSheet(f"color: {t.ink2};")
            lay.addWidget(lab)
        self.status_error.setFont(font)
        lay.addStretch(1)
        lay.addWidget(self.status_error)
        return bar

    def _apply_stylesheet(self) -> None:
        t = self.theme
        self.setStyleSheet(f"""
            QMainWindow, QWidget#central {{ background: {t.panel}; color: {t.ink}; }}
            QLabel {{ background: transparent; }}
            QFrame#toolbar {{ background: {t.panel2}; border: 0; border-bottom: 1px solid {t.line}; }}
            QFrame#main {{ border: 0; border-right: 1px solid {t.line}; }}
            QFrame#panel {{ border: 0; border-bottom: 1px solid {t.line}; }}
            QFrame#panelLast {{ border: 0; }}
            QFrame#timing {{ border: 0; border-left: 1px solid {t.line}; }}
            QFrame#rail {{ background: {t.panel2}; border: 0; }}
            QFrame#railBlock {{ border: 0; border-bottom: 1px solid {t.line}; }}
            QFrame#railBlockLast {{ border: 0; }}
            QFrame#encode {{ border: 0; border-top: 1px solid {t.line}; }}
            QFrame#encRight {{ background: {t.panel2}; border: 0; border-left: 1px solid {t.line}; }}
            QFrame#encOut {{ background: {t.panel2}; border: 1px solid {t.line}; border-radius: 4px; }}
            QFrame#statusbar {{ background: {t.chrome}; border: 0; border-top: 1px solid {t.line}; }}
            QPushButton {{ min-height: 26px; max-height: 26px; padding: 0 12px; border: 1px solid {t.line};
                           border-radius: 4px; background: {t.panel}; color: {t.ink}; }}
            QPushButton:hover {{ border-color: {t.ink3}; }}
            QPushButton:pressed {{ background: {t.panel2}; }}
            QPushButton:disabled {{ color: {t.ink3}; }}
            QPushButton#primary {{ background: {t.trace}; border-color: {t.trace}; color: #FFFFFF; }}
            QPushButton#primary:hover {{ background: {t.on}; }}
            QFrame#segFrame {{ border: 1px solid {t.line}; border-radius: 4px; background: {t.panel}; }}
            QPushButton#segL, QPushButton#segR {{ border: 0; border-radius: 0; background: {t.panel};
                                                  color: {t.ink2}; padding: 0 10px; }}
            QPushButton#segL {{ border-top-left-radius: 3px; border-bottom-left-radius: 3px; }}
            QPushButton#segR {{ border-top-right-radius: 3px; border-bottom-right-radius: 3px; }}
            QPushButton#segL:checked, QPushButton#segR:checked {{ background: {t.ink}; color: {t.panel}; }}
            QComboBox, QSpinBox, QLineEdit {{ min-height: 26px; max-height: 26px; padding: 0 8px;
                border: 1px solid {t.line}; border-radius: 4px; background: {t.panel}; color: {t.ink};
                selection-background-color: {t.spec}; selection-color: #FFFFFF; }}
            QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {t.focus}; }}
            QComboBox:disabled, QSpinBox:disabled, QLineEdit:disabled {{ color: {t.ink3}; }}
            QComboBox::drop-down {{ border: 0; width: 20px; }}
            QComboBox QAbstractItemView {{ background: {t.panel}; color: {t.ink}; border: 1px solid {t.line};
                selection-background-color: {t.panel2}; selection-color: {t.ink}; outline: 0; }}
            QPlainTextEdit#textOut {{ background: transparent; border: 0; color: {t.ink};
                selection-background-color: {t.spec}; selection-color: #FFFFFF; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {t.line}; border-radius: 4px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QToolTip {{ background: {t.ink}; color: {t.panel}; border: 0; padding: 4px 6px; }}
        """)

    # ------------------------------------------------------------- devices

    def _populate_devices(self) -> None:
        """Fill the device combo: the replay file in replay mode, else ``list_devices()``."""
        combo = self.device_combo
        combo.blockSignals(True)
        combo.clear()
        if self.replay or (self._source is not None and not hasattr(self._source, "running")):
            name = self._source.device_name if self._source is not None else "Replay"
            combo.addItem(name, -1)
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        try:
            from morse.audio_input import list_devices

            devices = list_devices()
        except Exception as exc:  # PortAudio unavailable or failing
            devices = []
            self.set_status_error(f"cannot query audio devices: {exc}")
        for dev in devices:
            raw = " (raw)" if "wdm-ks" in dev.hostapi.lower() else ""
            combo.addItem(f"{dev.name} — {dev.hostapi}{raw}", dev.index)
        if not devices:
            combo.addItem("No input devices", -1)
            combo.setEnabled(False)
        current = self._source.device_index if self._source is not None else -1
        pos = combo.findData(current)
        if pos >= 0:
            combo.setCurrentIndex(pos)
        combo.blockSignals(False)

    def _set_device_title(self) -> None:
        name = self._source.device_name if self._source is not None else "no input"
        self.setWindowTitle(f"Beeper Morse Console — {name} · {self.pipeline.fs / 1000:g} kHz")

    def _on_device_changed(self, position: int) -> None:
        index = self.device_combo.itemData(position)
        if index is None or int(index) < 0 or self.replay:
            return
        index = int(index)
        if self._source is not None and self._source.device_index == index:
            return
        self._switch_source(index)

    def _switch_source(self, device: int | str | None) -> None:
        from morse.audio_input import AudioInput

        old = self._source
        try:
            if old is not None:
                old.stop()
        except Exception as exc:
            self.set_status_error(f"stopping input failed: {exc}")
        self._source = None
        try:
            new = AudioInput(device=device, fs=self.pipeline.fs, block_size=self.pipeline.block_size)
            new.start()
        except Exception as exc:
            self.set_status_error(f"cannot open input: {exc}")
            self._set_device_title()
            return
        self._source = new
        try:
            self.pipeline.flush()
            self.pipeline.detector.reset()  # noise and signal levels belonged to the old device
        except Exception as exc:
            self.set_status_error(f"pipeline reset failed: {exc}")
        self._set_device_title()
        self.set_status_error("")

    # ---------------------------------------------------------- lifecycle

    def start(self, run_timer: bool = True) -> None:
        """Open the audio source (errors go to the status bar) and start the 30 Hz timer.

        With ``run_timer=False`` the source is started but the timer is not,
        so a caller can drive :meth:`on_tick` by hand (the offscreen test).
        """
        if self._source is not None:
            try:
                self._source.start()
            except Exception as exc:
                self.set_status_error(f"cannot start input: {exc}")
        if run_timer:
            self.timer.start()
        self._refresh_status()

    def stop(self) -> None:
        """Stop the timer, the audio source and any playback."""
        self.timer.stop()
        self._stop_play()
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.stop()
        super().closeEvent(event)

    # --------------------------------------------------------------- tick

    @QtCore.Slot()
    def _mix_injected(self, block: np.ndarray) -> np.ndarray:
        """Add the next slice of the playing tone to ``block`` while Feed the decoder is on.

        The slice advances with the input clock, so the injected tone keeps
        pace with real time whatever the output device does.  Once the tone
        is used up the injection ends on its own.
        """
        tone = self._inject
        if tone is None:
            return block
        start = self._inject_pos
        seg = tone[start:start + block.size]
        self._inject_pos = start + block.size
        if self._inject_pos >= tone.size:
            self._inject = None
        if seg.size == 0:
            return block
        mixed = np.array(block, dtype=np.float32, copy=True)
        mixed[:seg.size] += seg
        return mixed

    def on_tick(self) -> None:
        """Timer slot: drain the source, run the pipeline, redraw.  Never raises."""
        try:
            self._tick()
        except Exception as exc:  # anything from audio or DSP must not kill the window
            self.set_status_error(f"{type(exc).__name__}: {exc}")

    def _tick(self) -> None:
        blocks: list[np.ndarray] = []
        if self._source is not None:
            blocks = self._source.read_blocks()
        if self.paused:
            blocks = []  # discarded: the display stays frozen
        else:
            if blocks:
                tick_level = -math.inf
                for block in blocks:
                    tick_level = max(tick_level, self._process_block(block))
                self._level_db = tick_level
            if self.replay and not self.replay_finished and self._source is not None \
                    and getattr(self._source, "finished", False):
                self._finish_replay()
        self._advance_playhead()
        self._refresh(len(blocks))
        if (self._auto_blocks_left > 0 and self._auto_deadline is not None
                and time.monotonic() > self._auto_deadline):
            self._finish_auto_detect()  # the source went quiet: decide on what arrived

    def _process_block(self, block: np.ndarray) -> float:
        """Run one block through the pipeline and the history buffers; returns its level."""
        block = self._mix_injected(block)
        result = self.pipeline.process_block(block)
        self._last = result
        self._raw.extend(block)
        self._power.push(result.power_db)
        self._thr_hi.push(result.threshold_hi_db)
        self._thr_lo.push(result.threshold_lo_db)
        self._on.push(1.0 if result.on else 0.0)
        if result.filtered.size:
            self._filtered = result.filtered
        for run in result.runs:
            if run.on:
                self._marks_seen += 1
                if run.ms < self.pipeline.decoder.max_mark_ms:
                    self._mark_history.append(run.ms)
            else:
                self._gaps_seen += 1
        level = result.level_dbfs
        if level > self._peak_db:
            self._peak_db = level
            self._peak_age = 0
        else:
            self._peak_age += 1
            if self._peak_age > PEAK_HOLD_BLOCKS:
                self._peak_db -= PEAK_HOLD_DECAY_DB
        return level

    def _finish_replay(self) -> None:
        self.pipeline.flush()
        self.replay_finished = True
        try:
            if self._source is not None:
                self._source.stop()
        except Exception:
            pass

    # ------------------------------------------------------------ refresh

    def _refresh(self, blocks_this_tick: int = 0) -> None:
        self._refresh_spectrum(blocks_this_tick)
        self._refresh_power()
        self._refresh_decoded()
        self._refresh_rail()
        self._refresh_status()

    def _refresh_spectrum(self, blocks_this_tick: int) -> None:
        if self._raw.count < SPECTRUM_N or (blocks_this_tick == 0 and self._spec_smooth is not None):
            return
        frame = self._raw.last(SPECTRUM_N)
        freqs, power_db = spectrum(frame, self.pipeline.fs)
        if self._spec_freqs is None:
            self._spec_freqs = freqs
            self._spec_mask = freqs <= SPECTRUM_FMAX_HZ
        if self._auto_blocks_left > 0:
            self._auto_sum = power_db if self._auto_sum is None else self._auto_sum + power_db
            self._auto_count += 1
            self._auto_blocks_left -= max(1, blocks_this_tick)
            if self._auto_blocks_left <= 0:
                self._finish_auto_detect()
        if self._spec_smooth is None:
            self._spec_smooth = power_db.copy()
        else:
            self._spec_smooth = SPECTRUM_SMOOTHING * self._spec_smooth + (1.0 - SPECTRUM_SMOOTHING) * power_db
        shown = np.maximum(self._spec_smooth[self._spec_mask], SPECTRUM_DB_MIN)
        self.spectrum_curve.setData(self._spec_freqs[self._spec_mask], shown, fillLevel=SPECTRUM_DB_MIN)
        peak = float(shown.max()) if shown.size else SPECTRUM_DB_MIN
        target = max(SPECTRUM_DB_TOP_REST, math.ceil((peak + 6.0) / 10.0) * 10.0)
        if target > self._spec_top:
            self._spec_top = target
        elif target < self._spec_top:
            self._spec_top = max(target, self._spec_top - 0.5)
        self.spectrum_plot.setYRange(SPECTRUM_DB_MIN, self._spec_top, padding=0)

    def _refresh_power(self) -> None:
        power = self._power.view()
        hi = self._thr_hi.view()
        lo = self._thr_lo.view()
        finite = np.isfinite(power)
        data_min = float(min(power[finite].min() if finite.any() else -100.0, lo.min()))
        data_max = float(max(power[finite].max() if finite.any() else -100.0, hi.max()))
        y_lo = math.floor((data_min - 5.0) / 10.0) * 10.0
        y_hi = math.ceil((data_max + 5.0) / 10.0) * 10.0
        if y_hi - y_lo < 60.0:
            y_hi = y_lo + 60.0
        y_lo, y_hi = max(y_lo, -130.0), min(y_hi, 10.0)
        if y_hi - y_lo < 60.0:
            y_lo = y_hi - 60.0
        plot = self.power_plot
        plot.setYRange(y_lo, y_hi, padding=0)
        plot.getAxis("left").setTicks([[(db, str(int(db)) if db % 20 == 0 else "")
                                         for db in np.arange(y_lo, y_hi + 1, 10.0)]])
        x = self._power_x
        self.thr_hi_curve.setData(x, hi)
        self.thr_lo_curve.setData(x, lo)
        self.power_curve.setData(x, np.clip(power, y_lo, y_hi), fillLevel=y_lo)
        self.power_dot.setData([x[-1]], [float(np.clip(power[-1], y_lo, y_hi))])
        self.power_label_left.setPos(-HISTORY_S + 0.05, y_hi)
        self.power_label_right.setPos(-0.05, y_hi)
        self.power_label_thr.setPos(-HISTORY_S + 0.1, float(np.clip(hi[0], y_lo, y_hi)))
        rect = plot.getViewBox().sceneBoundingRect()
        self.strip.set_states(self._on.view(), rect.left(), rect.right())

    def _refresh_decoded(self) -> None:
        dec = self.pipeline.decoder
        det = self.pipeline.detector
        text = dec.text
        if text != self._text_shown:
            self._text_shown = text
            self.text_view.setPlainText(text)
            cursor = self.text_view.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.text_view.setTextCursor(cursor)
            self.text_view.ensureCursorVisible()
        # While the speed estimate settles the held-back runs are shown dimmed,
        # as they read under the current estimate; the final reading replaces them.
        pending = dec.pending_count > 0
        shown = (dec.provisional if pending else dec.buffer).replace(".", "·").replace("-", "−")
        current = det.current_run
        live = ""
        if current.on and current.blocks > 0 and not det.warming_up:
            live = "·" if (current.ms - dec.offset_ms) < 2.0 * dec.dit_ms else "−"
        t = self.theme
        html_text = html.escape(shown)
        if pending and html_text:
            html_text = f'<span style="color:{t.ink3}">{html_text}</span>'
        if live:
            html_text += f'<span style="color:{t.ink3}">{live}</span>'
        self.symbol_label.setText(html_text or " ")
        if pending:
            self.symbol_hint.setText("estimating speed…")
        elif dec.buffer:
            char = table.lookup(dec.buffer)
            self.symbol_hint.setText(f"→ {char if char else '?'}")
        else:
            self.symbol_hint.setText("")
        have = self._marks_seen >= 2 and self._gaps_seen >= 1
        values = {
            "Dit": f"{round(dec.dit_ms)} ms" if have else "— ms",
            "Dah": f"{round(3 * dec.dit_ms)} ms" if have else "— ms",
            "Letter gap": f"{round(3 * dec.dit_ms)} ms" if have else "— ms",
            "Reverb offset": f"{round(dec.offset_ms)} ms" if have else "— ms",
        }
        for key, value in values.items():
            self.timing_values[key].setText(value)
        self.histogram.set_marks(self._mark_history, dec.dit_ms)

    def _refresh_rail(self) -> None:
        dec = self.pipeline.decoder
        last = self._last
        self.meter.set_level(self._level_db, self._peak_db)
        n_wave = int(round(WAVE_MS * self.pipeline.fs / 1000.0))
        self.waveform.set_samples(self._filtered[-n_wave:] if self._filtered.size else self._filtered)
        self.pill.set_on(bool(last.on) if last else False)
        ro = self.readouts
        ro["Level"].set_value(f"{self._level_db:.0f}" if last else "—", "dBFS")
        snr = (last.peak_db - last.floor_db) if last else None
        ro["Signal / floor"].set_value(f"{snr:.0f}" if snr is not None and snr > 0.5 else "—", "dB")
        manual = self.speed_manual.isChecked()
        have = self._marks_seen >= 2 and self._gaps_seen >= 1
        if manual:
            ro["Speed"].set_value(f"{self.wpm_spin.value()}", "WPM")
        else:
            ro["Speed"].set_value(f"{dec.wpm:.1f}" if have else "—", "WPM")
            # The disabled speed spin follows the live estimate in Auto mode, so
            # a switch to Manual starts from the measured speed.
            if dec.timing_ready:
                live_wpm = int(round(min(WPM_MAX, max(WPM_MIN, dec.wpm))))
                if self.wpm_spin.value() != live_wpm:
                    self.wpm_spin.blockSignals(True)
                    self.wpm_spin.setValue(live_wpm)
                    self.wpm_spin.blockSignals(False)
        ro["Letters"].set_value(str(dec.letter_count))
        ro["Unknown"].set_value(str(dec.unknown_count))

    def _refresh_status(self) -> None:
        t = self.theme
        if self.paused:
            state, colour = "Paused", t.ink3
        elif self._source is None:
            state, colour = "Not listening", t.ink3
        elif self.replay:
            name = Path(getattr(self._source, "path", "file")).name
            if self.replay_finished:
                state, colour = f"Replay finished · {name}", t.good
            else:
                state, colour = f"Replaying {name}", t.good
        elif getattr(self._source, "running", True):
            state, colour = "Listening", t.good
        else:
            state, colour = "Input stopped", t.ink3
        self.status_state.setText(f'<span style="color:{colour}">●</span> {html.escape(state)}')
        dropped = int(getattr(self._source, "dropped", 0)) if self._source is not None else 0
        self.status_dropped.setText(f"{dropped} dropped")
        seconds = int(self.pipeline.elapsed_ms // 1000)
        self.status_clock.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
        self.status_error.setText(self._error)

    # ------------------------------------------------------------ controls

    def set_status_error(self, message: str) -> None:
        """Show ``message`` (or clear it with ``''``) at the right of the status bar.

        Safe to call while the window is still being built (for example when
        ``list_devices()`` fails during toolbar construction): the text is
        kept and shown once the status bar exists.
        """
        self._error = " ".join(str(message).split())
        label = getattr(self, "status_error", None)
        if label is not None:
            label.setText(self._error)

    def set_frequency(self, f0: float) -> None:
        """Retune the pipeline and every f0 readout; invalid values go to the status bar."""
        try:
            self.pipeline.set_frequency(float(f0))
        except ValueError as exc:
            self.set_status_error(str(exc))
            return
        f0 = self.pipeline.f0
        self.f0_header_label.setText(f"{f0:.0f} Hz")
        self.f0_marker.setValue(f0)
        self.f0_marker.label.setFormat(f" {f0:.0f} Hz")
        self.f3_marker.setValue(3 * f0)
        self.f3_marker.setVisible(3 * f0 <= SPECTRUM_FMAX_HZ)
        self.enc_readouts["Tone"].set_value(f"{f0:.0f}", "Hz")
        if self.freq_spin.value() != int(round(f0)):
            self.freq_spin.blockSignals(True)
            self.freq_spin.setValue(int(round(f0)))
            self.freq_spin.blockSignals(False)
        self._stop_play()

    def _on_freq_changed(self, value: int) -> None:
        self.set_frequency(float(value))

    def _on_auto_detect(self) -> None:
        """Average the spectrum over the next second, then snap f0 to the strongest peak."""
        if self._auto_blocks_left > 0:
            return
        if self._source is None:
            self.set_status_error("Auto-detect: no input is open")
            return
        self._auto_blocks_left = AUTO_DETECT_BLOCKS
        self._auto_sum = None
        self._auto_count = 0
        self._auto_deadline = time.monotonic() + AUTO_DETECT_TIMEOUT_S
        self.auto_button.setText("Listening…")
        self.auto_button.setEnabled(False)

    def _finish_auto_detect(self) -> None:
        self.auto_button.setText("Auto-detect")
        self.auto_button.setEnabled(True)
        self._auto_blocks_left = 0
        self._auto_deadline = None
        if self._auto_sum is None or self._auto_count == 0 or self._spec_freqs is None:
            self.set_status_error("Auto-detect: no audio arrived")
            return
        avg = self._auto_sum / self._auto_count
        self._auto_sum = None
        peak_hz, prominence = find_tone_frequency(self._spec_freqs, avg, AUTO_DETECT_FMIN_HZ, SPECTRUM_FMAX_HZ)
        if peak_hz <= 0 or prominence < AUTO_DETECT_MIN_PROMINENCE_DB:
            self.set_status_error(f"Auto-detect: no clear tone (best peak {peak_hz:.0f} Hz, {prominence:.0f} dB)")
            return
        self.set_status_error("")
        self.freq_spin.setValue(int(round(peak_hz)))  # triggers set_frequency

    def _on_speed_mode(self, manual: bool) -> None:
        self.wpm_spin.setEnabled(manual)
        self._replace_decoder(wpm=float(self.wpm_spin.value()) if manual else None, adaptive=not manual)

    def _on_wpm_changed(self, value: int) -> None:
        if self.speed_manual.isChecked():
            self._replace_decoder(wpm=float(value), adaptive=False)

    def _replace_decoder(self, wpm: float | None, adaptive: bool) -> None:
        """Swap the decoder for one with the new speed settings.

        Text, the pending letter and the counters carry over, and so does the
        timing history (``MorseDecoder.adopt_timing``), so Manual mode gets
        its reverb offset and a return to Auto gets its speed estimate
        without re-learning.
        """
        old = self.pipeline.decoder
        new = MorseDecoder(wpm=wpm, adaptive=adaptive)
        new.text = old.text
        new.buffer = old.buffer
        new.letter_count = old.letter_count
        new.unknown_count = old.unknown_count
        new.adopt_timing(old)  # measured speed and reverb offset carry over (same as the web app)
        self.pipeline.decoder = new

    def _on_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
        self._refresh_status()

    def _on_save(self) -> None:
        try:
            path = self.save_recent_wav()
        except Exception as exc:
            self.set_status_error(f"Save failed: {exc}")
            return
        self.save_button.setText(f"Saved {path.name}")
        self.save_button.setEnabled(False)
        QtCore.QTimer.singleShot(1800, self._reset_save_button)

    def _reset_save_button(self) -> None:
        self.save_button.setText("Save 30 s")
        self.save_button.setEnabled(True)

    def save_recent_wav(self, path: str | Path | None = None) -> Path:
        """Write the last 30 s of raw input as 16-bit PCM; returns the path written.

        Without ``path`` the file is ``capture_NNN.wav`` in the current
        directory, the first ``NNN`` that does not exist yet.
        """
        from scipy.io import wavfile

        if path is None:
            n = 1
            while True:
                candidate = Path.cwd() / f"capture_{n:03d}.wav"
                if not candidate.exists():
                    break
                n += 1
            path = candidate
        path = Path(path)
        samples = self._raw.last(min(self._raw.count, self._raw.size))
        if samples.size == 0:
            raise ValueError("nothing recorded yet")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(str(path), self.pipeline.fs, pcm)
        return path

    def _on_clear(self) -> None:
        self.pipeline.decoder.reset(keep_timing=True)
        self._refresh_decoded()
        self._refresh_rail()

    # ------------------------------------------------------------- encoder

    def _on_message_changed(self, *_: object) -> None:
        self._stop_play()
        self._rebuild_encoding()

    def _rebuild_encoding(self) -> None:
        t = self.theme
        try:
            enc = build_encoding(self.message_edit.text(), float(self.enc_wpm_spin.value()))
        except ValueError as exc:
            self.set_status_error(str(exc))
            return
        self._encoding = enc
        if enc.morse:
            parts = [html.escape(p) for p in enc.morse.split(table.WORD_SEP)]
            self.morse_output.setText(f' <span style="color:{t.ink3}">/</span> '.join(parts))
        else:
            self.morse_output.setText(f'<span style="color:{t.ink3}">Type a message above</span>')
        self.keying_guide.set_encoding(enc)
        dit = enc.dit_ms
        ro = self.enc_readouts
        ro["Duration"].set_value(f"{enc.total_ms / 1000:.1f}", "s")
        ro["Dit · dah"].set_value(f"{round(dit)} · {round(3 * dit)} ms")
        ro["Letter gap · word gap"].set_value(f"{round(3 * dit)} · {round(7 * dit)} ms")
        ro["Tone"].set_value(f"{self.pipeline.f0:.0f}", "Hz")

    def morse_output_text(self) -> str:
        """The encode strip's Morse output as plain text (``morse.table.encode`` of the message)."""
        return self._encoding.morse

    def decoded_text(self) -> str:
        """Everything decoded so far, as shown in the decoded-text widget."""
        return self.text_view.toPlainText()

    def _on_play(self) -> None:
        if self._play_started is not None:
            self._stop_play()
            return
        enc = self._encoding
        if not enc.timing:
            return
        try:
            samples = render_tone(enc.timing, self.pipeline.f0, fs=DEFAULT_FS)
            if self._player is None:
                self._player = TonePlayer(fs=DEFAULT_FS)
            self._player.play(samples)
        except Exception as exc:  # no output device, PortAudio error
            self.set_status_error(f"Play failed: {exc}")
            return
        self.set_status_error("")
        self._play_started = time.monotonic()
        if self.feed_check.isChecked() and self._source is not None:
            self._inject = np.asarray(samples, dtype=np.float32)
            self._inject_pos = 0
        else:
            self._inject = None
        self.play_button.setText("Stop")
        self.keying_guide.set_playhead(0.0)

    def _stop_play(self) -> None:
        if self._play_started is None:
            return
        self._play_started = None
        self._inject = None
        try:
            if self._player is not None:
                self._player.stop()
        except Exception as exc:
            self.set_status_error(f"Stop failed: {exc}")
        self.play_button.setText("Play tone")
        self.keying_guide.set_playhead(None)

    def _advance_playhead(self) -> None:
        if self._play_started is None:
            return
        elapsed_ms = (time.monotonic() - self._play_started) * 1000.0 - PLAY_LATENCY_MS
        if elapsed_ms >= self._encoding.total_ms + 50.0:
            self._stop_play()
        else:
            self.keying_guide.set_playhead(max(0.0, elapsed_ms))

    def _on_copy(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._encoding.morse)
        self.copy_button.setText("Copied")
        QtCore.QTimer.singleShot(1500, lambda: self.copy_button.setText("Copy"))


# ------------------------------------------------------------------ factory


def ensure_app(argv: Sequence[str] | None = None) -> QtWidgets.QApplication:
    """Return the running ``QApplication``, creating one (Fusion style) if needed."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(list(argv) if argv is not None else sys.argv[:1])
        app.setApplicationName("Beeper Morse Console")
        app.setStyle("Fusion")
    pg.setConfigOptions(antialias=True)
    return app


def make_window(args: Any = None, source: BlockSource | None = None) -> MainWindow:
    """Build the main window from parsed command-line ``args``.

    ``args`` is the namespace from ``morse.app.build_parser`` (``wav``,
    ``device``, ``freq``, ``wpm``); missing attributes take their defaults.
    With ``wav`` the input is a :class:`WavReplay` of that file at the file's
    own sample rate; otherwise ``source`` is used, or an
    :class:`morse.audio_input.AudioInput` on ``device`` is created (a failure
    to open it is shown in the status bar, not raised).  ``wpm`` selects
    Manual speed at that value.  The window is returned unshown and not yet
    started; call ``show()`` and :meth:`MainWindow.start`.
    """
    ensure_app()
    wav = getattr(args, "wav", None)
    device = getattr(args, "device", None)
    freq = float(getattr(args, "freq", None) or DEFAULT_FREQ_HZ)
    wpm = getattr(args, "wpm", None)
    error = ""
    if wav:
        source = WavReplay(wav)
        fs, block = source.fs, source.block_size
    elif source is not None:
        fs, block = int(source.fs), int(source.block_size)
    else:
        fs, block = DEFAULT_FS, DEFAULT_BLOCK_SIZE
        try:
            from morse.audio_input import AudioInput

            source = AudioInput(device=device, fs=fs, block_size=block)
        except Exception as exc:  # bad device spec, PortAudio missing, ...
            source = None
            error = f"cannot open input: {exc}"
    try:
        pipeline = Pipeline(fs=fs, block_size=block, f0=freq, wpm=wpm, adaptive=wpm is None)
    except ValueError as exc:
        error = f"{exc}; using {DEFAULT_FREQ_HZ:g} Hz"
        pipeline = Pipeline(fs=fs, block_size=block, f0=DEFAULT_FREQ_HZ, wpm=wpm, adaptive=wpm is None)
    window = MainWindow(pipeline, source, manual_wpm=wpm)
    if error:
        window.set_status_error(error)
    return window


def run_ui(args: Any = None) -> int:
    """Show the console for ``args`` (see :func:`make_window`) and run until it is closed."""
    app = ensure_app()
    window = make_window(args)
    window.show()
    window.start()
    return int(app.exec())
