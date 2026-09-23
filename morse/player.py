"""Key text as a Morse tone and play it through the speakers.

Three pieces, used by the encode strip of the desktop UI (``morse/ui.py``):

* :func:`build_timing` turns text into the ``[(on, ms), ...]`` keying
  sequence with exactly the rule of ``tools/beep_sender.py``: dit
  ``1200 / wpm`` ms, dah and letter gap 3 dits, intra-letter gap 1 dit, word
  gap 7 dits, every duration rounded to whole milliseconds. What the laptop
  plays is therefore sample-for-sample the timing the desktop beeper sends,
  so the decoder is exercised on the same signal either way.
* :func:`render_tone` renders that sequence as a ``float32`` sine keyed with
  short linear ramps at every edge so the speakers do not click. The ramps
  sit inside each mark, so gaps stay digitally silent and the length of the
  result is exactly the total duration of the sequence.
* :class:`TonePlayer` plays the samples through ``sounddevice.play`` without
  blocking, so the UI can run a playhead along the keying guide and offer a
  Stop button.

``sounddevice`` is imported lazily, inside the methods that need it, so this
module can be imported (and :func:`build_timing` and :func:`render_tone` can
be tested) without PortAudio, and tests can substitute a fake module. Playing
through the speakers while the microphone is open is fine: output and input
are separate PortAudio streams.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from morse.table import MORSE_TABLE

__all__ = ["MIC_GATE_GAIN", "MIC_GATE_TAIL_MS", "MicGate", "Timing", "TonePlayer", "build_timing",
           "farnsworth_gaps", "render_tone"]

Timing = list[tuple[bool, float]]
"""A keying sequence: ``(on, duration_ms)`` pairs, marks ``True``, gaps ``False``."""


# ----------------------------------------------------------------- timing


def _dit_ms(wpm: float) -> float:
    """Dit length in ms for ``wpm`` words per minute (PARIS standard)."""
    value = float(wpm)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"wpm must be positive, got {wpm!r}")
    return 1200.0 / value


def farnsworth_gaps(wpm: float, farnsworth_wpm: float | None) -> tuple[float, float]:
    """Letter and word gap in whole ms for characters at ``wpm`` and an overall ``farnsworth_wpm``.

    Standard spacing (``3`` and ``7`` dits) when ``farnsworth_wpm`` is None
    or not below ``wpm``. Otherwise the ARRL rule: with character speed
    ``c`` and overall speed ``s``, the extra time per 50-unit word is
    ``ta = (60 c - 37.2 s) / (s c)`` seconds, and the gaps become
    ``3 ta / 19`` (letter) and ``7 ta / 19`` (word). At 18/5 that is a
    1568 ms letter gap and a 3660 ms word gap around 67 ms elements, which
    is how Morse is taught by ear: characters at full speed, room between.
    """
    dit = _dit_ms(wpm)
    if farnsworth_wpm is None:
        return float(round(3 * dit)), float(round(7 * dit))
    s = float(farnsworth_wpm)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(f"farnsworth_wpm must be positive, got {farnsworth_wpm!r}")
    c = float(wpm)
    if s >= c:
        return float(round(3 * dit)), float(round(7 * dit))
    ta = (60.0 * c - 37.2 * s) / (s * c)
    return float(round(1000.0 * 3.0 * ta / 19.0)), float(round(1000.0 * 7.0 * ta / 19.0))


def build_timing(text: str, wpm: float, farnsworth_wpm: float | None = None) -> Timing:
    """Return the keying sequence for ``text`` as ``[(on, ms), ...]``.

    Same rule as ``build_timing`` in ``tools/beep_sender.py``: case-insensitive;
    characters without a Morse code are skipped (a word left empty by that
    is skipped too); any whitespace run separates words. Durations are
    ``round(k * 1200 / wpm)`` ms for ``k`` in 1 (dit, intra-letter gap),
    3 (dah, letter gap) and 7 (word gap), returned as whole-millisecond
    floats, so at 10 WPM a dit is 120 ms, a dah 360 ms and a word gap 840 ms.
    With ``farnsworth_wpm`` below ``wpm`` the letter and word gaps are
    stretched by :func:`farnsworth_gaps` while the elements keep their speed.
    The sequence starts with the first mark and ends with the last one: no
    leading or trailing gap, never two gaps in a row, and ``[]`` when
    nothing in ``text`` is encodable. ``wpm`` must be positive.
    """
    dit = _dit_ms(wpm)
    dit_ms = float(round(dit))
    dah_ms = float(round(3 * dit))
    letter_gap_ms, word_gap_ms = farnsworth_gaps(wpm, farnsworth_wpm)

    words: list[list[str]] = []
    for word in text.upper().split():
        codes = [MORSE_TABLE[ch] for ch in word if ch in MORSE_TABLE]
        if codes:
            words.append(codes)

    seq: Timing = []
    for wi, codes in enumerate(words):
        if wi > 0:
            seq.append((False, word_gap_ms))
        for li, code in enumerate(codes):
            if li > 0:
                seq.append((False, letter_gap_ms))
            for si, symbol in enumerate(code):
                if si > 0:
                    seq.append((False, dit_ms))
                seq.append((True, dah_ms if symbol == "-" else dit_ms))
    return seq


# ---------------------------------------------------------------- rendering


def render_tone(
    timing: Iterable[tuple[bool, float]],
    f0: float,
    fs: int = 48000,
    amplitude: float = 0.3,
    ramp_ms: float = 3.0,
) -> np.ndarray:
    """Render a keying sequence as a ``float32`` sine at ``f0`` Hz.

    Segment boundaries are placed at ``round(cumulative_ms * fs / 1000)``
    samples, so rounding never accumulates and the result has exactly
    ``round(total_ms * fs / 1000)`` samples. During a mark the signal is
    ``amplitude * sin(2 pi f0 t)`` with ``t`` measured from the start of the
    sequence (the phase is continuous across gaps); during a gap it is
    exactly zero.

    Each mark begins with a linear ramp up and ends with a linear ramp down,
    both ``ramp_ms`` long and both *inside* the mark, so the mark keeps its
    duration and the gaps stay silent. A ramp of ``r`` samples takes the
    envelope through ``1/(r+1), 2/(r+1), ..., r/(r+1)``: the step from the
    silence before it is the same size as every step within it. A mark
    shorter than two ramps gets ramps of half its length instead;
    ``ramp_ms = 0`` keys the sine hard.

    Raises ``ValueError`` for a non-positive ``fs``, an ``f0`` outside
    ``0 < f0 < fs/2``, an ``amplitude`` outside ``0..1``, a negative
    ``ramp_ms`` or a negative (or non-finite) segment duration. An empty
    sequence renders as an empty array.
    """
    fs_i = int(fs)
    if fs_i <= 0:
        raise ValueError(f"fs must be positive, got {fs!r}")
    f0_v = float(f0)
    if not (math.isfinite(f0_v) and 0.0 < f0_v < fs_i / 2.0):
        raise ValueError(f"f0 must satisfy 0 < f0 < fs/2 = {fs_i / 2:g} Hz, got {f0!r}")
    amp = float(amplitude)
    if not (math.isfinite(amp) and 0.0 <= amp <= 1.0):
        raise ValueError(f"amplitude must be in 0..1, got {amplitude!r}")
    ramp = float(ramp_ms)
    if not (math.isfinite(ramp) and ramp >= 0.0):
        raise ValueError(f"ramp_ms must be non-negative, got {ramp_ms!r}")

    ons: list[bool] = []
    durations: list[float] = []
    for on, ms in timing:
        ms_v = float(ms)
        if not (math.isfinite(ms_v) and ms_v >= 0.0):
            raise ValueError(f"segment duration must be a non-negative number of ms, got {ms!r}")
        ons.append(bool(on))
        durations.append(ms_v)

    # Boundaries from the cumulative time so per-segment rounding cannot drift.
    edges = np.rint(np.cumsum([0.0, *durations]) * fs_i / 1000.0).astype(np.int64)
    n = int(edges[-1])
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    env = np.zeros(n)
    ramp_samples = int(round(ramp * fs_i / 1000.0))
    for on, start, stop in zip(ons, edges[:-1], edges[1:]):
        if not on or stop <= start:
            continue
        env[start:stop] = 1.0
        r = min(ramp_samples, int(stop - start) // 2)
        if r > 0:
            slope = np.arange(1, r + 1) / (r + 1.0)
            env[start:start + r] = slope
            env[stop - r:stop] = slope[::-1]

    t = np.arange(n) / float(fs_i)
    signal = amp * env * np.sin(2.0 * np.pi * f0_v * t)
    return signal.astype(np.float32)


# ----------------------------------------------------------------- mic gate

MIC_GATE_GAIN = 0.01
"""Microphone gain (-40 dB) in the decoder's input while the app's own sound is fed to the decoder."""
MIC_GATE_TAIL_MS = 400.0
"""How long the microphone stays turned down after the app's sound stops, for the acoustic echo to die."""


class MicGate:
    """Decides when the microphone is turned down in the decoder's input.

    With "Feed the decoder" on, the app's own sound (Play, the hand key) is
    mixed straight into the decoder. A raw microphone hears the same sound
    from the speakers 30 to 150 ms later, and the two copies together fill the
    gaps between marks. So while the app is sounding, and for
    :data:`MIC_GATE_TAIL_MS` afterwards, the microphone is attenuated by
    :data:`MIC_GATE_GAIN` and the decoder effectively hears the feed alone. At
    every other moment the microphone passes untouched, so an external beeper,
    or a recording of the speakers played back later, decodes as usual. The
    same class exists in ``web/js/audio.js``.

    ``update(sounding, now_ms)`` is called once per block (or per event) with
    whether the app is sounding right now and the caller's clock; it returns
    whether the microphone should be attenuated for that moment.
    """

    def __init__(self, tail_ms: float = MIC_GATE_TAIL_MS) -> None:
        tail = float(tail_ms)
        if not (math.isfinite(tail) and tail >= 0.0):
            raise ValueError(f"tail_ms must be a non-negative number, got {tail_ms!r}")
        self.tail_ms: float = tail
        self._until: float | None = None

    def update(self, sounding: bool, now_ms: float) -> bool:
        """Record the state at ``now_ms``; True when the microphone should be attenuated."""
        now = float(now_ms)
        if sounding:
            self._until = now + self.tail_ms
            return True
        if self._until is not None and now < self._until:
            return True
        self._until = None
        return False

    @property
    def active(self) -> bool:
        """Whether the last :meth:`update` asked for attenuation."""
        return self._until is not None

    def reset(self) -> None:
        """Forget any tail in progress: the microphone passes at once."""
        self._until = None

    def __repr__(self) -> str:
        return f"MicGate(tail_ms={self.tail_ms:g}, active={self.active})"


# ----------------------------------------------------------------- playback


def _sd() -> Any:
    """Import sounddevice lazily so module import stays hardware-free."""
    import sounddevice  # deliberately local: loads PortAudio

    return sounddevice


class TonePlayer:
    """Non-blocking playback of rendered samples through ``sounddevice.play``.

    ``sounddevice.play`` keeps one global playback: starting a new one stops
    the previous one, which is exactly what a Play button wants. Nothing is
    imported or opened until :meth:`play` is called, so constructing a
    player, reading :attr:`playing` or calling :meth:`stop` on a player that
    never played does not load PortAudio.

    Attributes:
        fs: sample rate the samples are played at.
        device: output device passed to sounddevice: an index, a
            case-insensitive substring of the device name (sounddevice's own
            matching, ``'name, hostapi'`` allowed), or ``None`` for the
            system default output.
    """

    def __init__(self, fs: int = 48000, device: int | str | None = None) -> None:
        if int(fs) <= 0:
            raise ValueError(f"fs must be positive, got {fs!r}")
        self.fs: int = int(fs)
        self.device: int | str | None = device
        self._started: bool = False

    def play(self, samples: np.ndarray) -> None:
        """Start playing ``samples`` (mono ``float32`` in -1..1) and return at once.

        Any playback started earlier by this or another player is stopped
        first (``sounddevice.play`` does that itself). Other dtypes are cast
        to ``float32``; a ``(frames, channels)`` array is played as is. An
        empty array only stops what is playing. Errors from PortAudio (no
        output device, unsupported rate) propagate to the caller.
        """
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim not in (1, 2):  # checked before ascontiguousarray, which promotes 0-D to 1-D
            raise ValueError(f"samples must be 1-D (mono) or (frames, channels), got shape {data.shape}")
        data = np.ascontiguousarray(data)
        if data.shape[0] == 0:
            self.stop()
            return
        _sd().play(data, samplerate=self.fs, device=self.device)
        self._started = True

    def stop(self) -> None:
        """Stop playback. No-op (and no sounddevice import) if nothing was played."""
        if not self._started:
            return
        self._started = False
        _sd().stop()

    @property
    def playing(self) -> bool:
        """True while samples from :meth:`play` are still being played.

        Reads ``sounddevice.get_stream().active``; ``False`` before the first
        :meth:`play`, after :meth:`stop`, once the samples have run out, and
        whenever the query itself fails (no stream, or a stream that has
        already been closed).
        """
        if not self._started:
            return False
        try:
            return bool(_sd().get_stream().active)
        except Exception:  # RuntimeError (no stream yet) or a PortAudio error on a closed stream
            return False

    def __repr__(self) -> str:
        return f"TonePlayer(fs={self.fs}, device={self.device!r}, playing={self.playing})"
