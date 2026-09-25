"""A Morse key the operator works by hand: straight key or two-paddle keyer, and its tone.

Two pieces, kept apart so the timing logic can be tested without audio:

:class:`Keyer`
    Pure state machine.  Every method takes the current time in
    milliseconds and returns the tone transitions ``(t_ms, on)`` it decided
    on; some lie in the future (the end of a paddle element is known the
    moment it starts).  A bounded log of every transition lets callers render
    the envelope for any window of time (:func:`envelope`) or decode what was
    sent.  Mirrored line by line by ``web/js/keyer.js``.

    * ``straight`` mode: the tone follows the key.  ``key_down`` turns it on,
      ``key_up`` off; the operator supplies every duration.
    * ``paddle`` mode: an electronic keyer.  Pressing the dit paddle sends one
      dit of the configured length and keeps sending dits, one dit gap apart,
      while it is held; the dah paddle likewise with dahs.  Holding both
      alternates (iambic).  A paddle tapped during an element is remembered
      and sent after it, as real keyers do, so a quick "dit, dah" gives a
      clean A.  Releasing during an element never cuts it short.  Iambic
      ``A`` (default): letting go of both paddles ends the character with the
      element in progress.  Iambic ``B``: when both paddles were squeezed
      during an element and both are released, one more element of the
      opposite kind follows (the Curtis mode B habit: a squeeze released
      during the dah of an A gives an R).
    * ``bug`` mode: a semi-automatic key.  The dit paddle sends automatic
      dits while held; the dah paddle is a straight key, the tone sounding
      while it is held, so the operator makes every dah by hand.  Pressing
      the dah lever cuts a dit short; dits resume one space after it is let
      go if the dit paddle is still held.
    * Weighting applies to paddle and bug elements: ``dah_ratio`` is the dah
      length in dits (3 standard) and ``weight`` in percent (50 standard)
      lengthens every mark and shortens the space after it by the same
      amount, so the element period is unchanged: mark ``w T`` (dit) or
      ``(ratio + w - 1) T`` (dah) and space ``(2 - w) T`` with ``w =
      weight / 50``.

:class:`LiveKey`
    Plays the keyer's tone through a ``sounddevice`` output stream in real
    time.  The stream callback is the clock: it advances the keyer, renders
    the envelope with 3 ms ramps at every edge (no clicks) and writes the
    sine.  :meth:`LiveKey.feed_block` renders the same tone for an arbitrary
    time window so the desktop UI can mix it straight into the decoder's
    input ("Feed the decoder"), independent of speakers and microphone.
    ``sounddevice`` is imported lazily, inside :meth:`LiveKey.start`.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Literal

import numpy as np

__all__ = ["DAH_RATIO_RANGE", "Iambic", "Keyer", "LiveKey", "Mode", "Paddle", "WEIGHT_RANGE", "envelope"]

Mode = Literal["straight", "paddle", "bug"]
Paddle = Literal["dit", "dah"]
Iambic = Literal["A", "B"]

_MODES = ("straight", "paddle", "bug")
_PADDLES = ("dit", "dah")
_IAMBIC = ("A", "B")
DAH_RATIO_RANGE = (2.0, 5.0)
"""Allowed dah length in dits (3 is standard)."""
WEIGHT_RANGE = (25.0, 75.0)
"""Allowed weight in percent (50 is standard: mark and following space as the book says)."""
_LOG_MAX = 4000
"""Transitions kept in the log; 4000 is minutes of keying at any speed."""
_MAX_ELEMENTS_PER_TICK = 64
"""Guard against a caller that ticks after a very long pause."""

Transition = tuple[float, bool]


class Keyer:
    """Pure timing state machine for a straight key or a two-paddle keyer.

    All times are milliseconds on one monotonic clock chosen by the caller.
    Methods return the transitions they emit, in time order; the same
    transitions are appended to a log readable with :meth:`state_at` and
    :meth:`transitions_since`.
    """

    def __init__(self, dit_ms: float = 150.0, mode: Mode = "straight", iambic: Iambic = "A",
                 dah_ratio: float = 3.0, weight: float = 50.0) -> None:
        if not dit_ms > 0:
            raise ValueError(f"dit_ms must be positive, got {dit_ms!r}")
        if mode not in _MODES:
            raise ValueError(f"mode must be 'straight', 'paddle' or 'bug', got {mode!r}")
        if iambic not in _IAMBIC:
            raise ValueError(f"iambic must be 'A' or 'B', got {iambic!r}")
        self.dit_ms: float = float(dit_ms)
        self.mode: Mode = mode
        self.iambic: Iambic = iambic
        self.dah_ratio: float = 3.0
        self.weight: float = 50.0
        self.set_weighting(dah_ratio, weight)
        self._held: dict[str, bool] = {"dit": False, "dah": False}
        self._memory: list[str] = []
        self._phase: str = "idle"  # straight: 'idle' or 'mark'; paddle: 'idle', 'mark', 'gap'; bug adds 'manual'
        self._current: str | None = None
        self._squeezed: bool = False  # both paddles held at some moment during the current element
        self._manual: bool = False  # bug mode: the dah lever is holding the tone
        self._element_end: float = 0.0
        self._gap_end: float = 0.0
        self._log: deque[Transition] = deque(maxlen=_LOG_MAX)
        self._count: int = 0  # transitions ever emitted (the log may have dropped early ones)
        self._last_t: float = -math.inf

    # ------------------------------------------------------------- settings

    def set_speed(self, dit_ms: float) -> None:
        """Set the dit length used for paddle elements (``1200 / wpm``)."""
        if not dit_ms > 0:
            raise ValueError(f"dit_ms must be positive, got {dit_ms!r}")
        self.dit_ms = float(dit_ms)

    def set_mode(self, mode: Mode, t: float) -> list[Transition]:
        """Switch mode at time ``t``; releases everything and ends a sounding tone."""
        if mode not in _MODES:
            raise ValueError(f"mode must be 'straight', 'paddle' or 'bug', got {mode!r}")
        out = self.release_all(t)
        self.mode = mode
        return out

    def set_iambic(self, iambic: Iambic) -> None:
        """Choose iambic ``'A'`` or ``'B'`` for paddle mode; applies from the next decision."""
        if iambic not in _IAMBIC:
            raise ValueError(f"iambic must be 'A' or 'B', got {iambic!r}")
        self.iambic = iambic

    def set_weighting(self, dah_ratio: float | None = None, weight: float | None = None) -> None:
        """Set the dah length in dits and the weight in percent; None keeps a value. Next element on."""
        if dah_ratio is not None:
            r = float(dah_ratio)
            if not (math.isfinite(r) and DAH_RATIO_RANGE[0] <= r <= DAH_RATIO_RANGE[1]):
                raise ValueError(f"dah_ratio must be within {DAH_RATIO_RANGE}, got {dah_ratio!r}")
            self.dah_ratio = r
        if weight is not None:
            w = float(weight)
            if not (math.isfinite(w) and WEIGHT_RANGE[0] <= w <= WEIGHT_RANGE[1]):
                raise ValueError(f"weight must be within {WEIGHT_RANGE}, got {weight!r}")
            self.weight = w

    def mark_ms(self, which: str) -> float:
        """Length of an automatic element: ``w T`` for a dit, ``(ratio + w - 1) T`` for a dah."""
        w = self.weight / 50.0
        return self.dit_ms * (w if which == "dit" else self.dah_ratio + w - 1.0)

    @property
    def gap_ms(self) -> float:
        """The space after an automatic element: ``(2 - w) T``, so mark plus space stays constant."""
        return self.dit_ms * (2.0 - self.weight / 50.0)

    def release_all(self, t: float) -> list[Transition]:
        """Let go of every key and paddle at ``t``: the tone stops at once, memory is cleared."""
        self._held = {"dit": False, "dah": False}
        self._memory.clear()
        self._squeezed = False
        self._manual = False
        # An element's OFF edge is logged the moment it starts; releasing
        # earlier must cut it short, so drop the edges that lie ahead of t.
        self._drop_future_edges(t)
        out: list[Transition] = []
        if self.state_at(t):
            out.append(self._emit(t, False))
        self._phase = "idle"
        self._current = None
        return out

    def _drop_future_edges(self, t: float) -> None:
        while self._log and self._log[-1][0] > t:
            self._log.pop()
            self._count -= 1
        self._last_t = self._log[-1][0] if self._log else -math.inf

    # --------------------------------------------------------- straight key

    def key_down(self, t: float) -> list[Transition]:
        """Straight key pressed: tone on (ignored in paddle mode)."""
        if self.mode != "straight" or self._phase == "mark":
            return []
        self._phase = "mark"
        return [self._emit(t, True)]

    def key_up(self, t: float) -> list[Transition]:
        """Straight key released: tone off."""
        if self.mode != "straight" or self._phase != "mark":
            return []
        self._phase = "idle"
        return [self._emit(t, False)]

    # -------------------------------------------------------------- paddles

    def paddle_down(self, which: Paddle, t: float) -> list[Transition]:
        """Paddle pressed: start an element now if idle, else remember it (ignored in straight mode).

        In bug mode the dah paddle is a lever that keys the tone directly.
        """
        if which not in _PADDLES:
            raise ValueError(f"which must be 'dit' or 'dah', got {which!r}")
        if self.mode == "straight":
            return []
        if self.mode == "bug" and which == "dah":
            return self._bug_down(t)
        self._held[which] = True
        if self._manual:
            return []  # bug: the dah lever holds the tone; dits resume when it is let go
        if self._phase == "idle":
            return self._start_element(which, t)
        if self._held["dit"] and self._held["dah"]:
            self._squeezed = True
        if self.mode == "paddle" and which != self._current and which not in self._memory:
            self._memory.append(which)
        return []

    def paddle_up(self, which: Paddle, t: float) -> list[Transition]:
        """Paddle released; the element in progress completes on its own (a bug's dah lever stops the tone)."""
        if which not in _PADDLES:
            raise ValueError(f"which must be 'dit' or 'dah', got {which!r}")
        if self.mode == "bug" and which == "dah":
            return self._bug_up(t)
        self._held[which] = False
        return []

    def _bug_down(self, t: float) -> list[Transition]:
        """Bug dah lever pressed: tone on now; a dit in progress is cut short (its OFF edge dropped)."""
        if self._manual:
            return []
        self._manual = True
        self._memory.clear()
        self._drop_future_edges(t)
        self._phase = "manual"
        self._current = None
        if self.state_at(t):
            return []
        return [self._emit(t, True)]

    def _bug_up(self, t: float) -> list[Transition]:
        """Bug dah lever released: tone off; dits resume one space later if the dit paddle is held."""
        if not self._manual:
            return []
        self._manual = False
        out: list[Transition] = []
        if self.state_at(t):
            out.append(self._emit(t, False))
        if self._held["dit"]:
            self._phase = "gap"
            self._gap_end = t + self.gap_ms
        else:
            self._phase = "idle"
        return out

    def tick(self, t: float) -> list[Transition]:
        """Advance the paddle keyer to time ``t``; returns the transitions of any new elements.

        Call it at :attr:`next_wakeup_ms` (or more often).  In straight mode
        and while idle it does nothing.
        """
        out: list[Transition] = []
        if self.mode == "straight" or self._manual:
            return out
        for _ in range(_MAX_ELEMENTS_PER_TICK):
            if self._phase == "mark" and t >= self._element_end:
                self._phase = "gap"
            if self._phase == "gap" and t >= self._gap_end:
                nxt = self._next_element()
                if nxt is None:
                    self._phase = "idle"
                    self._current = None
                    return out
                out.extend(self._start_element(nxt, self._gap_end))
                continue
            return out
        return out

    @property
    def next_wakeup_ms(self) -> float | None:
        """When :meth:`tick` next has a decision to make, or ``None`` while idle or in straight mode."""
        if self.mode == "straight" or self._manual:
            return None
        if self._phase == "mark":
            return self._element_end
        if self._phase == "gap":
            return self._gap_end
        return None

    # ------------------------------------------------------------------ log

    def state_at(self, t: float) -> bool:
        """Tone state at time ``t`` according to the transitions emitted so far."""
        state = False
        for tt, on in self._log:
            if tt > t:
                break
            state = on
        return state

    @property
    def transition_count(self) -> int:
        """Transitions ever emitted; pass it to :meth:`transitions_since` to read only new ones."""
        return self._count

    def transitions_since(self, index: int) -> tuple[list[Transition], int]:
        """Transitions emitted after the ``index``-th; returns them and the new index."""
        dropped = self._count - len(self._log)
        start = max(index - dropped, 0)
        items = list(self._log)[start:]
        return items, self._count

    def transitions_in(self, t0: float, t1: float) -> list[Transition]:
        """Transitions with ``t0 <= t < t1``, in time order."""
        return [(tt, on) for tt, on in self._log if t0 <= tt < t1]

    # ------------------------------------------------------------- internals

    def _emit(self, t: float, on: bool) -> Transition:
        t = max(float(t), self._last_t)  # the log is never out of order
        self._last_t = t
        tr = (t, on)
        self._log.append(tr)
        self._count += 1
        return tr

    def _start_element(self, which: str, t: float) -> list[Transition]:
        self._current = which
        self._phase = "mark"
        self._element_end = t + self.mark_ms(which)
        self._gap_end = self._element_end + self.gap_ms
        self._squeezed = self._held["dit"] and self._held["dah"]
        if which in self._memory:
            self._memory.remove(which)
        return [self._emit(t, True), self._emit(self._element_end, False)]

    def _next_element(self) -> str | None:
        if self._memory:
            return self._memory.pop(0)
        dit, dah = self._held["dit"], self._held["dah"]
        if self.mode == "bug":
            return "dit" if dit else None
        if dit and dah:
            return "dah" if self._current == "dit" else "dit"
        if dit:
            return "dit"
        if dah:
            return "dah"
        if self.iambic == "B" and self._squeezed and self._current is not None:
            self._squeezed = False  # the one extra element of mode B, then silence
            return "dah" if self._current == "dit" else "dit"
        return None

    def __repr__(self) -> str:
        return (f"Keyer(mode={self.mode!r}, iambic={self.iambic!r}, dit_ms={self.dit_ms:.0f}, "
                f"dah_ratio={self.dah_ratio:g}, weight={self.weight:g}, phase={self._phase!r}, "
                f"held={self._held}, memory={self._memory})")


def envelope(
    transitions: list[Transition],
    initial: bool,
    t0_ms: float,
    n: int,
    fs: int,
    ramp_ms: float = 3.0,
) -> np.ndarray:
    """Amplitude envelope (0..1, ``float32``) for ``n`` samples starting at ``t0_ms``.

    ``initial`` is the tone state just before ``t0_ms - ramp_ms``;
    ``transitions`` are the ones that can affect the window (any with
    ``t > t0_ms - ramp_ms``), in time order.  Each transition ramps linearly
    over ``ramp_ms`` so the tone never clicks.
    """
    env = np.full(n, 1.0 if initial else 0.0, dtype=np.float32)
    ramp_n = max(1, int(round(ramp_ms * fs / 1000.0)))
    prev = 1.0 if initial else 0.0
    for tt, on in transitions:
        target = 1.0 if on else 0.0
        i0 = int(round((tt - t0_ms) * fs / 1000.0))
        if i0 >= n:
            break
        i1 = i0 + ramp_n
        lo, hi = max(i0, 0), min(i1, n)
        if lo < hi:
            k = np.arange(lo, hi, dtype=np.float32) - i0
            env[lo:hi] = prev + (target - prev) * (k + 1.0) / ramp_n
        if i1 < n:
            env[max(i1, 0):] = target
        prev = target
    return env


def _sd() -> Any:
    import sounddevice  # deliberately local: loads PortAudio

    return sounddevice


class LiveKey:
    """Real-time tone for a :class:`Keyer`, through a ``sounddevice`` output stream.

    The stream callback advances the keyer with the stream's own clock and
    renders the envelope, so paddle elements are sample-accurate.  Presses
    from the UI thread are stamped with the same clock (:meth:`now_ms`);
    every keyer call goes through one lock.

    Two pitches: the speakers play the *sidetone* (``sidetone_hz``, a
    comfortable pitch for the operator; None follows ``f0``) while
    :meth:`feed_block`, the decoder's software loopback, always renders at
    ``f0``, the frequency the detector is tuned to.
    """

    def __init__(
        self,
        keyer: Keyer,
        f0: float = 2491.0,
        fs: int = 48000,
        device: int | str | None = None,
        amplitude: float = 0.3,
        ramp_ms: float = 3.0,
        block_size: int = 480,
        sidetone_hz: float | None = None,
    ) -> None:
        if not 0 < f0 < fs / 2:
            raise ValueError(f"f0 must be between 0 and fs/2, got {f0!r}")
        if sidetone_hz is not None and not 0 < sidetone_hz < fs / 2:
            raise ValueError(f"sidetone_hz must be between 0 and fs/2, got {sidetone_hz!r}")
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError(f"amplitude must be in 0..1, got {amplitude!r}")
        self.keyer = keyer
        self.f0 = float(f0)
        self.sidetone_hz: float | None = None if sidetone_hz is None else float(sidetone_hz)
        self.fs = int(fs)
        self.device = device
        self.amplitude = float(amplitude)
        self.speakers: bool = True
        """Whether the sidetone reaches the speakers; :meth:`feed_block` is unaffected."""
        self.ramp_ms = float(ramp_ms)
        self.block_size = int(block_size)
        self._lock = threading.Lock()
        self._stream: Any = None
        self._t0: float | None = None
        self._stream_ms: float = 0.0
        self._phase_out: float = 0.0
        self._phase_feed: float = 0.0
        self.last_error: str = ""

    # ------------------------------------------------------------ lifecycle

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Open the output stream; the tone stays silent until a key is pressed."""
        if self._stream is not None:
            return
        sd = _sd()
        stream = sd.OutputStream(samplerate=self.fs, channels=1, dtype="float32",
                                 blocksize=self.block_size, device=self.device, callback=self._callback)
        self._stream_ms = 0.0
        self._t0 = time.monotonic()
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        """Release everything and close the stream."""
        stream, self._stream = self._stream, None
        with self._lock:
            self.keyer.release_all(self.now_ms())
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # PortAudio can complain on a device that vanished
                self.last_error = str(exc)
        self._t0 = None

    def now_ms(self) -> float:
        """The keyer clock: milliseconds since :meth:`start` (0 when stopped)."""
        return 0.0 if self._t0 is None else (time.monotonic() - self._t0) * 1000.0

    # ------------------------------------------------------------- controls

    def key_down(self) -> None:
        with self._lock:
            self.keyer.key_down(self.now_ms())

    def key_up(self) -> None:
        with self._lock:
            self.keyer.key_up(self.now_ms())

    def paddle_down(self, which: Paddle) -> None:
        with self._lock:
            self.keyer.paddle_down(which, self.now_ms())

    def paddle_up(self, which: Paddle) -> None:
        with self._lock:
            self.keyer.paddle_up(which, self.now_ms())

    def release_all(self) -> None:
        with self._lock:
            self.keyer.release_all(self.now_ms())

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            self.keyer.set_mode(mode, self.now_ms())

    def set_speed(self, dit_ms: float) -> None:
        with self._lock:
            self.keyer.set_speed(dit_ms)

    def set_iambic(self, iambic: Iambic) -> None:
        with self._lock:
            self.keyer.set_iambic(iambic)

    def set_weighting(self, dah_ratio: float | None = None, weight: float | None = None) -> None:
        with self._lock:
            self.keyer.set_weighting(dah_ratio, weight)

    def set_frequency(self, f0: float) -> None:
        """Retune the decoder feed (and the speakers when no sidetone is set)."""
        if not 0 < f0 < self.fs / 2:
            raise ValueError(f"f0 must be between 0 and fs/2, got {f0!r}")
        self.f0 = float(f0)

    def set_speakers(self, on: bool) -> None:
        """Send the sidetone to the speakers or not (takes effect at the next block)."""
        self.speakers = bool(on)

    def set_sidetone(self, hz: float | None) -> None:
        """Pitch the speakers play; None (or 0) follows :attr:`f0`. Takes effect at the next block."""
        if hz is None or float(hz) == 0.0:
            self.sidetone_hz = None
            return
        if not 0 < hz < self.fs / 2:
            raise ValueError(f"sidetone_hz must be between 0 and fs/2, got {hz!r}")
        self.sidetone_hz = float(hz)

    @property
    def speaker_hz(self) -> float:
        """The pitch the speakers play now: the sidetone, or ``f0`` without one."""
        return self.f0 if self.sidetone_hz is None else self.sidetone_hz

    def is_on(self) -> bool:
        """Whether the tone is sounding now."""
        with self._lock:
            return self.keyer.state_at(self.now_ms())

    def transitions_since(self, index: int) -> tuple[list[Transition], int]:
        """Thread-safe :meth:`Keyer.transitions_since`, for readers on the UI thread."""
        with self._lock:
            return self.keyer.transitions_since(index)

    # ------------------------------------------------------------ rendering

    def _render_env(self, t0_ms: float, n: int) -> np.ndarray:
        """Envelope for ``n`` samples from ``t0_ms``; caller holds the lock."""
        window_start = t0_ms - self.ramp_ms
        initial = self.keyer.state_at(window_start)
        trans = self.keyer.transitions_in(window_start, t0_ms + n * 1000.0 / self.fs + self.ramp_ms)
        # transitions_in is left-inclusive; a transition exactly at window_start is already in `initial`
        trans = [tr for tr in trans if tr[0] > window_start]
        return envelope(trans, initial, t0_ms, n, self.fs, self.ramp_ms)

    def _sine(self, n: int, phase: float, hz: float) -> tuple[np.ndarray, float]:
        step = 2.0 * math.pi * hz / self.fs
        k = np.arange(n, dtype=np.float64)
        samples = np.sin(phase + step * k).astype(np.float32)
        return samples, (phase + step * n) % (2.0 * math.pi)

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        block_ms = frames * 1000.0 / self.fs
        with self._lock:
            t0 = self._stream_ms
            self.keyer.tick(t0 + block_ms)
            env = self._render_env(t0, frames)
            self._stream_ms = t0 + block_ms
        tone, self._phase_out = self._sine(frames, self._phase_out, self.speaker_hz)
        outdata[:, 0] = (self.amplitude * env * tone) if self.speakers else 0.0

    def feed_block(self, t0_ms: float, n: int) -> np.ndarray:
        """The key's tone for ``n`` samples starting at keyer time ``t0_ms`` (for the decoder feed)."""
        with self._lock:
            env = self._render_env(t0_ms, n)
        tone, self._phase_feed = self._sine(n, self._phase_feed, self.f0)
        return (self.amplitude * env * tone).astype(np.float32)

    def __repr__(self) -> str:
        side = "follows f0" if self.sidetone_hz is None else f"{self.sidetone_hz:.0f}"
        return (f"LiveKey(f0={self.f0:.0f}, sidetone={side}, fs={self.fs}, running={self.running}, "
                f"keyer={self.keyer!r})")
