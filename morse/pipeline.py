"""Per-block decode pipeline: audio block -> tone power -> ON/OFF runs -> text.

:class:`Pipeline` chains the pieces described in ``docs/PLAN.md`` sections 3
and 4 for one stream of fixed-size mono blocks:

1. sanitise: a block containing NaN or inf is replaced by zeros and counted
   in :attr:`Pipeline.bad_blocks`, so one corrupt block cannot poison the
   band-pass state or the detector's level trackers,
2. input level in dBFS (rms of the raw block),
3. :class:`morse.dsp.BandPass` around ``f0`` for the display waveform,
4. :class:`morse.dsp.Goertzel` tone power at ``f0`` in dB,
5. :class:`morse.tone_detector.ToneDetector` turns the power series into
   debounced ON/OFF :class:`morse.runs.Run` objects,
6. :class:`morse.decoder.MorseDecoder` turns final runs into text, and is told
   how long the current OFF run has lasted so the last letter of a message
   appears without waiting for the next tone.

The Goertzel stage reads the *raw* block, not the band-passed one: the filter
has a group delay and a ring-down that would smear the tone edges, and the
Goertzel bin is already narrow.  The band-passed block is only for display.

The detector is constructed with the contract defaults
(``ToneDetector(block_ms=...)`` and nothing else): its thresholds are sized
from the tracked noise and signal statistics, and any per-application
override would make the two WAV fixtures and the detector's own acceptance
tests disagree about what "the defaults" mean.

:func:`decode_wav` runs the same pipeline over a WAV file, which is how the
regression fixtures under ``tests/fixtures`` are checked.  No Qt and no audio
hardware anywhere in this module.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morse.decoder import MorseDecoder
from morse.dsp import BandPass, Goertzel
from morse.runs import Run
from morse.tone_detector import ToneDetector, combine_db, is_tonal, neighbor_frequencies, tonality_db

__all__ = [
    "BlockResult",
    "Pipeline",
    "decode_samples",
    "decode_wav",
    "load_wav",
]

_POWER_FLOOR = 1e-12
"""Added to the mean square before ``log10`` so a silent block reads -120 dBFS."""


def _validate_f0(f0: float, fs: int) -> float:
    """Return ``f0`` as a float after checking ``0 < f0 < fs/2``.

    Raises ``ValueError`` otherwise (NaN and inf included): a tone at or above
    the Nyquist frequency cannot be detected, and ``BandPass`` alone would
    accept ``f0 = 0`` because it clamps the lower band edge to 1 Hz.
    """
    value = float(f0)
    nyquist = fs / 2.0
    if not (math.isfinite(value) and 0.0 < value < nyquist):
        raise ValueError(f"f0 must satisfy 0 < f0 < fs/2 = {nyquist:g} Hz, got {f0!r}")
    return value


@dataclass
class BlockResult:
    """Everything the pipeline learned from one block, for the UI and for tests."""

    power_db: float
    """Goertzel tone power at ``f0`` in dB (0 dB = full-scale sine)."""
    on: bool
    """The detector's current ON/OFF verdict after this block."""
    threshold_hi_db: float
    """Power above which the detector switches ON."""
    threshold_lo_db: float
    """Power below which the detector switches OFF."""
    floor_db: float
    """Tracked noise level (mean of OFF-block power)."""
    peak_db: float
    """Tracked signal level (mean of ON-block power)."""
    level_dbfs: float
    """``20*log10(rms of the raw block)``; -120 for digital silence."""
    new_text: str
    """Text emitted by the decoder during this block (empty when nothing new)."""
    runs: list[Run] = field(default_factory=list)
    """Runs that became final during this block, oldest first."""
    filtered: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    """Band-passed copy of the block (``float32``) for the waveform display."""
    tonality_db: float = 0.0
    peakiness_db: float = 0.0
    """How far the tone bin stands above its neighbour bands (see ``is_tonal``)."""
    """Tone power relative to the block's total power: 0 for a pure tone, about -24 for noise."""


class Pipeline:
    """Decode a stream of fixed-size mono ``float32`` blocks into Morse text.

    Feed blocks of ``block_size`` samples to :meth:`process_block`; each call
    returns a :class:`BlockResult`.  Call :meth:`flush` at the end of a stream
    to hand the still-pending runs to the decoder and get the last letters.

    Attributes:
        fs: sample rate in Hz.
        block_size: samples per block; ``block_ms`` is derived from it.
        block_ms: duration of one block in ms, stamped into every ``Run``.
        goertzel: the tone-power stage (``goertzel.f0`` is the tone frequency).
        bandpass: the display filter.
        detector: the ON/OFF run detector, built with the contract defaults.
        decoder: the timing decoder; ``decoder.text`` is everything decoded so far.
        blocks_processed: number of blocks fed so far (reset by :meth:`reset`).
        bad_blocks: number of blocks that contained NaN or inf and were
            replaced by zeros (reset by :meth:`reset`).
        last_flush_runs: runs handed to the decoder by the most recent
            :meth:`flush` or :meth:`set_frequency`, so callers that collect
            ``BlockResult.runs`` can complete their list.
    """

    def __init__(
        self,
        fs: int = 48000,
        block_size: int = 480,
        f0: float = 2491.0,
        wpm: float | None = None,
        adaptive: bool = True,
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.fs = int(fs)
        self.block_size = int(block_size)
        self.block_ms = 1000.0 * self.block_size / self.fs
        f0 = _validate_f0(f0, self.fs)
        self.bandpass = BandPass(f0, self.fs)
        self.goertzel = Goertzel(f0, self.fs, self.block_size)
        self._neighbors = [Goertzel(f, self.fs, self.block_size) for f in neighbor_frequencies(f0, self.fs)]
        # Contract defaults only: block_ms is timing, not a threshold parameter.
        self.detector = ToneDetector(block_ms=self.block_ms)
        self.decoder = MorseDecoder(wpm=wpm, adaptive=adaptive)
        self.blocks_processed: int = 0
        self.bad_blocks: int = 0
        self.last_flush_runs: list[Run] = []

    # ---------------------------------------------------------------- state

    @property
    def f0(self) -> float:
        """Tone frequency in Hz; change it with :meth:`set_frequency`."""
        return self.goertzel.f0

    @property
    def text(self) -> str:
        """Everything the decoder has emitted so far (``decoder.text``)."""
        return self.decoder.text

    @property
    def elapsed_ms(self) -> float:
        """Audio time processed so far, in ms."""
        return self.blocks_processed * self.block_ms

    # -------------------------------------------------------------- feeding

    def process_block(self, block: np.ndarray) -> BlockResult:
        """Run one block through every stage and return what was learned.

        ``block`` should hold ``block_size`` mono samples in -1..1; other
        lengths are accepted (the Goertzel stage normalises by the actual
        length) but still count as one ``block_ms`` for run timing.  A block
        containing NaN or inf is replaced by zeros before any filtering and
        counted in :attr:`bad_blocks`.
        """
        x = np.asarray(block, dtype=np.float32).ravel()
        if x.size and not np.isfinite(x).all():
            x = np.zeros(x.size, dtype=np.float32)
            self.bad_blocks += 1
        if x.size:
            mean_square = float(np.mean(np.square(x, dtype=np.float64)))
        else:
            mean_square = 0.0
        level_dbfs = float(10.0 * np.log10(mean_square + _POWER_FLOOR))

        filtered = self.bandpass.process(x)
        power_db = self.goertzel.power_db(x)
        neighbor_db = combine_db([g.power_db(x) for g in self._neighbors]) if self._neighbors else None

        det = self.detector
        # A loud but broadband block (click, speech), or one no sharper than its
        # neighbour bands, never switches the detector ON, though it still
        # teaches it the noise level.
        runs = det.update(power_db, tonal=is_tonal(power_db, level_dbfs, neighbor_db=neighbor_db))
        new_text = self._feed_runs(runs)
        current = det.current_run
        if not current.on:
            new_text += self.decoder.idle(current.ms)
        self.blocks_processed += 1

        return BlockResult(
            power_db=power_db,
            on=det.state,
            threshold_hi_db=det.threshold_hi_db,
            threshold_lo_db=det.threshold_lo_db,
            floor_db=det.floor_db,
            peak_db=det.peak_db,
            level_dbfs=level_dbfs,
            new_text=new_text,
            runs=runs,
            filtered=filtered,
            tonality_db=tonality_db(power_db, level_dbfs),
            peakiness_db=(power_db - neighbor_db) if neighbor_db is not None else 0.0,
        )

    def flush(self) -> str:
        """End of stream: finalise pending runs, decode them, return the new text.

        The detector's pending runs (including the in-progress one) are fed to
        the decoder and recorded in :attr:`last_flush_runs`; then the pending
        letter, if any, is decoded as if the silence went on forever.  The
        detector keeps its level trackers, so processing may continue after a
        flush.
        """
        text = self._flush_detector()
        text += self.decoder.idle(float("inf"))
        return text

    def set_frequency(self, f0: float) -> None:
        """Retune every stage to ``f0`` Hz (``0 < f0 < fs/2``, else ``ValueError``).

        What the detector holds is finalised first (``detector.flush()`` fed to
        the decoder and recorded in :attr:`last_flush_runs`), then the detector
        is reset because its noise and signal levels belonged to the old bin,
        and the Goertzel and band-pass stages are redesigned.  The decoder keeps
        its text, its pending letter and its timing estimates.  Nothing changes
        when the value is invalid.
        """
        f0 = _validate_f0(f0, self.fs)
        self._flush_detector()
        self.detector.reset()
        self.bandpass.set_frequency(f0)
        self.goertzel.set_frequency(f0)
        self._neighbors = [Goertzel(f, self.fs, self.block_size) for f in neighbor_frequencies(f0, self.fs)]

    def reset(self, keep_timing: bool = False) -> None:
        """Start over: clear filter state, detector trackers and decoded text.

        With ``keep_timing`` the decoder keeps its dit-length estimate.
        """
        self.bandpass.reset()
        self.detector.reset()
        self.decoder.reset(keep_timing=keep_timing)
        self.blocks_processed = 0
        self.bad_blocks = 0
        self.last_flush_runs = []

    # ------------------------------------------------------------ internals

    def _feed_runs(self, runs: list[Run]) -> str:
        return "".join(self.decoder.feed(run) for run in runs)

    def _flush_detector(self) -> str:
        """Hand every pending detector run to the decoder; return the emitted text."""
        runs = self.detector.flush()
        self.last_flush_runs = runs
        return self._feed_runs(runs)


# ---------------------------------------------------------------- offline


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Scale WAV samples of any integer or float type to ``float32`` in -1..1."""
    if data.dtype == np.uint8:  # 8-bit WAV is unsigned
        return ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        full_scale = float(2 ** (8 * data.dtype.itemsize - 1))  # 32768 for int16, 2**31 for int32 (and 24-bit)
        return (data.astype(np.float64) / full_scale).astype(np.float32)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    raise ValueError(f"unsupported WAV sample type {data.dtype}")


def _channel_zero(data: np.ndarray) -> np.ndarray:
    """Return channel 0 of ``(frames, channels)`` data; 1-D passes through."""
    if data.ndim > 2:
        raise ValueError(f"expected mono or (frames, channels) samples, got shape {data.shape}")
    if data.ndim == 2:
        return data[:, 0]
    return data.ravel()  # 0-D becomes one sample


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a WAV file as ``(fs, mono float32 samples in -1..1)``.

    Accepts 8/16/24/32-bit PCM and 32/64-bit float files; multi-channel files
    contribute channel 0 only.  Nothing is resampled.  ``OSError`` (missing
    file, a directory, no permission) propagates unchanged.
    """
    from scipy.io import wavfile  # local: keeps the module import cheap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # e.g. LIST/INFO chunks
        fs, data = wavfile.read(str(path))
    return int(fs), _to_float32(_channel_zero(np.asarray(data)))


def decode_samples(
    samples: np.ndarray,
    fs: int,
    f0: float,
    wpm: float | None = None,
    block_size: int = 480,
) -> tuple[str, list[Run]]:
    """Run a whole signal through a fresh :class:`Pipeline`.

    ``samples`` is mono (1-D) or ``(frames, channels)``, in which case channel
    0 is used; more than two dimensions raise ``ValueError``.  Integer samples
    are scaled to -1..1 like WAV data.  The signal is cut into ``block_size``
    blocks (a short tail is zero-padded to a full block),
    :meth:`Pipeline.flush` is called at the end, and ``(decoded_text,
    all_runs)`` is returned with every run in stream order.
    """
    x = _to_float32(_channel_zero(np.asarray(samples)))
    pipe = Pipeline(fs=fs, block_size=block_size, f0=f0, wpm=wpm)
    runs: list[Run] = []
    n_full = x.size // block_size
    for block in x[: n_full * block_size].reshape(n_full, block_size):
        runs.extend(pipe.process_block(block).runs)
    remainder = x.size - n_full * block_size
    if remainder:
        tail = np.zeros(block_size, dtype=np.float32)
        tail[:remainder] = x[n_full * block_size:]
        runs.extend(pipe.process_block(tail).runs)
    pipe.flush()
    runs.extend(pipe.last_flush_runs)
    return pipe.text, runs


def decode_wav(
    path: str | Path,
    f0: float,
    wpm: float | None = None,
    block_size: int = 480,
) -> tuple[str, list[Run]]:
    """Decode a WAV file offline; returns ``(decoded_text, all_runs)``.

    Any integer or float WAV format is accepted, mono or multi-channel
    (channel 0 used).  The file's own sample rate is used, so a ``block_size``
    of 480 is 10 ms only for 48 kHz files.  ``Pipeline.flush`` is called at
    the end so the last letter is included.
    """
    fs, x = load_wav(path)
    return decode_samples(x, fs, f0, wpm=wpm, block_size=block_size)
