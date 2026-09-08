"""Signal-processing building blocks for the Morse decoder.

Pure numpy/scipy: no Qt, no audio hardware.  Everything here works on mono
blocks of ``float32`` samples in the range -1..1 and is testable with
synthetic arrays.

Power normalisation
-------------------
``goertzel_power`` returns ``4 * |X(f0)|**2 / N**2`` where ``X(f0)`` is the
DTFT of the ``N``-sample block evaluated at ``f0`` (what the Goertzel
recursion computes).  For ``x[n] = A sin(w0 n + phi)``::

    X(w0) = (A / 2j) * (N e^{j phi} - e^{-j phi} * sum_n e^{-2j w0 n})

The second (image) term is a geometric sum bounded by ``1 / |sin w0|`` and
vanishes exactly when ``f0`` sits on a DFT bin, so ``|X(w0)| ~= A N / 2`` and
the normalised power is ``A**2``: a full-scale sine gives 1.0 regardless of
block length, exactly on-bin and within a few percent off-bin.
``spectrum`` uses the same idea with a Hann window: ``2 * |rfft| / sum(win)``
so a full-scale sine peaks near 0 dB (scalloping loss at most 1.4 dB).

Single bin, deliberately
------------------------
The tone power is one Goertzel bin at ``f0``; the ``f0 +- 50 Hz`` neighbour
bins mentioned in early planning notes are *not* summed.  At 10 ms the bin is
already about 100 Hz wide, so a slightly off ``f0`` is tolerated anyway, and a
three-bin sum would break the 0 dB normalisation in a block-length-dependent
way: ``+-50 Hz`` is on a DFT bin only when ``N`` is a multiple of 960, so the
sum reads about +2.6 dB at ``N = 480`` and +4.2 dB at ``N = 240`` for a
full-scale sine.  Ports (``web/js/dsp.js``) and the parity vectors written by
``tools/export_vectors.py`` must evaluate the single bin too.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from scipy.signal.windows import hann

_DB_FLOOR = 1e-12
"""Added to linear power before ``log10`` so silence reads -120 dB, not -inf."""


def _as_float64(block: np.ndarray) -> np.ndarray:
    """Return ``block`` as a flat float64 array (the recursions run in float64)."""
    return np.asarray(block, dtype=np.float64).ravel()


def _goertzel_mag2(x: np.ndarray, coeff: float) -> float:
    """``|X(w0)|**2`` of ``x`` via the Goertzel recursion, ``coeff = 2 cos(w0)``.

    The recursion ``s[n] = x[n] + coeff*s[n-1] - s[n-2]`` is the IIR filter
    ``b = [1], a = [1, -coeff, 1]``, evaluated in C by ``lfilter``.  After the
    last sample, ``|X|**2 = s1**2 + s2**2 - coeff*s1*s2``.
    """
    if x.size < 2:
        return 0.0
    s = lfilter([1.0], [1.0, -coeff, 1.0], x)
    s1 = float(s[-1])
    s2 = float(s[-2])
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def goertzel_power(block: np.ndarray, f0: float, fs: int) -> float:
    """Linear tone power at ``f0`` in ``block``.

    A single Goertzel bin, ``4 * |X(f0)|**2 / N**2``, normalised so that a
    full-scale sine at ``f0`` returns 1.0 regardless of block length (see the
    module docstring; neighbouring bins are intentionally not summed).
    Silence returns 0.0.
    """
    x = _as_float64(block)
    n = x.size
    if n < 2:
        return 0.0
    coeff = 2.0 * np.cos(2.0 * np.pi * f0 / fs)
    return 4.0 * _goertzel_mag2(x, coeff) / (n * n)


class Goertzel:
    """Single-frequency tone-power detector for fixed-size blocks.

    ``power`` equals ``goertzel_power(block, f0, fs)``; the class just caches
    the recursion coefficient and normalisation between calls.  Blocks of a
    length other than ``block_size`` are accepted and normalised by their
    actual length.
    """

    def __init__(self, f0: float, fs: int, block_size: int) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.fs = int(fs)
        self.block_size = int(block_size)
        self._norm = 4.0 / float(self.block_size * self.block_size)
        self._f0 = 0.0
        self._coeff = 0.0
        self.set_frequency(f0)

    @property
    def f0(self) -> float:
        """Tone frequency in Hz (change it with ``set_frequency``)."""
        return self._f0

    def set_frequency(self, f0: float) -> None:
        """Retune the detector to ``f0`` Hz."""
        self._f0 = float(f0)
        self._coeff = 2.0 * np.cos(2.0 * np.pi * self._f0 / self.fs)

    def power(self, block: np.ndarray) -> float:
        """Linear power at ``f0``, normalised so a full-scale sine gives 1.0."""
        x = _as_float64(block)
        n = x.size
        if n < 2:
            return 0.0
        norm = self._norm if n == self.block_size else 4.0 / float(n * n)
        return norm * _goertzel_mag2(x, self._coeff)

    def power_db(self, block: np.ndarray) -> float:
        """``10*log10(power + 1e-12)``: 0 dB for full scale, -120 dB for silence."""
        return float(10.0 * np.log10(self.power(block) + _DB_FLOOR))


class BandPass:
    """Butterworth band-pass ``f0 +- half_width`` with state carried between blocks.

    ``order`` is passed to ``scipy.signal.butter`` as ``N``; for a band-pass the
    resulting filter has ``order`` second-order sections (``2*order`` poles).
    ``process`` runs ``sosfilt`` with the saved ``zi`` so a stream fed block by
    block gives exactly the same output as filtering it in one call.
    """

    def __init__(self, f0: float, fs: int, half_width: float = 100.0, order: int = 4) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if half_width <= 0:
            raise ValueError("half_width must be positive")
        if order < 1:
            raise ValueError("order must be at least 1")
        self.fs = int(fs)
        self.half_width = float(half_width)
        self.order = int(order)
        self._f0 = 0.0
        self._sos = np.empty((0, 6))
        self._zi = np.empty((0, 2))
        self.set_frequency(f0)

    @property
    def f0(self) -> float:
        """Centre frequency in Hz."""
        return self._f0

    def set_frequency(self, f0: float) -> None:
        """Redesign the filter around ``f0`` and reset the filter state."""
        nyquist = self.fs / 2.0
        lo = max(f0 - self.half_width, 1.0)
        hi = min(f0 + self.half_width, nyquist * 0.999)
        if not lo < hi:
            raise ValueError(f"band {lo:.1f}..{hi:.1f} Hz is not valid for fs={self.fs}")
        self._f0 = float(f0)
        self._sos = butter(self.order, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        self.reset()

    def reset(self) -> None:
        """Zero the filter state without changing the design."""
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filter one block; returns ``float32`` of the same length."""
        x = _as_float64(block)
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)


def spectrum(frame: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed power spectrum of ``frame``.

    Returns ``(freqs_hz, power_db)`` from ``rfft``.  Magnitudes are scaled by
    ``2 / sum(window)`` so a full-scale sine peaks near 0 dB (exactly 0 dB on a
    bin, at worst -1.4 dB half-way between bins); silence reads -120 dB.
    """
    x = _as_float64(frame)
    n = x.size
    if n == 0:
        return np.empty(0), np.empty(0)
    win = hann(n, sym=False)
    scale = 2.0 / float(win.sum())
    mag = np.abs(np.fft.rfft(x * win)) * scale
    power_db = 10.0 * np.log10(mag * mag + _DB_FLOOR)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power_db


def find_tone_frequency(freqs: np.ndarray, power_db: np.ndarray,
                        fmin: float = 300.0, fmax: float = 8000.0) -> tuple[float, float]:
    """Locate the strongest spectral peak within ``[fmin, fmax]``.

    Returns ``(peak_hz, prominence_db)``: the peak frequency, refined to
    sub-bin accuracy by fitting a parabola through the peak bin and its two
    neighbours in dB, and the peak level minus the median of ``power_db``
    inside the band.  Returns ``(0.0, 0.0)`` when no bin lies in the band.
    """
    f = _as_float64(freqs)
    p = _as_float64(power_db)
    if f.shape != p.shape:
        raise ValueError("freqs and power_db must have the same shape")
    idx = np.flatnonzero((f >= fmin) & (f <= fmax))
    if idx.size == 0:
        return 0.0, 0.0
    band = p[idx]
    j = int(np.argmax(band))
    i = int(idx[j])
    peak_hz = f[i]
    if 0 < i < f.size - 1:
        a, b, c = p[i - 1], p[i], p[i + 1]
        denom = a - 2.0 * b + c
        if np.isfinite(denom) and denom < 0.0:
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            peak_hz = f[i] + delta * (f[i + 1] - f[i])
    prominence = band[j] - float(np.median(band))
    return float(peak_hz), float(prominence)


def synth_keyed_tone(pattern: list[tuple[bool, float]], f0: float, fs: int,
                     amplitude: float = 0.3, noise_rms: float = 0.0,
                     reverb_ms: float = 0.0, seed: int = 0) -> np.ndarray:
    """Synthesise a keyed sine for tests.

    ``pattern`` is ``[(on, duration_ms), ...]``; each segment is rounded to
    whole samples.  With ``reverb_ms > 0`` the on/off envelope gets an
    exponential release tail with that 1/e time constant (attack stays
    instant), so marks lengthen and gaps shorten as they do in a room.  White
    Gaussian noise with the given rms is added from
    ``numpy.random.default_rng(seed)``.  The result is ``float32`` clipped to
    -1..1.
    """
    lengths = [max(0, int(round(ms * fs / 1000.0))) for _, ms in pattern]
    n = int(sum(lengths))
    env = np.zeros(n)
    pos = 0
    for (on, _ms), length in zip(pattern, lengths):
        if on:
            env[pos:pos + length] = 1.0
        pos += length

    if reverb_ms > 0.0 and n > 0:
        tau = reverb_ms * fs / 1000.0
        idx = np.arange(n)
        last_on = np.maximum.accumulate(np.where(env > 0.0, idx, -1))
        heard = last_on >= 0
        dist = np.where(heard, idx - last_on, 0)
        env = np.where(heard, np.exp(-dist / tau), 0.0)

    t = np.arange(n) / float(fs)
    sig = amplitude * env * np.sin(2.0 * np.pi * f0 * t)
    if noise_rms > 0.0:
        rng = np.random.default_rng(seed)
        sig = sig + rng.normal(0.0, noise_rms, n)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)
