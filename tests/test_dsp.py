"""Tests for morse.dsp: Goertzel, BandPass, spectrum, tone finder, synthesiser."""
from __future__ import annotations

import numpy as np
import pytest

from morse.dsp import (
    BandPass,
    Goertzel,
    find_tone_frequency,
    goertzel_power,
    spectrum,
    synth_keyed_tone,
)

FS = 48000
N = 480            # 10 ms
BEEPER = 2491.0    # measured PC-speaker tone, off-bin for N=480


def sine(f: float, n: int = N, fs: int = FS, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


def db(power: float) -> float:
    return 10.0 * np.log10(power + 1e-30)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# --- goertzel_power -------------------------------------------------------

def test_goertzel_full_scale_on_bin_is_exactly_one():
    # 1000 Hz is bin 10 of a 480-sample block at 48 kHz: the image term vanishes.
    for phase in (0.0, 0.7, 1.9, 3.1):
        assert goertzel_power(sine(1000.0, phase=phase), 1000.0, FS) == pytest.approx(1.0, rel=1e-6)


def test_goertzel_full_scale_off_bin_is_near_one():
    for phase in (0.0, 0.7, 1.9, 3.1):
        assert goertzel_power(sine(BEEPER, phase=phase), BEEPER, FS) == pytest.approx(1.0, abs=0.05)


def test_goertzel_power_scales_with_amplitude_squared():
    assert goertzel_power(sine(1000.0, amp=0.3), 1000.0, FS) == pytest.approx(0.09, rel=1e-6)
    assert goertzel_power(sine(1000.0, amp=0.01), 1000.0, FS) == pytest.approx(1e-4, rel=1e-6)


@pytest.mark.parametrize("n", [240, 480, 960, 2048, 4800])
def test_goertzel_normalisation_independent_of_block_length(n):
    assert goertzel_power(sine(1000.0, n=n), 1000.0, FS) == pytest.approx(1.0, abs=0.05)
    assert goertzel_power(sine(BEEPER, n=n), BEEPER, FS) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("f0", [1000.0, BEEPER])
@pytest.mark.parametrize("offset", [500.0, -500.0])
def test_goertzel_rejects_tone_500hz_away_by_40db(f0, offset):
    p_on = goertzel_power(sine(f0), f0, FS)
    p_off = goertzel_power(sine(f0 + offset), f0, FS)
    assert db(p_on) - db(p_off) >= 40.0


def test_goertzel_silence():
    p = goertzel_power(np.zeros(N, dtype=np.float32), 1000.0, FS)
    assert p == 0.0
    assert 10 * np.log10(p + 1e-12) <= -90.0


def test_goertzel_accepts_lists_float64_and_column_vectors():
    ref = goertzel_power(sine(1000.0), 1000.0, FS)
    assert goertzel_power(sine(1000.0).tolist(), 1000.0, FS) == pytest.approx(ref)
    assert goertzel_power(sine(1000.0).astype(np.float64), 1000.0, FS) == pytest.approx(ref)
    assert goertzel_power(sine(1000.0).reshape(-1, 1), 1000.0, FS) == pytest.approx(ref)


def test_goertzel_degenerate_blocks():
    assert goertzel_power(np.zeros(0, dtype=np.float32), 1000.0, FS) == 0.0
    assert goertzel_power(np.ones(1, dtype=np.float32), 1000.0, FS) == 0.0


def dtft_power(x: np.ndarray, f0: float, fs: int = FS) -> float:
    """Reference: the single DTFT bin at f0, 4*|X(f0)|**2/N**2, by direct dot product."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    X = np.dot(x, np.exp(-2j * np.pi * f0 * np.arange(n) / fs))
    return float(4.0 * abs(X) ** 2 / (n * n))


@pytest.mark.parametrize("f0", [1000.0, BEEPER, 2510.0])
@pytest.mark.parametrize("n", [240, 480, 960, 2048])
def test_goertzel_is_exactly_the_single_bin_at_f0(f0, n):
    # Pins the design: one bin at f0, no neighbouring bins summed in.  Any
    # contribution from f0 +- 50 Hz would show up as a mismatch here.
    rng = np.random.default_rng(5)
    blocks = [sine(f0, n=n, phase=0.7), sine(f0 + 37.0, n=n), sine(1000.0, n=n) + sine(BEEPER, n=n, amp=0.5),
              rng.normal(0, 0.1, n).astype(np.float32)]
    for b in blocks:
        assert goertzel_power(b, f0, FS) == pytest.approx(dtft_power(b, f0), rel=1e-9, abs=1e-15)
        assert Goertzel(f0, FS, n).power(b) == pytest.approx(dtft_power(b, f0), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("f0", [1000.0, BEEPER])
def test_goertzel_neighbour_bin_sum_was_rejected_for_good_reason(f0):
    # Records why the "f0 +- 50 Hz summed" idea from early planning is not
    # implemented: the sum only reads 0 dB when +-50 Hz sits on a DFT bin (N a
    # multiple of 960), so it is block-length dependent and cannot satisfy the
    # "0 dB regardless of block length" contract that the single bin meets.
    def three_bin(n: int) -> float:
        x = sine(f0, n=n, phase=0.7)
        return sum(goertzel_power(x, f0 + d, FS) for d in (-50.0, 0.0, 50.0))

    for n in (240, 480, 960):
        assert goertzel_power(sine(f0, n=n, phase=0.7), f0, FS) == pytest.approx(1.0, abs=0.05)
    assert db(three_bin(240)) > 4.0            # about +4.2 dB
    assert db(three_bin(480)) > 2.4            # about +2.6 dB
    assert db(three_bin(960)) == pytest.approx(0.0, abs=0.1)


# --- Goertzel class ------------------------------------------------------

def test_goertzel_class_matches_function():
    rng = np.random.default_rng(42)
    g = Goertzel(BEEPER, FS, N)
    blocks = [sine(BEEPER), sine(1000.0), sine(BEEPER, amp=0.2, phase=1.0)]
    blocks += [rng.normal(0, 0.1, N).astype(np.float32) for _ in range(5)]
    for b in blocks:
        assert g.power(b) == pytest.approx(goertzel_power(b, BEEPER, FS), rel=1e-9, abs=1e-15)


def test_goertzel_class_power_db_formula():
    g = Goertzel(1000.0, FS, N)
    assert g.power_db(sine(1000.0)) == pytest.approx(0.0, abs=1e-6)
    assert g.power_db(sine(1000.0, amp=0.1)) == pytest.approx(-20.0, abs=1e-6)
    assert g.power_db(np.zeros(N, dtype=np.float32)) == pytest.approx(-120.0)


def test_goertzel_class_set_frequency_and_readonly_f0():
    g = Goertzel(1000.0, FS, N)
    assert g.f0 == 1000.0
    strong_at_1k = g.power(sine(1000.0))
    g.set_frequency(BEEPER)
    assert g.f0 == BEEPER
    assert g.power(sine(1000.0)) < 1e-4 * strong_at_1k
    assert g.power(sine(BEEPER)) == pytest.approx(1.0, abs=0.05)
    with pytest.raises(AttributeError):
        g.f0 = 500.0  # type: ignore[misc]


def test_goertzel_class_handles_other_block_lengths():
    g = Goertzel(1000.0, FS, N)
    assert g.power(sine(1000.0, n=960)) == pytest.approx(1.0, abs=0.05)
    assert g.power(sine(1000.0, n=200)) == pytest.approx(goertzel_power(sine(1000.0, n=200), 1000.0, FS))
    assert g.power(np.zeros(0, dtype=np.float32)) == 0.0


# --- BandPass ------------------------------------------------------------

@pytest.mark.parametrize("f0", [1000.0, BEEPER])
def test_bandpass_passes_f0(f0):
    bp = BandPass(f0, FS)
    x = sine(f0, n=FS)                       # 1 s
    y = bp.process(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    half = FS // 2                           # ignore the start-up transient
    gain_db = 20 * np.log10(rms(y[half:]) / rms(x[half:]))
    assert abs(gain_db) < 0.5


@pytest.mark.parametrize("f0", [1000.0, BEEPER])
@pytest.mark.parametrize("offset", [1000.0, -700.0])
def test_bandpass_attenuates_tone_1khz_away(f0, offset):
    bp = BandPass(f0, FS)
    x = sine(f0 + offset, n=FS)
    y = bp.process(x)
    half = FS // 2
    atten_db = 20 * np.log10(rms(x[half:]) / rms(y[half:]))
    assert atten_db >= 20.0


def test_bandpass_state_carry_blockwise_equals_whole():
    rng = np.random.default_rng(7)
    x = (rng.normal(0, 0.2, FS) + sine(BEEPER, n=FS, amp=0.3)).astype(np.float32)
    whole = BandPass(BEEPER, FS).process(x)
    bp = BandPass(BEEPER, FS)
    blocks = [bp.process(x[i:i + N]) for i in range(0, FS, N)]
    blockwise = np.concatenate(blocks)
    assert blockwise.shape == whole.shape
    # State is carried exactly, so the two agree everywhere, not just after the first blocks.
    assert np.allclose(blockwise, whole, atol=1e-6)
    assert np.allclose(blockwise[5 * N:], whole[5 * N:], atol=1e-6)


def test_bandpass_without_state_carry_would_differ():
    # Sanity check that the previous test is meaningful: restarting the filter on
    # every block gives a different answer than the carried-state stream.
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.2, 4 * N).astype(np.float32)
    carried = BandPass(BEEPER, FS)
    stream = np.concatenate([carried.process(x[i:i + N]) for i in range(0, x.size, N)])
    restarted = np.concatenate([BandPass(BEEPER, FS).process(x[i:i + N]) for i in range(0, x.size, N)])
    assert not np.allclose(stream, restarted, atol=1e-4)


def test_bandpass_set_frequency_resets_state():
    x = sine(BEEPER, n=4 * N, amp=0.5)
    fresh = BandPass(BEEPER, FS).process(x[:N])
    bp = BandPass(BEEPER, FS)
    for i in range(0, x.size, N):
        bp.process(x[i:i + N])
    bp.set_frequency(BEEPER)                 # same design, state must be cleared
    assert np.allclose(bp.process(x[:N]), fresh, atol=1e-6)
    bp.set_frequency(1000.0)
    assert bp.f0 == 1000.0
    y = bp.process(sine(1000.0, n=FS))
    assert abs(20 * np.log10(rms(y[FS // 2:]) / rms(sine(1000.0, n=FS)[FS // 2:]))) < 0.5


def test_bandpass_empty_block_and_invalid_band():
    bp = BandPass(BEEPER, FS)
    out = bp.process(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,) and out.dtype == np.float32
    with pytest.raises(ValueError):
        BandPass(BEEPER, FS, half_width=0.0)


# --- spectrum -------------------------------------------------------------

def test_spectrum_shapes_and_axis():
    n = 2048
    freqs, pdb = spectrum(np.zeros(n, dtype=np.float32), FS)
    assert freqs.shape == (n // 2 + 1,)
    assert pdb.shape == freqs.shape
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(FS / 2)
    assert freqs[1] == pytest.approx(FS / n)
    assert np.all(pdb <= -100.0)             # silence


def test_spectrum_on_bin_full_scale_peaks_at_zero_db():
    n = 2048
    k = 43
    f = k * FS / n                           # exactly on a bin
    freqs, pdb = spectrum(sine(f, n=n), FS)
    assert int(np.argmax(pdb)) == k
    assert pdb[k] == pytest.approx(0.0, abs=0.2)


def test_spectrum_off_bin_peak_within_one_bin_and_near_zero_db():
    n = 2048
    bin_hz = FS / n
    freqs, pdb = spectrum(sine(BEEPER, n=n), FS)
    i = int(np.argmax(pdb))
    assert abs(freqs[i] - BEEPER) <= bin_hz
    assert -1.5 <= pdb[i] <= 0.1             # Hann scalloping loss is at most 1.4 dB


def test_spectrum_level_tracks_amplitude():
    n = 2048
    k = 43
    f = k * FS / n
    _, pdb = spectrum(sine(f, n=n, amp=0.1), FS)
    assert pdb[k] == pytest.approx(-20.0, abs=0.2)


def test_spectrum_empty_frame():
    freqs, pdb = spectrum(np.zeros(0, dtype=np.float32), FS)
    assert freqs.size == 0 and pdb.size == 0


# --- find_tone_frequency -------------------------------------------------

def test_find_tone_frequency_finds_beeper_in_noise():
    n = 2048
    rng = np.random.default_rng(3)
    frame = sine(BEEPER, n=n, amp=0.05) + rng.normal(0, 0.02, n).astype(np.float32)
    freqs, pdb = spectrum(frame, FS)
    peak_hz, prominence_db = find_tone_frequency(freqs, pdb)
    assert abs(peak_hz - BEEPER) <= FS / n   # within one bin
    assert abs(peak_hz - BEEPER) < 5.0       # parabolic refinement does much better
    assert prominence_db > 20.0


def test_find_tone_frequency_averaged_frames_is_accurate():
    n = 2048
    rng = np.random.default_rng(11)
    acc = None
    for _ in range(20):
        frame = sine(BEEPER, n=n, amp=0.02, phase=rng.uniform(0, 2 * np.pi)) \
            + rng.normal(0, 0.02, n).astype(np.float32)
        freqs, pdb = spectrum(frame, FS)
        acc = pdb if acc is None else acc + pdb
    peak_hz, prominence_db = find_tone_frequency(freqs, acc / 20)
    assert abs(peak_hz - BEEPER) < 3.0
    assert prominence_db > 15.0


def test_find_tone_frequency_ignores_out_of_band_peaks():
    n = 4096
    frame = sine(100.0, n=n, amp=0.9) + sine(BEEPER, n=n, amp=0.05)
    freqs, pdb = spectrum(frame, FS)
    peak_hz, _ = find_tone_frequency(freqs, pdb, fmin=300.0, fmax=8000.0)
    assert abs(peak_hz - BEEPER) < 5.0
    peak_hz, _ = find_tone_frequency(freqs, pdb, fmin=50.0, fmax=8000.0)
    assert abs(peak_hz - 100.0) < 5.0


def test_find_tone_frequency_prominence_definition():
    freqs = np.arange(0, 8001, 100.0)
    pdb = np.full(freqs.shape, -80.0)
    pdb[freqs == 2500.0] = -20.0
    peak_hz, prominence_db = find_tone_frequency(freqs, pdb)
    assert peak_hz == pytest.approx(2500.0)
    assert prominence_db == pytest.approx(60.0)


def test_find_tone_frequency_empty_band():
    freqs = np.arange(0, 8001, 100.0)
    pdb = np.zeros_like(freqs)
    assert find_tone_frequency(freqs, pdb, fmin=9000.0, fmax=10000.0) == (0.0, 0.0)


# --- synth_keyed_tone -----------------------------------------------------

PATTERN = [(True, 100.0), (False, 100.0), (True, 300.0), (False, 100.0)]


def block_powers(x: np.ndarray, f0: float) -> np.ndarray:
    g = Goertzel(f0, FS, N)
    return np.array([g.power(x[i:i + N]) for i in range(0, x.size - N + 1, N)])


def test_synth_length_dtype_and_range():
    x = synth_keyed_tone(PATTERN, 1000.0, FS)
    assert x.dtype == np.float32
    assert x.shape == (int(0.6 * FS),)
    assert np.max(np.abs(x)) <= 1.0
    assert synth_keyed_tone([], 1000.0, FS).shape == (0,)


def test_synth_on_off_energy_pattern():
    x = synth_keyed_tone(PATTERN, 1000.0, FS, amplitude=0.3)
    p = block_powers(x, 1000.0)
    assert p.shape == (60,)
    on = np.r_[0:10, 20:50]
    off = np.r_[10:20, 50:60]
    assert np.allclose(p[on], 0.09, rtol=1e-5)
    assert np.all(p[off] < 1e-10)


def test_synth_off_bin_tone_energy_pattern():
    x = synth_keyed_tone(PATTERN, BEEPER, FS, amplitude=0.3)
    p = block_powers(x, BEEPER)
    on = np.r_[0:10, 20:50]
    off = np.r_[10:20, 50:60]
    assert np.allclose(p[on], 0.09, rtol=0.05)
    assert np.all(p[off] < 1e-10)


def test_synth_noise_is_deterministic_and_has_requested_rms():
    a = synth_keyed_tone(PATTERN, 1000.0, FS, noise_rms=0.05, seed=1)
    b = synth_keyed_tone(PATTERN, 1000.0, FS, noise_rms=0.05, seed=1)
    c = synth_keyed_tone(PATTERN, 1000.0, FS, noise_rms=0.05, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    gap = a[int(0.1 * FS):int(0.2 * FS)]                    # pure noise
    assert rms(gap) == pytest.approx(0.05, rel=0.1)
    p = block_powers(a, 1000.0)
    assert np.all(p[np.r_[0:10, 20:50]] > 0.05)             # marks still dominate
    assert np.all(p[np.r_[10:20, 50:60]] < 0.01)


def test_synth_reverb_lengthens_marks_and_shortens_gaps():
    dry = block_powers(synth_keyed_tone(PATTERN, 1000.0, FS), 1000.0)
    wet = block_powers(synth_keyed_tone(PATTERN, 1000.0, FS, reverb_ms=40.0), 1000.0)
    on = np.r_[0:10, 20:50]
    # Instant attack: marks themselves are unchanged.
    assert np.allclose(wet[on], dry[on], rtol=1e-5)
    # Tail spills into the gap and decays monotonically.
    assert dry[10] < 1e-10
    assert wet[10] > 0.5 * wet[9]
    assert wet[10] > wet[11] > wet[12] > wet[13]
    assert wet[19] < 0.05 * wet[9]
    assert wet[19] > 1e-8                                    # but a tail is still there
    # A 50 % power threshold now sees a longer mark and a shorter gap.
    thr = 0.5 * wet[9]
    assert np.sum(wet[:20] > thr) > np.sum(dry[:20] > thr)
