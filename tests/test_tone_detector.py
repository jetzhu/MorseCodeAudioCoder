"""Tests for morse.tone_detector.ToneDetector (version 2).

Synthetic noise is never added in the dB domain: every "noise" series here is
Gaussian audio at a stated dBFS level run through ``morse.dsp.Goertzel`` in
480-sample blocks, so the single-bin statistics (exponential power, 5.6 dB
standard deviation in dB) are the real ones.  The two recorded fixtures are
driven the same way.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from morse.dsp import Goertzel, synth_keyed_tone
from morse.runs import Run
from morse.table import encode
from morse.tone_detector import ToneDetector

FS = 48000
BLOCK = 480
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOOPBACK = FIXTURES / "loopback_sos_1khz_15wpm.wav"
BEEPER = FIXTURES / "beeper_long_2491hz_1m.wav"

# Clean levels for the structural tests: 70 dB apart, both well above the
# -100 dB silence clamp.
OFF_DB = -90.0
ON_DB = -20.0
# Structural patterns open with this many OFF blocks so the 30-block warm-up
# is over before the first mark.
LEAD = 40


# ------------------------------------------------------------------ helpers


def feed(det: ToneDetector, values) -> list[Run]:
    """Feed every value through update() and collect all emitted runs in order."""
    out: list[Run] = []
    for v in values:
        out.extend(det.update(float(v)))
    return out


def levels(pattern: list[tuple[bool, int]], on_db: float = ON_DB, off_db: float = OFF_DB) -> list[float]:
    """Expand [(on, blocks), ...] into one clean dB value per block."""
    out: list[float] = []
    for on, blocks in pattern:
        out.extend([on_db if on else off_db] * blocks)
    return out


def runs_of(pattern: list[tuple[bool, int]], block_ms: float = 10.0) -> list[Run]:
    return [Run(on=on, blocks=blocks, block_ms=block_ms) for on, blocks in pattern]


def to_db(power: float) -> float:
    return 10.0 * math.log10(power + 1e-12)


def from_db(power_db: float) -> float:
    return 10.0 ** (power_db / 10.0)


def reads(level_db: float) -> float:
    """What floor_db/peak_db report for a level tracked exactly at ``level_db``.

    The properties use ``10*log10(x + 1e-12)`` like ``Goertzel.power_db``, so
    -90 dB reads -89.9957 dB.
    """
    return to_db(from_db(level_db))


def goertzel_series(x: np.ndarray, f0: float, fs: int = FS, block: int = BLOCK) -> list[float]:
    """Goertzel dB per block of a mono signal, straight from morse.dsp."""
    g = Goertzel(f0, fs, block)
    n = x.size // block
    return [g.power_db(b) for b in np.asarray(x, dtype=np.float32)[: n * block].reshape(n, block)]


def gaussian(dbfs: float, samples: int, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise with the given rms level in dBFS."""
    return rng.normal(0.0, 10.0 ** (dbfs / 20.0), samples)


def noise_series(dbfs: float, seconds: float, seed: int, f0: float = 2491.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(goertzel_series(gaussian(dbfs, int(seconds * FS), rng), f0))


def keyed_pattern(text: str, wpm: float, lead_ms: float, tail_ms: float) -> list[tuple[bool, float]]:
    """(on, ms) sequence for text: dit T, dah 3T, gaps 1T / 3T / 7T."""
    dit = 1200.0 / wpm
    pat: list[tuple[bool, float]] = [(False, lead_ms)]
    for ch in encode(text):
        if ch == ".":
            pat += [(True, dit), (False, dit)]
        elif ch == "-":
            pat += [(True, 3 * dit), (False, dit)]
        elif ch == " ":
            pat[-1] = (False, 3 * dit)
        elif ch == "/":
            pat[-1] = (False, 7 * dit)
    pat[-1] = (False, tail_ms)
    return pat


def sos_in_noise(
    noise_dbfs: float,
    seed: int,
    snr_db: float = 20.0,
    lead_ms: float = 1000.0,
    tail_ms: float = 1000.0,
    silence_s: float = 0.0,
    reverb_ms: float = 0.0,
    f0: float = 1000.0,
) -> list[float]:
    """Goertzel series of SOS at 15 WPM, ``snr_db`` above Gaussian noise.

    The tone's rms is ``snr_db`` above the noise rms.  ``silence_s`` seconds
    of digital zeros are prepended before the noise starts.
    """
    rng = np.random.default_rng(seed)
    amplitude = 10.0 ** ((noise_dbfs + snr_db) / 20.0) * math.sqrt(2.0)
    tone = synth_keyed_tone(keyed_pattern("SOS", 15.0, lead_ms, tail_ms), f0, FS,
                            amplitude=amplitude, reverb_ms=reverb_ms)
    x = tone + gaussian(noise_dbfs, tone.size, rng)
    if silence_s:
        x = np.concatenate([np.zeros(int(silence_s * FS)), x])
    return goertzel_series(x, f0)


def decode(runs: list[Run]) -> str:
    """Run a list of final runs through MorseDecoder and return the text."""
    from morse.decoder import MorseDecoder

    dec = MorseDecoder()
    text = "".join(dec.feed(r) for r in runs)
    text += dec.idle(float("inf"))
    return text.strip()


def on_starts(runs: list[Run]) -> list[tuple[int, int]]:
    """(start_block, blocks) of every ON run."""
    out: list[tuple[int, int]] = []
    t = 0
    for r in runs:
        if r.on:
            out.append((t, r.blocks))
        t += r.blocks
    return out


def fixture_series(path: Path, f0: float) -> list[float]:
    """Goertzel dB per 10 ms block of a fixture."""
    from scipy.io import wavfile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    assert np.issubdtype(data.dtype, np.integer), data.dtype
    x = (data.astype(np.float64) / float(2 ** (8 * data.dtype.itemsize - 1))).astype(np.float32)
    return goertzel_series(x, f0, fs, round(fs / 100))


def expected_lifts(snr: float) -> tuple[float, float]:
    """(hi, lo) offsets above the noise level for a tracked SNR, per the contract.

    Noise-referenced lifts at low SNR; at high SNR the OFF threshold follows
    the signal (``S - release_db``) and hi sits ``hysteresis_db`` above it.
    """
    lift_on = min(30.0, max(12.0, 0.6 * snr))
    lift_off = min(24.0, max(6.0, 0.4 * snr))
    lo = max(lift_off, snr - 12.0)
    hi = max(lift_on, lo + 3.0)
    return hi, lo


# ----------------------------------------------------------------- defaults


def test_defaults_match_the_contract():
    det = ToneDetector()
    assert det.block_ms == 10.0
    assert (det.on_frac, det.off_frac) == (0.6, 0.4)
    assert det.min_run_blocks == 2
    assert (det.min_lift_db, det.max_lift_db, det.min_off_lift_db) == (12.0, 30.0, 6.0)
    # Equal alphas keep N an unbiased linear mean (module docstring).
    assert det.noise_alpha_up == det.noise_alpha_down == 0.1
    assert det.signal_alpha == 0.1
    assert det.signal_decay_db == 0.1
    assert det.silence_floor_db == -100.0
    assert det.warmup_blocks == 30
    assert det.max_on_blocks == 500
    assert det.warming_up is True
    assert det.state is False
    assert det.current_run == Run(on=False, blocks=0)


def test_pipeline_detector_uses_the_defaults():
    # The contract says Pipeline constructs ToneDetector() with the defaults
    # and overrides no threshold parameter.
    pipeline = pytest.importorskip("morse.pipeline")
    pipe = pipeline.Pipeline()
    ref = ToneDetector(block_ms=pipe.block_ms)
    for name in (
        "block_ms", "on_frac", "off_frac", "min_run_blocks", "min_lift_db", "max_lift_db",
        "min_off_lift_db", "noise_alpha_up", "noise_alpha_down", "signal_alpha",
        "signal_decay_db", "silence_floor_db", "warmup_blocks", "max_on_blocks",
    ):
        assert getattr(pipe.detector, name) == getattr(ref, name), name


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(block_ms=0.0),
        dict(on_frac=0.5, off_frac=0.6),
        dict(on_frac=1.2),
        dict(off_frac=-0.1),
        dict(min_run_blocks=0),
        dict(min_lift_db=-1.0),
        dict(max_lift_db=10.0),  # below min_lift_db
        dict(min_off_lift_db=-1.0),
        dict(noise_alpha_up=0.0),
        dict(noise_alpha_down=1.5),
        dict(signal_alpha=-0.1),
        dict(signal_decay_db=-0.1),
        dict(warmup_blocks=-1),
        dict(max_on_blocks=0),
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ToneDetector(**kwargs)


# ----------------------------------------------------------- level tracking


def test_first_block_primes_noise_and_signal_levels():
    det = ToneDetector()
    assert det.update(-50.0) == []
    assert det.floor_db == pytest.approx(-50.0, abs=1e-6)
    assert det.peak_db == pytest.approx(-50.0, abs=1e-6)
    # snr 0: lifts at their minimums
    assert det.threshold_hi_db == pytest.approx(-38.0, abs=1e-6)
    assert det.threshold_lo_db == pytest.approx(-44.0, abs=1e-6)
    assert det.state is False
    assert det.warming_up is True
    assert det.current_run == Run(on=False, blocks=1)


def test_warm_up_forces_off_and_smooths_with_alpha_0_3():
    det = ToneDetector()
    det.update(-90.0)
    n = from_db(-90.0)
    # every warm-up block, above or below N, uses alpha 0.3
    det.update(-80.0)
    n += 0.3 * (from_db(-80.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)
    det.update(-100.0)
    n += 0.3 * (from_db(-100.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)
    # a loud tone from the very start is learned as noise while warming up
    for i in range(3, 30):
        assert det.warming_up is True
        assert det.update(ON_DB) == []
        assert det.state is False
    assert det.warming_up is False
    assert det.floor_db == pytest.approx(ON_DB, abs=0.1)
    assert det.peak_db == pytest.approx(det.floor_db)
    assert det.update(ON_DB) == []
    assert det.state is False  # not above N + 12 dB
    assert det.flush() == runs_of([(False, 31)])


def test_warm_up_settles_on_a_fade_in_before_any_verdict():
    # A stream opening with a fade-in from digital silence (both fixtures do)
    # must land on the real level within the warm-up and not trip ON after it.
    fade = np.linspace(-119.0, -82.0, 25)
    det = ToneDetector()
    feed(det, fade)
    feed(det, [-82.0] * 5)
    assert det.warming_up is False
    assert det.floor_db == pytest.approx(-82.0, abs=1.0)
    for v in [-80.0, -84.0, -78.0, -86.0, -79.0] * 4:
        det.update(v)
        assert det.state is False
    assert det.flush() == runs_of([(False, 50)])


def test_noise_level_never_drops_below_the_silence_floor():
    det = ToneDetector()
    feed(det, [-120.0] * 60)  # digital silence, well past the warm-up
    assert det.floor_db == pytest.approx(reads(-100.0), abs=1e-9)
    assert -100.0 <= det.floor_db <= -99.9
    assert det.threshold_hi_db == pytest.approx(det.floor_db + 12.0)
    assert det.threshold_lo_db == pytest.approx(det.floor_db + 6.0)
    # a tone just above the clamped ON threshold is heard at once ...
    det.update(-87.0)
    assert det.state is True
    # ... one just below it is not (and, being OFF, it lifts the noise level)
    det = ToneDetector()
    feed(det, [-120.0] * 60)
    det.update(-89.0)
    assert det.state is False
    assert det.floor_db > reads(-100.0)


def test_custom_silence_floor_is_honoured():
    det = ToneDetector(silence_floor_db=-80.0)
    feed(det, [-120.0] * 40)
    assert det.floor_db == pytest.approx(-80.0, abs=0.05)
    assert det.threshold_hi_db == pytest.approx(det.floor_db + 12.0)


def test_noise_level_is_updated_only_by_off_blocks():
    det = ToneDetector()
    feed(det, [OFF_DB] * LEAD)
    floor = det.floor_db
    feed(det, [ON_DB] * 50)
    assert det.state is True
    assert det.floor_db == floor  # ON blocks leave N alone
    det.update(OFF_DB)
    assert det.floor_db == floor  # p == N: no change either way
    det.update(-85.0)
    assert det.floor_db > floor


def test_signal_level_follows_on_blocks_with_signal_alpha():
    det = ToneDetector()
    feed(det, [OFF_DB] * LEAD)
    s = from_db(OFF_DB)  # S starts at N, which sits exactly at the constant input
    for _ in range(3):
        det.update(ON_DB)
        s += 0.1 * (from_db(ON_DB) - s)
        assert det.state is True
        assert det.peak_db == pytest.approx(to_db(s), abs=1e-6)
    # first ON block puts S 10 dB under a tone that starts at N + 70 dB
    assert det.peak_db < ON_DB


def test_signal_level_decays_toward_noise_while_off_and_never_below():
    det = ToneDetector()
    feed(det, [OFF_DB] * LEAD)
    feed(det, [ON_DB] * 100)
    peak = det.peak_db
    assert peak == pytest.approx(ON_DB, abs=0.01)
    feed(det, [OFF_DB] * 100)
    assert det.state is False
    assert det.peak_db == pytest.approx(peak - 100 * det.signal_decay_db, abs=0.01)
    feed(det, [OFF_DB] * 3000)
    assert det.peak_db == pytest.approx(det.floor_db, abs=1e-9)
    assert det.peak_db >= det.floor_db


@pytest.mark.parametrize(
    "tone_db, lift_on, lift_off",
    [
        (-77.0, 12.0, 6.0),   # snr 13: noise-referenced minimums (a tone at
                              # exactly N + 12 dB is not *above* the threshold)
        (-75.0, 12.0, 6.0),   # snr 15: 0.4*snr == 6, still the minimum
        (-60.0, 21.0, 18.0),  # snr 30: lo = S - 12 = N + 18 beats 0.4*snr; hi = lo + 3
        (-45.0, 36.0, 33.0),  # snr 45: signal-referenced, lo = N + 33
        (-30.0, 51.0, 48.0),  # snr 60
        (-10.0, 71.0, 68.0),  # snr 80: thresholds keep following the signal
    ],
)
def test_lifts_follow_the_tracked_snr_with_min_and_max(tone_db, lift_on, lift_off):
    det = ToneDetector()
    feed(det, [-90.0] * LEAD)
    feed(det, [tone_db] * 150)  # S converges to the tone (0.9**150 ~ 1e-7)
    assert det.state is True
    assert det.peak_db == pytest.approx(tone_db, abs=0.01)
    assert det.floor_db == pytest.approx(reads(-90.0), abs=1e-9)
    assert det.threshold_hi_db == pytest.approx(det.floor_db + lift_on, abs=0.02)
    assert det.threshold_lo_db == pytest.approx(det.floor_db + lift_off, abs=0.02)


def test_noise_tracking_is_a_linear_ema_in_both_directions():
    det = ToneDetector()
    feed(det, [-90.0] * LEAD)
    n = from_db(-90.0)  # a constant input leaves the EMA exactly at that power
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-9)
    det.update(-80.0)  # above N: noise_alpha_up
    n += 0.1 * (from_db(-80.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)
    det.update(-100.0)  # below N: noise_alpha_down (same value by default)
    n += 0.1 * (from_db(-100.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)

    # The direction rule is honoured when the alphas differ.
    det = ToneDetector(noise_alpha_up=0.02, noise_alpha_down=0.2)
    feed(det, [-90.0] * LEAD)
    n = from_db(-90.0)
    det.update(-80.0)
    n += 0.02 * (from_db(-80.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)
    det.update(-100.0)
    n += 0.2 * (from_db(-100.0) - n)
    assert det.floor_db == pytest.approx(to_db(n), abs=1e-6)


def test_thresholds_are_always_consistent_with_the_levels():
    rng = np.random.default_rng(3)
    det = ToneDetector()
    values = np.concatenate([
        np.linspace(-119.0, -80.0, 20),
        -80.0 + 5.0 * rng.standard_normal(200),
        [-30.0] * 40, [-80.0] * 40, [-50.0] * 20, [-95.0] * 40, [-20.0] * 600,
    ])
    for v in values:
        det.update(float(v))
        snr = det.peak_db - det.floor_db
        assert snr >= 0.0
        lift_on, lift_off = expected_lifts(snr)
        assert det.threshold_hi_db == pytest.approx(det.floor_db + lift_on, abs=1e-6)
        assert det.threshold_lo_db == pytest.approx(det.floor_db + lift_off, abs=1e-6)
        assert det.floor_db <= det.threshold_lo_db < det.threshold_hi_db <= det.peak_db + 30.0


def test_stuck_on_timeout_ends_the_run_and_adopts_the_level():
    det = ToneDetector()
    got = feed(det, [OFF_DB] * LEAD)
    # 499 ON blocks: nothing final yet beyond the lead
    got += feed(det, [ON_DB] * 499)
    assert got == runs_of([(False, LEAD)])
    assert det.state is True
    # block 500 is still ON (the run reaches max_on_blocks) ...
    assert det.update(ON_DB) == []
    assert det.state is True
    assert det.current_run == Run(on=True, blocks=500)
    # ... block 501 is forced OFF and N jumps to S
    assert det.update(ON_DB) == []
    assert det.state is False
    assert det.floor_db == pytest.approx(det.peak_db)
    assert det.floor_db == pytest.approx(ON_DB, abs=0.01)
    # the ON run is final once the OFF run after it reaches min_run_blocks,
    # i.e. max_on_blocks + min_run_blocks blocks after it began
    assert det.update(ON_DB) == runs_of([(True, 500)])
    # the same level from now on is noise, never ON again
    got = feed(det, [ON_DB] * 200)
    assert got == []
    assert det.state is False
    assert det.flush() == runs_of([(False, 202)])


def test_stuck_on_timeout_recovers_when_the_level_drops_again():
    det = ToneDetector(max_on_blocks=100)
    feed(det, [OFF_DB] * LEAD)
    feed(det, [ON_DB] * 105)
    assert det.state is False and det.floor_db == pytest.approx(ON_DB, abs=0.01)
    # back to quiet: N follows down (0.1 per block in the linear domain, so
    # roughly 0.46 dB per block) and a tone is heard again once it is 12 dB up
    feed(det, [OFF_DB] * 200)
    assert det.floor_db == pytest.approx(OFF_DB, abs=0.5)
    det.update(ON_DB)
    assert det.state is True


def test_reset_restarts_the_warm_up_but_flush_does_not():
    det = ToneDetector()
    feed(det, levels([(False, LEAD), (True, 5)]))
    assert det.warming_up is False
    det.flush()
    assert det.warming_up is False
    assert det.floor_db == pytest.approx(reads(OFF_DB), abs=1e-9)  # levels survive a flush
    det.update(ON_DB)
    assert det.state is True
    det.reset()
    assert det.warming_up is True
    assert det.state is False
    assert det.current_run.blocks == 0
    assert det.flush() == []
    det.update(-30.0)
    assert det.floor_db == pytest.approx(reads(-30.0), abs=1e-9)
    assert det.peak_db == pytest.approx(reads(-30.0), abs=1e-9)
    assert det.state is False  # warming up again


def test_non_finite_power_is_treated_as_silence():
    det = ToneDetector()
    feed(det, [OFF_DB] * LEAD)
    det.update(float("nan"))
    det.update(float("inf"))
    det.update(float("-inf"))
    assert math.isfinite(det.floor_db) and math.isfinite(det.threshold_hi_db)
    assert det.state is False


# ---------------------------------------------------------- run generation


def test_clean_step_sequence_gives_exact_runs():
    pattern = [(False, LEAD), (True, 5), (False, 3), (True, 8), (False, 10)]
    det = ToneDetector()
    got = feed(det, levels(pattern))
    got += det.flush()
    assert got == runs_of(pattern)
    assert all(r.block_ms == 10.0 for r in got)


def test_block_ms_is_propagated_into_runs():
    det = ToneDetector(block_ms=5.0)
    pattern = [(False, LEAD), (True, 6), (False, 4)]
    got = feed(det, levels(pattern))
    got += det.flush()
    assert got == runs_of(pattern, block_ms=5.0)
    assert got[1].ms == 30.0


def test_one_block_dropout_inside_on_run_is_debounced():
    values = levels([(False, LEAD), (True, 5), (False, 1), (True, 5), (False, 10)])
    det = ToneDetector(min_run_blocks=2)
    got = feed(det, values)
    got += det.flush()
    assert got == runs_of([(False, LEAD), (True, 11), (False, 10)])


def test_one_block_spike_during_off_does_not_create_run():
    values = levels([(False, LEAD), (True, 1), (False, 10)])
    det = ToneDetector(min_run_blocks=2)
    feed(det, values[:LEAD])
    assert det.state is False
    # The spike does flip the raw verdict for one block...
    det.update(values[LEAD])
    assert det.state is True
    # ...but debouncing folds it back into one OFF run.
    got = feed(det, values[LEAD + 1:])
    got += det.flush()
    assert got == runs_of([(False, LEAD + 11)])


def test_current_run_tracks_in_progress_run_and_merges():
    det = ToneDetector(min_run_blocks=2)
    assert det.current_run.blocks == 0
    feed(det, [OFF_DB] * LEAD)
    assert det.current_run == Run(on=False, blocks=LEAD)
    feed(det, [ON_DB] * 3)
    assert det.current_run == Run(on=True, blocks=3)
    det.update(OFF_DB)
    assert det.current_run == Run(on=False, blocks=1)
    det.update(ON_DB)  # the 1-block OFF merges back into the ON run
    assert det.current_run == Run(on=True, blocks=5)


@pytest.mark.parametrize("min_run_blocks", [2, 3, 4])
def test_runs_are_not_returned_before_they_are_final(min_run_blocks: int):
    det = ToneDetector(min_run_blocks=min_run_blocks)
    assert feed(det, [OFF_DB] * LEAD) == []
    got = feed(det, [ON_DB] * 50)
    # the leading OFF run comes out once the ON run is long enough
    assert got == runs_of([(False, LEAD)])
    # a long ON run is still not final while the following OFF is too short
    for _ in range(min_run_blocks - 1):
        assert det.update(OFF_DB) == []
        assert det.current_run.on is False
    assert det.update(OFF_DB) == runs_of([(True, 50)])
    assert det.current_run == Run(on=False, blocks=min_run_blocks)


def test_flush_emits_the_tail_and_clears_pending():
    det = ToneDetector()
    assert det.flush() == []
    got = feed(det, levels([(False, LEAD), (True, 5)]))
    assert got == runs_of([(False, LEAD)])
    assert det.flush() == runs_of([(True, 5)])
    assert det.flush() == []
    assert det.current_run.blocks == 0

    # A short current run behind a pending run: both come out, oldest first.
    det = ToneDetector()
    feed(det, levels([(False, LEAD), (True, 5), (False, 1)]))
    assert det.flush() == runs_of([(True, 5), (False, 1)])

    # Levels survive a flush, so the stream can continue.
    assert det.floor_db == pytest.approx(reads(OFF_DB), abs=1e-9)
    got = feed(det, levels([(False, 4), (True, 3), (False, 2)]))
    got += det.flush()
    assert got == runs_of([(False, 4), (True, 3), (False, 2)])


def test_hysteresis_power_between_lo_and_hi_does_not_toggle():
    det = ToneDetector()
    got = feed(det, levels([(False, LEAD), (True, 20), (False, 20)]))
    assert det.state is False
    # While OFF, a block inside the band leaves the verdict OFF.  (It counts
    # as noise, so the thresholds move; check against the band of each block.)
    for _ in range(5):
        lo, hi = det.threshold_lo_db, det.threshold_hi_db
        v = 0.5 * (lo + hi)
        got += det.update(v)
        assert det.state is False
    # Switch ON, then hover inside the band: it must stay ON.
    got += feed(det, [ON_DB] * 5)
    assert det.state is True
    for _ in range(15):
        lo, hi = det.threshold_lo_db, det.threshold_hi_db
        v = 0.5 * (lo + hi)
        got += det.update(v)
        assert det.state is True
    got += det.flush()
    assert got == runs_of([(False, LEAD), (True, 20), (False, 25), (True, 20)])


def test_min_run_blocks_one_disables_debounce():
    pattern = [(False, LEAD), (True, 1), (False, 1), (True, 2), (False, 3)]
    det = ToneDetector(min_run_blocks=1)
    got = feed(det, levels(pattern))
    got += det.flush()
    assert got == runs_of(pattern)


def test_leading_short_run_joins_the_next_run():
    # One quiet block, then tone: the 1-block OFF has no earlier neighbour,
    # so it is absorbed into the ON run rather than emitted on its own.
    # (No warm-up, so the very first blocks are judged.)
    det = ToneDetector(min_run_blocks=2, warmup_blocks=0)
    got = feed(det, levels([(False, 1), (True, 10), (False, 10)]))
    got += det.flush()
    assert got == runs_of([(True, 11), (False, 10)])


def test_emitted_runs_alternate_conserve_blocks_and_respect_min_length():
    # Random keying only 10 dB above Gaussian noise, through the real
    # Goertzel: glitchy enough to exercise every merge path.
    rng = np.random.default_rng(1234)
    pattern: list[tuple[bool, float]] = [(False, 500.0)]
    for _ in range(40):
        pattern.append((True, float(rng.integers(1, 12)) * 10.0))
        pattern.append((False, float(rng.integers(1, 12)) * 10.0))
    amplitude = 10.0 ** ((-60.0 + 10.0) / 20.0) * math.sqrt(2.0)
    x = synth_keyed_tone(pattern, 1000.0, FS, amplitude=amplitude)
    x = x + gaussian(-60.0, x.size, rng)
    values = goertzel_series(x, 1000.0)
    for min_run_blocks in (2, 3):
        det = ToneDetector(min_run_blocks=min_run_blocks)
        got = feed(det, values)
        tail = det.flush()
        assert all(r.blocks >= min_run_blocks for r in got), "update() emitted a short run"
        assert all(r.blocks >= 1 for r in tail)
        everything = got + tail
        assert sum(r.blocks for r in everything) == len(values)
        assert all(a.on != b.on for a, b in zip(everything, everything[1:]))
        assert sum(r.on for r in everything) >= 10  # the keying was heard


# ------------------------------------------------------ noise, no chatter


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dbfs", [-110.0, -90.0, -70.0, -50.0, -40.0])
def test_a_minute_of_tone_free_noise_never_switches_on(dbfs: float, seed: int):
    # Acceptance: 60 s of Gaussian noise at any fixed level from -110 to
    # -40 dBFS produces zero ON runs with the defaults.
    values = noise_series(dbfs, 60.0, seed)
    assert values.size == 6000
    det = ToneDetector()
    for v in values:
        det.update(float(v))
        assert det.state is False, f"switched ON at {v:.1f} dB, hi {det.threshold_hi_db:.1f}"
    assert det.flush() == runs_of([(False, 6000)])
    settled = values[30:]
    assert det.threshold_hi_db > settled.max()
    linear_mean_db = to_db(float(np.mean(10.0 ** (settled / 10.0))))
    if linear_mean_db > -98.0:
        # above the silence clamp N is the unbiased linear mean of the noise
        assert det.floor_db == pytest.approx(linear_mean_db, abs=1.5)
    else:
        assert det.floor_db == pytest.approx(-100.0, abs=0.1)


# -------------------------------------------------------- keyed tone in noise


def assert_sos_runs(runs: list[Run], dit_ms: float = 80.0, tol_ms: float = 20.0) -> None:
    """Nine ON runs of dit, dit, dit, dah, dah, dah, dit, dit, dit."""
    marks = [r.ms for r in runs if r.on]
    assert len(marks) == 9, marks
    for i, ms in enumerate(marks):
        want = 3 * dit_ms if 3 <= i < 6 else dit_ms
        assert abs(ms - want) <= tol_ms, marks


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dbfs", [-110.0, -90.0, -70.0, -50.0, -40.0])
def test_a_tone_20_db_above_the_noise_decodes(dbfs: float, seed: int):
    det = ToneDetector()
    runs = feed(det, sos_in_noise(dbfs, seed, snr_db=20.0))
    runs += det.flush()
    assert runs[0].on is False and runs[0].ms >= 900.0  # no ON during the lead
    assert_sos_runs(runs)
    assert decode(runs) == "SOS"


@pytest.mark.parametrize("reverb_ms", [5.0, 10.0])
def test_a_reverberant_tone_20_db_above_the_noise_decodes(reverb_ms: float):
    det = ToneDetector()
    runs = feed(det, sos_in_noise(-60.0, 2, snr_db=20.0, reverb_ms=reverb_ms))
    runs += det.flush()
    assert len([r for r in runs if r.on]) == 9
    assert decode(runs) == "SOS"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dbfs", [-110.0, -90.0, -80.0])
def test_silence_then_noise_then_tone_decodes_without_early_runs(dbfs: float, seed: int):
    # Acceptance: 2 s of digital silence, then noise, then the keyed tone;
    # no ON run during the silence or the noise.  Levels up to -80 dBFS put
    # the single-bin noise at or under the -100 dB silence clamp; louder noise
    # arriving after a silence longer than the warm-up is covered by the
    # recovery test below.
    values = sos_in_noise(dbfs, seed, snr_db=20.0, lead_ms=2000.0, silence_s=2.0)
    det = ToneDetector()
    for i, v in enumerate(values[:400]):
        det.update(v)
        assert det.state is False, f"ON at block {i} ({v:.1f} dB)"
    assert det.warming_up is False
    runs = feed(det, values[400:])
    runs += det.flush()
    assert runs[0].on is False and runs[0].blocks >= 400
    assert_sos_runs(runs)
    assert decode(runs) == "SOS"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_loud_noise_after_a_long_silence_recovers_within_two_timeouts(seed: int):
    # Noise 40 dB above the silence clamp arriving 2 s in cannot be told from
    # a tone at its onset; the detector must nevertheless be OFF again within
    # 2 * max_on_blocks and then hear the tone normally.
    values = sos_in_noise(-50.0, seed, snr_db=20.0, lead_ms=20000.0, silence_s=2.0)
    noise_start, tone_start = 200, 2200
    det = ToneDetector()
    runs = feed(det, values)
    runs += det.flush()
    starts = on_starts(runs)
    assert all(start >= noise_start for start, _ in starts)
    early = [(s, b) for s, b in starts if s < tone_start]
    assert all(s + b <= noise_start + 2 * det.max_on_blocks for s, b in early), early
    # the ON phase may fragment into a few runs; in total it stays under two timeouts
    assert sum(b for _, b in early) <= 2 * det.max_on_blocks, early
    tone_runs = [r for r, (s, _) in zip((r for r in runs if r.on), starts) if s >= tone_start]
    assert len(tone_runs) == 9
    assert_sos_runs([Run(on=False, blocks=100)] + tone_runs)


# ------------------------------------------------------------ noise jumps


def test_a_clean_25_db_step_gives_one_timed_out_run():
    # The contract's statement on a clean series: at most one ON run, ending
    # within max_on_blocks + min_run_blocks blocks of the jump.
    det = ToneDetector()
    got = feed(det, [-80.0] * 200)
    got += feed(det, [-55.0] * 2000)
    got += det.flush()
    on = [r for r in got if r.on]
    assert len(on) == 1
    assert on[0].blocks == det.max_on_blocks
    assert got == runs_of([(False, 200), (True, 500), (False, 1500)])
    assert det.floor_db == pytest.approx(-55.0, abs=0.01)


@pytest.mark.parametrize("seed", range(6))
def test_a_25_db_noise_jump_recovers_and_stays_off(seed: int):
    # Real Goertzel noise jumping 25 dB dips below the OFF threshold now and
    # then, so the ON phase may fragment into a few runs; the total ON time
    # stays under one timeout and nothing is ON after two.
    rng = np.random.default_rng(seed)
    x = np.concatenate([gaussian(-70.0, 10 * FS, rng), gaussian(-45.0, 30 * FS, rng)])
    values = np.array(goertzel_series(x, 2491.0))
    det = ToneDetector()
    runs = feed(det, values)
    runs += det.flush()
    starts = on_starts(runs)
    assert starts, "the jump must at least be noticed"
    assert all(s >= 1000 for s, _ in starts), starts  # nothing before the jump
    assert max(s + b for s, b in starts) <= 1000 + 2 * det.max_on_blocks, starts
    assert sum(b for _, b in starts) <= 2 * det.max_on_blocks, starts
    linear_mean_db = to_db(float(np.mean(10.0 ** (values[3000:] / 10.0))))
    assert det.floor_db == pytest.approx(linear_mean_db, abs=3.0)


# --------------------------------------------- recorded fixtures, defaults only


def test_default_detector_finds_nine_marks_in_the_loopback_fixture():
    # 15 WPM SOS keyed as 80 / 240 ms marks and 80 / 240 ms gaps, played
    # through the laptop speakers.  Version 2 tracks the room's release tail
    # as the OFF level, so the marks come out within about a block of the
    # keyed lengths (dits 90-120 ms, dahs 250 ms) rather than the 110 / 270 ms
    # the version 1 thresholds measured.
    det = ToneDetector()
    got = feed(det, fixture_series(LOOPBACK, 1000.0))
    got += det.flush()
    marks = [r.ms for r in got if r.on]
    assert len(marks) == 9, marks
    dits = marks[0:3] + marks[6:9]
    dahs = marks[3:6]
    assert all(abs(ms - 100.0) <= 20.0 for ms in dits), dits
    assert all(abs(ms - 260.0) <= 20.0 for ms in dahs), dahs
    gaps = [r.ms for r in got[1:-1] if not r.on]
    assert len(gaps) == 8, gaps
    intra = gaps[0:2] + gaps[3:5] + gaps[6:8]
    letter = [gaps[2], gaps[5]]
    assert all(abs(ms - 55.0) <= 20.0 for ms in intra), intra
    assert all(abs(ms - 220.0) <= 20.0 for ms in letter), letter
    # no chatter during the fade-in and room noise before the message
    assert got[0].on is False
    assert abs(got[0].ms - 1370.0) <= 60.0, got[0]
    assert got[-1].on is False
    assert decode(got) == "SOS"


def test_default_detector_finds_four_tones_in_the_beeper_fixture():
    det = ToneDetector()
    got = feed(det, fixture_series(BEEPER, 2491.0))
    got += det.flush()
    on = [r.ms for r in got if r.on]
    assert len(on) == 4, on
    for ms, want in zip(on, (3700.0, 490.0, 570.0, 2020.0)):
        assert abs(ms - want) <= 100.0, on
    gaps = [r.ms for r in got[1:-1] if not r.on]
    assert len(gaps) == 3, gaps
    for ms, want in zip(gaps, (370.0, 250.0, 300.0)):
        assert abs(ms - want) <= 100.0, gaps
    # the first ON edge is near 0.69 s, not at the fade-in from digital silence
    assert got[0].on is False
    assert abs(got[0].ms - 690.0) <= 60.0, got[0]
    assert got[-1].on is False
