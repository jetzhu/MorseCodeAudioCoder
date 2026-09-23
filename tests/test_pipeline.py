"""Tests for morse.pipeline on synthetic audio, plus the morse.app command line."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from morse.dsp import synth_keyed_tone
from morse.pipeline import BlockResult, Pipeline, decode_samples, decode_wav, load_wav
from morse.runs import Run
from morse.table import encode
from morse.tone_detector import ToneDetector

ROOT = Path(__file__).resolve().parents[1]

FS = 48000
N = 480
NOISE_RMS = 1e-3          # -60 dBFS white noise
WPM = 8                   # the speed tools/beep_sender.py is meant to run at

REVERB_MS = 10.0
"""1/e time constant (amplitude) of the synthetic room tail.

The detector's OFF threshold sits at most 24 dB above the noise level, so at
the 60 dB signal-to-noise ratio of these tests a release is seen only once
the tail has decayed by about 36 dB, i.e. four time constants.  10 ms gives a
35-40 ms offset between keyed and measured runs, which is what the real
loopback capture showed (docs/PLAN.md section 4 measured 30 ms).  A 30 ms
tail would mean an 85 ms offset, longer than any gap at 15 WPM.
"""

Pattern = list[tuple[bool, float]]


# ------------------------------------------------------------------ helpers


def keyed_pattern(text: str, wpm: float, lead_ms: float = 500.0, tail_ms: float = 1000.0) -> Pattern:
    """Standard-timing ``(on, ms)`` keying for ``text``: dit 1, dah 3, gaps 1/3/7."""
    dit = 1200.0 / wpm
    pattern: Pattern = [(False, lead_ms)] if lead_ms > 0 else []
    for wi, word in enumerate(encode(text).split(" / ")):
        if wi:
            pattern.append((False, 7 * dit))
        for li, letter in enumerate(word.split(" ")):
            if li:
                pattern.append((False, 3 * dit))
            for si, symbol in enumerate(letter):
                if si:
                    pattern.append((False, dit))
                pattern.append((True, dit if symbol == "." else 3 * dit))
    if tail_ms > 0:
        pattern.append((False, tail_ms))
    return pattern


def synth(text: str, f0: float, amplitude: float = 0.1, wpm: float = WPM,
          reverb_ms: float = REVERB_MS, noise_rms: float = NOISE_RMS, **kwargs: float) -> np.ndarray:
    return synth_keyed_tone(keyed_pattern(text, wpm, **kwargs), f0, FS,
                            amplitude=amplitude, noise_rms=noise_rms, reverb_ms=reverb_ms)


def blocks_of(samples: np.ndarray) -> list[np.ndarray]:
    return [samples[i:i + N].copy() for i in range(0, samples.size - N + 1, N)]


def run_pipeline(pipe: Pipeline, samples: np.ndarray) -> tuple[str, list[Run], list[BlockResult], str]:
    """Feed ``samples`` in 480-sample blocks, then flush.

    Returns ``(text emitted, all runs in order, per-block results, flush text)``.
    """
    emitted = ""
    runs: list[Run] = []
    results: list[BlockResult] = []
    for block in blocks_of(samples):
        result = pipe.process_block(block)
        results.append(result)
        emitted += result.new_text
        runs.extend(result.runs)
    flushed = pipe.flush()
    runs.extend(pipe.last_flush_runs)
    return emitted + flushed, runs, results, flushed


def _write(path: Path, fs: int, data: np.ndarray) -> str:
    wavfile.write(str(path), fs, data)
    return str(path)


# ------------------------------------------------------------ decoding text


@pytest.mark.parametrize("amplitude", [0.1, 0.03])
def test_sos_at_1khz_with_noise_and_reverb(amplitude):
    x = synth("SOS", 1000.0, amplitude=amplitude)
    pipe = Pipeline(fs=FS, block_size=N, f0=1000.0)
    text, runs, _, _ = run_pipeline(pipe, x)
    assert text.strip() == "SOS"
    assert pipe.text == text
    assert sum(1 for r in runs if r.on) == 9
    assert pipe.decoder.unknown_count == 0


@pytest.mark.parametrize("amplitude", [0.1, 0.03])
def test_hello_world_at_beeper_frequency(amplitude):
    x = synth("HELLO WORLD", 2491.0, amplitude=amplitude)
    pipe = Pipeline(fs=FS, block_size=N, f0=2491.0)
    text, runs, _, _ = run_pipeline(pipe, x)
    assert text.strip() == "HELLO WORLD"
    assert pipe.decoder.letter_count == 10
    assert pipe.decoder.unknown_count == 0
    # Reverb lengthens marks and shortens gaps; the decoder's offset undoes it
    # and the dit estimate lands on the keyed 150 ms.
    assert pipe.decoder.offset_ms > 0
    assert pipe.decoder.dit_ms == pytest.approx(1200.0 / WPM, abs=20.0)


def test_sos_at_15_wpm_still_has_every_gap():
    # 80 ms dits: the intra-letter gaps survive the release lag with room to spare.
    x = synth("SOS", 1000.0, wpm=15)
    text, runs = decode_samples(x, FS, 1000.0)
    assert text.strip() == "SOS"
    assert sum(1 for r in runs if r.on) == 9
    assert min(r.ms for r in runs if not r.on) >= 20.0


def test_hello_world_with_wpm_hint_and_fixed_speed():
    x = synth("HELLO WORLD", 2491.0)
    text, _ = decode_samples(x, FS, 2491.0, wpm=WPM)
    assert text.strip() == "HELLO WORLD"
    pipe = Pipeline(fs=FS, f0=2491.0, wpm=WPM, adaptive=False)
    text, _, _, _ = run_pipeline(pipe, x)
    assert text.strip() == "HELLO WORLD"
    assert pipe.decoder.dit_ms == pytest.approx(1200.0 / WPM)


def test_default_pipeline_is_48k_10ms_at_2491hz():
    pipe = Pipeline()
    assert pipe.fs == 48000
    assert pipe.block_size == 480
    assert pipe.block_ms == pytest.approx(10.0)
    assert pipe.f0 == 2491.0
    assert pipe.goertzel.f0 == 2491.0
    assert pipe.bandpass.f0 == 2491.0
    assert pipe.text == ""
    assert pipe.bad_blocks == 0
    assert pipe.detector.block_ms == pytest.approx(10.0)


def test_detector_is_built_with_the_contract_defaults():
    import morse.pipeline as pipeline_module

    pipe = Pipeline(fs=44100, block_size=441)
    ref = ToneDetector(block_ms=pipe.block_ms)
    for name in (
        "block_ms", "on_frac", "off_frac", "min_run_blocks", "min_lift_db", "max_lift_db",
        "min_off_lift_db", "noise_alpha_up", "noise_alpha_down", "signal_alpha",
        "signal_decay_db", "silence_floor_db", "warmup_blocks", "max_on_blocks",
    ):
        assert getattr(pipe.detector, name) == getattr(ref, name), name
    # The old per-application threshold overrides are gone for good.
    for name in ("DETECTOR_ON_FRAC", "DETECTOR_OFF_FRAC", "DETECTOR_MIN_DYNAMIC_DB"):
        assert not hasattr(pipeline_module, name), name


# ------------------------------------------------------------- BlockResult


def test_block_result_fields_are_populated():
    amplitude = 0.1
    x = synth("SOS", 1000.0, amplitude=amplitude, lead_ms=500.0)
    pipe = Pipeline(fs=FS, block_size=N, f0=1000.0)
    text, runs, results, flushed = run_pipeline(pipe, x)

    assert len(results) == x.size // N
    for r in results:
        assert isinstance(r, BlockResult)
        assert isinstance(r.on, bool)
        assert isinstance(r.new_text, str)
        assert isinstance(r.runs, list)
        assert r.filtered.dtype == np.float32 and r.filtered.shape == (N,)
        assert np.isfinite([r.power_db, r.level_dbfs, r.floor_db, r.peak_db,
                            r.threshold_lo_db, r.threshold_hi_db]).all()
        assert r.threshold_lo_db <= r.threshold_hi_db
        assert r.floor_db <= r.peak_db
        assert r.floor_db <= r.threshold_lo_db

    # First 500 ms is noise only: level about -60 dBFS, tone power far below, OFF.
    lead = results[:50]
    assert all(not r.on for r in lead)
    assert np.mean([r.level_dbfs for r in lead]) == pytest.approx(-60.0, abs=1.5)
    assert max(r.power_db for r in lead) < -50.0

    # Blocks 52..61 are inside the first dit (150 ms from 500 ms): tone at -20 dB
    # power, level 20*log10(0.1/sqrt(2)) = -23 dBFS, detector ON, floor at the noise.
    mark = results[52:62]
    assert all(r.on for r in mark)
    assert np.mean([r.power_db for r in mark]) == pytest.approx(-20.0, abs=1.0)
    assert np.mean([r.level_dbfs for r in mark]) == pytest.approx(-23.0, abs=1.0)
    assert all(r.peak_db > r.threshold_hi_db > r.threshold_lo_db > r.floor_db for r in mark)
    assert all(r.floor_db < -70.0 for r in mark)

    # The band-passed block keeps the tone (about unity gain in the pass band)
    # and removes most of the broadband noise.
    tone_rms = float(np.sqrt(np.mean(np.concatenate([r.filtered for r in mark]) ** 2)))
    assert tone_rms == pytest.approx(amplitude / np.sqrt(2), rel=0.15)
    noise_rms = float(np.sqrt(np.mean(np.concatenate([r.filtered for r in lead[5:]]) ** 2)))
    assert noise_rms < 0.3 * NOISE_RMS

    # Runs alternate, every run finalised somewhere is reported exactly once,
    # and the emitted text adds up to the decoder's text.
    assert all(a.on != b.on for a, b in zip(runs, runs[1:]))
    assert len(runs) == sum(len(r.runs) for r in results) + len(pipe.last_flush_runs)
    assert sum(len(r.runs) for r in results) >= 18  # everything but the trailing silence
    assert "".join(r.new_text for r in results) + flushed == pipe.text == text
    assert text.strip() == "SOS"
    assert pipe.bad_blocks == 0


def test_last_letter_appears_during_trailing_silence_not_at_flush():
    # 1.5 s of trailing silence is longer than 7 dits at 8 WPM (1050 ms) minus the
    # reverb offset, so the idle rule emits the final S before flush() is called.
    x = synth("SOS", 1000.0, tail_ms=1500.0)
    pipe = Pipeline(fs=FS, f0=1000.0)
    emitted = "".join(pipe.process_block(b).new_text for b in blocks_of(x))
    assert emitted.strip() == "SOS"
    assert pipe.flush() == ""


def test_flush_emits_pending_letter_when_stream_ends_in_tone():
    # No trailing silence at all: the last dit is still the in-progress run.
    x = synth("SOS", 1000.0, tail_ms=0.0)
    pipe = Pipeline(fs=FS, f0=1000.0)
    emitted = "".join(pipe.process_block(b).new_text for b in blocks_of(x))
    assert emitted.strip() == "SO"
    flushed = pipe.flush()
    assert "S" in flushed
    assert pipe.text.strip() == "SOS"
    assert pipe.last_flush_runs and pipe.last_flush_runs[-1].on


def test_short_or_empty_blocks_do_not_crash():
    pipe = Pipeline()
    r = pipe.process_block(np.zeros(0, dtype=np.float32))
    assert r.level_dbfs == pytest.approx(-120.0)
    assert r.filtered.shape == (0,)
    r = pipe.process_block(np.zeros(100, dtype=np.float32))
    assert r.filtered.shape == (100,)
    assert r.power_db <= -110.0
    assert pipe.blocks_processed == 2
    assert pipe.bad_blocks == 0


# ---------------------------------------------------------- NaN / inf blocks


def test_nan_or_inf_block_is_zeroed_counted_and_does_not_poison_detection():
    x = synth("SOS", 1000.0)
    blocks = blocks_of(x)
    blocks[10][:] = np.nan          # whole block, in the lead-in noise
    blocks[20][7] = np.inf          # one sample is enough to condemn the block
    blocks[57][:] = -np.inf         # inside the first dit (500..650 ms)
    bad = {10, 20, 57}

    pipe = Pipeline(fs=FS, f0=1000.0)
    text = ""
    runs: list[Run] = []
    for i, block in enumerate(blocks):
        r = pipe.process_block(block)
        text += r.new_text
        runs.extend(r.runs)
        # Nothing downstream ever sees a non-finite value ...
        assert np.isfinite(r.filtered).all(), i
        assert np.isfinite([r.power_db, r.level_dbfs, r.floor_db, r.peak_db,
                            r.threshold_lo_db, r.threshold_hi_db]).all(), i
        if i in bad:
            # ... and the bad block itself was processed as digital silence.
            assert r.level_dbfs == pytest.approx(-120.0)
            assert r.power_db == pytest.approx(-120.0)
    text += pipe.flush()
    runs.extend(pipe.last_flush_runs)

    assert pipe.bad_blocks == 3
    # The lead-in blocks did not drag the noise tracker down (silence clamp at -100 dB
    # and one 0.1-weight update each), and the one-block hole in the first dit was
    # debounced away: the message still reads SOS with nine marks.
    assert text.strip() == "SOS"
    assert sum(1 for r in runs if r.on) == 9
    assert all(a.on != b.on for a, b in zip(runs, runs[1:]))


def test_nan_block_does_not_leave_the_bandpass_state_poisoned():
    pipe = Pipeline(fs=FS, f0=1000.0)
    x = synth("SOS", 1000.0)
    blocks = blocks_of(x)
    pipe.process_block(np.full(N, np.nan, dtype=np.float32))
    assert pipe.bad_blocks == 1
    # Without sanitising, sosfilt's carried state would be NaN forever.
    r = pipe.process_block(blocks[55])  # inside the first dit
    assert np.isfinite(r.filtered).all()
    assert float(np.sqrt(np.mean(r.filtered ** 2))) > 0.02


def test_reset_clears_bad_blocks():
    pipe = Pipeline()
    pipe.process_block(np.full(N, np.inf, dtype=np.float32))
    assert pipe.bad_blocks == 1
    pipe.reset()
    assert pipe.bad_blocks == 0


# ---------------------------------------------------- f0 validation / retune


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, 24000.0, 30000.0, float("nan"), float("inf")])
def test_constructor_rejects_f0_outside_the_open_nyquist_range(bad):
    with pytest.raises(ValueError):
        Pipeline(f0=bad)


@pytest.mark.parametrize("fs, bad", [(44100, 22050.0), (16000, 8000.0), (16000, 0.0)])
def test_constructor_validation_uses_the_given_sample_rate(fs, bad):
    with pytest.raises(ValueError):
        Pipeline(fs=fs, block_size=fs // 100, f0=bad)
    Pipeline(fs=fs, block_size=fs // 100, f0=fs / 2 - 1.0)  # inside the range: fine


@pytest.mark.parametrize("bad", [0, -5.0, 24000.0, 30000.0, float("nan")])
def test_set_frequency_rejects_out_of_range_and_leaves_the_pipeline_alone(bad):
    x = synth("SOS", 1000.0, tail_ms=0.0)
    pipe = Pipeline(fs=FS, f0=1000.0)
    for block in blocks_of(x):
        pipe.process_block(block)
    text_before = pipe.text
    pending_before = pipe.detector.current_run
    with pytest.raises(ValueError):
        pipe.set_frequency(bad)
    assert pipe.f0 == 1000.0
    assert pipe.goertzel.f0 == 1000.0 and pipe.bandpass.f0 == 1000.0
    assert pipe.text == text_before
    assert pipe.detector.current_run == pending_before  # nothing flushed or reset
    assert not pipe.detector.warming_up


def test_set_frequency_accepts_the_whole_open_range():
    pipe = Pipeline()
    pipe.set_frequency(1.0)
    assert pipe.f0 == 1.0
    pipe.set_frequency(23999.0)
    assert pipe.f0 == 23999.0 == pipe.goertzel.f0 == pipe.bandpass.f0


def test_set_frequency_retunes_every_stage():
    # A quiet 1 kHz message is invisible to a pipeline tuned to 2491 Hz ...
    x = synth("SOS", 1000.0, amplitude=0.03)
    pipe = Pipeline(fs=FS, f0=2491.0)
    wrong_text, wrong_runs, wrong_results, _ = run_pipeline(pipe, x)
    assert not any(r.on for r in wrong_runs)
    assert wrong_text == ""

    # ... until it is retuned; goertzel, bandpass and the pipeline all agree.
    pipe.set_frequency(1000.0)
    assert pipe.f0 == 1000.0
    assert pipe.goertzel.f0 == 1000.0
    assert pipe.bandpass.f0 == 1000.0
    text, runs, results, _ = run_pipeline(pipe, x)
    assert text.strip() == "SOS"
    assert sum(1 for r in runs if r.on) == 9
    assert max(r.power_db for r in results) > max(r.power_db for r in wrong_results) + 30.0


def test_set_frequency_flushes_the_detector_resets_it_and_keeps_the_decoder():
    wpm = 10  # 120 ms dit, distinct from the decoder's 150 ms seed
    x = synth("SOS", 1000.0, wpm=wpm, tail_ms=0.0)  # ends inside the last dit
    pipe = Pipeline(fs=FS, f0=1000.0)
    runs: list[Run] = []
    for block in blocks_of(x):
        runs.extend(pipe.process_block(block).runs)
    assert pipe.text == "SO"
    assert pipe.decoder.buffer == ".."          # two finalised dits of the last S
    assert pipe.detector.current_run.on          # the third dit is still in progress
    assert sum(1 for r in runs if r.on) == 8
    dit_before, offset_before = pipe.decoder.dit_ms, pipe.decoder.offset_ms
    assert dit_before == pytest.approx(120.0, abs=20.0)
    assert offset_before > 0.0

    pipe.set_frequency(2491.0)

    # The pending ON run went to the decoder: buffer complete, text untouched.
    assert pipe.last_flush_runs and pipe.last_flush_runs[-1].on
    assert pipe.decoder.buffer == "..."
    assert pipe.text == "SO"
    # Timing estimates survive (not back to the 150 ms / 0 ms seed).
    assert pipe.decoder.dit_ms == pytest.approx(120.0, abs=25.0)
    assert pipe.decoder.offset_ms > 0.0
    # The detector starts over: warm-up, no pending run, OFF.
    assert pipe.detector.warming_up
    assert pipe.detector.current_run.blocks == 0
    assert not pipe.detector.state
    assert pipe.f0 == 2491.0 == pipe.goertzel.f0 == pipe.bandpass.f0

    # Continue at the new frequency: the idle rule closes the pending S during
    # the 1.5 s lead-in, then the second message decodes normally.
    x2 = synth("SOS", 2491.0, wpm=wpm, lead_ms=1500.0)
    emitted = "".join(pipe.process_block(b).new_text for b in blocks_of(x2))
    emitted += pipe.flush()
    assert emitted == "S SOS "
    assert pipe.text == "SOS SOS "


def test_set_frequency_with_nothing_pending_only_resets_the_detector():
    pipe = Pipeline(fs=FS, f0=1000.0)
    run_pipeline(pipe, synth("SOS", 1000.0))  # ends with flush(): nothing pending
    text = pipe.text
    pipe.set_frequency(1000.0)  # same value: contract behaviour is unconditional
    assert pipe.last_flush_runs == []
    assert pipe.text == text
    assert pipe.detector.warming_up


def test_reset_clears_text_and_state():
    x = synth("SOS", 1000.0)
    pipe = Pipeline(fs=FS, f0=1000.0)
    run_pipeline(pipe, x)
    assert pipe.text.strip() == "SOS"
    pipe.reset()
    assert pipe.text == ""
    assert pipe.blocks_processed == 0
    assert pipe.decoder.letter_count == 0
    text, _, _, _ = run_pipeline(pipe, x)
    assert text.strip() == "SOS"


# --------------------------------------------------------------- decode_wav


def _as_format(x: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "int16":
        return (x * 32767).astype(np.int16)
    if fmt == "int32":
        return (x.astype(np.float64) * 2147483647).astype(np.int32)
    if fmt == "float32":
        return x.astype(np.float32)
    if fmt == "float64":
        return x.astype(np.float64)
    if fmt == "uint8":
        return np.clip(np.round(x * 127 + 128), 0, 255).astype(np.uint8)
    if fmt == "stereo_int16":
        # Channel 0 carries the message, channel 1 is unrelated noise.
        other = np.random.default_rng(1).normal(0, 0.05, x.size)
        return np.stack([x * 32767, other * 32767], axis=1).astype(np.int16)
    if fmt == "stereo_float32":
        other = np.random.default_rng(2).normal(0, 0.05, x.size)
        return np.stack([x, other], axis=1).astype(np.float32)
    raise AssertionError(fmt)


@pytest.mark.parametrize("fmt", ["int16", "int32", "float32", "float64", "uint8",
                                 "stereo_int16", "stereo_float32"])
def test_decode_wav_accepts_common_formats(tmp_path, fmt):
    x = synth("SOS", 1000.0, amplitude=0.1)
    path = _write(tmp_path / f"sos_{fmt}.wav", FS, _as_format(x, fmt))
    text, runs = decode_wav(path, 1000.0)
    assert text.strip() == "SOS", fmt
    assert sum(1 for r in runs if r.on) == 9
    assert all(a.on != b.on for a, b in zip(runs, runs[1:]))


def test_load_wav_scales_every_format_to_the_same_float32(tmp_path):
    x = synth("SOS", 1000.0, amplitude=0.1)
    reference = None
    for fmt in ("int16", "int32", "float32", "stereo_int16"):
        path = _write(tmp_path / f"sos_{fmt}.wav", FS, _as_format(x, fmt))
        fs, loaded = load_wav(path)
        assert fs == FS
        assert loaded.dtype == np.float32 and loaded.ndim == 1 and loaded.size == x.size
        if reference is None:
            reference = loaded
        assert np.allclose(loaded, reference, atol=1e-4), fmt


def test_decode_wav_uses_channel_zero_only(tmp_path):
    # Message in channel 1, digital silence in channel 0: nothing to decode.
    x = synth("SOS", 1000.0, noise_rms=0.0)
    data = np.stack([np.zeros_like(x), x], axis=1).astype(np.float32)
    path = _write(tmp_path / "sos_in_channel_1.wav", FS, data)
    text, runs = decode_wav(path, 1000.0)
    assert text == ""
    assert not any(r.on for r in runs)
    fs, loaded = load_wav(path)
    assert fs == FS and loaded.shape == (x.size,)
    assert not loaded.any()


def test_decode_samples_takes_channel_zero_and_rejects_3d():
    x = synth("SOS", 1000.0)
    mono = decode_samples(x, FS, 1000.0)
    assert mono[0].strip() == "SOS"
    stereo = decode_samples(np.stack([x, np.zeros_like(x)], axis=1), FS, 1000.0)
    assert stereo == mono
    swapped_text, swapped_runs = decode_samples(np.stack([np.zeros_like(x), x], axis=1), FS, 1000.0)
    assert swapped_text == "" and not any(r.on for r in swapped_runs)
    # Integer samples are scaled like WAV data.
    assert decode_samples((x * 32767).astype(np.int16), FS, 1000.0)[0].strip() == "SOS"
    with pytest.raises(ValueError):
        decode_samples(np.zeros((100, 2, 2), dtype=np.float32), FS, 1000.0)
    with pytest.raises(ValueError):
        decode_samples(x, FS, 0.0)  # f0 validation applies here too


def test_decode_wav_uses_the_files_sample_rate(tmp_path):
    fs = 44100
    x = synth_keyed_tone(keyed_pattern("SOS", WPM), 1000.0, fs,
                         amplitude=0.1, noise_rms=NOISE_RMS, reverb_ms=REVERB_MS)
    path = _write(tmp_path / "sos_44k.wav", fs, (x * 32767).astype(np.int16))
    text, runs = decode_wav(path, 1000.0)
    assert text.strip() == "SOS"
    # Run timing is in real milliseconds even though a block is now 10.88 ms.
    assert all(r.block_ms == pytest.approx(1000.0 * 480 / fs) for r in runs)
    dits = sorted(r.ms for r in runs if r.on)[:6]
    assert all(100.0 < ms < 300.0 for ms in dits)


def test_decode_wav_matches_pipeline_and_includes_flush(tmp_path):
    x = synth("SOS", 1000.0, tail_ms=0.0)  # ends inside the last dit
    path = _write(tmp_path / "sos_cut.wav", FS, (x * 32767).astype(np.int16))
    text, runs = decode_wav(path, 1000.0)
    assert text.strip() == "SOS"
    assert runs[-1].on  # the in-progress final run came from flush()
    # Same answer as feeding the loaded samples through a Pipeline by hand
    # (compare on the int16-quantised samples the file actually holds).
    fs, loaded = load_wav(path)
    assert fs == FS
    pipe = Pipeline(fs=FS, f0=1000.0)
    ref_text, ref_runs, _, _ = run_pipeline(pipe, loaded)
    assert text == ref_text
    assert runs == ref_runs
    assert decode_samples(loaded, FS, 1000.0) == (text, runs)


def test_decode_wav_missing_file_and_directory_raise_oserror(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_wav(str(tmp_path / "does_not_exist.wav"), 1000.0)
    with pytest.raises(OSError):
        decode_wav(str(tmp_path), 1000.0)


def test_decode_wav_rejects_non_wav_content(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not a RIFF file at all" * 4)
    with pytest.raises(ValueError):
        decode_wav(str(path), 1000.0)


# ---------------------------------------------------------------- morse.app


def _run_app(*argv: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, "-m", "morse.app", *argv], cwd=str(ROOT),
                          capture_output=True, env=env, timeout=120)


def test_app_wav_mode_prints_text(tmp_path, capsys):
    from morse import app

    x = synth("SOS", 1000.0)
    path = _write(tmp_path / "sos.wav", FS, (x * 32767).astype(np.int16))
    assert app.main(["--wav", path, "--freq", "1000"]) == 0
    assert capsys.readouterr().out.strip() == "SOS"


def test_app_wav_verbose_lists_runs_before_text(tmp_path, capsys):
    from morse import app

    x = synth("SOS", 1000.0)
    path = _write(tmp_path / "sos.wav", FS, (x * 32767).astype(np.int16))
    assert app.main(["--wav", path, "--freq", "1000", "--wpm", "8", "--verbose"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "SOS"
    run_lines = lines[:-1]
    assert len(run_lines) >= 18
    assert all(line.startswith(("ON ", "off ")) and line.endswith(" ms") for line in run_lines)
    assert sum(1 for line in run_lines if line.startswith("ON ")) == 9


def test_app_wav_unreadable_exits_2_with_one_line(tmp_path, capsys):
    from morse import app

    missing = str(tmp_path / "missing.wav")
    assert app.main(["--wav", missing]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and err.endswith("\n")
    assert err.startswith("error:") and "missing.wav" in err
    assert "Traceback" not in err

    assert app.main(["--wav", str(tmp_path)]) == 2  # a directory
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("error:") and tmp_path.name in err
    assert "Traceback" not in err


def test_app_wav_directory_subprocess_exits_2_without_traceback(tmp_path):
    proc = _run_app("--wav", str(tmp_path), "--freq", "1000")
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout == b""
    err = proc.stderr.decode("utf-8", errors="replace")
    assert len(err.strip().splitlines()) == 1, err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_app_wav_other_errors_exit_1(tmp_path, capsys):
    from morse import app

    x = synth("SOS", 1000.0)
    path = _write(tmp_path / "sos.wav", FS, (x * 32767).astype(np.int16))
    assert app.main(["--wav", path, "--freq", "30000"]) == 1  # above fs/2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav" * 10)
    assert app.main(["--wav", str(junk)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and err.count("\n") == 1


def test_app_parser_defaults_and_device_parsing():
    from morse import app

    args = app.build_parser().parse_args([])
    assert args.freq == 2491.0
    assert args.wpm is None
    assert args.device is None
    assert not args.no_ui and not args.list_devices and not args.verbose
    assert app.parse_device(None) is None
    assert app.parse_device("18") == 18
    assert app.parse_device(" 7 ") == 7
    assert app.parse_device("Microphone Array 1") == "Microphone Array 1"
    assert app.parse_device("") is None
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--freq", "-5"])
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--wpm", "0"])


def test_app_without_ui_module_points_at_no_ui(monkeypatch, capsys):
    from morse import app

    monkeypatch.setitem(sys.modules, "morse.ui", None)  # makes the import fail
    assert app.main(["--freq", "1000"]) == 1
    err = capsys.readouterr().err
    assert "--no-ui" in err
    assert "morse/ui.py" in err or "Qt UI" in err


def test_app_ui_mode_calls_run_ui_with_parsed_args(monkeypatch):
    import types

    from morse import app

    seen = {}
    fake = types.ModuleType("morse.ui")

    def run_ui(args):
        seen["args"] = args
        return 0

    fake.run_ui = run_ui
    monkeypatch.setitem(sys.modules, "morse.ui", fake)
    assert app.main(["--device", "18", "--freq", "1000", "--wpm", "12"]) == 0
    assert seen["args"].device == 18
    assert seen["args"].freq == 1000.0
    assert seen["args"].wpm == 12.0


# ------------------------------------------------------ --list-devices output

INTEL_NAME = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"
UNENCODABLE_NAME = "USB Mic ✓マイク"  # check mark and katakana: not in cp1252


def _fake_devices():
    from morse.audio_input import DeviceInfo

    return [
        DeviceInfo(1, INTEL_NAME, "MME", 6, 44100.0),
        DeviceInfo(7, UNENCODABLE_NAME, "Windows WASAPI", 2, 48000.0),
        DeviceInfo(18, "Microphone Array 1 ()", "Windows WDM-KS", 4, 48000.0),
    ]


def _install_fake_devices(monkeypatch, devices=None, default: int = 18):
    import morse.audio_input as audio_input

    listed = _fake_devices() if devices is None else devices
    monkeypatch.setattr(audio_input, "list_devices", lambda: list(listed))
    monkeypatch.setattr(audio_input, "resolve_device", lambda spec: default)


def test_app_list_devices_prints_every_device_and_marks_the_default(monkeypatch, capsys):
    from morse import app

    _install_fake_devices(monkeypatch)
    assert app.main(["--list-devices"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0].split() == ["idx", "ch", "rate", "host", "API", "name"]
    assert INTEL_NAME in out and UNENCODABLE_NAME in out  # utf-8 capture keeps everything
    device_lines = lines[1:-1]
    assert len(device_lines) == len(_fake_devices())
    marked = [line for line in device_lines if line.startswith("*")]
    assert len(marked) == 1 and " 18 " in marked[0] and "Microphone Array 1" in marked[0]
    assert lines[-1].startswith("* = device used")


def test_app_list_devices_survives_strict_cp1252_stdout(monkeypatch):
    from morse import app

    buffer = io.BytesIO()
    strict = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="", write_through=True)
    with pytest.raises(UnicodeEncodeError):  # prove the stream really is strict
        strict.write(UNENCODABLE_NAME)
    monkeypatch.setattr(sys, "stdout", strict)
    _install_fake_devices(monkeypatch)

    assert app.main(["--list-devices"]) == 0
    strict.flush()
    out = buffer.getvalue().decode("cp1252")
    assert INTEL_NAME in out                  # the registered sign exists in cp1252
    assert "USB Mic ????" in out               # replaced, not raised
    assert out.count("\n") == 1 + len(_fake_devices()) + 1


def test_app_list_devices_survives_strict_ascii_stdout(monkeypatch):
    from morse import app

    buffer = io.BytesIO()
    strict = io.TextIOWrapper(buffer, encoding="ascii", errors="strict", newline="", write_through=True)
    monkeypatch.setattr(sys, "stdout", strict)
    _install_fake_devices(monkeypatch)
    assert app.main(["--list-devices"]) == 0
    strict.flush()
    out = buffer.getvalue().decode("ascii")
    assert "Microphone Array (Intel? Smart Sound" in out


def test_app_list_devices_subprocess_with_pythonioencoding_cp1252_strict():
    # A control run shows that a plain print() of the same names aborts under
    # PYTHONIOENCODING=cp1252:strict, so the app's success below is meaningful.
    names = repr([INTEL_NAME, UNENCODABLE_NAME])
    env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
    env.pop("PYTHONUTF8", None)
    control = subprocess.run([sys.executable, "-c", f"for n in {names}: print(n)"],
                             capture_output=True, env=env, timeout=60)
    assert control.returncode != 0 and b"UnicodeEncodeError" in control.stderr

    code = textwrap.dedent(
        f"""
        import morse.audio_input as audio_input
        from morse.audio_input import DeviceInfo
        names = {names}
        audio_input.list_devices = lambda: [
            DeviceInfo(1, names[0], "MME", 6, 44100.0),
            DeviceInfo(7, names[1], "Windows WASAPI", 2, 48000.0),
        ]
        audio_input.resolve_device = lambda spec: 1
        from morse import app
        raise SystemExit(app.main(["--list-devices"]))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    out = proc.stdout.decode("cp1252")
    assert INTEL_NAME in out
    assert "USB Mic ????" in out
    assert out.strip().splitlines()[1].startswith("*   1 ")


def test_app_list_devices_reports_failures_and_empty_lists(monkeypatch, capsys):
    import morse.audio_input as audio_input
    from morse import app

    _install_fake_devices(monkeypatch, devices=[])
    assert app.main(["--list-devices"]) == 0
    assert "no input-capable" in capsys.readouterr().out

    def boom():
        raise RuntimeError("PortAudio not initialised")

    monkeypatch.setattr(audio_input, "list_devices", boom)
    assert app.main(["--list-devices"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "PortAudio" in err


def test_emit_never_raises_and_defaults_to_stdout(capsys):
    from morse import app

    app.emit("plain")
    app.emit("no newline", end="")
    app.emit("to stderr", file=sys.stderr, flush=True)
    out, err = capsys.readouterr()
    assert out == "plain\nno newline"
    assert err == "to stderr\n"

    class NoEncoding(io.StringIO):
        encoding = None  # type: ignore[assignment]

    sink = NoEncoding()
    app.emit(UNENCODABLE_NAME, file=sink)
    assert sink.getvalue() == UNENCODABLE_NAME + "\n"  # utf-8 assumed, nothing lost


# --------------------------------------------------------------- live modes


class _FakeAudioInput:
    """Stands in for morse.audio_input.AudioInput: serves queued blocks, then Ctrl-C."""

    chunks: list[list[np.ndarray]] = []

    def __init__(self, device=None, fs=48000, block_size=480, queue_size=1000):
        self.device_index = 18
        self.device_name = "Fake Microphone Array (Intel® Smart Sound)"
        self.fs = fs
        self.block_size = block_size
        self.dropped = 0
        self.started = False
        self.stopped = False
        self._chunks = [list(c) for c in type(self).chunks]

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def read_blocks(self):
        if not self._chunks:
            raise KeyboardInterrupt  # what Ctrl-C does to the polling loop
        return self._chunks.pop(0)


def _install_fake_audio(monkeypatch, samples: np.ndarray, blocks_per_poll: int = 3):
    import morse.audio_input as audio_input
    from morse import app

    blocks = blocks_of(samples)
    _FakeAudioInput.chunks = [blocks[i:i + blocks_per_poll] for i in range(0, len(blocks), blocks_per_poll)]
    monkeypatch.setattr(audio_input, "AudioInput", _FakeAudioInput)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)


def test_app_live_text_mode_without_hardware(monkeypatch, capsys):
    from morse import app

    _install_fake_audio(monkeypatch, synth("SOS", 1000.0))
    assert app.main(["--no-ui", "--freq", "1000", "--device", "18"]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == "SOS"
    assert "Listening" in err and "Intel® Smart Sound" in err and "1000" in err


def test_app_live_verbose_mode_without_hardware(monkeypatch, capsys):
    from morse import app

    _install_fake_audio(monkeypatch, synth("SOS", 1000.0, tail_ms=0.0))  # flush supplies the last S
    assert app.main(["--no-ui", "--freq", "1000", "--verbose"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert sum(1 for line in lines if line.startswith("ON ")) == 9
    assert lines[-1].startswith("text: ")
    assert lines[-1].split("text: ", 1)[1].strip() == "SOS"


def test_app_live_mode_rejects_bad_frequency_before_opening_audio(monkeypatch, capsys):
    import morse.audio_input as audio_input
    from morse import app

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("AudioInput must not be constructed for an invalid --freq")

    monkeypatch.setattr(audio_input, "AudioInput", must_not_be_called)
    assert app.main(["--no-ui", "--freq", "30000"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "fs/2" in err


# ------------------------------------------------------------ broadband gate


def _on_runs_of(x: np.ndarray, fs: int = FS, blk: int = 480, f0: float = 2491.0) -> list[Run]:
    pipe = Pipeline(fs=fs, block_size=blk, f0=f0)
    runs: list[Run] = []
    for i in range(x.size // blk):
        runs += pipe.process_block(x[i * blk:(i + 1) * blk]).runs
    runs += pipe.detector.flush()
    return [r for r in runs if r.on]


def test_broadband_click_never_switches_the_detector_on_but_a_tone_burst_does():
    # 3 s of room noise at -80 dBFS; at t = 1 s either a 60 ms broadband burst
    # at -30 dBFS (a keyboard click, 50 dB above the room) or a 60 ms tone burst
    # at -20 dBFS. The click spreads its energy over the whole band, so the tone
    # bin holds about 1/240 of it: the gate reports it at the noise floor.
    rng = np.random.default_rng(7)
    room = rng.normal(0.0, 10 ** (-80 / 20), 3 * FS).astype(np.float32)
    n = int(0.06 * FS)
    i0 = FS
    click = room.copy()
    click[i0:i0 + n] += rng.normal(0.0, 10 ** (-30 / 20), n).astype(np.float32)
    tone = room.copy()
    tone[i0:i0 + n] += (0.1 * np.sin(2 * np.pi * 2491.0 * np.arange(n) / FS)).astype(np.float32)
    assert _on_runs_of(click) == []
    on = _on_runs_of(tone)
    assert len(on) == 1 and abs(on[0].ms - 60.0) <= 30.0


def test_block_result_reports_tonality():
    pipe = Pipeline(fs=FS, block_size=480, f0=2491.0)
    t = np.arange(480) / FS
    pure = (0.3 * np.sin(2 * np.pi * 2491.0 * t)).astype(np.float32)
    assert abs(pipe.process_block(pure).tonality_db) < 1.0
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.01, 480).astype(np.float32)
    values = [pipe.process_block(noise).tonality_db for _ in range(50)]
    assert np.median(values) < -15.0
