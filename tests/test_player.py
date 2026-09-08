"""Tests for morse.player: timing builder, tone renderer, sounddevice player.

``TonePlayer`` is tested against a fake ``sounddevice`` module installed in
``sys.modules``: the player imports sounddevice lazily inside its methods, so
the fake is what it sees and no PortAudio stream is opened. One guarded smoke
test plays 20 ms of silence through the real default output and skips when
there is none.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morse.pipeline import decode_samples
from morse.player import TonePlayer, build_timing, render_tone

ROOT = Path(__file__).resolve().parents[1]
BEEP_SENDER = ROOT / "tools" / "beep_sender.py"

FS = 48000
AMP = 0.3
RAMP = 144  # 3 ms at 48 kHz

# 'SOS' at 10 WPM: dit 120, dah 360, letter gap 360 (same list as tests/test_beep_sender.py).
SOS_10WPM = [
    (True, 120), (False, 120), (True, 120), (False, 120), (True, 120),   # S
    (False, 360),
    (True, 360), (False, 120), (True, 360), (False, 120), (True, 360),   # O
    (False, 360),
    (True, 120), (False, 120), (True, 120), (False, 120), (True, 120),   # S
]
SOS_10WPM_TOTAL_MS = 3240


def total_ms(seq: list[tuple[bool, float]]) -> float:
    return sum(ms for _, ms in seq)


def segments(seq: list[tuple[bool, float]], fs: int = FS) -> list[tuple[bool, int, int]]:
    """``(on, start_sample, stop_sample)`` per segment, the way render_tone places them."""
    edges = np.rint(np.cumsum([0.0, *(ms for _, ms in seq)]) * fs / 1000.0).astype(int)
    return [(bool(on), int(a), int(b)) for (on, _), a, b in zip(seq, edges[:-1], edges[1:])]


def max_step(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x.astype(np.float64))).max())


# -- build_timing -------------------------------------------------------------


def test_sos_at_10_wpm_exact() -> None:
    assert build_timing("SOS", 10) == SOS_10WPM


def test_durations_are_whole_millisecond_floats() -> None:
    seq = build_timing("SOS", 7)  # dit 171.43 -> 171, dah 514.29 -> 514
    assert all(isinstance(on, bool) and isinstance(ms, float) for on, ms in seq)
    assert all(ms == round(ms) for _, ms in seq)
    assert seq[0] == (True, 171.0)
    assert seq[6] == (True, 514.0)


def test_word_gap_is_seven_dits() -> None:
    assert build_timing("E E", 10) == [(True, 120), (False, 840), (True, 120)]


def test_whitespace_runs_and_surrounding_whitespace_make_one_word_gap() -> None:
    assert build_timing("  E \t\n  E  ", 10) == [(True, 120), (False, 840), (True, 120)]


def test_two_words_share_one_word_gap() -> None:
    seq = build_timing("SOS SOS", 10)
    assert seq == SOS_10WPM + [(False, 840)] + SOS_10WPM
    assert seq.count((False, 840)) == 1


def test_case_insensitive_and_unknown_characters_skipped() -> None:
    assert build_timing("sos", 10) == SOS_10WPM
    assert build_timing("S#O~S", 10) == SOS_10WPM
    assert build_timing("##", 10) == []
    assert build_timing("", 10) == []
    assert build_timing("   ", 10) == []
    # A word made entirely of unknown characters does not leave a stray gap.
    assert build_timing("E ## E", 10) == [(True, 120), (False, 840), (True, 120)]


def test_no_leading_trailing_or_adjacent_gaps() -> None:
    seq = build_timing("HELLO WORLD 73", 12)
    assert seq[0][0] is True and seq[-1][0] is True
    assert all(a[0] != b[0] for a, b in zip(seq, seq[1:]))  # strictly alternating


def test_dit_length_scales_with_wpm() -> None:
    assert build_timing("E", 20) == [(True, 60)]
    assert build_timing("E", 8) == [(True, 150)]
    assert build_timing("T", 8) == [(True, 450)]
    assert build_timing("E", 7) == [(True, 171)]  # 171.43 rounded
    assert build_timing("E E", 7) == [(True, 171), (False, 1200), (True, 171)]
    assert build_timing("E", 12.5) == [(True, 96)]


def test_punctuation_and_digits_use_the_package_table() -> None:
    seq = build_timing("?", 10)  # ..--..
    assert [on for on, _ in seq] == [True, False] * 5 + [True]
    assert [ms for on, ms in seq if on] == [120, 120, 360, 360, 120, 120]
    assert total_ms(build_timing("0", 10)) == 5 * 360 + 4 * 120


@pytest.mark.parametrize("wpm", [0, -5, float("nan"), float("inf")])
def test_non_positive_wpm_raises(wpm: float) -> None:
    with pytest.raises(ValueError):
        build_timing("SOS", wpm)


def _load_beep_sender() -> Any:
    spec = importlib.util.spec_from_file_location("beep_sender_for_player_test", BEEP_SENDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wpm", [5, 7, 8, 10, 12.5, 15, 20, 33])
def test_matches_tools_beep_sender_exactly(wpm: float) -> None:
    """The laptop plays what the desktop beeper sends: same rule, same rounding."""
    bs = _load_beep_sender()
    texts = [
        "SOS",
        "sos sos",
        "HELLO WORLD 73",
        "S#O~S",
        "  E \t\n  E  ",
        "E ## E",
        "The quick brown fox, jumps? over: 1234567890 @ !",
        ".,?/='():;+-_\"@!",
        "",
    ]
    for text in texts:
        ours = build_timing(text, wpm)
        theirs = bs.build_timing(text, wpm)
        assert ours == theirs, (text, wpm)
        assert [type(ms) for _, ms in ours] == [float] * len(ours)
        assert [type(ms) for _, ms in theirs] == [int] * len(theirs)


# -- render_tone: shape ---------------------------------------------------------


def test_length_equals_total_ms_at_48k() -> None:
    seq = build_timing("SOS", 10)
    assert total_ms(seq) == SOS_10WPM_TOTAL_MS
    x = render_tone(seq, 1000.0)
    assert x.shape == (SOS_10WPM_TOTAL_MS * FS // 1000,)
    assert x.dtype == np.float32
    assert x.ndim == 1


@pytest.mark.parametrize("fs", [8000, 22050, 44100, 96000])
def test_length_equals_total_ms_at_other_rates(fs: int) -> None:
    seq = build_timing("HELLO WORLD", 7)  # odd durations: 171, 514, 1200
    x = render_tone(seq, 1000.0, fs=fs)
    assert x.size == round(total_ms(seq) * fs / 1000)


def test_empty_timing_renders_empty() -> None:
    x = render_tone([], 1000.0)
    assert x.shape == (0,) and x.dtype == np.float32
    x = render_tone([(True, 0.0), (False, 0.0)], 1000.0)
    assert x.shape == (0,) and x.dtype == np.float32


def test_zero_length_segments_add_nothing() -> None:
    a = render_tone([(True, 100), (False, 50), (True, 100)], 1000.0)
    b = render_tone([(True, 0), (True, 100), (False, 0), (False, 50), (True, 100), (False, 0)], 1000.0)
    np.testing.assert_array_equal(a, b)


def test_accepts_integer_flags_and_durations() -> None:
    a = render_tone([(1, 100), (0, 50), (1, 100)], 1000.0)  # type: ignore[list-item]
    b = render_tone([(True, 100.0), (False, 50.0), (True, 100.0)], 1000.0)
    np.testing.assert_array_equal(a, b)


# -- render_tone: content ---------------------------------------------------------


def test_gaps_are_digitally_silent() -> None:
    seq = build_timing("SOS SOS", 10)
    x = render_tone(seq, 2491.0)
    for on, a, b in segments(seq):
        if not on:
            assert not x[a:b].any(), (a, b)
        else:
            assert np.abs(x[a:b]).max() > 0.9 * AMP, (a, b)


def test_marks_are_the_sine_at_amplitude_with_continuous_phase() -> None:
    seq = build_timing("SOS", 10)
    f0 = 2491.0
    x = render_tone(seq, f0, amplitude=AMP)
    t = np.arange(x.size) / FS
    ref = AMP * np.sin(2 * np.pi * f0 * t)
    for on, a, b in segments(seq):
        if on:  # between the ramps the mark is exactly the reference sine
            np.testing.assert_allclose(x[a + RAMP:b - RAMP], ref[a + RAMP:b - RAMP], atol=2e-6)
    assert np.abs(x).max() <= AMP + 1e-6
    assert np.abs(x).max() == pytest.approx(AMP, abs=1e-3)


def test_amplitude_scales_output() -> None:
    seq = [(True, 100)]
    a = render_tone(seq, 1000.0, amplitude=0.3)
    b = render_tone(seq, 1000.0, amplitude=0.6)
    np.testing.assert_allclose(b, 2 * a, atol=1e-6)
    assert not render_tone(seq, 1000.0, amplitude=0.0).any()


def test_edges_ramp_no_sample_jump_larger_than_a_fifth_of_amplitude() -> None:
    # At 1000 Hz the sine itself moves at most 0.3*2*pi*1000/48000 = 0.039 per
    # sample, so any jump above 0.06 would have to come from a keying edge.
    seq = build_timing("SOS HELLO", 15)
    x = render_tone(seq, 1000.0, amplitude=AMP)
    assert max_step(x) <= AMP * 0.2


def test_ramps_remove_the_clicks_hard_keying_produces() -> None:
    # Boundaries at 100 and 150 ms fall on sin = 1 and sin = 0.7 at 1002.5 Hz.
    seq = [(True, 100), (False, 50), (True, 100)]
    hard = render_tone(seq, 1002.5, amplitude=AMP, ramp_ms=0.0)
    soft = render_tone(seq, 1002.5, amplitude=AMP)
    assert max_step(hard) > AMP * 0.9
    assert max_step(soft) <= AMP * 0.2
    assert hard.size == soft.size


def test_ramp_envelope_is_linear_inside_each_mark() -> None:
    # Holds at any frequency, 2491 Hz included, where the raw sine steps 0.098.
    seq = build_timing("SOS", 10)
    f0 = 2491.0
    x = render_tone(seq, f0, amplitude=AMP, ramp_ms=3.0)
    t = np.arange(x.size) / FS
    ref = AMP * np.sin(2 * np.pi * f0 * t)
    slope = np.arange(1, RAMP + 1) / (RAMP + 1.0)
    for on, a, b in segments(seq):
        if not on:
            continue
        np.testing.assert_allclose(x[a:a + RAMP], ref[a:a + RAMP] * slope, atol=2e-6)
        np.testing.assert_allclose(x[b - RAMP:b], ref[b - RAMP:b] * slope[::-1], atol=2e-6)
        # First and last ramp samples are one step above silence, never full scale.
        assert abs(x[a]) <= AMP / (RAMP + 1) + 1e-6
        assert abs(x[b - 1]) <= AMP / (RAMP + 1) + 1e-6


def test_ramp_length_follows_ramp_ms_and_fs() -> None:
    seq = [(True, 100), (False, 100)]
    for fs, ramp_ms in ((48000, 3.0), (48000, 10.0), (44100, 3.0), (8000, 5.0)):
        r = round(ramp_ms * fs / 1000)
        x = render_tone(seq, 1000.0, fs=fs, amplitude=1.0, ramp_ms=ramp_ms)
        t = np.arange(x.size) / fs
        ref = np.sin(2 * np.pi * 1000.0 * t)
        n_mark = round(100 * fs / 1000)
        # Inside the ramp the envelope is below 1; right after it the sine is untouched.
        env = np.abs(x[:n_mark]) / np.maximum(np.abs(ref[:n_mark]), 1e-3)
        assert env[r - 1] < 1.0
        np.testing.assert_allclose(x[r:n_mark - r], ref[r:n_mark - r], atol=2e-6)


def test_short_mark_gets_half_length_ramps() -> None:
    # 2 ms = 96 samples < 2 * 144: ramps shrink to 48 samples each and meet in the middle.
    x = render_tone([(True, 2.0)], 1000.0, amplitude=1.0)
    assert x.size == 96
    t = np.arange(96) / FS
    ref = np.sin(2 * np.pi * 1000.0 * t)
    slope = np.arange(1, 49) / 49.0
    np.testing.assert_allclose(x[:48], ref[:48] * slope, atol=2e-6)
    np.testing.assert_allclose(x[48:], ref[48:] * slope[::-1], atol=2e-6)
    # A one-sample mark has no room for a ramp and is simply the sine sample.
    one = render_tone([(False, 0.5), (True, 1.0 / 48)], 1000.0, amplitude=1.0)
    assert one.size == 25
    assert one[24] == pytest.approx(np.sin(2 * np.pi * 1000.0 * 24 / FS), abs=1e-6)


def test_zero_ramp_keys_hard() -> None:
    seq = [(True, 100)]
    x = render_tone(seq, 1002.5, amplitude=AMP, ramp_ms=0.0)
    t = np.arange(x.size) / FS
    np.testing.assert_allclose(x, AMP * np.sin(2 * np.pi * 1002.5 * t), atol=2e-6)


# -- render_tone: validation ---------------------------------------------------------


@pytest.mark.parametrize("f0", [0.0, -100.0, 24000.0, 30000.0, float("nan"), float("inf")])
def test_f0_outside_open_nyquist_interval_raises(f0: float) -> None:
    with pytest.raises(ValueError):
        render_tone([(True, 10)], f0)


def test_f0_just_inside_the_interval_is_accepted() -> None:
    assert render_tone([(True, 10)], 23999.0).size == 480
    assert render_tone([(True, 10)], 1.0).size == 480
    assert render_tone([(True, 10)], 3999.0, fs=8000).size == 80


@pytest.mark.parametrize("amplitude", [-0.1, 1.01, float("nan")])
def test_amplitude_outside_unit_range_raises(amplitude: float) -> None:
    with pytest.raises(ValueError):
        render_tone([(True, 10)], 1000.0, amplitude=amplitude)


def test_negative_ramp_raises() -> None:
    with pytest.raises(ValueError):
        render_tone([(True, 10)], 1000.0, ramp_ms=-1.0)


@pytest.mark.parametrize("fs", [0, -48000])
def test_non_positive_fs_raises(fs: int) -> None:
    with pytest.raises(ValueError):
        render_tone([(True, 10)], 1000.0, fs=fs)


@pytest.mark.parametrize("ms", [-1.0, float("nan"), float("inf")])
def test_bad_segment_duration_raises(ms: float) -> None:
    with pytest.raises(ValueError):
        render_tone([(True, 100), (False, ms)], 1000.0)


# -- render_tone -> pipeline round trip --------------------------------------------------


@pytest.mark.parametrize(
    ("text", "wpm", "dit_blocks"),
    [("SOS", 15, 8), ("HELLO WORLD", 8, 15), ("PARIS 73", 20, 6)],
)
def test_rendered_tone_decodes_through_the_pipeline(text: str, wpm: float, dit_blocks: int) -> None:
    """What Play produces is what the decoder expects, mark for mark."""
    seq = build_timing(text, wpm)
    tone = render_tone(seq, 2491.0)
    lead = np.zeros(FS // 2, dtype=np.float32)   # 500 ms: past the detector warm-up
    tail = np.zeros(FS, dtype=np.float32)        # 1 s: lets the last letter flush
    decoded, runs = decode_samples(np.concatenate([lead, tone, tail]), FS, 2491.0)
    assert decoded.strip() == text
    marks = [r.blocks for r in runs if r.on]
    keyed = [ms for on, ms in seq if on]
    assert len(marks) == len(keyed)
    for got, ms in zip(marks, keyed):
        assert got == round(ms / 10), (got, ms)
    assert all(b in (dit_blocks, 3 * dit_blocks) for b in marks)


# -- TonePlayer with a fake sounddevice -----------------------------------------------------


class FakeStream:
    """Stands in for the OutputStream behind sounddevice.play/get_stream."""

    def __init__(self) -> None:
        self._active = True
        self.closed = False

    @property
    def active(self) -> bool:
        if self.closed:  # PortAudio raises on a closed stream pointer
            raise RuntimeError("Error querying stream: Invalid stream pointer [PaErrorCode -9988]")
        return self._active

    def finish(self) -> None:
        """Pretend the samples ran out (stream open but no longer active)."""
        self._active = False


def install_fake_sd(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``sounddevice`` with play/stop/get_stream and call records."""
    fake = types.ModuleType("sounddevice")
    fake.play_calls = []  # type: ignore[attr-defined]
    fake.stop_calls = 0  # type: ignore[attr-defined]
    fake.stream = None  # type: ignore[attr-defined]

    def close_stream() -> None:
        if fake.stream is not None:  # type: ignore[attr-defined]
            fake.stream._active = False  # type: ignore[attr-defined]
            fake.stream.closed = True  # type: ignore[attr-defined]

    def play(data: Any, samplerate: Any = None, mapping: Any = None, blocking: bool = False,
             loop: bool = False, **kwargs: Any) -> None:
        close_stream()  # the real play() stops the previous playback first
        fake.stream = FakeStream()  # type: ignore[attr-defined]
        fake.play_calls.append(  # type: ignore[attr-defined]
            {"data": data, "samplerate": samplerate, "mapping": mapping,
             "blocking": blocking, "loop": loop, **kwargs}
        )

    def stop(ignore_errors: bool = True) -> None:
        # Counts only the player's own sounddevice.stop() calls.
        fake.stop_calls += 1  # type: ignore[attr-defined]
        close_stream()

    def get_stream() -> FakeStream:
        if fake.stream is None:  # type: ignore[attr-defined]
            raise RuntimeError("play()/rec()/playrec() was not called yet")
        return fake.stream  # type: ignore[attr-defined]

    fake.play = play  # type: ignore[attr-defined]
    fake.stop = stop  # type: ignore[attr-defined]
    fake.get_stream = get_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_importing_module_does_not_import_sounddevice() -> None:
    code = "import sys, morse.player; print('sounddevice' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_idle_player_never_imports_sounddevice(monkeypatch) -> None:
    # A None entry in sys.modules makes `import sounddevice` raise ImportError.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    player = TonePlayer()
    assert player.fs == 48000 and player.device is None
    assert player.playing is False
    player.stop()
    player.play(np.zeros(0, dtype=np.float32))  # empty: nothing to start
    assert player.playing is False
    assert "TonePlayer" in repr(player)


def test_constructor_validates_fs() -> None:
    with pytest.raises(ValueError):
        TonePlayer(fs=0)
    assert TonePlayer(fs=44100, device="Speakers").fs == 44100


def test_play_is_non_blocking_and_passes_rate_and_device(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    samples = render_tone(build_timing("E", 10), 1000.0)
    player = TonePlayer()
    player.play(samples)
    assert len(fake.play_calls) == 1
    call = fake.play_calls[0]
    assert call["samplerate"] == 48000
    assert call["device"] is None
    assert call["blocking"] is False
    assert call["loop"] is False
    assert call["mapping"] is None
    data = call["data"]
    assert data.dtype == np.float32 and data.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, samples)
    assert player.playing is True

    other = TonePlayer(fs=44100, device="Speakers")
    other.play(samples)
    assert fake.play_calls[1]["samplerate"] == 44100
    assert fake.play_calls[1]["device"] == "Speakers"
    third = TonePlayer(device=7)
    third.play(samples)
    assert fake.play_calls[2]["device"] == 7


def test_play_casts_to_float32_and_keeps_shape(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    player.play(np.linspace(-1.0, 1.0, 100))  # float64
    assert fake.play_calls[-1]["data"].dtype == np.float32
    assert fake.play_calls[-1]["data"].shape == (100,)
    player.play([0.0, 0.1, 0.2])  # a plain list works too
    assert fake.play_calls[-1]["data"].shape == (3,)
    player.play(np.zeros((50, 2), dtype=np.float32))  # (frames, channels) passes through
    assert fake.play_calls[-1]["data"].shape == (50, 2)
    with pytest.raises(ValueError):
        player.play(np.float32(0.5))  # 0-D
    with pytest.raises(ValueError):
        player.play(np.zeros((2, 2, 2), dtype=np.float32))


def test_stop_stops_and_playing_reflects_it(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    assert player.playing is False
    player.stop()  # before any play: no sounddevice call
    assert fake.stop_calls == 0

    player.play(np.zeros(480, dtype=np.float32))
    assert player.playing is True
    player.stop()
    assert fake.stop_calls == 1
    assert player.playing is False
    # The closed stream now raises on .active; playing must stay False, not raise.
    assert fake.stream.closed is True
    with pytest.raises(RuntimeError):
        fake.stream.active
    assert player.playing is False
    player.stop()  # idempotent
    assert fake.stop_calls == 1


def test_playing_false_once_samples_have_run_out(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    player.play(np.zeros(480, dtype=np.float32))
    assert player.playing is True
    fake.stream.finish()
    assert player.playing is False
    assert fake.stop_calls == 0


def test_playing_guards_missing_stream(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    player.play(np.zeros(480, dtype=np.float32))
    fake.stream = None  # as if someone else's stop() discarded it
    assert player.playing is False


def test_second_play_replaces_the_first(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    player.play(np.zeros(480, dtype=np.float32))
    first = fake.stream
    player.play(np.ones(480, dtype=np.float32) * 0.1)
    assert fake.stream is not first
    assert first.closed is True
    assert player.playing is True
    assert len(fake.play_calls) == 2


def test_play_empty_only_stops(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)
    player = TonePlayer()
    player.play(np.zeros(480, dtype=np.float32))
    assert player.playing is True
    player.play(np.zeros(0, dtype=np.float32))
    assert len(fake.play_calls) == 1
    assert player.playing is False
    assert fake.stream.closed is True


def test_play_errors_propagate_and_leave_player_stopped(monkeypatch) -> None:
    fake = install_fake_sd(monkeypatch)

    def failing_play(data: Any, **kwargs: Any) -> None:
        raise OSError("Error opening OutputStream: Device unavailable")

    fake.play = failing_play  # type: ignore[attr-defined]
    player = TonePlayer()
    with pytest.raises(OSError):
        player.play(np.zeros(480, dtype=np.float32))
    assert player.playing is False
    player.stop()  # nothing started, so nothing to stop
    assert fake.stop_calls == 0


# -- smoke test against the real sounddevice --------------------------------------------------


def test_real_output_smoke() -> None:
    """Play 20 ms of silence through the default output; skip when there is none."""
    try:
        import sounddevice  # noqa: F401  (may fail without PortAudio)
    except Exception as exc:  # ImportError or OSError from the PortAudio loader
        pytest.skip(f"sounddevice not importable: {exc}")
    player = TonePlayer()
    try:
        player.play(np.zeros(FS // 50, dtype=np.float32))
    except Exception as exc:  # no output device (CI), unsupported rate, ...
        pytest.skip(f"sounddevice could not open an output stream: {exc}")
    assert isinstance(player.playing, bool)
    player.stop()
    assert player.playing is False
