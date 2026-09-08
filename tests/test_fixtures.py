"""Regression tests against the recorded WAV fixtures (docs/PLAN.md section 8).

Both fixtures are decoded with the contract defaults: ``decode_wav`` builds a
``Pipeline`` whose ``ToneDetector()`` takes no threshold overrides.  The run
lengths asserted here are the ones the version 2 detector measures on the
recordings; the tolerances reflect how it works.  It sees the attack within
one 10 ms block but the release only once the room tail has decayed to about
24 dB above the noise, so every mark comes out a few tens of milliseconds
longer than keyed and every gap the same amount shorter.  The decoder's
offset ``d`` undoes exactly that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from morse.pipeline import decode_wav, load_wav
from morse.runs import Run

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOOPBACK = FIXTURES / "loopback_sos_1khz_15wpm.wav"
BEEPER = FIXTURES / "beeper_long_2491hz_1m.wav"

MARK_LAG_MS = (-100.0, 80.0)
"""Allowed (measured - nominal) for an ON run. The nominal figures came from a midpoint
threshold; the signal-referenced release lets go 12 dB below the tone, trimming up to
70 ms of tail from each mark, so marks may run short as well as long."""
GAP_LAG_MS = (-80.0, 100.0)
"""Allowed (measured - nominal) for an OFF run between marks: gaps may run short."""
START_TOLERANCE_MS = 60.0


def _on_runs(runs: list[Run]) -> list[Run]:
    return [r for r in runs if r.on]


def _gaps_between_on_runs(runs: list[Run]) -> list[float]:
    """OFF run lengths strictly between the first and the last ON run."""
    on_idx = [i for i, r in enumerate(runs) if r.on]
    return [runs[i].ms for i in range(on_idx[0] + 1, on_idx[-1]) if not runs[i].on]


def _within(got: float, want: float, window: tuple[float, float]) -> bool:
    lo, hi = window
    return lo <= got - want <= hi


def test_fixtures_are_48k_mono_int16():
    expected_seconds = {LOOPBACK: 4.4, BEEPER: 12.0}
    for path in (LOOPBACK, BEEPER):
        fs, raw = wavfile.read(str(path))
        assert fs == 48000, path.name
        assert raw.dtype == np.int16, (path.name, raw.dtype)
        assert raw.ndim == 1, (path.name, raw.shape)
        assert raw.size / fs == expected_seconds[path], (path.name, raw.size / fs)

        fs2, x = load_wav(path)
        assert fs2 == fs
        assert x.dtype == np.float32 and x.shape == raw.shape
        assert 0.0 < float(np.abs(x).max()) < 1.0
        assert np.array_equal(x, raw.astype(np.float32) / 32768.0)


def test_loopback_sos_decodes():
    text, runs = decode_wav(str(LOOPBACK), 1000.0)
    assert text.strip() == "SOS"
    assert all(a.on != b.on for a, b in zip(runs, runs[1:]))
    assert not runs[0].on and not runs[-1].on

    # 9 marks (3 letters of 3 symbols) keyed 80 / 240 ms at 15 WPM.
    marks = [r.ms for r in _on_runs(runs)]
    assert len(marks) == 9, marks
    dits = marks[0:3] + marks[6:9]
    dahs = marks[3:6]
    assert all(_within(ms, 80.0, MARK_LAG_MS) for ms in dits), dits
    assert all(_within(ms, 240.0, MARK_LAG_MS) for ms in dahs), dahs

    # 8 gaps: six inside letters (keyed 80 ms) and two between letters (keyed 240 ms).
    gaps = _gaps_between_on_runs(runs)
    assert len(gaps) == 8, gaps
    inner = gaps[0:2] + gaps[3:5] + gaps[6:8]
    letter = [gaps[2], gaps[5]]
    assert all(_within(ms, 80.0, GAP_LAG_MS) and ms >= 20.0 for ms in inner), inner
    assert all(_within(ms, 240.0, GAP_LAG_MS) for ms in letter), letter
    assert len(runs) == 19  # lead-in, 9 marks, 8 gaps, trailing silence: no chatter


def test_beeper_long_capture_has_four_tones():
    _, runs = decode_wav(str(BEEPER), 2491.0)
    on = _on_runs(runs)
    lengths = [r.ms for r in on]
    assert len(on) == 4, f"expected 4 ON runs, got {len(on)}: {lengths}"

    expected_on = [3700.0, 490.0, 570.0, 2020.0]
    for got, want in zip(lengths, expected_on):
        assert _within(got, want, MARK_LAG_MS), f"ON runs {lengths} vs {expected_on}"

    gaps = _gaps_between_on_runs(runs)
    expected_off = [370.0, 250.0, 300.0]
    assert len(gaps) == 3, gaps
    for got, want in zip(gaps, expected_off):
        assert _within(got, want, GAP_LAG_MS), f"gaps {gaps} vs {expected_off}"

    # The first tone starts near 0.69 s and there is no chatter in the
    # leading silence: everything before the first ON run is one OFF run.
    first_on = next(i for i, r in enumerate(runs) if r.on)
    assert first_on == 1
    assert abs(runs[0].ms - 690.0) <= START_TOLERANCE_MS, runs[0].ms

    # Nothing but silence after the last tone (about 3.6 s of quiet room).
    last_on = max(i for i, r in enumerate(runs) if r.on)
    assert all(not r.on for r in runs[last_on + 1:])
    assert sum(r.ms for r in runs[last_on + 1:]) > 3000.0
    assert len(runs) == 9


def test_beeper_capture_is_not_morse_but_decodes_without_crashing():
    text, runs = decode_wav(str(BEEPER), 2491.0)
    assert isinstance(text, str)
    assert len(runs) >= 8
    # Two dahs and a dit-dah are what the four tones look like as Morse; the
    # decoder must not have produced an unknown-symbol marker for them.
    assert "?" not in text
