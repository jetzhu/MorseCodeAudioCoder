"""Tests for morse.keyer: the pure Keyer state machine, the envelope renderer and LiveKey.

LiveKey is exercised against a fake ``sounddevice`` whose OutputStream hands
the callback back to the test, so PortAudio is never loaded.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from morse.decoder import MorseDecoder
from morse.keyer import Keyer, LiveKey, envelope
from morse.runs import Run

T = 100.0  # dit length used below (12 WPM)


def runs_from(transitions: list[tuple[float, bool]], end_ms: float) -> list[Run]:
    """Turn a transition log into detector-style runs, closing the last one at ``end_ms``."""
    out: list[Run] = []
    for (t0, on), (t1, _) in zip(transitions, transitions[1:] + [(end_ms, None)]):
        out.append(Run(on, max(1, round((t1 - t0) / 10.0))))
    return out


def decode(transitions: list[tuple[float, bool]], end_ms: float) -> str:
    dec = MorseDecoder(wpm=1200.0 / T)
    text = "".join(dec.feed(r) for r in runs_from(transitions, end_ms))
    return (text + dec.idle(float("inf"))).strip()


# ------------------------------------------------------------ straight key


def test_straight_key_follows_the_hand():
    k = Keyer(T)
    assert k.key_down(10.0) == [(10.0, True)]
    assert k.key_down(20.0) == []  # already down
    assert k.state_at(15.0) is True
    assert k.key_up(250.0) == [(250.0, False)]
    assert k.key_up(260.0) == []
    assert k.state_at(300.0) is False
    assert k.next_wakeup_ms is None
    assert k.tick(1000.0) == []  # nothing automatic in straight mode


def test_straight_key_ignores_paddles_and_vice_versa():
    k = Keyer(T)
    assert k.paddle_down("dit", 0.0) == []
    p = Keyer(T, "paddle")
    assert p.key_down(0.0) == []


def test_straight_key_hand_keyed_sos_decodes():
    k = Keyer(T)
    t = 0.0
    trans: list[tuple[float, bool]] = []
    for symbol in "... --- ...":
        if symbol == " ":
            t += 2 * T  # letter gap: 3T in total with the trailing element gap
            continue
        trans += k.key_down(t)
        t += T if symbol == "." else 3 * T
        trans += k.key_up(t)
        t += T
    assert decode(trans, t + 1000.0) == "SOS"


# ----------------------------------------------------------------- paddles


def test_a_tap_sends_one_full_element():
    k = Keyer(T, "paddle")
    assert k.paddle_down("dit", 0.0) == [(0.0, True), (T, False)]
    assert k.paddle_up("dit", 30.0) == []  # released long before the dit ends
    assert k.next_wakeup_ms == T
    assert k.tick(T) == []  # element over, gap begins
    assert k.next_wakeup_ms == 2 * T
    assert k.tick(2 * T) == []  # nothing held: idle
    assert k.next_wakeup_ms is None
    assert k.state_at(50.0) is True and k.state_at(150.0) is False


def test_holding_a_paddle_repeats_with_one_dit_gaps():
    k = Keyer(T, "paddle")
    trans = k.paddle_down("dah", 0.0)
    while len(trans) < 8:  # a tick at an element's end only opens the gap; the next one starts the element
        trans += k.tick(k.next_wakeup_ms)
    assert trans == [
        (0.0, True), (3 * T, False),
        (4 * T, True), (7 * T, False),
        (8 * T, True), (11 * T, False),
        (12 * T, True), (15 * T, False),
    ]
    k.paddle_up("dah", 12.5 * T)  # released during the fourth dah
    assert k.tick(15 * T) == []  # it completes
    assert k.tick(16 * T) == []  # and nothing follows
    assert k.next_wakeup_ms is None


def test_a_late_tick_catches_up_element_by_element():
    k = Keyer(T, "paddle")
    k.paddle_down("dit", 0.0)
    trans = k.tick(4.5 * T)  # three elements overdue
    assert trans == [(2 * T, True), (3 * T, False), (4 * T, True), (5 * T, False)]


def test_paddle_memory_makes_a_clean_a():
    k = Keyer(T, "paddle")
    trans = k.paddle_down("dit", 0.0)
    trans += k.paddle_down("dah", 40.0)  # tapped while the dit sounds: remembered
    k.paddle_up("dah", 60.0)
    k.paddle_up("dit", 70.0)
    trans += k.tick(2 * T)  # gap over: the remembered dah goes out
    assert trans == [(0.0, True), (T, False), (2 * T, True), (5 * T, False)]
    assert k.tick(6 * T) == []
    assert decode(trans, 10 * T) == "A"


def test_holding_both_paddles_alternates():
    k = Keyer(T, "paddle")
    trans = k.paddle_down("dit", 0.0)
    k.paddle_down("dah", 10.0)
    while len(trans) < 8:
        trans += k.tick(k.next_wakeup_ms)
    lengths = [b - a for (a, _), (b, _) in zip(trans[::2], trans[1::2])]
    assert lengths == [T, 3 * T, T, 3 * T]


def test_release_all_and_set_mode_end_a_sounding_tone():
    k = Keyer(T, "paddle")
    k.paddle_down("dah", 0.0)  # logs the ON edge and the OFF edge at 3T
    assert k.release_all(50.0) == [(50.0, False)]  # the future OFF edge is dropped, the tone ends now
    assert k.state_at(60.0) is False
    assert k.transitions_since(0)[0] == [(0.0, True), (50.0, False)]
    assert k.tick(1000.0) == [] and k.next_wakeup_ms is None
    s = Keyer(T)
    s.key_down(0.0)
    assert s.set_mode("paddle", 30.0) == [(30.0, False)]
    assert s.mode == "paddle"
    assert s.key_down(40.0) == []


def test_set_speed_applies_to_the_next_element():
    k = Keyer(T, "paddle")
    k.paddle_down("dit", 0.0)
    k.set_speed(60.0)
    assert k.tick(2 * T) == [(2 * T, True), (2 * T + 60.0, False)]


def test_log_is_monotonic_and_readable_incrementally():
    k = Keyer(T)
    k.key_down(100.0)
    k.key_up(50.0)  # out of order: clamped to the last time
    items, idx = k.transitions_since(0)
    assert items == [(100.0, True), (100.0, False)] and idx == 2
    assert k.transitions_since(idx) == ([], 2)
    assert k.transitions_in(0.0, 100.0) == []
    assert k.transitions_in(100.0, 101.0) == [(100.0, True), (100.0, False)]


def test_validation():
    with pytest.raises(ValueError):
        Keyer(0)
    with pytest.raises(ValueError):
        Keyer(T, "iambic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Keyer(T, "paddle").paddle_down("squeeze", 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Keyer(T, iambic="C")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Keyer(T, dah_ratio=1.5)
    with pytest.raises(ValueError):
        Keyer(T, weight=80)
    k = Keyer(T)
    with pytest.raises(ValueError):
        k.set_iambic("ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        k.set_weighting(weight=float("nan"))
    k.set_weighting(dah_ratio=None, weight=None)  # keeps everything
    assert (k.dah_ratio, k.weight) == (3.0, 50.0)


# ---------------------------------------------------------- iambic A and B


def _squeeze_release_during_dah(iambic: str) -> list[tuple[float, bool]]:
    """Squeeze both paddles from idle (dit first) and let go of both during the dah."""
    k = Keyer(T, "paddle", iambic=iambic)  # type: ignore[arg-type]
    trans = k.paddle_down("dit", 0.0)
    trans += k.paddle_down("dah", 30.0)  # squeezed during the dit: dah remembered
    trans += k.tick(2 * T)  # gap over: the dah starts at 2T and ends at 5T
    k.paddle_up("dit", 3 * T)
    k.paddle_up("dah", 3.5 * T)  # both released during the dah
    for _ in range(4):
        w = k.next_wakeup_ms
        if w is None:
            break
        trans += k.tick(w)
    return trans


def test_iambic_a_ends_with_the_element_in_progress():
    trans = _squeeze_release_during_dah("A")
    assert trans == [(0.0, True), (T, False), (2 * T, True), (5 * T, False)]
    assert decode(trans, 12 * T) == "A"


def test_iambic_b_adds_one_opposite_element_after_a_squeeze():
    trans = _squeeze_release_during_dah("B")
    assert trans == [(0.0, True), (T, False), (2 * T, True), (5 * T, False), (6 * T, True), (7 * T, False)]
    assert decode(trans, 12 * T) == "R"


def test_iambic_b_squeeze_released_during_the_first_element_gives_a_plain_a():
    k = Keyer(T, "paddle", iambic="B")
    trans = k.paddle_down("dit", 0.0)
    k.paddle_down("dah", 20.0)
    k.paddle_up("dit", 40.0)
    k.paddle_up("dah", 50.0)  # both let go while the dit still sounds
    while k.next_wakeup_ms is not None:
        trans += k.tick(k.next_wakeup_ms)
    assert trans == [(0.0, True), (T, False), (2 * T, True), (5 * T, False)]  # the remembered dah, no extra


def test_iambic_b_without_a_squeeze_behaves_like_a():
    k = Keyer(T, "paddle", iambic="B")
    trans = k.paddle_down("dah", 0.0)
    k.paddle_up("dah", 50.0)
    while k.next_wakeup_ms is not None:
        trans += k.tick(k.next_wakeup_ms)
    assert trans == [(0.0, True), (3 * T, False)]
    k.set_iambic("A")
    assert k.iambic == "A"


# ----------------------------------------------------------------- bug mode


def test_bug_dits_are_automatic_and_dahs_by_hand():
    k = Keyer(T, "bug")
    assert k.key_down(0.0) == []  # no straight key in bug mode
    trans = k.paddle_down("dit", 0.0)
    while len(trans) < 4:
        trans += k.tick(k.next_wakeup_ms)
    assert trans == [(0.0, True), (T, False), (2 * T, True), (3 * T, False)]
    k.paddle_up("dit", 2.5 * T)
    assert k.tick(4 * T) == [] and k.next_wakeup_ms is None
    # The dah lever is a straight key: on while held, any length.
    assert k.paddle_down("dah", 5 * T) == [(5 * T, True)]
    assert k.next_wakeup_ms is None and k.tick(6 * T) == []
    assert k.paddle_down("dah", 6 * T) == []  # already down
    assert k.paddle_up("dah", 8.7 * T) == [(8.7 * T, False)]
    assert k.paddle_up("dah", 9 * T) == []
    assert k.state_at(7 * T) is True and k.state_at(9 * T) is False


def test_bug_lever_cuts_a_dit_short_and_dits_resume_after_it():
    k = Keyer(T, "bug")
    trans = k.paddle_down("dit", 0.0)  # dit: ON at 0, OFF logged at T
    trans += k.paddle_down("dah", 0.5 * T)  # lever pressed mid-dit: the tone simply stays on
    assert trans == [(0.0, True), (T, False)]  # what the calls returned; the log below dropped the OFF edge
    assert k.transitions_since(0)[0] == [(0.0, True)]
    assert k.paddle_down("dit", 1.5 * T) == []  # dits wait while the lever is down (still held)
    trans += k.paddle_up("dah", 3 * T)
    assert trans[-1] == (3 * T, False)
    assert k.next_wakeup_ms == 4 * T  # one space, then dits resume because the dit paddle is held
    trans += k.tick(4 * T)
    assert trans[-2:] == [(4 * T, True), (5 * T, False)]
    k.paddle_up("dit", 4.5 * T)
    assert k.release_all(4.6 * T) == [(4.6 * T, False)]
    assert k.tick(10 * T) == []


def test_bug_memory_is_not_used():
    k = Keyer(T, "bug")
    k.paddle_down("dit", 0.0)
    k.paddle_down("dah", 0.3 * T)  # lever during a dit: manual tone
    k.paddle_up("dit", 0.4 * T)
    trans = k.paddle_up("dah", 2 * T)
    assert trans == [(2 * T, False)]
    assert k.next_wakeup_ms is None  # nothing remembered, nothing held


# ---------------------------------------------------------------- weighting


def test_weighting_shapes_marks_and_spaces_but_keeps_the_period():
    k = Keyer(T, "paddle", dah_ratio=4.0, weight=60.0)
    assert k.mark_ms("dit") == pytest.approx(1.2 * T)
    assert k.mark_ms("dah") == pytest.approx(4.2 * T)
    assert k.gap_ms == pytest.approx(0.8 * T)
    trans = k.paddle_down("dit", 0.0)
    while len(trans) < 4:
        trans += k.tick(k.next_wakeup_ms)
    assert trans == pytest.approx([(0.0, True), (1.2 * T, False), (2 * T, True), (3.2 * T, False)])
    k.release_all(3.5 * T)
    k.set_weighting(weight=40.0)  # lighter: shorter marks, longer spaces, same period
    trans = k.paddle_down("dah", 10 * T)
    while len(trans) < 4:
        trans += k.tick(k.next_wakeup_ms)
    assert trans == pytest.approx([(10 * T, True), (13.8 * T, False), (15 * T, True), (18.8 * T, False)])
    k.set_weighting(3.0, 50.0)
    assert (k.mark_ms("dit"), k.mark_ms("dah"), k.gap_ms) == (T, 3 * T, T)
    assert "dah_ratio=3" in repr(k) and "iambic='A'" in repr(k)


def test_livekey_passes_keyer_variants_through(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    keyer = Keyer(T, "paddle")
    lk = LiveKey(keyer, f0=1000.0)
    lk.set_iambic("B")
    lk.set_weighting(3.5, 55.0)
    assert keyer.iambic == "B" and keyer.dah_ratio == 3.5 and keyer.weight == 55.0
    lk.set_mode("bug")
    assert keyer.mode == "bug"


# ---------------------------------------------------------------- envelope


def test_envelope_ramps_at_every_edge():
    fs = 48000
    env = envelope([(10.0, True), (50.0, False)], False, 0.0, 480 * 10, fs, ramp_ms=3.0)
    assert env.dtype == np.float32
    assert env[0] == 0.0
    assert env[int(0.020 * fs)] == 1.0  # fully on after the 3 ms ramp
    mid_on = env[int(0.0115 * fs)]
    assert 0.0 < mid_on < 1.0  # inside the rising ramp
    assert env[int(0.060 * fs)] == 0.0
    assert 0.0 < env[int(0.0515 * fs)] < 1.0  # inside the falling ramp
    assert np.all(np.diff(env[int(0.010 * fs):int(0.013 * fs)]) >= 0)


def test_envelope_honours_the_initial_state_and_later_transitions():
    fs = 48000
    assert np.all(envelope([], True, 0.0, 100, fs) == 1.0)
    assert np.all(envelope([], False, 0.0, 100, fs) == 0.0)
    env = envelope([(500.0, True)], False, 0.0, 480, fs)  # transition after the window
    assert np.all(env == 0.0)


# ----------------------------------------------------------------- LiveKey


class FakeOutputStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def install_fake_sd(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake = types.ModuleType("sounddevice")
    fake.streams = []  # type: ignore[attr-defined]

    def output_stream(**kwargs: Any) -> FakeOutputStream:
        s = FakeOutputStream(**kwargs)
        fake.streams.append(s)  # type: ignore[attr-defined]
        return s

    fake.OutputStream = output_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_livekey_never_imports_sounddevice_until_started(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import would raise
    lk = LiveKey(Keyer(T), f0=2491.0)
    assert lk.running is False and lk.is_on() is False
    lk.key_down()  # harmless while stopped
    assert lk.feed_block(0.0, 480).shape == (480,)


def test_livekey_renders_the_keyed_tone_in_the_callback_and_the_feed(monkeypatch):
    fake = install_fake_sd(monkeypatch)
    lk = LiveKey(Keyer(T), f0=2491.0, fs=48000, amplitude=0.3)
    lk.start()
    stream = fake.streams[0]  # type: ignore[attr-defined]
    assert stream.started and stream.kwargs["samplerate"] == 48000 and stream.kwargs["channels"] == 1
    out = np.zeros((480, 1), dtype=np.float32)
    stream.callback(out, 480, None, None)  # silent before any press
    assert np.all(out == 0.0)
    lk.key_down()
    assert lk.is_on() is True
    # The press is stamped at now_ms (a few ms after start); the callback's clock
    # sits at 10 ms, so within a couple of blocks the tone is fully up.
    for _ in range(3):
        stream.callback(out, 480, None, None)
    assert np.max(np.abs(out)) == pytest.approx(0.3, abs=0.02)
    feed = lk.feed_block(lk.now_ms(), 480)
    assert np.max(np.abs(feed)) == pytest.approx(0.3, abs=0.02)
    lk.key_up()
    assert lk.is_on() is False
    for _ in range(3):
        stream.callback(out, 480, None, None)
    assert np.max(np.abs(out)) == 0.0
    lk.stop()
    assert stream.closed and lk.running is False


def test_livekey_paddle_elements_are_driven_by_the_stream_clock(monkeypatch):
    fake = install_fake_sd(monkeypatch)
    keyer = Keyer(T, "paddle")
    lk = LiveKey(keyer, f0=1000.0)
    lk.start()
    stream = fake.streams[0]  # type: ignore[attr-defined]
    lk.paddle_down("dit")
    out = np.zeros((480, 1), dtype=np.float32)
    levels = []
    for _ in range(45):  # 450 ms of callbacks: a dit, a gap, another dit while held...
        stream.callback(out, 480, None, None)
        levels.append(float(np.max(np.abs(out))))
    on_blocks = [lvl > 0.1 for lvl in levels]
    # about 10 blocks on, 10 off, 10 on, 10 off (ramps blur the edges by a block)
    assert 8 <= sum(on_blocks[0:12]) <= 12
    assert sum(on_blocks[12:19]) <= 1
    assert 8 <= sum(on_blocks[19:32]) <= 12
    lk.paddle_up("dit")
    lk.stop()


def _dominant_hz(block: np.ndarray, fs: int) -> float:
    spec = np.abs(np.fft.rfft(block * np.hanning(len(block))))
    return float(np.argmax(spec) * fs / len(block))


def test_livekey_sidetone_goes_to_the_speakers_and_f0_to_the_feed(monkeypatch):
    fake = install_fake_sd(monkeypatch)
    lk = LiveKey(Keyer(T), f0=2491.0, fs=48000, amplitude=0.3, sidetone_hz=600.0)
    assert lk.speaker_hz == 600.0
    lk.start()
    stream = fake.streams[0]  # type: ignore[attr-defined]
    lk.key_down()
    out = np.zeros((4800, 1), dtype=np.float32)
    for _ in range(3):
        stream.callback(out, 4800, None, None)
    assert abs(_dominant_hz(out[:, 0], 48000) - 600.0) < 15
    feed = lk.feed_block(lk.now_ms() + 200.0, 4800)
    assert abs(_dominant_hz(feed, 48000) - 2491.0) < 15
    # Follow the tone again: the speakers move to f0; retuning f0 moves both.
    lk.set_sidetone(None)
    assert lk.sidetone_hz is None and lk.speaker_hz == 2491.0
    lk.set_frequency(1000.0)
    for _ in range(3):
        stream.callback(out, 4800, None, None)
    assert abs(_dominant_hz(out[:, 0], 48000) - 1000.0) < 15
    lk.set_sidetone(0)
    assert lk.sidetone_hz is None
    lk.set_sidetone(700)
    assert lk.speaker_hz == 700.0
    with pytest.raises(ValueError):
        lk.set_sidetone(30000)
    with pytest.raises(ValueError):
        LiveKey(Keyer(T), f0=1000.0, sidetone_hz=-1.0)
    assert "sidetone=700" in repr(lk)
    lk.stop()
