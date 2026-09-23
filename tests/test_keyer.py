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
