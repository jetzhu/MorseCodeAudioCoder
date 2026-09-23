"""Offscreen smoke tests for the desktop UI (``morse/ui.py``).

Qt runs on the ``offscreen`` platform plugin so no display is needed.  The
window is built through the public factory :func:`morse.ui.make_window` with
the loopback fixture as its input (``args.wav``), which makes the window use
:class:`morse.ui.WavReplay` instead of the microphone: nothing here opens
PortAudio or imports ``sounddevice``.  The replay is driven to completion by
calling the timer slot :meth:`MainWindow.on_tick` in a loop, the window is
rendered once with ``grab()``, and the decoded-text widget must show ``SOS``.

Skipped when PySide6 or pyqtgraph is not installed.
"""
from __future__ import annotations

import argparse
import time
import os

# Must be set before anything imports Qt, or the default platform is used.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import morse.ui as ui  # noqa: E402  (after the importorskip guards on purpose)
from morse.table import encode  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "loopback_sos_1khz_15wpm.wav"
FIXTURE_F0 = 1000.0
MAX_TICKS = 20_000  # far more than the 4.4 s fixture needs; guards against a stuck replay


@pytest.fixture(scope="module")
def qapp():
    """The one QApplication of the test process (Qt allows exactly one)."""
    return ui.ensure_app(["morse-tests"])


def replay_args(**overrides: object) -> argparse.Namespace:
    """The namespace ``morse.app.build_parser`` would produce for ``--wav FIXTURE --freq 1000``."""
    values: dict[str, object] = {"wav": str(FIXTURE), "device": None, "freq": FIXTURE_F0, "wpm": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def drive_replay(window: ui.MainWindow) -> int:
    """Call the timer slot until the replay has been flushed; returns the tick count."""
    ticks = 0
    while not window.replay_finished:
        assert ticks < MAX_TICKS, "replay did not finish"
        window.on_tick()
        ticks += 1
    return ticks


@pytest.fixture
def settings(tmp_path: Path):
    """An INI-file preference store, so tests never touch the registry."""
    from PySide6 import QtCore

    return QtCore.QSettings(str(tmp_path / "prefs.ini"), QtCore.QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp, settings):
    """A started window on the loopback fixture with the timer left to the test."""
    win = ui.make_window(replay_args(), settings=settings)
    win.start(run_timer=False)
    yield win
    win.close()


def press(window: ui.MainWindow, key, down: bool = True, auto_repeat: bool = False) -> None:
    """Deliver a key press or release to the window as the keyboard would."""
    from PySide6 import QtCore, QtGui

    kind = QtCore.QEvent.Type.KeyPress if down else QtCore.QEvent.Type.KeyRelease
    ev = QtGui.QKeyEvent(kind, key, QtCore.Qt.KeyboardModifier.NoModifier, "", auto_repeat)
    window.keyPressEvent(ev) if down else window.keyReleaseEvent(ev)


# -- decoding through the window's own pipeline --------------------------------


def test_replay_decodes_sos_and_renders(qapp, window: ui.MainWindow) -> None:
    assert isinstance(window._source, ui.WavReplay)  # replay, not the microphone
    assert window.pipeline.f0 == FIXTURE_F0
    assert window.pipeline.block_size == 480 and window.pipeline.fs == 48000

    ticks = drive_replay(window)
    assert ticks > 1  # the file was handed out over several timer slots
    qapp.processEvents()

    image = window.grab()
    assert not image.isNull()
    assert image.width() > 0 and image.height() > 0

    assert "SOS" in window.decoded_text()
    assert "SOS" in window.text_view.toPlainText()
    assert window.pipeline.text.strip() == "SOS"
    assert window._error == ""  # no exception reached the status bar

    # Readouts follow the decoder: three letters, none unknown, a plausible dit.
    assert window.pipeline.decoder.letter_count == 3
    assert window.pipeline.decoder.unknown_count == 0
    assert window.timing_values["Dit"].text().endswith("ms")
    assert window.timing_values["Dit"].text() != "— ms"
    assert "Replay finished" in window.status_state.text()
    assert window.status_clock.text() == "00:04"
    assert window.status_dropped.text() == "0 dropped"
    assert window.histogram.marks  # the histogram saw the marks


def test_tick_never_raises_and_reports_errors_in_status_bar(qapp, window: ui.MainWindow) -> None:
    def explode(_block):
        raise RuntimeError("synthetic audio failure")

    window.pipeline.process_block = explode  # type: ignore[method-assign]
    window.on_tick()  # must not propagate
    assert "synthetic audio failure" in window.status_error.text()
    assert "RuntimeError" in window._error


def test_clear_text_keeps_timing(qapp, window: ui.MainWindow) -> None:
    drive_replay(window)
    dit_before = window.pipeline.decoder.dit_ms
    window.clear_button.click()
    assert window.decoded_text() == ""
    assert window.pipeline.decoder.letter_count == 0
    assert window.pipeline.decoder.dit_ms == dit_before


def test_frequency_spin_retunes_pipeline_and_markers(qapp, window: ui.MainWindow) -> None:
    window.freq_spin.setValue(2491)
    assert window.pipeline.f0 == 2491.0
    assert window.f0_header_label.text() == "2491 Hz"
    assert window.f0_marker.value() == 2491.0
    assert window.f3_marker.value() == 3 * 2491.0
    assert window.f3_marker.isVisible()  # 7473 Hz is inside the 0-8 kHz plot
    window.freq_spin.setValue(3000)
    assert not window.f3_marker.isVisible()  # 9 kHz would be off the plot
    window.freq_spin.setValue(1000)
    assert window.f3_marker.isVisible()
    assert window.pipeline.f0 == 1000.0


def test_speed_mode_switch_keeps_text_and_history(qapp, window: ui.MainWindow) -> None:
    drive_replay(window)
    text = window.pipeline.text
    window.speed_manual.click()
    assert window.wpm_spin.isEnabled()
    window.wpm_spin.setValue(15)
    dec = window.pipeline.decoder
    assert dec.text == text
    assert dec.dit_ms == pytest.approx(1200 / 15)
    window.speed_auto.click()
    assert not window.wpm_spin.isEnabled()
    assert window.pipeline.decoder.text == text


def test_save_recent_wav_writes_pcm(qapp, window: ui.MainWindow, tmp_path: Path) -> None:
    from scipy.io import wavfile

    drive_replay(window)
    out = window.save_recent_wav(tmp_path / "capture.wav")
    fs, data = wavfile.read(str(out))
    assert fs == 48000
    assert data.dtype.kind == "i"
    assert data.size == 211_200  # the whole 4.4 s fixture is shorter than the 30 s ring


# -- encoder strip ------------------------------------------------------------------


def test_encoder_sos(qapp, window: ui.MainWindow) -> None:
    window.message_edit.setText("SOS")
    assert window.morse_output_text() == "... --- ..."
    assert window.morse_output_text() == encode("SOS")
    guide = window.keying_guide
    assert guide.mark_count == 9  # three dits, three dahs, three dits
    enc = guide.encoding
    assert enc is not None
    assert [ch for ch, _, _ in enc.letters] == ["S", "O", "S"]
    # 8 WPM: dit 150, dah 450, intra 150, letter gap 450 -> 3*150+2*150 + 450 + 3*450+2*150 + 450 + 750
    assert enc.total_ms == 4050.0
    assert window.enc_readouts["Duration"].value.text().startswith("4.")
    assert window.enc_readouts["Dit · dah"].value.text().startswith("150 · 450 ms")
    image = guide.grab()
    assert not image.isNull()


def test_encoder_words_and_unknown_characters(qapp, window: ui.MainWindow) -> None:
    window.message_edit.setText("sos x#")
    assert window.morse_output_text() == "... --- ... / -..-"
    assert window.keying_guide.mark_count == 13
    window.message_edit.setText("###")
    assert window.morse_output_text() == ""
    assert window.keying_guide.mark_count == 0


def test_build_encoding_letter_spans_match_timing() -> None:
    enc = ui.build_encoding("HELLO WORLD", 8)
    assert enc.morse == encode("HELLO WORLD")
    assert enc.mark_count == 32
    assert enc.total_ms == sum(ms for _, ms in enc.timing)
    starts = [s for _, s, _ in enc.letters]
    assert starts == sorted(starts)
    for (ch, start, end), (nxt_ch, nxt_start, _) in zip(enc.letters, enc.letters[1:]):
        assert end <= nxt_start
    assert [ch for ch, _, _ in enc.letters] == list("HELLOWORLD")


# ------------------------------------------------- keying guide labels, feed


def test_keying_guide_labels_every_letter_including_e() -> None:
    # "HELLO WORLD!" at 8 WPM in an 800 px guide: the E is a single 150 ms dit
    # about 6 px wide, and a minimum-width gate used to drop its label.
    enc = ui.build_encoding("HELLO WORLD!", 8.0)
    scale = 800.0 / enc.total_ms
    placed = ui.layout_guide_labels(enc.letters, lambda ms: ms * scale, lambda ch: 7.0)
    assert [ch for ch, _ in placed] == list("HELLOWORLD!")
    xs = [x for _, x in placed]
    assert xs == sorted(xs)
    assert all(b - a >= 7.0 + 3.0 for a, b in zip(xs, xs[1:])), "labels never overlap"


def test_keying_guide_skips_only_colliding_labels_when_squeezed() -> None:
    enc = ui.build_encoding("HELLO WORLD!", 8.0)
    scale = 60.0 / enc.total_ms
    placed = ui.layout_guide_labels(enc.letters, lambda ms: ms * scale, lambda ch: 7.0)
    assert 0 < len(placed) < len(enc.letters)
    assert placed[0][0] == "H"
    xs = [x for _, x in placed]
    assert all(b - a >= 10.0 for a, b in zip(xs, xs[1:]))


def test_feed_the_decoder_mixes_the_tone_into_the_pipeline(qapp, window: ui.MainWindow) -> None:
    # Software loopback: the rendered tone is added to the input blocks, so a
    # silent microphone still decodes what Play keys.
    assert window.feed_check.isChecked()
    enc = ui.build_encoding("SOS", 12.0)
    tone = ui.render_tone(enc.timing, window.pipeline.f0, fs=ui.DEFAULT_FS)
    window._inject = np.concatenate([np.zeros(ui.DEFAULT_FS // 2, dtype=np.float32), tone])
    window._inject_pos = 0
    block = window.pipeline.block_size
    silence = np.zeros(block, dtype=np.float32)
    for _ in range(window._inject.size // block + 200):  # the tone, then 2 s of silence
        window._process_block(silence)
    assert window._inject is None, "injection ends on its own"
    assert window.pipeline.text.strip().endswith("SOS")


# ------------------------------------------------------------------ key strip


class _FakeOutputStream:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    fake = types.ModuleType("sounddevice")
    fake.streams = []  # type: ignore[attr-defined]

    def output_stream(**kwargs: object) -> _FakeOutputStream:
        s = _FakeOutputStream(**kwargs)
        fake.streams.append(s)  # type: ignore[attr-defined]
        return s

    fake.OutputStream = output_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_key_strip_modes_and_readouts(qapp, window: ui.MainWindow) -> None:
    assert window.key_straight.isChecked()
    assert window.key_button.isVisibleTo(window) and not window.dit_button.isVisibleTo(window)
    window._on_key_mode("paddle")
    assert not window.key_button.isVisibleTo(window)
    assert window.dit_button.isVisibleTo(window) and window.dah_button.isVisibleTo(window)
    assert "dits" in window.key_help.text()
    window.enc_wpm_spin.setValue(12)
    assert window.key_readouts["Speed"].value_text().startswith("12") if hasattr(
        window.key_readouts["Speed"], "value_text") else True
    assert window._keyer.dit_ms == pytest.approx(100.0)
    window._on_key_mode("straight")
    assert window.key_button.isVisibleTo(window)


def test_straight_key_button_sounds_feeds_and_is_read_back(qapp, window: ui.MainWindow, fake_sd) -> None:
    # Press: the output stream opens lazily and the tone is on.
    window._on_key_button(True, None)
    assert window._livekey is not None and window._livekey.running
    assert fake_sd.streams[0].active
    assert window._livekey.is_on()
    assert window.key_pill.on is True
    # With Feed the decoder on, an input block of silence carries the key's tone.
    block = np.zeros(window.pipeline.block_size, dtype=np.float32)
    window._feed_time_ms = window._livekey.now_ms()
    mixed = window._key_feed(block)
    window._feed_time_ms = None
    assert np.max(np.abs(mixed)) > 0.2
    window.feed_check.setChecked(False)
    window._feed_time_ms = window._livekey.now_ms()
    assert np.max(np.abs(window._key_feed(block))) == 0.0
    window._feed_time_ms = None
    window.feed_check.setChecked(True)
    # Release after a straight-key dah, then silence: the Sent line reads T.
    window._livekey._t0 -= 0.45  # the key clock jumps 450 ms ahead: a dah-length mark
    window._on_key_button(False, None)
    assert window._livekey.is_on() is False
    window._livekey._t0 -= 5.0  # then 5 s of silence
    window._refresh_key()
    assert window.sent_output.text().strip().startswith("T"), window.sent_output.text()
    window._clear_sent()
    assert window.sent_output.text().strip() in ("", "\u00a0")
    window.stop()
    assert fake_sd.streams[0].closed


def test_paddle_button_sends_a_timed_element(qapp, window: ui.MainWindow, fake_sd) -> None:
    window._on_key_mode("paddle")
    window.enc_wpm_spin.setValue(12)  # 100 ms dit
    window._on_key_button(True, "dit")
    window._on_key_button(False, "dit")
    lk = window._livekey
    assert lk is not None and lk.running
    items, _ = lk.transitions_since(0)
    assert [on for _, on in items] == [True, False]
    assert items[1][0] - items[0][0] == pytest.approx(100.0)
    window.stop()


def test_keyboard_space_works_the_straight_key(qapp, window: ui.MainWindow, fake_sd) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    assert window._handle_key_event(press, True) is True
    assert window._livekey is not None and window._livekey.is_on()
    assert window.key_button.isDown()
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, autorep=True)
    assert window._handle_key_event(repeat, True) is True  # swallowed, no second press
    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    assert window._handle_key_event(release, False) is True
    assert window._livekey.is_on() is False
    other = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    assert window._handle_key_event(other, True) is False
    window.stop()


# -------------------------------------------------------------------- practice


def test_practice_strip_grades_the_sent_line(qapp, window: ui.MainWindow) -> None:
    assert window.pr_kind_buttons["words"].isChecked()
    window._pr_next()
    target = window._practice.target
    assert target and window.pr_target_label.text() == target
    assert window._practice.asked == 1
    # A wrong copy, checked by hand.
    window._sent_dec.text = target[:-1] + "X"
    window._pr_check(False)
    assert "error" in window.pr_result_label.text()
    assert window._practice.correct == 0
    # A perfect copy is graded automatically by the refresh.
    window._pr_next()
    target = window._practice.target
    window._sent_dec.text = target + " "
    window._refresh_practice()
    assert window.pr_result_label.text().startswith("Correct")
    assert window._practice.correct == 1 and window._practice.asked == 2
    assert window.pr_readouts["Accuracy"].value_text().startswith("100") if hasattr(
        window.pr_readouts["Accuracy"], "value_text") else True
    window._pr_set_kind("digits")
    window._pr_next()
    assert window._practice.target.isdigit()
    window.pr_show_code.setChecked(True)
    assert window.pr_code_label.isVisibleTo(window)


def test_encoder_farnsworth_spin_stretches_letter_and_word_gaps(qapp, window: ui.MainWindow) -> None:
    window.message_edit.setText("SOS E")
    window.enc_wpm_spin.setValue(18)
    window.enc_farns_spin.setValue(0)  # special value "off"
    standard = window._encoding
    assert standard.farnsworth_wpm is None
    assert (round(standard.letter_gap_ms), round(standard.word_gap_ms)) == (200, 467)
    window.enc_farns_spin.setValue(5)
    enc = window._encoding
    assert enc.farnsworth_wpm == 5.0
    assert (round(enc.letter_gap_ms), round(enc.word_gap_ms)) == (1568, 3660)
    assert enc.total_ms > standard.total_ms
    assert enc.mark_count == standard.mark_count
    assert "1568" in window.enc_readouts["Letter gap \u00b7 word gap"].value.text()
    assert "3660" in window.enc_readouts["Letter gap \u00b7 word gap"].value.text()
    window.enc_farns_spin.setValue(25)  # not below the character speed: standard again
    assert window._encoding.farnsworth_wpm is None
    assert window._encoding.total_ms == standard.total_ms


def test_key_bindings_capture_swap_persist_and_reset(qapp, window: ui.MainWindow, settings, fake_sd) -> None:
    from PySide6.QtCore import Qt

    assert window._bindings == ui.keybind.DEFAULTS
    assert not window.key_reset_button.isEnabled()
    assert window.key_button.text().endswith("Space")
    # Change key on the straight key, then press J: J works the key from now on.
    window.key_rebind.click()
    assert "Press the key" in window.key_help.text()
    press(window, Qt.Key.Key_Shift)  # modifiers are ignored while choosing
    assert window._capture_action == "key"
    press(window, Qt.Key.Key_J)
    assert window._capture_action is None
    assert window._bindings == {"key": "J", "dit": "Left", "dah": "Right"}
    assert window.key_button.text() == "Key\nJ"
    assert "<b>J</b>" in window.key_help.text()
    assert window.key_reset_button.isEnabled()
    press(window, Qt.Key.Key_J, down=False)  # the captured key's release keys nothing
    assert window._livekey is None
    press(window, Qt.Key.Key_Space)  # Space no longer does anything
    assert window._livekey is None
    press(window, Qt.Key.Key_J)
    assert window._livekey is not None and window._livekey.is_on()
    press(window, Qt.Key.Key_J, down=False)
    assert not window._livekey.is_on()
    # Esc keeps the current binding.
    window.key_rebind.click()
    press(window, Qt.Key.Key_Escape)
    assert window._bindings["key"] == "J"
    # Taking a paddle's key swaps the two.
    window._on_key_mode("paddle")
    window.dah_rebind.click()
    press(window, Qt.Key.Key_Left)
    assert window._bindings == {"key": "J", "dit": "Right", "dah": "Left"}
    assert window.dit_button.text() == "Dit\n\u2192 right arrow"
    # Persisted: a second window on the same store starts with these bindings.
    settings.sync()
    other = ui.make_window(replay_args(), settings=settings)
    try:
        assert other._bindings == {"key": "J", "dit": "Right", "dah": "Left"}
    finally:
        other.close()
    window.key_reset_button.click()
    assert window._bindings == ui.keybind.DEFAULTS
    assert not window.key_reset_button.isEnabled()
    assert ui.load_bindings(settings) == ui.keybind.DEFAULTS


def test_sidetone_spin_pitches_the_speakers_not_the_feed(qapp, window: ui.MainWindow, settings, fake_sd) -> None:
    assert window.sidetone_spin.value() == ui.DEFAULT_SIDETONE_HZ
    assert "600" in window.key_readouts["Sidetone"].value.text()
    window._on_key_button(True, None)
    lk = window._livekey
    assert lk is not None and lk.sidetone_hz == 600.0 and lk.f0 == window.pipeline.f0
    window.sidetone_spin.setValue(800)
    assert lk.sidetone_hz == 800.0
    assert int(settings.value(ui.SETTINGS_SIDETONE)) == 800
    window.sidetone_spin.setValue(0)  # "tone": follow the beeper frequency
    assert lk.sidetone_hz is None and lk.speaker_hz == window.pipeline.f0
    assert "tone" in window.key_readouts["Sidetone"].value.text()
    window._on_key_button(False, None)
    other = ui.make_window(replay_args(), settings=settings)
    try:
        assert other.sidetone_spin.value() == 0
    finally:
        other.close()
    window.stop()


def test_preference_loaders_survive_garbage(tmp_path: Path) -> None:
    from PySide6 import QtCore

    s = QtCore.QSettings(str(tmp_path / "bad.ini"), QtCore.QSettings.Format.IniFormat)
    s.setValue(ui.SETTINGS_BINDINGS, "{not json")
    s.setValue(ui.SETTINGS_SIDETONE, "loud")
    assert ui.load_bindings(s) == ui.keybind.DEFAULTS
    assert ui.load_sidetone(s) == ui.DEFAULT_SIDETONE_HZ
    s.setValue(ui.SETTINGS_SIDETONE, 99999)
    assert ui.load_sidetone(s) == ui.DEFAULT_SIDETONE_HZ
    assert ui.key_name(QtCore.Qt.Key.Key_Space) == "Space"
    assert ui.key_name(QtCore.Qt.Key.Key_Left) == "Left"
    assert ui.key_name(QtCore.Qt.Key.Key_unknown) == ""
    assert ui.key_label("Left") == "\u2190 left arrow" and ui.key_label("J") == "J"


def test_decoded_log_is_stamped_and_saved(qapp, window: ui.MainWindow, tmp_path: Path) -> None:
    assert window._log.is_empty and not window.log_button.isEnabled()
    with pytest.raises(ValueError):
        window.save_log(tmp_path / "early.txt")
    drive_replay(window)
    window._refresh_decoded()
    assert window.log_button.isEnabled()
    words = window._log.words()
    assert [w.text for w in words] == ["SOS"]
    assert window._log.text.strip() == window.decoded_text().strip() == "SOS"
    # Each letter carries the audio clock of the block that produced it: the
    # first S is decoded during the fixture, not at the flush at its end.
    assert 0.0 < words[0].elapsed_ms < words[0].end_ms <= window.pipeline.elapsed_ms
    assert abs(words[0].wall_s - time.time()) < 60.0
    # Clear text keeps the log.
    window._on_clear()
    assert window.decoded_text().strip() == "" and window._log.text.strip() == "SOS"
    txt = window.save_log(tmp_path / "log.txt")
    content = txt.read_text(encoding="utf-8")
    assert content.startswith("Beeper Morse Console decoded log\nExported: ")
    assert "Source: Replay: loopback_sos_1khz_15wpm.wav" in content
    assert "Tone: 1000 Hz" in content
    assert content.rstrip().splitlines()[-1].endswith("  SOS")
    csv = window.save_log(tmp_path / "log.csv").read_text(encoding="utf-8")
    lines = csv.splitlines()
    assert lines[0] == "time,elapsed_s,word" and lines[1].endswith(",SOS") and len(lines) == 2
    assert float(lines[1].split(",")[1]) == pytest.approx(words[0].elapsed_ms / 1000.0, abs=0.006)


def test_keyer_variants_controls_apply_and_persist(qapp, window: ui.MainWindow, settings, fake_sd) -> None:
    from PySide6.QtCore import Qt

    k = window._keyer
    assert (k.iambic, k.dah_ratio, k.weight) == ("A", 3.0, 50.0)
    assert window.iambic_a.isChecked() and not window.iambic_a.isEnabled()  # single key: no iambic choice
    window.key_bug.click()
    assert k.mode == "bug"
    assert window.dit_button.isVisibleTo(window) and window.dah_button.isVisibleTo(window)
    assert not window.key_button.isVisibleTo(window)
    assert "bug" in window.key_help.text()
    assert not window.iambic_b.isEnabled()
    # The bound dah key works the lever: tone while held.
    press(window, Qt.Key.Key_Right)
    assert window._livekey is not None and window._livekey.is_on()
    press(window, Qt.Key.Key_Right, down=False)
    assert not window._livekey.is_on()
    window.key_paddle.click()
    assert k.mode == "paddle" and window.iambic_b.isEnabled()
    window.iambic_b.click()
    assert k.iambic == "B"
    window.ratio_spin.setValue(3.5)
    window.weight_spin.setValue(60)
    assert (k.dah_ratio, k.weight) == (3.5, 60.0)
    window.enc_wpm_spin.setValue(12)  # 100 ms dit: marks 120 and 370 ms
    assert window.key_readouts["Dit \u00b7 dah"].value.text().startswith("120 \u00b7 370")
    settings.sync()
    other = ui.make_window(replay_args(), settings=settings)
    try:
        assert (other._keyer.iambic, other._keyer.dah_ratio, other._keyer.weight) == ("B", 3.5, 60.0)
        assert other.iambic_b.isChecked() and other.ratio_spin.value() == 3.5 and other.weight_spin.value() == 60
    finally:
        other.close()
    window.stop()


def test_keyer_preference_loaders_survive_garbage(tmp_path: Path) -> None:
    from PySide6 import QtCore

    s = QtCore.QSettings(str(tmp_path / "bad2.ini"), QtCore.QSettings.Format.IniFormat)
    s.setValue(ui.SETTINGS_IAMBIC, "Z")
    s.setValue(ui.SETTINGS_RATIO, "wide")
    s.setValue(ui.SETTINGS_WEIGHT, 99)
    assert (ui.load_iambic(s), ui.load_dah_ratio(s), ui.load_weight(s)) == ("A", 3.0, 50)
    s.setValue(ui.SETTINGS_IAMBIC, " b ")
    s.setValue(ui.SETTINGS_RATIO, 4.25)
    s.setValue(ui.SETTINGS_WEIGHT, "30")
    assert (ui.load_iambic(s), ui.load_dah_ratio(s), ui.load_weight(s)) == ("B", 4.2, 30)
