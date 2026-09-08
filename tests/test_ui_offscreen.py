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
import os

# Must be set before anything imports Qt, or the default platform is used.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

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
def window(qapp):
    """A started window on the loopback fixture with the timer left to the test."""
    win = ui.make_window(replay_args())
    win.start(run_timer=False)
    yield win
    win.close()


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
