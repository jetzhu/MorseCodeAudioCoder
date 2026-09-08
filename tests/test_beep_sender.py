"""Tests for tools/beep_sender.py (standalone script, imported via importlib)."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "beep_sender.py"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

LINE_RE = re.compile(r"^\((True|False), (\d+)\)$")


def load_script():
    spec = importlib.util.spec_from_file_location("beep_sender", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = load_script()

# 'SOS' at 10 WPM: dit 120, dah 360, letter gap 360.
SOS_10WPM = [
    (True, 120), (False, 120), (True, 120), (False, 120), (True, 120),   # S
    (False, 360),
    (True, 360), (False, 120), (True, 360), (False, 120), (True, 360),   # O
    (False, 360),
    (True, 120), (False, 120), (True, 120), (False, 120), (True, 120),   # S
]


# -- build_timing -------------------------------------------------------------


def test_sos_at_10_wpm_exact() -> None:
    assert bs.build_timing("SOS", 10) == SOS_10WPM


def test_word_gap_is_seven_dits() -> None:
    assert bs.build_timing("E E", 10) == [(True, 120), (False, 840), (True, 120)]


def test_multiple_spaces_and_surrounding_whitespace_make_one_word_gap() -> None:
    assert bs.build_timing("  E \t\n  E  ", 10) == [(True, 120), (False, 840), (True, 120)]


def test_two_words_share_one_word_gap() -> None:
    seq = bs.build_timing("SOS SOS", 10)
    assert seq == SOS_10WPM + [(False, 840)] + SOS_10WPM
    assert seq.count((False, 840)) == 1


def test_case_insensitive_and_unknown_characters_skipped() -> None:
    assert bs.build_timing("sos", 10) == SOS_10WPM
    assert bs.build_timing("S#O~S", 10) == SOS_10WPM
    assert bs.build_timing("##", 10) == []
    assert bs.build_timing("", 10) == []
    # A word made entirely of unknown characters does not leave a stray gap.
    assert bs.build_timing("E ## E", 10) == [(True, 120), (False, 840), (True, 120)]


def test_no_leading_trailing_or_adjacent_gaps() -> None:
    seq = bs.build_timing("HELLO WORLD 73", 12)
    assert seq[0][0] is True and seq[-1][0] is True
    assert all(a[0] != b[0] for a, b in zip(seq, seq[1:]))  # strictly alternating


def test_dit_length_scales_with_wpm() -> None:
    assert bs.build_timing("E", 20) == [(True, 60)]
    assert bs.build_timing("E", 8) == [(True, 150)]
    assert bs.build_timing("T", 8) == [(True, 450)]
    assert bs.build_timing("E", 7) == [(True, 171)]  # 171.43 rounded
    assert bs.build_timing("E E", 7) == [(True, 171), (False, 1200), (True, 171)]  # 1200.0


def test_table_covers_letters_digits_and_punctuation() -> None:
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/='():;+-_\"@!":
        assert ch in bs.MORSE_TABLE, ch
    assert bs.MORSE_TABLE["S"] == "..." and bs.MORSE_TABLE["O"] == "---"
    assert len(set(bs.MORSE_TABLE.values())) == len(bs.MORSE_TABLE)  # no duplicate codes


def test_repeat_timing_inserts_word_gap_between_copies() -> None:
    assert bs.repeat_timing(SOS_10WPM, 1, 10) == SOS_10WPM
    assert bs.repeat_timing(SOS_10WPM, 2, 10) == SOS_10WPM + [(False, 840)] + SOS_10WPM
    assert bs.repeat_timing([], 3, 10) == []


def test_unknown_characters_helper() -> None:
    assert bs.unknown_characters("sos") == ""
    assert bs.unknown_characters("a#b#c~ d") == "#~"


def test_script_does_not_import_the_morse_package() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+morse\b", source, re.MULTILINE)


# -- command line ---------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )


def test_dry_run_prints_one_tuple_per_line_and_exits_zero() -> None:
    result = _run("SOS", "--wpm", "10", "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == len(SOS_10WPM)
    for line in lines:
        assert LINE_RE.match(line), line
    parsed = [(m.group(1) == "True", int(m.group(2))) for m in map(LINE_RE.match, lines)]
    assert parsed == SOS_10WPM
    assert lines[0] == "(True, 120)"


def test_dry_run_repeat_expands_sequence() -> None:
    result = _run("SOS", "--wpm", "10", "--dry-run", "--repeat", "2")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2 * len(SOS_10WPM) + 1
    assert lines[len(SOS_10WPM)] == "(False, 840)"


def test_dry_run_joins_positional_words() -> None:
    quoted = _run("E E", "--wpm", "10", "--dry-run")
    unquoted = _run("E", "E", "--wpm", "10", "--dry-run")
    assert quoted.returncode == 0 and unquoted.returncode == 0
    assert quoted.stdout == unquoted.stdout
    assert quoted.stdout.splitlines() == ["(True, 120)", "(False, 840)", "(True, 120)"]


def test_dry_run_default_wpm_is_8() -> None:
    result = _run("E", "--dry-run")
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["(True, 150)"]


def test_unknown_characters_are_reported_on_stderr_and_skipped() -> None:
    result = _run("S#O#S", "--wpm", "10", "--dry-run")
    assert result.returncode == 0
    assert "#" in result.stderr
    assert len(result.stdout.splitlines()) == len(SOS_10WPM)


def test_nothing_encodable_is_an_error() -> None:
    result = _run("###", "--dry-run")
    assert result.returncode == 1
    assert result.stdout == ""


@pytest.mark.parametrize(
    "args",
    [
        ("SOS", "--wpm", "0"),
        ("SOS", "--freq", "0"),
        ("SOS", "--freq", "-5"),
        ("SOS", "--freq", "0.3"),  # rounds to 0 Hz
        ("SOS", "--freq", "abc"),
        ("SOS", "--freq", "nan"),
        ("SOS", "--repeat", "0"),
    ],
)
def test_invalid_arguments_exit_with_usage_error(args: tuple[str, ...]) -> None:
    result = _run(*args, "--dry-run")
    assert result.returncode == 2
    assert "error" in result.stderr


# -- --freq: float accepted and rounded; winsound range only for winsound ----------


def test_freq_accepts_float_and_rounds_to_int() -> None:
    # The decoder's --freq is a float; a value copied from its auto-detect
    # readout (2490.97 Hz, docs/PLAN.md section 8) must work here unchanged.
    for text in ("2491", "2491.0", "2490.97", "2491.4"):
        freq = bs.parse_args(["SOS", "--freq", text]).freq
        assert freq == 2491, text
        assert type(freq) is int, text
    assert bs.parse_args(["E", "--freq", "440.6"]).freq == 441
    assert bs.parse_args(["E"]).freq == bs.DEFAULT_FREQ == 2491


def test_freq_float_on_the_command_line_dry_run_exits_zero() -> None:
    for text in ("2491.0", "2490.97"):
        result = _run("SOS", "--wpm", "10", "--freq", text, "--dry-run")
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.splitlines()) == len(SOS_10WPM)


def test_dry_run_ignores_winsound_frequency_range() -> None:
    # --dry-run makes no sound, so winsound's 37..32767 Hz limit does not apply.
    for text in ("20", "36", "32768", "40000"):
        result = _run("E", "--freq", text, "--dry-run")
        assert result.returncode == 0, (text, result.stderr)
        assert result.stdout.splitlines() == ["(True, 150)"]


def test_backend_frequency_error_only_for_winsound() -> None:
    assert bs.backend_frequency_error(bs.WINSOUND_BACKEND, 2491) is None
    assert bs.backend_frequency_error(bs.WINSOUND_BACKEND, 37) is None
    assert bs.backend_frequency_error(bs.WINSOUND_BACKEND, 32767) is None
    for bad in (1, 36, 32768, 40000):
        message = bs.backend_frequency_error(bs.WINSOUND_BACKEND, bad)
        assert message and "37..32767" in message, bad
    # The beep command and the print-only backend enforce nothing here.
    for name in ("/usr/bin/beep -f FREQ -l MS", "none"):
        for freq in (1, 20, 2491, 40000):
            assert bs.backend_frequency_error(name, freq) is None, (name, freq)


def test_winsound_backend_rejects_out_of_range_before_beeping(monkeypatch, capsys) -> None:
    beeps: list[tuple[int, int]] = []
    fake_winsound = type(sys)("winsound")
    fake_winsound.Beep = lambda freq, ms: beeps.append((freq, ms))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
    monkeypatch.setattr(bs.sys, "platform", "win32")
    monkeypatch.setattr(bs.time, "sleep", lambda s: None)

    assert bs.main(["E", "--freq", "20"]) == 2
    assert beeps == []
    assert "37..32767" in capsys.readouterr().err

    assert bs.main(["E", "--freq", "440.4"]) == 0
    assert beeps == [(440, 150)]


def test_beep_command_backend_passes_low_frequency_through(monkeypatch, capsys) -> None:
    # Simulated Linux with the `beep` command: its own range (0 < f <= 20000)
    # differs from winsound's, so the script must not apply 37..32767 there.
    commands: list[list[str]] = []
    monkeypatch.setattr(bs.sys, "platform", "linux")
    monkeypatch.setattr(bs.shutil, "which", lambda name: "/usr/bin/beep" if name == "beep" else None)
    monkeypatch.setattr(bs.subprocess, "run", lambda cmd, check: commands.append(list(cmd)))
    monkeypatch.setattr(bs.time, "sleep", lambda s: None)

    assert bs.main(["E", "--freq", "20"]) == 0
    assert commands == [["/usr/bin/beep", "-f", "20", "-l", "150"]]
    assert "via /usr/bin/beep" in capsys.readouterr().err


def test_no_backend_prints_timing_regardless_of_frequency(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bs.sys, "platform", "linux")
    monkeypatch.setattr(bs.shutil, "which", lambda name: None)

    assert bs.main(["E", "--freq", "20"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["(True, 150)"]
    assert "no tone backend" in captured.err


def test_value_error_from_backend_is_reported_as_beep_failure(monkeypatch, capsys) -> None:
    def failing_beep(freq: int, ms: int) -> None:
        raise ValueError("frequency must be in 37 thru 32767")

    monkeypatch.setattr(bs, "select_backend", lambda: ("fake", failing_beep))
    assert bs.main(["E"]) == 1
    assert "beep failed" in capsys.readouterr().err


def test_runtime_error_from_backend_is_reported_as_beep_failure(monkeypatch, capsys) -> None:
    def failing_beep(freq: int, ms: int) -> None:
        raise RuntimeError("Failed to beep")

    monkeypatch.setattr(bs, "select_backend", lambda: ("fake", failing_beep))
    assert bs.main(["E"]) == 1
    assert "beep failed" in capsys.readouterr().err


def test_main_callable_in_process_dry_run(capsys) -> None:
    assert bs.main(["SOS", "--wpm", "10", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"({on}, {ms})" for on, ms in SOS_10WPM]


def test_play_uses_beep_for_marks_and_sleep_for_gaps(monkeypatch) -> None:
    calls: list[tuple[str, int | float]] = []
    monkeypatch.setattr(bs.time, "sleep", lambda s: calls.append(("sleep", round(s * 1000))))
    bs.play([(True, 120), (False, 360), (True, 360)], 2491, lambda f, ms: calls.append(("beep", ms)))
    assert calls == [("beep", 120), ("sleep", 360), ("beep", 360)]
