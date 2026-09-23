#!/usr/bin/env python3
"""Send text as Morse code through the PC speaker of *this* machine.

Standalone and standard-library only: copy this single file to the beeper
machine and run it there. It must not import the ``morse`` package.

    python beep_sender.py "HELLO WORLD" [--wpm 8] [--freq 2491] [--dry-run] [--repeat N]

Timing (PARIS standard): dit = 1200 / wpm ms, dah = 3 dits, gap between the
symbols of one letter = 1 dit, gap between letters = 3 dits, gap between
words = 7 dits. With ``--repeat N`` the message is sent N times with a word
gap between repetitions.

``--freq`` takes any positive number of Hz (``2491``, ``2491.0`` or the
``2490.97`` the decoder's auto-detect reports) and rounds it to whole Hz,
which is what every backend needs.

Backends: on Windows ``winsound.Beep(freq, ms)`` keys each mark and
``time.sleep`` waits out each gap; ``winsound.Beep`` only accepts 37..32767 Hz,
so that range is enforced when, and only when, it is the backend in use.
Elsewhere the ``beep`` command is used when present (``beep -f FREQ -l MS``,
which applies its own limits); otherwise the timing is printed with a note.
``--dry-run`` prints one ``(on, ms)`` tuple per line and exits 0 without
touching any backend.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

# Kept inline on purpose: this script travels alone to another machine.
MORSE_TABLE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-",
    "'": ".----.", "(": "-.--.", ")": "-.--.-", ":": "---...", ";": "-.-.-.",
    "+": ".-.-.", "-": "-....-", "_": "..--.-", '"': ".-..-.", "@": ".--.-.",
    "!": "-.-.--",
}

DEFAULT_WPM = 8.0
DEFAULT_FREQ = 2491  # Hz, the measured beeper fundamental (docs/PLAN.md, section 8)

# winsound.Beep raises ValueError outside this range. It is a limit of that one
# backend, not of the script: --dry-run makes no sound, and the Linux `beep`
# command checks its own (different) range itself.
WINSOUND_BACKEND = "winsound.Beep"
WINSOUND_MIN_FREQ = 37
WINSOUND_MAX_FREQ = 32767

Timing = list[tuple[bool, int]]


def dit_ms(wpm: float) -> float:
    """Dit length in milliseconds for a given words-per-minute speed."""
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    return 1200.0 / wpm


def unknown_characters(text: str) -> str:
    """Characters of ``text`` (upper-cased) that have no Morse code, in order, unique."""
    seen: list[str] = []
    for ch in text.upper():
        if not ch.isspace() and ch not in MORSE_TABLE and ch not in seen:
            seen.append(ch)
    return "".join(seen)


def farnsworth_gaps(wpm: float, farnsworth: float | None) -> tuple[int, int]:
    """Letter and word gap in ms; ARRL Farnsworth spacing when ``farnsworth`` is below ``wpm``.

    ``ta = (60 c - 37.2 s) / (s c)`` seconds; letter gap ``3 ta / 19``, word
    gap ``7 ta / 19`` (same rule as ``morse.player.farnsworth_gaps``).
    """
    dit = dit_ms(wpm)
    if farnsworth is None or farnsworth >= wpm:
        return int(round(3 * dit)), int(round(7 * dit))
    if farnsworth <= 0:
        raise ValueError(f"farnsworth must be positive, got {farnsworth!r}")
    ta = (60.0 * wpm - 37.2 * farnsworth) / (farnsworth * wpm)
    return int(round(1000.0 * 3.0 * ta / 19.0)), int(round(1000.0 * 7.0 * ta / 19.0))


def build_timing(text: str, wpm: float, farnsworth: float | None = None) -> Timing:
    """Return the keying sequence for ``text`` as ``[(on, ms), ...]``.

    Case-insensitive. Characters without a Morse code are skipped. Whitespace
    of any kind and length separates words. The sequence starts with the first
    mark and ends with the last mark: no leading or trailing gap, and never two
    adjacent gaps. Durations are ``round(k * 1200 / wpm)`` for k in
    {1, 3, 7}, so at 10 WPM a dit is 120 ms, a dah 360 ms, a word gap 840 ms.
    A ``farnsworth`` speed below ``wpm`` stretches the letter and word gaps
    (:func:`farnsworth_gaps`) while the elements keep their speed.
    """
    dit = dit_ms(wpm)
    dit_i = int(round(dit))
    dah_i = int(round(3 * dit))
    letter_gap_i, word_gap_i = farnsworth_gaps(wpm, farnsworth)

    words: list[list[str]] = []
    for word in text.upper().split():
        codes = [MORSE_TABLE[ch] for ch in word if ch in MORSE_TABLE]
        if codes:
            words.append(codes)

    seq: Timing = []
    for wi, codes in enumerate(words):
        if wi > 0:
            seq.append((False, word_gap_i))
        for li, code in enumerate(codes):
            if li > 0:
                seq.append((False, letter_gap_i))
            for si, symbol in enumerate(code):
                if si > 0:
                    seq.append((False, dit_i))
                seq.append((True, dah_i if symbol == "-" else dit_i))
    return seq


def repeat_timing(seq: Timing, times: int, wpm: float) -> Timing:
    """Concatenate ``seq`` ``times`` times with a word gap between copies."""
    if times < 1 or not seq:
        return []
    gap = (False, int(round(7 * dit_ms(wpm))))
    out: Timing = []
    for i in range(times):
        if i > 0:
            out.append(gap)
        out.extend(seq)
    return out


def total_ms(seq: Sequence[tuple[bool, int]]) -> int:
    """Total duration of a keying sequence in milliseconds."""
    return sum(ms for _, ms in seq)


def format_timing(seq: Sequence[tuple[bool, int]]) -> str:
    """One ``(on, ms)`` tuple per line, e.g. ``(True, 120)``."""
    return "\n".join(f"({on}, {ms})" for on, ms in seq)


# -- playback backends -------------------------------------------------------

BeepFn = Callable[[int, int], None]


def select_backend() -> tuple[str, BeepFn | None]:
    """Pick the tone generator for this platform.

    Returns ``(name, beep)`` where ``beep(freq_hz, ms)`` blocks for ``ms``
    milliseconds, or ``(name, None)`` when nothing can make a sound here.
    ``name`` is :data:`WINSOUND_BACKEND` on Windows, the ``beep`` command line
    elsewhere when the command exists, and ``"none"`` otherwise.
    """
    if sys.platform == "win32":
        import winsound  # Windows only

        def beep_winsound(freq: int, ms: int) -> None:
            winsound.Beep(freq, ms)

        return WINSOUND_BACKEND, beep_winsound

    beep_cmd = shutil.which("beep")
    if beep_cmd:

        def beep_command(freq: int, ms: int) -> None:
            subprocess.run([beep_cmd, "-f", str(freq), "-l", str(ms)], check=False)

        return f"{beep_cmd} -f FREQ -l MS", beep_command

    return "none", None


def backend_frequency_error(name: str, freq: int) -> str | None:
    """Usage-error text when backend ``name`` cannot produce ``freq`` Hz, else ``None``.

    Only :data:`WINSOUND_BACKEND` has a range the script must enforce up front
    (``winsound.Beep`` refuses anything outside 37..32767 Hz). The ``beep``
    command validates its own arguments and the ``"none"`` backend only prints.
    """
    if name == WINSOUND_BACKEND and not WINSOUND_MIN_FREQ <= freq <= WINSOUND_MAX_FREQ:
        return (f"--freq must be in {WINSOUND_MIN_FREQ}..{WINSOUND_MAX_FREQ} Hz "
                f"for {WINSOUND_BACKEND}, got {freq}")
    return None


def play(seq: Sequence[tuple[bool, int]], freq: int, beep: BeepFn) -> None:
    """Key ``seq`` through ``beep`` for marks and ``time.sleep`` for gaps."""
    for on, ms in seq:
        if on:
            beep(freq, ms)
        else:
            time.sleep(ms / 1000.0)


# -- command line -------------------------------------------------------------


def _frequency(text: str) -> int:
    """argparse type for ``--freq``: a positive number of Hz, rounded to whole Hz.

    Accepts ``2491``, ``2491.0`` and ``2490.97`` alike (all become 2491) so a
    value copied from the decoder's auto-detect readout works unchanged; every
    backend takes an integer frequency.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"frequency must be a number, got {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"frequency must be positive, got {text!r}")
    freq = int(round(value))
    if freq < 1:
        raise argparse.ArgumentTypeError(f"frequency must round to at least 1 Hz, got {text!r}")
    return freq


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` defaults to ``sys.argv[1:]``).

    ``args.freq`` is always an ``int``. Backend-specific frequency limits are
    not applied here; see :func:`backend_frequency_error`.
    """
    parser = argparse.ArgumentParser(
        description="Send text as Morse code through this machine's PC speaker.",
        epilog="Timing: dit 1200/wpm ms, dah 3 dits, letter gap 3 dits, word gap 7 dits; "
               "--farnsworth stretches the letter and word gaps.",
    )
    parser.add_argument("text", nargs="+", help="text to send (words joined with spaces)")
    parser.add_argument("--wpm", type=float, default=DEFAULT_WPM,
                        help=f"speed in words per minute (default {DEFAULT_WPM:g})")
    parser.add_argument("--farnsworth", type=float, default=None, metavar="WPM",
                        help="overall speed for Farnsworth spacing: characters at --wpm, longer gaps "
                             "between them (ARRL rule); no effect when not below --wpm")
    parser.add_argument("--freq", type=_frequency, default=DEFAULT_FREQ, metavar="HZ",
                        help=f"tone frequency in Hz, rounded to whole Hz (default {DEFAULT_FREQ})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print one (on, ms) per line instead of beeping")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="send the message N times, a word gap apart (default 1)")
    args = parser.parse_args(argv)
    if args.wpm <= 0:
        parser.error("--wpm must be positive")
    if args.farnsworth is not None and args.farnsworth <= 0:
        parser.error("--farnsworth must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    text = " ".join(args.text)

    skipped = unknown_characters(text)
    if skipped:
        print(f"note: no Morse code for {skipped!r}; skipped", file=sys.stderr)

    seq = repeat_timing(build_timing(text, args.wpm, args.farnsworth), args.repeat, args.wpm)
    if not seq:
        print("nothing to send: text contains no encodable characters", file=sys.stderr)
        return 1

    if args.dry_run:
        print(format_timing(seq))
        return 0

    name, beep = select_backend()
    summary = (f"{text!r} at {args.wpm:g} WPM (dit {dit_ms(args.wpm):.0f} ms), "
               f"{args.freq} Hz, x{args.repeat}, {total_ms(seq) / 1000:.1f} s")
    if beep is None:
        print(f"no tone backend on this platform ({sys.platform}); install 'beep' or "
              "run on Windows. Timing that would have been sent for "
              f"{summary}:", file=sys.stderr)
        print(format_timing(seq))
        return 0

    range_error = backend_frequency_error(name, args.freq)
    if range_error:
        print(f"error: {range_error}", file=sys.stderr)
        return 2

    print(f"sending {summary} via {name}", file=sys.stderr)
    try:
        play(seq, args.freq, beep)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        # winsound.Beep: RuntimeError when there is no speaker / RPC failure,
        # ValueError when it rejects the frequency or duration.
        print(f"beep failed: {exc}", file=sys.stderr)
        return 1
    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
