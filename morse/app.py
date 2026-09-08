"""Command-line entry point for the Morse code audio decoder.

::

    python -m morse.app --list-devices
    python -m morse.app --wav PATH [--freq HZ] [--wpm N] [--verbose]     # offline: print decoded text
    python -m morse.app --no-ui [--device SPEC] [--freq HZ] [--wpm N]     # live, text to stdout
    python -m morse.app [--device SPEC] [--freq HZ] [--wpm N]             # live with the Qt UI

``--freq`` defaults to 2491 Hz, the measured PC-beeper tone.  ``--verbose``
prints every finalised run as ``ON 110 ms`` / ``off 50 ms`` before the text.
Live modes read the microphone through :class:`morse.audio_input.AudioInput`
and poll it every 20 ms; the UI mode imports ``morse.ui`` lazily and exits
with a clear message pointing at ``--no-ui`` when it is missing.

Exit codes: 0 on success; 2 when the WAV named by ``--wav`` cannot be opened
(missing file, a directory, no permission: any ``OSError``), reported as one
line on stderr and never as a traceback; 1 for every other error.

Everything written to stdout or stderr goes through :func:`emit`, which
encodes with ``errors="replace"`` for the stream's own encoding: device names
on the target laptop contain a registered-trademark sign, and a redirected or
legacy console stream (cp1252, cp437, ascii) must not abort ``--list-devices``
with a ``UnicodeEncodeError``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Sequence
from typing import TextIO

from morse.pipeline import Pipeline, decode_wav

__all__ = ["build_parser", "emit", "main", "parse_device"]

DEFAULT_FREQ_HZ = 2491.0
"""Default tone frequency: the BIOS-style beep measured in docs/PLAN.md section 8."""

POLL_INTERVAL_S = 0.02
"""How often the live text mode drains the audio queue."""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WAV_UNREADABLE = 2
"""Exit code when the ``--wav`` file cannot be opened (any ``OSError``)."""

_UI_HINT = "run with --no-ui for text output, or --wav FILE to decode a recording"


# ------------------------------------------------------------------- output


def emit(text: str = "", *, end: str = "\n", file: TextIO | None = None, flush: bool = False) -> None:
    """Write ``text + end`` to ``file`` (default ``sys.stdout``) without raising for encoding.

    The text is first encoded for the stream's own encoding with
    ``errors="replace"`` and decoded back, so a character the console cannot
    show becomes ``?`` instead of a ``UnicodeEncodeError``.  Works for any
    text stream, including wrappers that have no ``reconfigure`` method.
    """
    stream = file if file is not None else sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    payload = text + end
    try:
        safe = payload.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except LookupError:  # unknown codec name on an exotic stream: let the stream cope
        safe = payload
    stream.write(safe)
    if flush:
        stream.flush()


def _error(message: str) -> None:
    """One line on stderr, prefixed with ``error:``."""
    emit(f"error: {' '.join(str(message).split())}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------- parsing


def parse_device(spec: str | None) -> int | str | None:
    """Turn the ``--device`` argument into what ``AudioInput`` expects.

    A string of digits (optionally negative) becomes a device index, anything
    else is a case-insensitive name fragment, and ``None`` stays ``None``
    (the preferred raw endpoint, else the system default input).
    """
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _positive(kind: str):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{kind} must be a number, got {text!r}") from exc
        if not value > 0:
            raise argparse.ArgumentTypeError(f"{kind} must be positive, got {text!r}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests and the UI can reuse it)."""
    parser = argparse.ArgumentParser(
        prog="python -m morse.app",
        description="Decode a PC beeper's Morse code from the microphone or from a WAV file.",
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="list input-capable audio devices and exit")
    parser.add_argument("--wav", metavar="PATH",
                        help="decode this WAV file offline instead of listening")
    parser.add_argument("--no-ui", action="store_true",
                        help="live decoding with text on stdout, no Qt window")
    parser.add_argument("--device", metavar="SPEC", default=None,
                        help="input device index or name fragment (default: the raw "
                             "'Microphone Array 1' WDM-KS endpoint if present, else the system default)")
    parser.add_argument("--freq", metavar="HZ", type=_positive("frequency"), default=DEFAULT_FREQ_HZ,
                        help=f"tone frequency in Hz (default {DEFAULT_FREQ_HZ:g})")
    parser.add_argument("--wpm", metavar="N", type=_positive("wpm"), default=None,
                        help="keying speed in words per minute (default: adaptive)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print every finalised run as 'ON 110 ms' / 'off 50 ms'")
    return parser


# ----------------------------------------------------------------- commands


def _cmd_list_devices() -> int:
    from morse.audio_input import list_devices, resolve_device

    try:
        devices = list_devices()
    except Exception as exc:  # sounddevice / PortAudio failures
        _error(f"cannot query audio devices: {exc}")
        return EXIT_ERROR
    if not devices:
        emit("no input-capable audio devices found")
        return EXIT_OK
    try:
        default_index = resolve_device(None)
    except Exception:
        default_index = -1
    emit(f"{'':1} {'idx':>3}  {'ch':>2}  {'rate':>6}  {'host API':<16} name")
    for dev in devices:
        mark = "*" if dev.index == default_index else " "
        emit(f"{mark} {dev.index:>3}  {dev.max_input_channels:>2}  {dev.default_samplerate:>6.0f}  "
             f"{dev.hostapi:<16} {dev.name}")
    if default_index >= 0:
        emit("* = device used when --device is not given")
    return EXIT_OK


def _cmd_wav(args: argparse.Namespace) -> int:
    try:
        text, runs = decode_wav(args.wav, args.freq, wpm=args.wpm)
    except OSError as exc:  # missing file, a directory, permission denied, ...
        reason = exc.strerror or exc.__class__.__name__
        _error(f"cannot open WAV file {args.wav}: {reason}")
        return EXIT_WAV_UNREADABLE
    except ValueError as exc:  # not a WAV, unsupported sample type, bad --freq for this fs
        _error(str(exc))
        return EXIT_ERROR
    if args.verbose:
        for run in runs:
            emit(str(run))
    emit(text.strip())
    return EXIT_OK


def _cmd_live(args: argparse.Namespace) -> int:
    from morse.audio_input import AudioInput

    try:
        pipeline = Pipeline(f0=args.freq, wpm=args.wpm)
    except ValueError as exc:
        _error(str(exc))
        return EXIT_ERROR
    try:
        audio = AudioInput(device=args.device, fs=pipeline.fs, block_size=pipeline.block_size)
    except (ValueError, RuntimeError, TypeError) as exc:
        _error(str(exc))
        return EXIT_ERROR
    except Exception as exc:  # PortAudio errors have no useful common base class
        _error(f"cannot open audio input: {exc}")
        return EXIT_ERROR

    emit(f"Listening on [{audio.device_index}] {audio.device_name} for {args.freq:g} Hz; "
         f"Ctrl-C to stop.", file=sys.stderr, flush=True)
    try:
        with audio:
            while True:
                for block in audio.read_blocks():
                    result = pipeline.process_block(block)
                    if args.verbose:
                        for run in result.runs:
                            emit(str(run))
                        if result.new_text:
                            emit(f"text: {pipeline.text}")
                    elif result.new_text:
                        emit(result.new_text, end="", flush=True)
                time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        emit("", file=sys.stderr)
        _error(f"audio input failed: {exc}")
        return EXIT_ERROR

    rest = pipeline.flush()
    if args.verbose:
        for run in pipeline.last_flush_runs:
            emit(str(run))
        emit(f"text: {pipeline.text}")
    else:
        emit(rest, flush=True)  # ends the line of streamed text
    if audio.dropped:
        emit(f"dropped {audio.dropped} blocks (queue full)", file=sys.stderr)
    return EXIT_OK


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        ui = importlib.import_module("morse.ui")
    except ModuleNotFoundError as exc:
        if exc.name == "morse.ui":
            reason = "the Qt UI (morse/ui.py) is not available in this checkout"
        else:
            reason = f"the Qt UI needs the {exc.name!r} package, which is not installed"
        _error(f"{reason}; {_UI_HINT}.")
        return EXIT_ERROR
    except ImportError as exc:
        _error(f"the Qt UI failed to import ({exc}); {_UI_HINT}.")
        return EXIT_ERROR
    return int(ui.run_ui(args) or 0)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``) and run the selected mode."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.device = parse_device(args.device)
    if args.list_devices:
        return _cmd_list_devices()
    if args.wav:
        return _cmd_wav(args)
    if args.no_ui:
        return _cmd_live(args)
    return _cmd_ui(args)


if __name__ == "__main__":
    sys.exit(main())
