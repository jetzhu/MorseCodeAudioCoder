"""Export parity vectors for the JavaScript port to ``web/test/vectors.json``.

Usage (from any directory)::

    .venv/Scripts/python.exe tools/export_vectors.py            # (re)write the file
    .venv/Scripts/python.exe tools/export_vectors.py --check    # exit 1 if the file is stale
    .venv/Scripts/python.exe tools/export_vectors.py --out PATH

The JSON is generated from the Python reference implementation; do not edit
it by hand.  It carries three things (docs/INTERFACES.md, "Parity vectors"):

``cases``
    Decoder vectors.  Each case is a fixed list of final runs
    ``[[on, blocks], ...]`` in 10 ms blocks, the text
    :class:`morse.decoder.MorseDecoder` has emitted after feeding every run
    and calling ``idle(inf)``, and the final ``dit_ms`` / ``offset_ms``.  The
    run lists are built here from text keyed at a WPM with standard timing,
    optionally jittered (``numpy.random.default_rng(seed)``) and optionally
    shifted by a reverb offset ``d`` (marks measure ``+d``, gaps ``-d``).
    The ``idle-flush`` case also records what ``idle(off_ms)`` returned for a
    few finite ``off_ms`` values before the final flush.

``fixtures``
    For each WAV under ``tests/fixtures``: the :class:`morse.dsp.Goertzel` dB
    series per 480-sample block at the fixture's tone frequency, rounded to
    0.1 dB; the runs a default :class:`morse.tone_detector.ToneDetector`
    produces from the unrounded series (``update`` per block, then ``flush``);
    and the decoder text, cross-checked against :func:`morse.pipeline.decode_samples`.

``params``
    The detector and decoder defaults, read from the constructors, and the
    constants every port must share.

The output is deterministic: running the script twice writes identical bytes.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morse import decoder as decoder_module  # noqa: E402
from morse import tone_detector as detector_module  # noqa: E402
from morse.decoder import DEFAULT_DIT_MS, MorseDecoder  # noqa: E402
from morse.dsp import Goertzel  # noqa: E402
from morse.pipeline import decode_samples, load_wav  # noqa: E402
from morse.runs import Run  # noqa: E402
from morse.table import encode  # noqa: E402
from morse.tone_detector import ToneDetector, is_tonal  # noqa: E402

__all__ = [
    "build_cases",
    "build_fixture",
    "build_params",
    "build_vectors",
    "code_to_pattern",
    "main",
    "make_runs",
    "pattern_to_runs",
    "render_json",
    "speed_change_runs",
]

DEFAULT_OUTPUT = ROOT / "web" / "test" / "vectors.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures"

FS = 48000
BLOCK_SIZE = 480
BLOCK_MS = 1000.0 * BLOCK_SIZE / FS  # 10.0
SERIES_ROUNDING_DB = 0.1
MAX_BYTES = 2 * 1024 * 1024
LINE_WIDTH = 100

FIXTURES: list[tuple[str, float]] = [
    ("loopback_sos_1khz_15wpm", 1000.0),
    ("beeper_long_2491hz_1m", 2491.0),
]
"""(file stem, tone frequency) of the recorded fixtures (docs/INTERFACES.md, Conventions)."""

Pattern = list[tuple[bool, float]]


# ------------------------------------------------------------- run builders


def code_to_pattern(code: str, dit_ms: float) -> Pattern:
    """Keyed ``(on, ms)`` pairs for a dot/dash string such as ``'... --- / -..-'``.

    Standard timing in dits: dit 1, dah 3, intra-letter gap 1, letter gap 3,
    word gap 7.  Letters are separated by one space and words by ``' / '``
    (the output format of :func:`morse.table.encode`).  Leading and trailing
    gaps are not included.
    """
    pattern: Pattern = []
    for wi, word in enumerate(code.split(" / ")):
        if wi:
            pattern.append((False, 7 * dit_ms))
        for li, letter in enumerate(word.split(" ")):
            if li:
                pattern.append((False, 3 * dit_ms))
            for si, symbol in enumerate(letter):
                if si:
                    pattern.append((False, dit_ms))
                pattern.append((True, 3 * dit_ms if symbol == "-" else dit_ms))
    return pattern


def pattern_to_runs(
    pattern: Pattern,
    jitter: float = 0.0,
    offset_ms: float = 0.0,
    seed: int = 0,
    block_ms: float = BLOCK_MS,
) -> list[Run]:
    """Distort a keyed pattern and quantise it to detector blocks.

    ``jitter`` is a fraction: each duration is scaled by ``1 + u`` with ``u``
    drawn uniformly from ``[-jitter, jitter)`` by
    ``numpy.random.default_rng(seed)``, one draw per run in pattern order.
    ``offset_ms`` mimics room reverb and the detector's release: marks are
    lengthened by it and gaps shortened by it.  Each run is then
    ``max(1, round(ms / block_ms))`` blocks long.
    """
    rng = np.random.default_rng(seed) if jitter else None
    runs: list[Run] = []
    for on, ms in pattern:
        if rng is not None:
            ms *= 1.0 + float(rng.uniform(-jitter, jitter))
        ms += offset_ms if on else -offset_ms
        runs.append(Run(on, max(1, round(ms / block_ms)), block_ms))
    return runs


def make_runs(text: str, wpm: float, **kwargs: Any) -> list[Run]:
    """Runs for ``text`` keyed at ``wpm``; keyword arguments go to :func:`pattern_to_runs`."""
    return pattern_to_runs(code_to_pattern(encode(text), 1200.0 / wpm), **kwargs)


def speed_change_runs(first_text: str, first_wpm: float, second_text: str, second_wpm: float) -> list[Run]:
    """``first_text`` at ``first_wpm``, a word gap at the slower speed, ``second_text`` at ``second_wpm``."""
    slow_dit = 1200.0 / min(first_wpm, second_wpm)
    gap = Run(False, round(7 * slow_dit / BLOCK_MS), BLOCK_MS)
    return make_runs(first_text, first_wpm) + [gap] + make_runs(second_text, second_wpm)


# ------------------------------------------------------------- decoder cases


def _text_parameters(text: str, wpm: float, jitter: float = 0.0, seed: int | None = None,
                     offset_ms: float = 0.0) -> dict[str, Any]:
    params: dict[str, Any] = {"text": text, "wpm": wpm, "jitter": jitter, "offset_ms": offset_ms}
    if jitter:
        params["seed"] = seed
        params["rng"] = ("numpy.random.default_rng(seed): each run's ms is scaled by "
                         "1 + uniform(-jitter, jitter), one draw per run in order")
    return params


def _decode_case(
    name: str,
    description: str,
    runs: Sequence[Run],
    parameters: dict[str, Any],
    idle_probe_ms: Iterable[float] = (),
    **decoder_kwargs: Any,
) -> dict[str, Any]:
    """Feed ``runs`` to a fresh decoder, probe ``idle`` if asked, flush with ``idle(inf)``."""
    dec = MorseDecoder(**decoder_kwargs)
    text_before_idle = "".join(dec.feed(run) for run in runs)
    probes = [[off_ms, dec.idle(float(off_ms))] for off_ms in idle_probe_ms]
    idle_text = dec.idle(math.inf)
    assert dec.text == text_before_idle + "".join(p[1] for p in probes) + idle_text
    decoder_args = {**_constructor_defaults(MorseDecoder), **decoder_kwargs}
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "decoder": {key: decoder_args[key] for key in ("wpm", "adaptive", "window")},
        "runs": [[run.on, run.blocks] for run in runs],
        "text_before_idle": text_before_idle,
        "idle_probes": probes,
        "idle_text": idle_text,
        "text": dec.text,
        "dit_ms": dec.dit_ms,
        "offset_ms": dec.offset_ms,
        "letter_count": dec.letter_count,
        "unknown_count": dec.unknown_count,
    }


def build_cases() -> list[dict[str, Any]]:
    """The fixed case list from docs/INTERFACES.md, "Parity vectors"."""
    dit_15wpm = 1200.0 / 15
    idle_probes = (560, 570, 600, 5000)
    return [
        _decode_case(
            "sos-15wpm-clean",
            "SOS keyed at 15 WPM with exact standard timing (80 ms dit, 240 ms dah).",
            make_runs("SOS", 15),
            _text_parameters("SOS", 15),
        ),
        _decode_case(
            "hello-world-8wpm",
            "HELLO WORLD at 8 WPM (150 ms dit): letter gaps and one word gap.",
            make_runs("HELLO WORLD", 8),
            _text_parameters("HELLO WORLD", 8),
        ),
        _decode_case(
            "jitter-20pct-12wpm-seed1",
            "HELLO WORLD at 12 WPM (100 ms dit), every run scaled by a uniform factor in "
            "1 +- 0.2 drawn from numpy.random.default_rng(1).",
            make_runs("HELLO WORLD", 12, jitter=0.2, seed=1),
            _text_parameters("HELLO WORLD", 12, jitter=0.2, seed=1),
        ),
        _decode_case(
            "reverb-30ms-15wpm",
            "SOS at 15 WPM with a 30 ms reverb offset: marks measure 110/270 ms, gaps 50/210 ms "
            "(docs/PLAN.md section 4); the decoder must recover d = 30 ms.",
            make_runs("SOS", 15, offset_ms=30.0),
            _text_parameters("SOS", 15, offset_ms=30.0),
        ),
        _decode_case(
            "speed-change-12-to-6wpm",
            "PARIS PARIS at 12 WPM, a 7-dit word gap at the slower speed, then PARIS PARIS PARIS "
            "PARIS at 6 WPM. While the 100 ms dits drain out of the 30-run window the 200 ms dits "
            "sit exactly on the 2T boundary, so the transitional text depends on the nearest-rank "
            "percentile.",
            speed_change_runs("PARIS PARIS", 12, "PARIS PARIS PARIS PARIS", 6),
            {
                "segments": [
                    {"text": "PARIS PARIS", "wpm": 12},
                    {"gap_blocks": round(7 * (1200.0 / 6) / BLOCK_MS), "note": "7 dits at 6 WPM"},
                    {"text": "PARIS PARIS PARIS PARIS", "wpm": 6},
                ],
            },
        ),
        _decode_case(
            "oso-15wpm-dah-leading",
            "OSO at 15 WPM: the message opens with a dah-only letter. The first runs are held "
            "back until both mark classes have been seen, then replayed; the old seed-based rule "
            "read this as SSO.",
            make_runs("OSO", 15),
            _text_parameters("OSO", 15),
        ),
        _decode_case(
            "sos-hello-3wpm",
            "SOS HELLO at 3 WPM (400 ms dit): slow keying, held back until the first dah of O.",
            make_runs("SOS HELLO", 3),
            _text_parameters("SOS HELLO", 3),
        ),
        _decode_case(
            "hello-world-2wpm",
            "HELLO WORLD at 2 WPM (600 ms dit), the slowest supported speed.",
            make_runs("HELLO WORLD", 2),
            _text_parameters("HELLO WORLD", 2),
        ),
        _decode_case(
            "lone-o-idle-flush",
            "O alone at 15 WPM, then silence: three equal marks 3x the intra gaps. Three marks "
            "make the estimate trusted and the dah rule gives T = 80 ms, so idle() flushes once "
            "off_ms > 7T = 560 (560 is not > 560, 570 is).",
            make_runs("O", 15),
            {**_text_parameters("O", 15), "idle_probe_ms": [560, 570, 600, 5000]},
            idle_probe_ms=(560, 570, 600, 5000),
        ),
        _decode_case(
            "glitches-after-hello-8wpm",
            "HELLO WORLD at 8 WPM followed by keyboard clicks: 20 to 50 ms marks separated by "
            "2 to 3 s of silence. Once the estimate is trusted, marks shorter than 0.4 T are "
            "ignored and leave the timing untouched.",
            make_runs("HELLO WORLD", 8) + [
                Run(False, 200), Run(True, 3), Run(False, 200), Run(True, 4),
                Run(False, 300), Run(True, 2), Run(False, 200), Run(True, 5), Run(False, 200),
            ],
            {**_text_parameters("HELLO WORLD", 8), "glitch_blocks": [3, 4, 2, 5]},
        ),
        _decode_case(
            "slowdown-20-to-4wpm",
            "PARIS at 20 WPM, a word gap at the slow speed, then HELLO WORLD at 4 WPM. Three marks of "
            "at least 1.5 T with no dit between them mean the estimate is stale-low: the window is "
            "rebuilt from those runs and the letter in progress re-read, so only the first slow "
            "letter is damaged.",
            make_runs("PARIS", 20) + [Run(False, round(7 * (1200.0 / 4) / BLOCK_MS))] + make_runs("HELLO WORLD", 4),
            {"segments": [{"text": "PARIS", "wpm": 20}, {"gap_blocks": round(7 * (1200.0 / 4) / BLOCK_MS)},
                          {"text": "HELLO WORLD", "wpm": 4}]},
        ),
        _decode_case(
            "unknown-symbol",
            "Six dits in one letter (not in the table), a word gap, then S: emits '?' once and "
            "counts it in unknown_count.",
            pattern_to_runs(code_to_pattern("...... / ...", dit_15wpm)),
            {"code": "...... / ...", "wpm": 15, "jitter": 0.0, "offset_ms": 0.0},
        ),
        _decode_case(
            "idle-flush",
            "SOS at 15 WPM with no gap after the last dit, so the last S is pending when the "
            "runs end. idle(off_ms) flushes it only once off_ms + offset_ms > 7 * dit_ms "
            "(560 is not > 560, 570 is) and later calls emit nothing.",
            make_runs("SOS", 15),
            {**_text_parameters("SOS", 15), "idle_probe_ms": list(idle_probes)},
            idle_probe_ms=idle_probes,
        ),
    ]


# ---------------------------------------------------------------- fixtures


def _full_blocks(x: np.ndarray, block_size: int) -> np.ndarray:
    """Cut ``x`` into ``(n, block_size)``, zero-padding a short tail like ``decode_samples``."""
    remainder = x.size % block_size
    if remainder:
        x = np.concatenate([x, np.zeros(block_size - remainder, dtype=x.dtype)])
    return x.reshape(-1, block_size)


def _detector_runs(
    series: Iterable[float],
    decoder: MorseDecoder | None = None,
    tonal: Sequence[bool] | None = None,
) -> list[Run]:
    """Default detector over ``series`` then ``flush``; optionally decode like ``Pipeline`` does.

    ``tonal`` gives the per-block flag ``Pipeline.process_block`` derives from
    the block level (:func:`morse.tone_detector.is_tonal`); without it every
    block counts as tonal.
    """
    det = ToneDetector()
    runs: list[Run] = []
    for i, power_db in enumerate(series):
        new = det.update(power_db, tonal=True if tonal is None else bool(tonal[i]))
        runs.extend(new)
        if decoder is not None:
            for run in new:
                decoder.feed(run)
            current = det.current_run
            if not current.on:
                decoder.idle(current.ms)
    tail = det.flush()
    runs.extend(tail)
    if decoder is not None:
        for run in tail:
            decoder.feed(run)
        decoder.idle(math.inf)
    return runs


def build_fixture(stem: str, f0: float) -> dict[str, Any]:
    """Goertzel series, default-detector runs and decoder text for one WAV fixture."""
    path = FIXTURE_DIR / f"{stem}.wav"
    fs, x = load_wav(path)
    if fs != FS:
        raise SystemExit(f"{path.name}: expected {FS} Hz, got {fs} Hz")
    blocks = _full_blocks(x, BLOCK_SIZE)
    goertzel = Goertzel(f0, fs, BLOCK_SIZE)
    series = [goertzel.power_db(block) for block in blocks]
    levels = [float(10.0 * np.log10(float(np.mean(np.square(block, dtype=np.float64))) + 1e-12))
              for block in blocks]
    tonal = [is_tonal(p, lvl) for p, lvl in zip(series, levels)]

    decoder = MorseDecoder()
    runs = _detector_runs(series, decoder, tonal)

    # The offline pipeline must agree, or the vectors would not describe decode_wav.
    ref_text, ref_runs = decode_samples(x, fs, f0, block_size=BLOCK_SIZE)
    if (ref_text, ref_runs) != (decoder.text, runs):
        raise SystemExit(f"{path.name}: direct detector/decoder pass disagrees with decode_samples")

    rounded = [round(value, 1) for value in series]
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "f0": f0,
        "fs": fs,
        "block_size": BLOCK_SIZE,
        "block_ms": BLOCK_MS,
        "samples": int(x.size),
        "blocks": len(series),
        "power_db": rounded,
        "level_dbfs": [round(value, 1) for value in levels],
        "tonal": [int(flag) for flag in tonal],
        "runs": [[run.on, run.blocks] for run in runs],
        "runs_from_rounded_power_db_identical": _detector_runs(rounded, tonal=tonal) == runs,
        "text": decoder.text,
    }


# ------------------------------------------------------------------ params


def _constructor_defaults(cls: type) -> dict[str, Any]:
    """Keyword defaults of ``cls.__init__`` in declaration order."""
    return {
        name: parameter.default
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def build_params() -> dict[str, Any]:
    """Detector and decoder defaults plus the constants the ports must share."""
    return {
        "fs": FS,
        "block_size": BLOCK_SIZE,
        "block_ms": BLOCK_MS,
        "run_format": "[on, blocks]; a run lasts blocks * block_ms milliseconds",
        "run_generation": "pattern ms -> blocks = max(1, round(ms / block_ms)); marks +offset_ms, gaps -offset_ms",
        "goertzel": {
            "power": "4 * |X(f0)|^2 / N^2 over the single bin at f0; a full-scale sine at f0 gives 1.0",
            "power_db": "10 * log10(power + 1e-12)",
            "series_rounding_db": SERIES_ROUNDING_DB,
        },
        "detector": _constructor_defaults(ToneDetector),
        "detector_constants": {
            "db_floor": detector_module._DB_FLOOR,
            "warmup_alpha": detector_module._WARMUP_ALPHA,
        },
        "decoder": _constructor_defaults(MorseDecoder),
        "decoder_constants": {
            "default_dit_ms": DEFAULT_DIT_MS,
            "percentile": decoder_module._PERCENTILE,
            "percentile_method": "nearest rank: sort ascending, take the element at 1-based rank ceil(q / 100 * n)",
            "gap_cap_factor": decoder_module._GAP_CAP_FACTOR,
            "dit_dah_split": decoder_module._DIT_DAH_SPLIT,
            "letter_gap": decoder_module._LETTER_GAP,
            "word_gap": decoder_module._WORD_GAP,
            "idle_flush": decoder_module._IDLE_FLUSH,
        },
    }


def build_vectors() -> dict[str, Any]:
    """The whole document."""
    return {
        "generated_by": "tools/export_vectors.py",
        "note": "Generated from the Python reference implementation; do not edit by hand. "
                "Every case feeds its runs to a fresh MorseDecoder, applies the idle probes in "
                "order, then calls idle(+Infinity); 'text' is decoder.text afterwards.",
        "schema_version": 1,
        "params": build_params(),
        "cases": build_cases(),
        "fixtures": {stem: build_fixture(stem, f0) for stem, f0 in FIXTURES},
    }


# -------------------------------------------------------------- rendering


def _scalar(value: Any) -> str:
    return json.dumps(value, allow_nan=False)


def _is_flat(items: list[Any]) -> bool:
    """True for a list of scalars or of short scalar lists (runs, idle probes)."""
    for item in items:
        if isinstance(item, dict):
            return False
        if isinstance(item, list):
            if len(item) > 3 or any(isinstance(x, (dict, list)) for x in item):
                return False
    return True


def _wrap(tokens: list[str], indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current}, {token}"
        if current and len(inner) + len(candidate) > LINE_WIDTH:
            lines.append(current)
            current = token
        else:
            current = candidate
    lines.append(current)
    if len(lines) == 1 and len(pad) + len(current) + 2 <= LINE_WIDTH:
        return f"[{current}]"
    return "[\n" + ",\n".join(inner + line for line in lines) + f"\n{pad}]"


def _render(value: Any, indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{_scalar(str(key))}: {_render(item, indent + 2)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if _is_flat(value):
            tokens = [
                "[" + ", ".join(_scalar(x) for x in item) + "]" if isinstance(item, list) else _scalar(item)
                for item in value
            ]
            return _wrap(tokens, indent)
        items = [f"{inner}{_render(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return _scalar(value)


def render_json(data: dict[str, Any]) -> str:
    """Pretty JSON with numeric arrays and run lists packed onto wrapped lines."""
    text = _render(data, 0) + "\n"
    if json.loads(text) != data:
        raise RuntimeError("rendered JSON does not round-trip")
    return text


# ------------------------------------------------------------------- main


def _summary(data: dict[str, Any], out: Path, size: int) -> str:
    shown = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
    lines = [f"{shown}: {size} bytes"]
    for case in data["cases"]:
        lines.append(f"  case {case['name']:<26} text={case['text']!r:<64} "
                     f"dit_ms={case['dit_ms']:g} offset_ms={case['offset_ms']:g}")
    for stem, fixture in data["fixtures"].items():
        on_ms = [int(blocks * fixture["block_ms"]) for on, blocks in fixture["runs"] if on]
        lines.append(f"  fixture {stem}: {fixture['blocks']} blocks, {len(fixture['runs'])} runs, "
                     f"ON ms {on_ms}, text={fixture['text']!r}, "
                     f"rounded-series runs identical={fixture['runs_from_rounded_power_db_identical']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the vectors; write them (default) or check the file on disk (``--check``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output path (default {DEFAULT_OUTPUT})")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if the file on disk differs from what would be written")
    args = parser.parse_args(argv)

    data = build_vectors()
    text = render_json(data)
    size = len(text.encode("utf-8"))
    if size > MAX_BYTES:
        raise SystemExit(f"vectors.json would be {size} bytes, above the {MAX_BYTES} byte limit")

    out: Path = args.out
    existing = out.read_text(encoding="utf-8") if out.is_file() else None
    if args.check:
        if existing == text:
            print(f"{out}: up to date")
            return 0
        print(f"{out}: {'missing' if existing is None else 'stale'}; run tools/export_vectors.py")
        return 1
    if existing == text:
        print(f"{out}: unchanged")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{out}: written")
    print(_summary(data, out, size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
