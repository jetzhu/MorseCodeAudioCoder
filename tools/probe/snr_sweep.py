"""Decode-success sweep over signal-to-noise ratio, keying speed and room tail.

Runs the full Python pipeline on synthetic keyed tones and prints, for each
speed and reverb tail, how many of three noise seeds decode the message
exactly at each SNR.  Use it after any change to the detector or decoder:

    .venv\\Scripts\\python.exe tools\\probe\\snr_sweep.py [--text "SOS HELLO"] [--f0 2491]

SNR here is the tone's Goertzel power over the single-bin Goertzel noise
mean for 480-sample blocks (white noise bin power = 2/N * sigma^2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from morse.dsp import synth_keyed_tone  # noqa: E402
from morse.pipeline import Pipeline  # noqa: E402
from morse.table import encode  # noqa: E402

FS = 48000
BLOCK = 480


def pattern(text: str, wpm: float) -> list[tuple[bool, float]]:
    t = 1200.0 / wpm
    pat: list[tuple[bool, float]] = [(False, 800.0)]
    words = text.split(" ")
    for wi, word in enumerate(words):
        for ci, ch in enumerate(word):
            code = encode(ch)
            for si, sym in enumerate(code):
                pat.append((True, t if sym == "." else 3 * t))
                if si < len(code) - 1:
                    pat.append((False, t))
            if ci < len(word) - 1:
                pat.append((False, 3 * t))
        if wi < len(words) - 1:
            pat.append((False, 7 * t))
    pat.append((False, 1500.0))
    return pat


def decode(text: str, wpm: float, snr_db: float, reverb_ms: float, f0: float,
           amp: float = 0.1, seed: int = 0) -> tuple[str, float, float]:
    sigma = amp * np.sqrt(BLOCK / 2) * 10 ** (-snr_db / 20)
    x = synth_keyed_tone(pattern(text, wpm), f0, FS, amplitude=amp, noise_rms=sigma,
                         reverb_ms=reverb_ms, seed=seed)
    pipe = Pipeline(fs=FS, block_size=BLOCK, f0=f0)
    for i in range(len(x) // BLOCK):
        pipe.process_block(x[i * BLOCK:(i + 1) * BLOCK])
    pipe.flush()
    return pipe.text.strip(), pipe.decoder.dit_ms, pipe.decoder.offset_ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--text", default="SOS HELLO")
    ap.add_argument("--f0", type=float, default=2491.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    snrs = (12, 15, 20, 30, 46, 60)
    for reverb in (10.0, 30.0, 60.0):
        print(f"\nreverb tail {reverb:.0f} ms (1/e), text '{args.text}':")
        for wpm in (8, 12, 15, 20, 25):
            cells = []
            for snr in snrs:
                ok = sum(decode(args.text, wpm, snr, reverb, args.f0, seed=s)[0] == args.text
                         for s in range(args.seeds))
                cells.append(f"{snr:>2}dB:{ok}/{args.seeds}")
            text, dit, off = decode(args.text, wpm, 46, reverb, args.f0)
            print(f"  {wpm:>2} WPM  " + "  ".join(cells)
                  + f"   at 46 dB: dit {dit:.0f} ms, offset {off:.0f} ms, '{text}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
