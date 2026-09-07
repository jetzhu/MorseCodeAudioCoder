# Module interfaces

Contract for the `morse` package. Every module is written against this file
so that modules built independently fit together. Read `docs/PLAN.md` first
for the reasoning; measured constants come from section 8 of that plan.

Conventions

- Python 3.13, type hints everywhere, `from __future__ import annotations`.
- Audio is `numpy.ndarray` of `float32` in the range −1..1, mono.
- Sample rate `fs = 48000`, block size `480` samples = 10 ms, unless told otherwise.
- Power values are in dB. Goertzel power is normalised so a full-scale sine at
  `f0` gives 0 dB regardless of block length. Silence gives roughly −90 dB or less.
- No module below `morse/ui.py` imports Qt, sounddevice, or anything that needs
  hardware. `morse/audio_input.py` is the only module that imports sounddevice.
- Tests run with `.venv\Scripts\python.exe -m pytest -q` from the project root.
- Fixtures: `tests/fixtures/loopback_sos_1khz_15wpm.wav` decodes to `SOS`
  (1000 Hz tone, 80 ms dit, room reverb). `tests/fixtures/beeper_long_2491hz_1m.wav`
  is the real beeper at 2491 Hz: four ON runs of about 3700, 490, 570 and
  2020 ms separated by gaps of about 370, 250 and 300 ms, starting near 0.69 s.

## morse/runs.py (shared, already written)

```python
@dataclass(frozen=True)
class Run:
    on: bool        # True = tone present
    blocks: int     # length in blocks
    block_ms: float = 10.0
    @property
    def ms(self) -> float: ...
```

## morse/table.py

```python
MORSE_TABLE: dict[str, str]          # 'A' -> '.-', digits, . , ? / = ' ( ) : ; + - _ " @ !
INVERSE: dict[str, str]              # '.-' -> 'A'
def lookup(symbols: str) -> str | None       # None when unknown
def encode(text: str, unknown: str = "") -> str
    # 'SOS X' -> '... --- ... / -..-'  letters separated by one space, words by ' / '
```

## morse/dsp.py

```python
def goertzel_power(block: np.ndarray, f0: float, fs: int) -> float
    # linear power, normalised so a full-scale sine at f0 returns 1.0

class Goertzel:
    def __init__(self, f0: float, fs: int, block_size: int): ...
    f0: float                                   # read-only property
    def set_frequency(self, f0: float) -> None
    def power(self, block: np.ndarray) -> float      # linear, normalised as above
    def power_db(self, block: np.ndarray) -> float   # 10*log10(power + 1e-12)

class BandPass:
    def __init__(self, f0: float, fs: int, half_width: float = 100.0, order: int = 4): ...
    def set_frequency(self, f0: float) -> None       # resets filter state
    def process(self, block: np.ndarray) -> np.ndarray   # same length, state carried between calls

def spectrum(frame: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]
    # Hann window, rfft; returns (freqs_hz, power_db) with power_db normalised
    # so a full-scale sine peaks near 0 dB

def find_tone_frequency(freqs: np.ndarray, power_db: np.ndarray,
                        fmin: float = 300.0, fmax: float = 8000.0) -> tuple[float, float]
    # returns (peak_hz, prominence_db) where prominence is peak minus the
    # median of power_db within [fmin, fmax]

def synth_keyed_tone(pattern: list[tuple[bool, float]], f0: float, fs: int,
                     amplitude: float = 0.3, noise_rms: float = 0.0,
                     reverb_ms: float = 0.0, seed: int = 0) -> np.ndarray
    # test helper: pattern is [(on, duration_ms), ...]; reverb_ms > 0 applies a
    # simple exponential tail so marks lengthen and gaps shorten
```

## morse/tone_detector.py

```python
class ToneDetector:
    def __init__(self, block_ms: float = 10.0, on_frac: float = 0.6, off_frac: float = 0.4,
                 min_run_blocks: int = 2, floor_rise_db: float = 0.02,
                 peak_fall_db: float = 0.02, min_dynamic_db: float = 12.0): ...
    def update(self, power_db: float) -> list[Run]
        # feed one block; returns runs that are now final (debounced), oldest first,
        # usually [] and occasionally one or more
    def flush(self) -> list[Run]        # emit everything pending, including the current run
    def reset(self) -> None
    state: bool                          # current ON/OFF verdict
    floor_db: float                      # tracked noise floor (drops instantly, rises floor_rise_db per block)
    peak_db: float                       # tracked signal peak (rises instantly, falls peak_fall_db per block)
    threshold_hi_db: float               # floor + on_frac * max(peak - floor, min_dynamic_db)
    threshold_lo_db: float               # floor + off_frac * (same span)
    current_run: Run                     # the in-progress run (not yet final)
```

Semantics: ON when power rises above `threshold_hi_db`, OFF when it falls
below `threshold_lo_db`. A completed run shorter than `min_run_blocks` is
merged into the runs on either side (the three become one run with the outer
state). A run is final only when the run after it has reached
`min_run_blocks`, so a run's length can no longer change.

## morse/decoder.py

```python
class MorseDecoder:
    def __init__(self, wpm: float | None = None, adaptive: bool = True, window: int = 30): ...
    def feed(self, run: Run) -> str      # consume one final run, return newly emitted text ('' if none)
    def idle(self, off_ms: float) -> str  # length of the current, unfinished OFF run; flushes the
                                          # pending letter (and adds a space) once off_ms + offset_ms > 7*dit_ms
    def reset(self, keep_timing: bool = False) -> None
    dit_ms: float                        # T
    offset_ms: float                     # d, the reverb correction
    wpm: float                           # 1200 / dit_ms
    buffer: str                          # pending symbols, e.g. '.-'
    text: str                            # everything emitted so far
    letter_count: int
    unknown_count: int
```

Timing rule (from the plan): keep the last `window` mark lengths and gap
lengths. `M` = 10th percentile of marks, `G` = 10th percentile of gaps after
capping each gap at `20*M`. `T = (M+G)/2`, `d = (M−G)/2`; if `d < 0` use
`T = M, d = 0`. With fewer than two marks or one gap, use `T = 1200/wpm` if
`wpm` was given, else `T = 150 ms`, and `d = 0`. If `adaptive` is False and
`wpm` is given, `T` stays fixed and only `d` adapts. Marks are corrected as
`ms − d`, gaps as `ms + d`. Mark `< 2T` is a dit, else a dah. Gap `< 2T` is
inside a letter; `2T..5T` ends the letter; `≥ 5T` ends the word (one space).
Unknown symbol sequences emit `?` and increment `unknown_count`. A leading gap
before any mark emits nothing.

## morse/pipeline.py

```python
@dataclass
class BlockResult:
    power_db: float
    on: bool
    threshold_hi_db: float
    threshold_lo_db: float
    floor_db: float
    peak_db: float
    level_dbfs: float                    # 20*log10(rms of the raw block)
    new_text: str                        # text emitted by the decoder during this block
    runs: list[Run]                      # runs finalised during this block
    filtered: np.ndarray                 # band-passed block for display

class Pipeline:
    def __init__(self, fs: int = 48000, block_size: int = 480, f0: float = 2491.0,
                 wpm: float | None = None, adaptive: bool = True): ...
    def process_block(self, block: np.ndarray) -> BlockResult
    def set_frequency(self, f0: float) -> None
    def flush(self) -> str               # end of stream: finalise runs, decode, return remaining text
    goertzel: Goertzel
    bandpass: BandPass
    detector: ToneDetector
    decoder: MorseDecoder
    text: str                            # decoder.text

def decode_wav(path: str, f0: float, wpm: float | None = None,
               block_size: int = 480) -> tuple[str, list[Run]]
    # reads a WAV (any int or float format, mono or first channel), resamples
    # nothing (uses the file's fs), returns (decoded_text, all_runs)
```

`process_block` order: rms level, band-pass, Goertzel dB, detector update,
decoder feed for each finalised run, then `decoder.idle(current OFF run ms)`
when the current run is OFF.

## morse/audio_input.py

```python
@dataclass
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    default_samplerate: float

def list_devices() -> list[DeviceInfo]             # input-capable only
def resolve_device(spec: str | int | None) -> int
    # int -> that index; str -> case-insensitive substring of name, prefer
    # hostapi containing 'WDM-KS'; None -> a device whose name contains
    # 'Microphone Array 1' on WDM-KS if present, else the system default input

class AudioInput:
    def __init__(self, device: int | str | None = None, fs: int = 48000, block_size: int = 480,
                 queue_size: int = 1000): ...
    def start(self) -> None
    def stop(self) -> None
    def read_blocks(self) -> list[np.ndarray]     # drain everything queued, non-blocking
    dropped: int                                   # blocks lost to a full queue
    device_index: int
    device_name: str
    def __enter__(self) / __exit__(...)
```

The PortAudio callback copies `indata[:, 0]` into a `queue.Queue` and does
nothing else. Only one input stream may be open at a time on this machine.

## morse/app.py

```
python -m morse.app --list-devices
python -m morse.app --wav PATH [--freq HZ] [--wpm N] [--verbose]     # offline: print decoded text
python -m morse.app --no-ui [--device SPEC] [--freq HZ] [--wpm N]     # live, text to stdout
python -m morse.app [--device SPEC] [--freq HZ] [--wpm N]             # live with the Qt UI
```

Defaults: `--freq 2491`. `--verbose` prints every finalised run as
`ON 110 ms` / `off 50 ms` before the text. Without `--wav`/`--no-ui` it
imports `morse.ui` and calls `run_ui(args)`; if `morse.ui` is missing it exits
with a clear message pointing at `--no-ui`. Exit code 0 on success.

## tools/beep_sender.py

Standalone, standard library only, runs on the beeper machine:

```
python beep_sender.py "HELLO WORLD" [--wpm 8] [--freq 2491] [--dry-run] [--repeat N]
```

Windows: `winsound.Beep(freq, ms)` for marks, `time.sleep` for gaps. Other
platforms: use the `beep` command if present (`beep -f FREQ -l MS`), else
print the timing. `--dry-run` prints the `(on, ms)` sequence and exits.
Timing: dit `1200/wpm` ms, dah 3 dits, intra-letter gap 1 dit, letter gap 3,
word gap 7.

## morse/player.py (built with the UI phase)

```python
def build_timing(text: str, wpm: float) -> list[tuple[bool, float]]
    # (on, ms) sequence: dit 1200/wpm, dah 3 dits, intra-letter gap 1, letter gap 3, word gap 7;
    # unknown characters are skipped; leading/trailing gaps are not included

def render_tone(timing: list[tuple[bool, float]], f0: float, fs: int = 48000,
                amplitude: float = 0.3, ramp_ms: float = 3.0) -> np.ndarray
    # float32 sine keyed by the timing, with linear ramps at each edge to avoid clicks

class TonePlayer:
    def __init__(self, fs: int = 48000, device: int | str | None = None): ...
    def play(self, samples: np.ndarray) -> None     # non-blocking, sounddevice.play
    def stop(self) -> None
    playing: bool
```

Imports sounddevice lazily. Playing through the laptop speakers while the
microphone is open is allowed: output and input are separate streams.

## morse/ui.py (built after the design review)

`run_ui(args) -> int`. pyqtgraph + PySide6, layout per the approved mock
(artifact "Beeper Morse Console"):

- Toolbar: input device, tone frequency with Auto-detect, speed Auto/Manual,
  Pause, Save 30 s, Clear text.
- Spectrum 0–8 kHz with markers at f0 and 3f0.
- Tone power over the last 10 s with the hysteresis band and the ON/OFF strip.
- Decoded text with the pending letter, dit/dah/gap/offset readouts and a
  histogram of the last 30 mark lengths.
- Right rail: input level in dBFS with peak hold, 6 ms band-passed waveform,
  detector state, level, signal/floor, speed, letter and unknown counts.
- Encode strip, full width under the plots: message field, speed, Play tone,
  Copy; Morse output with letters separated by a space and words by ` / `;
  a keying guide drawn to scale (marks and gaps, letters labelled above) with
  a playhead while playing; duration, dit/dah and gap readouts. Uses
  `morse.table.encode`, `morse.player.build_timing`, `render_tone`, `TonePlayer`.
- Status bar: listening state, block size, dropped blocks, elapsed time.

A 30 Hz QTimer drains `AudioInput` and feeds `Pipeline`.
