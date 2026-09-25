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
  2020 ms separated by gaps of about 370, 250 and 300 ms, starting near 0.69 s
  (those figures come from a midpoint threshold; the signal-referenced release
  trims 20 to 70 ms from each mark, so tests allow 100 ms).

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

Version 2 (2026-09-07). Version 1 tracked the floor as a slowly rising
minimum of the dB series, which fails in three measured ways: digital silence
at stream start (about −119 dB in both fixtures) primes it far below the real
room noise; after tens of seconds without a tone the peak decays until the
threshold sits inside the noise spread (single-bin Goertzel noise has a
standard deviation near 5.6 dB, so its spikes reach 15 dB above the mean);
and a jump in the noise level can leave the detector ON with nothing that
brings it back. Version 2 tracks averages in the linear domain and sizes the
lift above the noise mean to those statistics.

Measured adjustments (2026-09-08, see the module docstring for the numbers):
the noise alphas are equal, 0.1 up and 0.1 down. An asymmetric linear average
of exponentially distributed noise power is biased low by about 4 dB, which
put the OFF threshold at the noise mean and produced 8 to 14 false runs per
minute; equal alphas make N the unbiased mean, and 0.1 lets N follow the
600 to 700 ms fade-in at stream start that both fixtures show. A NaN or +inf
power is treated as digital silence. Thresholds are recomputed after the level
update so `threshold_hi_db == floor_db + lift_on` holds at all times.

```python
class ToneDetector:
    def __init__(self, block_ms: float = 10.0, on_frac: float = 0.6, off_frac: float = 0.4,
                 min_run_blocks: int = 2, min_lift_db: float = 12.0, max_lift_db: float = 30.0,
                 min_off_lift_db: float = 6.0, noise_alpha_up: float = 0.1,
                 noise_alpha_down: float = 0.1, signal_alpha: float = 0.1,
                 signal_decay_db: float = 0.1, silence_floor_db: float = -100.0,
                 warmup_blocks: int = 30, max_on_blocks: int = 500,
                 release_db: float = 12.0, hysteresis_db: float = 3.0): ...
    def update(self, power_db: float, tonal: bool = True) -> list[Run]
        # feed one block; returns runs that are now final (debounced), oldest first,
        # usually [] and occasionally one or more; tonal=False (a broadband block) can
        # never switch an OFF detector ON but still updates the noise level
    def flush(self) -> list[Run]        # emit everything pending, including the current run
    def reset(self) -> None
    state: bool                          # current ON/OFF verdict
    floor_db: float                      # tracked noise level N (mean of OFF-block power, in dB)
    peak_db: float                       # tracked signal level S (mean of ON-block power, in dB)
    threshold_hi_db: float               # max(N + lift_on, threshold_lo_db + hysteresis_db)
    threshold_lo_db: float               # max(N + lift_off, S - release_db)
    current_run: Run                     # the in-progress run (not yet final)
    warming_up: bool                     # True during the first warmup_blocks after construction or reset
```

Level tracking, all in the linear power domain, converted to dB for the
properties (`10*log10(x + 1e-12)`):

- `N` (noise) starts at the first block's power and is updated only by blocks
  whose verdict is OFF: `N += alpha * (p - N)` with `alpha = noise_alpha_down`
  when `p < N`, else `noise_alpha_up`. `N` is clamped so that `floor_db >=
  silence_floor_db`; digital silence therefore never lowers the thresholds
  below `silence_floor_db + min_lift_db`.
- `S` (signal) starts at `N` and is updated only by ON blocks with
  `signal_alpha`. While the verdict is OFF, `S` decays toward `N` by
  `signal_decay_db` per block (never below `N`).
- `snr = peak_db - floor_db`. `lift_on = min(max_lift_db, max(min_lift_db,
  on_frac * snr))`; `lift_off = min(max_lift_db - 6, max(min_off_lift_db,
  off_frac * snr))`. Then `threshold_lo_db = max(floor_db + lift_off,
  peak_db - release_db)` and `threshold_hi_db = max(floor_db + lift_on,
  threshold_lo_db + hysteresis_db)`, with `release_db = 12` and
  `hysteresis_db = 3` as constructor keywords. At low SNR both thresholds are
  referenced to the noise level and protect against noise spikes; at high SNR
  they follow the tracked signal, so a loud tone is released as soon as its
  room tail has dropped 12 dB instead of when it has decayed to the
  noise-referenced lift (measured 2026-09-08: the noise-referenced release
  alone failed at 12 WPM and above with a 30 ms tail once the tone was 46 dB
  or more above the noise). `signal_decay_db` defaults to 0.1 so a beeper
  moved farther away is followed within about a second. Thresholds are
  recomputed every block before the verdict.
- Warm-up: during the first `warmup_blocks` after construction or `reset()`
  every block is OFF and updates `N` with `alpha = 0.3`, so a stream that
  opens with silence or a fade-in settles onto the real noise level before any
  verdict is made. `flush()` does not restart the warm-up.
- Stuck-ON timeout: when the current ON run reaches `max_on_blocks`, the run
  is ended as OFF from that block on and `N` is set to the current `S` (the
  level was noise after all). No Morse mark lasts 5 s; this recovers from a
  noise-level jump within `max_on_blocks`.

Verdict: ON when `p_db > threshold_hi_db`, OFF when `p_db < threshold_lo_db`,
otherwise unchanged (hysteresis). Debounce and finality are unchanged from
version 1: a completed run shorter than `min_run_blocks` is merged into the
runs on either side (the three become one run with the outer state); a run is
final only when the run after it has reached `min_run_blocks`. A short first
run is joined to the run that follows it.

Acceptance, all with the defaults: both fixtures produce the run lists in
section "Conventions" above (the loopback dahs measure about 250 ms under
v2, since the OFF threshold now sits above the room's release tail); 60 s of
Gaussian noise at any fixed level from −110 to −40 dBFS produces zero ON
runs; a tone 20 dB above that noise keyed as SOS at 15 WPM decodes; 2 s of
digital silence followed by noise no louder than −80 dBFS, then the same
keyed tone, decodes with no ON run during the silence or the noise (louder
noise arriving after a silence longer than the warm-up is indistinguishable
from a tone; it then produces ON runs totalling under `2 * max_on_blocks`
before the detector settles, and the tone that follows decodes); a noise
level that jumps 25 dB and stays there produces ON runs totalling at most
`2 * max_on_blocks` blocks, nothing ON after `jump + 2 * max_on_blocks`, and a
floor within 3 dB of the new mean (a clean dB step gives exactly one run of
`max_on_blocks`).

Known limitation: `lift_off` is capped at `max_lift_db − 6 = 24 dB`, so at
very high signal-to-noise ratios the release is detected only once the room
tail has decayed by the full lift, which lengthens marks and shortens gaps by
more than at moderate ratios. Real recordings at 1 m gave an offset near
10 ms; synthetic tests use a 10 ms tail. Fast keying above 20 WPM in a very
reverberant room is the case most likely to suffer.

Module functions (2026-09-22): `tonality_db(power_db, level_dbfs)` returns
`power_db − level_dbfs − 3.01`, which is 0 for a pure tone at `f0` and about
−24 for white noise or a click (a 100 Hz bin holds 1/240 of broadband
energy). `is_tonal(power_db, level_dbfs, min_tonality_db=TONALITY_MIN_DB)` is the
`tonal` flag for `update`: `tonality_db >= −15`. `update(power_db,
tonal=True)`: a block that is not tonal can never switch an OFF detector ON,
but it still updates the noise level like any OFF block (an earlier design
that reported such blocks at the floor starved the noise tracker and produced
false marks during the loopback fixture's fade-in); while ON the flag is
ignored, so a click during a mark does not chop it. Mirrored by `tonalityDb`,
`isTonal`, `TONALITY_MIN_DB` and `update(powerDb, tonal)` in
`web/js/detector.js`.

## morse/decoder.py

```python
class MorseDecoder:
    def __init__(self, wpm: float | None = None, adaptive: bool = True, window: int = 30): ...
    def feed(self, run: Run) -> str      # consume one final run, return newly emitted text ('' if none)
    def idle(self, off_ms: float) -> str  # length of the current, unfinished OFF run; flushes the
                                          # pending letter (and adds a space) once off_ms + offset_ms > 7*dit_ms
    def reset(self, keep_timing: bool = False) -> None
    def adopt_timing(self, other: MorseDecoder) -> None   # take other's run windows and trust; recompute T, d
    dit_ms: float                        # T
    offset_ms: float                     # d, the reverb correction
    wpm: float                           # 1200 / dit_ms
    timing_ready: bool                   # False while the first runs are held back (see below)
    pending_count: int                   # runs held back
    provisional: str                     # what the held runs read as under the current estimate, e.g. '.. -'; '' when none
    buffer: str                          # pending symbols, e.g. '.-'
    text: str                            # everything emitted so far
    letter_count: int
    unknown_count: int
```

Hold-back (2026-09-22). Without a WPM seed a mark alone cannot tell a slow
dit from a fast dah, and the old 150 ms seed misread every message that
opened with T, M, O or a digit ("OSO" became "SSO", "TEST" became "IST") and
every message slower than 5 WPM. Runs are therefore held back until the
estimate is trusted: both mark classes seen (longest mark ≥ 2 × shortest)
or three marks, except that three marks of one class that look like dahs
(≥ 2.5 × the shortest gap) with no letter gap yet to arbitrate wait for a
letter gap or five marks, since reverberant dits with jitter look the same.
The held runs are then classified in one go. Once trusted,
a mark whose measured length is below `0.4 T` is a glitch (a click): it is
neither a symbol nor timing evidence, and a gap below `0.4 T` is treated the
same way, so clicks after a message cannot shrink the dit estimate (which
used to turn everything after them into dahs and word gaps). Four such marks
in a row with Morse-like spacing (each preceded by a gap shorter than eight
times its own length) are a faster keying speed, not clicks: a valve opens
and short marks are accepted until a long gap closes it again, so the window
adapts; a click after silence always resets the count. Measured lengths, not
corrected ones, are compared, so a transitional offset estimate cannot lock a
new speed out. Slowdown resync (2026-09-22): a speed-up re-locks within
three marks because new short dits become the 10th percentile at once, but
a slowdown used to wait for the whole 30-run window to drain while every new
dit read as a dah ("T T T T"). Now three consecutive marks of at least
`1.5 T`, with no mark or gap as short as a dit between them, rebuild the
windows from those recent runs, re-read the letter in progress and suspend
the glitch rule for eight marks; marks between `2.2 T` and `4.5 T` could
genuinely be dahs (T, M, O, digits; jitter and the low-biased percentile
widen the band), so when any falls in that band five are required. An
intra-letter gap ends the streak, so a run of dahs inside a letter never
triggers it, and a fixed speed (`wpm` given, `adaptive` False) disables it.
While holding,
`idle()` flushes after the longer of `7T` and `3.5 ×` the longest held mark
(that mark may be a dah whose letter gap is as long as itself), using the
best estimate available; a lone mark falls back to the seed (below 300 ms is
E, else T). A seed (`wpm` given) disables the hold-back. Supported speeds:
2 to 40 WPM.

Timing rule (from the plan): keep the last `window` mark lengths and gap
lengths. `M` = 10th percentile of marks, `G` = 10th percentile of gaps after
capping each gap at `20*M`. Percentiles use the nearest-rank method: sort
ascending and take the element at 1-based rank `ceil(q/100 * n)`, never
interpolating (so the minimum for `n <= 10`). A mark of `max_mark_ms = 5000`
or longer (`ms >= max_mark_ms`) is not a symbol: it is ignored, the pending
buffer is cleared, and nothing is emitted. The detector's stuck-ON timeout
produces a run of exactly `max_on_blocks * block_ms = 5000 ms`, which this
rule therefore drops; long button presses and detector timeouts never produce
`?` or a stray dah. `T = (M+G)/2`, `d = (M−G)/2`; if `d < 0` use
`T = M, d = 0`. Dah rule: when only one mark class has been seen (longest
mark < 2 × shortest) and `M ≥ 2.5 G`, the marks are dahs: `T = (M+G)/4`,
`d = (M−3G)/4` (clamped at 0). With fewer than two marks or one gap, use `T = 1200/wpm` if
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
    tonality_db: float                   # tone power minus block power minus 3 dB: 0 for a pure tone, about -24 for noise

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

`process_block` order: sanitise (a block containing NaN or inf is replaced by
zeros and counted in `bad_blocks: int`), rms level, band-pass, Goertzel dB,
detector update with `tonal=is_tonal(power_db, level_dbfs)` (a block whose
tone power is more than 15 dB below its total power plus 3 dB is a broadband
transient: keyboard clicks, doors and speech teach the noise level but never
switch the detector ON), decoder feed for each finalised run, then
`decoder.idle(current OFF run ms)` when the current run is OFF. The
`Pipeline` constructs `ToneDetector()` with the contract defaults and does not
override any threshold parameter. `set_frequency(f0)` validates
`0 < f0 < fs/2` (ValueError otherwise), then finalises what the detector holds
(`detector.flush()` fed to the decoder), resets the detector (level statistics
belong to the old frequency), and re-tunes the Goertzel and band-pass; the
decoder keeps its timing estimates and text. `Pipeline(f0=...)` applies the
same validation. `decode_wav` takes channel 0 of multi-channel files;
`decode_samples(samples, fs, ...)`, if present, takes channel 0 of a 2-D array
and raises on more than two dimensions.

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

The PortAudio callback copies `indata[:, 0]` into a `queue.Queue`, counts
`dropped` on a full queue and `overflows: int` when PortAudio reports an input
overflow, and does nothing else. Only one input stream may be open at a time
on this machine.

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
with a clear message pointing at `--no-ui`. Exit code 0 on success. Any
`OSError` from opening the WAV (missing file, a directory, permission) exits 2
with a one-line message, never a traceback. Output is written with
`errors="replace"` (or `sys.stdout.reconfigure(encoding="utf-8",
errors="replace")`) because device names on this machine contain `®`.

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
def build_timing(text: str, wpm: float, farnsworth_wpm: float | None = None) -> list[tuple[bool, float]]
    # (on, ms) sequence: dit 1200/wpm, dah 3 dits, intra-letter gap 1, letter gap 3, word gap 7;
    # unknown characters are skipped; leading/trailing gaps are not included.
    # farnsworth_wpm below wpm stretches the letter and word gaps (ARRL rule, see farnsworth_gaps)

def farnsworth_gaps(wpm: float, farnsworth_wpm: float | None) -> tuple[float, float]
    # (letter_gap_ms, word_gap_ms): 3 and 7 dits normally; with overall speed s below character
    # speed c, ta = (60c - 37.2s)/(s c) seconds and the gaps are 3ta/19 and 7ta/19 (18/5: 1568, 3660)

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

## Web app (`web/`)

Plain ES modules, no bundler, no framework. Everything under `web/js/` except
`audio.js`, `player.js`, `app.js` and `worklet.js` (an AudioWorkletProcessor,
which only exists inside an audio rendering thread) must run in Node 24
without a DOM so the tests can exercise it. Mirror the Python names so the two implementations can
be compared side by side.

```
web/index.html            page; starts from the approved mock (same layout, tokens, themes)
web/css/app.css           styles extracted from the mock
web/js/table.js           MORSE_TABLE, INVERSE, lookup(symbols), encode(text) — same semantics as morse/table.py
web/js/runs.js            class Run { on, blocks, blockMs = 10; get ms() }
web/js/dsp.js             goertzelPower(block: Float32Array, f0, fs) normalised like Python;
                          class Goertzel(f0, fs, blockSize) { setFrequency, power, powerDb }
web/js/detector.js        class ToneDetector — same constructor options, update(powerDb) -> Run[], flush(), reset(),
                          state, floorDb, peakDb, thresholdHiDb, thresholdLoDb, currentRun
web/js/decoder.js         class MorseDecoder({wpm, adaptive, window}) — feed(run) -> string, idle(offMs) -> string,
                          reset(), ditMs, offsetMs, wpm, buffer, text, letterCount, unknownCount
web/js/player.js          buildTiming(text, wpm) -> [{on, ms}], layoutGuideLabels(letters, xOf, widthOf) -> [{ch, x}],
                          class TonePlayer(audioContext) { play(timing, f0, gain), stop(), onProgress, addOutput(node), removeOutput(node), clearOutputs(), outputs }
web/js/keyer.js           class Keyer (port of morse/keyer.py) and class LiveKey(keyer, {gain}) { start(audioContext, f0),
                          stop(), keyDown(), keyUp(), paddleDown(w), paddleUp(w), releaseAll(), setMode(m), setSpeed(ditMs),
                          setFrequency(f0), isOn(), nowMs, addOutput(node), removeOutput(node), outputs, onChange }:
                          a persistent oscillator whose gain is automated with setTargetAtTime on the AudioContext clock
web/js/worklet.js         AudioWorkletProcessor 'block-meter': accumulates blocks of round(sampleRate/100) samples,
                          posts {rmsDb, powerDb, blockIndex}; accepts {f0} messages
web/js/audio.js           class MicInput { static listDevices(); start(deviceId, onBlock); stop(); bus (GainNode: the input of the analysis chain); analyser (AnalyserNode 2048);
                          filtered (BiquadFilterNode band-pass at f0, Q = f0/200); setFrequency(f0); sampleRate }
web/js/app.js             wiring: controls, ring buffers, canvas renderers (ported from the mock), status, hints
web/test/vectors.json     generated by tools/export_vectors.py; do not edit by hand
web/test/*.test.mjs       node --test: table, dsp, detector, decoder parity against vectors.json, WAV fixture decode
```

Behavioural requirements

- Listening starts from a Start button click. `getUserMedia` constraints:
  `{ audio: { deviceId, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } }`
  (the `Mic: Raw` default, 2026-09-22). `Mic: Browser default` leaves the three
  processing constraints out so the browser applies its own defaults, a
  different capture path for setups where raw yields nothing (some Firefox on
  Windows machines); the choice is kept in localStorage `morse.micProcessing`
  and changing it while listening reopens the microphone. The status bar shows
  what the browser reports having applied (`track.getSettings()`): `raw`,
  `processed`, or nothing when it does not say.
  Show the actual sample rate and block size in the status bar.
- Device picker lists inputs from `enumerateDevices()`; labels fill in after
  permission is granted. Remember the last device in `localStorage`, wrapped in try/catch.
- All plots, readouts, encoder and Play behave as in the mock. Auto-detect
  averages the analyser spectrum for one second and picks the most prominent
  peak between 300 and 8000 Hz.
- Feed the decoder (checkbox in the encode strip, default on): while listening,
  `TonePlayer.addOutput(mic.bus)` mixes the keyed tone straight into the
  analysis chain, so Play is decoded regardless of speakers, microphone and
  OS processing. Firefox on Windows receives the processed microphone, which
  removes the machine's own output; Chromium browsers open the raw path.
- Per block the page calls `detector.update(powerDb, isTonal(powerDb, rmsDb))`,
  exactly like `Pipeline.process_block`.
- While the decoder holds runs back, the pending-letter slot shows
  `decoder.provisional` dimmed with the hint "estimating speed…". In Auto
  mode the disabled speed field follows the live estimate once
  `timingReady`, so switching to Manual starts from the measured speed.
- Key strip (section E, 2026-09-22): Single key (Space or a press-and-hold
  button: tone while held) or Two keys (left arrow dits, right arrow dahs,
  at the encoder speed, repeating while held, both alternate, a tap during an
  element is remembered). Keyboard events are ignored while an input has
  focus and on auto-repeat; window blur releases everything. The tone goes
  to the speakers and, with Feed the decoder on while listening, to the
  decoder's input bus like Play does. A Sent line decodes the operator's own
  keying locally (an adaptive `MorseDecoder` fed from the keyer's transition
  log) with a Clear button.
- Channels, lamp, full-screen light, vibration, handset layout (2026-09-25,
  v0.1.13; step 1 of the light work): `morse/channels.py` and
  `web/js/channels.js` share the selection rules (`sanitize`, `effective`,
  `timing_state_at`, `vibration_pattern`); chips Listen with (Microphone;
  Camera greyed until step 2) and Send with (Audio, Screen light, Torch greyed
  until step 3, Vibration on devices with `navigator.vibrate`), any
  combination, all on by default, remembered (localStorage `morse.channels`,
  QSettings `channels`). Audio off mutes the speaker branch of `TonePlayer`
  (`setSpeakers`, a gain before the destination) and `LiveKey` (`speakers`
  flag in both languages) while the decoder feed is unaffected; Start needs a
  listen source; the mic gate needs Audio and Microphone. `web/js/light.js`
  `Lamp` drives every `.lamp` element and the full-screen overlay (`rx` ring
  while the detector is ON, `tx` fill while `player.stateAtNow()` or the key
  is on); the desktop has a `Lamp` widget in the rail and a `LightWindow`
  full-screen window, with a photosensitivity notice the first time
  (`morse.light.notice` / QSettings `light/notice`). Vibration follows Play
  (`vibrationPattern`) and the key (`updateKeyVibration`). The page has two
  layouts on one URL: `body.view-handset` hides the plots, rail, About and
  notes and reorders lamp, text, encoder and key with CSS; chosen by
  `?view=handset|desktop`, then localStorage `morse.view`, then narrow +
  touch; a More button reveals the hidden settings, practice and chart.
- Star on GitHub (2026-09-23, v0.1.12): a quiet link in the page footer and
  download panel with the live star count (one `GET` of the repository from
  api.github.com, named in the About privacy sentence), and in the desktop
  status bar; a one-time nudge ("Useful? Star it on GitHub") after the first
  decoded word or the first correct practice target, remembered in
  localStorage `morse.star.nudged` / QSettings `star/nudged`. No third-party
  scripts. README carries shields.io badges.
- Reference chart (2026-09-23, v0.1.11): section G in both apps, built from
  `morse/reference.py` / `web/js/reference.js` (`chart_entries(table)`:
  letters, digits, then punctuation by code length, code, character;
  `cell_state(code, buffer)` -> `"match" | "prefix" | ""`). Collapsed behind
  Show chart (QSettings `reference/open`, localStorage
  `morse.reference.open`). While open, the cells the letter in progress could
  still become are lit (the hand key's Sent buffer while the key runs, else
  the decoder's buffer) and the exact match stands out; clicking a cell plays
  the character at the encoder speed through the speakers and, with Feed on,
  into the decoder (no keying-guide playhead).
- Mic gate (2026-09-23, v0.1.10): `MicGate` in `morse/player.py` and
  `web/js/audio.js` (`update(sounding, now_ms) -> bool`, `active`, `reset`,
  tail `MIC_GATE_TAIL_MS` 400 ms, gain `MIC_GATE_GAIN` 0.01). With Feed on
  and the decoder listening, the microphone is attenuated 40 dB in the
  decoder's input while Play or the hand key sounds and for the tail after,
  so a raw microphone's delayed echo of the speakers cannot fill the gaps
  between marks; the speakers still play the beeper frequency, and outside
  those moments the microphone passes untouched (external beepers and
  recordings of the speakers decode as before). Web: a `gate` GainNode
  between the microphone source and the bus (`MicInput.setGate`), driven by
  `updateMicGate()` on player and key events and a timer; desktop:
  `_gate_mic` in `_process_block` on the pipeline's audio clock. With Feed
  off the gate never engages.
- Encode layout (2026-09-22): the message box has a full-width line of its
  own (`.enc-msg`) and the controls wrap on the line below, so a long message
  stays readable and clickable at any width. The decoded row is
  `minmax(168px, auto)` so the timing panel is never clipped; the level meter
  canvas is absolutely positioned (a canvas sized from its own clientHeight in
  an auto grid row grew on every resize in the single-column layout).
- Sent line flush (2026-09-22): `updateKeyDom` flags a redraw while the tone
  sounds or a letter is pending, so `trackSent` keeps calling `idle()` and the
  last letter closes after seven dit lengths with no further input even when
  the main decoder is not listening (before, the frame loop slept and the
  dots and dashes stayed until the next key press). The frame loop clears
  `dirty` before drawing, not after, so a draw that asks for another frame is
  not wiped (2026-09-23: with the decoder stopped the flag set during the draw
  was being reset at once and the flush never came). The desktop refreshes
  the Key strip on its timer and never had the gap.
- IndexNow (2026-09-23): `web/<key>.txt` holds the IndexNow key (the file
  name is the key); the page URL is submitted to api.indexnow.org, which feeds
  Bing, Yandex, Seznam and Naver. Google does not take IndexNow; it needs the
  owner to verify the site in Search Console.
- Search engines (2026-09-22): the page has a descriptive title and
  description, `rel=canonical` to the Pages URL, Open Graph and Twitter cards
  with `web/og.png` (1200x630, rendered by a Qt script), JSON-LD
  (`WebApplication` + `SoftwareApplication` graph, and a `FAQPage` whose
  questions are the h3s of the About section), an About/FAQ section above the
  footer, `web/robots.txt` (mock.html and test/ disallowed) and
  `web/sitemap.xml`; `mock.html` is `noindex`. Verification meta tags for
  Search Console and Bing go in the marked place in the head. The repository
  carries the Pages URL as homepage and topics. `web/test/seo.test.mjs`
  guards all of it.
- Decoded log export (2026-09-22, v0.1.8): `morse/declog.py` and
  `web/js/declog.js` (`DecodedLog.add(text, elapsed_ms, wall)`, `words()`,
  `render_text(header, now, tz)`, `render_csv(tz)`; identical output for
  identical input, clock in seconds in Python and ms in JS). Every emission
  point feeds it: `BlockResult.new_text` and `Pipeline.flush()` on the
  desktop; `decoder.feed`, `decoder.idle` (per block, at Stop and when the
  decoder is rebuilt) on the web. Clear text leaves the log alone; a new
  stream (web Start) resets it. Save log (desktop, file dialog, `.txt` table
  or `.csv` by extension) and Download log (web, Blob) export one line per
  word: computer clock of the first letter, audio time since the stream
  started, the word.
- Key bindings and sidetone (2026-09-22, v0.1.7): `morse/bindings.py` and
  `web/js/bindings.js` share the binding rules (three actions `key`, `dit`,
  `dah`; `sanitize`, `rebind` swaps a taken key, `action_for`, `is_default`).
  Each key button has a Change key link: the next press becomes the binding
  (Esc cancels, modifiers ignored, the captured key's release is swallowed);
  Reset keys restores Space and the arrows. Desktop stores Qt portable key
  names in `QSettings("MorseConsole", "Beeper Morse Console")` under
  `key/bindings` (JSON) and `key/sidetone_hz`; the web app stores
  `KeyboardEvent.code` values in localStorage `morse.key.bindings` and
  `morse.key.sidetone`. The sidetone (default 600 Hz; 0 / empty follows the
  beeper frequency) is what the speakers play while keying; the decoder feed
  is always rendered at `f0`: `LiveKey.set_sidetone` / `speaker_hz` in
  Python, `setSidetone` / `speakerHz` in JS, where a second oscillator chain
  drives the extra outputs.
- Farnsworth (2026-09-22): an optional overall speed next to the encoder speed,
  empty for off; when below the character speed the keying guide, Play, the
  duration and gap readouts use `farnsworthGaps` / `farnsworth_gaps`. The
  key strip is unaffected (it keys elements, not gaps). `tools/beep_sender.py`
  has `--farnsworth WPM`. Note that a decoder reads Farnsworth letter gaps as
  word gaps, since they exceed five dits by design.
- Keying-guide labels: every letter is labelled (`layoutGuideLabels`); only a
  label that would overlap its predecessor is skipped. Speed inputs accept
  2 to 40 WPM.
- Chopped-signal hint: if the detector's debounce removed more than 4
  fragments per second over the last 3 seconds while ON runs were present,
  show a dismissible hint: "The microphone signal looks processed. In Windows
  Sound settings turn off Audio enhancements for this microphone."
- Save 30 s writes a WAV of the last 30 s of raw input via a Blob download
  (kept in a ring buffer in the worklet or main thread).
- Works in current Chrome and Edge; degrade with a message elsewhere.
- No external scripts or fonts beyond fonts.googleapis.com; relative paths
  only, so the page works at `https://jetzhu.github.io/MorseCodeAudioCoder/`.

Parity vectors (`tools/export_vectors.py`, Python side): for each case in a
fixed list (SOS at 15 WPM clean; HELLO WORLD at 8 WPM; 20 % jitter at 12 WPM
with seed 1; reverb offset 30 ms at 15 WPM; speed change 12 to 6 WPM; unknown
symbol; idle flush), write the run list `[[on, blocks], ...]`, the expected
text, and the final `dit_ms` and `offset_ms` as produced by
`morse.decoder.MorseDecoder`. Also write, for both WAV fixtures, the Goertzel
dB series per block from `morse.dsp.Goertzel` (rounded to 0.1 dB) so the
JavaScript Goertzel can be checked against Python on real audio.

Deployment: `.github/workflows/pages.yml` runs on push to `main`: checkout,
Node 24, `node --test "web/test/*.test.mjs"` (Node 21+ treats the argument as a glob, not a directory), `actions/upload-pages-artifact` with
`path: web`, `actions/deploy-pages`. Repository Pages source is "GitHub
Actions".

Download panel (web, milestone M8): a section under the encoder titled
"Desktop app" that fetches
`https://api.github.com/repos/jetzhu/MorseCodeAudioCoder/releases/latest`,
lists each asset as a link with its size in MB and the release tag, and shows
"unzip, run morse-console.exe, no internet needed". On any fetch error or when
there is no release yet, show a link to
`https://github.com/jetzhu/MorseCodeAudioCoder/releases` and the
`pip install` alternative.

## Packaging (milestone M7 and M8)

```
pyproject.toml                    [project] name "morse-console", version from morse.__version__,
                                  dependencies as in requirements.txt (pytest under an optional "dev" extra),
                                  [project.scripts] morse-console = "morse.app:main"
packaging/morse-console.spec      PyInstaller spec: entry morse/__main__.py (add it: calls morse.app.main()),
                                  one-folder, console=True, name "morse-console", collects PySide6 and pyqtgraph
                                  data, includes the sounddevice PortAudio DLL, excludes tests and tools
.github/workflows/release.yml     on push of tag v*: windows-latest, Python 3.13, pip install . pyinstaller pytest,
                                  pytest -q, pyinstaller packaging/morse-console.spec, zip dist/morse-console
                                  as morse-console-windows-x64.zip, create the GitHub Release with that zip
                                  and a source zip (git archive)
.github/workflows/ci.yml          on push and pull request: run pytest on windows-latest and ubuntu-latest,
                                  and node --test "web/test/*.test.mjs" on ubuntu-latest
```

`morse.app.main(argv=None) -> int` must exist and be the single entry point.

## morse/keyer.py (2026-09-22)

A Morse key worked by hand, in two parts so the timing can be tested without
audio; mirrored by `web/js/keyer.js`.

```python
class Keyer:                      # pure state machine; all times are ms on one caller-chosen clock
    def __init__(self, dit_ms: float = 150.0, mode: Literal["straight", "paddle", "bug"] = "straight",
                 iambic: Literal["A", "B"] = "A", dah_ratio: float = 3.0, weight: float = 50.0): ...
    def set_speed(self, dit_ms: float) -> None
    def set_iambic(self, iambic) -> None                            # paddle mode: 'A' or 'B'
    def set_weighting(self, dah_ratio=None, weight=None) -> None    # ranges DAH_RATIO_RANGE (2..5), WEIGHT_RANGE (25..75)
    def mark_ms(self, which) -> float; gap_ms: float                # w T / (ratio + w - 1) T; (2 - w) T, w = weight / 50
    def set_mode(self, mode, t: float) -> list[tuple[float, bool]]   # releases everything, ends a sounding tone
    def release_all(self, t) -> list[tuple[float, bool]]             # drops edges logged ahead of t, then OFF at t
    def key_down(self, t) / key_up(self, t) -> list[tuple[float, bool]]        # straight key
    def paddle_down(self, which: Literal["dit", "dah"], t) / paddle_up(which, t)  # paddles
    def tick(self, t) -> list[tuple[float, bool]]   # paddle mode: start the next element when its time comes
    next_wakeup_ms: float | None                    # when tick() next has a decision to make
    def state_at(self, t) -> bool
    def transitions_since(self, index) -> tuple[list[tuple[float, bool]], int]
    def transitions_in(self, t0, t1) -> list[tuple[float, bool]]
    transition_count: int

def envelope(transitions, initial: bool, t0_ms, n, fs, ramp_ms=3.0) -> np.ndarray   # 0..1 float32 with linear ramps

class LiveKey:                    # real-time tone through a sounddevice OutputStream (lazy import)
    def __init__(self, keyer, f0=2491.0, fs=48000, device=None, amplitude=0.3, ramp_ms=3.0, block_size=480): ...
    def start(self) / stop(self); running: bool; def now_ms(self) -> float   # the keyer clock: ms since start
    def key_down(self) / key_up(self) / paddle_down(which) / paddle_up(which) / release_all(self)
    def set_mode(mode) / set_speed(dit_ms) / set_frequency(f0) / set_sidetone(hz) / set_iambic(x) / set_weighting(r, w)
    def is_on(self) -> bool
    def transitions_since(self, index)              # thread-safe
    def feed_block(self, t0_ms, n) -> np.ndarray    # the key's tone for a time window, for the decoder feed
```

Semantics. Straight: the tone follows the key. Paddle: an electronic keyer.
`paddle_down` while idle starts an element at once (`dit_ms` or `3 dit_ms`)
and logs both its edges; while an element or its trailing one-dit gap is
running, the other paddle is remembered and sent next. At the gap end
`tick` starts the next element when a paddle is held (both held alternate,
memory first), else goes idle; a tick at the element end only opens the
gap. Releasing a paddle never cuts an element short; `release_all` does.
Iambic A (default) stops with the element in progress when both paddles are
let go; iambic B sends one more opposite element when both paddles were
squeezed during the element and both are released (`_squeezed`, set when
both are held during an element, cleared when the extra element goes out).
Bug mode: the dit paddle repeats automatic dits; the dah paddle is a lever
that keys the tone directly (`_manual`): pressing it drops a dit's future
OFF edge so the tone runs on, ticks are suspended, and releasing it emits
OFF and, with the dit paddle still held, opens one space before dits
resume. Weighting: `mark_ms(which)` and `gap_ms` replace the fixed 1/3/1
dits (period constant). Both apps keep Setup (Single key / Two keys / Bug),
Iambic A/B, Dah ratio and Weight; the desktop in QSettings `key/iambic`,
`key/dah_ratio`, `key/weight`, the web in localStorage `morse.key.iambic`,
`morse.key.ratio`, `morse.key.weight` (v0.1.9, 2026-09-22).
The log is bounded (4000) and never out of time order. `LiveKey`'s stream
callback advances the keyer with the stream clock (sample-accurate elements)
and renders the envelope with 3 ms ramps; presses from the UI thread are
stamped with `now_ms()`; one lock guards every keyer call.

## morse/practice.py (2026-09-22)

Pure logic shared with `web/js/practice.js`: `WORDS` (about 150 common words),
`call_sign(rng)`, `digit_group(rng, length=5)`, `class Practice(kind, seed)`
with `next_target()`, `record(score)`, `reset_session()`, `asked`, `correct`;
`score(target, sent) -> Score(correct, errors, total, accuracy, perfect)`
by edit distance with a match count from the alignment; `rhythm(runs,
dit_ms) -> Rhythm(mark_error_pct, gap_error_pct, marks, gaps, error_pct)`
comparing each run with the nearest ideal element (marks 1 or 3 dits, gaps
1, 3 or 7). Both UIs show a Practice strip (section F): drill kind, Next
target, Check, Show the code, the target in large type, and readouts for
accuracy, the operator's speed, timing error and the session tally. The copy
is the Sent line; a perfect copy is graded automatically.

## morse/ui.py (milestone M7)

`run_ui(args) -> int`. pyqtgraph + PySide6, layout per the approved mock
(`web/mock.html`, also the artifact "Beeper Morse Console"). Must be
smoke-testable offscreen: `QT_QPA_PLATFORM=offscreen`, construct the window,
feed the loopback fixture through its pipeline in 480-sample blocks with the
timer driven manually, render once, and read `SOS` from the decoded-text
widget. Design detail:

- Toolbar: input device, tone frequency with Auto-detect, speed Auto/Manual,
  Pause, Save 30 s, Save log, Clear text.
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
- Feed the decoder checkbox in the encode strip (default on): while a source
  is open, `Play` also mixes the rendered tone into the blocks fed to the
  pipeline, advancing with the input clock (a software loopback). Keying-guide
  labels use `layout_guide_labels`, the mirror of the web function. Speed
  spin boxes accept 2 to 40 WPM. The Auto/Manual switch hands timing over with
  `MorseDecoder.adopt_timing`. In Auto mode the disabled speed spin follows the
  live estimate once `timing_ready`; while runs are held back the pending-letter
  label shows `provisional` dimmed with the hint "estimating speed…".
- Key strip (section E): the same controls and behaviour as the web page.
  Space, Left and Right are handled in `keyPressEvent`/`keyReleaseEvent` when
  no text field has focus; the big buttons take no keyboard focus. The
  `LiveKey` output stream opens on first use. With Feed the decoder on, each
  input block is stamped with the key clock at capture and `LiveKey.feed_block`
  mixes the key's tone into it, so the pipeline decodes the operator's keying.
- Status bar: listening state, block size, dropped blocks, elapsed time.

A 30 Hz QTimer drains `AudioInput` and feeds `Pipeline`.
