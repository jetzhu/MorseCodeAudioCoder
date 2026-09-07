# Morse Code Audio Decoder — Implementation Plan

Toy project: listen to the laptop microphone, isolate the tone of a PC onboard
beeper coming from another machine, show what is heard in a live plot, and
decode the on/off pattern as Morse code.

## 1. Stack

**Python 3.13** (already installed) with:

| Package | Role | Status on this machine |
|---|---|---|
| `sounddevice` | mic capture (PortAudio is bundled in the wheel) | verified working today |
| `numpy`, `scipy` | DSP (Goertzel, band-pass, FFT) | already installed |
| `pyqtgraph` + `PySide6` | real-time plots and the control UI | to install |
| `pytest` | tests | to install |

Why Python and not a browser page: numpy/scipy are already here, the DSP is
easy to unit-test offline, and pyqtgraph redraws several traces at 30 fps
without effort. matplotlib is installed too but is too slow for a smooth
scrolling display, so it is only used for offline debug plots.

Use a project venv (`python -m venv .venv`) so the Windows Store Python stays
untouched.

## 2. Architecture

```
mic ──► AudioInput ──► DSP (per 10 ms block) ──► ToneDetector ──► MorseDecoder ──► text
         (thread)         │ Goertzel power at f0        │ on/off + duration     │
                          │ band-pass waveform          │                       │
                          └─► FFT (every ~50 ms) ───────┴─► UI (30 fps timer) ◄─┘
```

| Module | Responsibility | Notes |
|---|---|---|
| `morse/audio_input.py` | Opens a `sounddevice.InputStream`, 48 kHz mono, 480-sample blocks (10 ms). The callback only copies the block into a `queue.Queue`. | Device selectable by index or by name substring plus host API. Never do DSP inside the PortAudio callback. |
| `morse/dsp.py` | Pure functions and small stateful classes: `goertzel_power(block, f0, fs)`, stateful band-pass (`scipy.signal.sosfilt` with carried `zi`), `spectrum(frames)` for the display, `find_tone_frequency(spectrum)` for auto-detect. | No Qt, no audio; fully testable with synthetic arrays. |
| `morse/tone_detector.py` | Turns the per-block tone power series into ON/OFF events with durations. Adaptive threshold with hysteresis and a minimum run length. | Emits `(state, duration_ms)` on each transition. |
| `morse/decoder.py` | Timing state machine: ON runs become dit/dah, OFF runs become symbol/letter/word gaps, adaptive dit-length estimate, table lookup. | Pure Python, fully unit-tested with synthetic timing sequences. |
| `morse/table.py` | Morse table (letters, digits, common punctuation) and its inverse. | |
| `morse/ui.py` | pyqtgraph window: spectrum, tone power with threshold, ON/OFF strip, decoded text, controls. A `QTimer` at ~30 Hz drains the queue and drives the pipeline. | The only module that imports Qt. |
| `morse/app.py` | CLI entry: `--device`, `--freq`, `--wpm`, `--wav file` (offline mode), `--list-devices`. | Offline mode runs the same pipeline over a WAV file, for debugging with saved recordings. |
| `tools/beep_sender.py` | Runs **on the beeper machine**: plays text as Morse through the PC speaker (`winsound.Beep` on Windows; `beep` or the console ioctl on Linux). Args: text, WPM, frequency. | Makes end-to-end tests repeatable. |
| `tools/probe/` | Today's probe scripts (record, analyze, device comparison, live monitor, acoustic loopback), kept as diagnostics. | |
| `tests/` | Unit tests for decoder and tone detector; a synthetic-audio test; regression tests against recorded WAV fixtures in `tests/fixtures/`. | |

## 3. Signal chain

| Stage | Choice | Reason |
|---|---|---|
| Sample rate / block | 48 kHz, 480 samples = 10 ms | 10 ms resolution is plenty: a dit at 20 WPM is 60 ms, at 5 WPM 240 ms. |
| Tone energy | Goertzel at `f0` per block, plus one bin either side (±50 Hz) summed | Cheap, narrow, the classic Morse-decoder approach. Tolerates a slightly off `f0`. |
| Display waveform | 4th-order Butterworth band-pass, `f0 ± 100 Hz`, `sosfilt` with carried state | Shows the isolated beeper signal instead of room noise. |
| Spectrum | 2048-point FFT over the last ~43 ms, Hann window, 300 to 5000 Hz shown, marker at `f0` | Lets the user see the beeper peak and tune or auto-detect `f0`. |
| Auto-detect `f0` | Average the spectrum over ~1 s while the user holds a beep, pick the most prominent peak (peak over in-band median) in 300 to 5000 Hz | PC-speaker beeps vary by machine (BIOS about 1 kHz, Windows `Beep` default 800 Hz, Linux `beep` default 440 Hz). |
| Threshold | Track noise floor (slow-rising, fast-falling min tracker) and signal peak (fast-rising, slow-decaying max tracker) in dB. ON when power > floor + 60 % of (peak − floor), OFF when < floor + 40 %. | Hysteresis prevents chatter; adaptive tracking survives level changes. |
| Debounce | A run shorter than 2 blocks (20 ms) is merged into its neighbours | Rejects clicks and the 10 ms edge dropouts seen in the loopback test (section 8). |

## 4. Morse timing decoder

Room reverb and the detector's attack/release lengthen every mark and shorten
every gap by a roughly constant offset `d` (30 ms in today's loopback test).
A shortest mark of `T + d` and a shortest gap of `T − d` therefore average to
the true dit length, and the decoder uses that to undo the distortion.

1. Maintain robust estimates of the shortest mark `M` and the shortest gap
   `G` (10th percentile over a sliding window of recent runs, or the smaller
   cluster after a two-way split). Then `T = (M + G) / 2` and
   `d = (M − G) / 2`. If `--wpm` is given, seed `T = 1200 / WPM` ms and
   `d = 0` until enough runs have arrived.
2. Correct each run before classifying it: marks become `ms − d`, gaps
   `ms + d`.
3. Corrected ON run `< 2T` is a dit, otherwise a dah. Append to the current
   symbol buffer.
4. Corrected OFF run `< 2T` is an intra-character gap, nothing to do.
   `2T ≤ gap < 5T` ends the letter: look up the buffer, append the character
   (or `?` if unknown), clear the buffer.
   `≥ 5T` ends the word: as above, then append a space.
5. Idle flush: if OFF has lasted `> 7T` with a non-empty buffer, decode it
   without waiting for the next ON edge, so the last letter appears promptly.
6. Expose `T`, `d`, current WPM, the pending symbol buffer, and the decoded
   text so the UI can show them.

Validated on the loopback fixture: measured runs were marks 110/270 ms and
gaps 50/210 ms for keyed 80/240 ms. Without the correction the dit estimate
is 110 ms, the 210 ms letter gaps fall below the 220 ms threshold, and the
whole message collapses into one unknown symbol. With it, `T` comes out at
exactly 80 ms and the fixture decodes to `SOS`.

## 5. UI layout (pyqtgraph)

```
┌──────────────────────────────────────────────────────────┐
│ Device [▼]  f0 [ 1000 ] Hz [Auto]  WPM [auto] [Clear]     │
├──────────────────────────────────────────────────────────┤
│ Spectrum 300–5000 Hz, vertical marker at f0              │
├──────────────────────────────────────────────────────────┤
│ Tone power (dB) last 10 s, threshold line, ON/OFF strip  │
├──────────────────────────────────────────────────────────┤
│ Symbols: · · ·  − − −    WPM 12   level −38 dB           │
│ Decoded: SOS SOS TEST                                    │
└──────────────────────────────────────────────────────────┘
```

Plots update from a 30 Hz `QTimer`; each tick drains all queued blocks, runs
them through the pipeline, and appends to ring buffers backing the plots. The
live input level is always visible so a dead or muted input is obvious.

**Encode strip** (added 2026-09-07 at the design review): a full-width strip
under the plots turns typed English into Morse. It shows the dots and dashes
(letters separated by a space, words by a slash), a keying guide drawn to
scale at the chosen speed with letters labelled so the operator can follow it
while pressing the button, duration and timing readouts, a Copy button, and
Play tone, which keys a sine at the tone frequency through the laptop speakers
so the decoder can be exercised without the desktop. Backend: `morse.table.encode`
plus a small `morse/player.py` (timing builder, tone renderer, sounddevice
playback); see `docs/INTERFACES.md`.

The approved interactive mock is the artifact "Beeper Morse Console"; the
spectrum range there is 0–8 kHz so the third harmonic is visible, superseding
the 300–5000 Hz range in section 3.

## 6. Tests

- `tests/test_decoder.py`: feed hand-built `(state, ms)` sequences for known
  strings at several WPM, with ±20 % jitter, assert the decoded text.
- `tests/test_tone_detector.py`: step signals with noise; assert transitions
  and debounce behaviour.
- `tests/test_synthetic_audio.py`: generate a keyed 1 kHz tone with noise at
  a few SNRs, run the full pipeline offline, assert the text.
- `tests/fixtures/*.wav`: real recordings (the loopback SOS from today, and
  beeper recordings once captured); a regression test decodes each and
  compares to its known text.

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | Repo scaffold, venv, `requirements.txt`, `--list-devices`, offline WAV mode, probe scripts moved in | `python -m morse.app --wav tests/fixtures/loopback_sos.wav` prints tone power per block |
| M1 | Live spectrum and tone-power plot | Beeper peak visible in the spectrum, power trace rises on each beep |
| M2 | Tone detector with adaptive threshold and debounce, ON/OFF strip | Strip follows the beeps cleanly, no chatter |
| M3 | Morse decoder plus unit tests, wired into the UI | Synthetic-audio test passes; loopback fixture decodes to SOS; live SOS decodes |
| M4 | `tools/beep_sender.py` and an end-to-end run against the real beeper | A sentence sent from the beeper machine decodes with at most one error |
| M5 | Polish: auto-detect `f0`, WPM adaptation, device picker, settings persistence | |

## 8. Findings from today's probes (2026-09-07)

- **Default mic path is heavily processed.** The default endpoint
  ("Microphone Array (Intel Smart Sound Technology for Digital Microphones)",
  MME and WASAPI shared) records an ambient floor near −90 dBFS, about 40 dB
  below the raw WDM-KS endpoint ("Microphone Array 1", device index 18 in
  `sounddevice` today). That pattern is driver noise suppression or gating,
  which can attenuate a steady tone. Default to the WDM-KS endpoint on this
  laptop, selected by name and host API rather than by index.
- **Acoustic loopback validates the chain.** A 15 WPM SOS at 1000 Hz played
  through the laptop speakers and captured on the raw endpoint gave a 60 dB
  signal-to-floor ratio at the Goertzel output and a spectral peak exactly at
  1000 Hz. After a 20 ms debounce the runs were: dits 110 ms (keyed 80),
  dahs 270 ms (keyed 240), intra-symbol gaps 50 ms (keyed 80), letter gaps
  210 ms (keyed 240). Tone edges produced 10 ms dropouts before debouncing.
  The recording is kept as `tests/fixtures/loopback_sos_1khz_15wpm.wav` and
  decodes to `SOS` with the offset correction described in section 4.
- WASAPI exclusive mode refused 48 kHz on the array; not needed.
- Windows microphone privacy consent is Allow for desktop apps.
- Opening the default MME endpoint while the WDM-KS endpoint is open fails
  with "Device unavailable"; the app must hold only one input stream.
- **Beeper measured: 2491 Hz.** The source is a desktop PC speaker about
  1 m from the laptop. A 7.6 s beep captured on the raw endpoint gave:
  fundamental 2490.97 Hz with under 5 Hz drift, matching the PC timer
  (1.193182 MHz / 479 = 2491 Hz), so this is a classic BIOS-style beep;
  square-wave harmonics at 3f −11 dB, 4f −25 dB, 2f −31 dB, 5f −34 dB;
  steady level −41 dBFS rms; Goertzel signal-to-floor 46 dB; attack under
  10 ms; release drops about 15 dB within 10 ms then a room tail of roughly
  50 ms. Default `f0` to 2491 Hz. Fixture:
  `tests/fixtures/beeper_long_2491hz_1m.wav` (first 12 s of the capture).
- **The "one long beep" was four tones.** The 7.6 s capture contains ON
  runs of 3.7 s, 0.49 s, 0.57 s and 2.0 s separated by silences of 370, 250
  and 300 ms; during the silences the spectrum is identical to the quiet room.
  Whatever drives the desktop speaker inserts gaps, which Morse timing will
  see as symbol or letter boundaries. Before M4, establish how the beeper is
  driven and use `tools/beep_sender.py` (explicit per-symbol on/off timing)
  rather than a held key or a repeated `beep` command.
- Earlier windows (20 s, 20 s, 60 s, 300 s) captured nothing because the
  beeps were not made while a window was open; a listening window has to
  stay open for minutes and the user reports back when done.

## 9. Risks

- **Beeper too quiet or too far.** The loopback used the laptop's own
  speakers at moderate volume. A small PC-speaker piezo across a room may not
  clear the floor. Mitigation: bring the beeper within a metre, and show the
  live level and spectrum so the user can see when it does register.
- **Driver processing eats the tone.** Mitigation: raw endpoint; if that
  still fails, disable audio enhancements for the array in Windows Sound
  settings or use a USB or headset mic.
- **Room reverb smears gaps** and turns two dits into a dah. Mitigation:
  hysteresis, 20 ms debounce, adaptive dit estimate.
- **Harmonics.** A square-wave beeper has strong odd harmonics; the Goertzel
  stage only looks at the fundamental, so this is harmless.
- **Very slow or irregular hand keying** breaks the 3:1 timing assumption.
  Mitigation: manual WPM override; decoding is best-effort.

## 10. Setup

```powershell
cd C:\Users\jetzhu\Projects\OOPolaris\MorseCodeAudioDecoder
python -m venv .venv
.\.venv\Scripts\python -m pip install sounddevice pyqtgraph PySide6 numpy scipy pytest
.\.venv\Scripts\python -m morse.app --list-devices
.\.venv\Scripts\python -m morse.app --device "Microphone Array 1" --freq 1000
```
