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
| Threshold | Track the noise level as a linear-domain average of OFF blocks and the signal level as an average of ON blocks. ON when power exceeds noise + lift, where lift is 60 % of the tracked signal-to-noise ratio, at least 12 dB and at most 30 dB; OFF at 40 % (at least 6 dB), or 12 dB below the tracked signal level when that is higher, so a loud tone is released as soon as its room tail drops 12 dB. Digital silence is clamped at −100 dB, a 300 ms warm-up settles the noise estimate, and an ON run longer than 5 s is reclassified as noise. Full rule in `docs/INTERFACES.md`. | The 12 dB minimum lift sits above the spikes of single-bin Goertzel noise (standard deviation near 5.6 dB); the earlier min-tracking design primed on stream-start silence and produced false marks after long quiet stretches. |
| Debounce | A run shorter than 2 blocks (20 ms) is merged into its neighbours | Rejects clicks and the 10 ms edge dropouts seen in the loopback test (section 8). |
| Broadband gate (2026-09-22) | Compare the tone-bin power with the block's total power. A block whose tone power is more than 15 dB below total power + 3 dB is not tonal: it can never switch an OFF detector ON, though it still teaches the noise level. The decoder additionally ignores marks and gaps shorter than 0.4 × the trusted dit (four in a row are accepted as a faster speed). | A pure tone puts all its energy in the bin (0 dB); a keyboard click or speech spreads it over the band (about −24 dB in a 100 Hz bin). Clicks used to become E and shrink the dit estimate so that everything after read as T. |

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

Speed changes (2026-09-22): the 10th-percentile window re-locks a speed-up
within three marks but was blind to slowdowns until the window drained;
three consecutive marks of at least 1.5 T now rebuild the estimate from the
recent runs (five when they could be genuine dahs). Full rule in
`docs/INTERFACES.md`.

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

### Measured operating envelope (2026-09-08, `tools/probe/snr_sweep.py`)

Synthetic keyed tone at 2491 Hz in white noise, full Python pipeline, three
noise seeds per cell, "SOS HELLO" must decode exactly. SNR is tone power
over the single-bin Goertzel noise mean.

| Room tail (1/e) | 8 WPM | 12 WPM | 15 WPM | 20 WPM | 25 WPM |
|---|---|---|---|---|---|
| 10 ms | ≥ 20 dB | ≥ 20 dB | ≥ 20 dB | ≥ 20 dB | ≥ 30 dB |
| 30 ms | ≥ 20 dB | ≥ 30 dB (2 of 3 at 20) | ≥ 30 dB (2 of 3 at 20) | fails | fails |
| 60 ms | ≥ 30 dB | fails | fails | fails | fails |

Nothing decodes at 12 or 15 dB SNR: the 12 dB minimum lift is the price of
zero false marks on noise. The real beeper at 1 m gives 46 dB and the
loopback recording's tail is near 10 ms, so hand keying up to about 15 WPM
has a wide margin. Before the signal-referenced release was added, the 30 ms
row failed at 12 WPM and above whenever the tone was 46 dB or more above the
noise.

## 8b. Hand key (added 2026-09-22)

Both apps gained a Key strip so the operator can send Morse from the
computer: a single key (Space, or a press-and-hold button) that behaves like
a straight key, and a two-key setup (left arrow dits, right arrow dahs) that
behaves like an electronic keyer: each press sends one correctly timed
element at the encoder speed, repeats while held, holding both alternates,
and a tap during an element is remembered. The tone reaches the speakers
and, with Feed the decoder on, the decoder itself, so the page doubles as a
practice tool; a Sent line reads the operator's keying back even when the
decoder is not listening. The timing state machine is shared logic
(`morse/keyer.py`, `web/js/keyer.js`), tested in both languages.

Candidate enhancements noted at the same time, not built:

- A practice mode: show a word or call sign, key it, score against the
  decoder's reading and measure speed and rhythm errors. Built 2026-09-22
  (v0.1.5, section F).
- Farnsworth timing for the encoder (characters at one speed, gaps at a
  slower one), the standard way to learn by ear. Built 2026-09-22 (v0.1.6).
- Configurable key bindings and a sidetone pitch separate from the beeper
  frequency. Built 2026-09-22 (v0.1.7).
- Export of the decoded log with timestamps. Built 2026-09-22 (v0.1.8).
- Keyer weighting and mode variants (iambic A and B, bug mode). Built
  2026-09-22 (v0.1.9).

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

## 10. Web UI on GitHub Pages (added 2026-09-07)

GitHub Pages serves static files, so the online version runs entirely in the
browser: the page captures the microphone with `getUserMedia`, computes tone
power per 10 ms block in an `AudioWorklet`, and runs JavaScript ports of the
tone detector and decoder. The Python package stays the reference
implementation, the command-line tool and the test bed; the two decoders are
kept in lock-step by shared test vectors.

| Piece | Choice |
|---|---|
| Location | `web/` in this repository, plain HTML, CSS and ES modules, no build step. `web/index.html` starts from the approved mock. |
| Capture | `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all `false`; device picker from `enumerateDevices`. Starts on a click, as browsers require. |
| Per-block DSP | `AudioWorkletProcessor` accumulates render quanta into blocks of `round(fs/100)` samples and posts rms and Goertzel power at `f0`. Spectrum from an `AnalyserNode` (2048-point), display waveform from a `BiquadFilterNode` band-pass. |
| Detector and decoder | Line-by-line ports of `morse/tone_detector.py` and `morse/decoder.py` in `web/js/`. |
| Parity | `tools/export_vectors.py` writes `web/test/vectors.json` from the Python decoder (run sequences and expected text, dit and offset estimates). Node tests (`node --test web/test`) replay them and also decode both WAV fixtures. |
| Encoder and player | `web/js/table.js` and `web/js/player.js` (Web Audio oscillator with 3 ms ramps). |
| Deployment | `.github/workflows/pages.yml`: on push to `main`, run the Node tests, upload `web/`, deploy with `actions/deploy-pages`. Pages source set to GitHub Actions. URL: `https://jetzhu.github.io/MorseCodeAudioCoder/`. |

**Finding that shapes this (2026-09-07, refined 2026-09-22).** Chromium
browsers (Edge, Chrome) open the microphone in Windows raw mode when a page
disables echo cancellation, noise suppression and gain control, so they
bypass the enhancement processors and receive the beeper cleanly. Firefox
uses the shared, processed path. Measured on that path on 2026-09-22 with a
4 WPM SOS played from the laptop's own speakers: the 300 and 900 ms tones
survived only as 40 to 60 ms fragments at each onset (noise suppression
learns a steady tone in about 50 ms; echo cancellation removes the machine's
own output), decoding as "E E E". This is why slow beeps read as T and why
Play was never heard in Firefox, while Edge worked. The web app therefore
offers "Feed the decoder", which mixes the encoder's tone straight into the
decoder's input, and the page recommends Edge or Chrome. The 2026-09-07
measurement follows.

A loopback SOS at 2491 Hz through that path (device "Microphone Array", MME)
came out chopped: dahs mostly suppressed, marks fragmented into 10 to 30 ms
pieces, despite 42 dB of tone above the floor. The same test through the raw
WDM-KS endpoint gave clean 120 and 320 ms marks with 64 dB of headroom. The
registry shows seven audio-enhancement processors active on the array with
"disable all enhancements" unset. Mitigations, in order:

1. Use Edge or Chrome, which open the microphone in raw mode. (Since
   2026-09-22 the page also offers Mic: Browser default, which leaves the
   processing constraints to the browser, as an alternative path to try.)
   With a raw microphone and Feed on, the speaker echo used to add to the
   feed 30 to 150 ms late and fill the gaps; since v0.1.10 the microphone is
   turned down in the decoder's input while the app sounds (MicGate).
2. In Windows Settings, Sound, Input, Microphone Array: turn Audio
   enhancements off. Re-run `tools/probe/probe_loopback.py 1` to confirm the
   processed path then passes the tone cleanly.
3. Use a USB or headset microphone in the browser.
4. Fall back to the Python app, which reads the raw endpoint.

The web page detects the symptom itself: when the debounce is removing many
sub-30 ms fragments per second, it shows a hint pointing at the setting.

### Two solutions from one design (decided 2026-09-07)

1. **JavaScript solution**: the web page itself decodes, as described above.
2. **Python solution**: the desktop app with the pyqtgraph UI from section 5
   and the encoder strip, downloadable from the same web page and runnable
   offline. It reads the raw microphone endpoint, so it is also the answer on
   machines where the browser only gets a processed signal.

**Desktop distribution**

| Piece | Choice |
|---|---|
| Packaging metadata | `pyproject.toml` with a console script `morse-console = morse.app:main`; `pip install .` works for anyone with Python. |
| Self-contained build | PyInstaller one-folder build from `packaging/morse-console.spec`, console mode so the CLI flags keep working; zipped as `morse-console-windows-x64.zip`. Expected size 120 to 180 MB because of Qt. |
| Release automation | `.github/workflows/release.yml`: on a tag `v*`, build on `windows-latest`, run the tests, zip, and attach the zip plus a source zip to a GitHub Release. |
| Download from the page | A Download panel on `web/index.html` fetches `releases/latest` from the GitHub API and lists the assets with sizes; if the fetch fails it links to the releases page. Includes one-line instructions: unzip, run `morse-console.exe`, no internet needed. |
| Other platforms | Source zip plus `pip install .`; native builds for macOS and Linux can be added to the release workflow later. |

**Milestones added**

| # | Deliverable | Done when |
|---|---|---|
| M6 | Web app in `web/`, parity tests, Pages deployment | Node tests pass in CI; the page decodes the loopback fixture in a browser test harness and decodes live beeps on a machine with enhancements off |
| M7 | Desktop UI `morse/ui.py` and `morse/player.py`, `pyproject.toml`, PyInstaller spec | Offscreen smoke test renders the window and decodes the loopback fixture through the UI's own pipeline; a local PyInstaller build starts and lists devices |
| M8 | Release workflow and Download panel | Tag `v0.1.0` produces a Release with the Windows zip; the Pages site lists it |

## 11. Setup

```powershell
cd C:\Users\jetzhu\Projects\OOPolaris\MorseCodeAudioDecoder
python -m venv .venv
.\.venv\Scripts\python -m pip install sounddevice pyqtgraph PySide6 numpy scipy pytest
.\.venv\Scripts\python -m morse.app --list-devices
.\.venv\Scripts\python -m morse.app --device "Microphone Array 1" --freq 1000
```

## Search engines (2026-09-22)

Tiers 1 and 2 of the discoverability plan are on the page: canonical URL,
title and description with the words people search for, Open Graph and
Twitter cards with a preview image, JSON-LD (web app, desktop app, FAQ), an
About and FAQ section in plain prose, robots.txt, sitemap.xml, noindex on the
design mock, and repository homepage and topics. Still needing the owner:
verifying the site in Google Search Console and Bing Webmaster Tools (paste
the tags into the marked place in `web/index.html`, then submit the sitemap),
and any off-site links (a mention in ham-radio or maker forums) that tier 3
would bring.

Search terms (2026-09-23): the title, description, eyebrow, an About lead
paragraph, three FAQ entries, JSON-LD `alternateName`/`keywords`, the README
title and the repository description and topics now carry the words people
type: Morse code simulator, translator, trainer, practice, chart, CW decoder,
learn Morse code. The chart is pre-rendered in the HTML (app.js reuses the
cells) so crawlers see it without running scripts. Off-site steps for the
owner, in order of effect: verify the site in Google Search Console and Bing
Webmaster Tools and submit the sitemap; get a handful of real links (ham-radio
and maker forums, a Show HN, an AlternativeTo listing, awesome-lists, club
newsletters); keep the release cadence visible on GitHub (stars and forks are
themselves signals); pin the repository on the GitHub profile.

## Reference chart (2026-09-23)

Section G, a Morse code chart built from the shared table in both apps
(v0.1.11): collapsed by default, lit by the letter in progress, click to hear.
Chosen over a separate chart page (no live highlight) and over extending the
Practice strip's Show the code (covers only the current target).

## Light: a hearing-impaired user, and two phones talking by light (2026-09-25)

Decided with the user: both Android and iPhone matter; the main scenario is
two phones each sending and receiving by light; browser first. Steps:
1. Channels (Listen with / Send with, any combination, all on by default),
   the signal lamp, full-screen light behind a photosensitivity notice,
   vibration on Android, and a handset layout on the same URL. Built
   2026-09-25 (v0.1.13).
2. Camera as a listen source: preview, tap the light to track, brightness
   fed to the existing detector and decoder at the frame rate (about 30 Hz,
   which caps light decoding near 10 to 12 WPM); synthetic-frame tests.
3. Torch on Android Chrome behind capability detection, capped at 8 WPM.
4. A link check between two phones (one sends a known word, the other
   grades it, reusing the practice scoring).
Desktop: the lamp and full-screen window in step 1; a USB-serial DTR/RTS
key line would be the way to drive an external lamp or a rig later.
