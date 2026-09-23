# Morse Code Audio Decoder

Listens to the laptop microphone, isolates the tone of a PC beeper on another
machine (2491 Hz, a classic BIOS-style beep), shows what it hears, and decodes
the on/off pattern as Morse code. Pure Python: numpy/scipy for the DSP,
sounddevice for capture, pyqtgraph + PySide6 for the live window.

Signal chain: 48 kHz mono in 10 ms blocks -> Goertzel tone power at `f0` ->
adaptive ON/OFF detector (version 2: noise and signal levels tracked as linear
means, hysteresis sized to the measured signal-to-noise ratio, 20 ms debounce,
5 s stuck-ON timeout) -> timing decoder that estimates the dit length and
undoes the room-reverb offset -> text. The design and the measurements behind
every constant are in [`docs/PLAN.md`](docs/PLAN.md); the module contract is
[`docs/INTERFACES.md`](docs/INTERFACES.md). Two recordings under
`tests/fixtures/` pin the behaviour: `loopback_sos_1khz_15wpm.wav` decodes to
`SOS`, and `beeper_long_2491hz_1m.wav` (the real beeper) yields exactly four
ON runs, all with the detector's default settings.

Three ways to use it: download the self-contained Windows app, `pip install`
the package on any platform with Python, or open the browser version. The
desktop app is the reference implementation and the one to use when the
browser only receives a processed microphone signal (Firefox on Windows).

## Send Morse by hand

Both apps have a Key strip. **Single key**: hold Space or the big button and
the tone sounds while held, like a straight key. **Two keys**: the left arrow
sends dits and the right arrow dahs, timed at the encoder speed and repeating
while held, like an electronic keyer; holding both alternates and a tap during
an element is remembered. The Sent line reads back what you keyed; with Feed
the decoder on and the decoder listening, the decoder reads it too.

## Download (Windows, no Python needed)

Get `morse-console-windows-x64.zip` from the
[latest release](https://github.com/jetzhu/MorseCodeAudioCoder/releases/latest),
unzip it anywhere and run `morse-console.exe` from the `morse-console` folder.
Python, Qt and PortAudio are bundled: nothing to install, no internet
connection needed. The executable accepts the same options as
`python -m morse.app` below, so `morse-console.exe --list-devices` and
`morse-console.exe --wav recording.wav --freq 1000` work from a terminal, and
starting it without arguments opens the live window. The first launch of an
unsigned download may trigger a SmartScreen prompt ("More info", "Run anyway").

All releases: <https://github.com/jetzhu/MorseCodeAudioCoder/releases>.

## Install from source

Any platform with Python 3.13 or newer, from a checkout or from the source zip
attached to a release:

```powershell
pip install .              # the morse package plus the morse-console command
pip install .[dev]         # adds pytest
morse-console --list-devices
```

The command is declared in `pyproject.toml` (`morse-console = morse.app:main`)
and takes the same options as `python -m morse.app`. On Linux install the
PortAudio library first (Debian/Ubuntu: `sudo apt install libportaudio2`); the
sounddevice wheel bundles it only on Windows and macOS.

## Web version

The same decoder runs in the browser at
<https://jetzhu.github.io/MorseCodeAudioCoder/> (current Chrome or Edge; the
source is in `web/`, deployed by `.github/workflows/pages.yml`). Chromium
browsers open the microphone in Windows raw mode when a page turns processing
off, so the beeper arrives clean. Firefox gets the shared, processed path,
which removes steady tones after about 50 ms and removes the machine's own
output entirely (`docs/PLAN.md`, section 10); use Edge or Chrome there, or the
encoder's "Feed the decoder" option, which mixes Play straight into the
decoder. When the
page reports a processed signal, turn off Audio enhancements for the
microphone in Windows Sound settings, or use the desktop app, which reads the
raw endpoint. The page also lists the desktop download.

## Setup (development)

```powershell
cd C:\Users\jetzhu\Projects\OOPolaris\MorseCodeAudioDecoder
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
```

Always run the project with the venv interpreter, `.\.venv\Scripts\python`.
Every command below also works as `morse-console ...` after `pip install .`
and as `morse-console.exe ...` from the Windows download.

## Running

```powershell
# 1. Which microphones does PortAudio see?  (* marks the one used by default)
.\.venv\Scripts\python -m morse.app --list-devices

# 2. Decode a recording offline; --verbose lists every ON/off run first
.\.venv\Scripts\python -m morse.app --wav tests\fixtures\loopback_sos_1khz_15wpm.wav --freq 1000
.\.venv\Scripts\python -m morse.app --wav tests\fixtures\beeper_long_2491hz_1m.wav --verbose

# 3. Listen live, text only (Ctrl-C to stop)
.\.venv\Scripts\python -m morse.app --no-ui --device "Microphone Array 1"

# 4. Listen live with the Qt window
.\.venv\Scripts\python -m morse.app --device "Microphone Array 1" --freq 2491
```

Common options: `--freq HZ` (default 2491; must be below half the sample
rate), `--wpm N` (seed the speed instead of adapting), `--device SPEC` (index
from `--list-devices` or a name fragment; default is the raw WDM-KS
`Microphone Array 1` endpoint when present).

Exit codes: 0 on success, 2 when the `--wav` file cannot be opened (missing,
a directory, no permission; one line on stderr), 1 for any other error.
WAV files may be any integer or float format, mono or multi-channel (channel
0 is used), at any sample rate.

To send test messages from the beeper machine copy `tools/beep_sender.py`
there (standard library only) and run
`python beep_sender.py "HELLO WORLD" --wpm 8 --freq 2491`.

## Layout

| Path | What |
|---|---|
| `morse/` | the package: `dsp`, `tone_detector`, `decoder`, `table`, `pipeline`, `audio_input`, `app`, `ui` |
| `tests/` | pytest suite; `tests/fixtures/` holds the recorded WAVs (`loopback_sos_1khz_15wpm.wav` decodes to `SOS`; `beeper_long_2491hz_1m.wav` is the real beeper, four tones) |
| `tools/beep_sender.py` | keys text as Morse through the PC speaker of the machine it runs on |
| `tools/probe/` | the diagnostic scripts from the first measurement session (device comparison, recording, analysis, loopback) |
| `docs/PLAN.md` | plan, algorithm and measured constants |
| `docs/INTERFACES.md` | the module contract |
| `web/` | the browser app (`index.html`, `js/`, `css/`), its Node tests, and `mock.html`, the approved design |
| `pyproject.toml` | packaging metadata: `pip install .` installs the package and the `morse-console` command |
| `packaging/morse-console.spec` | PyInstaller spec for the self-contained Windows build (`dist/morse-console/`) |
| `.github/workflows/` | `ci.yml` (tests on Windows and Ubuntu), `pages.yml` (web deploy), `release.yml` (Windows build and GitHub Release on a `v*` tag) |

## Making a release

Set `__version__` in `morse/__init__.py`, commit, then tag and push:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/release.yml` refuses a tag that does not equal
`v` + `__version__`, runs the test suite on `windows-latest`, builds the app
with PyInstaller, smoke-tests the executable (`--list-devices` and decoding
the loopback fixture to `SOS`), and publishes a Release with
`morse-console-windows-x64.zip` and a source zip, with generated notes. To
build the same folder locally:

```powershell
.\.venv\Scripts\python -m pip install pyinstaller
.\.venv\Scripts\python -m PyInstaller --noconfirm --clean packaging\morse-console.spec
dist\morse-console\morse-console.exe --wav tests\fixtures\loopback_sos_1khz_15wpm.wav --freq 1000
```
