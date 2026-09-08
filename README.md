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

## Setup

```powershell
cd C:\Users\jetzhu\Projects\OOPolaris\MorseCodeAudioDecoder
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
```

Always run the project with the venv interpreter, `.\.venv\Scripts\python`.

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
