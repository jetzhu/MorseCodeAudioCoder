# Beeper Morse Console 0.2.0: sound and light

0.2.0 marks the point where the console works with both sound and light.
Everything from the 0.1 series is included.

## Sound (both apps)

- Live decoder for a PC beeper or any CW tone, 2 to 40 WPM, with an adaptive
  speed estimate that re-locks within a few marks.
- Noise rejection: a block counts as tone only if it holds a fair share of
  the energy and stands 8 dB above the bands 300 and 600 Hz either side, so
  clicks and taps are ignored even on a phone's band-limited microphone.
- Encoder with keying guide, Farnsworth spacing, and Feed the decoder, with
  the microphone turned down while the app itself sounds.
- Hand key: straight key, iambic A/B keyer, bug; rebindable keys, sidetone,
  dah ratio and weight.
- Practice drills with grading, a Morse reference chart, and decoded-log
  export.

## Light (web page)

- Listen with Microphone and/or Camera; send with Audio, Screen light
  (lamp or full screen), Vibration and the Torch (Chrome on Android), in any
  combination.
- Camera decoding: tap the light in the preview; exposure is locked after the
  camera settles, and marks are timed from each frame's capture time.
- A link check between two devices ("VVV PARIS 73"), graded with advice.
- A handset layout for phones with every section, and Show Audio / Show Light
  with S / M / L heights in both layouts.

## Desktop app

Windows build: unzip `morse-console-windows-x64.zip` and run
`morse-console.exe`; no Python or internet needed. It has the sound features,
the signal lamp and full-screen light. Camera, torch and vibration are
browser features.
