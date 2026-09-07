import sys, time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

FS = 48000
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "probe.wav"

dev = sd.query_devices(kind="input")
print(f"Recording {DUR:.0f}s from: {dev['name']} @ {FS} Hz", flush=True)
audio = sd.rec(int(DUR * FS), samplerate=FS, channels=1, dtype="float32")
sd.wait()
x = audio[:, 0]
wavfile.write(OUT, FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))
print(f"saved {OUT}, peak={np.abs(x).max():.3f}", flush=True)

# --- level timeline, 100 ms bins ---
hop = FS // 10
n = len(x) // hop
rms = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop] ** 2)) for i in range(n)])
db = 20 * np.log10(rms + 1e-9)
floor = np.percentile(db, 20)
print(f"\nnoise floor ~{floor:.1f} dBFS; timeline (each char = 100 ms, '#' = >10 dB above floor, '+' = >5 dB):")
line = "".join("#" if d > floor + 10 else "+" if d > floor + 5 else "." for d in db)
for s in range(0, n, 50):
    print(f"{s/10:5.1f}s  {line[s:s+50]}")

# --- spectrum of loud segments vs quiet segments ---
loud = np.where(db > floor + 10)[0]
if len(loud) == 0:
    print("\nNo loud segment detected (nothing >10 dB above floor).")
    sys.exit(0)
seg = np.concatenate([x[i*hop:(i+1)*hop] for i in loud])
win = np.hanning(len(seg))
spec = np.abs(np.fft.rfft(seg * win))
freqs = np.fft.rfftfreq(len(seg), 1 / FS)
mask = freqs > 100
spec_m, freqs_m = spec[mask], freqs[mask]
order = np.argsort(spec_m)[::-1]
print(f"\nLoud segments total {len(seg)/FS:.2f}s. Dominant frequency: {freqs_m[order[0]]:.1f} Hz")
print("Top spectral peaks (Hz, relative dB):")
picked = []
for i in order:
    f = freqs_m[i]
    if all(abs(f - p) > 30 for p in picked):
        picked.append(f)
        print(f"  {f:8.1f} Hz  {20*np.log10(spec_m[i]/spec_m[order[0]]):6.1f} dB")
    if len(picked) >= 8:
        break

# --- on/off runs at the dominant frequency using Goertzel per 10 ms ---
f0 = freqs_m[order[0]]
blk = FS // 100
nb = len(x) // blk
k = 2 * np.cos(2 * np.pi * f0 / FS)
pw = np.empty(nb)
for b in range(nb):
    s1 = s2 = 0.0
    for v in x[b*blk:(b+1)*blk]:
        s0 = v + k * s1 - s2
        s2, s1 = s1, s0
    pw[b] = s1*s1 + s2*s2 - k*s1*s2
pdb = 10 * np.log10(pw + 1e-12)
thr = (np.percentile(pdb, 20) + np.percentile(pdb, 99)) / 2
on = pdb > thr
runs, cur, cnt = [], on[0], 0
for v in on:
    if v == cur: cnt += 1
    else:
        runs.append((cur, cnt)); cur, cnt = v, 1
runs.append((cur, cnt))
print(f"\nTone on/off runs at {f0:.0f} Hz (10 ms blocks, threshold {thr:.1f} dB):")
print("  " + " ".join(f"{'ON' if r[0] else 'off'}:{r[1]*10}ms" for r in runs if r[1] < 500))
