import sys, numpy as np, sounddevice as sd
from scipy.io import wavfile
FS, F0, WPM, IN_DEV, OUT_DEV = 48000, 2491.0, 12, int(sys.argv[1]) if len(sys.argv) > 1 else 18, 3
T = 1.2 / WPM  # dit seconds
CODE = {"S": "...", "O": "---", "E": ".", "T": "-"}
def keyed(text):
    env = [np.zeros(int(FS * 0.5))]
    for ch in text:
        for s in CODE[ch]:
            env.append(np.ones(int(FS * (T if s == "." else 3 * T)))); env.append(np.zeros(int(FS * T)))
        env.append(np.zeros(int(FS * 2 * T)))
    env.append(np.zeros(int(FS * 0.5)))
    e = np.concatenate(env)
    return e, (0.3 * e * np.sin(2 * np.pi * F0 * np.arange(len(e)) / FS)).astype(np.float32)
env, tone = keyed("SOS")
chunks = []
def cb(indata, frames, t, status): chunks.append(indata[:, 0].copy())
with sd.InputStream(device=IN_DEV, samplerate=FS, channels=1, dtype="float32", blocksize=480, callback=cb):
    sd.play(tone, samplerate=FS, device=OUT_DEV); sd.wait(); sd.sleep(300)
x = np.concatenate(chunks); wavfile.write(f"loopback_dev{IN_DEV}_2491.wav", FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))
print(f"played {len(tone)/FS:.1f}s SOS @ {F0:.0f} Hz via device {OUT_DEV}; recorded {len(x)/FS:.1f}s on device {IN_DEV}: peak {20*np.log10(np.abs(x).max()+1e-9):.1f} dBFS")
# Goertzel per 10 ms at F0
blk = 480; nb = len(x) // blk
w = 2 * np.pi * F0 / FS; c = np.exp(-1j * w * np.arange(blk))
pw = np.array([abs(np.dot(x[i*blk:(i+1)*blk], c))**2 for i in range(nb)])
pdb = 10 * np.log10(pw + 1e-12)
floor, peak = np.percentile(pdb, 30), np.percentile(pdb, 98)
print(f"Goertzel@{F0:.0f}: floor {floor:.1f} dB, peak {peak:.1f} dB, SNR {peak-floor:.1f} dB")
thr = floor + 0.5 * (peak - floor); on = pdb > thr
runs, cur, cnt = [], on[0], 0
for v in on:
    if v == cur: cnt += 1
    else: runs.append((cur, cnt)); cur, cnt = v, 1
runs.append((cur, cnt))
print("runs:", " ".join(f"{'ON' if r[0] else 'off'}:{r[1]*10}" for r in runs))
# simple decode
marks = [r for r in runs if r[0]]
if marks:
    dit = min(r[1] for r in marks)
    out, buf = "", ""
    INV = {v: k for k, v in CODE.items()}
    for state, n in runs[1:-1]:
        if state: buf += "." if n < 2 * dit else "-"
        elif n >= 2 * dit: out += INV.get(buf, "?"); buf = ""
    if buf: out += INV.get(buf, "?")
    print(f"dit estimate {dit*10} ms (true {T*1000:.0f} ms) -> decoded: {out}")
# where is the true peak frequency in the loud part?
seg = x[np.abs(x) > 0.3 * np.abs(x).max()] if False else x
spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))**2; f = np.fft.rfftfreq(len(x), 1/FS); b = (f > 300) & (f < 5000)
print(f"spectral peak in 300-5000 Hz: {f[b][spec[b].argmax()]:.1f} Hz")
