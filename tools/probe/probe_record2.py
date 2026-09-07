import sys, threading, numpy as np, sounddevice as sd
from scipy.io import wavfile
from scipy.signal import stft
FS, DUR = 48000, float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
DEVS = {"raw_wdmks": 18, "default_mme": 1}
bufs = {k: [] for k in DEVS}
streams = {}
for name, dev in DEVS.items():
    def cb(indata, frames, t, status, _n=name):
        bufs[_n].append(indata[:, 0].copy())
    try:
        s = sd.InputStream(device=dev, samplerate=FS, channels=1, dtype="float32", blocksize=480, callback=cb)
        s.start(); streams[name] = s
    except Exception as e:
        print(f"{name} (dev {dev}) failed to open: {e}", flush=True)
print(f"recording {DUR:.0f}s on: {list(streams)}", flush=True)
sd.sleep(int(DUR * 1000))
for s in streams.values(): s.stop(); s.close()

def analyze(name, x):
    wavfile.write(f"probe2_{name}.wav", FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print(f"\n=== {name}: {len(x)/FS:.1f}s, peak {20*np.log10(np.abs(x).max()+1e-9):.1f} dBFS, rms {20*np.log10(np.sqrt(np.mean(x**2))+1e-9):.1f} dBFS")
    nper = int(FS * 0.025)
    f, t, Z = stft(x, fs=FS, nperseg=nper, noverlap=nper // 2, window="hann")
    P = np.abs(Z) ** 2
    band = (f >= 300) & (f <= 5000); fb, Pb = f[band], P[band]
    pi = Pb.argmax(axis=0); pk = Pb.max(axis=0); med = np.median(Pb, axis=0)
    prom = 10 * np.log10(pk / (med + 1e-15) + 1e-15); pf = fb[pi]; pdb = 10 * np.log10(pk + 1e-15)
    tonal = prom > 15
    # sustained runs: consecutive tonal frames with freq within 60 Hz of run start, >= 4 frames (50 ms)
    runs, i = [], 0
    while i < len(t):
        if tonal[i]:
            j = i
            while j + 1 < len(t) and tonal[j + 1] and abs(pf[j + 1] - pf[i]) <= 60: j += 1
            if j - i + 1 >= 4:
                runs.append((t[i], t[j] - t[i] + 0.0125, np.median(pf[i:j+1]), pdb[i:j+1].max()))
            i = j + 1
        else: i += 1
    print(f"sustained tone runs (>=50 ms, stable freq): {len(runs)}")
    for s0, d, fr, lv in runs:
        print(f"  t={s0:6.2f}s dur={d*1000:5.0f}ms  f={fr:7.1f} Hz  level={lv:6.1f} dB")
    if runs:
        fr = np.array([r[2] for r in runs]); d = np.array([r[1] for r in runs])
        print(f"run-frequency median {np.median(fr):.0f} Hz (spread {fr.min():.0f}-{fr.max():.0f}); durations ms: {sorted((d*1000).astype(int).tolist())}")

for name in streams:
    analyze(name, np.concatenate(bufs[name]))
