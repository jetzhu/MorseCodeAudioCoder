import sys, numpy as np
from scipy.io import wavfile
from scipy.signal import stft
path = sys.argv[1]; min_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 80
fs, x = wavfile.read(path); x = x.astype(np.float32) / 32768
print(f"{path}: {len(x)/fs:.1f}s, peak {20*np.log10(np.abs(x).max()+1e-9):.1f} dBFS, rms {20*np.log10(np.sqrt(np.mean(x**2))+1e-9):.1f} dBFS")
nper = int(fs * 0.025)
f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann")
P = np.abs(Z) ** 2; band = (f >= 250) & (f <= 6000); fb, Pb = f[band], P[band]
pi = Pb.argmax(axis=0); pk = Pb.max(axis=0); med = np.median(Pb, axis=0)
prom = 10 * np.log10(pk / (med + 1e-15) + 1e-15); pf = fb[pi]; pdb = 10 * np.log10(pk + 1e-15)
tonal = prom > 15
runs, i = [], 0
while i < len(t):
    if tonal[i]:
        j = i
        while j + 1 < len(t) and tonal[j + 1] and abs(pf[j + 1] - np.median(pf[i:j+1])) <= 80: j += 1
        dur = (j - i + 1) * 0.0125
        if dur * 1000 >= min_ms: runs.append((t[i], dur, float(np.median(pf[i:j+1])), pdb[i:j+1].max(), prom[i:j+1].mean()))
        i = j + 1
    else: i += 1
print(f"\nsustained stable-frequency tone runs >= {min_ms:.0f} ms: {len(runs)}")
for s0, d, fr, lv, pr in runs:
    print(f"  t={s0:6.2f}s dur={d*1000:5.0f}ms  f={fr:7.1f} Hz  level={lv:6.1f} dB  prom={pr:4.1f} dB")
if runs:
    fr = np.array([r[2] for r in runs]); hist, edges = np.histogram(fr, bins=np.arange(250, 6001, 100))
    print("\nfrequency histogram of runs (100 Hz bins, non-empty):")
    for h, e in zip(hist, edges[:-1]):
        if h: print(f"  {e:5.0f}-{e+100:5.0f} Hz : {'#'*h} {h}")
    # gaps between consecutive runs at the dominant frequency band
    dom = edges[hist.argmax()]
    sel = [r for r in runs if dom <= r[2] < dom + 100]
    if len(sel) > 1:
        print(f"\nruns in dominant band {dom:.0f}-{dom+100:.0f} Hz: durations ms {[int(r[1]*1000) for r in sel]}")
        print(f"gaps between them ms: {[int((sel[k+1][0]-(sel[k][0]+sel[k][1]))*1000) for k in range(len(sel)-1)]}")
