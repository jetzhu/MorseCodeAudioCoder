import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

fs, x = wavfile.read(sys.argv[1])
x = x.astype(np.float32) / 32768
print(f"{sys.argv[1]}: {len(x)/fs:.1f}s, peak {20*np.log10(np.abs(x).max()+1e-9):.1f} dBFS, rms {20*np.log10(np.sqrt(np.mean(x**2))+1e-9):.1f} dBFS")

nper = int(fs * 0.05)  # 50 ms frames
f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann")
P = np.abs(Z) ** 2
band = (f >= 300) & (f <= 4000)
fb, Pb = f[band], P[band]
# tonal prominence per frame: peak power vs median power in band
peak_idx = Pb.argmax(axis=0)
peak_pow = Pb.max(axis=0)
med_pow = np.median(Pb, axis=0)
prom_db = 10 * np.log10(peak_pow / (med_pow + 1e-15))
peak_f = fb[peak_idx]
peak_db = 10 * np.log10(peak_pow + 1e-15)

print("\nFrames with a prominent narrowband peak (>18 dB above in-band median):")
tonal = prom_db > 18
for i in np.where(tonal)[0]:
    print(f"  t={t[i]:6.2f}s  f={peak_f[i]:7.1f} Hz  level={peak_db[i]:6.1f} dB  prominence={prom_db[i]:5.1f} dB")
print(f"\n{tonal.sum()} tonal frames of {len(t)}")
if tonal.sum():
    hist, edges = np.histogram(peak_f[tonal], bins=np.arange(300, 4001, 50))
    top = np.argsort(hist)[::-1][:5]
    print("Most common tonal-frame frequencies (50 Hz bins):")
    for i in top:
        if hist[i]:
            print(f"  {edges[i]:5.0f}-{edges[i+1]:5.0f} Hz : {hist[i]} frames")

# overall long-term average spectrum in band, top peaks
avg = Pb.mean(axis=1)
order = np.argsort(avg)[::-1]
print("\nLong-term average spectrum, top peaks:")
picked = []
for i in order:
    if all(abs(fb[i] - p) > 40 for p in picked):
        picked.append(fb[i]); print(f"  {fb[i]:7.1f} Hz  {10*np.log10(avg[i]/avg[order[0]]):6.1f} dB")
    if len(picked) >= 6: break
