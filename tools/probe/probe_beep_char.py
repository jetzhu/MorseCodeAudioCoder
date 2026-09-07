"""Characterize a single long beep: exact frequency, drift, harmonics, level, edge times."""
import sys, numpy as np
from scipy.io import wavfile
path, f_hint = sys.argv[1], float(sys.argv[2])
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None
fs, x = wavfile.read(path); x = x.astype(np.float32) / 32768
x = x[int(t0 * fs): int(t1 * fs) if t1 else None]
BLK = fs // 100
c = np.exp(-1j * 2 * np.pi * f_hint / fs * np.arange(BLK)); nb = len(x) // BLK
pw = np.array([abs(np.dot(x[i*BLK:(i+1)*BLK], c))**2 for i in range(nb)]); pdb = 10 * np.log10(pw + 1e-12)
floor, peak = np.percentile(pdb, 20), np.percentile(pdb, 99); thr = floor + 0.5 * (peak - floor)
on = pdb > thr; idx = np.where(on)[0]
if len(idx) == 0: print("no tone found"); sys.exit(0)
s, e = idx[0], idx[-1]
print(f"tone ON from {t0 + s/100:.2f}s to {t0 + (e+1)/100:.2f}s  ({(e - s + 1)*10} ms), Goertzel SNR {peak - floor:.1f} dB, gaps inside: {int((~on[s:e+1]).sum())*10} ms total")
seg = x[(s + 20) * BLK:(e - 20) * BLK]  # steady part
print(f"steady-part level: rms {20*np.log10(np.sqrt(np.mean(seg**2))+1e-9):.1f} dBFS, peak {20*np.log10(np.abs(seg).max()+1e-9):.1f} dBFS")
# exact frequency: zero-padded FFT + parabolic interpolation
N = 1 << (int(np.ceil(np.log2(len(seg)))) + 2)
S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)); f = np.fft.rfftfreq(N, 1 / fs)
b = (f > f_hint * 0.9) & (f < f_hint * 1.1); k = np.argmax(np.where(b, S, 0))
a, bb, cc = np.log(S[k-1] + 1e-12), np.log(S[k] + 1e-12), np.log(S[k+1] + 1e-12); p = 0.5 * (a - cc) / (a - 2*bb + cc)
f0 = f[k] + p * (f[1] - f[0])
print(f"exact fundamental: {f0:.2f} Hz")
# drift: peak per 250 ms
hop = fs // 4; drift = []
for i in range(0, len(seg) - hop, hop):
    w = seg[i:i+hop] * np.hanning(hop); Sw = np.abs(np.fft.rfft(w, 1 << 18)); fw = np.fft.rfftfreq(1 << 18, 1 / fs)
    bw = (fw > f_hint * 0.9) & (fw < f_hint * 1.1); drift.append(fw[bw][Sw[bw].argmax()])
print(f"frequency per 250 ms: min {min(drift):.1f}, max {max(drift):.1f} Hz (drift {max(drift)-min(drift):.1f} Hz)")
# harmonics relative to fundamental
def level_at(fq):
    b = (f > fq - 30) & (f < fq + 30); return 20 * np.log10(S[b].max() / S[k] + 1e-12) if b.any() else float("nan")
print("harmonics (dB rel fundamental): " + ", ".join(f"{n}f={n*f0:.0f}Hz {level_at(n*f0):.0f}dB" for n in range(2, 8) if n * f0 < fs / 2))
sub = [(fq, level_at(fq)) for fq in (f0/2, f0/3)]
print("subharmonics: " + ", ".join(f"{fq:.0f}Hz {lv:.0f}dB" for fq, lv in sub))
# spectral peaks anywhere, to see what else is there
bb2 = (f > 150) & (f < fs/2 - 100); order = np.argsort(np.where(bb2, S, 0))[::-1]; picked = []
for i in order:
    if all(abs(f[i] - q) > 80 for q in picked): picked.append(f[i])
    if len(picked) >= 6: break
print("top spectral peaks: " + ", ".join(f"{q:.0f}Hz {20*np.log10(S[np.argmin(abs(f-q))]/S[k]):.0f}dB" for q in picked))
# edges: 10-90 % rise/fall of the Goertzel envelope (linear power) around start and end
lin = pw / pw[s+20:e-20].mean()
def edge(i0, i1):
    seg_ = lin[i0:i1]; lo_, hi_ = np.where(seg_ > 0.1)[0], np.where(seg_ > 0.9)[0]
    return (hi_[0] - lo_[0]) * 10 if len(lo_) and len(hi_) else float("nan")
rise = edge(max(0, s - 10), s + 10)
fall_seg = lin[e - 10:e + 10][::-1]; lo_, hi_ = np.where(fall_seg > 0.1)[0], np.where(fall_seg > 0.9)[0]
fall = (hi_[0] - lo_[0]) * 10 if len(lo_) and len(hi_) else float("nan")
print(f"attack (10-90%) {rise:.0f} ms, release (90-10%) {fall:.0f} ms")
print("envelope around start (10 ms blocks, dB):", " ".join(f"{v:.0f}" for v in pdb[max(0, s-5):s+8]))
print("envelope around end   (10 ms blocks, dB):", " ".join(f"{v:.0f}" for v in pdb[e-7:e+6]))
