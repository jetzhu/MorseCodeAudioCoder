"""Offline prototype of the decode pipeline: WAV -> Goertzel -> runs -> debounce -> offset-corrected Morse."""
import sys, numpy as np
from scipy.io import wavfile
path = sys.argv[1]; f0_hint = float(sys.argv[2]); t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else None
fs, x = wavfile.read(path); x = x.astype(np.float32) / 32768
x = x[int(t0 * fs): int(t1 * fs) if t1 else None]
BLK = fs // 100  # 10 ms
TABLE = {"A":".-","B":"-...","C":"-.-.","D":"-..","E":".","F":"..-.","G":"--.","H":"....","I":"..","J":".---","K":"-.-","L":".-..","M":"--","N":"-.","O":"---","P":".--.","Q":"--.-","R":".-.","S":"...","T":"-","U":"..-","V":"...-","W":".--","X":"-..-","Y":"-.--","Z":"--..","0":"-----","1":".----","2":"..---","3":"...--","4":"....-","5":".....","6":"-....","7":"--...","8":"---..","9":"----.",".":".-.-.-",",":"--..--","?":"..--..","/":"-..-.","=":"-...-"}
INV = {v: k for k, v in TABLE.items()}

# 1) refine f0: average spectrum of the loudest 20 % of 100 ms frames, peak within ±15 % of the hint
hop = fs // 10; n = len(x) // hop
rms = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2)) for i in range(n)])
loud = np.argsort(rms)[-max(3, n // 5):]
seg = np.concatenate([x[i*hop:(i+1)*hop] for i in sorted(loud)])
spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))**2; f = np.fft.rfftfreq(len(seg), 1 / fs)
b = (f > f0_hint * 0.85) & (f < f0_hint * 1.15); f0 = f[b][spec[b].argmax()]
bb = (f > 200) & (f < fs / 2 - 100); order = np.argsort(spec[bb])[::-1]; picked = []
for i in order:
    if all(abs(f[bb][i] - p[0]) > 60 for p in picked): picked.append((f[bb][i], 10 * np.log10(spec[bb][i] / spec[bb][order[0]])))
    if len(picked) >= 6: break
print(f"{path} [{t0:.1f}s..{t1 if t1 else len(x)/fs + t0:.1f}s]  refined f0 = {f0:.1f} Hz")
print("  spectral peaks of loud frames (Hz, dB rel):", ", ".join(f"{p[0]:.0f}Hz {p[1]:.0f}dB" for p in picked))

# 2) Goertzel power per 10 ms block, in dB
nb = len(x) // BLK; c = np.exp(-1j * 2 * np.pi * f0 / fs * np.arange(BLK))
pdb = 10 * np.log10(np.array([abs(np.dot(x[i*BLK:(i+1)*BLK], c))**2 for i in range(nb)]) + 1e-12)
floor, peak = np.percentile(pdb, 20), np.percentile(pdb, 99)
print(f"  Goertzel@{f0:.0f}: floor {floor:.1f} dB, peak {peak:.1f} dB, SNR {peak - floor:.1f} dB")

# 3) hysteresis threshold -> runs
hi, lo = floor + 0.6 * (peak - floor), floor + 0.4 * (peak - floor)
state = pdb[0] > hi; runs = [[bool(state), 0]]
for v in pdb:
    if state and v < lo: state = False; runs.append([False, 0])
    elif not state and v > hi: state = True; runs.append([True, 0])
    runs[-1][1] += 1
# 4) debounce: absorb runs < 2 blocks into neighbours
while True:
    for i in range(1, len(runs) - 1):
        if runs[i][1] < 2: runs[i-1][1] += runs[i][1] + runs[i+1][1]; del runs[i:i+2]; break
    else: break
inner = runs[1:-1] if len(runs) > 2 else []
print(f"  runs ({len(inner)} inner):", " ".join(f"{'ON' if r[0] else 'off'}:{r[1]*10}" for r in inner)[:1500])
marks = np.array([r[1] * 10.0 for r in inner if r[0]]); gaps = np.array([r[1] * 10.0 for r in inner if not r[0]])
if len(marks) < 2 or len(gaps) < 1:
    print("  not enough runs to decode"); sys.exit(0)

# 5) dit estimate with reverb-offset correction: shortest mark M = T + d, shortest gap G = T - d
M = np.percentile(marks, 10); G = np.percentile(np.minimum(gaps, 20 * M), 10)
T = (M + G) / 2; d = (M - G) / 2
if d < 0: T, d = M, 0.0   # gaps longer than marks: no reverb correction needed
print(f"  shortest mark {M:.0f} ms, shortest gap {G:.0f} ms -> dit T={T:.0f} ms ({1200/T:.1f} WPM), offset d={d:.0f} ms")
print(f"  mark durations ms: {sorted(marks.astype(int).tolist())}")
print(f"  gap durations ms:  {sorted(gaps.astype(int).tolist())}")

# 6) classify and decode
out, buf, sym = "", "", []
for st, n in inner:
    ms = n * 10 - d if st else n * 10 + d
    if st: buf += "." if ms < 2 * T else "-"
    else:
        if ms >= 2 * T and buf: out += INV.get(buf, "?"); sym.append(buf); buf = ""
        if ms >= 5 * T: out += " "
if buf: out += INV.get(buf, "?"); sym.append(buf)
print(f"  symbols: {' '.join(sym)}")
print(f"  DECODED: '{out.strip()}'")
