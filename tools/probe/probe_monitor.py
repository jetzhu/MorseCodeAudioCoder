import sys, time, numpy as np, sounddevice as sd
from scipy.io import wavfile
FS, DEV = 48000, int(sys.argv[2]) if len(sys.argv) > 2 else 18
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "probe3.wav"
chunks, ring = [], np.zeros(FS // 4, dtype=np.float32)   # 250 ms analysis window
state = {"n": 0, "floor": None, "last_sec": -1, "events": []}
def cb(indata, frames, t, status):
    global ring
    x = indata[:, 0].copy(); chunks.append(x)
    ring = np.roll(ring, -len(x)); ring[-len(x):] = x
    state["n"] += len(x)
    if state["n"] % (FS // 4) != 0: return
    now = state["n"] / FS
    rms_db = 20*np.log10(np.sqrt(np.mean(ring**2)) + 1e-9)
    if state["floor"] is None: state["floor"] = rms_db
    state["floor"] = min(state["floor"] + 0.02, rms_db) if rms_db > state["floor"] else rms_db  # slow-rise fast-fall floor
    spec = np.abs(np.fft.rfft(ring * np.hanning(len(ring))))**2
    f = np.fft.rfftfreq(len(ring), 1/FS); b = (f >= 300) & (f <= 5000)
    fb, sb = f[b], spec[b]; i = sb.argmax()
    prom = 10*np.log10(sb[i] / (np.median(sb) + 1e-15) + 1e-15)
    tag = ""
    if prom > 22 and rms_db > state["floor"] + 6:
        tag = f"  TONE f={fb[i]:6.0f} Hz prom={prom:4.0f} dB"; state["events"].append((now, fb[i], rms_db, prom))
    elif rms_db > state["floor"] + 15:
        tag = "  LOUD (broadband)"
    if tag or int(now) != state["last_sec"]:
        if tag or int(now) % 5 == 0:
            print(f"t={now:5.1f}s level={rms_db:6.1f} dBFS floor={state['floor']:6.1f}{tag}", flush=True)
        state["last_sec"] = int(now)
print(f"monitor: device {DEV} '{sd.query_devices(DEV)['name'][:40]}' for {DUR:.0f}s", flush=True)
with sd.InputStream(device=DEV, samplerate=FS, channels=1, dtype="float32", blocksize=480, callback=cb):
    sd.sleep(int(DUR * 1000))
x = np.concatenate(chunks); wavfile.write(OUT, FS, (np.clip(x, -1, 1)*32767).astype(np.int16))
ev = state["events"]
print(f"\nDONE. saved {OUT}. {len(ev)} tone windows detected.")
if ev:
    fr = np.array([e[1] for e in ev]); print(f"tone frequency median {np.median(fr):.0f} Hz, IQR {np.percentile(fr,25):.0f}-{np.percentile(fr,75):.0f} Hz")
