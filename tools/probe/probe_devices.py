import numpy as np, sounddevice as sd
FS = 48000
def rec(dev, secs=3, extra=None, ch=1, fs=FS):
    try:
        info = sd.query_devices(dev)
        a = sd.rec(int(secs*fs), samplerate=fs, channels=ch, dtype="float32", device=dev, extra_settings=extra)
        sd.wait()
        x = a[:, 0]
        rms = 20*np.log10(np.sqrt(np.mean(x**2))+1e-9); pk = 20*np.log10(np.abs(x).max()+1e-9)
        # spectral flatness of the ambient: real acoustic noise is broadband
        spec = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
        f = np.fft.rfftfreq(len(x), 1/fs); b = (f>200)&(f<4000)
        print(f"dev {dev:2d} {info['name'][:45]:45s} api={sd.query_hostapis(info['hostapi'])['name'][:18]:18s} rms={rms:6.1f} dBFS peak={pk:6.1f} dBFS band200-4k={10*np.log10(spec[b].mean()+1e-15):6.1f}")
    except Exception as e:
        print(f"dev {dev:2d} FAILED: {type(e).__name__}: {str(e)[:80]}")
print("3 s ambient recordings (stay quiet):", flush=True)
rec(1)                                     # MME default
rec(9)                                     # WASAPI shared
rec(9, extra=sd.WasapiSettings(exclusive=True))  # WASAPI exclusive (bypasses Windows APO effects)
rec(18)                                    # WDM-KS raw array 1
rec(20)                                    # WDM-KS raw array 3
