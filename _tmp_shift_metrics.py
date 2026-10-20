"""临时分析：D2/D6/D7 相对 D1 的频域偏移指纹（site 0, n=100/组）"""
import numpy as np
from CIF import fun

PRE = "SaveData/PreTrain/D1-PreTrain.npy"
DIRS = {"D2": "Traces/D2-Undefence", "D6": "Traces/D6-Undefence", "D7": "Traces/D7-Undefence"}
SITE, N = 0, 100

pre = np.load(PRE, allow_pickle=True).item()
mask = pre["label"] == SITE
d1_x = pre["dataset"][mask].astype(np.float64)
d1_slot = pre["slot_duration"][mask].astype(np.float64)

def trace_features(path):
    times, sizes = [], []
    with open(path) as f:
        for line in f:
            t, s = line.rstrip("\n").split("\t")[:2]
            times.append(float(t)); sizes.append(int(float(s)))
    times = np.asarray(times) - float(times[0])
    feat, _, slot = fun(times, np.asarray(sizes))
    return np.array(feat, dtype=np.float64), slot

groups = {"D1": (d1_x, d1_slot)}
for name, d in DIRS.items():
    xs, ss = [], []
    for i in range(N):
        f_, s_ = trace_features(f"{d}/{SITE}-{i}")
        xs.append(f_); ss.append(s_)
    groups[name] = (np.array(xs), np.array(ss))

nyq = min(1 / (2 * g[1].max()) for g in groups.values())
fc = np.linspace(0, nyq, 1000)

specs = {}
for name, (x, slot) in groups.items():
    S = np.empty((len(x), 2, len(fc)))
    for i in range(len(x)):
        fr = np.fft.rfftfreq(x.shape[-1], d=slot[i])
        amp = np.abs(np.fft.rfft(x[i], axis=-1)) / x.shape[-1]
        for ch in range(2):
            S[i, ch] = np.interp(fc, fr, amp[ch])
    specs[name] = S

def db(a):
    return 20 * np.log10(np.clip(a, 1e-9, None))

i05, i3, i08, i4 = (int(v / nyq * 999) for v in (0.5, 3.0, 0.8, 4.0))
bands = {"low<0.5Hz": (1, i05), "mid0.5-3Hz": (i05, i3), "high>3Hz": (i3, 999)}

print(f"nyq={nyq:.3f} Hz, bins: low=1..{i05}, mid={i05}..{i3}, high={i3}..999")
for name in ["D2", "D6", "D7"]:
    S, S1 = specs[name], specs["D1"]
    m, m1 = db(S.mean(axis=0)), db(S1.mean(axis=0))       # 域均值谱(dB)
    wd, wd1 = db(S).std(axis=0), db(S1).std(axis=0)       # 域内逐trace dB 标准差
    print(f"--- {name} vs D1 (site {SITE}) ---")
    for ch in range(2):
        off = " | ".join(f"{bn}: { (m[ch][a:b+1] - m1[ch][a:b+1]).mean():+.1f}dB (域内std {wd[ch][a:b+1].mean():.1f} vs D1 {wd1[ch][a:b+1].mean():.1f})"
                         for bn, (a, b) in bands.items())
        dcr = S[:, ch, 0].mean() / S1[:, ch, 0].mean()
        pk1 = np.array([fc[i08 + int(np.argmax(S1[i, ch, i08:i4]))] for i in range(len(S1))])
        pk = np.array([fc[i08 + int(np.argmax(S[i, ch, i08:i4]))] for i in range(len(S))])
        print(f" ch{ch}: {off}")
        print(f"       DC ratio {dcr:.2f} | 中频主峰位置 D1 {pk1.mean():.2f}+/-{pk1.std():.2f}Hz vs {name} {pk.mean():.2f}+/-{pk.std():.2f}Hz")
