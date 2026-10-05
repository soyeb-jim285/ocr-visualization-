"""Original score for the brag video: A minor, 96 bpm, synced to composition/index.html scene times."""
import sys
import wave

import numpy as np
from scipy.signal import butter, lfilter

SR = 48000
DUR = 21.4
BEAT = 60 / 96
BAR = 4 * BEAT
REVEAL, TRAY, OUTRO = 3.0, 7.4, 18.2
TEXT_HITS = [(0.15, 69), (3.7, 76), (7.75, 72), (11.35, 74), (14.95, 76), (18.4, 81)]  # (time, midi)
rng = np.random.default_rng(7)
t = np.arange(int(SR * DUR)) / SR
L = np.zeros_like(t)
R = np.zeros_like(t)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    sig = sig[: max(0, len(t) - i)]
    L[i:i + len(sig)] += sig * gain * np.sqrt(0.5 * (1 - pan))
    R[i:i + len(sig)] += sig * gain * np.sqrt(0.5 * (1 + pan))


def lp(x, f, order=2):
    b, a = butter(order, f / (SR / 2), "low")
    return lfilter(b, a, x)


def hp(x, f, order=2):
    b, a = butter(order, f / (SR / 2), "high")
    return lfilter(b, a, x)


def soft_saw(f, n, det=0.0):
    tt = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * k * f * (1 + det) * tt) * (0.62 ** k) / k for k in range(1, 8))


def mallet(m, dur=1.2):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = hz(m)
    env = np.exp(-tt * 5.5) * (1 - np.exp(-tt * 900))
    return (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt * 14)) * env


# --- pad: one chord per bar
CHORDS = [[57, 60, 64], [57, 60, 64], [53, 57, 60], [48, 52, 55, 60], [57, 60, 64], [53, 57, 60, 65],
          [48, 52, 55, 64], [55, 59, 62], [45, 57, 60, 64]]
pad_L = np.zeros_like(t)
pad_R = np.zeros_like(t)
for bi, chord in enumerate(CHORDS):
    s0 = bi * BAR
    n = int((BAR + 1.2) * SR)
    tt = np.arange(n) / SR
    env = np.clip(tt / 0.5, 0, 1) * np.clip((BAR + 1.2 - tt) / 1.2, 0, 1)
    vl = sum(soft_saw(hz(m), n, -0.0025) for m in chord) * env
    vr = sum(soft_saw(hz(m), n, 0.0025) for m in chord) * env
    i = int(s0 * SR)
    m_ = min(n, len(t) - i)
    if m_ > 0:
        pad_L[i:i + m_] += vl[:m_]
        pad_R[i:i + m_] += vr[:m_]
pad_cut = 500 + 1700 * np.clip((t - 1.5) / 3, 0, 1)            # pad opens up into the reveal
pad_L = lp(pad_L, 900) * 0.6 + lp(pad_L, 2200) * 0.4 * np.clip((pad_cut - 500) / 1700, 0, 1)
pad_R = lp(pad_R, 900) * 0.6 + lp(pad_R, 2200) * 0.4 * np.clip((pad_cut - 500) / 1700, 0, 1)
L += pad_L * 0.055
R += pad_R * 0.055

# --- pencil-on-paper texture while the stroke is drawn
n = int(1.45 * SR)
scratch = hp(lp(rng.standard_normal(n), 5200), 1800) * (0.5 + 0.5 * np.sin(np.arange(n) / SR * 2 * np.pi * 9) ** 2)
scratch *= np.clip(np.arange(n) / (0.08 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.15 * SR), 0, 1)
add(scratch, 0.12, pan=0.25, gain=0.035)

# --- riser into the reveal
n = int(1.4 * SR)
tt = np.arange(n) / SR
noise = rng.standard_normal(n)
rise = sum(hp(noise, f) * ((tt > i * 0.35) & (tt <= (i + 1) * 0.35 + 0.05)) for i, f in enumerate([600, 1500, 3000, 5000]))
rise = lp(rise, 7000) * (tt / 1.4) ** 2.2 * 0.06
add(rise, REVEAL - 1.4)
# soft impact on reveal
n = int(2.2 * SR)
tt = np.arange(n) / SR
boom = np.sin(2 * np.pi * (55 * tt + 40 * (1 - np.exp(-tt * 9)) / 9)) * np.exp(-tt * 2.2)
add(boom, REVEAL, gain=0.32)

# --- pulse: soft kick on 1 & 3, offbeat ticks after the tray
def kick():
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    ph = 2 * np.pi * (48 * tt + 70 * (1 - np.exp(-tt * 30)) / 30)
    return np.sin(ph) * np.exp(-tt * 9)


def tick():
    n = int(0.08 * SR)
    return hp(rng.standard_normal(n), 7000) * np.exp(-np.arange(n) / SR * 60)


beat = REVEAL
while beat < OUTRO - 0.05:
    k = round((beat - REVEAL) / BEAT)
    if k % 2 == 0:
        add(kick(), beat, gain=0.30)
    if beat >= TRAY:
        add(tick(), beat + BEAT / 2, pan=0.4 if k % 2 else -0.4, gain=0.05)
    beat += BEAT
add(kick(), OUTRO, gain=0.38)

# --- arpeggio from the reveal: chord tones, 8ths, gentle
arp_bus = np.zeros((2, len(t)))
s = REVEAL + BEAT / 2
step = 0
while s < OUTRO + BAR:
    chord = CHORDS[min(int(s / BAR), len(CHORDS) - 1)]
    m = chord[step % len(chord)] + 12
    sig = mallet(m, 0.6) * 0.5
    i = int(s * SR)
    m_ = min(len(sig), len(t) - i)
    if m_ > 0:
        pan = -0.5 if step % 2 else 0.5
        arp_bus[0, i:i + m_] += sig[:m_] * np.sqrt(0.5 * (1 - pan))
        arp_bus[1, i:i + m_] += sig[:m_] * np.sqrt(0.5 * (1 + pan))
    s += BEAT / 2
    step += 1
arp_env = np.clip((t - REVEAL) / 1.5, 0, 1) * np.clip((DUR - 0.8 - t) / 2.0, 0, 1)
L += lp(arp_bus[0], 3500) * arp_env * 0.06
R += lp(arp_bus[1], 3500) * arp_env * 0.06

# --- text-entry mallet hits (in key)
for at, m in TEXT_HITS:
    add(mallet(m, 1.6), at, pan=0.15, gain=0.10)
    add(mallet(m - 12, 1.6), at, pan=-0.15, gain=0.05)


# --- light reverb (Schroeder) on the whole mix
def reverb(x):
    out = np.zeros_like(x)
    for d, g in [(1557, 0.80), (1617, 0.79), (1491, 0.81), (1422, 0.78)]:
        a = np.zeros(d + 1)
        a[0], a[d] = 1, -g
        out += lfilter([1], a, x)
    for d, g in [(225, 0.5), (556, 0.5)]:
        bcoef = np.zeros(d + 1)
        bcoef[0], bcoef[d] = -g, 1
        a = np.zeros(d + 1)
        a[0], a[d] = 1, -g
        out = lfilter(bcoef, a, out)
    return lp(out, 6000) * 0.12


L, R = L + reverb(L), R + reverb(R)
mix = np.stack([L, R], 1)
mix *= np.clip(t / 0.05, 0, 1)[:, None] * np.clip((DUR - t) / 0.6, 0, 1)[:, None]
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix *= 0.89 / np.max(np.abs(mix))                                # ≈ -1 dBFS peak

out = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, f"{DUR}s")
