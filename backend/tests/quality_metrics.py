"""Objective quality harness for the loop ('band') instrumental engine.

Renders a few genre/key/tempo profiles through the REAL loop engine and prints
the metrics that actually correlate with 'sounds professional / release-ready':

  - integrated LUFS            (commercial masters land in a known window)
  - true peak (dBTP)           (must stay under ~ -1.0 to survive lossy codecs)
  - crest factor (dB)          (peak-to-RMS: too low = squashed, too high = weak)
  - spectral tilt (dB/oct)     (overall tonal balance: boomy vs harsh vs neutral)
  - low/mid/high energy split  (is the low end controlled, are highs present)
  - stereo width + mono compat (does it collapse to mono cleanly)
  - DC offset                  (should be ~0)

Run:  .venv/Scripts/python.exe tests/quality_metrics.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enjoi.core import config  # noqa: E402
from enjoi.modules import band  # noqa: E402

SR = config.SAMPLE_RATE


def _lufs(x):
    import pyloudnorm as pyln
    try:
        return float(pyln.Meter(SR).integrated_loudness(x.T.astype(np.float64)))
    except Exception:
        return float("nan")


def _true_peak_db(x):
    from scipy.signal import resample_poly
    up = resample_poly(x.astype(np.float64), 4, 1, axis=-1)
    p = float(np.max(np.abs(up))) if up.size else 0.0
    return 20.0 * np.log10(max(p, 1e-9))


def _crest_db(x):
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * np.log10(peak / rms)


def _bands(x):
    """Return (low<200, mid 200-4k, high>4k) energy fractions and tilt dB/oct."""
    mono = x.mean(axis=0).astype(np.float64)
    n = len(mono)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(mono * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    total = float(spec.sum()) + 1e-20
    lo = float(spec[freqs < 200].sum()) / total
    mid = float(spec[(freqs >= 200) & (freqs < 4000)].sum()) / total
    hi = float(spec[freqs >= 4000].sum()) / total
    # spectral tilt: linear fit of 10*log10(power) vs log2(freq) over 50..16k
    m = (freqs >= 50) & (freqs <= 16000) & (spec > 0)
    tilt = float(np.polyfit(np.log2(freqs[m]), 10.0 * np.log10(spec[m]), 1)[0])
    return lo, mid, hi, tilt


def _stereo(x):
    if x.shape[0] < 2:
        return 0.0, 1.0
    l, r = x[0].astype(np.float64), x[1].astype(np.float64)
    mid = (l + r) / 2.0
    side = (l - r) / 2.0
    width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
    mono = l + r
    mono_loss = float(np.sqrt(np.mean(mono ** 2)) / (np.sqrt(np.mean(l ** 2) + np.mean(r ** 2)) + 1e-12))
    return width, mono_loss


PROFILES = [
    dict(name="pop-Cmaj-100", bpm=100.0, key={"tonic": "C", "mode": "major"},
         genre_tags=["pop"]),
    dict(name="trap-Gmin-140", bpm=140.0, key={"tonic": "G", "mode": "minor"},
         genre_tags=["trap"]),
    dict(name="folk-Emin-116", bpm=116.0, key={"tonic": "E", "mode": "minor"},
         genre_tags=["folk"]),
]
STRUCTURE = [{"label": "intro", "bars": 2}, {"label": "verse", "bars": 8},
             {"label": "chorus", "bars": 8}, {"label": "verse", "bars": 8},
             {"label": "chorus", "bars": 8}, {"label": "outro", "bars": 4}]


def run():
    print(f"loop library available: {band.library_available()}  "
          f"dir={band.library_dir()}")
    for prof in PROFILES:
        plan = dict(bpm=prof["bpm"], key=prof["key"], time_signature="4/4",
                    genre_tags=prof["genre_tags"], structure=STRUCTURE, seed=7)
        t = time.time()
        out = band._render_loops(plan, None)
        dt = time.time() - t
        lo, mid, hi, tilt = _bands(out)
        width, mono = _stereo(out)
        print(f"\n=== {prof['name']}  ({dt:.1f}s, {out.shape[1]/SR:.0f}s) ===")
        print(f"  LUFS        {_lufs(out):7.2f}   (target ~ -9..-11 for an instr bed)")
        print(f"  true peak   {_true_peak_db(out):7.2f} dBTP  (must be < -1.0)")
        print(f"  crest       {_crest_db(out):7.2f} dB    (mainstream ~ 8..13)")
        print(f"  tilt        {tilt:7.2f} dB/oct (neutral ~ -3 .. -4.5)")
        print(f"  lo/mid/hi   {lo*100:4.1f}% / {mid*100:4.1f}% / {hi*100:4.1f}%")
        print(f"  width       {width:7.3f}      mono-keep {mono:7.3f}")


if __name__ == "__main__":
    run()
