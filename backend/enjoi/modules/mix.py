"""Mixing & mastering module (spec 4.9).

Two-source render graph (ownership guarantee, contract rule 6): the ONLY audio
read here is project.instrumental_path + the tuned vocal chops referenced by
the arrangement. `_ref_cache/` is never touched.

Buses:
  vocal bus   : HPF 90 Hz -> de-esser (6-9 kHz band compression) -> compressor
                (2.5:1, threshold tuned for ~4 dB GR from measured bus RMS) ->
                peak EQ (preset) -> plate reverb send (60 ms pre-delay) ->
                tempo-synced 1/4-note delay send.
  chorus chops: parallel tanh saturation + stereo doubler (±12 ms, ±6 cents,
                −6 dB) before joining the bus; bridge chops get an extra
                150 Hz HPF and +1 dB reverb send.
  instr bus   : gentle bus compressor -> sidechain duck (up to −1.5 dB, keyed
                by the vocal-bus envelope, 50 ms attack / 250 ms release).
  master      : glue compressor (~2 dB GR, slow attack) -> tilt EQ -> loudness
                normalize (pyloudnorm) -> 4x-oversampled lookahead true-peak
                limiter at −1 dBTP -> clip guard.

Pedalboard is used when importable; every stage has a scipy/numpy fallback
(contract rule 2). Module top level: stdlib + numpy + enjoi.core only.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from ..core import audio as core_audio
from ..core import config, deps
from ..core.errors import PipelineError
from ..core.storage import write_json

log = logging.getLogger("enjoi.mix")

ENV_HOP = 512                 # control-rate hop for envelope followers
SILENCE_FLOOR_DB = -55.0
REVERB_PREDELAY_SEC = 0.060
DELAY_FEEDBACK = 0.25
DOUBLER_OFFSET_SEC = 0.012
DOUBLER_CENTS = 6.0
DOUBLER_GAIN_DB = -6.0
DUCK_MAX_DB = 1.5
TP_CEILING_DB = -1.0

# Gain-staging: the lead vocal must sit clearly ON TOP of the instrumental
# (the #1 product complaint is drowned vocals). We measure both buses and set
# the vocal level RELATIVE to the instrumental so the lead sits in a modern,
# lead-vocal-forward window above it.
# Vocal sits just UNDER the lead melody: present and clear, but the melody leads.
# Measured over the whole instrumental (melody is its loudest stem), so a small
# positive offset keeps the vocal ~even-with / just-below the melody. Generous
# range preserves artistic expression.
VOCAL_OVER_INSTR_LU = 1.5      # target: lead vocal ~this many LU over the instrumental avg
VOCAL_OVER_INSTR_MIN = 0.5     # never drowned
VOCAL_OVER_INSTR_MAX = 3.0     # nor shouty
VOCAL_STAGE_GAIN_LIMIT_DB = 18.0  # clamp the relative move so one bus can't run away

# Preset flavors: small sensible variations per genre (spec 4.9).
PRESETS: dict[str, dict] = {
    "pop": {
        "eq": [(4000.0, 2.0, 1.0), (300.0, -2.0, 1.0)],
        "reverb_size": 0.50, "reverb_mix": 0.15, "delay_mix": 0.08,
        "saturation_mix": 0.20, "inst_ratio": 1.5, "inst_gr_db": 1.5,
        "tilt_db": 0.5,
    },
    "hiphop": {
        "eq": [(4500.0, 1.5, 1.0), (250.0, -2.5, 1.0)],
        "reverb_size": 0.35, "reverb_mix": 0.12, "delay_mix": 0.05,
        "saturation_mix": 0.25, "inst_ratio": 1.4, "inst_gr_db": 1.0,
        "tilt_db": -1.0,
    },
    "rnb": {
        "eq": [(3500.0, 1.5, 0.9), (300.0, -1.5, 1.0)],
        "reverb_size": 0.55, "reverb_mix": 0.18, "delay_mix": 0.10,
        "saturation_mix": 0.15, "inst_ratio": 1.5, "inst_gr_db": 1.5,
        "tilt_db": -0.5,
    },
    "rock": {
        "eq": [(4200.0, 2.5, 1.1), (350.0, -2.0, 1.0)],
        "reverb_size": 0.45, "reverb_mix": 0.13, "delay_mix": 0.06,
        "saturation_mix": 0.30, "inst_ratio": 1.6, "inst_gr_db": 2.0,
        "tilt_db": 1.0,
    },
    "acoustic": {
        "eq": [(5000.0, 1.0, 0.8), (250.0, -1.0, 0.9)],
        "reverb_size": 0.60, "reverb_mix": 0.16, "delay_mix": 0.04,
        "saturation_mix": 0.08, "inst_ratio": 1.3, "inst_gr_db": 1.0,
        "tilt_db": 0.0,
    },
}


# ---------------------------------------------------------------------------
# small numpy/scipy DSP primitives (always available — also the fallbacks)
# ---------------------------------------------------------------------------

def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x])
    elif x.shape[0] == 1:
        x = np.vstack([x, x])
    elif x.shape[0] > 2:
        x = x[:2]
    return np.ascontiguousarray(x, dtype=np.float32)


def _butter_filter(x: np.ndarray, freq, sr: int, btype: str, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(order, freq, btype=btype, fs=sr, output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def _biquad(x: np.ndarray, b, a) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter(np.asarray(b) / a[0], np.asarray(a) / a[0], x, axis=-1).astype(np.float32)


def _peak_eq(x: np.ndarray, sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """Peaking EQ — pedalboard.PeakFilter when available, RBJ biquad fallback."""
    if abs(gain_db) < 1e-3:
        return x
    pb = deps.optional_import("pedalboard")
    if pb is not None and hasattr(pb, "PeakFilter"):
        try:
            board = pb.Pedalboard([pb.PeakFilter(cutoff_frequency_hz=f0, gain_db=gain_db, q=q)])
            return np.asarray(board(x.astype(np.float32), sr), dtype=np.float32)
        except Exception:
            pass
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = [1.0 + alpha * A, -2.0 * cw, 1.0 - alpha * A]
    a = [1.0 + alpha / A, -2.0 * cw, 1.0 - alpha / A]
    return _biquad(x, b, a)


def _shelf(x: np.ndarray, sr: int, f0: float, gain_db: float, high: bool) -> np.ndarray:
    """Low/high shelf — pedalboard when available, RBJ biquad fallback."""
    if abs(gain_db) < 1e-3:
        return x
    pb = deps.optional_import("pedalboard")
    name = "HighShelfFilter" if high else "LowShelfFilter"
    if pb is not None and hasattr(pb, name):
        try:
            flt = getattr(pb, name)(cutoff_frequency_hz=f0, gain_db=gain_db, q=0.707)
            return np.asarray(pb.Pedalboard([flt])(x.astype(np.float32), sr), dtype=np.float32)
        except Exception:
            pass
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt(2.0)
    sq = 2.0 * math.sqrt(A) * alpha
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - sq]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - sq]
    return _biquad(x, b, a)


def _hpf(x: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    pb = deps.optional_import("pedalboard")
    if pb is not None and hasattr(pb, "HighpassFilter"):
        try:
            board = pb.Pedalboard([pb.HighpassFilter(cutoff_frequency_hz=freq)])
            return np.asarray(board(x.astype(np.float32), sr), dtype=np.float32)
        except Exception:
            pass
    return _butter_filter(x, freq, sr, "high", order=order)


# ---- envelope follower / compressor ----------------------------------------

def _frame_rms_db(mono: np.ndarray, hop: int = ENV_HOP) -> np.ndarray:
    n = len(mono)
    nf = max(n // hop, 1)
    trimmed = mono[: nf * hop].astype(np.float64).reshape(nf, hop)
    r = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-20)
    return 20.0 * np.log10(r + 1e-12)


def _smooth_env_db(env_db: np.ndarray, sr: int, hop: int, attack: float, release: float) -> np.ndarray:
    a_att = math.exp(-hop / (sr * max(attack, 1e-4)))
    a_rel = math.exp(-hop / (sr * max(release, 1e-4)))
    out = np.empty_like(env_db)
    prev = float(env_db[0])
    for i in range(len(env_db)):
        v = float(env_db[i])
        a = a_att if v > prev else a_rel
        prev = a * prev + (1.0 - a) * v
        out[i] = prev
    return out


def _frames_gain_to_samples(gain_db_frames: np.ndarray, hop: int, n: int) -> np.ndarray:
    centers = np.arange(len(gain_db_frames), dtype=np.float64) * hop + hop / 2.0
    g = np.interp(np.arange(n, dtype=np.float64), centers, 10.0 ** (gain_db_frames / 20.0))
    return g.astype(np.float32)


def _pick_threshold(env_db_active: np.ndarray, ratio: float, target_gr_db: float) -> float:
    """Scan thresholds so mean gain reduction over active frames ~= target."""
    lo = float(np.min(env_db_active)) - 5.0
    hi = float(np.max(env_db_active))
    k = 1.0 - 1.0 / max(ratio, 1.01)
    best_t, best_err = hi, 1e9
    for t in np.linspace(lo, hi, 80):
        gr = float(np.mean(np.maximum(env_db_active - t, 0.0))) * k
        err = abs(gr - target_gr_db)
        if err < best_err:
            best_err, best_t = err, float(t)
    return best_t


def _bus_compress(
    x: np.ndarray, sr: int, ratio: float, target_gr_db: float,
    attack: float, release: float,
) -> np.ndarray:
    """Compressor with threshold auto-tuned from the measured bus envelope.

    Uses pedalboard.Compressor when available; numpy feed-forward fallback.
    """
    mono = np.mean(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    env = _frame_rms_db(mono)
    env = _smooth_env_db(env, sr, ENV_HOP, attack, release)
    active = env[env > SILENCE_FLOOR_DB]
    if active.size < 4:
        return x
    threshold = _pick_threshold(active, ratio, target_gr_db)

    pb = deps.optional_import("pedalboard")
    if pb is not None and hasattr(pb, "Compressor"):
        try:
            board = pb.Pedalboard([pb.Compressor(
                threshold_db=float(threshold), ratio=float(ratio),
                attack_ms=attack * 1000.0, release_ms=release * 1000.0,
            )])
            return np.asarray(board(x.astype(np.float32), sr), dtype=np.float32)
        except Exception:
            pass
    k = 1.0 - 1.0 / ratio
    gain_db = -np.maximum(env - threshold, 0.0) * k
    g = _frames_gain_to_samples(gain_db, ENV_HOP, x.shape[-1])
    return (x * g[None, :] if x.ndim == 2 else x * g).astype(np.float32)


def _deesser(x: np.ndarray, sr: int, max_gr_db: float = 6.0) -> np.ndarray:
    """Split-band de-esser: compress the 6-9 kHz band 4:1 above its p85 level."""
    band = _butter_filter(x, [6000.0, 9000.0], sr, "band", order=2)
    mono = np.mean(np.abs(band), axis=0)
    env = _frame_rms_db(mono)
    env = _smooth_env_db(env, sr, ENV_HOP, attack=0.003, release=0.05)
    active = env[env > -60.0]
    if active.size < 4:
        return x
    threshold = float(np.percentile(active, 85.0))
    gr = np.minimum(np.maximum(env - threshold, 0.0) * 0.75, max_gr_db)  # 4:1 capped
    g = _frames_gain_to_samples(-gr, ENV_HOP, x.shape[-1])
    return (x - band + band * g[None, :]).astype(np.float32)


# ---- reverb / delay ----------------------------------------------------------

def _comb(x: np.ndarray, delay: int, g: float) -> np.ndarray:
    """y[n] = x[n] + g*y[n-D], computed block-wise (exact, vectorized)."""
    n = len(x)
    if delay <= 0 or delay >= n:
        return x.copy()
    y = np.zeros_like(x)
    for k in range(0, n, delay):
        e = min(k + delay, n)
        if k == 0:
            y[k:e] = x[k:e]
        else:
            y[k:e] = x[k:e] + g * y[k - delay:k - delay + (e - k)]
    return y


def _allpass(x: np.ndarray, delay: int, g: float) -> np.ndarray:
    """y[n] = -g*x[n] + x[n-D] + g*y[n-D], block-wise."""
    n = len(x)
    if delay <= 0 or delay >= n:
        return x.copy()
    xd = np.zeros_like(x)
    xd[delay:] = x[:-delay]
    y = np.zeros_like(x)
    for k in range(0, n, delay):
        e = min(k + delay, n)
        past = y[k - delay:k - delay + (e - k)] if k >= delay else 0.0
        y[k:e] = -g * x[k:e] + xd[k:e] + g * past
    return y


def _reverb_wet(x: np.ndarray, sr: int, size: float) -> np.ndarray:
    """Full-wet plate-style reverb; pedalboard.Reverb or Schroeder fallback."""
    pb = deps.optional_import("pedalboard")
    if pb is not None and hasattr(pb, "Reverb"):
        try:
            board = pb.Pedalboard([pb.Reverb(
                room_size=float(min(max(size, 0.0), 1.0)), damping=0.5,
                wet_level=1.0, dry_level=0.0, width=1.0,
            )])
            return np.asarray(board(x.astype(np.float32), sr), dtype=np.float32)
        except Exception:
            pass
    # Schroeder fallback: 4 parallel combs -> 2 series allpasses, per channel.
    size = float(min(max(size, 0.0), 1.0))
    base_delays = np.array([0.0297, 0.0371, 0.0411, 0.0437]) * (0.6 + 0.9 * size)
    feedback = 0.70 + 0.18 * size
    pre_lp = _butter_filter(x, 6000.0, sr, "low", order=1)  # plate damping feel
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        xin = pre_lp[c]
        acc = np.zeros_like(xin)
        for i, d in enumerate(base_delays):
            D = max(int(d * sr * (1.013 if c == 1 else 1.0) * (1.0 + 0.007 * i)), 1)
            acc += _comb(xin, D, feedback)
        acc *= 0.25
        for dm, ga in ((0.0050, 0.7), (0.0017, 0.7)):
            acc = _allpass(acc, max(int(dm * sr), 1), ga)
        out[c] = acc
    return (out * 0.6).astype(np.float32)


def _delay_wet(x: np.ndarray, sr: int, delay_sec: float, feedback: float, taps: int = 6) -> np.ndarray:
    n = x.shape[-1]
    D = int(delay_sec * sr)
    wet = np.zeros_like(x)
    if D < 1:
        return wet
    g = 1.0
    for k in range(1, taps + 1):
        off = k * D
        if off >= n:
            break
        wet[:, off:] += x[:, : n - off] * g
        g *= feedback
    return wet.astype(np.float32)


# ---- loudness / limiting ------------------------------------------------------

def _measure_lufs(x: np.ndarray, sr: int) -> float:
    pyln = deps.optional_import("pyloudnorm")
    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            val = float(meter.integrated_loudness(x.T.astype(np.float64)))
            if math.isfinite(val):
                return val
        except Exception as exc:
            log.debug("pyloudnorm failed (%s), using K-weighted approximation", exc)
    # K-weighting approximation (BS.1770-ish): HPF 38 Hz + high shelf +4 dB.
    y = _butter_filter(x, 38.0, sr, "high", order=2)
    y = _shelf(y, sr, 1650.0, 4.0, high=True)
    msq = float(np.mean(np.sum(y.astype(np.float64) ** 2, axis=0) / y.shape[0]))
    if msq <= 0:
        return float("-inf")
    return -0.691 + 10.0 * math.log10(msq)


def _active_loudness(x: np.ndarray, sr: int) -> float:
    """Loudness (LUFS, else RMS dBFS) measured over the bus's ACTIVE regions only.

    For gain-staging we care about how loud the lead vocal is *while it is
    sounding*, not its average over a track full of inter-phrase silence. We
    gate to frames within 25 dB of the bus's loud peak, then measure that
    subset's loudness. Returns -inf for an empty/silent bus.
    """
    if x.size == 0:
        return float("-inf")
    mono = np.mean(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    env_db = _frame_rms_db(mono, ENV_HOP)
    if env_db.size == 0:
        return float("-inf")
    peak = float(np.max(env_db))
    if not math.isfinite(peak) or peak <= SILENCE_FLOOR_DB:
        return float("-inf")
    gate = max(peak - 25.0, SILENCE_FLOOR_DB)
    active = env_db >= gate
    if not np.any(active):
        return float("-inf")
    # expand the frame mask to samples and measure loudness of the active audio
    n = x.shape[-1]
    centers = np.arange(len(env_db), dtype=np.float64) * ENV_HOP + ENV_HOP / 2.0
    samp_active = np.interp(
        np.arange(n, dtype=np.float64), centers, active.astype(np.float64)
    ) >= 0.5
    if not np.any(samp_active):
        return float("-inf")
    sel = x[:, samp_active] if x.ndim == 2 else x[samp_active]
    sel = np.ascontiguousarray(_ensure_stereo(sel))
    lufs = _measure_lufs(sel, sr)
    if math.isfinite(lufs):
        return lufs
    return core_audio.lin_to_db(core_audio.rms(sel))


def _true_peak_db(x: np.ndarray, sr: int) -> float:
    from scipy.signal import resample_poly

    up = resample_poly(x.astype(np.float64), 4, 1, axis=-1)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return core_audio.lin_to_db(peak)


def _limit_true_peak(x: np.ndarray, sr: int, ceiling_db: float = TP_CEILING_DB) -> np.ndarray:
    """4x-oversampled peak detection + lookahead brickwall gain + clip guard."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import resample_poly

    n = x.shape[-1]
    if n == 0:
        return x
    ceiling = core_audio.db_to_lin(ceiling_db)
    up = resample_poly(x.astype(np.float64), 4, 1, axis=-1)
    if up.shape[-1] < 4 * n:
        up = np.pad(up, [(0, 0)] * (up.ndim - 1) + [(0, 4 * n - up.shape[-1])])
    tp = np.abs(up[..., : 4 * n]).reshape(x.shape[0], n, 4).max(axis=-1).max(axis=0)
    required = np.minimum(1.0, ceiling / np.maximum(tp, 1e-9))
    win = max(int(0.004 * sr) | 1, 3)  # ~4 ms lookahead, odd width
    gain = minimum_filter1d(required, size=win, mode="nearest")
    gain = uniform_filter1d(gain, size=win, mode="nearest")
    gain = np.minimum(gain, required)  # belt & suspenders: never exceed required
    out = (x * gain[None, :]).astype(np.float32)
    np.clip(out, -ceiling, ceiling, out=out)  # final clip guard
    return out


# ---------------------------------------------------------------------------
# placement loading
# ---------------------------------------------------------------------------

def _owned_source(project, rel: str) -> Path:
    """Resolve a project-relative audio path; enforce the ownership guarantee."""
    rel_path = Path(str(rel))
    if rel_path.is_absolute() or "_ref_cache" in rel_path.parts or ".." in rel_path.parts:
        raise PipelineError("Render sources must be project-owned audio (ownership guarantee).")
    full = (project.dir / rel_path).resolve()
    if not str(full).startswith(str(project.dir.resolve())):
        raise PipelineError("Render sources must live inside the project directory.")
    if not full.exists():
        raise PipelineError(f"Missing render source: {rel} — re-run tuning.")
    return full


def _build_vocal_events(project, placements: list[dict], preset: dict, sr: int, progress):
    """Load+condition each placed chop. Returns (events, boost_events).

    events: list of (channel, start_sample, mono_audio); channel 2 = both.
    boost_events feed the bridge's +1 dB reverb-send boost.
    """
    events: list[tuple[int, int, np.ndarray]] = []
    boost_events: list[tuple[int, np.ndarray]] = []
    sat_mix = float(preset["saturation_mix"])
    total = max(len(placements), 1)

    for i, p in enumerate(placements):
        rel = p.get("tuned_file") or p.get("chop_file")
        if not rel:
            continue
        role = str(p.get("role", "vocal"))
        progress(i / total, f"Placing {role} chop {i + 1}/{len(placements)}")
        path = _owned_source(project, rel)
        y, _ = core_audio.load_audio(path, mono=True)
        if y.size == 0:
            continue

        stretch = float(p.get("stretch") or 1.0)
        stretch = min(max(stretch, 0.5), 2.0)
        if abs(stretch - 1.0) > 1e-3:
            # stretch > 1 -> chop must fill MORE time -> rate = 1/stretch (<1)
            y = core_audio.time_stretch(y, rate=1.0 / stretch, sr=sr)

        y = y * core_audio.db_to_lin(float(p.get("gain_db") or 0.0))
        y = core_audio.fade_edges(y, 0.004, 0.010, sr).astype(np.float32)
        start = max(int(float(p.get("target_start") or 0.0) * sr), 0)

        if p.get("bridge_fx"):
            # thinner bridge: extra HPF 150 Hz; +1 dB reverb send via boost bus
            y = _butter_filter(y[None, :], 150.0, sr, "high", order=2)[0]
            boost_events.append((start, y * (core_audio.db_to_lin(1.0) - 1.0)))

        if role == "chorus":
            sat = np.tanh(3.0 * y) / math.tanh(3.0)
            main = ((1.0 - sat_mix) * y + sat_mix * sat).astype(np.float32)
            events.append((2, start, main))
            off = int(DOUBLER_OFFSET_SEC * sr)
            dgain = core_audio.db_to_lin(DOUBLER_GAIN_DB)
            dl = core_audio.pitch_shift(y, +DOUBLER_CENTS / 100.0, sr) * dgain
            dr = core_audio.pitch_shift(y, -DOUBLER_CENTS / 100.0, sr) * dgain
            events.append((0, max(start - off, 0), dl.astype(np.float32)))
            events.append((1, start + off, dr.astype(np.float32)))
        else:
            events.append((2, start, y.astype(np.float32)))
    return events, boost_events


def _render_bus(events: list[tuple[int, int, np.ndarray]], n: int) -> np.ndarray:
    bus = np.zeros((2, n), dtype=np.float32)
    for ch, start, y in events:
        end = min(start + len(y), n)
        if end <= start:
            continue
        seg = y[: end - start]
        if ch == 2:
            bus[0, start:end] += seg
            bus[1, start:end] += seg
        else:
            bus[ch, start:end] += seg
    return bus


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def mix_and_master(project, arrangement: dict, grid: dict, preset: str,
                   loudness_preset: str, progress) -> Path:
    """Render, mix and master the song; returns exports/_master_tmp.wav."""
    sr = config.SAMPLE_RATE
    flavor = PRESETS.get(str(preset).lower()) or PRESETS["pop"]
    loud = config.LOUDNESS_PRESETS.get(str(loudness_preset).lower()) \
        or config.LOUDNESS_PRESETS["streaming"]
    target_lufs = float(loud["lufs"])
    bpm = float(grid.get("bpm") or 120.0)
    if not math.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    progress(0.02, "Loading instrumental")
    inst, _ = core_audio.load_audio(project.instrumental_path, sr=sr, mono=False)
    inst = _ensure_stereo(inst)

    placements = [p for p in (arrangement or {}).get("placements") or []
                  if p.get("tuned_file") or p.get("chop_file")]
    events, boost_events = _build_vocal_events(
        project, placements, flavor, sr,
        lambda f, m: progress(0.04 + 0.21 * f, m),
    )

    n = inst.shape[-1]
    for _, start, y in events:
        n = max(n, start + len(y))
    inst = np.pad(inst, [(0, 0), (0, n - inst.shape[-1])]) if inst.shape[-1] < n else inst

    vocal = _render_bus(events, n)
    boost = _render_bus([(2, s, y) for s, y in boost_events], n)
    has_vocals = bool(np.any(np.abs(vocal) > 1e-7))

    # ---- vocal bus chain (spec 4.9) ---------------------------------------
    if has_vocals:
        progress(0.28, "Vocal bus: HPF + de-esser")
        vocal = _hpf(vocal, sr, 90.0)
        vocal = _deesser(vocal, sr)
        progress(0.34, "Vocal bus: compression (~4 dB GR)")
        vocal = _bus_compress(vocal, sr, ratio=2.5, target_gr_db=4.0,
                              attack=0.008, release=0.120)
        progress(0.40, "Vocal bus: EQ")
        for f0, gain_db, q in flavor["eq"]:
            vocal = _peak_eq(vocal, sr, f0, gain_db, q)

        progress(0.45, "Vocal bus: reverb + delay sends")
        send_in = (vocal + _hpf(boost, sr, 90.0)).astype(np.float32)
        wet = _reverb_wet(send_in, sr, flavor["reverb_size"])
        pre = int(REVERB_PREDELAY_SEC * sr)
        wet_delayed = np.zeros_like(wet)
        if wet.shape[-1] > pre:
            wet_delayed[:, pre:] = wet[:, :-pre]
        vocal = vocal + wet_delayed * float(flavor["reverb_mix"])
        dly = _delay_wet(vocal, sr, 60.0 / bpm, DELAY_FEEDBACK)
        vocal = (vocal + dly * float(flavor["delay_mix"])).astype(np.float32)
        vocal = np.nan_to_num(vocal, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- instrumental bus ---------------------------------------------------
    progress(0.55, "Instrumental bus: glue + sidechain duck")
    inst = _bus_compress(inst, sr, ratio=float(flavor["inst_ratio"]),
                         target_gr_db=float(flavor["inst_gr_db"]),
                         attack=0.030, release=0.250)
    if has_vocals:
        env = _frame_rms_db(np.mean(np.abs(vocal), axis=0))
        env = _smooth_env_db(env, sr, ENV_HOP, attack=0.050, release=0.250)
        activity = np.clip((env + 40.0) / 15.0, 0.0, 1.0)  # 0 below -40 dBFS
        duck = _frames_gain_to_samples(-DUCK_MAX_DB * activity, ENV_HOP, n)
        inst = (inst * duck[None, :]).astype(np.float32)
    inst = np.nan_to_num(inst, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- gain-staging: put the lead vocal ON TOP of the instrumental --------
    # Measure both buses' ACTIVE loudness and lift/trim the vocal bus so the
    # lead sits +2..+4 LU above the instrumental (lead-vocal-forward balance).
    # This is what stops the one-take from being drowned by the backing track.
    if has_vocals:
        progress(0.60, "Gain-staging: seating the vocal above the instrumental")
        inst_loud = _active_loudness(inst, sr)
        voc_loud = _active_loudness(vocal, sr)
        if math.isfinite(inst_loud) and math.isfinite(voc_loud):
            desired_voc = inst_loud + VOCAL_OVER_INSTR_LU
            stage_db = desired_voc - voc_loud
            stage_db = float(np.clip(
                stage_db, -VOCAL_STAGE_GAIN_LIMIT_DB, VOCAL_STAGE_GAIN_LIMIT_DB
            ))
            vocal = (vocal * core_audio.db_to_lin(stage_db)).astype(np.float32)
            log.debug(
                "gain-stage: inst=%.1f LU voc=%.1f LU -> +%.1f dB (target +%.1f LU)",
                inst_loud, voc_loud, stage_db, VOCAL_OVER_INSTR_LU,
            )
        vocal = np.nan_to_num(vocal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- stems (post-bus, pre-master-sum) -----------------------------------
    progress(0.62, "Writing stems")
    exports_dir = project.exports_dir
    core_audio.save_wav(exports_dir / "_stem_instrumental.wav", inst, sr, subtype="FLOAT")
    core_audio.save_wav(exports_dir / "_stem_vocals.wav", vocal, sr, subtype="FLOAT")

    # ---- master bus ----------------------------------------------------------
    progress(0.68, "Master: glue compression")
    mix = (inst + vocal).astype(np.float32)
    mix = _bus_compress(mix, sr, ratio=4.0, target_gr_db=2.0,
                        attack=0.030, release=0.300)
    tilt = float(flavor["tilt_db"])
    if abs(tilt) > 1e-3:
        mix = _shelf(mix, sr, 250.0, -tilt, high=False)
        mix = _shelf(mix, sr, 3500.0, +tilt, high=True)

    progress(0.78, f"Master: loudness normalize to {target_lufs:g} LUFS")
    measured = _measure_lufs(mix, sr)
    if math.isfinite(measured):
        mix = (mix * core_audio.db_to_lin(target_lufs - measured)).astype(np.float32)

    progress(0.86, "Master: true-peak limiting (-1 dBTP)")
    mix = _limit_true_peak(mix, sr, TP_CEILING_DB)
    after = _measure_lufs(mix, sr)
    if math.isfinite(after) and (target_lufs - after) > 0.5:
        # limiter ate some loudness (Loud preset) — one refinement pass
        mix = (mix * core_audio.db_to_lin(min(target_lufs - after, 3.0))).astype(np.float32)
        mix = _limit_true_peak(mix, sr, TP_CEILING_DB)

    mix = np.nan_to_num(mix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    final_lufs = _measure_lufs(mix, sr)
    final_tp = _true_peak_db(mix, sr)

    progress(0.94, "Writing master render")
    master_path = exports_dir / "_master_tmp.wav"
    core_audio.save_wav(master_path, mix, sr, subtype="FLOAT")
    write_json(exports_dir / "_master_meta.json", {
        "lufs": round(final_lufs, 2) if math.isfinite(final_lufs) else None,
        "true_peak_db": round(final_tp, 2),
        "preset": str(preset),
        "loudness_preset": str(loudness_preset),
    })
    progress(1.0, "Mix & master complete")
    return master_path
