"""Autotune module (spec 4.8): pitch-correct vocal chops to the instrumental key.

Pipeline per chop:
  f0 track (torchcrepe when available, librosa.pyin baseline) -> note
  segmentation on the median-filtered f0 -> per-note scale snap -> strength /
  hard-tune correction curve with smoothed note transitions -> per-note (or
  sub-note, for hard tune) pitch shifting via core.audio.pitch_shift with 10 ms
  crossfades -> optional de-breath pass -> vocal_tuned/t###.wav.

Engineering rules: heavy libs (librosa, scipy, torchcrepe) imported inside
functions only; module top level is stdlib + numpy + enjoi.core.*.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ..core import audio as core_audio
from ..core.errors import PipelineError
from ..core import deps

log = logging.getLogger("enjoi.tune")

# f0 tracking / segmentation constants (spec'd)
HOP = 256
FRAME = 2048
FMIN = 70.0
FMAX = 900.0
JUMP_SEMITONES = 0.6          # note-boundary jump threshold
JUMP_SUSTAIN_SEC = 0.040      # jump must hold this long to start a new note
CROSSFADE_SEC = 0.010         # segment re-join crossfade
HARD_TUNE_KNEE = 0.7          # strength above this engages hard-tune flatten
DEBREATH_GAIN_DB = -9.0
MIN_SHIFT_SEMITONES = 0.02    # below this, leave the segment untouched

_NOTE_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "F": 5, "E#": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}
_MAJOR_IV = (0, 2, 4, 5, 7, 9, 11)
_MINOR_IV = (0, 2, 3, 5, 7, 8, 10)


# ---------------------------------------------------------------------------
# key / scale helpers
# ---------------------------------------------------------------------------

def _scale_pitch_classes(grid: dict) -> list[int]:
    """Pitch classes (0..11) of the instrumental's key scale.

    Prefers grid["key"]["scale_midi"] mod 12; falls back to deriving the
    major/minor scale from tonic+mode; chromatic (no-op snap bias) if unknown.
    """
    key = (grid or {}).get("key") or {}
    scale_midi = key.get("scale_midi")
    if scale_midi:
        try:
            pcs = sorted({int(round(float(m))) % 12 for m in scale_midi})
            if pcs:
                return pcs
        except (TypeError, ValueError):
            pass
    tonic = str(key.get("tonic", "") or "").strip().upper()
    pc = _NOTE_PC.get(tonic)
    if pc is None:
        return list(range(12))  # unknown key: snap to nearest semitone
    mode = str(key.get("mode", "major") or "major").lower()
    intervals = _MINOR_IV if "min" in mode else _MAJOR_IV
    return sorted({(pc + iv) % 12 for iv in intervals})


def _snap_to_scale(midi_val: float, pcs: list[int]) -> float:
    """Nearest MIDI pitch whose pitch class is in the scale."""
    best = midi_val
    best_d = 1e9
    for pc in pcs:
        cand = pc + 12.0 * round((midi_val - pc) / 12.0)
        d = abs(cand - midi_val)
        if d < best_d:
            best_d, best = d, cand
    return float(best)


# ---------------------------------------------------------------------------
# f0 tracking
# ---------------------------------------------------------------------------

def _track_f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame f0 (Hz, 0 where unvoiced) + voiced flags at hop=HOP."""
    n_frames = 1 + len(y) // HOP
    frame_times = np.arange(n_frames) * (HOP / sr)

    # torchcrepe (optional, higher quality) ---------------------------------
    torchcrepe = deps.optional_import("torchcrepe")
    torch = deps.optional_import("torch")
    if torchcrepe is not None and torch is not None:
        try:
            sr16 = 16000
            y16 = core_audio.resample(y.astype(np.float32), sr, sr16)
            device = "cuda" if deps.gpu_available() else "cpu"
            hop16 = 80  # 5 ms
            with torch.no_grad():
                f0_t, per_t = torchcrepe.predict(
                    torch.from_numpy(y16).float().unsqueeze(0),
                    sr16,
                    hop_length=hop16,
                    fmin=FMIN,
                    fmax=FMAX,
                    model="full" if device == "cuda" else "tiny",
                    batch_size=512,
                    device=device,
                    return_periodicity=True,
                )
            f0_c = f0_t.squeeze(0).cpu().numpy().astype(np.float64)
            per = per_t.squeeze(0).cpu().numpy().astype(np.float64)
            t_c = np.arange(len(f0_c)) * (hop16 / sr16)
            f0 = np.interp(frame_times, t_c, f0_c)
            per_i = np.interp(frame_times, t_c, per)
            voiced = (per_i > 0.5) & (f0 > FMIN * 0.9) & np.isfinite(f0)
            if voiced.any():
                return np.where(np.isfinite(f0), f0, 0.0), voiced
        except Exception as exc:  # any torch trouble -> pyin baseline
            log.debug("torchcrepe failed, falling back to pyin: %s", exc)

    # librosa.pyin baseline ---------------------------------------------------
    import librosa

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        y.astype(np.float32),
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME,
        hop_length=HOP,
    )
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)
    return np.where(np.isfinite(f0), f0, 0.0), voiced


# ---------------------------------------------------------------------------
# note segmentation
# ---------------------------------------------------------------------------

def _voiced_runs(voiced: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = len(voiced)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _segment_notes(
    midi: np.ndarray, voiced: np.ndarray, sustain_frames: int, min_note_frames: int
) -> list[tuple[int, int]]:
    """Split voiced runs into notes where the (median-filtered) pitch jumps
    > JUMP_SEMITONES sustained for >= sustain_frames."""
    notes: list[list[int]] = []
    for run_s, run_e in _voiced_runs(voiced):
        s = run_s
        while s < run_e:
            ref = float(np.median(midi[s:min(s + 8, run_e)]))
            e = s + 1
            while e < run_e:
                if abs(midi[e] - ref) > JUMP_SEMITONES:
                    k_end = min(e + sustain_frames, run_e)
                    seg = midi[e:k_end]
                    if len(seg) >= sustain_frames and bool(
                        np.all(np.abs(seg - ref) > JUMP_SEMITONES)
                    ):
                        break  # sustained jump -> new note starts at e
                # track slow drift while the note is young
                if e - s <= 24:
                    ref = float(np.median(midi[s:e + 1]))
                e += 1
            notes.append([s, e])
            s = e
    # merge fragments shorter than min_note_frames into the preceding note
    merged: list[list[int]] = []
    for s, e in notes:
        if merged and (e - s) < min_note_frames and merged[-1][1] == s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e > s]


# ---------------------------------------------------------------------------
# correction core
# ---------------------------------------------------------------------------

def _correct_chop(
    y: np.ndarray, sr: int, pcs: list[int], strength: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pitch-correct one mono chop. Returns (audio, voiced_frames_or_None).

    Length is preserved exactly; unvoiced/breath regions are untouched.
    """
    from scipy.signal import lfilter, medfilt

    out = np.ascontiguousarray(y, dtype=np.float32).copy()
    n = len(out)
    if n < FRAME:
        return out, None

    f0, voiced = _track_f0(out, sr)
    if int(voiced.sum()) < 4:
        return out, voiced

    midi = np.zeros(len(f0), dtype=np.float64)
    midi[voiced] = 69.0 + 12.0 * np.log2(np.maximum(f0[voiced], 1e-6) / 440.0)

    # median-filter pitch inside each voiced run (kills octave blips)
    midi_s = midi.copy()
    for s, e in _voiced_runs(voiced):
        if e - s >= 5:
            midi_s[s:e] = medfilt(midi[s:e], kernel_size=5)

    sustain_frames = max(int(round(JUMP_SUSTAIN_SEC * sr / HOP)), 2)
    min_note_frames = max(int(round(0.030 * sr / HOP)), 2)
    notes = _segment_notes(midi_s, voiced, sustain_frames, min_note_frames)
    if not notes:
        return out, voiced

    # per-frame corrected pitch curve --------------------------------------
    strength = float(min(max(strength, 0.0), 1.0))
    flatten = 0.0 if strength <= HARD_TUNE_KNEE else (
        (strength - HARD_TUNE_KNEE) / (1.0 - HARD_TUNE_KNEE)
    )
    shift_curve = np.zeros(len(midi_s), dtype=np.float64)
    for s, e in notes:
        note_median = float(np.median(midi_s[s:e]))
        target = _snap_to_scale(note_median, pcs)
        corrected = midi_s[s:e] + strength * (target - note_median)
        if flatten > 0.0:
            # hard tune: progressively flatten intra-note drift/vibrato
            corrected = (1.0 - flatten) * corrected + flatten * target
        shift_curve[s:e] = corrected - midi_s[s:e]

    # smooth note transitions: one-pole, tau 120 ms (natural) -> 5 ms (hard)
    tau = 0.120 - (0.120 - 0.005) * strength
    alpha = math.exp(-HOP / (sr * max(tau, 1e-3)))
    shift_smooth = lfilter([1.0 - alpha], [1.0, -alpha], shift_curve)

    # apply per NOTE: constant shift = mean of the smoothed correction curve
    src = out.copy()  # pristine source: every segment is shifted from this
    cross = max(int(CROSSFADE_SEC * sr), 8)
    ctx = max(int(0.045 * sr), cross)  # extra STFT context for clean shifting
    for s, e in notes:
        semis = float(np.mean(shift_smooth[s:e]))
        if not math.isfinite(semis) or abs(semis) < MIN_SHIFT_SEMITONES:
            continue
        s0 = max(s * HOP - cross, 0)
        e0 = min(e * HOP + cross, n)
        seg_len = e0 - s0
        if seg_len < 64:
            continue
        a0 = max(s * HOP - ctx, 0)
        b0 = min(e * HOP + ctx, n)
        shifted_full = core_audio.pitch_shift(src[a0:b0].astype(np.float32), semis, sr)
        if len(shifted_full) != b0 - a0:  # guard: keep exact length (no drift)
            if len(shifted_full) > b0 - a0:
                shifted_full = shifted_full[: b0 - a0]
            else:
                shifted_full = np.pad(shifted_full, (0, (b0 - a0) - len(shifted_full)))
        shifted = shifted_full[s0 - a0:e0 - a0]
        # trapezoid blend window: 10 ms crossfade against existing content
        w = np.ones(seg_len, dtype=np.float32)
        f = min(cross, seg_len // 2)
        if f > 1:
            if s0 > 0:
                w[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
            if e0 < n:
                w[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
        out[s0:e0] = out[s0:e0] * (1.0 - w) + shifted * w

    if flatten > 0.0:
        # Hard tune: continuously flatten intra-note pitch drift / vibrato via
        # a smooth time-varying resampling pass (zero-mean residual per note,
        # so cumulative timing drift stays negligible — T-Pain snap at 100).
        out = _flatten_intranote(out, sr, midi_s, notes, flatten, alpha)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, voiced


def _flatten_intranote(
    x: np.ndarray, sr: int, midi_s: np.ndarray, notes: list[tuple[int, int]],
    flatten: float, alpha: float,
) -> np.ndarray:
    """Remove `flatten` x (intra-note pitch deviation) by variable-rate resampling.

    The residual is ~zero-mean inside each note (deviation from the note
    median), so the read-position drift is bounded by a fraction of a vibrato
    cycle; a final linear rescale pins the output length exactly (pitch error
    of that rescale is a few cents at most). Length preserved exactly.
    """
    from scipy.signal import lfilter

    n = len(x)
    if n < 32:
        return x
    deviation = np.zeros(len(midi_s), dtype=np.float64)
    for s, e in notes:
        med = float(np.median(midi_s[s:e]))
        deviation[s:e] = flatten * (midi_s[s:e] - med)
    deviation = np.clip(deviation, -1.5, 1.5)
    deviation = lfilter([1.0 - alpha], [1.0, -alpha], deviation)
    dev_samples = np.interp(
        np.arange(n, dtype=np.float64),
        np.arange(len(deviation), dtype=np.float64) * HOP,
        deviation,
    )
    rate = 2.0 ** (-dev_samples / 12.0)  # local read speed removes the residual
    pos = np.cumsum(rate)
    pos -= pos[0]
    if pos[-1] <= 0:
        return x
    pos *= (n - 1) / pos[-1]  # exact length, ~cents-level uniform pitch cost
    return np.interp(pos, np.arange(n, dtype=np.float64), x).astype(np.float32)


# ---------------------------------------------------------------------------
# de-breath
# ---------------------------------------------------------------------------

def _debreath(y: np.ndarray, sr: int, voiced: np.ndarray) -> np.ndarray:
    """Subtle -9 dB dip on unvoiced, high-centroid, low-energy frames."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    if y.size < FRAME or voiced is None or not voiced.any():
        return y
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=1024, hop_length=HOP))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    frame_rms = librosa.feature.rms(S=S)[0]
    m = min(len(voiced), len(centroid), len(frame_rms))
    if m < 4:
        return y
    voiced_rms = frame_rms[:m][voiced[:m]]
    if voiced_rms.size == 0:
        return y
    energy_thresh = float(np.median(voiced_rms)) * 0.35
    mask = (
        (~voiced[:m])
        & (centroid[:m] > 3500.0)
        & (frame_rms[:m] < energy_thresh)
        & (frame_rms[:m] > 1e-5)
    )
    if not mask.any():
        return y
    gains = np.where(mask, core_audio.db_to_lin(DEBREATH_GAIN_DB), 1.0)
    gains = uniform_filter1d(gains.astype(np.float64), size=5, mode="nearest")
    sample_gain = np.interp(
        np.arange(len(y), dtype=np.float64), np.arange(m, dtype=np.float64) * HOP, gains
    )
    return (y * sample_gain).astype(np.float32)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def tune_vocals(project, arrangement: dict, grid: dict, retune_speed: int, progress) -> dict:
    """Pitch-correct every placed chop to the grid key scale (spec 4.8).

    Writes vocal_tuned/t###.wav per placement and sets placement["tuned_file"].
    Returns the (mutated) arrangement; the caller persists it.
    """
    arrangement = arrangement or {}
    placements = arrangement.get("placements") or []
    todo = [p for p in placements if p.get("chop_file")]
    if not todo:
        progress(1.0, "No vocal chops to tune")
        return arrangement

    pcs = _scale_pitch_classes(grid)
    strength = min(max(int(retune_speed), 0), 100) / 100.0
    tuned_dir = project.tuned_dir  # ensures dir exists
    total = len(todo)

    for i, placement in enumerate(todo):
        role = str(placement.get("role", "vocal"))
        progress(i / total, f"Tuning {role} chop {i + 1}/{total}…")

        chop_path = project.dir / str(placement["chop_file"])
        if not chop_path.exists():
            raise PipelineError(
                f"Vocal chop missing: {placement['chop_file']} — re-run arrangement."
            )
        y, sr = core_audio.load_audio(chop_path, mono=True)
        if y.size == 0:
            raise PipelineError(f"Vocal chop is empty: {placement['chop_file']}")

        voiced = None
        try:
            tuned, voiced = _correct_chop(y, sr, pcs, strength)
        except PipelineError:
            raise
        except Exception as exc:
            # Never fail the whole render on one stubborn chop: pass through.
            log.warning("pitch correction failed for %s: %s", chop_path.name, exc)
            tuned = y.astype(np.float32).copy()

        if retune_speed > 0:
            try:
                tuned = _debreath(tuned, sr, voiced)
            except Exception as exc:
                log.debug("de-breath skipped for %s: %s", chop_path.name, exc)

        if placement.get("bridge_fx"):
            # spec 4.6: bridge contrast — extra −2 semitone shift
            tuned = core_audio.pitch_shift(tuned, -2.0, sr)
            if len(tuned) != len(y):  # length guard after global shift
                if len(tuned) > len(y):
                    tuned = tuned[: len(y)]
                else:
                    tuned = np.pad(tuned, (0, len(y) - len(tuned)))

        tuned = np.nan_to_num(tuned, nan=0.0, posinf=0.0, neginf=0.0)
        tuned = np.clip(tuned, -1.0, 1.0).astype(np.float32)

        pid = int(placement.get("id", i))
        name = f"t{pid:03d}.wav"
        core_audio.save_wav(tuned_dir / name, tuned, sr, subtype="FLOAT")
        placement["tuned_file"] = f"vocal_tuned/{name}"

    progress(1.0, f"Tuned {total} vocal chop{'s' if total != 1 else ''}")
    return arrangement
