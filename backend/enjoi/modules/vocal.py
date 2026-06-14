"""Vocal upload & analysis (Module: vocal, spec §4.5).

process_vocal: load the one-take upload → pre-clean (noise reduction +
silence-aware normalization) → save vocal_raw.wav → transcribe with
faster-whisper (optional; energy-only mode when unavailable) → phrase
segmentation (energy gating with hysteresis + word-timestamp gaps) →
per-phrase features (rms, f0, pitch height, vibrato, brightness).

Writes vocal_analysis.json (with "sections" empty — score.py fills them)
and returns the analysis dict per docs/API_CONTRACT.md.

Heavy libraries (librosa, faster_whisper, noisereduce, torchcrepe) are only
imported inside functions / via deps.optional_import.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config, deps
from ..core.errors import PipelineError
from ..core.storage import write_json

SR = config.SAMPLE_RATE

_FRAME = 2048
_HOP = 512

_MIN_TAKE_SEC = 5.0
_MAX_TAKE_SEC = 12 * 60.0           # cap: 12 minutes

_MIN_SILENCE_SEC = 0.35             # gaps shorter than this never split phrases
_MIN_PHRASE_SEC = 1.0
_MAX_PHRASE_SEC = 12.0
_WORD_GAP_SEC = 0.45                # word-timestamp gap that forces a boundary
_SNAP_TOL_SEC = 0.30                # energy boundary → word boundary snap window
_EDGE_PAD_SEC = 0.06                # padding around detected voiced regions

_TARGET_RMS_DB = -18.0              # normalization target (95th-pct voiced RMS)
_PEAK_CAP_DB = -1.0

ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def process_vocal(project, uploaded_path: Path, progress: ProgressFn) -> dict:
    """Resample → clean → transcribe → phrase segmentation. Returns analysis dict."""
    progress(0.02, "Loading vocal take…")
    audio, sr = core_audio.load_audio(uploaded_path, sr=SR, mono=True)

    if audio.size == 0:
        raise PipelineError("The uploaded file contains no audio.")
    duration = len(audio) / sr
    if duration < _MIN_TAKE_SEC:
        raise PipelineError(
            f"The vocal take is too short ({duration:.1f} s) — please upload at least 5 seconds of singing."
        )
    if duration > _MAX_TAKE_SEC:
        progress(0.04, "Take longer than 12 minutes — trimming to the first 12 minutes")
        audio = audio[: int(_MAX_TAKE_SEC * sr)]
        duration = len(audio) / sr
    if float(np.max(np.abs(audio))) < 1e-5:
        raise PipelineError("The uploaded take appears to be silent.")

    # ---- pre-clean ---------------------------------------------------------
    progress(0.08, "Cleaning up the recording…")
    audio = _denoise(audio, sr)
    audio = _normalize_silence_aware(audio)

    progress(0.14, "Saving working file…")
    core_audio.save_wav(project.vocal_raw_path, audio, sr=sr, subtype="PCM_16")

    # ---- transcription (optional) -------------------------------------------
    progress(0.18, "Transcribing lyrics…")
    lyrics, words = _transcribe(project.vocal_raw_path, progress)

    # ---- phrase segmentation -------------------------------------------------
    progress(0.55, "Segmenting phrases…")
    spans = _segment_phrases(audio, words)
    if not spans:
        raise PipelineError("Couldn't find any singing in the take — it appears to be silent.")

    # ---- per-phrase features ---------------------------------------------------
    phrases = _phrase_features(audio, sr, spans, words, progress)

    # Drop laughs / mistakes / non-sung noise so they're never placed in the song.
    # Keep everything if the gate would leave too little to work with.
    usable = [p for p in phrases if p.get("usable", True)]
    if len(usable) >= max(2, len(phrases) // 2):
        dropped = len(phrases) - len(usable)
        phrases = usable
        if dropped:
            progress(0.93, f"Skipped {dropped} unusable take(s) (laughs/fumbles)")
    for i, p in enumerate(phrases):  # re-index so phrase ids stay contiguous
        p["id"] = i

    analysis = {
        "file": "vocal_raw.wav",
        "duration_sec": round(duration, 3),
        "lyrics": lyrics,
        "words": words,
        "phrases": phrases,
        "sections": [],  # filled by score.score_sections
        "weights": dict(config.IMPACT_WEIGHTS_DEFAULT),
    }
    write_json(project.vocal_analysis_path, analysis)
    progress(1.0, f"Vocal analyzed — {len(phrases)} phrases found")
    return analysis


# ---------------------------------------------------------------------------
# pre-clean
# ---------------------------------------------------------------------------

def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mild non-stationary noise reduction when noisereduce is available."""
    nr = deps.optional_import("noisereduce")
    if nr is None:
        return audio
    try:
        out = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.5)
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        return audio  # cleaning is best-effort, never fatal


def _normalize_silence_aware(audio: np.ndarray) -> np.ndarray:
    """Normalize by the 95th-percentile RMS of voiced frames to ~-18 dBFS RMS,
    peak-capped at -1 dB. Silence does not drag the measurement down."""
    rms_f = _frame_rms(audio)
    if rms_f.size == 0:
        return audio
    on, _ = _gate_thresholds(rms_f)
    voiced = rms_f[rms_f >= on]
    ref = float(np.percentile(voiced if voiced.size else rms_f, 95))
    if ref < 1e-7:
        return audio
    gain = core_audio.db_to_lin(_TARGET_RMS_DB) / ref
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    cap = core_audio.db_to_lin(_PEAK_CAP_DB)
    if peak > cap:
        out *= cap / peak
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------

def _transcribe(wav_path: Path, progress: ProgressFn) -> tuple[str, list[dict]]:
    """faster-whisper "small" with word timestamps. Missing/failing whisper →
    ("", []) and the pipeline continues in energy-only mode."""
    fw = deps.optional_import("faster_whisper")
    if fw is None:
        progress(0.50, "Speech model not installed — continuing without lyrics")
        return "", []
    try:
        if deps.gpu_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        # Model size is env-configurable so a CPU server can use the faster
        # "base" while a GPU box uses "small"/"medium" for accuracy.
        model_size = os.environ.get(
            "ENJOI_WHISPER_MODEL", "small" if deps.gpu_available() else "base"
        ).strip() or "base"
        model = fw.WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(config.models_dir()),
        )
        segments, _info = model.transcribe(str(wav_path), word_timestamps=True)
        words: list[dict] = []
        texts: list[str] = []
        for seg in segments:  # generator — consumes lazily
            if seg.text:
                texts.append(seg.text.strip())
            for w in seg.words or []:
                token = (w.word or "").strip()
                if token:
                    words.append({"w": token, "start": round(float(w.start), 3), "end": round(float(w.end), 3)})
        words.sort(key=lambda w: w["start"])
        return " ".join(texts).strip(), words
    except Exception:
        progress(0.50, "Transcription failed — continuing without lyrics")
        return "", []


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def _frame_rms(audio: np.ndarray, frame: int = _FRAME, hop: int = _HOP) -> np.ndarray:
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    n = 1 + (len(audio) - frame) // hop
    csum = np.concatenate([[0.0], np.cumsum(np.square(audio, dtype=np.float64))])
    starts = np.arange(n) * hop
    energy = csum[starts + frame] - csum[starts]
    return np.sqrt(np.maximum(energy / frame, 0.0))


def _gate_thresholds(rms_f: np.ndarray) -> tuple[float, float]:
    """Hysteresis on/off thresholds relative to the take's voiced level and floor."""
    nz = rms_f[rms_f > 0]
    if nz.size == 0:
        return 1e-4, 5e-5
    peak = float(np.percentile(nz, 95))
    floor = float(np.percentile(rms_f, 10))
    on = max(1e-4, floor * 2.5, peak * 0.12)
    off = max(floor * 1.5, on * 0.45)
    off = min(off, on * 0.6)
    return on, off


def _frame_time(i: int) -> float:
    return (i * _HOP + _FRAME / 2) / SR


def _segment_phrases(audio: np.ndarray, words: list[dict]) -> list[tuple[float, float]]:
    """Energy gating with hysteresis + word-gap boundaries → phrase spans (sec)."""
    rms_f = _frame_rms(audio)
    on, off = _gate_thresholds(rms_f)
    total = len(audio) / SR

    # --- hysteresis gating → raw voiced regions
    segs: list[tuple[float, float]] = []
    active = False
    start_i = 0
    for i, v in enumerate(rms_f):
        if not active and v >= on:
            active, start_i = True, i
        elif active and v < off:
            active = False
            segs.append((_frame_time(start_i) - _FRAME / (2 * SR), _frame_time(i)))
    if active:
        segs.append((_frame_time(start_i) - _FRAME / (2 * SR), total))
    segs = [(max(0.0, s - _EDGE_PAD_SEC), min(total, e + _EDGE_PAD_SEC)) for s, e in segs]

    # --- merge gaps shorter than the minimum silence
    segs = _merge_close(segs, _MIN_SILENCE_SEC)

    # --- word-timestamp refinement
    if words:
        segs = _snap_to_words(segs, words)
        segs = _split_by_word_gaps(segs, words)
        segs = _merge_close(segs, _MIN_SILENCE_SEC)

    # --- enforce min phrase length (merge into closest neighbor, else drop)
    segs = _enforce_min_length(segs)

    # --- enforce max phrase length (split at weakest interior gap)
    out: list[tuple[float, float]] = []
    for seg in segs:
        out.extend(_split_long(seg, rms_f))
    out.sort(key=lambda se: se[0])
    return [(round(s, 3), round(e, 3)) for s, e in out if e - s > 0.2]


def _merge_close(segs: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda se: se[0])
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < min_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _snap_to_words(segs: list[tuple[float, float]], words: list[dict]) -> list[tuple[float, float]]:
    starts = np.array([w["start"] for w in words], dtype=np.float64)
    ends = np.array([w["end"] for w in words], dtype=np.float64)
    snapped: list[tuple[float, float]] = []
    for s, e in segs:
        i = int(np.argmin(np.abs(starts - s)))
        if abs(starts[i] - s) <= _SNAP_TOL_SEC:
            s = float(starts[i])
        j = int(np.argmin(np.abs(ends - e)))
        if abs(ends[j] - e) <= _SNAP_TOL_SEC:
            e = float(ends[j])
        if e - s > 0.2:
            snapped.append((s, e))
    # resolve any overlaps introduced by snapping
    out: list[tuple[float, float]] = []
    for s, e in snapped:
        if out and s < out[-1][1]:
            s = out[-1][1]
        if e - s > 0.2:
            out.append((s, e))
    return out


def _split_by_word_gaps(segs: list[tuple[float, float]], words: list[dict]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in segs:
        inside = [w for w in words if w["start"] >= s - 0.05 and w["end"] <= e + 0.05]
        cur = s
        for w, nw in zip(inside, inside[1:]):
            gap = nw["start"] - w["end"]
            if gap > _WORD_GAP_SEC and w["end"] - cur >= 0.3:
                out.append((cur, w["end"]))
                cur = nw["start"]
        if e - cur > 0.2:
            out.append((cur, e))
    return out


def _enforce_min_length(segs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(segs) <= 1:
        return list(segs)
    result: list[tuple[float, float]] = []
    i = 0
    while i < len(segs):
        s, e = segs[i]
        if e - s >= _MIN_PHRASE_SEC:
            result.append((s, e))
            i += 1
            continue
        gap_prev = s - result[-1][1] if result else math.inf
        gap_next = segs[i + 1][0] - e if i + 1 < len(segs) else math.inf
        if gap_prev <= gap_next and gap_prev <= 0.75 and result:
            result[-1] = (result[-1][0], e)
        elif gap_next <= 0.75 and i + 1 < len(segs):
            segs[i + 1] = (s, segs[i + 1][1])
        # else: too isolated AND too short → drop
        i += 1
    return result


def _split_long(seg: tuple[float, float], rms_f: np.ndarray) -> list[tuple[float, float]]:
    s, e = seg
    if e - s <= _MAX_PHRASE_SEC:
        return [seg]
    margin = 0.5
    i0 = max(0, int(((s + margin) * SR - _FRAME / 2) / _HOP))
    i1 = min(len(rms_f), int(((e - margin) * SR - _FRAME / 2) / _HOP))
    if i1 - i0 >= 2:
        k = i0 + int(np.argmin(rms_f[i0:i1]))
        t = _frame_time(k)
    else:
        t = (s + e) / 2
    if not (s + 0.4 < t < e - 0.4):
        t = (s + e) / 2
    return _split_long((s, t), rms_f) + _split_long((t, e), rms_f)


# ---------------------------------------------------------------------------
# per-phrase features
# ---------------------------------------------------------------------------

def _phrase_features(
    audio: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    words: list[dict],
    progress: ProgressFn,
) -> list[dict]:
    n = len(spans)
    f0_per_phrase: list[tuple[np.ndarray, float]] = []
    for k, (s, e) in enumerate(spans):
        seg = audio[int(s * sr): int(e * sr)]
        f0_per_phrase.append(_track_f0(seg, sr))
        progress(0.62 + 0.30 * (k + 1) / max(n, 1), f"Analyzing phrase {k + 1}/{n}…")

    # global voiced-f0 distribution → pitch_height percentile positions
    all_voiced = np.concatenate(
        [f0[np.isfinite(f0)] for f0, _ in f0_per_phrase]
        + [np.zeros(0, dtype=np.float64)]
    )
    global_sorted = np.sort(all_voiced)

    phrases = []
    for k, (s, e) in enumerate(spans):
        seg = audio[int(s * sr): int(e * sr)]
        f0, hop_dur = f0_per_phrase[k]
        voiced = f0[np.isfinite(f0)]

        rms_v = core_audio.rms(seg)
        f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
        if voiced.size >= 2:
            p5, p95 = np.percentile(voiced, [5, 95])
            f0_range = 12.0 * math.log2(max(p95, 1e-6) / max(p5, 1e-6))
        else:
            f0_range = 0.0
        if voiced.size and global_sorted.size:
            pos = np.searchsorted(global_sorted, voiced) / max(len(global_sorted), 1)
            pitch_height = float(np.mean(pos))
        else:
            pitch_height = 0.0

        # --- musicality gate: skip laughs / fumbles / non-sung noise ---
        # A laugh or vocal mistake is erratically pitched (big frame-to-frame
        # jumps) and/or barely voiced; a sung/rapped line is more stable.
        voiced_ratio = float(voiced.size) / max(int(f0.size), 1)
        if voiced.size >= 4:
            steps = 12.0 * np.log2(np.clip(voiced[1:], 1e-6, None)
                                   / np.clip(voiced[:-1], 1e-6, None))
            jitter = float(np.median(np.abs(steps)))
        else:
            jitter = 9.9 if voiced.size < 2 else 0.0
        # generous: only reject clearly non-musical segments
        usable = (voiced_ratio >= 0.18) and (jitter <= 2.6) and (max(f0_range, 0.0) <= 22.0)

        phrases.append({
            "id": k,
            "start": s,
            "end": e,
            "text": _words_in_span(words, s, e),
            "usable": bool(usable),
            "features": {
                "rms": round(float(rms_v), 4),
                "f0_mean_hz": round(f0_mean, 1),
                "f0_range_semitones": round(max(f0_range, 0.0), 2),
                "pitch_height": round(pitch_height, 3),
                "vibrato": round(_vibrato(f0, hop_dur), 3),
                "brightness": round(_brightness(seg, sr), 3),
                "voiced_ratio": round(voiced_ratio, 3),
                "pitch_jitter": round(jitter, 3),
            },
        })
    return phrases


def _words_in_span(words: list[dict], start: float, end: float) -> str:
    toks = [w["w"] for w in words if start <= (w["start"] + w["end"]) / 2 <= end]
    return " ".join(toks)


def _track_f0(seg: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """f0 contour in Hz (NaN = unvoiced) + hop duration. torchcrepe → pyin."""
    res = _f0_torchcrepe(seg, sr)
    if res is not None:
        return res
    return _f0_pyin(seg, sr)


def _f0_torchcrepe(seg: np.ndarray, sr: int) -> tuple[np.ndarray, float] | None:
    torchcrepe = deps.optional_import("torchcrepe")
    torch = deps.optional_import("torch")
    if torchcrepe is None or torch is None or seg.size < sr // 10:
        return None
    try:
        target_sr, hop = 16000, 160  # 10 ms frames
        x = core_audio.resample(seg, sr, target_sr)
        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        device = "cuda" if deps.gpu_available() else "cpu"
        f0, periodicity = torchcrepe.predict(
            tensor, target_sr, hop, 80.0, 800.0, "tiny",
            batch_size=512, device=device, return_periodicity=True,
        )
        f0 = f0[0].cpu().numpy().astype(np.float64)
        per = periodicity[0].cpu().numpy()
        f0[per < 0.4] = np.nan
        return f0, hop / target_sr
    except Exception:
        return None


def _f0_pyin(seg: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    try:
        import librosa

        f0, _vflag, _vprob = librosa.pyin(
            seg.astype(np.float32), fmin=80.0, fmax=800.0, sr=sr,
            frame_length=_FRAME, hop_length=_HOP,
        )
        return np.asarray(f0, dtype=np.float64), _HOP / sr
    except Exception:
        return np.full(0, np.nan), _HOP / sr


def _vibrato(f0: np.ndarray, hop_dur: float) -> float:
    """Mean 4–8 Hz modulation depth (semitones / 1.5, capped at 1) over
    sustained voiced stretches >= 0.5 s."""
    if f0.size == 0 or hop_dur <= 0:
        return 0.0
    min_len = max(int(round(0.5 / hop_dur)), 4)
    depths: list[float] = []
    finite = np.isfinite(f0)
    run_start = None
    for i in range(len(f0) + 1):
        if i < len(f0) and finite[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            run = f0[run_start:i]
            if len(run) >= min_len:
                d = _vibrato_depth(run, hop_dur)
                if d is not None:
                    depths.append(min(d / 1.5, 1.0))
            run_start = None
    return float(np.mean(depths)) if depths else 0.0


def _vibrato_depth(run_hz: np.ndarray, hop_dur: float) -> float | None:
    st = 12.0 * np.log2(np.maximum(run_hz, 1e-6) / 440.0)
    w = max(3, int(round(0.25 / hop_dur)) | 1)  # ~0.25 s moving average, odd
    if len(st) <= w + 4:
        trend = np.full_like(st, np.mean(st))
        resid = st - trend
    else:
        trend = np.convolve(st, np.ones(w) / w, mode="same")
        resid = (st - trend)[w // 2: len(st) - w // 2]
    if len(resid) < 8:
        return None
    win = np.hanning(len(resid))
    spec = np.abs(np.fft.rfft(resid * win))
    freqs = np.fft.rfftfreq(len(resid), hop_dur)
    band = (freqs >= 4.0) & (freqs <= 8.0)
    if not np.any(band):
        return None
    # peak sinusoid amplitude in the band (hann coherent gain = 0.5)
    amp = spec[band].max() / (len(resid) / 2) / 0.5
    return float(amp)


def _brightness(seg: np.ndarray, sr: int) -> float:
    """Mean spectral centroid normalized by 4000 Hz, capped at 1."""
    if seg.size < _FRAME:
        return 0.0
    try:
        import librosa

        cent = librosa.feature.spectral_centroid(
            y=seg.astype(np.float32), sr=sr, n_fft=_FRAME, hop_length=_HOP
        )[0]
        if cent.size == 0:
            return 0.0
        return float(min(np.mean(cent) / 4000.0, 1.0))
    except Exception:
        return 0.0
