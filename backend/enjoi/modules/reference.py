"""Reference acquisition + full MIR analysis → reference_profile.json (spec §4.2).

Downloads the chosen YouTube reference into the analysis-only ``_ref_cache``
sandbox, decodes it to 44.1 kHz WAV, and extracts every descriptor in the
contract schema: beat grid, downbeats, key, time signature, structure,
energy curve, instrumentation, groove, tags, and the uniqueness-guard
fingerprints (melody interval n-grams, per-bar chords, downbeat chroma,
spectral landmark hashes).

librosa is the required baseline; madmom / Essentia / Demucs / torchcrepe are
probed via ``deps.optional_import`` and upgrade quality when present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config, deps
from ..core.errors import PipelineError
from ..core.storage import Project, write_json

ProgressFn = Callable[[float, str], None]

_SR = 22050          # internal analysis sample rate (timestamps stay in real seconds)
_HOP = 512
_NFFT = 2048

_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def acquire_and_analyze(project: Project, url: str, progress: ProgressFn) -> dict:
    """Download reference audio into ``project.ref_cache_dir``, analyze it, and
    write + return the reference profile dict (contract schema)."""
    p = _safe_progress(progress)

    meta, src_path = _download_reference(project, url, p)        # 0.00 – 0.25

    p(0.25, "Decoding audio…")
    y44, _ = core_audio.load_audio(src_path, sr=config.SAMPLE_RATE, mono=True)
    duration = len(y44) / float(config.SAMPLE_RATE)
    if duration > config.MAX_REFERENCE_DURATION_SEC + 1.0:
        raise PipelineError(
            f"That video is {duration / 60:.1f} minutes long — references must be 10 minutes or shorter."
        )
    if duration < 15.0:
        raise PipelineError("That video is too short to use as a reference (need at least 15 seconds).")

    ref_wav = project.ref_cache_dir / "reference.wav"
    core_audio.save_wav(ref_wav, y44, config.SAMPLE_RATE, subtype="PCM_16")
    try:
        if src_path != ref_wav:
            src_path.unlink(missing_ok=True)
    except OSError:
        pass

    p(0.28, "Preparing analysis…")
    y = np.ascontiguousarray(core_audio.resample(y44, config.SAMPLE_RATE, _SR), dtype=np.float32)
    del y44

    profile = _analyze_audio(y, _SR, duration, ref_wav, meta, p)  # 0.30 – 0.99
    profile["source"] = {
        "title": meta.get("title", ""),
        "channel": meta.get("channel", ""),
        "url": meta.get("url", url),
        "video_id": meta.get("video_id", ""),
        "duration_sec": round(duration, 2),
        "thumbnail_url": meta.get("thumbnail_url", ""),
    }
    profile["ref_audio"] = "_ref_cache/reference.wav"
    profile = _jsonable(profile)
    write_json(project.reference_profile_path, profile)
    p(1.0, "Reference analysis complete")
    return profile


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------

def _download_reference(project: Project, url: str, p: ProgressFn) -> tuple[dict, Path]:
    """yt-dlp bestaudio download into the ref-cache sandbox. Returns (meta, path)."""
    yt_dlp = deps.optional_import("yt_dlp")
    if yt_dlp is None:
        raise PipelineError("yt-dlp is not installed — cannot fetch the reference audio.")

    cache = project.ref_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("source.*"):
        try:
            stale.unlink()
        except OSError:
            pass

    p(0.01, "Looking up video…")

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            frac = min(done / total, 1.0) if total else 0.0
            p(0.05 + 0.19 * frac, "Downloading reference audio…")
        elif d.get("status") == "finished":
            p(0.24, "Download complete")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 20,
        "progress_hooks": [hook],
        "cachedir": False,
    }
    ffmpeg = deps.ffmpeg_path()
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and info.get("entries"):
                entries = [e for e in info["entries"] if e]
                if not entries:
                    raise PipelineError("No playable video was found at that link.")
                info = entries[0]
            if not info:
                raise PipelineError("Could not read video information from that link.")
            if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                raise PipelineError("Live streams can't be used as a reference — pick a regular video.")
            dur = info.get("duration")
            if dur and float(dur) > config.MAX_REFERENCE_DURATION_SEC:
                raise PipelineError(
                    f"That video is {float(dur) / 60:.1f} minutes long — references must be 10 minutes or shorter."
                )
            ydl.download([info.get("webpage_url") or url])
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "Could not download the reference audio — check your internet connection and try again."
        ) from exc

    files = sorted(
        (f for f in cache.glob("source.*") if f.suffix not in (".part", ".ytdl")),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    if not files:
        raise PipelineError("The download finished but no audio file was produced. Try another video.")

    meta = {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "url": info.get("webpage_url") or url,
        "video_id": info.get("id") or "",
        "duration_sec": float(info.get("duration") or 0.0),
        "thumbnail_url": info.get("thumbnail") or "",
        "tags": list(info.get("tags") or []),
        "categories": list(info.get("categories") or []),
    }
    return meta, files[0]


# ---------------------------------------------------------------------------
# analysis driver
# ---------------------------------------------------------------------------

def _analyze_audio(y: np.ndarray, sr: int, duration: float, ref_wav: Path,
                   meta: dict, p: ProgressFn) -> dict:
    """Full MIR analysis of mono audio ``y`` at ``sr``; timestamps in real seconds."""
    import librosa

    p(0.30, "Tracking beats…")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    bpm, beat_times = _track_beats(y, sr, onset_env, ref_wav)

    # Shared spectral workspace (one STFT for everything downstream).
    D = librosa.stft(y, n_fft=_NFFT, hop_length=_HOP)
    S = np.abs(D).astype(np.float32)
    n_frames = S.shape[1]
    onset_env = _fit_length(onset_env, n_frames)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    low_env = (S[freqs < 150.0] ** 2).sum(axis=0)

    p(0.40, "Finding downbeats…")
    downbeats, time_signature = _track_downbeats(ref_wav, beat_times, onset_env, low_env, bpm)
    bar_spans = _bar_spans(downbeats, duration)

    p(0.46, "Separating harmonic content…")
    H, P_mag = librosa.decompose.hpss(S)
    mask_h = H / (S + 1e-9)
    y_harm = librosa.istft(D * mask_h, hop_length=_HOP, length=len(y)).astype(np.float32)
    e_harm = float((H.astype(np.float64) ** 2).sum())
    e_perc = float((P_mag.astype(np.float64) ** 2).sum())
    del D, P_mag

    p(0.52, "Detecting key…")
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=_HOP)
    beat_frames = np.clip(
        librosa.time_to_frames(beat_times, sr=sr, hop_length=_HOP), 0, max(n_frames - 1, 0)
    )
    key = _detect_key(ref_wav, chroma, beat_frames)

    p(0.58, "Mapping song structure…")
    rms_env = _fit_length(librosa.feature.rms(S=S)[0], n_frames)
    band_vocal = (freqs >= 200.0) & (freqs <= 4000.0)
    vocal_env = (H[band_vocal] ** 2).sum(axis=0)
    structure = _segment_structure(
        y, sr, chroma, beat_frames, beat_times, downbeats, duration,
        rms_env, vocal_env, frame_times, bpm,
    )

    p(0.70, "Measuring energy curve…")
    energy_curve = _energy_curve(bar_spans, rms_env, onset_env, frame_times)

    p(0.74, "Profiling instrumentation…")
    instrumentation = _instrumentation(ref_wav, S, H, freqs, e_harm, e_perc)

    p(0.80, "Analyzing groove…")
    groove = _groove(
        S, freqs, onset_env, frame_times, beat_times, bar_spans, duration,
        sr, instrumentation.get("drums", 0.5),
    )

    # Fingerprints MUST be computed with the exact same extractors the
    # Uniqueness Guard applies to generated candidates (unique.py canonical
    # helpers), otherwise the audit compares unlike representations.
    from . import unique as _uq

    del y_harm
    y_fp = y[: int(_uq.MAX_ANALYSIS_SEC * sr)]
    try:
        bpb_fp = max(2, min(12, int(time_signature.split("/")[0])))
    except (ValueError, IndexError):
        bpb_fp = 4

    p(0.85, "Fingerprinting melody…")
    melody_ngrams = sorted(_uq.interval_ngrams(_uq.melody_midi_sequence(y_fp, sr)))

    p(0.93, "Fingerprinting harmony…")
    chord_sequence = _uq.chords_per_bar(y_fp, sr, bpb_fp)
    chroma_downbeat = [
        [round(float(v), 4) for v in row] for row in _uq.downbeat_chroma(y_fp, sr, bpb_fp)
    ]

    p(0.96, "Computing audio fingerprint…")
    fp_arr, _fp_times = _uq.landmark_hashes(y_fp, sr)
    fp_hashes = [int(h) for h in fp_arr]
    del S, H

    energy_mean = (
        float(np.mean(energy_curve["per_bar_rms"])) if energy_curve["per_bar_rms"] else 0.5
    )
    genre_tags, mood_tags = _tags(meta, bpm, key, energy_mean, groove, instrumentation)

    return {
        "duration_sec": round(duration, 2),
        "bpm": round(float(bpm), 2),
        "beat_times": [round(float(t), 3) for t in beat_times],
        "downbeats": [round(float(t), 3) for t in downbeats],
        "time_signature": time_signature,
        "key": key,
        "structure": structure,
        "energy_curve": energy_curve,
        "instrumentation": instrumentation,
        "groove": groove,
        "genre_tags": genre_tags,
        "mood_tags": mood_tags,
        "fingerprints": {
            "melody_interval_ngrams": melody_ngrams,
            "chord_sequence": chord_sequence,
            "chroma_downbeat": chroma_downbeat,
            "fp_hashes": fp_hashes,
        },
    }


# ---------------------------------------------------------------------------
# beats / downbeats
# ---------------------------------------------------------------------------

def _octave_correct(bpm: float) -> float:
    """Fold BPM into the 60–200 range by octave doubling/halving."""
    bpm = float(bpm) if bpm and np.isfinite(bpm) and bpm > 0 else 120.0
    while bpm < 60.0:
        bpm *= 2.0
    while bpm > 200.0:
        bpm /= 2.0
    return bpm


def _track_beats(y: np.ndarray, sr: int, onset_env: np.ndarray,
                 ref_wav: Path) -> tuple[float, np.ndarray]:
    """Beat grid: madmom DBN tracker if available, else librosa beat_track."""
    import librosa

    beat_times = _madmom_beats(ref_wav)
    if beat_times is None:
        tempo, frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=_HOP, start_bpm=120.0, trim=False
        )
        t0 = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
        corrected = _octave_correct(t0)
        if abs(corrected - t0) > 1.0:  # re-track with the corrected tempo hint
            tempo, frames = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=_HOP, start_bpm=corrected, trim=False
            )
        beat_times = librosa.frames_to_time(frames, sr=sr, hop_length=_HOP)

    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 4:  # degenerate audio — synthesize a 120 BPM grid
        end = max(len(y) / sr, 1.0)
        beat_times = np.arange(0.0, end, 0.5)
    ibis = np.diff(beat_times)
    ibis = ibis[ibis > 1e-3]
    bpm = _octave_correct(60.0 / float(np.median(ibis))) if len(ibis) else 120.0
    return float(bpm), beat_times


def _madmom_beats(ref_wav: Path) -> np.ndarray | None:
    if deps.optional_import("madmom") is None:
        return None
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

        act = RNNBeatProcessor()(str(ref_wav))
        beats = DBNBeatTrackingProcessor(fps=100, min_bpm=55.0, max_bpm=215.0)(act)
        return np.asarray(beats, dtype=float) if len(beats) >= 4 else None
    except Exception:
        return None


def _track_downbeats(ref_wav: Path, beat_times: np.ndarray, onset_env: np.ndarray,
                     low_env: np.ndarray, bpm: float) -> tuple[np.ndarray, str]:
    """Downbeat grid + time signature; madmom DBN → phase/meter heuristic fallback."""
    mm = _madmom_downbeats(ref_wav)
    if mm is not None:
        downbeats, meter = mm
    else:
        downbeats, meter = _heuristic_downbeats(beat_times, onset_env, low_env)

    time_signature = "4/4"
    if meter == 3:
        if bpm >= 150.0 and len(downbeats) >= 4:
            time_signature = "6/8"           # fast triple feel → compound meter
            downbeats = downbeats[::2]       # bars span 6 tracked beats
        else:
            time_signature = "3/4"
    if len(downbeats) == 0:
        downbeats = beat_times[:1] if len(beat_times) else np.array([0.0])
    return np.asarray(downbeats, dtype=float), time_signature


def _madmom_downbeats(ref_wav: Path) -> tuple[np.ndarray, int] | None:
    if deps.optional_import("madmom") is None:
        return None
    try:
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        act = RNNDownBeatProcessor()(str(ref_wav))
        res = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        if res is None or len(res) < 4:
            return None
        res = np.asarray(res, dtype=float)
        downbeats = res[res[:, 1] == 1][:, 0]
        meter = int(res[:, 1].max())
        return (downbeats, meter) if len(downbeats) >= 2 else None
    except Exception:
        return None


def _heuristic_downbeats(beat_times: np.ndarray, onset_env: np.ndarray,
                         low_env: np.ndarray) -> tuple[np.ndarray, int]:
    """Assume 4/4 unless onset autocorrelation strongly suggests 3; pick the
    downbeat phase maximizing onset strength + low-frequency energy."""
    import librosa

    if len(beat_times) < 8:
        return beat_times[:1], 4

    n = len(onset_env)
    bf = np.clip(
        librosa.time_to_frames(beat_times, sr=_SR, hop_length=_HOP), 0, max(n - 1, 0)
    )
    strength = onset_env[bf] / (float(onset_env[bf].max()) + 1e-9)
    low = low_env[np.clip(bf, 0, len(low_env) - 1)]
    low = low / (float(low.max()) + 1e-9)
    score_vec = strength + low

    def autocorr(lag: int) -> float:
        v = score_vec - score_vec.mean()
        if len(v) <= lag:
            return 0.0
        denom = float(np.sum(v * v)) + 1e-9
        return float(np.sum(v[:-lag] * v[lag:])) / denom

    def best_phase(meter: int) -> tuple[int, float]:
        best = (0, -1.0)
        for ph in range(meter):
            sel = score_vec[ph::meter]
            sc = float(sel.mean()) if len(sel) else -1.0
            if sc > best[1]:
                best = (ph, sc)
        return best

    ac3, ac4 = autocorr(3), autocorr(4)
    ph3, s3 = best_phase(3)
    ph4, s4 = best_phase(4)
    meter = 3 if (ac3 > 1.15 * ac4 and s3 > s4) else 4
    phase = ph3 if meter == 3 else ph4
    return beat_times[phase::meter], meter


def _bar_spans(downbeats: np.ndarray, duration: float) -> list[tuple[float, float]]:
    """(start, end) per bar from the downbeat grid, plus the trailing partial bar."""
    db = [float(t) for t in downbeats if 0.0 <= float(t) < duration]
    if not db:
        return [(0.0, duration)]
    spans = [(db[i], db[i + 1]) for i in range(len(db) - 1)]
    med = float(np.median([e - s for s, e in spans])) if spans else max(duration - db[0], 0.5)
    if duration - db[-1] > 0.25 * med:
        spans.append((db[-1], duration))
    return spans or [(0.0, duration)]


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------

def _detect_key(ref_wav: Path, chroma: np.ndarray, beat_frames: np.ndarray) -> dict:
    """Essentia KeyExtractor → Krumhansl-Schmuckler on beat-averaged chroma."""
    ess = _essentia_key(ref_wav)
    if ess is not None:
        return ess

    import librosa

    bf = np.unique(np.clip(beat_frames, 0, max(chroma.shape[1] - 1, 0)))
    sync = librosa.util.sync(chroma, bf, aggregate=np.mean) if len(bf) >= 2 else chroma
    mean_chroma = sync.mean(axis=1)
    if float(mean_chroma.max()) <= 1e-9:
        return {"tonic": "C", "mode": "major", "confidence": 0.05}

    scores: list[tuple[float, str, str]] = []
    for mode_name, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for tonic_idx in range(12):
            r = float(np.corrcoef(np.roll(prof, tonic_idx), mean_chroma)[0, 1])
            scores.append((r if np.isfinite(r) else -1.0, _PITCHES[tonic_idx], mode_name))
    scores.sort(key=lambda t: -t[0])
    (r1, tonic, mode), (r2, _, _) = scores[0], scores[1]
    confidence = float(np.clip(max(r1, 0.0) * 0.5 + 2.5 * (r1 - r2), 0.05, 0.99))
    return {"tonic": tonic, "mode": mode, "confidence": round(confidence, 3)}


def _essentia_key(ref_wav: Path) -> dict | None:
    es = deps.optional_import("essentia.standard")
    if es is None:
        return None
    try:
        audio16 = es.MonoLoader(filename=str(ref_wav), sampleRate=16000)()
        key, scale, strength = es.KeyExtractor()(audio16)
        tonic = _FLAT_TO_SHARP.get(str(key), str(key))
        if tonic not in _PITCHES:
            return None
        mode = "minor" if str(scale).lower() == "minor" else "major"
        return {"tonic": tonic, "mode": mode, "confidence": round(float(strength), 3)}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _zscore(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def _segment_structure(y: np.ndarray, sr: int, chroma: np.ndarray,
                       beat_frames: np.ndarray, beat_times: np.ndarray,
                       downbeats: np.ndarray, duration: float,
                       rms_env: np.ndarray, vocal_env: np.ndarray,
                       frame_times: np.ndarray, bpm: float) -> list[dict]:
    """Novelty/agglomerative segmentation on beat-synced MFCC+chroma, snapped to
    downbeats, then heuristically labeled intro/verse/prechorus/chorus/bridge/outro/inst."""
    import librosa

    if len(beat_times) < 16 or duration < 30.0 or len(downbeats) < 4:
        bars = max(1, len(_bar_spans(downbeats, duration)))
        return [{"label": "inst", "start": 0.0, "end": round(duration, 3), "bars": bars}]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=_HOP)
    n_cols = min(mfcc.shape[1], chroma.shape[1])
    bf = np.unique(np.clip(beat_frames, 0, n_cols - 1))
    fm = librosa.util.sync(mfcc[:, :n_cols], bf, aggregate=np.mean)
    fc = librosa.util.sync(chroma[:, :n_cols], bf, aggregate=np.mean)
    feat = np.vstack([_zscore(fm), _zscore(fc)])

    # Column j of the synced features starts at: 0 (j=0) or time of beat frame j-1.
    col_starts = np.concatenate(
        [[0.0], librosa.frames_to_time(bf, sr=sr, hop_length=_HOP)]
    )

    n_segs = int(np.clip(round(duration / 18.0), 6, 12))
    n_segs = max(2, min(n_segs, feat.shape[1] // 4))
    bounds = librosa.segment.agglomerative(feat, n_segs)
    bound_times = col_starts[np.clip(bounds, 0, len(col_starts) - 1)]

    bar_dur = (
        float(np.median(np.diff(downbeats))) if len(downbeats) >= 2 else 4.0 * 60.0 / bpm
    )
    min_gap = max(0.9 * bar_dur, 1.0)
    snapped: list[float] = []
    for t in sorted(float(t) for t in bound_times[1:]):
        idx = int(np.argmin(np.abs(downbeats - t)))
        cand = float(downbeats[idx]) if abs(downbeats[idx] - t) <= 2.0 else t
        if cand <= min_gap or cand >= duration - min_gap:
            continue
        if snapped and cand - snapped[-1] < min_gap:
            continue
        snapped.append(cand)
    boundaries = [0.0] + snapped + [duration]
    spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    seg_feats, seg_energy, seg_vocal = [], [], []
    for s0, e0 in spans:
        i0, i1 = np.searchsorted(col_starts, [s0, e0])
        i1 = max(int(i1), int(i0) + 1)
        seg_feats.append(feat[:, int(i0):min(int(i1), feat.shape[1])].mean(axis=1))
        j0, j1 = np.searchsorted(frame_times, [s0, e0])
        j1 = max(int(j1), int(j0) + 1)
        seg_energy.append(float(rms_env[int(j0):int(j1)].mean()))
        seg_vocal.append(float(vocal_env[int(j0):min(int(j1), len(vocal_env))].mean()))

    clusters = _cluster_segments(np.stack(seg_feats))
    labels = _label_segments(spans, seg_energy, seg_vocal, clusters, duration)

    structure = []
    for (s0, e0), label in zip(spans, labels):
        i0 = int(np.searchsorted(downbeats, s0 - 1e-3))
        i1 = int(np.searchsorted(downbeats, e0 - 1e-3))
        structure.append(
            {"label": label, "start": round(s0, 3), "end": round(e0, 3), "bars": max(1, i1 - i0)}
        )
    return structure


def _cluster_segments(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n < 4:
        return np.arange(1, n + 1)
    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(x, method="ward")
    k = max(2, min(n - 1, int(round(n * 0.5))))
    return fcluster(z, t=k, criterion="maxclust")


def _label_segments(spans: list[tuple[float, float]], seg_energy: list[float],
                    seg_vocal: list[float], clusters: np.ndarray,
                    duration: float) -> list[str]:
    n = len(spans)
    labels = ["verse"] * n
    if n == 0:
        return labels
    e_n = np.asarray(seg_energy, dtype=float) / (max(seg_energy) + 1e-9)
    cl = np.asarray(clusters)
    counts = {int(c): int(np.sum(cl == c)) for c in set(cl.tolist())}

    # Chorus: most-repeated high-energy cluster.
    chorus_c, best = None, -1.0
    for c, cnt in counts.items():
        if cnt < 2:
            continue
        score = cnt * (0.6 + float(e_n[cl == c].mean()))
        if score > best:
            best, chorus_c = score, c
    if chorus_c is not None:
        for i in range(n):
            if int(cl[i]) == chorus_c:
                labels[i] = "chorus"
    else:  # nothing repeats — pick the highest-energy interior segment
        inner = list(range(1, n - 1)) if n > 2 else list(range(n))
        idx = max(inner, key=lambda i: e_n[i])
        labels[idx] = "chorus"
        chorus_c = int(cl[idx])

    if labels[0] != "chorus":
        labels[0] = "intro"
    if n > 1 and labels[-1] != "chorus":
        labels[-1] = "outro"

    # Bridge: a late unique segment.
    for i in range(n - 2, 0, -1):
        if (labels[i] == "verse" and counts.get(int(cl[i]), 0) == 1
                and spans[i][0] >= 0.5 * duration):
            labels[i] = "bridge"
            break

    # Prechorus: a repeated cluster whose members all lead directly into a chorus.
    for c, cnt in counts.items():
        if c == chorus_c or cnt < 2:
            continue
        members = [i for i in range(n) if int(cl[i]) == c]
        if all(labels[i] == "verse" and i + 1 < n and labels[i + 1] == "chorus" for i in members):
            for i in members:
                labels[i] = "prechorus"

    # Inst: interior segments with very low vocal-band energy.
    if n >= 3:
        vmed = float(np.median(seg_vocal)) + 1e-9
        for i in range(n):
            if labels[i] == "verse" and seg_vocal[i] < 0.45 * vmed:
                labels[i] = "inst"
    return labels


# ---------------------------------------------------------------------------
# energy / instrumentation / groove
# ---------------------------------------------------------------------------

def _energy_curve(bar_spans: list[tuple[float, float]], rms_env: np.ndarray,
                  onset_env: np.ndarray, frame_times: np.ndarray) -> dict:
    per_rms, per_flux = [], []
    for s0, e0 in bar_spans:
        i0, i1 = np.searchsorted(frame_times, [s0, e0])
        i1 = max(int(i1), int(i0) + 1)
        per_rms.append(float(rms_env[int(i0):int(i1)].mean()))
        per_flux.append(float(onset_env[int(i0):int(i1)].mean()))

    def norm(vals: list[float]) -> list[float]:
        peak = max(vals) if vals else 0.0
        return [round(v / peak, 4) if peak > 0 else 0.0 for v in vals]

    return {"per_bar_rms": norm(per_rms), "per_bar_flux": norm(per_flux)}


def _instrumentation(ref_wav: Path, S: np.ndarray, H: np.ndarray,
                     freqs: np.ndarray, e_harm: float, e_perc: float) -> dict:
    """Demucs stem activity if available; HPSS + band-energy heuristics otherwise."""
    via_demucs = _instrumentation_demucs(ref_wav)
    if via_demucs is not None:
        return via_demucs

    total_hp = e_harm + e_perc + 1e-12
    frac_perc = e_perc / total_hp
    frac_harm = 1.0 - frac_perc

    p_full = (S.astype(np.float64) ** 2)
    total_e = float(p_full.sum()) + 1e-12
    bass_frac = float(p_full[freqs < 150.0].sum()) / total_e
    del p_full

    p_harm = (H.astype(np.float64) ** 2)
    total_h = float(p_harm.sum()) + 1e-12
    voc_band = float(p_harm[(freqs >= 200.0) & (freqs <= 4000.0)].sum()) / total_h
    formant = float(p_harm[(freqs >= 300.0) & (freqs <= 3400.0)].sum()) / total_h
    del p_harm
    voc_raw = 0.5 * voc_band + 0.5 * formant

    clip01 = lambda v: float(np.clip(v, 0.0, 1.0))
    return {
        "drums": round(clip01(frac_perc / 0.35), 2),
        "bass": round(clip01(bass_frac / 0.18), 2),
        "melodic": round(clip01(frac_harm / 0.65), 2),
        "vocals": round(clip01((voc_raw - 0.30) / 0.45), 2),
    }


def _instrumentation_demucs(ref_wav: Path) -> dict | None:
    if not (deps.has("demucs") and deps.has("torch")):
        return None
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        wav, _ = core_audio.load_audio(ref_wav, sr=int(model.samplerate), mono=False)
        if wav.ndim == 1:
            wav = np.stack([wav, wav])
        if wav.shape[0] == 1:
            wav = np.vstack([wav, wav])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(wav[:2][None], dtype=torch.float32)
        with torch.no_grad():
            stems = apply_model(model, tensor, device=device, split=True)[0].cpu().numpy()
        stem_rms = {
            name: float(np.sqrt(np.mean(stem**2)))
            for name, stem in zip(model.sources, stems)
        }
        peak = max(stem_rms.values()) + 1e-12
        act = {name: round(min(1.0, r / peak), 2) for name, r in stem_rms.items()}
        return {
            "drums": act.get("drums", 0.5),
            "bass": act.get("bass", 0.5),
            "melodic": act.get("other", 0.5),
            "vocals": act.get("vocals", 0.5),
        }
    except Exception:
        return None


def _bar_histogram(env: np.ndarray, frame_times: np.ndarray,
                   bar_spans: list[tuple[float, float]], nbins: int = 16) -> np.ndarray:
    """Average per-bar histogram of an envelope sampled at nbins bar positions."""
    acc = np.zeros(nbins)
    count = 0
    for s0, e0 in bar_spans:
        if e0 - s0 < 0.4:
            continue
        ts = s0 + (e0 - s0) * (np.arange(nbins) + 0.5) / nbins
        v = np.interp(ts, frame_times, env[: len(frame_times)])
        peak = float(v.max())
        if peak > 1e-9:
            acc += v / peak
            count += 1
    if count:
        acc /= count
    if acc.max() > 1e-9:
        acc /= acc.max()
    return acc


def _groove(S: np.ndarray, freqs: np.ndarray, onset_env: np.ndarray,
            frame_times: np.ndarray, beat_times: np.ndarray,
            bar_spans: list[tuple[float, float]], duration: float,
            sr: int, drums_activity: float) -> dict:
    """16-bin onset histogram, swing %, and kick/snare-band pattern class."""
    import librosa

    hist = _bar_histogram(onset_env, frame_times, bar_spans)

    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP, units="time"
    )
    fracs = []
    for t in onsets:
        i = int(np.searchsorted(beat_times, t)) - 1
        if i < 0 or i >= len(beat_times) - 1:
            continue
        ibi = beat_times[i + 1] - beat_times[i]
        if ibi <= 1e-3:
            continue
        fr = (t - beat_times[i]) / ibi
        if 0.30 <= fr <= 0.80:  # off-beat onsets only
            fracs.append(fr)
    swing = 0.0
    if fracs:
        med = float(np.median(fracs))
        swing = float(np.clip((med - 0.5) / (2.0 / 3.0 - 0.5), 0.0, 1.0))

    low = (S[freqs < 150.0] ** 2).sum(axis=0)
    hi = (S[(freqs >= 2000.0) & (freqs <= 8000.0)] ** 2).sum(axis=0)
    kick_env = np.maximum(0.0, np.diff(low, prepend=low[:1]))
    snare_env = np.maximum(0.0, np.diff(hi, prepend=hi[:1]))
    khist = _bar_histogram(kick_env, frame_times, bar_spans)
    shist = _bar_histogram(snare_env, frame_times, bar_spans)

    quarters = [0, 4, 8, 12]
    off = [i for i in range(16) if i not in quarters]
    k_q = float(khist[quarters].mean())
    k_off = float(khist[off].mean())
    backbeat_s = float((shist[4] + shist[12]) / 2.0)
    half_s = float(shist[8])
    onset_rate = len(onsets) / max(duration, 1.0)

    if onset_rate < 0.8 or drums_activity < 0.15:
        pattern = "sparse"
    elif swing >= 0.45:
        pattern = "shuffle"
    elif float(khist[quarters].min()) >= 0.45 and k_q >= 1.5 * (k_off + 1e-9):
        pattern = "four_on_floor"
    elif half_s >= 0.5 and half_s > 1.3 * shist[4] and half_s > 1.3 * shist[12]:
        pattern = "halftime"
    elif backbeat_s >= 0.35:
        pattern = "backbeat"
    else:
        pattern = "backbeat" if drums_activity >= 0.5 else "sparse"

    return {
        "swing": round(swing, 3),
        "pattern_class": pattern,
        "onset_histogram": [round(float(v), 4) for v in hist],
    }



# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------

_GENRE_KEYWORDS = [
    ("hip hop", ("hip hop", "hip-hop", "rap", "trap", "drill", "boom bap")),
    ("edm", ("edm", "house", "techno", "electro", "dubstep", "drum & bass", "dnb", "dance", "club mix")),
    ("rock", ("rock", "punk", "grunge", "indie rock")),
    ("metal", ("metal", "metalcore")),
    ("r&b", ("r&b", "rnb", "soul", "neo soul")),
    ("country", ("country", "bluegrass")),
    ("folk", ("folk", "singer-songwriter")),
    ("jazz", ("jazz", "bossa", "swing band")),
    ("blues", ("blues",)),
    ("classical", ("classical", "orchestra", "symphony", "concerto")),
    ("lofi", ("lofi", "lo-fi", "chillhop")),
    ("latin", ("latin", "reggaeton", "salsa", "bachata")),
    ("k-pop", ("k-pop", "kpop")),
    ("acoustic", ("acoustic", "unplugged")),
    ("pop", ("pop",)),
]
_MOOD_KEYWORDS = [
    ("party", ("party", "club", "banger")),
    ("romantic", ("love", "romance", "valentine")),
    ("melancholic", ("sad", "heartbreak", "cry", "lonely")),
    ("chill", ("chill", "relax", "calm", "study")),
    ("happy", ("happy", "feel good", "sunshine")),
    ("dark", ("dark", "night", "shadow")),
    ("epic", ("epic", "cinematic")),
]


def _tags(meta: dict, bpm: float, key: dict, energy_mean: float,
          groove: dict, instrumentation: dict) -> tuple[list[str], list[str]]:
    """Genre + mood tags from YouTube metadata and audio heuristics (≥1 of each)."""
    text = " ".join(
        str(x).lower()
        for x in (
            [meta.get("title", ""), meta.get("channel", "")]
            + list(meta.get("tags") or [])
            + list(meta.get("categories") or [])
        )
        if x
    )

    genres: list[str] = []
    for name, words in _GENRE_KEYWORDS:
        if any(w in text for w in words) and name not in genres:
            genres.append(name)
        if len(genres) >= 3:
            break

    pattern = groove.get("pattern_class", "")
    swing = float(groove.get("swing", 0.0))
    if not genres:  # audio-only heuristics
        if pattern == "four_on_floor" and 112 <= bpm <= 140:
            genres.append("dance")
        elif bpm <= 95 and pattern == "halftime":
            genres.append("hip hop")
        elif swing >= 0.45:
            genres.append("jazz")
    if not genres:
        genres.append("pop")

    moods: list[str] = []
    for name, words in _MOOD_KEYWORDS:
        if any(w in text for w in words) and name not in moods:
            moods.append(name)
    mode = key.get("mode", "major")
    if energy_mean >= 0.6 and bpm >= 118 and "energetic" not in moods:
        moods.append("energetic")
    if mode == "major" and energy_mean >= 0.45 and "bright" not in moods:
        moods.append("bright")
    if mode == "minor" and "moody" not in moods:
        moods.append("moody")
    if bpm < 90 and energy_mean < 0.5 and "laid-back" not in moods:
        moods.append("laid-back")
    if not moods:
        moods.append("warm")
    return genres[:3], moods[:3]


# ---------------------------------------------------------------------------
# misc utilities
# ---------------------------------------------------------------------------

def _safe_progress(progress: ProgressFn) -> ProgressFn:
    def fn(frac: float, msg: str) -> None:
        try:
            progress(float(frac), str(msg))
        except Exception:
            pass

    return fn


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so the profile is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
