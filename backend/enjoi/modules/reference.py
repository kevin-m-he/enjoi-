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
    """Download a YouTube reference into ``project.ref_cache_dir``, analyze it, and
    write + return the reference profile dict (contract schema)."""
    p = _safe_progress(progress)
    meta, src_path = _download_reference(project, url, p)        # 0.00 – 0.25
    return _decode_validate_analyze(project, src_path, meta, p)


def analyze_uploaded(project: Project, uploaded_path: Path, title: str,
                     progress: ProgressFn) -> dict:
    """Analyze a user-UPLOADED audio file as the reference — same analysis as the
    YouTube path, but works for ANY user and can never be bot-blocked."""
    import shutil

    p = _safe_progress(progress)
    p(0.05, "Reading your audio file…")
    cache = project.ref_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("source.*"):
        try:
            stale.unlink()
        except OSError:
            pass
    suffix = (uploaded_path.suffix or ".wav").lower()
    src_path = cache / f"source{suffix}"
    shutil.copyfile(uploaded_path, src_path)
    meta = {
        "title": (title or uploaded_path.stem or "Uploaded audio").strip(),
        "channel": "", "url": "", "video_id": "", "duration_sec": 0.0,
        "thumbnail_url": "", "uploader": "", "tags": [], "categories": [],
        "keywords": [], "description": "",
    }
    return _decode_validate_analyze(project, src_path, meta, p)


def _decode_validate_analyze(project: Project, src_path: Path, meta: dict,
                             p: ProgressFn) -> dict:
    """Shared tail for both reference paths: decode → validate length → save
    reference.wav → analyze → write + return the profile dict."""
    p(0.25, "Decoding audio…")
    y44, _ = core_audio.load_audio(src_path, sr=config.SAMPLE_RATE, mono=True)
    duration = len(y44) / float(config.SAMPLE_RATE)
    if duration > config.MAX_REFERENCE_DURATION_SEC + 1.0:
        raise PipelineError(
            f"That audio is {duration / 60:.1f} minutes long — references must be 10 minutes or shorter."
        )
    if duration < 15.0:
        raise PipelineError("That audio is too short to use as a reference (need at least 15 seconds).")

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
        "url": meta.get("url", ""),
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
        # On a datacenter/VPS IP, YouTube's default `web` client triggers the
        # "Sign in to confirm you're not a bot" block. These clients ship their
        # streams without that gate far more often; yt-dlp tries them in order.
        "extractor_args": {
            "youtube": {"player_client": ["ios", "tv", "mweb", "android", "web_safari"]}
        },
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
        "uploader": info.get("uploader") or "",
        "tags": list(info.get("tags") or []),
        "categories": list(info.get("categories") or []),
        "keywords": list(info.get("categories") or []),
        # Trim the description so the genre miner sees real keywords without a
        # giant blob (links, credits, hashtags often carry genre cues).
        "description": (str(info.get("description") or "")[:1500]),
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
    instrumentation = _instrumentation(ref_wav, S, H, freqs, e_harm, e_perc, p)

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
                     freqs: np.ndarray, e_harm: float, e_perc: float,
                     p: ProgressFn) -> dict:
    """Per-instrument activity 0..1, natural instruments first.

    PRIMARY path: Demucs ``htdemucs_6s`` source separation (drums, bass, other,
    vocals, guitar, piano) on the GPU when available. Each stem's activity is a
    blend of its loudness (RMS relative to the mix) and how much of the track it
    is *active* (fraction of frames above a per-stem noise floor), normalized so
    the most prominent instrument is ~1.0. ``melodic`` (= max of guitar/piano/
    other) is kept for backward compatibility, and ``other`` carries the
    residual (synths / strings / etc.).

    FALLBACK (demucs/torch absent or failed): the HPSS + band-energy heuristic,
    which still emits every key — guitar/piano are estimated from spectral cues
    so consumers always see the full dict.
    """
    via_demucs = _instrumentation_demucs(ref_wav, p)
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
    # Plucked-string / piano cues live in the mid/upper-mid harmonic band; a
    # crude split lets us still surface guitar vs piano without separation.
    guitar_band = float(p_harm[(freqs >= 250.0) & (freqs <= 2500.0)].sum()) / total_h
    piano_band = float(p_harm[(freqs >= 150.0) & (freqs <= 5000.0)].sum()) / total_h
    del p_harm
    voc_raw = 0.5 * voc_band + 0.5 * formant

    clip01 = lambda v: float(np.clip(v, 0.0, 1.0))
    melodic = clip01(frac_harm / 0.65)
    guitar = round(clip01(0.6 * melodic + 0.4 * clip01(guitar_band / 0.5)), 2)
    piano = round(clip01(0.5 * melodic + 0.5 * clip01(piano_band / 0.6)), 2)
    return {
        "drums": round(clip01(frac_perc / 0.35), 2),
        "bass": round(clip01(bass_frac / 0.18), 2),
        "guitar": guitar,
        "piano": piano,
        "other": round(melodic, 2),
        "melodic": round(max(melodic, guitar, piano), 2),
        "vocals": round(clip01((voc_raw - 0.30) / 0.45), 2),
    }


# Natural instruments are listed first so consumers that iterate the dict
# encounter real instruments (guitar/piano/drums/bass) before the synth-ish
# residual ("other"). ``melodic`` stays for backward compatibility.
_DEMUCS_MODEL = "htdemucs_6s"
_NATURAL_ORDER = ("drums", "bass", "guitar", "piano", "vocals", "other")


def _instrumentation_demucs(ref_wav: Path, p: ProgressFn) -> dict | None:
    """htdemucs_6s stem activity (drums/bass/guitar/piano/vocals/other).

    Returns the activity dict, or ``None`` if demucs/torch is unavailable or the
    separation fails (the caller then uses the spectral fallback)."""
    if not (deps.has("demucs") and deps.has("torch")):
        return None
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        p(0.74, "Separating stems (guitar, piano, drums, bass)…")
        model = get_model(_DEMUCS_MODEL)
        model.eval()
        wav, _ = core_audio.load_audio(ref_wav, sr=int(model.samplerate), mono=False)
        if wav.ndim == 1:
            wav = np.stack([wav, wav])
        if wav.shape[0] == 1:
            wav = np.vstack([wav, wav])
        wav = np.ascontiguousarray(wav[:2], dtype=np.float32)

        device = "cuda" if deps.gpu_available() and torch.cuda.is_available() else "cpu"
        # Per-channel mix energy for relative-loudness normalization.
        mix_rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) + 1e-9

        tensor = torch.from_numpy(wav[None])
        with torch.no_grad():
            stems = apply_model(
                model, tensor, device=device, split=True, overlap=0.10, progress=False
            )[0].cpu().numpy()
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        p(0.79, "Measuring instrument activity…")
        sr = int(model.samplerate)
        act_raw: dict[str, float] = {}
        for name, stem in zip(model.sources, stems):
            mono = stem.mean(axis=0).astype(np.float64)
            loud = float(np.sqrt(np.mean(mono ** 2))) / mix_rms  # relative loudness
            # Frame activity: fraction of ~46 ms frames whose RMS clears a noise
            # floor set relative to the stem's own peak frame energy.
            win = max(int(0.046 * sr), 1)
            n_win = max(len(mono) // win, 1)
            frames = mono[: n_win * win].reshape(n_win, win)
            fr_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            floor = 0.06 * float(fr_rms.max() if fr_rms.size else 0.0)
            active = float(np.mean(fr_rms > floor)) if floor > 0 else 0.0
            act_raw[name] = 0.7 * loud + 0.3 * (loud * active)

        peak = max(act_raw.values()) + 1e-12
        norm = {name: float(np.clip(v / peak, 0.0, 1.0)) for name, v in act_raw.items()}

        guitar = norm.get("guitar", 0.0)
        piano = norm.get("piano", 0.0)
        other = norm.get("other", 0.0)
        out = {
            "drums": round(norm.get("drums", 0.0), 2),
            "bass": round(norm.get("bass", 0.0), 2),
            "guitar": round(guitar, 2),
            "piano": round(piano, 2),
            "vocals": round(norm.get("vocals", 0.0), 2),
            "other": round(other, 2),
            "melodic": round(max(guitar, piano, other), 2),
        }
        # Emit in natural-first order.
        return {k: out[k] for k in (*_NATURAL_ORDER, "melodic")}
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

# Genre lexicon. Order matters: more specific genres are listed before the
# broad "pop" catch-all so a folk/country song is never mislabeled. Each entry's
# keywords are matched (substring) against the pooled YouTube metadata text.
_GENRE_KEYWORDS = [
    ("country", ("country", "americana", "bluegrass", "nashville", "honky tonk",
                 "alt-country", "alt country")),
    ("folk", ("folk", "singer-songwriter", "singer songwriter", "indie folk",
              "folk rock", "stomp")),
    ("acoustic", ("acoustic", "unplugged")),
    ("hip hop", ("hip hop", "hip-hop", "hiphop", "rap", "trap", "drill",
                 "boom bap", "trapsoul")),
    ("edm", ("edm", "house", "techno", "electro", "dubstep", "drum & bass",
             "dnb", "future bass", "club mix", "edm mix")),
    ("dance", ("dance pop", "dance-pop", "dancefloor")),
    ("metal", ("metal", "metalcore", "djent", "deathcore", "thrash")),
    ("rock", ("rock", "punk", "grunge", "indie rock", "alt rock", "alternative",
              "garage")),
    ("r&b", ("r&b", "rnb", "r & b", "soul", "neo soul", "neo-soul", "motown")),
    ("jazz", ("jazz", "bossa", "swing band", "bebop", "fusion")),
    ("blues", ("blues",)),
    ("classical", ("classical", "orchestra", "orchestral", "symphony",
                   "concerto", "soundtrack", "score", "cinematic")),
    ("lofi", ("lofi", "lo-fi", "chillhop")),
    ("latin", ("latin", "reggaeton", "salsa", "bachata", "samba", "afrobeat",
               "cumbia")),
    ("k-pop", ("k-pop", "kpop", "k pop")),
    ("pop", ("pop", "synthpop", "electropop")),
]
_MOOD_KEYWORDS = [
    ("party", ("party", "club", "banger", "turn up")),
    ("romantic", ("love", "romance", "valentine")),
    ("melancholic", ("sad", "heartbreak", "heartbroken", "cry", "lonely",
                     "lonesome", "grief", "tears", "stick season")),
    ("reflective", ("reflective", "introspective", "nostalgia", "nostalgic",
                    "memories")),
    ("intimate", ("intimate", "stripped", "stripped back", "bedroom")),
    ("chill", ("chill", "relax", "calm", "study", "mellow")),
    ("happy", ("happy", "feel good", "feel-good", "sunshine", "upbeat")),
    ("dark", ("dark", "night", "shadow", "moody")),
    ("epic", ("epic", "cinematic", "anthem")),
]

# Acoustic/organic instrumentation evidence → these genres should win over pop.
_ACOUSTIC_GENRES = ("folk", "country", "acoustic", "singer-songwriter")


def _tags(meta: dict, bpm: float, key: dict, energy_mean: float,
          groove: dict, instrumentation: dict) -> tuple[list[str], list[str]]:
    """Genre + mood tags from YouTube metadata (weighted heaviest) and the
    now-accurate instrumentation, with sensible audio rules (≥1 of each).

    An acoustic-guitar-led, organically-drummed, moderate-tempo track resolves
    to folk / country / acoustic — never "pop" by default. "pop" is only a last
    resort when the evidence is genuinely ambiguous.
    """
    text = " ".join(
        str(x).lower()
        for x in (
            [meta.get("title", ""), meta.get("channel", ""),
             meta.get("description", ""), meta.get("uploader", "")]
            + list(meta.get("tags") or [])
            + list(meta.get("categories") or [])
            + list(meta.get("keywords") or [])
        )
        if x
    )

    # 1) Metadata is the strongest signal — mine it hard.
    genres: list[str] = []
    for name, words in _GENRE_KEYWORDS:
        if any(w in text for w in words) and name not in genres:
            genres.append(name)
        if len(genres) >= 3:
            break

    # 2) Instrumentation + tempo + groove rules. These either CONFIRM the
    #    metadata or, when metadata is silent, classify from the audio. Using
    #    the demucs-accurate stems, an acoustic track reads as folk/country.
    pattern = str(groove.get("pattern_class", ""))
    swing = float(groove.get("swing", 0.0))
    inst = instrumentation or {}
    drums = float(inst.get("drums") or 0.0)
    bass = float(inst.get("bass") or 0.0)
    guitar = float(inst.get("guitar") or 0.0)
    piano = float(inst.get("piano") or 0.0)
    other = float(inst.get("other") or inst.get("melodic") or 0.0)
    vocals = float(inst.get("vocals") or 0.0)
    # "Other" carries synths/strings; a high other vs low guitar/piano is the
    # electronic signature.
    organic = max(guitar, piano)
    synthy = other > 0.55 and other > organic + 0.12

    audio_genre: str | None = None
    if synthy and pattern == "four_on_floor":
        audio_genre = "edm"
    elif synthy and pattern in ("backbeat", "four_on_floor") and bpm >= 110:
        audio_genre = "pop"
    elif drums >= 0.35 and bass >= 0.3 and pattern == "halftime" and bpm <= 100 \
            and organic < 0.5:
        audio_genre = "hip hop"
    elif guitar >= 0.55 and other > guitar + 0.15 and drums >= 0.55 \
            and energy_mean >= 0.6:
        audio_genre = "rock"  # loud, distorted-leaning guitar + loud drums
    elif organic >= 0.45 and not synthy and bpm <= 150:
        # Acoustic/real-instrument led, not electronic → folk / country / acoustic.
        if drums >= 0.3 and bass >= 0.3 and 80 <= bpm <= 150:
            audio_genre = "country"   # full band, gentle backbeat
        else:
            audio_genre = "folk"      # sparse guitar/piano + voice
    elif swing >= 0.45:
        audio_genre = "jazz"

    if not genres:
        if audio_genre:
            genres.append(audio_genre)
    else:
        # If metadata is generic ("pop" only) but the audio is clearly organic,
        # prefer the acoustic reading so a folk/country song isn't called pop.
        if genres == ["pop"] and audio_genre in _ACOUSTIC_GENRES:
            genres = [audio_genre, "pop"]
        elif audio_genre and audio_genre not in genres and len(genres) < 3:
            genres.append(audio_genre)

    # When the AUDIO is clearly acoustic/organic, an acoustic-family genre must
    # LEAD even if the metadata also carried a broader tag (e.g. an indie-folk
    # song tagged "rock"/"pop"). This keeps the secondary tag but ensures the
    # generation palette resolves to acoustic timbres, not electric/synth.
    if audio_genre in _ACOUSTIC_GENRES:
        acoustic_in = [g for g in genres if g in _ACOUSTIC_GENRES]
        if not acoustic_in:
            genres.insert(0, audio_genre)
        elif genres[0] not in _ACOUSTIC_GENRES:
            lead = audio_genre if audio_genre in acoustic_in else acoustic_in[0]
            genres = [lead] + [g for g in genres if g != lead]

    if not genres:
        genres.append("pop")  # truly ambiguous

    # 3) Mood — metadata keywords first, then key mode + energy + tempo.
    moods: list[str] = []
    for name, words in _MOOD_KEYWORDS:
        if any(w in text for w in words) and name not in moods:
            moods.append(name)
    mode = key.get("mode", "major")
    low_energy = energy_mean < 0.5
    if mode == "minor":
        if "melancholic" not in moods:
            moods.append("melancholic")
        if low_energy and "reflective" not in moods:
            moods.append("reflective")
    if low_energy and bpm < 110:
        if "intimate" not in moods:
            moods.append("intimate")
        if "mellow" not in moods and "chill" not in moods:
            moods.append("mellow")
    # A song's emotional character shouldn't read as both sad and upbeat. If a
    # melancholic/reflective/intimate mood already leads (usually from the
    # metadata, the strongest signal — e.g. a heartbreak ballad in a major key),
    # don't bolt on a contradictory "bright"/"energetic" tag from the audio rules.
    somber = bool({"melancholic", "reflective", "intimate"} & set(moods))
    if not somber:
        if energy_mean >= 0.6 and bpm >= 118 and "energetic" not in moods:
            moods.append("energetic")
        if mode == "major" and energy_mean >= 0.55 and bpm >= 100 and "bright" not in moods:
            moods.append("bright")
    if mode == "major" and not moods:
        moods.append("warm")
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
