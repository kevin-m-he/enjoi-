"""Shared audio I/O and DSP helpers.

All pipeline audio is float32 numpy at 44.1 kHz. Time-stretch / pitch-shift use
Rubber Band when available (formant-safe) and fall back to librosa.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import config, deps
from .errors import PipelineError

SR = config.SAMPLE_RATE


# ---- I/O -------------------------------------------------------------------

def load_audio(path: Path | str, sr: int = SR, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load any audio file → float32 ndarray (samples,) or (channels, samples)."""
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Audio file not found: {path.name}")
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        data = data.T  # (channels, samples)
    except Exception:
        # Compressed formats (mp3/m4a) → decode via ffmpeg to a temp wav
        data, file_sr = _ffmpeg_decode(path)
    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    if file_sr != sr:
        data = resample(data, file_sr, sr)
    out = data[0] if mono else data
    return np.ascontiguousarray(out, dtype=np.float32), sr


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = deps.ffmpeg_path()
    if not ffmpeg:
        raise PipelineError(
            f"Cannot decode {path.suffix} — FFmpeg not found. Install FFmpeg and restart."
        )
    cmd = [
        ffmpeg, "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SR), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise PipelineError(f"FFmpeg failed to decode {path.name}: {proc.stderr.decode(errors='ignore')[:300]}")
    interleaved = np.frombuffer(proc.stdout, dtype=np.float32)
    data = interleaved.reshape(-1, 2).T.copy()
    return data, SR


def save_wav(path: Path | str, audio: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> Path:
    """Write float32 audio ((samples,) mono or (channels, samples)) as WAV."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype=subtype)
    return path


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    soxr = deps.optional_import("soxr")
    squeeze = audio.ndim == 1
    x = audio[None, :] if squeeze else audio
    if soxr is not None:
        out = np.stack([soxr.resample(ch, from_sr, to_sr) for ch in x])
    else:
        import librosa

        out = np.stack([librosa.resample(ch, orig_sr=from_sr, target_sr=to_sr) for ch in x])
    out = out.astype(np.float32)
    return out[0] if squeeze else out


def duration_sec(path: Path | str) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        audio, sr = load_audio(path)
        return len(audio) / sr


# ---- gain / metering ---------------------------------------------------------

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin: float, floor: float = -120.0) -> float:
    if lin <= 0:
        return floor
    return float(max(20.0 * np.log10(lin), floor))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def normalize_peak(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-9:
        return audio
    return (audio * (db_to_lin(peak_db) / peak)).astype(np.float32)


# ---- editing -----------------------------------------------------------------

def crossfade_concat(chunks: list[np.ndarray], fade_sec: float = 0.05, sr: int = SR) -> np.ndarray:
    """Concatenate mono chunks with equal-power crossfades at the joins."""
    chunks = [c.astype(np.float32) for c in chunks if c is not None and c.size > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    fade = max(int(fade_sec * sr), 1)
    out = chunks[0]
    for nxt in chunks[1:]:
        f = min(fade, len(out), len(nxt))
        if f < 2:
            out = np.concatenate([out, nxt])
            continue
        t = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
        a, b = np.cos(t), np.sin(t)
        joined = out[-f:] * a + nxt[:f] * b
        out = np.concatenate([out[:-f], joined, nxt[f:]])
    return out


def fade_edges(audio: np.ndarray, fade_in_sec: float = 0.01, fade_out_sec: float = 0.01, sr: int = SR) -> np.ndarray:
    out = audio.copy()
    fi = min(int(fade_in_sec * sr), len(out))
    fo = min(int(fade_out_sec * sr), len(out))
    if fi > 1:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 1:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def time_stretch(audio: np.ndarray, rate: float, sr: int = SR) -> np.ndarray:
    """Stretch duration by 1/rate (rate>1 = shorter). Rubber Band → librosa fallback."""
    if abs(rate - 1.0) < 1e-4 or audio.size == 0:
        return audio
    if deps.has("pyrubberband") and deps.rubberband_cli():
        try:
            import pyrubberband as pyrb

            return pyrb.time_stretch(audio.astype(np.float64), sr, rate).astype(np.float32)
        except Exception:
            pass
    import librosa

    return librosa.effects.time_stretch(audio, rate=float(rate)).astype(np.float32)


def pitch_shift(audio: np.ndarray, semitones: float, sr: int = SR) -> np.ndarray:
    if abs(semitones) < 1e-3 or audio.size == 0:
        return audio
    if deps.has("pyrubberband") and deps.rubberband_cli():
        try:
            import pyrubberband as pyrb

            return pyrb.pitch_shift(audio.astype(np.float64), sr, semitones).astype(np.float32)
        except Exception:
            pass
    import librosa

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(semitones)).astype(np.float32)
