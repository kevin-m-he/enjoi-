"""Export module (spec 4.10): final WAV/MP3 encode, metadata, stems, manifest.

Sources are the master render + stem files produced by mix.py — never the
reference cache (contract rule 6). Returns the exports list for ProjectState
with paths relative to the project directory (served as /media URLs).
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core import audio as core_audio
from ..core import deps
from ..core.errors import PipelineError
from ..core.storage import read_json, write_json

log = logging.getLogger("enjoi.export")

MP3_BITRATE = "320k"
COMMENT_TEXT = "Made with enjoi"


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------

def _encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = deps.ffmpeg_path()
    if not ffmpeg:
        raise PipelineError(
            "MP3 export needs FFmpeg, which was not found. "
            "Install FFmpeg (or `pip install imageio-ffmpeg`) and try again."
        )
    attempts = [
        ["-codec:a", "libmp3lame", "-b:a", MP3_BITRATE],
        ["-b:a", MP3_BITRATE],  # fallback: let ffmpeg pick its mp3 encoder
    ]
    last_err = ""
    for extra in attempts:
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(wav_path), *extra, str(mp3_path)]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode == 0 and mp3_path.exists():
            return
        last_err = proc.stderr.decode(errors="ignore")[:300]
    raise PipelineError(f"FFmpeg could not encode the MP3: {last_err}")


def _id3_key(key_str: str) -> str:
    """'A minor' -> 'Am', 'C major' -> 'C' (ID3 TKEY format, best effort)."""
    parts = str(key_str or "").strip().split()
    if not parts:
        return ""
    tonic = parts[0]
    minor = any("min" in p.lower() for p in parts[1:])
    return tonic + ("m" if minor else "")


def _tag_mp3(mp3_path: Path, metadata: dict) -> None:
    from mutagen.id3 import COMM, ID3, ID3NoHeaderError, TBPM, TIT2, TKEY, TPE1

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()
    title = str(metadata.get("title") or "")
    artist = str(metadata.get("artist") or "")
    if title:
        tags.add(TIT2(encoding=3, text=title))
    if artist:
        tags.add(TPE1(encoding=3, text=artist))
    bpm = metadata.get("bpm")
    if bpm:
        tags.add(TBPM(encoding=3, text=str(int(round(float(bpm))))))
    key = _id3_key(metadata.get("key", ""))
    if key:
        tags.add(TKEY(encoding=3, text=key))
    tags.add(COMM(encoding=3, lang="eng", desc="", text=COMMENT_TEXT))
    tags.save(str(mp3_path))


def _tag_wav(wav_path: Path, metadata: dict) -> None:
    """Best-effort metadata on the WAV via mutagen (wrapped per contract)."""
    try:
        from mutagen.id3 import COMM, TBPM, TIT2, TKEY, TPE1
        from mutagen.wave import WAVE

        w = WAVE(str(wav_path))
        if w.tags is None:
            w.add_tags()
        title = str(metadata.get("title") or "")
        artist = str(metadata.get("artist") or "")
        if title:
            w.tags.add(TIT2(encoding=3, text=title))
        if artist:
            w.tags.add(TPE1(encoding=3, text=artist))
        bpm = metadata.get("bpm")
        if bpm:
            w.tags.add(TBPM(encoding=3, text=str(int(round(float(bpm))))))
        key = _id3_key(metadata.get("key", ""))
        if key:
            w.tags.add(TKEY(encoding=3, text=key))
        w.tags.add(COMM(encoding=3, lang="eng", desc="", text=COMMENT_TEXT))
        w.save()
    except Exception as exc:
        log.debug("WAV metadata tagging skipped: %s", exc)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def export_song(project, master_wav: Path, metadata: dict,
                include_stems: bool, progress) -> list[dict]:
    """Encode final formats + metadata + song_manifest.json (spec 4.10)."""
    metadata = metadata or {}
    exports_dir = project.exports_dir
    master_wav = Path(master_wav)
    if not master_wav.exists():
        raise PipelineError("Master render not found — run the mix stage first.")

    # ---- song.wav (44.1 kHz / 24-bit) -------------------------------------
    progress(0.05, "Writing song.wav (24-bit)")
    audio, sr = core_audio.load_audio(master_wav, mono=False)
    audio = np.nan_to_num(np.atleast_2d(audio), nan=0.0, posinf=0.0, neginf=0.0)
    wav_path = exports_dir / "song.wav"
    core_audio.save_wav(wav_path, audio.astype(np.float32), sr, subtype="PCM_24")
    _tag_wav(wav_path, metadata)

    # ---- song.mp3 (320 kbps) -----------------------------------------------
    progress(0.35, "Encoding song.mp3 (320 kbps)")
    mp3_path = exports_dir / "song.mp3"
    _encode_mp3(wav_path, mp3_path)
    try:
        _tag_mp3(mp3_path, metadata)
    except Exception as exc:  # tags are nice-to-have, never fail the export
        log.warning("MP3 tagging failed: %s", exc)

    exports = [
        {"file": "exports/song.wav", "format": "wav"},
        {"file": "exports/song.mp3", "format": "mp3"},
    ]

    # ---- optional stems ------------------------------------------------------
    if include_stems:
        progress(0.6, "Writing stems")
        stem_map = [
            ("_stem_instrumental.wav", "instrumental_stem.wav"),
            ("_stem_vocals.wav", "vocal_stem.wav"),
        ]
        for src_name, dst_name in stem_map:
            src = exports_dir / src_name
            if not src.exists():
                log.warning("Stem source missing, skipping: %s", src_name)
                continue
            stem_audio, stem_sr = core_audio.load_audio(src, mono=False)
            stem_audio = np.nan_to_num(np.atleast_2d(stem_audio))
            core_audio.save_wav(exports_dir / dst_name, stem_audio.astype(np.float32),
                                stem_sr, subtype="PCM_24")
            exports.append({"file": f"exports/{dst_name}", "format": "wav"})

    # ---- song_manifest.json ---------------------------------------------------
    progress(0.85, "Writing song manifest")
    meta_path = exports_dir / "_master_meta.json"
    lufs = None
    true_peak = None
    if meta_path.exists():
        try:
            master_meta = read_json(meta_path)
            lufs = master_meta.get("lufs")
            true_peak = master_meta.get("true_peak_db")
        except Exception as exc:
            log.warning("Could not read master meta: %s", exc)

    uniqueness_report = None
    if project.uniqueness_report_path.exists():
        try:
            uniqueness_report = read_json(project.uniqueness_report_path)
        except Exception as exc:
            log.warning("Could not read uniqueness report: %s", exc)

    engine = ""
    try:
        engine = (project.read_state().get("instrumental") or {}).get("engine", "")
    except Exception:
        pass
    instrumental_source = (
        f"generated instrumental (local {engine})" if engine
        else "generated instrumental (local MusicGen/procedural)"
    )

    manifest_exports = []
    for entry in exports:
        item = {"file": Path(entry["file"]).name, "format": entry["format"]}
        if entry["file"] == "exports/song.wav":
            item["lufs"] = lufs
            item["true_peak_db"] = true_peak
        manifest_exports.append(item)

    manifest = {
        "title": str(metadata.get("title") or ""),
        "artist": str(metadata.get("artist") or ""),
        "bpm": float(metadata.get("bpm") or 0.0),
        "key": str(metadata.get("key") or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [instrumental_source, "user one-take vocal"],
        "reference_audio_in_output": False,
        "uniqueness_report": uniqueness_report,
        "exports": manifest_exports,
    }
    write_json(project.manifest_path, manifest)

    # _master_tmp.wav stays (render cache); nothing else temporary to clean.
    progress(1.0, "Export complete")
    return exports
