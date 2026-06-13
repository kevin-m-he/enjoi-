"""Optional-dependency probing and external tool resolution.

Heavy/optional libraries are NEVER imported at module top level anywhere in the
codebase; modules call `optional_import` (or import lazily inside functions) and
fall back gracefully per docs/API_CONTRACT.md rule 2.
"""
from __future__ import annotations

import functools
import importlib
import shutil
from typing import Any


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def has(name: str) -> bool:
    return optional_import(name) is not None


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    mod = optional_import("imageio_ffmpeg")
    if mod is not None:
        try:
            return mod.get_ffmpeg_exe()
        except Exception:
            return None
    return None


@functools.lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


@functools.lru_cache(maxsize=1)
def rubberband_cli() -> str | None:
    return shutil.which("rubberband")


def gpu_available() -> bool:
    torch = optional_import("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def capabilities() -> dict:
    """Capability report for /api/health and UI feature flags."""
    return {
        "ffmpeg": ffmpeg_path() is not None,
        "gpu": gpu_available(),
        "musicgen": has("audiocraft"),
        "whisper": has("faster_whisper"),
        "demucs": has("demucs"),
        "pedalboard": has("pedalboard"),
        "rubberband": has("pyrubberband") and rubberband_cli() is not None,
        "madmom": has("madmom"),
        "essentia": has("essentia"),
        "crepe": has("torchcrepe") or has("crepe"),
        "noisereduce": has("noisereduce"),
        "pyloudnorm": has("pyloudnorm"),
    }
