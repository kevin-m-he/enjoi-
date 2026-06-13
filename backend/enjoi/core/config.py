"""Application configuration and well-known directories."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "enjoi"
VERSION = "0.1.0"
HOST = "127.0.0.1"
PORT = 8723

# Hard limits / defaults from the build spec
SAMPLE_RATE = 44100
SEARCH_DEBOUNCE_MS = 400
SEARCH_RESULT_LIMIT = 12
MAX_REFERENCE_DURATION_SEC = 10 * 60          # duration cap for references
LENGTH_TOLERANCE = 0.05                       # output length within ±5% of reference
MAX_VOCAL_STRETCH = 0.06                      # ±6% time-stretch budget for chops
DEFAULT_RETUNE_SPEED = 35

LOUDNESS_PRESETS = {
    "streaming": {"lufs": -14.0, "true_peak_db": -1.0, "label": "Streaming (-14 LUFS)"},
    "loud": {"lufs": -9.0, "true_peak_db": -1.0, "label": "Loud (-9 LUFS)"},
    "dynamic": {"lufs": -16.0, "true_peak_db": -1.0, "label": "Dynamic (-16 LUFS)"},
}
MIX_PRESETS = ["pop", "hiphop", "rnb", "rock", "acoustic"]

IMPACT_WEIGHTS_DEFAULT = {
    "energy": 0.20,
    "pitch_range": 0.15,
    "pitch_height": 0.10,
    "vibrato": 0.15,
    "repetition": 0.20,
    "brightness": 0.10,
    "hookiness": 0.10,
}


def data_dir() -> Path:
    override = os.environ.get("ENJOI_DATA_DIR")
    if override:
        d = Path(override)
    else:
        appdata = os.environ.get("APPDATA")
        d = (Path(appdata) if appdata else Path.home() / ".config") / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def projects_dir() -> Path:
    d = data_dir() / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
