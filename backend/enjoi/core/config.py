"""Application configuration and well-known directories."""
from __future__ import annotations

import math
import os
from pathlib import Path

APP_NAME = "enjoi"
VERSION = "0.1.0"
HOST = "127.0.0.1"
PORT = 8723

# Hard limits / defaults from the build spec
SAMPLE_RATE = 44100

# Concert-pitch reference for ALL generated audio. The product is tuned to
# A4 = 432 Hz ("Verdi tuning") for a warmer, more enveloping sound. The
# procedural engine synthesizes notes at this reference and the autotuner snaps
# the vocal to the same reference, so the instrumental and vocal stay coherent.
TUNING_HZ = 432.0
STANDARD_TUNING_HZ = 440.0
# Constant pitch offset (semitones) from standard 440 Hz to 432 Hz (~ -31.77 c).
# Add this to a 440-based pitch shift to retune audio to 432.
TUNING_OFFSET_SEMITONES = 12.0 * math.log2(TUNING_HZ / STANDARD_TUNING_HZ)


def midi_to_hz(midi: float, tuning: float = TUNING_HZ) -> float:
    """MIDI note number → frequency (Hz) at the given concert-A reference.

    With the default ``tuning`` this returns 432 Hz-based frequencies; pass
    ``STANDARD_TUNING_HZ`` for conventional 440 Hz math.
    """
    return float(tuning) * 2.0 ** ((float(midi) - 69.0) / 12.0)
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
