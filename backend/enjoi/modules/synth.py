"""Procedural instrumental engine — the guaranteed fallback when MusicGen is absent.

Pure numpy + scipy.signal synthesis (no ML dependencies). Renders a complete,
genre-appropriate, radio-style instrumental from a generation plan (see
docs/API_CONTRACT.md): a real-ish drum kit, bass / sub / 808, plucked acoustic
& electric guitars (Karplus-Strong), pianos, electric pianos with tine
character, organ, pads, strings, synth keys/plucks/leads, brass and bells —
selected and voiced per the plan's ``genre`` and ``instrument_palette`` so the
output clearly belongs to the reference's genre and emotional feel rather than
sounding like generic "video-game" synth music.

Design goals:
* REALISM — layered oscillators, per-instrument ADSR, subtle detune/chorus,
  gentle saturation, body/noise components (Karplus-Strong strings, tine
  electric piano, real-ish drums).
* GENRE MATCHING — ``plan["genre"]`` + ``plan["instrument_palette"]`` pick the
  timbres, groove and arrangement density.
* MIX BALANCE — hi-hats/cymbals are among the QUIETEST elements; kick + bass are
  the foundation; the vocal range (~300 Hz–4 kHz) is left uncluttered; one bass
  element owns the low end; tasteful panning (hats/keys off-centre,
  kick/bass/snare centred).
* 432 Hz TUNING — every note→frequency conversion goes through
  ``config.midi_to_hz`` (A4 / MIDI 69 → 432.0 Hz).

Reproducibility: every musical choice derives from ``random.Random`` instances
seeded by ``plan["seed"]``. Two optional plan keys let the orchestrator
(generate.py) regenerate ONLY a failing aspect after a uniqueness audit:

* ``plan["_melody_seed"]``  — overrides the seed for the chorus lead melody.
* ``plan["_harmony_seed"]`` — overrides the seed for chord-progression choice.

The melody and chords are generated purely from the seed and the key/scale —
never from any reference data — so the output is original by construction.

Public API:
    render_section(plan, section, sr=44100) -> np.ndarray (2, n)
    render_song(plan, progress)             -> np.ndarray (2, n)
    scale_midi_notes(tonic, mode)           -> list[int]   (7 MIDI notes)
    NOTE_TO_PC                              -> dict (imported by unique.py)
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config

SR = config.SAMPLE_RATE

# ---------------------------------------------------------------------------
# Music theory tables
# ---------------------------------------------------------------------------

NOTE_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]
MODE_INTERVALS = {
    "major": _MAJOR, "ionian": _MAJOR,
    "minor": _MINOR, "aeolian": _MINOR,
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}
_MINORISH_MODES = {"minor", "aeolian", "dorian", "phrygian"}

# Curated diatonic progressions as 0-based scale-degree lists. Triad quality
# falls out of the diatonic scale automatically (thirds stacked in-scale).
MAJOR_PROGRESSIONS = [
    [0, 4, 5, 3],   # I-V-vi-IV
    [0, 5, 3, 4],   # I-vi-IV-V
    [5, 3, 0, 4],   # vi-IV-I-V
    [0, 3, 5, 4],   # I-IV-vi-V
    [1, 4, 0, 3],   # ii-V-I-IV
    [0, 3, 0, 4],   # I-IV-I-V
]
MINOR_PROGRESSIONS = [
    [0, 5, 2, 6],   # i-VI-III-VII
    [0, 3, 5, 4],   # i-iv-VI-v
    [0, 6, 5, 6],   # i-VII-VI-VII
    [0, 5, 3, 4],   # i-VI-iv-v
    [0, 2, 6, 5],   # i-III-VII-VI
    [0, 3, 4, 4],   # i-iv-v-v
]

_LABEL_CODE = {"intro": 1, "verse": 2, "prechorus": 3, "chorus": 4,
               "bridge": 5, "outro": 6, "inst": 7}
_DEFAULT_ENERGY = {"intro": 0.30, "verse": 0.55, "prechorus": 0.65,
                   "chorus": 0.85, "bridge": 0.50, "outro": 0.28, "inst": 0.60}

_DEFAULT_STRUCTURE = [
    {"label": "intro", "bars": 4}, {"label": "verse", "bars": 8},
    {"label": "chorus", "bars": 8}, {"label": "verse", "bars": 8},
    {"label": "chorus", "bars": 8}, {"label": "outro", "bars": 4},
]

# Per-bar lead rhythm motifs in beats (sum == 4; adapted for other meters).
_LEAD_RHYTHMS_4 = [
    [1.0, 0.5, 0.5, 1.0, 1.0],
    [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
    [1.5, 0.5, 1.0, 1.0],
    [0.5, 0.5, 0.5, 0.5, 2.0],
    [1.0, 1.0, 0.5, 0.5, 1.0],
]


# ---------------------------------------------------------------------------
# Frequency (432 Hz tuning) — ALL note→Hz conversions go through here.
# ---------------------------------------------------------------------------

def _hz(m: float) -> float:
    """MIDI note → frequency at the project tuning (A4/MIDI 69 → 432 Hz)."""
    return config.midi_to_hz(float(m))


# ---------------------------------------------------------------------------
# Scale / chord helpers
# ---------------------------------------------------------------------------

def scale_midi_notes(tonic: str, mode: str) -> list[int]:
    """The 7 scale MIDI notes for a key, octave 3 (contract: A minor -> 57..67)."""
    name = str(tonic or "A").strip().upper().replace("♯", "#").replace("♭", "B")
    pc = NOTE_TO_PC.get(name[:2], NOTE_TO_PC.get(name[:1], 9))
    intervals = MODE_INTERVALS.get(
        str(mode or "").strip().lower(),
        _MINOR if "min" in str(mode or "").lower() else _MAJOR,
    )
    base = 48 + pc
    return [base + iv for iv in intervals]


def _diatonic_triad(scale: list[int], degree: int) -> list[int]:
    """Stack diatonic thirds on a 0-based scale degree → 3 MIDI notes."""
    out = []
    for step in (0, 2, 4):
        d = degree + step
        out.append(scale[d % 7] + 12 * (d // 7))
    return out


# ---------------------------------------------------------------------------
# Genre → engine voicing.
#
# A genre profile tells the renderer HOW each role should sound: drum-kit
# flavour, bass timbre, default keys voicing, and a per-bus level trim so the
# mix sits in the right pocket for the style. Roles in plan["instrument_palette"]
# are mapped onto concrete synth voices.
# ---------------------------------------------------------------------------

_GENRE_VOICE = {
    "pop":     {"kit": "acoustic", "bass": "electric", "keys": "piano",  "warmth": 0.45},
    "rock":    {"kit": "rock",     "bass": "pick",     "keys": "piano",  "warmth": 0.35},
    "metal":   {"kit": "rock",     "bass": "pick",     "keys": "piano",  "warmth": 0.20},
    "country": {"kit": "acoustic", "bass": "electric", "keys": "piano",  "warmth": 0.55},
    "folk":    {"kit": "brush",    "bass": "upright",  "keys": "piano",  "warmth": 0.6},
    "acoustic":{"kit": "brush",    "bass": "upright",  "keys": "piano",  "warmth": 0.6},
    "hip hop": {"kit": "trap",     "bass": "808",      "keys": "epiano", "warmth": 0.5},
    "trap":    {"kit": "trap",     "bass": "808",      "keys": "synth",  "warmth": 0.4},
    "r&b":     {"kit": "soft",     "bass": "electric", "keys": "epiano", "warmth": 0.6},
    "edm":     {"kit": "edm",      "bass": "sub",      "keys": "synth",  "warmth": 0.3},
    "dance":   {"kit": "edm",      "bass": "sub",      "keys": "synth",  "warmth": 0.3},
    "house":   {"kit": "edm",      "bass": "sub",      "keys": "synth",  "warmth": 0.35},
    "lofi":    {"kit": "lofi",     "bass": "electric", "keys": "epiano", "warmth": 0.7},
    "jazz":    {"kit": "brush",    "bass": "upright",  "keys": "piano",  "warmth": 0.55},
    "blues":   {"kit": "brush",    "bass": "upright",  "keys": "piano",  "warmth": 0.5},
    "latin":   {"kit": "acoustic", "bass": "electric", "keys": "piano",  "warmth": 0.5},
    "k-pop":   {"kit": "edm",      "bass": "sub",      "keys": "synth",  "warmth": 0.35},
    "classical": {"kit": "brush",  "bass": "upright",  "keys": "piano",  "warmth": 0.5},
}
_DEFAULT_VOICE = _GENRE_VOICE["pop"]

# Which palette roles drive which synth voice. (Several roles share a generator
# but with different parameters, set in _render_section_internal.)
_GUITAR_ROLES = {"acoustic_guitar", "electric_guitar", "dist_guitar", "pedal_steel"}
_KEYS_ROLES = {"piano", "epiano", "organ", "synth_keys", "keys", "chords", "melodic"}
_PAD_ROLES = {"pad", "pads", "strings"}
_PLUCK_ROLES = {"pluck", "arp", "bell"}
_LEAD_ROLES = {"lead", "synth_lead", "brass"}
_LOW_ROLES = {"bass", "sub_bass", "808"}


# ---------------------------------------------------------------------------
# Plan context
# ---------------------------------------------------------------------------

class _Ctx:
    __slots__ = ("bpm", "spb", "bpb", "scale", "seed", "palette", "swing",
                 "pattern", "structure", "energies", "minorish", "genre",
                 "voice", "density")


def _ctx(plan: dict) -> _Ctx:
    c = _Ctx()
    c.bpm = float(plan.get("bpm") or 120.0)
    if not (40.0 <= c.bpm <= 250.0):
        c.bpm = min(max(c.bpm, 40.0), 250.0)
    c.spb = 60.0 / c.bpm
    ts = str(plan.get("time_signature") or "4/4")
    try:
        c.bpb = max(2, min(12, int(ts.split("/")[0])))
    except (ValueError, IndexError):
        c.bpb = 4
    key = plan.get("key") or {}
    mode = str(key.get("mode") or "minor")
    c.scale = scale_midi_notes(key.get("tonic") or "A", mode)
    c.minorish = mode.strip().lower() in _MINORISH_MODES or "min" in mode.lower()
    c.seed = int(plan.get("seed") or 0)
    raw_palette = plan.get("instrument_palette") or []
    c.palette = [str(p).strip().lower() for p in raw_palette]
    if not c.palette:
        c.palette = ["drums", "bass", "piano"]
    groove = plan.get("groove") or {}
    try:
        c.swing = float(groove.get("swing") or 0.0)
    except (TypeError, ValueError):
        c.swing = 0.0
    c.swing = min(max(c.swing, 0.0), 1.0)
    c.pattern = str(groove.get("pattern_class") or "backbeat").strip().lower()
    structure = plan.get("structure") or _DEFAULT_STRUCTURE
    c.structure = [
        {"label": str(s.get("label", "verse")), "bars": max(1, int(s.get("bars", 4)))}
        for s in structure
    ]
    c.energies = list(plan.get("energy_targets") or [])
    c.genre = str(plan.get("genre") or "pop").strip().lower()
    c.voice = _GENRE_VOICE.get(c.genre, _DEFAULT_VOICE)
    try:
        c.density = float(plan.get("genre_density") or 0.55)
    except (TypeError, ValueError):
        c.density = 0.55
    return c


def _section_energy(ctx: _Ctx, label: str, index: int | None) -> float:
    if index is not None and 0 <= index < len(ctx.energies):
        try:
            return min(max(float(ctx.energies[index]), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    return _DEFAULT_ENERGY.get(label, 0.55)


def _section_index(plan: dict, section: dict) -> int | None:
    if "_index" in section:
        return int(section["_index"])
    structure = plan.get("structure") or []
    probe = {"label": section.get("label"), "bars": section.get("bars")}
    for i, s in enumerate(structure):
        if s.get("label") == probe["label"] and s.get("bars") == probe["bars"]:
            return i
    return None


def _progressions(plan: dict, ctx: _Ctx) -> dict:
    """Deterministic label → scale-degree progression map (chorus/verse differ;
    the chorus progression is identical at every chorus)."""
    hseed = int(plan.get("_harmony_seed", ctx.seed + 20011))
    rng = random.Random(hseed)
    pool = MINOR_PROGRESSIONS if ctx.minorish else MAJOR_PROGRESSIONS
    picks = rng.sample(pool, k=min(4, len(pool)))
    mapping = {
        "chorus": picks[0],
        "verse": picks[1 % len(picks)],
        "bridge": picks[2 % len(picks)],
        "prechorus": picks[3 % len(picks)],
    }
    mapping["intro"] = mapping["verse"]
    mapping["outro"] = mapping["verse"]
    mapping["inst"] = mapping["chorus"]
    return mapping


def _prog_for(plan: dict, ctx: _Ctx, label: str) -> list[int]:
    table = _progressions(plan, ctx)
    return table.get(label, table["verse"])


# ---------------------------------------------------------------------------
# Filters / envelopes / oscillators (all vectorized)
# ---------------------------------------------------------------------------

def _lowpass(x: np.ndarray, cutoff: float, sr: int, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, lfilter

    c = min(max(cutoff, 40.0), 0.45 * sr)
    b, a = butter(order, c / (0.5 * sr))
    return lfilter(b, a, x)


def _highpass(x: np.ndarray, cutoff: float, sr: int, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, lfilter

    c = min(max(cutoff, 20.0), 0.45 * sr)
    b, a = butter(order, c / (0.5 * sr), btype="high")
    return lfilter(b, a, x)


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    from scipy.signal import butter, lfilter

    lo = max(lo, 20.0)
    hi = min(hi, 0.45 * sr)
    b, a = butter(2, [lo / (0.5 * sr), hi / (0.5 * sr)], btype="band")
    return lfilter(b, a, x)


def _peak_eq(x: np.ndarray, freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """A single biquad peaking-EQ band (used for body resonances)."""
    from scipy.signal import lfilter

    w0 = 2.0 * np.pi * min(max(freq, 30.0), 0.45 * sr) / sr
    a = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * max(q, 1e-3))
    cw = np.cos(w0)
    b0 = 1 + alpha * a
    b1 = -2 * cw
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cw
    a2 = 1 - alpha / a
    return lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], x)


def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float,
          release: float) -> np.ndarray:
    n = max(n, 1)
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    if a + r >= n:
        half = max(1, n // 2)
        return np.concatenate([
            np.linspace(0.0, 1.0, half, endpoint=False),
            np.linspace(1.0, 0.0, n - half),
        ])
    body = n - a - r
    d = min(max(1, int(decay * sr)), body)
    return np.concatenate([
        np.linspace(0.0, 1.0, a, endpoint=False),
        np.linspace(1.0, sustain, d, endpoint=False),
        np.full(body - d, sustain),
        np.linspace(sustain, 0.0, r),
    ])


def _saw_detuned(freq: float, n: int, sr: int, detune: float = 0.003,
                 voices: int = 2) -> np.ndarray:
    t = np.arange(n) / sr
    out = np.zeros(n)
    offsets = np.linspace(-detune, detune, max(voices, 1))
    for d in offsets:
        ph = freq * (1.0 + d) * t
        out += 2.0 * (ph % 1.0) - 1.0
    return out / max(voices, 1)


def _saturate(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Gentle tanh saturation for analog-ish warmth / glue."""
    if drive <= 0:
        return x
    return np.tanh(drive * x) / np.tanh(drive) if drive != 1.0 else np.tanh(x)


def _add(bus: np.ndarray, start: int, sig: np.ndarray) -> None:
    if start >= len(bus) or sig.size == 0:
        return
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(len(bus), start + len(sig))
    bus[start:end] += sig[: end - start]


# ---------------------------------------------------------------------------
# Karplus-Strong plucked string — the realistic basis for guitars / plucks.
# ---------------------------------------------------------------------------

def _karplus_strong(freq: float, dur_sec: float, sr: int, *, damping: float = 0.498,
                    pick: float = 0.5, body: bool = True, brightness: float = 1.0,
                    seed: int = 0) -> np.ndarray:
    """A plucked string via the extended Karplus-Strong algorithm.

    ``damping`` ~0.5 → bright/long sustain; lower → duller/shorter. ``pick``
    blends the initial excitation between noise (1.0) and a softer filtered burst
    (0.0). ``body`` adds resonant body formants. Vectorized inner loop is a
    fixed-length IIR comb realized with scipy's lfilter for speed.
    """
    from scipy.signal import lfilter

    n = max(int(dur_sec * sr), 64)
    p = max(int(round(sr / max(freq, 20.0))), 2)
    rng = np.random.default_rng((seed ^ int(freq)) & 0x7FFFFFFF)
    burst = rng.standard_normal(p)
    if pick < 1.0:
        # Soften the pick: low-pass the excitation a touch (rounder attack).
        burst = _lowpass(burst, 2000.0 + 6000.0 * pick, sr)
    burst *= np.hanning(p) ** 0.25
    x = np.zeros(n)
    x[:p] = burst[:p]
    # Feedback comb with a 2-tap averaging lowpass in the loop (string damping).
    # y[i] = x[i] + damping*(y[i-p] + y[i-p-1]) ; realize via lfilter.
    a = np.zeros(p + 2)
    a[0] = 1.0
    a[p] = -damping
    a[p + 1] = -damping
    y = lfilter([1.0], a, x)
    # Brightness / decay shaping and a soft attack.
    t = np.arange(n) / sr
    y *= np.exp(-t * (2.2 / max(dur_sec, 0.2)))
    if brightness < 1.0:
        y = _lowpass(y, 2000.0 + 6000.0 * brightness, sr)
    if body:
        # Acoustic-guitar-ish body resonances.
        y = _peak_eq(y, 100.0, 4.0, 1.2, sr)
        y = _peak_eq(y, 215.0, 3.0, 1.5, sr)
    env = _adsr(n, sr, 0.002, 0.05, 0.7, min(0.15, 0.4 * dur_sec))
    out = y * env
    m = float(np.max(np.abs(out)))
    return out / m if m > 1e-9 else out


# ---------------------------------------------------------------------------
# Drum kit one-shots (deterministic, cached per (sr, kit))
# ---------------------------------------------------------------------------

_KIT_CACHE: dict[tuple, dict] = {}


def _norm1(x: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(x)))
    return x / p if p > 1e-9 else x


def _make_kick(sr: int, kit: str, rng) -> np.ndarray:
    if kit in ("trap", "edm"):
        n = int(0.5 * sr); t = np.arange(n) / sr
        f = 45.0 + 95.0 * np.exp(-t * 22.0)          # long, deep 808-ish boom
        k = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * (5.0 if kit == "trap" else 9.0))
    elif kit == "rock":
        n = int(0.3 * sr); t = np.arange(n) / sr
        f = 55.0 + 90.0 * np.exp(-t * 34.0)
        k = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 16.0)
    else:  # acoustic / soft / brush / lofi
        n = int(0.34 * sr); t = np.arange(n) / sr
        f = 52.0 + 96.0 * np.exp(-t * 30.0)
        k = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 14.0)
    cn = max(int(0.004 * sr), 8)
    click = rng.standard_normal(cn) * np.linspace(1.0, 0.0, cn)
    click_amt = 0.25 if kit in ("trap", "edm") else 0.45
    k[:cn] += click_amt * np.diff(np.concatenate([[0.0], click]))
    if kit == "lofi":
        k = _lowpass(k, 4500.0, sr)
    return _norm1(k)


def _make_snare(sr: int, kit: str, rng) -> np.ndarray:
    if kit == "brush":  # soft brushed snare for jazz/folk/acoustic
        n = int(0.18 * sr); t = np.arange(n) / sr
        noise = _bandpass(rng.standard_normal(n), 1200.0, 6000.0, sr) * np.exp(-t * 22.0)
        tone = np.sin(2.0 * np.pi * 210.0 * t) * np.exp(-t * 40.0)
        return _norm1(0.9 * _norm1(noise) + 0.25 * tone)
    if kit in ("trap", "edm"):  # tight, snappy clap-ish snare
        n = int(0.18 * sr); t = np.arange(n) / sr
        noise = _bandpass(rng.standard_normal(n), 900.0, 9000.0, sr) * np.exp(-t * 26.0)
        tone = np.sin(2.0 * np.pi * 200.0 * t) * np.exp(-t * 40.0)
        return _norm1(0.95 * _norm1(noise) + 0.35 * tone)
    if kit == "soft":  # r&b / soft snare
        n = int(0.2 * sr); t = np.arange(n) / sr
        noise = _bandpass(rng.standard_normal(n), 700.0, 6500.0, sr) * np.exp(-t * 20.0)
        tone = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 34.0)
        return _norm1(0.8 * _norm1(noise) + 0.45 * tone)
    # rock / acoustic / pop / lofi — full backbeat snare with body
    n = int(0.22 * sr); t = np.arange(n) / sr
    noise = _bandpass(rng.standard_normal(n), 600.0, 7800.0, sr) * np.exp(-t * 18.0)
    tone = np.sin(2.0 * np.pi * 186.0 * t) * np.exp(-t * 30.0)
    snare = 0.85 * _norm1(noise) + 0.5 * tone
    if kit == "lofi":
        snare = _lowpass(snare, 6000.0, sr)
    return _norm1(snare)


def _make_hats(sr: int, kit: str, rng) -> tuple[np.ndarray, np.ndarray]:
    hp = 8000.0 if kit in ("trap", "edm") else 7200.0
    n = int(0.07 * sr); t = np.arange(n) / sr
    decay = 80.0 if kit in ("trap", "edm") else 60.0
    hat_c = _norm1(_highpass(rng.standard_normal(n), hp, sr, order=4) * np.exp(-t * decay))
    n = int(0.4 * sr); t = np.arange(n) / sr
    hat_o = _norm1(_highpass(rng.standard_normal(n), hp - 400.0, sr, order=4) * np.exp(-t * 7.0))
    if kit == "lofi":
        hat_c = _lowpass(hat_c, 11000.0, sr)
        hat_o = _lowpass(hat_o, 11000.0, sr)
    return hat_c, hat_o


def _build_kit(sr: int, kit: str = "acoustic") -> dict:
    key = (sr, kit)
    if key in _KIT_CACHE:
        return _KIT_CACHE[key]
    rng = np.random.default_rng(0xE17501 ^ hash(kit) & 0xFFFFFF)
    hat_c, hat_o = _make_hats(sr, kit, rng)
    kit_d = {
        "kick": _make_kick(sr, kit, rng),
        "snare": _make_snare(sr, kit, rng),
        "hat_c": hat_c,
        "hat_o": hat_o,
    }
    _KIT_CACHE[key] = kit_d
    return kit_d


# ---------------------------------------------------------------------------
# Event generation (positions in beats relative to section start)
# ---------------------------------------------------------------------------

def _swing_pos(pos: float, swing: float) -> float:
    """Delay off-8ths by the groove swing amount (full swing → triplet feel)."""
    frac = pos % 1.0
    if abs(frac - 0.5) < 1e-6 and swing > 0.02:
        return float(np.floor(pos) + 0.5 + swing * (2.0 / 3.0 - 0.5))
    return pos


def _drum_events(pattern: str, bpb: int, bars: int, energy: float, swing: float,
                 rng: random.Random, fill_at_end: bool) -> list[tuple]:
    """Returns (beat_pos, instrument, velocity) events for the section."""
    ev: list[tuple] = []
    sw = max(swing, 0.6) if pattern == "shuffle" else swing
    vel_scale = 0.62 + 0.38 * energy

    if energy > 0.72:
        hat_div = 0.25
    elif energy > 0.38:
        hat_div = 0.5
    else:
        hat_div = 1.0

    mid = bpb // 2
    for bar in range(bars):
        b0 = bar * bpb
        kicks: list[float] = []
        snares: list[float] = []

        if pattern == "four_on_floor":
            kicks = [float(b) for b in range(bpb)]
            snares = [1.0, 3.0] if bpb >= 4 else [float(mid)]
        elif pattern == "halftime":
            kicks = [0.0]
            if rng.random() < 0.30 * energy:
                kicks.append(bpb - 0.5)
            snares = [2.0] if bpb >= 4 else [float(mid)]
        elif pattern == "shuffle":
            kicks = [0.0, float(mid)]
            snares = [1.0, 3.0] if bpb >= 4 else [float(mid)]
        elif pattern == "sparse":
            kicks = [0.0]
            snares = [2.0] if (bar % 2 == 1 and bpb >= 4) else []
        else:  # backbeat (default)
            kicks = [0.0, float(mid)]
            if rng.random() < 0.25 + 0.45 * energy:
                kicks.append(mid + 1.5 if mid + 1.5 < bpb else bpb - 0.5)
            snares = [1.0, 3.0] if bpb >= 4 else [float(mid)]

        for k in kicks:
            ev.append((b0 + _swing_pos(k, sw), "kick",
                       0.95 * vel_scale * rng.uniform(0.92, 1.0)))
        for s in snares:
            ev.append((b0 + s, "snare", 0.85 * vel_scale * rng.uniform(0.9, 1.0)))

        # Hats
        if pattern == "sparse":
            if energy > 0.35:
                for b in range(bpb):
                    ev.append((b0 + b, "hat_c", 0.30 * vel_scale))
        else:
            pos = 0.0
            while pos < bpb - 1e-9:
                accent = 0.55 if pos % 1.0 == 0.0 else 0.40
                ev.append((b0 + _swing_pos(pos, sw), "hat_c",
                           accent * vel_scale * rng.uniform(0.85, 1.0)))
                pos += hat_div
            if energy > 0.65 and pattern in ("four_on_floor", "backbeat"):
                if pattern == "four_on_floor" and energy > 0.8:
                    for b in range(bpb):
                        ev.append((b0 + _swing_pos(b + 0.5, sw), "hat_o",
                                   0.40 * vel_scale))
                else:
                    ev.append((b0 + _swing_pos(bpb - 0.5, sw), "hat_o",
                               0.42 * vel_scale))

    if fill_at_end and bars >= 1:
        last_beat = bars * bpb - 1.0
        ev = [e for e in ev if e[0] < last_beat - 1e-6]
        hits = 6 if energy > 0.6 else 4
        for i, frac in enumerate(np.linspace(0.0, 1.0, hits, endpoint=False)):
            ev.append((last_beat + float(frac), "snare",
                       (0.45 + 0.5 * i / max(hits - 1, 1)) * vel_scale))
    return ev


# ---------------------------------------------------------------------------
# Instrument voices
# ---------------------------------------------------------------------------

def _bass_note(freq: float, dur_sec: float, sr: int, energy: float,
               kind: str = "electric") -> np.ndarray:
    """Foundation bass. ``kind`` ∈ electric|pick|upright|sub|808."""
    n = max(int(dur_sec * sr), 32)
    t = np.arange(n) / sr
    sub = np.sin(2.0 * np.pi * freq * t)
    if kind == "sub":
        x = sub + 0.06 * np.sin(2.0 * np.pi * 2.0 * freq * t)
        x = _lowpass(x, 140.0, sr)
        env = _adsr(n, sr, 0.008, 0.06, 0.92, 0.04)
    elif kind == "808":
        # Pitch-gliding 808 with long decay (hip hop / trap).
        gl = freq * (1.0 + 0.4 * np.exp(-t * 30.0))   # slight pitch drop at attack
        x = np.sin(2.0 * np.pi * np.cumsum(gl) / sr)
        x = _saturate(x * 1.4, 1.4)
        x = _lowpass(x, 200.0 + 200.0 * energy, sr)
        env = _adsr(n, sr, 0.004, 0.4, 0.6, min(0.5, 0.6 * dur_sec))
    elif kind == "upright":
        saw = 2.0 * ((freq * t) % 1.0) - 1.0
        x = 0.8 * sub + 0.18 * saw
        x = _lowpass(x, 350.0 + 200.0 * energy, sr)
        x = _peak_eq(x, 120.0, 3.0, 1.0, sr)          # woody body
        env = _adsr(n, sr, 0.01, 0.12, 0.55, 0.05)
    elif kind == "pick":
        saw = 2.0 * ((freq * t) % 1.0) - 1.0
        sq = np.sign(np.sin(2.0 * np.pi * freq * t))
        x = 0.7 * sub + 0.3 * saw + 0.1 * sq
        x = _saturate(x * 1.2, 1.2)
        x = _lowpass(x, 700.0 + 900.0 * energy, sr)   # picked = brighter
        env = _adsr(n, sr, 0.004, 0.06, 0.75, 0.03)
    else:  # electric (fingered)
        saw = 2.0 * ((freq * t) % 1.0) - 1.0
        x = 0.85 * sub + 0.22 * saw
        x = _lowpass(x, 180.0 + 520.0 * energy, sr)
        env = _adsr(n, sr, 0.006, 0.08, 0.8, 0.03)
    return x * env


def _piano_note(midi: int, dur_sec: float, sr: int, energy: float) -> np.ndarray:
    """Acoustic-piano-ish voice: a few inharmonic partials + hammer thump."""
    n = max(int(dur_sec * sr), 64)
    t = np.arange(n) / sr
    f0 = _hz(midi)
    x = np.zeros(n)
    # Partials with slight inharmonicity and per-partial decay (bright→dull).
    for k, amp in ((1, 1.0), (2, 0.5), (3, 0.28), (4, 0.16), (6, 0.08)):
        inh = 1.0 + 0.0006 * k * k
        x += amp * np.sin(2.0 * np.pi * f0 * k * inh * t) * np.exp(-t * (1.6 + 0.5 * k))
    # Hammer thump.
    x[: max(int(0.004 * sr), 8)] += 0.3 * np.random.default_rng(midi).standard_normal(
        max(int(0.004 * sr), 8))
    x = _lowpass(x, 4000.0 + 4000.0 * energy, sr)
    env = _adsr(n, sr, 0.003, 0.4, 0.45, min(0.25, 0.4 * dur_sec))
    return x * env


def _epiano_note(midi: int, dur_sec: float, sr: int, energy: float) -> np.ndarray:
    """Electric-piano (Rhodes-ish): sine fundamental + bell-like tine FM."""
    n = max(int(dur_sec * sr), 64)
    t = np.arange(n) / sr
    f0 = _hz(midi)
    tine_env = np.exp(-t * 18.0)                       # bright tine attack
    mod = 2.0 * tine_env * np.sin(2.0 * np.pi * f0 * 2.0 * t)
    x = np.sin(2.0 * np.pi * f0 * t + mod) * np.exp(-t * 1.4)
    x += 0.25 * np.sin(2.0 * np.pi * f0 * 3.0 * t) * tine_env
    x = _saturate(x * 1.1, 1.1)
    env = _adsr(n, sr, 0.003, 0.25, 0.55, min(0.2, 0.4 * dur_sec))
    return x * env


def _organ_note(midi: int, dur_sec: float, sr: int) -> np.ndarray:
    """Drawbar organ: stacked octave/fifth sines, fast attack, slight vibrato."""
    n = max(int(dur_sec * sr), 64)
    t = np.arange(n) / sr
    f0 = _hz(midi)
    vib = 1.0 + 0.004 * np.sin(2.0 * np.pi * 6.0 * t)
    x = np.zeros(n)
    for mult, amp in ((1.0, 1.0), (2.0, 0.6), (3.0, 0.4), (4.0, 0.25), (0.5, 0.3)):
        x += amp * np.sin(2.0 * np.pi * f0 * mult * vib * t)
    env = _adsr(n, sr, 0.006, 0.05, 0.95, 0.06)
    return x * env


def _keys_chord(midis: list[int], dur_sec: float, sr: int, energy: float,
                voice: str = "piano") -> np.ndarray:
    """Render a chord with the chosen keyboard voice."""
    n = max(int(dur_sec * sr), 64)
    x = np.zeros(n)
    if voice == "piano":
        for m in midis:
            x[: len(x)] += _piano_note(m, dur_sec, sr, energy)[: n]
    elif voice == "epiano":
        for m in midis:
            x[: len(x)] += _epiano_note(m, dur_sec, sr, energy)[: n]
    elif voice == "organ":
        for m in midis:
            x[: len(x)] += _organ_note(m, dur_sec, sr)[: n]
    else:  # synth (detuned saw stack, lowpassed — clean modern keys)
        for m in midis:
            x += _saw_detuned(_hz(m), n, sr, detune=0.004, voices=2)[: n]
        x = _lowpass(x, 900.0 + 3000.0 * energy, sr)
        x *= _adsr(n, sr, 0.008, 0.2, 0.6, 0.08)
    return x / max(len(midis), 1)


def _guitar_chord(midis: list[int], dur_sec: float, sr: int, *, kind: str,
                  seed: int) -> np.ndarray:
    """A strummed/plucked guitar chord built from Karplus-Strong strings."""
    n = max(int(dur_sec * sr), 64)
    out = np.zeros(n)
    bright = {"acoustic_guitar": 0.85, "electric_guitar": 0.7,
              "dist_guitar": 0.9, "pedal_steel": 0.8}.get(kind, 0.8)
    body = kind in ("acoustic_guitar", "pedal_steel")
    pick = 0.7 if kind == "acoustic_guitar" else 0.5
    strum = int(0.012 * sr)                             # spread strings in time
    for i, m in enumerate(sorted(midis)):
        s = _karplus_strong(_hz(m), dur_sec, sr, damping=0.495, pick=pick,
                            body=body, brightness=bright, seed=seed + i)
        _add(out, i * strum, s[: n - i * strum] if i * strum < n else s[:0])
    if kind == "dist_guitar":
        out = _saturate(out * 3.0, 3.0)                 # overdrive
        out = _lowpass(out, 5500.0, sr)
    elif kind == "electric_guitar":
        out = _saturate(out * 1.4, 1.4)
    return out


def _pad_chord(midis: list[int], dur_sec: float, sr: int, *, strings: bool = False) -> np.ndarray:
    n = max(int(dur_sec * sr), 64)
    x = np.zeros(n)
    for m in midis:
        x += _saw_detuned(_hz(m), n, sr, detune=0.007, voices=3)
    x /= max(len(midis), 1)
    if strings:
        # Bowed-strings shimmer: slow vibrato + brighter top.
        t = np.arange(n) / sr
        x *= (1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * t))
        x = _lowpass(x, 3200.0, sr)
        x = _peak_eq(x, 1500.0, 2.0, 1.0, sr)
    else:
        x = _lowpass(x, 1100.0, sr)
    attack = min(1.2, 0.35 * dur_sec)
    return x * _adsr(n, sr, attack, 0.3, 0.85, min(0.6, 0.3 * dur_sec))


def _pluck_note(midi: int, dur_sec: float, sr: int, *, kind: str, seed: int) -> np.ndarray:
    """Short plucky synth / bell for EDM/pop plucks and bells."""
    n = max(int(dur_sec * sr), 32)
    t = np.arange(n) / sr
    f0 = _hz(midi)
    if kind == "bell":
        x = (np.sin(2.0 * np.pi * f0 * t) * np.exp(-t * 3.0)
             + 0.5 * np.sin(2.0 * np.pi * f0 * 2.76 * t) * np.exp(-t * 5.0)
             + 0.3 * np.sin(2.0 * np.pi * f0 * 5.4 * t) * np.exp(-t * 8.0))
        env = _adsr(n, sr, 0.001, 0.2, 0.0, min(0.4, dur_sec))
    else:  # synth pluck (Karplus + filter)
        x = _karplus_strong(f0, dur_sec, sr, damping=0.49, pick=0.4, body=False,
                            brightness=0.7, seed=seed)
        x = _lowpass(x, 4000.0, sr)
        env = _adsr(n, sr, 0.002, 0.08, 0.25, min(0.2, dur_sec))
    return x[: n] * env


def _lead_note(midi: float, dur_sec: float, sr: int, *, kind: str = "lead") -> np.ndarray:
    n = max(int(dur_sec * sr), 32)
    t = np.arange(n) / sr
    f0 = _hz(midi)
    vib = 1.0 + 0.007 * np.sin(2.0 * np.pi * 5.3 * t) * np.clip((t - 0.12) / 0.1, 0.0, 1.0)
    ph = 2.0 * np.pi * np.cumsum(f0 * vib) / sr
    if kind == "synth_lead":
        x = (2.0 * ((f0 * np.cumsum(vib) / sr) % 1.0) - 1.0)   # saw lead
        x = _lowpass(x, 3500.0, sr)
        x = _saturate(x * 1.5, 1.5)
    elif kind == "brass":
        x = np.sin(ph) + 0.5 * np.sin(2.0 * ph) + 0.3 * np.sin(3.0 * ph) + 0.15 * np.sin(4.0 * ph)
        x = _saturate(x * 1.3, 1.3)
        x = _peak_eq(x, 1200.0, 3.0, 1.0, sr)
    else:  # sine-ish lead with light harmonics
        x = np.sin(ph) + 0.35 * np.sin(2.0 * ph) + 0.12 * np.sin(3.0 * ph)
    return x * _adsr(n, sr, 0.012, 0.1, 0.75, 0.05)


# ---------------------------------------------------------------------------
# Chorus lead melody — generated entirely from the seed (never from reference)
# ---------------------------------------------------------------------------

def _lead_rhythms(bpb: int) -> list[list[float]]:
    if bpb == 4:
        return _LEAD_RHYTHMS_4
    quarters = [1.0] * bpb
    eighths_end = [1.0] * (bpb - 1) + [0.5, 0.5]
    syncopated = [1.5, 0.5] + [1.0] * (bpb - 2) if bpb >= 2 else quarters
    return [quarters, eighths_end, syncopated]


def _chorus_melody(plan: dict, ctx: _Ctx) -> tuple[list[tuple], int]:
    """One melody per plan (re-used at every chorus). Returns (notes, bars)
    where notes are (beat_offset, dur_beats, midi)."""
    mseed = int(plan.get("_melody_seed", ctx.seed + 10007))
    rng = random.Random((mseed * 2654435761) % (2 ** 31))

    bars = 8
    for s in ctx.structure:
        if s["label"] == "chorus":
            bars = s["bars"]
            break

    prog = _prog_for(plan, ctx, "chorus")
    ext = sorted(m + 12 * o for o in (1, 2) for m in ctx.scale)  # 2 octaves up
    lo, hi = 2, len(ext) - 3

    rhythms = _lead_rhythms(ctx.bpb)
    motif_a = rng.choice(rhythms)
    motif_b = rng.choice([r for r in rhythms if r is not motif_a] or rhythms)

    idx = rng.randint(len(ext) // 2 - 2, len(ext) // 2 + 2)
    notes: list[tuple] = []
    for bar in range(bars):
        rhythm = motif_a if bar % 2 == 0 else motif_b
        if bar % 4 == 3 and rng.random() < 0.5:
            rhythm = rng.choice(rhythms)
        degree = prog[bar % len(prog)]
        chord_pcs = {m % 12 for m in _diatonic_triad(ctx.scale, degree)}
        pos = float(bar * ctx.bpb)
        for j, d in enumerate(rhythm):
            step = (rng.choice([-1, 0, 1]) if j == 0
                    else rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3]))
            idx = min(max(idx + step, lo), hi)
            if j == 0 or j == len(rhythm) - 1:
                if ext[idx] % 12 not in chord_pcs:
                    for off in (1, -1, 2, -2, 3, -3):
                        k = idx + off
                        if lo <= k <= hi and ext[k] % 12 in chord_pcs:
                            idx = k
                            break
            if 0 < j < len(rhythm) - 1 and rng.random() < 0.1:
                pos += d
                continue
            notes.append((pos, d * 0.95, ext[idx]))
            pos += d
    return notes, bars


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------

_TAIL_SEC = 0.8


def _haas(sig: np.ndarray, delay_sec: float, sr: int) -> np.ndarray:
    d = max(int(delay_sec * sr), 1)
    if d >= len(sig):
        return sig.copy()
    return np.concatenate([np.zeros(d), sig[:-d]])


def _palette_voices(ctx: _Ctx) -> dict:
    """Resolve the abstract palette roles into concrete engine flags/voices."""
    pal = set(ctx.palette)
    v = {
        "drums": "drums" in pal,
        "low": None,          # bass voice kind, or None
        "keys": None,         # keyboard voice (piano/epiano/organ/synth), or None
        "guitar": None,       # guitar role to render as chords, or None
        "pad": False,
        "strings": False,
        "pluck": None,        # pluck/bell role
        "lead": None,         # lead role
    }
    # Low end — exactly one source (keep the low end clean).
    if "808" in pal:
        v["low"] = "808"
    elif "sub_bass" in pal:
        v["low"] = "sub"
    elif "bass" in pal:
        v["low"] = ctx.voice["bass"]
    # Keys — prefer explicit role, else the genre's default keyboard.
    if "epiano" in pal:
        v["keys"] = "epiano"
    elif "organ" in pal:
        v["keys"] = "organ"
    elif "synth_keys" in pal:
        v["keys"] = "synth"
    elif "piano" in pal or (pal & {"keys", "chords", "melodic"}):
        v["keys"] = ctx.voice["keys"]
    # Guitar — pick the most prominent guitar role present.
    for role in ("dist_guitar", "electric_guitar", "acoustic_guitar", "pedal_steel"):
        if role in pal:
            v["guitar"] = role
            break
    # Pads / strings.
    if "strings" in pal:
        v["pad"] = True
        v["strings"] = True
    if pal & {"pad", "pads"}:
        v["pad"] = True
    # Plucks / bells.
    if "bell" in pal:
        v["pluck"] = "bell"
    elif pal & {"pluck", "arp"}:
        v["pluck"] = "pluck"
    # Lead / topline.
    if "synth_lead" in pal:
        v["lead"] = "synth_lead"
    elif "brass" in pal:
        v["lead"] = "brass"
    elif "lead" in pal:
        v["lead"] = "lead"
    return v


def _bus_signals(plan: dict, section: dict, sr: int) -> tuple[dict, _Ctx, str, float]:
    """Synthesize every instrument bus for one section.

    Returns (buses, ctx, label, energy). Buses are mono float64 of length
    n_section + tail. Exposed so tests can measure per-bus RMS (mix balance).
    """
    ctx = _ctx(plan)
    label = str(section.get("label", "verse"))
    bars = max(1, int(section.get("bars", 4)))
    index = _section_index(plan, section)
    energy = _section_energy(ctx, label, index)
    next_label = section.get("_next_label")

    spb, bpb = ctx.spb, ctx.bpb
    n = int(round(bars * bpb * spb * sr))
    tail = int(_TAIL_SEC * sr)
    total = n + tail

    rng = random.Random(ctx.seed * 1009 + 7919 * (index if index is not None else 0)
                        + _LABEL_CODE.get(label, 8))
    voices = _palette_voices(ctx)

    prog = _prog_for(plan, ctx, label)
    bar_chords = []
    for bar in range(bars):
        degree = prog[bar % len(prog)]
        triad = _diatonic_triad(ctx.scale, degree)
        if energy > 0.5 and rng.random() < 0.3:
            d7 = degree + 6
            triad = triad + [ctx.scale[d7 % 7] + 12 * (d7 // 7)]
        bar_chords.append((degree, triad))

    buses = {k: np.zeros(total) for k in
             ("kick", "snare", "hat_c", "hat_o", "bass", "keys", "guitar",
              "pad", "pluck", "lead")}

    def beat_idx(beat: float) -> int:
        return int(round(beat * spb * sr))

    # ---- drums ----
    pattern = ctx.pattern
    if label in ("intro", "outro") and energy < 0.5:
        pattern = "sparse"
    if voices["drums"] and energy >= 0.18:
        kit = _build_kit(sr, ctx.voice["kit"])
        fill = (next_label == "chorus")
        for beat, instr, vel in _drum_events(pattern, bpb, bars, energy,
                                             ctx.swing, rng, fill):
            _add(buses[instr], beat_idx(beat), kit[instr] * vel)

    # ---- bass / low end ----
    if voices["low"]:
        kind = voices["low"]
        sw = max(ctx.swing, 0.6) if pattern == "shuffle" else ctx.swing
        for bar in range(bars):
            b0 = bar * bpb
            degree, _triad = bar_chords[bar]
            root = ctx.scale[degree % 7] + 12 * (degree // 7) - 12  # octave 2
            if kind in ("sub", "808") or energy < 0.35:
                events = [(b0, root, bpb * 0.95)]
                if kind == "808" and energy >= 0.5 and rng.random() < 0.5:
                    events = [(b0, root, bpb / 2 * 0.95), (b0 + bpb / 2, root, bpb / 2 * 0.95)]
            elif energy < 0.65:
                events = [(b0, root, 1.6)]
                if bpb >= 4:
                    events.append((b0 + bpb / 2, root, 1.4))
                if rng.random() < 0.5:
                    events.append((b0 + bpb - 0.5, root, 0.45))
            else:
                events = []
                pos = 0.0
                while pos < bpb - 1e-9:
                    if rng.random() > 0.08:
                        r = rng.random()
                        if pos == 0.0 or r < 0.60:
                            m = root
                        elif r < 0.78:
                            m = root + 12
                        else:
                            m = root + 7
                        events.append((b0 + _swing_pos(pos, sw), m, 0.45))
                    pos += 0.5
            for beat, midi, dur in events:
                sig = _bass_note(_hz(midi), dur * spb, sr, energy, kind=kind)
                _add(buses["bass"], beat_idx(beat), sig * rng.uniform(0.92, 1.0))

    # ---- keyboard chords ----
    if voices["keys"]:
        kv = voices["keys"]
        stab = rng.random() < 0.5
        for bar in range(bars):
            b0 = bar * bpb
            _degree, triad = bar_chords[bar]
            voiced = [m + 12 for m in triad]
            if energy < 0.35:
                hits = [(b0, bpb * 0.98)]
            elif energy < 0.65:
                hits = [(b0, bpb / 2 * 0.95), (b0 + bpb / 2, bpb / 2 * 0.95)]
            elif stab:
                hits = [(b0 + p + 0.5, 0.45) for p in range(bpb)]
            else:
                hits = [(b0 + p, 0.9) for p in range(bpb)]
            for beat, dur in hits:
                sig = _keys_chord(voiced, dur * spb, sr, energy, voice=kv)
                _add(buses["keys"], beat_idx(_swing_pos(beat, ctx.swing)),
                     sig * rng.uniform(0.9, 1.0))

    # ---- guitar (strummed / picked Karplus-Strong) ----
    if voices["guitar"]:
        gkind = voices["guitar"]
        for bar in range(bars):
            b0 = bar * bpb
            _degree, triad = bar_chords[bar]
            voiced = [m + 12 for m in triad] + [triad[0] + 24]
            if energy < 0.4:
                hits = [(b0, bpb * 0.95)]                # let chords ring
            elif energy < 0.7:
                hits = [(b0 + p, 1.0) for p in range(0, bpb, max(1, bpb // 2))]
            else:
                hits = [(b0 + p * 0.5, 0.5) for p in range(bpb * 2)]   # 8th strums
            for beat, dur in hits:
                sig = _guitar_chord(voiced, dur * spb + 0.3, sr, kind=gkind,
                                    seed=ctx.seed + bar * 7 + int(beat * 4))
                _add(buses["guitar"], beat_idx(_swing_pos(beat, ctx.swing)),
                     sig * rng.uniform(0.85, 1.0))

    # ---- pad / strings ----
    if voices["pad"]:
        bar = 0
        while bar < bars:
            degree, triad = bar_chords[bar]
            span = 1
            while bar + span < bars and bar_chords[bar + span][0] == degree:
                span += 1
            voiced = [m + 12 for m in triad] + [triad[0] + 24]
            sig = _pad_chord(voiced, span * bpb * spb + 0.4, sr, strings=voices["strings"])
            _add(buses["pad"], beat_idx(bar * bpb), sig)
            bar += span

    # ---- pluck / bell (off-beat arpeggio sparkle) ----
    if voices["pluck"] and energy >= 0.4:
        pk = voices["pluck"]
        for bar in range(bars):
            b0 = bar * bpb
            _degree, triad = bar_chords[bar]
            arp = [m + 24 for m in triad]
            step = 0.5
            for j in range(int(bpb / step)):
                m = arp[j % len(arp)]
                sig = _pluck_note(m, step * spb * 1.1, sr, kind=pk,
                                  seed=ctx.seed + bar * 13 + j)
                _add(buses["pluck"], beat_idx(_swing_pos(b0 + j * step, ctx.swing)),
                     sig * 0.8)

    # ---- lead melody (chorus only; seeded random walk) ----
    if voices["lead"] and label == "chorus":
        notes, mel_bars = _chorus_melody(plan, ctx)
        section_beats = bars * bpb
        mel_beats = max(mel_bars * bpb, 1)
        offset = 0.0
        while offset < section_beats - 1e-6:
            for pos, dur, midi in notes:
                p = offset + pos
                if p >= section_beats - 1e-6:
                    break
                sig = _lead_note(midi, dur * spb, sr, kind=voices["lead"])
                _add(buses["lead"], beat_idx(p), sig)
            offset += mel_beats

    return buses, ctx, label, energy


# Mix levels — hats/cymbals are the QUIETEST; kick + bass are the foundation.
# Vocal-range instruments (keys/guitar/lead) sit moderate to leave space.
_BASE_LVL = {
    "kick": 0.98, "snare": 0.72, "hat_c": 0.20, "hat_o": 0.17,
    "bass": 0.92, "keys": 0.42, "guitar": 0.46, "pad": 0.26,
    "pluck": 0.24, "lead": 0.40,
}
# Stereo placement: kick/snare/bass centred; everything else tasteful off-centre.
_PAN = {"hat_c": 0.32, "hat_o": -0.28, "lead": 0.14, "pluck": -0.35, "guitar": 0.22}


def _mix_buses(buses: dict, ctx: _Ctx, label: str, sr: int) -> np.ndarray:
    total = len(next(iter(buses.values())))
    lvl = dict(_BASE_LVL)
    # Genre warmth darkens the very top (less hat sizzle, rounder mix).
    warmth = float(ctx.voice.get("warmth", 0.45))
    lvl["hat_c"] *= (1.0 - 0.35 * warmth)
    lvl["hat_o"] *= (1.0 - 0.35 * warmth)
    if label in ("intro", "outro"):
        lvl["keys"] *= 0.85
        lvl["guitar"] *= 0.85
        lvl["hat_o"] = 0.0

    def pan_gains(p: float) -> tuple[float, float]:
        a = (p + 1.0) * np.pi / 4.0
        return float(np.cos(a)), float(np.sin(a))

    left = np.zeros(total)
    right = np.zeros(total)
    # Centred foundation.
    for name in ("kick", "snare", "bass"):
        sig = buses[name] * lvl[name]
        left += sig * 0.7071
        right += sig * 0.7071
    # Panned mono elements.
    for name, p in _PAN.items():
        if name not in buses:
            continue
        gl, gr = pan_gains(p)
        sig = buses[name] * lvl[name]
        left += sig * gl
        right += sig * gr
    # Haas-widened chords and pad for width without phase smear in mono.
    keys = buses["keys"] * lvl["keys"]
    left += keys * 0.74
    right += _haas(keys, 0.012, sr) * 0.74
    pad = buses["pad"] * lvl["pad"]
    left += _haas(pad, 0.018, sr) * 0.74
    right += pad * 0.74
    return np.stack([left, right])


def _render_section_internal(plan: dict, section: dict, sr: int) -> np.ndarray:
    """Render one section + release tail → float64 (2, n_section + tail)."""
    buses, ctx, label, _energy = _bus_signals(plan, section, sr)
    return _mix_buses(buses, ctx, label, sr)


def section_bus_rms(plan: dict, section: dict, sr: int = SR) -> dict:
    """Debug hook: per-instrument-bus RMS (level-weighted, as mixed) for one
    section. Used by the smoke test to assert mix balance (hats < kick/bass).
    """
    buses, ctx, label, _energy = _bus_signals(plan, section, sr)
    lvl = dict(_BASE_LVL)
    warmth = float(ctx.voice.get("warmth", 0.45))
    lvl["hat_c"] *= (1.0 - 0.35 * warmth)
    lvl["hat_o"] *= (1.0 - 0.35 * warmth)
    out = {}
    for name, sig in buses.items():
        out[name] = float(core_audio.rms(sig * lvl.get(name, 1.0)))
    return out


def render_section(plan: dict, section: dict, sr: int = 44100) -> np.ndarray:
    """Render a single section → float32 stereo (2, n), peak-normalized −3 dBFS."""
    ctx = _ctx(plan)
    bars = max(1, int(section.get("bars", 4)))
    n = int(round(bars * ctx.bpb * ctx.spb * sr))
    out = _render_section_internal(plan, section, sr)[:, :n]
    out = np.tanh(1.25 * out) / np.tanh(1.25)
    return core_audio.normalize_peak(out.astype(np.float32), -3.0)


# ---------------------------------------------------------------------------
# Full song rendering
# ---------------------------------------------------------------------------

def _safe_progress(progress) -> Callable[[float, str], None]:
    def fn(frac: float, msg: str) -> None:
        if progress is None:
            return
        try:
            progress(min(max(float(frac), 0.0), 1.0), msg)
        except Exception:
            pass
    return fn


def render_song(plan: dict, progress=None) -> np.ndarray:
    """Render the full plan → float32 stereo (2, n) at 44.1 kHz, −3 dBFS peak.

    Sections are rendered on an arithmetic beat grid (sample-exact starts) with
    release tails ringing over into the following section.
    """
    sr = SR
    ctx = _ctx(plan)
    p = _safe_progress(progress)
    structure = ctx.structure

    starts = [0]
    for s in structure:
        starts.append(starts[-1] + s["bars"] * ctx.bpb)
    total_beats = starts[-1]
    n_total = int(round(total_beats * ctx.spb * sr))
    tail = int(_TAIL_SEC * sr)
    out = np.zeros((2, n_total + tail))

    n_sections = max(len(structure), 1)
    for i, sec in enumerate(structure):
        p(0.02 + 0.9 * i / n_sections,
          f"Synthesizing {sec['label']} ({i + 1}/{n_sections})")
        s2 = dict(sec)
        s2["_index"] = i
        s2["_next_label"] = structure[i + 1]["label"] if i + 1 < len(structure) else None
        chunk = _render_section_internal(plan, s2, sr)
        start_idx = int(round(starts[i] * ctx.spb * sr))
        end = min(out.shape[1], start_idx + chunk.shape[1])
        out[:, start_idx:end] += chunk[:, : end - start_idx]

    p(0.95, "Finalizing mix")
    out = out[:, :n_total]
    out = np.tanh(1.25 * out) / np.tanh(1.25)         # gentle glue / soft clip
    fade_in = min(int(0.01 * sr), out.shape[1])
    fade_out = min(int(0.4 * sr), out.shape[1])
    if fade_in > 1:
        out[:, :fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out > 1:
        out[:, -fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    out = core_audio.normalize_peak(out.astype(np.float32), -3.0)
    p(1.0, "Instrumental synthesized")
    return out
