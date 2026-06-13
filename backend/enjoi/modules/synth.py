"""Procedural instrumental engine — the guaranteed fallback when MusicGen is absent.

Pure numpy + scipy.signal synthesis (no ML dependencies). Renders a complete,
listenable instrumental from a generation plan (see docs/API_CONTRACT.md):
drums (kick/snare/hats), bass, chords/keys, optional pad, and a seeded lead
melody on chorus sections.

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


def _midi_hz(m: float) -> float:
    return float(440.0 * 2.0 ** ((m - 69) / 12.0))


# ---------------------------------------------------------------------------
# Plan context
# ---------------------------------------------------------------------------

class _Ctx:
    __slots__ = ("bpm", "spb", "bpb", "scale", "seed", "palette", "swing",
                 "pattern", "structure", "energies", "minorish")


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
    c.palette = {str(p).strip().lower() for p in raw_palette}
    if not c.palette:
        c.palette = {"drums", "bass", "piano"}
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


def _add(bus: np.ndarray, start: int, sig: np.ndarray) -> None:
    if start >= len(bus) or sig.size == 0:
        return
    if start < 0:
        sig = sig[-start:]
        start = 0
    end = min(len(bus), start + len(sig))
    bus[start:end] += sig[: end - start]


# ---------------------------------------------------------------------------
# Drum kit one-shots (deterministic, cached per sample rate)
# ---------------------------------------------------------------------------

_KIT_CACHE: dict[int, dict] = {}


def _norm1(x: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(x)))
    return x / p if p > 1e-9 else x


def _build_kit(sr: int) -> dict:
    if sr in _KIT_CACHE:
        return _KIT_CACHE[sr]
    rng = np.random.default_rng(0xE17501)

    # Kick: decaying sine sweep 150 -> 52 Hz + click transient.
    n = int(0.32 * sr)
    t = np.arange(n) / sr
    f = 52.0 + 98.0 * np.exp(-t * 30.0)
    kick = np.sin(2.0 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 14.0)
    cn = max(int(0.004 * sr), 8)
    click = rng.standard_normal(cn) * np.linspace(1.0, 0.0, cn)
    kick[:cn] += 0.45 * np.diff(np.concatenate([[0.0], click]))
    kick = _norm1(kick)

    # Snare: filtered noise burst + 186 Hz tone.
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    noise = _bandpass(rng.standard_normal(n), 600.0, 7800.0, sr) * np.exp(-t * 18.0)
    tone = np.sin(2.0 * np.pi * 186.0 * t) * np.exp(-t * 30.0)
    snare = _norm1(0.85 * _norm1(noise) + 0.5 * tone)

    # Hi-hats: highpassed noise, short closed / longer open.
    n = int(0.09 * sr)
    t = np.arange(n) / sr
    hat_c = _norm1(_highpass(rng.standard_normal(n), 7200.0, sr, order=4)
                   * np.exp(-t * 60.0))
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    hat_o = _norm1(_highpass(rng.standard_normal(n), 6800.0, sr, order=4)
                   * np.exp(-t * 7.0))

    kit = {"kick": kick, "snare": snare, "hat_c": hat_c, "hat_o": hat_o}
    _KIT_CACHE[sr] = kit
    return kit


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


def _bass_note(freq: float, dur_sec: float, sr: int, energy: float) -> np.ndarray:
    n = max(int(dur_sec * sr), 32)
    t = np.arange(n) / sr
    sub = np.sin(2.0 * np.pi * freq * t)
    saw = 2.0 * ((freq * t) % 1.0) - 1.0
    x = 0.85 * sub + 0.22 * saw
    x = _lowpass(x, 180.0 + 520.0 * energy, sr)
    return x * _adsr(n, sr, 0.006, 0.08, 0.8, 0.03)


def _keys_chord(midis: list[int], dur_sec: float, sr: int, energy: float) -> np.ndarray:
    n = max(int(dur_sec * sr), 64)
    x = np.zeros(n)
    for m in midis:
        x += _saw_detuned(_midi_hz(m), n, sr, detune=0.0035, voices=2)
    x /= max(len(midis), 1)
    x = _lowpass(x, 800.0 + 2600.0 * energy, sr)
    return x * _adsr(n, sr, 0.012, 0.25, 0.55, 0.08)


def _pad_chord(midis: list[int], dur_sec: float, sr: int) -> np.ndarray:
    n = max(int(dur_sec * sr), 64)
    x = np.zeros(n)
    for m in midis:
        x += _saw_detuned(_midi_hz(m), n, sr, detune=0.006, voices=3)
    x /= max(len(midis), 1)
    x = _lowpass(x, 900.0, sr)
    attack = min(1.2, 0.35 * dur_sec)
    return x * _adsr(n, sr, attack, 0.3, 0.85, min(0.5, 0.25 * dur_sec))


def _lead_note(midi: float, dur_sec: float, sr: int) -> np.ndarray:
    n = max(int(dur_sec * sr), 32)
    t = np.arange(n) / sr
    f0 = _midi_hz(midi)
    vib = 1.0 + 0.007 * np.sin(2.0 * np.pi * 5.3 * t) * np.clip((t - 0.12) / 0.1, 0.0, 1.0)
    ph = 2.0 * np.pi * np.cumsum(f0 * vib) / sr
    x = np.sin(ph) + 0.35 * np.sin(2.0 * ph) + 0.12 * np.sin(3.0 * ph)
    return x * _adsr(n, sr, 0.012, 0.1, 0.75, 0.05)


# ---------------------------------------------------------------------------
# Chorus lead melody — generated entirely from the seed (never from reference)
# ---------------------------------------------------------------------------

def _lead_rhythms(bpb: int) -> list[list[float]]:
    if bpb == 4:
        return _LEAD_RHYTHMS_4
    # Generic motifs for other meters: quarters / 8th pairs filling the bar.
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
                # Snap phrase boundaries to the bar's chord tones.
                if ext[idx] % 12 not in chord_pcs:
                    for off in (1, -1, 2, -2, 3, -3):
                        k = idx + off
                        if lo <= k <= hi and ext[k] % 12 in chord_pcs:
                            idx = k
                            break
            if 0 < j < len(rhythm) - 1 and rng.random() < 0.1:
                pos += d  # breathe: occasional rest mid-phrase
                continue
            notes.append((pos, d * 0.95, ext[idx]))
            pos += d
    return notes, bars


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------

_TAIL_SEC = 0.8

_KEYS_ALIASES = {"piano", "keys", "chords", "guitar", "melodic", "synth", "epiano", "organ"}
_PAD_ALIASES = {"pad", "pads", "strings"}


def _haas(sig: np.ndarray, delay_sec: float, sr: int) -> np.ndarray:
    d = max(int(delay_sec * sr), 1)
    if d >= len(sig):
        return sig.copy()
    return np.concatenate([np.zeros(d), sig[:-d]])


def _render_section_internal(plan: dict, section: dict, sr: int) -> np.ndarray:
    """Render one section + release tail → float64 (2, n_section + tail)."""
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
             ("kick", "snare", "hat_c", "hat_o", "bass", "keys", "pad", "lead")}

    def beat_idx(beat: float) -> int:
        return int(round(beat * spb * sr))

    # ---- drums ----
    pattern = ctx.pattern
    if label in ("intro", "outro") and energy < 0.5:
        pattern = "sparse"
    if "drums" in ctx.palette and energy >= 0.18:
        kit = _build_kit(sr)
        fill = (next_label == "chorus")
        for beat, instr, vel in _drum_events(pattern, bpb, bars, energy,
                                             ctx.swing, rng, fill):
            _add(buses[instr], beat_idx(beat), kit[instr] * vel)

    # ---- bass ----
    if "bass" in ctx.palette:
        sw = max(ctx.swing, 0.6) if pattern == "shuffle" else ctx.swing
        for bar in range(bars):
            b0 = bar * bpb
            degree, _triad = bar_chords[bar]
            root = ctx.scale[degree % 7] + 12 * (degree // 7) - 12  # octave 2
            if energy < 0.35:
                events = [(b0, root, bpb * 0.97)]
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
                sig = _bass_note(_midi_hz(midi), dur * spb, sr, energy)
                _add(buses["bass"], beat_idx(beat), sig * rng.uniform(0.92, 1.0))

    # ---- chords / keys ----
    if ctx.palette & _KEYS_ALIASES:
        stab = rng.random() < 0.5  # per-section pattern choice (seeded)
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
                sig = _keys_chord(voiced, dur * spb, sr, energy)
                _add(buses["keys"], beat_idx(_swing_pos(beat, ctx.swing)),
                     sig * rng.uniform(0.9, 1.0))

    # ---- pad ----
    if ctx.palette & _PAD_ALIASES:
        bar = 0
        while bar < bars:
            degree, triad = bar_chords[bar]
            span = 1
            while bar + span < bars and bar_chords[bar + span][0] == degree:
                span += 1
            voiced = [m + 12 for m in triad] + [triad[0] + 24]
            sig = _pad_chord(voiced, span * bpb * spb + 0.4, sr)
            _add(buses["pad"], beat_idx(bar * bpb), sig)
            bar += span

    # ---- lead melody (chorus only; seeded random walk) ----
    if label == "chorus":
        notes, mel_bars = _chorus_melody(plan, ctx)
        section_beats = bars * bpb
        mel_beats = max(mel_bars * bpb, 1)
        offset = 0.0
        while offset < section_beats - 1e-6:
            for pos, dur, midi in notes:
                p = offset + pos
                if p >= section_beats - 1e-6:
                    break
                sig = _lead_note(midi, dur * spb, sr)
                _add(buses["lead"], beat_idx(p), sig)
            offset += mel_beats

    # ---- stereo assembly ----
    lvl = {"kick": 0.95, "snare": 0.80, "hat_c": 0.45, "hat_o": 0.40,
           "bass": 0.80, "keys": 0.50, "pad": 0.30, "lead": 0.50}
    if label in ("intro", "outro"):
        lvl["keys"] *= 0.85
        lvl["hat_o"] = 0.0

    def pan_gains(p: float) -> tuple[float, float]:
        a = (p + 1.0) * np.pi / 4.0
        return float(np.cos(a)), float(np.sin(a))

    left = np.zeros(total)
    right = np.zeros(total)
    # Centered: kick, snare, bass
    for name in ("kick", "snare", "bass"):
        sig = buses[name] * lvl[name]
        left += sig * 0.7071
        right += sig * 0.7071
    # Panned hats and lead
    for name, p in (("hat_c", 0.3), ("hat_o", -0.25), ("lead", 0.12)):
        gl, gr = pan_gains(p)
        sig = buses[name] * lvl[name]
        left += sig * gl
        right += sig * gr
    # Haas-widened chords and pad
    keys = buses["keys"] * lvl["keys"]
    left += keys * 0.74
    right += _haas(keys, 0.012, sr) * 0.74
    pad = buses["pad"] * lvl["pad"]
    left += _haas(pad, 0.018, sr) * 0.74
    right += pad * 0.74

    return np.stack([left, right])


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
