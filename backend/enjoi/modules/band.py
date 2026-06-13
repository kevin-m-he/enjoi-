"""Real-instrument 'band' engine — a sample-based producer, not a synth.

Renders a genre-appropriate arrangement (drums, bass, guitar/keys, melody,
strings) using REAL recorded instrument samples from a General-MIDI SoundFont
(FluidR3_GM) via tinysoundfont, then mixes and volume-levels the stems into one
cohesive instrumental the way a producer would in a DAW.

Public API:
    available() -> bool
    render_band(plan, progress=None) -> np.ndarray  # (2, n) float32 @ 44.1 kHz

Heavy/optional deps (tinysoundfont, requests, pedalboard, pyloudnorm) are
imported lazily; generate.py falls back to the procedural synth if this engine
is unavailable.
"""
from __future__ import annotations

import math
import os
import random
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config, deps
from ..core.errors import PipelineError

SR = config.SAMPLE_RATE
SOUNDFONT_FILE = "FluidR3_GM.sf2"
SOUNDFONT_URL = "https://github.com/Jacalz/fluid-soundfont/raw/master/original-files/FluidR3_GM.sf2"
SOUNDFONT_BYTES = 148398306

# ---------------------------------------------------------------------------
# Music theory
# ---------------------------------------------------------------------------
NOTE_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]

# Diatonic progressions as 0-based scale degrees (triads built in-scale).
_MAJOR_PROGS = [[0, 4, 5, 3], [0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 4, 4], [0, 3, 0, 4]]
_MINOR_PROGS = [[0, 5, 2, 6], [0, 6, 5, 6], [0, 5, 3, 4], [0, 2, 6, 5], [0, 3, 0, 6]]

# GM programs (0-based). Drum kit lives on MIDI channel 9 (bank 128).
GM = {
    "piano": 0, "epiano": 4, "nylon_guitar": 24, "steel_guitar": 25,
    "clean_guitar": 27, "od_guitar": 29, "acoustic_bass": 32, "finger_bass": 33,
    "pick_bass": 34, "synth_bass": 38, "strings": 48, "slow_strings": 49,
    "synth_pad": 90, "organ": 19, "sax": 65,
}
# GM drum keys (channel 9).
KICK, SNARE, SIDE, CLAP, CHH, PHH, OHH, CRASH, RIDE, TOM_L, TOM_M, SHAKER, TAMB = (
    36, 38, 37, 39, 42, 44, 46, 49, 51, 45, 47, 70, 54)

# ---------------------------------------------------------------------------
# Genre → band recipe.  harmony/lead/bass map to GM instrument keys above.
# "feel" tunes drum pattern + swing; "drum_kit" picks velocities/elements.
# ---------------------------------------------------------------------------
_GENRES = {
    "country": dict(harmony="steel_guitar", harmony2="piano", lead="steel_guitar",
                    bass="acoustic_bass", strings="slow_strings", feel="country"),
    "folk": dict(harmony="steel_guitar", harmony2="piano", lead="nylon_guitar",
                 bass="acoustic_bass", strings="slow_strings", feel="folk"),
    "acoustic": dict(harmony="nylon_guitar", harmony2="piano", lead="nylon_guitar",
                     bass="acoustic_bass", strings="slow_strings", feel="folk"),
    "singer-songwriter": dict(harmony="piano", harmony2="steel_guitar", lead="piano",
                              bass="acoustic_bass", strings="slow_strings", feel="folk"),
    "rock": dict(harmony="od_guitar", harmony2="clean_guitar", lead="od_guitar",
                 bass="pick_bass", strings="strings", feel="rock"),
    "metal": dict(harmony="od_guitar", harmony2="od_guitar", lead="od_guitar",
                  bass="pick_bass", strings="strings", feel="rock"),
    "pop": dict(harmony="piano", harmony2="clean_guitar", lead="piano",
                bass="finger_bass", strings="strings", feel="pop"),
    "r&b": dict(harmony="epiano", harmony2="clean_guitar", lead="epiano",
                bass="finger_bass", strings="slow_strings", feel="rnb"),
    "soul": dict(harmony="epiano", harmony2="organ", lead="sax",
                 bass="finger_bass", strings="slow_strings", feel="rnb"),
    "gospel": dict(harmony="organ", harmony2="piano", lead="organ",
                   bass="finger_bass", strings="strings", feel="rnb"),
    "hip hop": dict(harmony="epiano", harmony2="piano", lead="epiano",
                    bass="synth_bass", strings="strings", feel="trap"),
    "rap": dict(harmony="epiano", harmony2="piano", lead="epiano",
                bass="synth_bass", strings="strings", feel="trap"),
    "trap": dict(harmony="epiano", harmony2="synth_pad", lead="epiano",
                 bass="synth_bass", strings="strings", feel="trap"),
    "lofi": dict(harmony="epiano", harmony2="piano", lead="epiano",
                 bass="finger_bass", strings="slow_strings", feel="lofi"),
    "jazz": dict(harmony="piano", harmony2="epiano", lead="sax",
                 bass="acoustic_bass", strings="slow_strings", feel="jazz"),
    "blues": dict(harmony="clean_guitar", harmony2="organ", lead="od_guitar",
                  bass="acoustic_bass", strings="slow_strings", feel="blues"),
    "edm": dict(harmony="synth_pad", harmony2="piano", lead="synth_pad",
                bass="synth_bass", strings="strings", feel="edm"),
    "dance": dict(harmony="synth_pad", harmony2="piano", lead="piano",
                  bass="synth_bass", strings="strings", feel="edm"),
    "house": dict(harmony="epiano", harmony2="synth_pad", lead="piano",
                  bass="synth_bass", strings="strings", feel="edm"),
    "latin": dict(harmony="nylon_guitar", harmony2="piano", lead="nylon_guitar",
                  bass="acoustic_bass", strings="strings", feel="latin"),
}
_DEFAULT_GENRE = dict(harmony="piano", harmony2="clean_guitar", lead="piano",
                      bass="finger_bass", strings="strings", feel="pop")

# Per-section intensity 0..1 (which layers play, how busy).
_SECTION_INTENSITY = {"intro": 0.35, "verse": 0.6, "prechorus": 0.75,
                      "chorus": 1.0, "bridge": 0.7, "outro": 0.4, "inst": 0.8}


# ---------------------------------------------------------------------------
# SoundFont resolution
# ---------------------------------------------------------------------------

def _soundfont_path() -> str | None:
    """Find FluidR3_GM.sf2 in known locations; download to models_dir if absent."""
    from pathlib import Path

    candidates = [
        config.models_dir() / SOUNDFONT_FILE,
        Path(__file__).resolve().parents[2] / ".wheels" / SOUNDFONT_FILE,
    ]
    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 1_000_000:
                return str(c)
        except OSError:
            continue
    # download to models_dir
    target = config.models_dir() / SOUNDFONT_FILE
    requests = deps.optional_import("requests")
    try:
        if requests is not None:
            with requests.get(SOUNDFONT_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = target.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(target)
        else:
            import urllib.request

            urllib.request.urlretrieve(SOUNDFONT_URL, str(target))
        if target.is_file() and target.stat().st_size > 1_000_000:
            return str(target)
    except Exception:
        return None
    return None


def available() -> bool:
    return deps.has("tinysoundfont")


# ---------------------------------------------------------------------------
# Plan context
# ---------------------------------------------------------------------------

class _Ctx:
    __slots__ = ("bpm", "spb", "bpb", "scale", "minorish", "seed", "genre",
                 "recipe", "structure", "rng")


def _ctx(plan: dict) -> _Ctx:
    c = _Ctx()
    c.bpm = float(plan.get("bpm") or 120.0)
    c.bpm = min(max(c.bpm, 50.0), 200.0)
    c.spb = 60.0 / c.bpm
    ts = str(plan.get("time_signature") or "4/4")
    try:
        c.bpb = max(2, min(12, int(ts.split("/")[0])))
    except (ValueError, IndexError):
        c.bpb = 4
    key = plan.get("key") or {}
    tonic = str(key.get("tonic") or "C").strip().upper()
    pc = NOTE_TO_PC.get(tonic[:2], NOTE_TO_PC.get(tonic[:1], 0))
    mode = str(key.get("mode") or "major").lower()
    c.minorish = "min" in mode
    intervals = _MINOR if c.minorish else _MAJOR
    c.scale = [48 + pc + iv for iv in intervals]  # octave-3 scale
    c.seed = int(plan.get("seed") or 0)
    c.rng = random.Random(c.seed or 1234)
    genre = "pop"
    for tag in plan.get("genre_tags") or []:
        t = str(tag).lower()
        for name in _GENRES:
            if name in t or t in name:
                genre = name
                break
    c.genre = genre
    c.recipe = _GENRES.get(genre, _DEFAULT_GENRE)
    structure = plan.get("structure") or [
        {"label": "intro", "bars": 2}, {"label": "verse", "bars": 8},
        {"label": "chorus", "bars": 8}, {"label": "verse", "bars": 8},
        {"label": "chorus", "bars": 8}, {"label": "outro", "bars": 4}]
    c.structure = [{"label": str(s.get("label", "verse")), "bars": max(1, int(s.get("bars", 4)))}
                   for s in structure]
    return c


def _triad(scale: list[int], degree: int) -> list[int]:
    return [scale[(degree + s) % 7] + 12 * ((degree + s) // 7) for s in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Arrangement → note events per stem.  Event = (start_sec, dur_sec, key, vel).
# ---------------------------------------------------------------------------

def _build_arrangement(ctx: _Ctx) -> dict:
    rng = ctx.rng
    bpb, spb = ctx.bpb, ctx.spb
    prog = rng.choice(_MINOR_PROGS if ctx.minorish else _MAJOR_PROGS)
    feel = ctx.recipe["feel"]

    drums: list[tuple] = []
    bass: list[tuple] = []
    harmony: list[tuple] = []
    lead: list[tuple] = []
    strings: list[tuple] = []

    bar = 0
    chorus_root_lift = 0
    for sec in ctx.structure:
        label = sec["label"]
        inten = _SECTION_INTENSITY.get(label, 0.6)
        for b in range(sec["bars"]):
            t0 = bar * bpb * spb
            degree = prog[(bar) % len(prog)]
            triad = _triad(ctx.scale, degree)
            root = ctx.scale[degree % 7] + 12 * (degree // 7) - 12  # bass octave

            # --- drums (velocity-balanced: kick strong, hats quiet) ---
            if inten >= 0.3:
                _drum_bar(drums, t0, bpb, spb, feel, inten, rng)

            # --- bass: root motion locked to kick ---
            if inten >= 0.45:
                _bass_bar(bass, t0, bpb, spb, root, triad, feel, inten, rng)

            # --- harmony comping (guitar strum / piano chords) ---
            voiced = [m + 12 for m in triad]
            _harmony_bar(harmony, t0, bpb, spb, voiced, feel, inten, rng)

            # --- strings pad on bigger sections ---
            if inten >= 0.7:
                strings.append((t0, bpb * spb * 0.98,
                                [m + 12 for m in triad] + [triad[0] + 24], int(46 * inten)))

            # --- lead melody only on chorus/inst, sparse, leaves room ---
            if label in ("chorus", "inst") and inten >= 0.8:
                _lead_bar(lead, t0, bpb, spb, ctx.scale, triad, rng)
            bar += 1

    return {"drums": drums, "bass": bass, "harmony": harmony,
            "lead": lead, "strings": strings}


def _swing(pos: float, amt: float) -> float:
    if amt > 0.02 and abs((pos % 1.0) - 0.5) < 1e-6:
        return math.floor(pos) + 0.5 + amt * (2.0 / 3.0 - 0.5)
    return pos


def _drum_bar(out, t0, bpb, spb, feel, inten, rng):
    def add(beat, key, vel):
        out.append((t0 + max(0.0, beat) * spb + rng.uniform(-0.004, 0.004),
                    0.18, key, max(1, min(127, int(vel)))))
    mid = bpb // 2
    if feel == "trap":
        add(0, KICK, 122); add(mid + 0.5, KICK, 112)
        if rng.random() < 0.5 * inten:
            add(bpb - 1, KICK, 100)
        add(mid, SNARE, 112)
        div = 0.25 if inten > 0.7 else 0.5
        p = 0.0
        while p < bpb - 1e-9:
            v = 52 + (18 if (p % 1.0 == 0.0) else 0)
            add(p, CHH, v)
            if inten > 0.85 and rng.random() < 0.18:  # hat roll
                add(p + div / 2, CHH, 44)
            p += div
    elif feel == "edm":
        for k in range(bpb):
            add(k, KICK, 120)
        for k in range(bpb):
            add(k + 0.5, OHH, 50)
        add(mid, CLAP, 100)
    elif feel in ("folk", "country", "lofi", "jazz", "blues"):
        add(0, KICK, 104)
        if bpb >= 4:
            add(2, KICK, 92)
        for s in ([1, 3] if bpb >= 4 else [mid]):
            add(s, SNARE if feel != "jazz" else SIDE, 92)
        # soft shaker/ride 8ths, quiet
        p = 0.0
        while p < bpb - 1e-9:
            add(_swing(p, 0.2 if feel in ("jazz", "blues") else 0.1),
                SHAKER if feel in ("folk", "country", "lofi") else RIDE, 38)
            p += 0.5
    else:  # pop / rock / rnb / latin
        add(0, KICK, 118)
        add(mid, KICK if feel == "rock" else SNARE, 96 if feel == "rock" else 0) if False else None
        add(0, KICK, 118)
        if rng.random() < 0.5 + 0.4 * inten:
            add(mid + 1.5 if mid + 1.5 < bpb else bpb - 0.5, KICK, 90)
        for s in ([1, 3] if bpb >= 4 else [mid]):
            add(s, SNARE, 108)
        div = 0.25 if inten > 0.7 else 0.5
        p = 0.0
        while p < bpb - 1e-9:
            add(p, CHH, 50 + (16 if p % 1.0 == 0 else 0))
            p += div
    # crash on downbeat of loud sections
    if inten >= 0.95 and rng.random() < 0.5:
        add(0, CRASH, 78)


def _bass_bar(out, t0, bpb, spb, root, triad, feel, inten, rng):
    fifth = root + 7
    if feel in ("folk", "country", "jazz", "blues"):
        notes = [(0, root, bpb / 2), (bpb / 2, fifth, bpb / 2)] if bpb >= 4 else [(0, root, bpb)]
    elif feel == "trap":
        notes = [(0, root, bpb * 0.6)]
        if rng.random() < 0.6:
            notes.append((bpb * 0.66, root, bpb * 0.3))
    elif feel == "edm":
        notes = [(k, root, 0.9) for k in range(bpb)]
    else:  # pop / rock / rnb
        notes = [(0, root, 1.4)]
        if bpb >= 4:
            notes += [(1, root, 0.9), (2, fifth, 0.9), (3, root, 0.9)]
    for beat, key, dur in notes:
        out.append((t0 + beat * spb, dur * spb, key, int(96 * (0.7 + 0.3 * inten))))


def _harmony_bar(out, t0, bpb, spb, voiced, feel, inten, rng):
    vel = int((58 + 30 * inten))
    if feel in ("folk", "country", "acoustic", "latin"):
        # gentle strum: stagger chord notes, two strums per bar
        for beat in ([0, 2] if bpb >= 4 else [0]):
            for i, m in enumerate(voiced):
                out.append((t0 + beat * spb + i * 0.012, (bpb / 2) * spb * 0.9, m, vel))
    elif feel in ("pop", "rnb", "lofi", "jazz", "gospel"):
        out.append((t0, bpb * spb * 0.95, voiced, vel))  # sustained chord (list = stack)
    elif feel == "rock":
        for beat in range(bpb):  # 8th/quarter chord chugs
            for m in voiced:
                out.append((t0 + beat * spb, spb * 0.9, m, vel))
    elif feel in ("edm", "house"):
        p = 0.5
        while p < bpb - 1e-9:
            for m in voiced:
                out.append((t0 + p * spb, spb * 0.45, m, vel))
            p += 1.0
    else:
        out.append((t0, bpb * spb * 0.95, voiced, vel))


def _lead_bar(out, t0, bpb, spb, scale, triad, rng):
    ext = sorted(m + 12 for m in scale) + sorted(m + 24 for m in scale)
    chord_pcs = {m % 12 for m in triad}
    pos = 0.0
    idx = rng.randint(2, len(ext) - 4)
    rhythm = rng.choice([[1, 1, 2], [0.5, 0.5, 1, 1], [1, 0.5, 0.5, 1], [2, 1, 1]])
    for j, d in enumerate(rhythm):
        idx = min(max(idx + rng.choice([-2, -1, 0, 1, 2]), 1), len(ext) - 2)
        if (j == 0 or j == len(rhythm) - 1) and ext[idx] % 12 not in chord_pcs:
            for off in (1, -1, 2, -2):
                if 0 <= idx + off < len(ext) and ext[idx + off] % 12 in chord_pcs:
                    idx += off
                    break
        if rng.random() < 0.2:  # leave space
            pos += d
            continue
        out.append((t0 + pos * spb, d * spb * 0.9, ext[idx], 78))
        pos += d


# ---------------------------------------------------------------------------
# SoundFont rendering (one stem per instrument; event-boundary chunks)
# ---------------------------------------------------------------------------

def _render_stem(events: list[tuple], program: tuple[int, int], total_sec: float,
                 sf_path: str) -> np.ndarray:
    """Render one instrument's events → (2, n) float32. program=(bank,preset);
    bank 128 ⇒ drum channel."""
    import tinysoundfont as tsf

    bank, preset = program
    ch = 9 if bank == 128 else 0
    synth = tsf.Synth(samplerate=SR, gain=0.4)
    sfid = synth.sfload(sf_path)
    try:
        synth.program_select(ch, sfid, bank, preset)
    except Exception:
        synth.program_select(ch, sfid, 0, preset)

    tail = 1.5
    total_n = int((total_sec + tail) * SR)
    # actions: (sample, on?, key, vel)
    actions: list[tuple] = []
    for ev in events:
        start, dur = ev[0], ev[1]
        keys = ev[2] if isinstance(ev[2], (list, tuple)) else [ev[2]]
        vel = ev[3]
        s0 = int(start * SR)
        s1 = int((start + max(dur, 0.05)) * SR)
        for k in keys:
            actions.append((s0, True, int(k), int(vel)))
            actions.append((s1, False, int(k), 0))
    actions.sort(key=lambda a: a[0])

    chunks: list[np.ndarray] = []
    pos = 0
    ai = 0
    n_act = len(actions)
    while pos < total_n:
        nxt = actions[ai][0] if ai < n_act else total_n
        nxt = min(nxt, total_n)
        if nxt > pos:
            count = nxt - pos
            mv = synth.generate(count)
            chunks.append(np.frombuffer(mv, dtype=np.float32).reshape(-1, 2).copy())
            pos = nxt
        while ai < n_act and actions[ai][0] <= pos:
            _, on, key, vel = actions[ai]
            if on:
                synth.noteon(ch, key, vel)
            else:
                synth.noteoff(ch, key)
            ai += 1
    synth.sounds_off(ch)
    if not chunks:
        return np.zeros((2, total_n), dtype=np.float32)
    out = np.concatenate(chunks, axis=0).T  # (2, n)
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mixing / leveling (producer chain)
# ---------------------------------------------------------------------------

def _hpf(x: np.ndarray, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(2, max(20.0, cutoff) / (0.5 * SR), btype="high", output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def _pan(stem: np.ndarray, pan: float) -> np.ndarray:
    a = (pan + 1.0) * math.pi / 4.0
    gl, gr = math.cos(a), math.sin(a)
    mono = stem.mean(axis=0)
    return np.stack([mono * gl + (stem[0] - mono) * gl,
                     mono * gr + (stem[1] - mono) * gr]).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


# Per-stem mix targets: gain (dB), highpass (Hz), pan (-1..1).
_MIX = {
    "drums": (0.0, 30.0, 0.0),
    "bass": (-1.0, 30.0, 0.0),
    "harmony": (-7.5, 120.0, -0.25),
    "strings": (-12.0, 180.0, 0.3),
    "lead": (-9.0, 250.0, 0.18),
}


def _mix_stems(stems: dict, progress) -> np.ndarray:
    n = max((s.shape[1] for s in stems.values() if s.size), default=SR)
    bus = np.zeros((2, n), dtype=np.float32)
    # Level each stem to a reference RMS then apply its mix gain, so the balance
    # is consistent regardless of how hot the soundfont rendered each instrument.
    ref_rms = {"drums": 0.20, "bass": 0.16, "harmony": 0.10, "strings": 0.07, "lead": 0.09}
    for name, stem in stems.items():
        if stem.size == 0 or _rms(stem) < 1e-5:
            continue
        gain_db, hp, pan = _MIX.get(name, (-8.0, 120.0, 0.0))
        s = stem[:, :n] if stem.shape[1] >= n else np.pad(stem, ((0, 0), (0, n - stem.shape[1])))
        if hp > 25:
            s = _hpf(s, hp)
        cur = _rms(s)
        if cur > 1e-6:
            s = s * (ref_rms.get(name, 0.1) / cur)
        s = s * core_audio.db_to_lin(gain_db)
        if abs(pan) > 0.01:
            s = _pan(s, pan)
        bus[:, : s.shape[1]] += s[:, :n]
    return bus


def _master(bus: np.ndarray, loudness_lufs: float, progress) -> np.ndarray:
    pedalboard = deps.optional_import("pedalboard")
    x = bus
    if pedalboard is not None:
        try:
            from pedalboard import Compressor, HighShelfFilter, Limiter, Pedalboard, Reverb

            board = Pedalboard([
                Reverb(room_size=0.18, wet_level=0.10, dry_level=0.92, width=0.9),
                Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=12, release_ms=180),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5),
                Limiter(threshold_db=-1.0, release_ms=120),
            ])
            x = board(x, SR)
        except Exception:
            pass
    # Loudness normalize to target (mainstream), then true-peak guard.
    pyln = deps.optional_import("pyloudnorm")
    if pyln is not None:
        try:
            meter = pyln.Meter(SR)
            loud = meter.integrated_loudness(x.T)
            if math.isfinite(loud):
                x = (x * core_audio.db_to_lin(loudness_lufs - loud)).astype(np.float32)
        except Exception:
            pass
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > core_audio.db_to_lin(-1.0):
        x = x * (core_audio.db_to_lin(-1.0) / peak)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def _p(progress, frac, msg):
    if progress:
        try:
            progress(min(max(frac, 0.0), 1.0), msg)
        except Exception:
            pass


def render_band(plan: dict, progress: Callable | None = None) -> np.ndarray:
    """Render a real-instrument, mixed, volume-leveled instrumental → (2, n)."""
    if not available():
        raise PipelineError("Real-instrument engine unavailable (tinysoundfont missing).")
    _p(progress, 0.02, "Loading real instruments (SoundFont)…")
    sf_path = _soundfont_path()
    if not sf_path:
        raise PipelineError("Could not obtain the instrument SoundFont.")

    ctx = _ctx(plan)
    _p(progress, 0.08, "Arranging the band…")
    arr = _build_arrangement(ctx)
    total_sec = sum(s["bars"] for s in ctx.structure) * ctx.bpb * ctx.spb

    programs = {
        "drums": (128, 0),
        "bass": (0, GM[ctx.recipe["bass"]]),
        "harmony": (0, GM[ctx.recipe["harmony"]]),
        "strings": (0, GM[ctx.recipe["strings"]]),
        "lead": (0, GM[ctx.recipe["lead"]]),
    }
    order = ["drums", "bass", "harmony", "strings", "lead"]
    stems: dict = {}
    for i, name in enumerate(order):
        _p(progress, 0.12 + 0.70 * i / len(order), f"Recording {name}…")
        evs = arr.get(name) or []
        if not evs:
            continue
        try:
            stems[name] = _render_stem(evs, programs[name], total_sec, sf_path)
        except Exception as exc:
            # one instrument failing must not kill the song
            _p(progress, 0.12 + 0.70 * i / len(order), f"{name} skipped ({type(exc).__name__})")
    if not stems:
        raise PipelineError("The instrument engine produced no audio.")

    _p(progress, 0.85, "Mixing the stems…")
    bus = _mix_stems(stems, progress)
    _p(progress, 0.93, "Leveling & mastering…")
    out = _master(bus, float(os.environ.get("ENJOI_INSTR_LUFS", "-11.0")), progress)
    _p(progress, 0.99, "Instrumental ready")
    return out
