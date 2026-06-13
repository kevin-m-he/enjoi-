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


def _render_soundfont(plan: dict, progress: Callable | None = None) -> np.ndarray:
    """Fallback engine: render via the General-MIDI SoundFont (sampled GM
    instruments) when no local loop library is present."""
    if not deps.has("tinysoundfont"):
        raise PipelineError("SoundFont engine unavailable (tinysoundfont missing).")
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


# ===========================================================================
# LOOP ENGINE — uses a local library of REAL commercial loops/one-shots,
# warped to the target tempo & key and arranged like a producer. This is the
# primary engine; the SoundFont path above is the fallback when no library
# exists. The library is the user's own licensed samples and is NOT shipped.
# ===========================================================================

import re as _re

_LIB_CACHE: dict = {}


def library_dir():
    from pathlib import Path

    env = os.environ.get("ENJOI_SAMPLE_LIB", "").strip()
    cands = ([Path(env)] if env else []) + [
        Path(__file__).resolve().parents[2] / "sample_library",
        config.data_dir() / "sample_library",
    ]
    for d in cands:
        try:
            if d.is_dir() and any(d.glob("*.wav")):
                return d
        except OSError:
            continue
    return None


def library_available() -> bool:
    return library_dir() is not None


_NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
            "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def _parse_bpm(name: str):
    for pat in (r'(\d{2,3})\s*BPM', r'BPM\s*(\d{2,3})', r'(\d{2,3})\s*bpm',
                r'_(\d{2,3})_', r'-\s*(\d{2,3})\s*-', r'(\d{2,3})bpm'):
        m = _re.search(pat, name, _re.I)
        if m and 50 <= int(m.group(1)) <= 200:
            return int(m.group(1))
    return None


def _parse_key(name: str):
    for m in _re.finditer(r'(?<![A-Za-z#b])([A-G])([#b]?)\s*(maj|min|m)\b', name):
        pc = _NOTE_PC.get(m.group(1).upper() + (m.group(2) or "").upper())
        if pc is not None:
            return pc, ("major" if m.group(3).lower() == "maj" else "minor")
    m = _re.search(r'[Kk]ey([A-G])([#b]?)(min|maj|m)?', name)
    if m:
        pc = _NOTE_PC.get(m.group(1).upper() + (m.group(2) or "").upper())
        if pc is not None:
            return pc, ("major" if (m.group(3) or "").lower() == "maj" else "minor")
    return None


def _categorize(name: str) -> str:
    n = name.lower()
    if "full drum" in n or "drum loop" in n or "drum_loop" in n or "full_drum" in n:
        return "drumloop"
    if "808" in n:
        return "b808"
    if "kick" in n:
        return "kick"
    if "snare" in n:
        return "snare"
    if "clap" in n:
        return "clap"
    if "crash" in n:
        return "crash"
    if "open hat" in n or "open_hat" in n or "openhat" in n:
        return "openhat"
    if "hihat" in n or "hi-hat" in n or "hat" in n:
        return "hat"
    if any(k in n for k in ("guitar", "banjo", "charango", "ukulele", "mandolin")):
        return "guitar"
    if any(k in n for k in ("pad", "texture", "swell", "atmos")):
        return "pad"
    if "arp" in n:
        return "arp"
    if "pluck" in n:
        return "pluck"
    if any(k in n for k in ("synth", "lead", "melody", "chords", "poly")):
        return "synth"
    return "other"


def _index_library() -> list[dict]:
    d = library_dir()
    if d is None:
        return []
    key = str(d)
    if _LIB_CACHE.get("dir") == key:
        return _LIB_CACHE["index"]
    import soundfile as sf

    idx = []
    for p in sorted(d.glob("*.wav")):
        name = p.name
        try:
            info = sf.info(str(p))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if dur < 0.4 or dur > 130:
            continue
        key_parsed = _parse_key(name)
        idx.append({
            "path": str(p), "name": name, "cat": _categorize(name),
            "bpm": _parse_bpm(name),
            "pc": key_parsed[0] if key_parsed else None,
            "mode": key_parsed[1] if key_parsed else None,
            "dur": dur,
        })
    _LIB_CACHE.clear()
    _LIB_CACHE.update({"dir": key, "index": idx})
    return idx


# ---- loop loading + warping ------------------------------------------------

def _load_loop(path: str) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    y = data.T
    if y.shape[0] == 1:
        y = np.vstack([y[0], y[0]])
    elif y.shape[0] > 2:
        y = y[:2]
    if sr != SR:
        y = np.stack([core_audio.resample(np.ascontiguousarray(ch), sr, SR) for ch in y])
    return np.ascontiguousarray(y, dtype=np.float32)


def _fold_bpm(native: float, target: float) -> float:
    n = float(native)
    if n <= 0:
        return target
    while target / n > 1.42:
        n *= 2.0
    while target / n < 0.71:
        n /= 2.0
    return n


def _warp(y: np.ndarray, native_bpm, target_bpm: float, semitones: float,
          is_drum: bool) -> np.ndarray:
    import librosa

    if native_bpm:
        rate = target_bpm / _fold_bpm(native_bpm, target_bpm)
        if abs(rate - 1.0) > 0.01:
            y = np.stack([librosa.effects.time_stretch(np.ascontiguousarray(ch), rate=float(rate))
                          for ch in y])
    if not is_drum and abs(semitones) >= 0.5:
        y = np.stack([librosa.effects.pitch_shift(np.ascontiguousarray(ch), sr=SR,
                                                  n_steps=float(semitones)) for ch in y])
    return np.ascontiguousarray(y, dtype=np.float32)


def _tile(y: np.ndarray, n: int) -> np.ndarray:
    """Loop (2, m) up to length n with a short equal-power crossfade at the seam."""
    m = y.shape[1]
    if m == 0:
        return np.zeros((2, n), dtype=np.float32)
    if m >= n:
        return y[:, :n]
    xf = int(min(0.03 * SR, m * 0.12))
    out = y.copy()
    while out.shape[1] < n + m:
        a, b = out, y
        if xf >= 2:
            t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)
            fade_o, fade_i = np.cos(t), np.sin(t)
            head = a[:, -xf:] * fade_o + b[:, :xf] * fade_i
            out = np.concatenate([a[:, :-xf], head, b[:, xf:]], axis=1)
        else:
            out = np.concatenate([a, b], axis=1)
    return out[:, :n]


# ---- selection -------------------------------------------------------------

def _semitones_to(target_pc, native_pc) -> float:
    if target_pc is None or native_pc is None:
        return 0.0
    return float(((target_pc - native_pc + 6) % 12) - 6)


def _pick(index, cats, target_pc, target_bpm, rng, prefer=(), avoid=(), mode=None):
    pool = [s for s in index if s["cat"] in cats]
    if not pool:
        return None
    def score(s):
        sc = 0.0
        nm = s["name"].lower()
        for kw in prefer:
            if kw in nm:
                sc += 3.0
        for kw in avoid:
            if kw in nm:
                sc -= 2.5
        if s["bpm"]:
            ratio = target_bpm / _fold_bpm(s["bpm"], target_bpm)
            sc -= abs(math.log2(max(ratio, 1e-3))) * 4.0  # penalize stretch
        if s["pc"] is not None and target_pc is not None:
            sc -= abs(_semitones_to(target_pc, s["pc"])) * 0.25
        if mode and s["mode"] == mode:
            sc += 0.8
        return sc
    pool.sort(key=score, reverse=True)
    top = pool[: max(1, min(3, len(pool)))]
    return rng.choice(top)


# ---- genre recipes (which loops + how loud the drums) ----------------------

_LOOP_GENRE = {
    "country": dict(gtr=("acoustic",), drum=0.5, b808=0.0, synth=0.15, pad=0.4),
    "folk": dict(gtr=("acoustic", "classical"), drum=0.45, b808=0.0, synth=0.12, pad=0.45),
    "acoustic": dict(gtr=("acoustic", "classical", "nylon"), drum=0.4, b808=0.0, synth=0.1, pad=0.4),
    "singer-songwriter": dict(gtr=("acoustic", "classical"), drum=0.45, b808=0.0, synth=0.1, pad=0.45),
    "lofi": dict(gtr=("lofi", "classical", "sadboi"), drum=0.6, b808=0.5, synth=0.4, pad=0.5),
    "latin": dict(gtr=("spanish", "latin", "classical"), drum=0.8, b808=0.7, synth=0.3, pad=0.3),
    "pop": dict(gtr=("acoustic", "classic"), drum=0.85, b808=0.7, synth=0.6, pad=0.4),
    "r&b": dict(gtr=("classic", "spanish"), drum=0.7, b808=0.8, synth=0.6, pad=0.5),
    "soul": dict(gtr=("classic",), drum=0.65, b808=0.7, synth=0.5, pad=0.5),
    "hip hop": dict(gtr=("spanish", "trap", "latin"), drum=1.0, b808=1.0, synth=0.6, pad=0.3),
    "rap": dict(gtr=("spanish", "trap"), drum=1.0, b808=1.0, synth=0.6, pad=0.3),
    "trap": dict(gtr=("spanish", "trap", "latin"), drum=1.0, b808=1.0, synth=0.7, pad=0.3),
    "rock": dict(gtr=("electric",), drum=0.9, b808=0.3, synth=0.3, pad=0.3),
    "edm": dict(gtr=(), drum=0.95, b808=0.9, synth=0.9, pad=0.6),
    "dance": dict(gtr=(), drum=0.95, b808=0.9, synth=0.9, pad=0.6),
    "house": dict(gtr=(), drum=0.95, b808=0.85, synth=0.85, pad=0.6),
}
_DEFAULT_LOOP_GENRE = dict(gtr=("acoustic", "classic", "spanish"), drum=0.7, b808=0.6,
                           synth=0.5, pad=0.4)

# Per-section layer gains (arrangement dynamics).
_LAYER = {
    "harmony": {"intro": 0.85, "verse": 1.0, "prechorus": 1.0, "chorus": 1.0,
                "bridge": 0.9, "outro": 0.8, "inst": 1.0},
    "drums": {"intro": 0.0, "verse": 0.8, "prechorus": 0.92, "chorus": 1.0,
              "bridge": 0.55, "outro": 0.3, "inst": 0.95},
    "bass": {"intro": 0.0, "verse": 0.85, "prechorus": 0.95, "chorus": 1.0,
             "bridge": 0.75, "outro": 0.35, "inst": 0.95},
    "synth": {"intro": 0.45, "verse": 0.3, "prechorus": 0.6, "chorus": 0.9,
              "bridge": 0.6, "outro": 0.3, "inst": 0.8},
    "pad": {"intro": 0.7, "verse": 0.5, "prechorus": 0.6, "chorus": 0.85,
            "bridge": 0.85, "outro": 0.6, "inst": 0.6},
}
_LOOP_MIX = {  # (gain_db, highpass_hz, pan), plus reference rms for consistent balance
    "drums": (-0.5, 28.0, 0.0, 0.20),
    "bass": (-1.5, 28.0, 0.0, 0.15),
    "harmony": (-4.0, 90.0, -0.15, 0.13),
    "synth": (-8.0, 200.0, 0.2, 0.07),
    "pad": (-11.0, 160.0, 0.3, 0.06),
}


def _section_envelope(role: str, ctx: _Ctx, n_total: int) -> np.ndarray:
    env = np.zeros(n_total, dtype=np.float32)
    table = _LAYER[role]
    cursor = 0
    ramp = int(0.06 * SR)
    for sec in ctx.structure:
        seclen = int(sec["bars"] * ctx.bpb * ctx.spb * SR)
        g = table.get(sec["label"], 0.7)
        end = min(n_total, cursor + seclen)
        if end > cursor:
            env[cursor:end] = g
        cursor = end
        if cursor >= n_total:
            break
    # smooth boundaries to avoid clicks
    if ramp > 2 and n_total > 4 * ramp:
        kernel = np.ones(ramp, dtype=np.float32) / ramp
        env = np.convolve(env, kernel, mode="same").astype(np.float32)
    return env


def _drum_bus(index, ctx, n_total, intensity, rng) -> np.ndarray:
    """Layer kick+snare+hat+clap (+full loop if present) into one drum stem,
    hats kept quiet. Time-stretched only (never pitch-shifted)."""
    target = ctx.bpm
    elems = [
        ("kick", ("kick",), 1.0), ("snare", ("snare",), 0.9),
        ("hat", ("hat",), 0.42), ("openhat", ("openhat",), 0.36),
        ("clap", ("clap",), 0.7), ("crash", ("crash",), 0.5),
    ]
    bus = np.zeros((2, n_total), dtype=np.float32)
    got = False
    # If a full drum loop fits well, prefer it as the spine.
    full = _pick(index, ("drumloop",), None, target, rng)
    if full is not None:
        y = _warp(_load_loop(full["path"]), full["bpm"], target, 0.0, True)
        bus += _tile(y, n_total)
        got = True
    for role, cats, lvl in elems:
        s = _pick(index, cats, None, target, rng)
        if s is None:
            continue
        y = _warp(_load_loop(s["path"]), s["bpm"], target, 0.0, True) * lvl
        bus[:, : n_total] += _tile(y, n_total)
        got = True
    return bus * intensity if got else bus


def _render_loops(plan: dict, progress) -> np.ndarray:
    index = _index_library()
    if not index:
        raise PipelineError("No sample library found.")
    ctx = _ctx(plan)
    rng = ctx.rng
    recipe = _LOOP_GENRE.get(ctx.genre, _DEFAULT_LOOP_GENRE)
    tonic_pc = ctx.scale[0] % 12
    n_total = int(sum(s["bars"] for s in ctx.structure) * ctx.bpb * ctx.spb * SR)
    if n_total < SR:
        n_total = SR

    stems: dict = {}

    _p(progress, 0.12, "Laying down the guitars…")
    gtr = _pick(index, ("guitar",), tonic_pc, ctx.bpm, rng,
                prefer=recipe["gtr"], mode=("minor" if ctx.minorish else "major"))
    if gtr is None:
        gtr = _pick(index, ("guitar", "synth", "pluck"), tonic_pc, ctx.bpm, rng)
    if gtr is not None:
        y = _warp(_load_loop(gtr["path"]), gtr["bpm"], ctx.bpm,
                  _semitones_to(tonic_pc, gtr["pc"]), False)
        stems["harmony"] = _tile(y, n_total)

    _p(progress, 0.35, "Programming the drums…")
    drums = _drum_bus(index, ctx, n_total, max(0.05, recipe["drum"]), rng)
    if float(np.abs(drums).max()) > 1e-4:
        stems["drums"] = drums

    if recipe["b808"] > 0.05:
        _p(progress, 0.55, "Dropping the 808/bass…")
        b = _pick(index, ("b808",), tonic_pc, ctx.bpm, rng)
        if b is not None:
            y = _warp(_load_loop(b["path"]), b["bpm"], ctx.bpm,
                      _semitones_to(tonic_pc, b["pc"]), False) * recipe["b808"]
            stems["bass"] = _tile(y, n_total)

    if recipe["synth"] > 0.05:
        _p(progress, 0.68, "Adding synths…")
        s = _pick(index, ("synth", "arp", "pluck", "pad"), tonic_pc, ctx.bpm, rng,
                  mode=("minor" if ctx.minorish else "major"))
        if s is not None:
            y = _warp(_load_loop(s["path"]), s["bpm"], ctx.bpm,
                      _semitones_to(tonic_pc, s["pc"]), False)
            stems["synth"] = _tile(y, n_total)

    if recipe["pad"] > 0.05:
        _p(progress, 0.76, "Warming it with pads…")
        s = _pick(index, ("pad", "synth"), tonic_pc, ctx.bpm, rng, prefer=("pad", "texture"))
        if s is not None:
            y = _warp(_load_loop(s["path"]), s["bpm"], ctx.bpm,
                      _semitones_to(tonic_pc, s["pc"]), False)
            stems["pad"] = _tile(y, n_total)

    if not stems:
        raise PipelineError("Could not select any loops from the library.")

    _p(progress, 0.84, "Arranging sections…")
    bus = np.zeros((2, n_total), dtype=np.float32)
    for name, stem in stems.items():
        gain_db, hp, pan, ref = _LOOP_MIX.get(name, (-8.0, 120.0, 0.0, 0.08))
        s = stem[:, :n_total]
        if s.shape[1] < n_total:
            s = np.pad(s, ((0, 0), (0, n_total - s.shape[1])))
        if hp > 25:
            s = _hpf(s, hp)
        cur = _rms(s)
        if cur > 1e-6:
            s = s * (ref / cur)
        s = s * core_audio.db_to_lin(gain_db)
        env = _section_envelope("bass" if name == "bass" else name, ctx, n_total) \
            if name in _LAYER else np.ones(n_total, dtype=np.float32)
        s = s * env
        if abs(pan) > 0.01:
            s = _pan(s, pan)
        bus += s

    _p(progress, 0.92, "Mixing & leveling…")
    out = _master(bus, float(os.environ.get("ENJOI_INSTR_LUFS", "-11.0")), progress)
    _p(progress, 0.99, "Instrumental ready")
    return out


def available() -> bool:  # noqa: F811  (final definition — loop OR soundfont)
    return library_available() or deps.has("tinysoundfont")


def render_band(plan: dict, progress: Callable | None = None) -> np.ndarray:
    """Render a real-sample, mixed, leveled instrumental → (2, n).

    Primary: the local loop library (real commercial loops warped to the target
    tempo & key and arranged). Fallback: the General-MIDI SoundFont.
    """
    if library_available() and os.environ.get("ENJOI_ENGINE", "").lower() != "soundfont":
        try:
            return _render_loops(plan, progress)
        except Exception as exc:
            _p(progress, 0.02, f"Loop engine fell back ({type(exc).__name__})")
    return _render_soundfont(plan, progress)
