"""Similarity slider → generation plan (spec §4.3, table columns 0/25/50/75/100).

Pure functions over the reference profile — no audio I/O, stdlib only.
The slider maps *style descriptors* (tempo, key, structure, energy shape,
palette, groove) to a plan dict for the generator. Melody, chords and the
reference audio itself are never copied at any value ("style, never substance").
"""
from __future__ import annotations

import hashlib
import math
import random

_COLUMNS = (0, 25, 50, 75, 100)
_TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
_ROLES = ["drums", "bass", "piano", "pad", "lead", "strings", "guitar"]
_PATTERNS = ("four_on_floor", "backbeat", "halftime", "shuffle", "sparse")

_GENRE_BPM = {
    "pop": (85, 135), "dance": (118, 134), "edm": (118, 140), "house": (120, 128),
    "hip hop": (70, 105), "rap": (70, 105), "trap": (70, 105), "r&b": (65, 105),
    "rock": (95, 165), "metal": (100, 180), "jazz": (80, 160), "blues": (60, 120),
    "lofi": (60, 90), "country": (80, 140), "folk": (80, 130), "latin": (88, 105),
    "acoustic": (75, 125), "classical": (60, 140), "k-pop": (90, 135),
}
_GENRE_PATTERN = {
    "pop": "backbeat", "rock": "backbeat", "metal": "backbeat", "country": "backbeat",
    "folk": "backbeat", "r&b": "backbeat", "k-pop": "backbeat",
    "dance": "four_on_floor", "edm": "four_on_floor", "house": "four_on_floor",
    "latin": "four_on_floor", "hip hop": "halftime", "rap": "halftime", "trap": "halftime",
    "jazz": "shuffle", "blues": "shuffle", "lofi": "sparse", "acoustic": "sparse",
    "classical": "sparse",
}
_LABEL_ENERGY = {
    "intro": 0.30, "verse": 0.55, "prechorus": 0.65, "chorus": 0.90,
    "bridge": 0.60, "outro": 0.30, "inst": 0.50,
}
_BAR_WEIGHT = {
    "intro": 0.5, "verse": 1.5, "prechorus": 0.8, "chorus": 1.2,
    "bridge": 0.8, "outro": 0.5, "inst": 1.0,
}
_PATTERN_PHRASE = {
    "four_on_floor": "four-on-the-floor beat",
    "backbeat": "steady backbeat groove",
    "halftime": "half-time groove",
    "shuffle": "shuffled swing groove",
    "sparse": "sparse minimal rhythm",
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_generation_plan(profile: dict, similarity: int, rng_seed: int | None = None) -> dict:
    """Map slider value 0..100 + reference profile → generation plan dict.

    Implements spec table 4.3 exactly at the 0/25/50/75/100 columns and
    interpolates behaviour between them. Reproducible via ``rng_seed`` (a
    deterministic seed is derived from the profile + similarity otherwise).
    """
    s = _clamp_similarity(similarity)
    col = _column(s)
    duration = float(profile.get("duration_sec") or 180.0)
    ref_bpm = _safe_float(profile.get("bpm"), 120.0)
    ref_tonic, ref_mode = _ref_key(profile)
    genre, bpm_range = _primary_genre(profile)

    seed = _derive_seed(profile, s, rng_seed)
    rng = random.Random(seed)

    bpm = _plan_bpm(s, ref_bpm, bpm_range, rng)
    tonic, mode = _plan_key(col, ref_tonic, ref_mode, rng)
    time_signature = (profile.get("time_signature") or "4/4") if col >= 50 else "4/4"
    beats_per_bar = {"3/4": 3, "6/8": 6}.get(time_signature, 4)
    total_bars = max(8, int(round(duration * bpm / 60.0 / beats_per_bar)))

    ref_struct = [
        {"label": str(sec.get("label", "verse")), "bars": max(1, int(sec.get("bars") or 1))}
        for sec in (profile.get("structure") or [])
    ]
    structure = _plan_structure(col, ref_struct, total_bars, rng)
    energy_targets = _plan_energy(col, structure, ref_struct, profile, rng)
    palette = _plan_palette(s, col, profile, rng)
    groove = _plan_groove(col, profile, genre, rng)

    plan: dict = {
        "similarity": s,
        "seed": int(seed),
        "target_duration_sec": round(duration, 2),
        "bpm": round(bpm, 1),
        "key": {"tonic": tonic, "mode": mode},
        "time_signature": time_signature,
        "structure": structure,
        "energy_targets": energy_targets,
        "instrument_palette": palette,
        "groove": groove,
        "prompt": _build_prompt(profile, bpm, tonic, mode, palette, groove, energy_targets),
        "summary": similarity_summary(profile, s),
    }
    if s >= 75:
        plan["energy_per_bar"] = _plan_energy_per_bar(s, structure, profile)
    return plan


def similarity_summary(profile: dict, similarity: int) -> str:
    """Deterministic human-readable label for the slider (no RNG needed)."""
    s = _clamp_similarity(similarity)
    col = _column(s)
    key_phrase = {
        0: "any key", 25: "related key", 50: "same mode",
        75: "same key family", 100: "same key",
    }[col]
    structure_phrase = {
        0: "free structure", 25: "loose structure with a chorus",
        50: "same section count", 75: "same section order",
        100: "same structure & bar lengths",
    }[col]
    if s >= 100:
        tempo_phrase = "exact tempo"
    elif s <= 0:
        tempo_phrase = "free tempo"
    else:
        tol = _tempo_tolerance(s)
        tempo_phrase = f"tempo within {max(1, int(round(tol * 100)))}%"
    groove_phrase = {
        0: "free groove", 25: "genre-typical groove", 50: "similar swing",
        75: "similar groove", 100: "matched groove",
    }[col]
    return (
        f"{s}% — {key_phrase}, {structure_phrase}, {tempo_phrase}, {groove_phrase}"
        " — melody & chords 100% original."
    )


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _clamp_similarity(similarity: int) -> int:
    try:
        return max(0, min(100, int(round(float(similarity)))))
    except (TypeError, ValueError):
        return 50


def _column(s: int) -> int:
    """Nearest spec-table column (ties resolve to the lower column)."""
    return min(_COLUMNS, key=lambda c: abs(c - s))


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation over sorted (x, y) points."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return points[-1][1]


def _tempo_tolerance(s: int) -> float:
    return _interp(float(s), [(0, 0.30), (25, 0.15), (50, 0.07), (75, 0.03), (100, 0.0)])


def _safe_float(value, default: float) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) and v > 0 else default
    except (TypeError, ValueError):
        return default


def _derive_seed(profile: dict, s: int, rng_seed: int | None) -> int:
    if rng_seed is not None:
        return int(rng_seed) % (2**31 - 1)
    basis = "|".join(
        str(profile.get(k, "")) for k in ("bpm", "duration_sec", "time_signature")
    ) + f"|{profile.get('key', {})}|{s}"
    digest = hashlib.sha256(basis.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


# ---------------------------------------------------------------------------
# per-attribute planners
# ---------------------------------------------------------------------------

def _primary_genre(profile: dict) -> tuple[str, tuple[float, float]]:
    for tag in profile.get("genre_tags") or []:
        t = str(tag).lower()
        for name, rng in _GENRE_BPM.items():
            if name in t or t in name:
                return name, rng
    return "pop", _GENRE_BPM["pop"]


def _plan_bpm(s: int, ref_bpm: float, bpm_range: tuple[float, float], rng: random.Random) -> float:
    if s >= 100:
        bpm = ref_bpm
    elif s <= 0:
        bpm = rng.uniform(*bpm_range)  # random in genre range
    else:
        tol = _tempo_tolerance(s)
        bpm = ref_bpm * (1.0 + rng.uniform(-tol, tol))
    return min(200.0, max(60.0, bpm))


def _ref_key(profile: dict) -> tuple[str, str]:
    key = profile.get("key") or {}
    tonic = str(key.get("tonic") or "C")
    tonic = _FLAT_TO_SHARP.get(tonic, tonic)
    if tonic not in _TONICS:
        tonic = "C"
    mode = "minor" if str(key.get("mode", "")).lower() == "minor" else "major"
    return tonic, mode


def _shift_tonic(tonic: str, semitones: int) -> str:
    return _TONICS[(_TONICS.index(tonic) + semitones) % 12]


def _related_keys(tonic: str, mode: str) -> list[tuple[str, str]]:
    """Relative, dominant and subdominant keys of (tonic, mode)."""
    relative = (_shift_tonic(tonic, -3), "minor") if mode == "major" else (_shift_tonic(tonic, 3), "major")
    return [relative, (_shift_tonic(tonic, 7), mode), (_shift_tonic(tonic, 5), mode)]


def _plan_key(col: int, ref_tonic: str, ref_mode: str, rng: random.Random) -> tuple[str, str]:
    if col == 100:
        return ref_tonic, ref_mode
    if col == 75:  # same key family: same tonic letter (either mode) or relative
        relative = _related_keys(ref_tonic, ref_mode)[0]
        other_mode = "minor" if ref_mode == "major" else "major"
        return rng.choice([(ref_tonic, ref_mode), (ref_tonic, other_mode), relative])
    if col == 50:  # same mode, free tonic
        return rng.choice(_TONICS), ref_mode
    if col == 25:  # related key
        return rng.choice(_related_keys(ref_tonic, ref_mode))
    return rng.choice(_TONICS), rng.choice(["major", "minor"])


# ---- structure ---------------------------------------------------------------

def _scale_bars(bars: list[int], total: int) -> list[int]:
    """Proportionally rescale bar counts to sum to ``total`` (each ≥ 1)."""
    raw = [max(1, int(b)) for b in bars] or [total]
    ssum = sum(raw)
    scaled = [max(1, int(round(b * total / ssum))) for b in raw]
    diff = total - sum(scaled)
    guard = 0
    while diff != 0 and guard < 10_000:
        order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
        for i in order:
            if diff == 0:
                break
            if diff > 0:
                scaled[i] += 1
                diff -= 1
            elif scaled[i] > 1:
                scaled[i] -= 1
                diff += 1
        guard += 1
    return scaled


def _template_sections(total_bars: int, rng: random.Random, jitter: float) -> list[dict]:
    """A sensible free song template scaled to ``total_bars``."""
    if total_bars >= 56:
        base = [("intro", 4), ("verse", 16), ("chorus", 8), ("verse", 16),
                ("chorus", 8), ("bridge", 8), ("chorus", 8), ("outro", 4)]
    elif total_bars >= 28:
        base = [("intro", 2), ("verse", 8), ("chorus", 8), ("verse", 8),
                ("chorus", 8), ("outro", 2)]
    else:
        base = [("intro", 2), ("verse", 8), ("chorus", 8), ("outro", 2)]
    bars = [b for _, b in base]
    if jitter > 0:
        bars = [max(1, int(round(b * rng.uniform(1.0 - jitter, 1.0 + jitter)))) for b in bars]
    bars = _scale_bars(bars, total_bars)
    return [{"label": lbl, "bars": b} for (lbl, _), b in zip(base, bars)]


def _labels_for_count(n: int, rng: random.Random) -> list[str]:
    """A sensible label sequence of exactly ``n`` sections, with ≥1 chorus."""
    if n <= 1:
        return ["chorus"]
    if n == 2:
        return ["verse", "chorus"]
    mid = ["verse" if i % 2 == 0 else "chorus" for i in range(n - 2)]
    if mid and mid[-1] != "chorus":
        mid[-1] = "chorus"
    if len(mid) >= 5 and mid[-2] == "verse" and rng.random() < 0.8:
        mid[-2] = "bridge"
    labels = ["intro"] + mid + ["outro"]
    if "chorus" not in labels:
        labels[len(labels) // 2] = "chorus"
    return labels


def _plan_structure(col: int, ref_struct: list[dict], total_bars: int,
                    rng: random.Random) -> list[dict]:
    if col == 100 and ref_struct:
        return [dict(sec) for sec in ref_struct]  # same order + same bar lengths
    if col == 75 and ref_struct:
        bars = _scale_bars([sec["bars"] for sec in ref_struct], total_bars)
        return [{"label": sec["label"], "bars": b} for sec, b in zip(ref_struct, bars)]
    if col == 50 and ref_struct:
        labels = _labels_for_count(len(ref_struct), rng)
        weights = [_BAR_WEIGHT.get(lbl, 1.0) * 8 for lbl in labels]
        bars = _scale_bars([max(1, int(round(w))) for w in weights], total_bars)
        return [{"label": lbl, "bars": b} for lbl, b in zip(labels, bars)]
    # 0% free template / 25% loose-with-chorus (template always has a chorus)
    return _template_sections(total_bars, rng, jitter=0.25 if col == 25 else 0.0)


# ---- energy -------------------------------------------------------------------

def _normalized_curve(profile: dict) -> list[float]:
    curve = [
        _safe_float(v, 0.0) if v else 0.0
        for v in (profile.get("energy_curve") or {}).get("per_bar_rms") or []
    ]
    peak = max(curve) if curve else 0.0
    if peak <= 0:
        return []
    return [min(1.0, v / peak) for v in curve]


def _resample_list(vals: list[float], n: int) -> list[float]:
    """Area-averaging resample of ``vals`` to length ``n``."""
    if n <= 0:
        return []
    if not vals:
        return [0.5] * n
    m = len(vals)
    out: list[float] = []
    for i in range(n):
        lo, hi = i * m / n, (i + 1) * m / n
        acc = w = 0.0
        for j in range(int(lo), min(m, int(math.ceil(hi)))):
            ov = min(hi, j + 1) - max(lo, j)
            if ov > 0:
                acc += vals[j] * ov
                w += ov
        out.append(acc / w if w > 0 else vals[min(int(lo), m - 1)])
    return out


def _smooth(vals: list[float], window: int) -> list[float]:
    if window <= 1 or len(vals) < 3:
        return list(vals)
    half = window // 2
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _ref_section_energies(profile: dict, ref_struct: list[dict]) -> list[float]:
    """Mean normalized per-bar RMS for each reference section."""
    curve = _normalized_curve(profile)
    if not curve or not ref_struct:
        return [_LABEL_ENERGY.get(sec.get("label", "verse"), 0.55) for sec in ref_struct]
    total = sum(sec["bars"] for sec in ref_struct) or 1
    out, cum = [], 0
    for sec in ref_struct:
        i0 = int(round(cum / total * len(curve)))
        cum += sec["bars"]
        i1 = max(i0 + 1, int(round(cum / total * len(curve))))
        chunk = curve[min(i0, len(curve) - 1):min(i1, len(curve))] or [curve[-1]]
        out.append(sum(chunk) / len(chunk))
    return out


def _plan_energy(col: int, structure: list[dict], ref_struct: list[dict],
                 profile: dict, rng: random.Random) -> list[float]:
    n = len(structure)
    if col >= 50 and ref_struct and len(ref_struct) == n:
        # per-section match (and the per-bar tiers refine this further)
        targets = _ref_section_energies(profile, ref_struct)
    elif col == 25:
        # trend only: heavily smoothed reference curve resampled to our sections
        trend = _smooth(_resample_list(_normalized_curve(profile), n), 3)
        targets = [t + rng.uniform(-0.08, 0.08) for t in trend]
    else:
        # free: sensible defaults per label + jitter
        targets = [
            _LABEL_ENERGY.get(sec["label"], 0.55) + rng.uniform(-0.08, 0.08)
            for sec in structure
        ]
    return [round(min(1.0, max(0.0, t)), 3) for t in targets]


def _plan_energy_per_bar(s: int, structure: list[dict], profile: dict) -> list[float]:
    total_bars = sum(sec["bars"] for sec in structure) or 1
    vals = _resample_list(_normalized_curve(profile) or [0.5], total_bars)
    if s < 100:  # "per-bar loose" — follow the shape, not every wiggle
        vals = _smooth(vals, 4)
    return [round(min(1.0, max(0.0, v)), 3) for v in vals]


# ---- palette -------------------------------------------------------------------

def _reference_palette(profile: dict) -> list[str]:
    """Ranked instrument roles implied by the reference's stem activity."""
    inst = profile.get("instrumentation") or {}
    drums = _safe_float(inst.get("drums"), 0.7)
    bass = _safe_float(inst.get("bass"), 0.6)
    melodic = _safe_float(inst.get("melodic"), 0.5)
    vocals = _safe_float(inst.get("vocals"), 0.0)
    scored: list[tuple[str, float]] = [
        ("drums", drums), ("bass", bass), ("piano", melodic), ("pad", melodic * 0.85),
    ]
    if melodic >= 0.6:
        scored.append(("lead", melodic * 0.75))
    if vocals >= 0.5:
        scored.append(("lead", vocals * 0.8))  # vocal melody → lead instrument role
    best: dict[str, float] = {}
    for role, act in scored:
        best[role] = max(best.get(role, 0.0), act)
    ranked = [r for r, a in sorted(best.items(), key=lambda kv: -kv[1]) if a >= 0.2][:5]
    for fallback in ("drums", "bass", "piano", "pad"):
        if len(ranked) >= 3:
            break
        if fallback not in ranked:
            ranked.append(fallback)
    return ranked


def _plan_palette(s: int, col: int, profile: dict, rng: random.Random) -> list[str]:
    ref_palette = _reference_palette(profile)
    if col == 100:
        return list(ref_palette)  # full palette match
    n_shared = int(round(_interp(
        float(s), [(0, 0), (25, 1), (50, 2), (75, 3), (100, len(ref_palette))]
    )))
    n_shared = max(0, min(n_shared, len(ref_palette)))
    palette = ref_palette[:n_shared]
    pool = [r for r in _ROLES if r not in palette]
    rng.shuffle(pool)
    while len(palette) < max(4, n_shared) and pool:
        palette.append(pool.pop())
    return palette


# ---- groove --------------------------------------------------------------------

def _plan_groove(col: int, profile: dict, genre: str, rng: random.Random) -> dict:
    ref = profile.get("groove") or {}
    ref_swing = min(1.0, max(0.0, _safe_float(ref.get("swing"), 0.0)))
    ref_pattern = ref.get("pattern_class") if ref.get("pattern_class") in _PATTERNS else "backbeat"
    genre_pattern = _GENRE_PATTERN.get(genre, "backbeat")

    if col == 100:  # matched pattern class + swing
        swing, pattern = ref_swing, ref_pattern
    elif col == 75:  # similar pattern class
        swing = min(1.0, max(0.0, ref_swing + rng.uniform(-0.03, 0.03)))
        pattern = ref_pattern
    elif col == 50:  # similar swing
        swing = min(1.0, max(0.0, ref_swing + rng.uniform(-0.05, 0.05)))
        pattern = genre_pattern
    elif col == 25:  # genre-typical
        pattern = genre_pattern
        swing = 0.25 if pattern == "shuffle" else rng.uniform(0.0, 0.12)
    else:  # free
        pattern = rng.choice(_PATTERNS[:4])
        swing = rng.uniform(0.0, 0.3)
    return {"swing": round(swing, 3), "pattern_class": pattern}


# ---- prompt --------------------------------------------------------------------

def _build_prompt(profile: dict, bpm: float, tonic: str, mode: str,
                  palette: list[str], groove: dict, energy_targets: list[float]) -> str:
    """MusicGen text prompt from style descriptors only (never the source title)."""
    genres = [str(t) for t in (profile.get("genre_tags") or []) if t] or ["pop"]
    moods = [str(t) for t in (profile.get("mood_tags") or []) if t] or ["warm"]
    mean_energy = sum(energy_targets) / len(energy_targets) if energy_targets else 0.5
    if mean_energy >= 0.65:
        energy_word = "high-energy, driving"
    elif mean_energy <= 0.4:
        energy_word = "laid-back, mellow"
    else:
        energy_word = "mid-energy"
    swing = float(groove.get("swing", 0.0))
    groove_phrase = _PATTERN_PHRASE.get(groove.get("pattern_class", ""), "steady groove")
    if swing >= 0.4:
        groove_phrase += " with heavy swing"
    elif swing >= 0.15:
        groove_phrase += " with a light swing"
    parts = [
        f"{moods[0]} {genres[0]} instrumental",
        f"{int(round(bpm))} bpm",
        f"{tonic} {mode}",
        groove_phrase,
        "featuring " + ", ".join(palette) if palette else "minimal arrangement",
        energy_word,
        "no vocals",
        "polished modern production",
    ]
    return ", ".join(parts)
