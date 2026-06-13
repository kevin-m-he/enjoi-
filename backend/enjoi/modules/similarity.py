"""Similarity slider → generation plan (spec §4.3, table columns 0/25/50/75/100).

Pure functions over the reference profile — no audio I/O, stdlib only.

The slider maps *style descriptors* (key, structure, energy shape, palette,
groove) to a plan dict for the generator. Melody, chords and the reference
audio itself are never copied at any value ("style, never substance").

TEMPO IS NOT A STYLISTIC VARIABLE. At every similarity value the plan BPM
equals the reference BPM exactly (octave-sanity-clamped to 60–200). The slider
never nudges, randomizes, or scales tempo — see ``_plan_bpm``.

GENRE → INSTRUMENTATION. The reference's ``genre_tags`` select a genre profile
(``_GENRE_PROFILES``) that drives which synthesized instruments are used, the
default groove, swing feel, and arrangement density. The procedural engine
(synth.py) reads ``plan["genre"]`` and ``plan["instrument_palette"]`` to choose
genre-appropriate timbres (e.g. country → acoustic guitar + real-ish drums +
bass + piano, NOT a generic saw synth). The goal is to MIRROR the reference's
genre and emotional feel, not to impose a house style.
"""
from __future__ import annotations

import hashlib
import math
import random

_COLUMNS = (0, 25, 50, 75, 100)
_TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
_PATTERNS = ("four_on_floor", "backbeat", "halftime", "shuffle", "sparse")

# ---------------------------------------------------------------------------
# Genre profiles — the heart of genre-appropriate instrumentation.
#
# Each canonical genre maps to:
#   instruments  : ordered instrument-role palette the engine should synthesize.
#                  These are the genre's "default" full-band arrangement; the
#                  similarity slider trims/blends it with the reference's own
#                  stem activity. Roles are interpreted by synth.py.
#   pattern      : the genre-typical drum pattern_class.
#   swing        : a sensible swing default (overridden by the reference at high
#                  similarity).
#   density      : 0..1 arrangement density hint (how busy / how many layers).
#   aliases      : substrings matched against the reference's genre_tags.
#
# Instrument-role vocabulary understood by synth.py:
#   drums, bass, sub_bass, acoustic_guitar, electric_guitar, dist_guitar,
#   piano, epiano, organ, synth_keys, pad, strings, pluck, lead, synth_lead,
#   brass, bell, arp, 808.
# ---------------------------------------------------------------------------

_GENRE_PROFILES: dict[str, dict] = {
    "pop": {
        "instruments": ["drums", "bass", "piano", "synth_keys", "pad", "lead"],
        "pattern": "backbeat", "swing": 0.04, "density": 0.6,
        "aliases": ["pop", "synthpop", "electropop", "dance-pop", "indie pop"],
    },
    "rock": {
        "instruments": ["drums", "bass", "electric_guitar", "piano", "lead"],
        "pattern": "backbeat", "swing": 0.0, "density": 0.65,
        "aliases": ["rock", "indie rock", "alt rock", "punk", "garage", "grunge"],
    },
    "metal": {
        "instruments": ["drums", "bass", "dist_guitar", "dist_guitar", "lead"],
        "pattern": "backbeat", "swing": 0.0, "density": 0.8,
        "aliases": ["metal", "heavy metal", "metalcore", "djent", "hardcore"],
    },
    "country": {
        "instruments": ["drums", "bass", "acoustic_guitar", "piano", "pedal_steel"],
        "pattern": "backbeat", "swing": 0.08, "density": 0.5,
        "aliases": ["country", "americana", "bluegrass", "nashville"],
    },
    "folk": {
        "instruments": ["acoustic_guitar", "bass", "strings", "piano", "drums"],
        "pattern": "sparse", "swing": 0.06, "density": 0.4,
        "aliases": ["folk", "singer-songwriter", "indie folk"],
    },
    "acoustic": {
        "instruments": ["acoustic_guitar", "bass", "piano", "strings", "drums"],
        "pattern": "sparse", "swing": 0.05, "density": 0.38,
        "aliases": ["acoustic", "unplugged"],
    },
    "hip hop": {
        "instruments": ["drums", "808", "epiano", "pad", "bell"],
        "pattern": "halftime", "swing": 0.12, "density": 0.45,
        "aliases": ["hip hop", "hip-hop", "hiphop", "rap", "boom bap", "conscious"],
    },
    "trap": {
        "instruments": ["drums", "808", "synth_keys", "bell", "pad"],
        "pattern": "halftime", "swing": 0.06, "density": 0.5,
        "aliases": ["trap", "drill"],
    },
    "r&b": {
        "instruments": ["drums", "bass", "epiano", "pad", "synth_keys", "lead"],
        "pattern": "halftime", "swing": 0.14, "density": 0.5,
        "aliases": ["r&b", "rnb", "soul", "neo soul", "neo-soul", "contemporary r&b"],
    },
    "edm": {
        "instruments": ["drums", "sub_bass", "synth_keys", "pluck", "pad", "synth_lead"],
        "pattern": "four_on_floor", "swing": 0.0, "density": 0.7,
        "aliases": ["edm", "electronic", "future bass", "dubstep", "electro"],
    },
    "dance": {
        "instruments": ["drums", "sub_bass", "synth_keys", "pluck", "pad", "synth_lead"],
        "pattern": "four_on_floor", "swing": 0.0, "density": 0.7,
        "aliases": ["dance", "dance pop"],
    },
    "house": {
        "instruments": ["drums", "sub_bass", "synth_keys", "pluck", "pad", "organ"],
        "pattern": "four_on_floor", "swing": 0.04, "density": 0.65,
        "aliases": ["house", "deep house", "tech house", "progressive house"],
    },
    "lofi": {
        "instruments": ["drums", "bass", "epiano", "pad", "pluck"],
        "pattern": "halftime", "swing": 0.18, "density": 0.35,
        "aliases": ["lofi", "lo-fi", "chillhop", "chill"],
    },
    "jazz": {
        "instruments": ["drums", "bass", "piano", "brass", "lead"],
        "pattern": "shuffle", "swing": 0.5, "density": 0.55,
        "aliases": ["jazz", "swing", "bebop", "fusion"],
    },
    "blues": {
        "instruments": ["drums", "bass", "electric_guitar", "organ", "lead"],
        "pattern": "shuffle", "swing": 0.4, "density": 0.5,
        "aliases": ["blues", "rhythm and blues"],
    },
    "latin": {
        "instruments": ["drums", "bass", "piano", "brass", "acoustic_guitar"],
        "pattern": "four_on_floor", "swing": 0.08, "density": 0.65,
        "aliases": ["latin", "reggaeton", "salsa", "bossa", "samba", "afrobeat"],
    },
    "k-pop": {
        "instruments": ["drums", "bass", "synth_keys", "pluck", "pad", "synth_lead"],
        "pattern": "backbeat", "swing": 0.03, "density": 0.7,
        "aliases": ["k-pop", "kpop", "k pop"],
    },
    "classical": {
        "instruments": ["strings", "piano", "brass", "pad"],
        "pattern": "sparse", "swing": 0.0, "density": 0.45,
        "aliases": ["classical", "orchestral", "cinematic", "soundtrack", "score"],
    },
}

# Canonical fallback when no genre tag matches.
_DEFAULT_GENRE = "pop"

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

# Human-readable instrument names for the MusicGen text prompt (generic default).
_PROMPT_INSTRUMENT = {
    "drums": "drums", "bass": "bass", "sub_bass": "deep sub bass", "808": "808 bass",
    "acoustic_guitar": "acoustic guitar", "electric_guitar": "electric guitar",
    "dist_guitar": "distorted guitar", "pedal_steel": "pedal steel",
    "piano": "piano", "epiano": "electric piano", "organ": "organ",
    "synth_keys": "synth keys", "pad": "warm pad", "strings": "strings",
    "pluck": "synth pluck", "lead": "lead melody", "synth_lead": "synth lead",
    "brass": "brass", "bell": "bell", "arp": "arpeggio",
}

# Producer-grade, genre-specific instrument phrasing. For a given genre we pick
# the most evocative real-instrument description so MusicGen renders a believable
# band in that style instead of generic timbres. Falls back to _PROMPT_INSTRUMENT.
_GENRE_INSTRUMENT_WORDS: dict[str, dict[str, str]] = {
    "folk": {
        "drums": "soft brushed drums", "bass": "warm upright bass",
        "acoustic_guitar": "fingerpicked steel-string acoustic guitar",
        "electric_guitar": "warm clean electric guitar", "piano": "soft felt piano",
        "strings": "gentle string section", "pedal_steel": "weeping pedal steel",
        "organ": "warm hammond organ",
    },
    "country": {
        "drums": "gentle brushed drum kit", "bass": "warm electric bass",
        "acoustic_guitar": "bright strummed acoustic guitar",
        "electric_guitar": "twangy telecaster electric guitar",
        "piano": "honky-tonk piano", "pedal_steel": "crying pedal steel guitar",
        "strings": "fiddle", "organ": "hammond organ",
    },
    "acoustic": {
        "drums": "soft cajon and brushed drums", "bass": "warm upright bass",
        "acoustic_guitar": "fingerpicked acoustic guitar",
        "piano": "intimate acoustic piano", "strings": "soft strings",
    },
    "rock": {
        "drums": "punchy live drum kit", "bass": "driving electric bass",
        "electric_guitar": "crunchy overdriven electric guitars",
        "dist_guitar": "distorted electric guitars", "piano": "rock piano",
        "organ": "hammond organ",
    },
    "metal": {
        "drums": "aggressive double-kick drums", "bass": "heavy distorted bass",
        "dist_guitar": "heavy palm-muted distorted guitars",
        "electric_guitar": "high-gain electric guitars",
    },
    "blues": {
        "drums": "shuffling drum kit", "bass": "walking electric bass",
        "electric_guitar": "soulful overdriven blues guitar",
        "organ": "hammond organ", "piano": "bluesy piano",
    },
    "jazz": {
        "drums": "brushed jazz drums", "bass": "walking upright bass",
        "piano": "jazz piano comping", "brass": "muted trumpet and saxophone",
        "electric_guitar": "warm hollow-body jazz guitar",
    },
    "r&b": {
        "drums": "tight laid-back drums", "bass": "round electric bass",
        "epiano": "lush rhodes electric piano", "piano": "soulful piano",
        "pad": "warm analog pad", "synth_keys": "smooth synth keys",
    },
    "gospel": {
        "drums": "energetic live drums", "bass": "round electric bass",
        "organ": "soaring hammond organ", "piano": "gospel piano",
        "strings": "full choir-backed strings",
    },
    "latin": {
        "drums": "live latin percussion and drums", "bass": "syncopated electric bass",
        "piano": "montuno piano", "brass": "bright brass section",
        "acoustic_guitar": "nylon-string guitar",
    },
    "lofi": {
        "drums": "dusty lo-fi drums", "bass": "mellow electric bass",
        "epiano": "warm vintage rhodes", "pad": "hazy tape pad",
        "piano": "soft jazzy piano",
    },
    "hip hop": {
        "drums": "boom-bap drums", "bass": "deep electric bass",
        "epiano": "soulful rhodes", "808": "deep 808 bass", "bell": "vibraphone bells",
    },
    "trap": {
        "drums": "crisp trap drums with rolling hi-hats", "808": "booming 808 bass",
        "synth_keys": "dark synth keys", "bell": "glassy bells", "pad": "atmospheric pad",
    },
    "edm": {
        "drums": "punchy four-on-the-floor drums", "sub_bass": "deep sub bass",
        "synth_keys": "bright supersaw synths", "pluck": "rhythmic synth plucks",
        "pad": "wide synth pad", "synth_lead": "soaring synth lead",
    },
    "pop": {
        "drums": "polished pop drums", "bass": "tight electric bass",
        "piano": "bright pop piano", "synth_keys": "modern synth keys",
        "pad": "lush pad", "acoustic_guitar": "bright acoustic guitar",
    },
}

# Genre-level production/character descriptors (the "studio vibe").
_GENRE_PRODUCTION = {
    "folk": "organic live-band studio recording, warm analog, intimate, high fidelity",
    "country": "warm Nashville studio recording, live band, analog warmth, high fidelity",
    "acoustic": "intimate unplugged studio recording, natural room sound, high fidelity",
    "singer-songwriter": "intimate organic studio recording, warm analog, high fidelity",
    "rock": "punchy live rock band, big drum room, analog warmth, high fidelity",
    "metal": "tight modern metal production, powerful and aggressive, high fidelity",
    "blues": "warm live blues club recording, analog, high fidelity",
    "jazz": "warm acoustic jazz club recording, natural dynamics, high fidelity",
    "r&b": "smooth modern R&B production, warm and lush, high fidelity",
    "soul": "warm vintage soul recording, live band, analog, high fidelity",
    "gospel": "uplifting live gospel recording, full and warm, high fidelity",
    "latin": "lively latin production, crisp percussion, high fidelity",
    "lofi": "warm lo-fi production, vinyl crackle, mellow, cozy",
    "hip hop": "warm boom-bap production, punchy and dusty, high fidelity",
    "trap": "modern trap production, clean and hard-hitting, high fidelity",
    "edm": "polished electronic production, wide and energetic, club-ready, high fidelity",
    "dance": "polished dance production, energetic, club-ready, high fidelity",
    "house": "deep house production, groovy and warm, club-ready, high fidelity",
    "pop": "polished modern pop production, radio-ready, clean and bright, high fidelity",
    "k-pop": "glossy K-pop production, energetic and modern, high fidelity",
    "classical": "lush orchestral recording, concert-hall ambience, high fidelity",
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_generation_plan(profile: dict, similarity: int, rng_seed: int | None = None) -> dict:
    """Map slider value 0..100 + reference profile → generation plan dict.

    Implements spec table 4.3 at the 0/25/50/75/100 columns and interpolates
    behaviour between them — EXCEPT tempo, which is always the reference tempo
    (BPM is fixed to the reference at every similarity value). Reproducible via
    ``rng_seed`` (a deterministic seed is derived from the profile + similarity
    otherwise).
    """
    s = _clamp_similarity(similarity)
    col = _column(s)
    duration = float(profile.get("duration_sec") or 180.0)
    ref_tonic, ref_mode = _ref_key(profile)
    genre, gp = _primary_genre(profile)

    seed = _derive_seed(profile, s, rng_seed)
    rng = random.Random(seed)

    bpm = _plan_bpm(profile)  # FIXED to reference at every similarity value.
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
    palette = _plan_palette(s, col, profile, genre, gp, rng)
    groove = _plan_groove(col, profile, genre, gp, rng)

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
        # Richer descriptors consumed by synth.py / generate.py (additive keys).
        "genre": genre,
        "genre_density": round(float(gp.get("density", 0.55)), 3),
        "mood_tags": [str(t) for t in (profile.get("mood_tags") or []) if t],
        "prompt": _build_prompt(profile, genre, bpm, tonic, mode, palette,
                                groove, energy_targets),
        "summary": similarity_summary(profile, s),
    }
    if s >= 75:
        plan["energy_per_bar"] = _plan_energy_per_bar(s, structure, profile)
    return plan


def similarity_summary(profile: dict, similarity: int) -> str:
    """Deterministic human-readable label for the slider (no RNG needed).

    Tempo is fixed to the reference, so the label always reads "same tempo". At
    100% the label reflects that the track is as close to the reference as the
    originality audit allows — same key, tempo, structure and the reference's
    full natural instrument palette — with melody & chords still 100% original
    (the audit only gates melody/chords).
    """
    s = _clamp_similarity(similarity)
    col = _column(s)
    if col == 100:
        return (
            "100% — as close to the reference as possible: same key, same tempo, "
            "same structure & bar lengths, matched groove, and the reference's "
            "full natural instrument palette — melody & chords still 100% original."
        )
    key_phrase = {
        0: "any key", 25: "related key", 50: "same mode",
        75: "same key family",
    }[col]
    structure_phrase = {
        0: "free structure", 25: "loose structure with a chorus",
        50: "same section count", 75: "same section order",
    }[col]
    groove_phrase = {
        0: "free groove", 25: "genre-typical groove", 50: "similar swing",
        75: "similar groove",
    }[col]
    return (
        f"{s}% — {key_phrase}, {structure_phrase}, same tempo, {groove_phrase}"
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
# tempo — FIXED to the reference (not a stylistic variable)
# ---------------------------------------------------------------------------

def _octave_clamp_bpm(bpm: float) -> float:
    """Sanity-clamp a tempo into 60–200 by octave halving/doubling.

    Beat trackers commonly latch onto a double- or half-tempo; we keep the
    musical tempo but bring it into a sensible range without changing its feel.
    """
    if not math.isfinite(bpm) or bpm <= 0:
        return 120.0
    while bpm > 200.0:
        bpm /= 2.0
    while bpm < 60.0:
        bpm *= 2.0
    return min(200.0, max(60.0, bpm))


def _plan_bpm(profile: dict) -> float:
    """The reference BPM, octave-clamped to 60–200. Identical at every
    similarity value — the slider never varies tempo."""
    return _octave_clamp_bpm(_safe_float(profile.get("bpm"), 120.0))


# ---------------------------------------------------------------------------
# genre selection
# ---------------------------------------------------------------------------

def _primary_genre(profile: dict) -> tuple[str, dict]:
    """Resolve the reference's primary genre → (canonical name, genre profile)."""
    tags = [str(t).lower().strip() for t in (profile.get("genre_tags") or []) if t]
    for tag in tags:
        for name, gp in _GENRE_PROFILES.items():
            for alias in gp["aliases"]:
                if alias in tag or tag in alias:
                    return name, gp
    return _DEFAULT_GENRE, _GENRE_PROFILES[_DEFAULT_GENRE]


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------

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
    """A sensible free song template scaled to ``total_bars``.

    "Less is more": keep section counts modest so the engine can repeat a clean
    16–24 bar idea rather than cramming. Vocals are added later — leave space.
    """
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

# Electronic genres are the only ones that should get synth roles by default;
# everywhere else the palette stays natural (real instruments first).
_ELECTRONIC_GENRES = {"edm", "dance", "house", "trap", "k-pop"}
# Synth/electronic engine roles we strip from natural-genre palettes.
_SYNTH_ROLES = {"synth_keys", "synth_lead", "pluck", "arp", "pad", "sub_bass",
                "808", "lead"}


def _reference_activity(profile: dict) -> dict:
    """Per-instrument activity 0..1 from the reference profile.

    Reads the natural-first instrumentation keys produced by reference.py
    (drums/bass/guitar/piano/other/vocals) and keeps ``melodic`` for backward
    compatibility (falling back to it when a specific key is missing)."""
    inst = profile.get("instrumentation") or {}
    melodic = _safe_float(inst.get("melodic"), 0.5)
    guitar = _safe_float(inst.get("guitar"), 0.0)
    piano = _safe_float(inst.get("piano"), 0.0)
    other = _safe_float(inst.get("other"), 0.0)
    # If only the legacy "melodic" key exists, treat it as the melodic content.
    if guitar <= 0.0 and piano <= 0.0 and other <= 0.0:
        other = melodic
    return {
        "drums": _safe_float(inst.get("drums"), 0.7),
        "bass": _safe_float(inst.get("bass"), 0.6),
        "guitar": guitar,
        "piano": piano,
        "other": other,
        "melodic": max(melodic, guitar, piano, other),
        "vocals": _safe_float(inst.get("vocals"), 0.0),
    }


def _natural_palette(act: dict, gp: dict, electronic: bool) -> list[str]:
    """Ordered NATURAL-instrument-first palette from the reference's detected
    instruments.

    Real instruments that are actually present (drums, bass, guitar, piano) lead
    the palette; the genre profile's own roles supply timbre detail and (for
    electronic genres only) synth roles. The reference's vocal melodic line is
    reserved for the user's vocal, so no synth lead is forced into natural genres.
    """
    # Pick the guitar timbre the genre prefers (acoustic by default for natural
    # genres; the genre band may specify electric/distorted).
    gp_set = set(gp.get("instruments") or [])
    if "dist_guitar" in gp_set:
        guitar_role = "dist_guitar"
    elif "electric_guitar" in gp_set:
        guitar_role = "electric_guitar"
    else:
        guitar_role = "acoustic_guitar"

    # The genre band tells us whether a rhythm section is idiomatic; most
    # full-band genres (pop/rock/country/folk/etc.) imply drums + bass even when
    # one stem reads quietly in the mix (e.g. a soft folk bassline). We keep the
    # foundation unless the reference shows essentially NO trace of it.
    genre_has_drums = "drums" in gp_set
    genre_has_low = bool(gp_set & {"bass", "sub_bass", "808"})

    palette: list[str] = []
    # 1) Rhythm foundation, natural first.
    if act["drums"] >= 0.05 or (genre_has_drums and act["drums"] >= 0.02):
        palette.append("drums")
    if act["bass"] >= 0.05 or (genre_has_low and act["bass"] >= 0.02):
        # Electronic genres prefer sub/808; natural genres get real bass.
        if electronic and "sub_bass" in gp_set:
            palette.append("sub_bass")
        elif electronic and "808" in gp_set:
            palette.append("808")
        else:
            palette.append("bass")
    # 2) Natural melodic/harmonic instruments that are present, prominent first.
    melodic_present = []
    if act["guitar"] >= 0.2:
        melodic_present.append((act["guitar"], guitar_role))
    if act["piano"] >= 0.2:
        melodic_present.append((act["piano"], "piano"))
    melodic_present.sort(key=lambda t: -t[0])
    for _, role in melodic_present:
        if role not in palette:
            palette.append(role)
    # Ensure at least one chordal instrument even if activity was modest. For
    # electronic genres the synth keys (added later from the genre band) cover
    # this, so only force a natural chordal instrument for natural genres.
    has_chordal = set(palette) & {"acoustic_guitar", "electric_guitar",
                                  "dist_guitar", "piano", "epiano", "organ"}
    if not has_chordal and not electronic:
        fallback = guitar_role if act["guitar"] >= act["piano"] else "piano"
        palette.append(fallback)
    return palette


def _plan_palette(s: int, col: int, profile: dict, genre: str, gp: dict,
                  rng: random.Random) -> list[str]:
    """Natural-instrument-first instrument palette built from the reference's
    detected instruments.

    Real instruments (drums, bass, guitar, piano) that are actually present lead
    the palette. Synth/electronic roles (synth_keys, plucks, pads, sub/808,
    synth_lead) are only added for electronic genres. For a country/folk
    reference the palette is e.g. ["drums","bass","acoustic_guitar","piano"] —
    never generic saw synths. The density/similarity hints decide how many extra
    genre-detail layers to keep so the result stays clean ("less is more").
    """
    act = _reference_activity(profile)
    electronic = genre in _ELECTRONIC_GENRES

    palette = _natural_palette(act, gp, electronic)

    # Genre-detail layers from the genre band, filtered by genre type. For
    # natural genres we only allow natural roles (extra real instruments such as
    # strings/organ/epiano/pedal_steel); synth roles are reserved for electronic.
    extras: list[str] = []
    for role in dict.fromkeys(gp.get("instruments") or []):
        if role in palette:
            continue
        if role in _LOW_ROLE_SET or role == "drums":
            continue
        if not electronic and role in _SYNTH_ROLES:
            continue  # natural genres never get synth/pad/lead by default
        extras.append(role)

    # Density → how many extra detail layers to keep on top of the natural core.
    density = float(gp.get("density", 0.55))
    melodic = act["melodic"]
    sim_factor = _interp(float(s), [(0, 0.7), (50, 0.85), (100, 1.0)])
    target = density * (0.6 + 0.5 * melodic) * sim_factor
    total_layers = int(round(_interp(target, [(0.0, 3), (0.5, 4), (0.75, 5), (1.0, 6)])))
    total_layers = max(3, total_layers)
    n_extra = max(0, total_layers - len(palette))
    palette += extras[:n_extra]

    # Electronic genres may take a synth lead/topline at higher similarity; the
    # natural-genre vocal slot is left for the user's vocal (no forced lead).
    if electronic and s >= 50 and melodic >= 0.5:
        lead = next((c for c in (gp.get("instruments") or []) if c in
                     ("synth_lead", "lead")), None)
        if lead and lead not in palette and len(palette) < 6:
            palette.append(lead)

    # De-dupe while preserving order.
    seen: set[str] = set()
    out = [p for p in palette if not (p in seen or seen.add(p))]
    return out or ["drums", "bass", "acoustic_guitar", "piano"]


# Low-end engine roles (only one is ever used; chosen in _natural_palette).
_LOW_ROLE_SET = {"bass", "sub_bass", "808"}


# ---- groove --------------------------------------------------------------------

def _plan_groove(col: int, profile: dict, genre: str, gp: dict,
                 rng: random.Random) -> dict:
    ref = profile.get("groove") or {}
    ref_swing = min(1.0, max(0.0, _safe_float(ref.get("swing"), 0.0)))
    ref_pattern = ref.get("pattern_class") if ref.get("pattern_class") in _PATTERNS else None
    genre_pattern = gp.get("pattern", "backbeat")
    genre_swing = float(gp.get("swing", 0.0))

    if col == 100:  # matched pattern class + swing (genre pattern if ref absent)
        swing = ref_swing if ref_swing > 0 else genre_swing
        pattern = ref_pattern or genre_pattern
    elif col == 75:  # similar pattern class
        swing = min(1.0, max(0.0, (ref_swing or genre_swing) + rng.uniform(-0.03, 0.03)))
        pattern = ref_pattern or genre_pattern
    elif col == 50:  # similar swing, genre pattern
        swing = min(1.0, max(0.0, (ref_swing or genre_swing) + rng.uniform(-0.05, 0.05)))
        pattern = genre_pattern
    elif col == 25:  # genre-typical
        pattern = genre_pattern
        swing = genre_swing if genre_swing > 0 else rng.uniform(0.0, 0.1)
    else:  # free, but still anchored to the genre so it reads as the right style
        pattern = genre_pattern
        swing = min(1.0, max(0.0, genre_swing + rng.uniform(-0.05, 0.08)))
    return {"swing": round(swing, 3), "pattern_class": pattern}


# ---- prompt --------------------------------------------------------------------

def _genre_name(genre: str) -> str:
    """How the genre should read in the prompt (slashes help MusicGen)."""
    return {
        "country": "folk/country", "folk": "folk", "acoustic": "acoustic",
        "r&b": "R&B/soul", "hip hop": "hip hop", "edm": "electronic/EDM",
        "lofi": "lo-fi hip hop", "k-pop": "K-pop",
    }.get(genre, genre)


def _instrument_words(genre: str, palette: list[str]) -> list[str]:
    """Producer-grade, genre-specific instrument descriptions for the palette."""
    genre_words = _GENRE_INSTRUMENT_WORDS.get(genre, {})
    out: list[str] = []
    seen: set[str] = set()
    for role in palette:
        word = genre_words.get(role) or _PROMPT_INSTRUMENT.get(role, role.replace("_", " "))
        if word not in seen:
            out.append(word)
            seen.add(word)
    return out


def _build_prompt(profile: dict, genre: str, bpm: float, tonic: str, mode: str,
                  palette: list[str], groove: dict, energy_targets: list[float]) -> str:
    """Rich, producer-grade MusicGen text prompt from style descriptors only
    (never the source title or melody).

    Names REAL, genre-appropriate instruments (e.g. for folk/country:
    fingerpicked steel-string acoustic guitar, warm upright bass, gentle brushed
    drums, soft piano, subtle pedal steel/fiddle) plus the mood, tempo, key,
    groove and a genre-specific production character. Acoustic/organic genres
    never receive synth language. This steers MusicGen toward a believable live
    band in the reference's genre rather than synthetic bleeps.
    """
    moods = [str(t) for t in (profile.get("mood_tags") or []) if t] or ["warm"]
    mood_word = moods[0]
    mean_energy = sum(energy_targets) / len(energy_targets) if energy_targets else 0.5
    if mean_energy >= 0.65:
        energy_word = "energetic and driving"
    elif mean_energy <= 0.4:
        energy_word = "laid-back and mellow"
    else:
        energy_word = "mid-energy"

    swing = float(groove.get("swing", 0.0))
    groove_phrase = _PATTERN_PHRASE.get(groove.get("pattern_class", ""), "steady groove")
    if swing >= 0.4:
        groove_phrase += " with heavy swing"
    elif swing >= 0.15:
        groove_phrase += " with a light swing"

    instr_words = _instrument_words(genre, palette)
    production = _GENRE_PRODUCTION.get(genre, "polished modern production, high fidelity")

    parts = [
        f"{mood_word} {_genre_name(genre)} instrumental",
        ("featuring " + ", ".join(instr_words)) if instr_words else "minimal arrangement",
        f"{int(round(bpm))} BPM",
        f"{tonic} {mode}",
        groove_phrase,
        energy_word,
        production,
        "instrumental only, no vocals, no singing",
    ]
    return ", ".join(parts)
