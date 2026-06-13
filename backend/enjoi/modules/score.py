"""Impact scoring & chorus/bridge detection (Module: score, spec §4.6).

score_sections is a PURE, deterministic function over the vocal analysis dict:
it groups consecutive phrases into candidate sections, computes seven
sub-scores (each min-max normalized 0..1 across sections), combines them with
the (validated, renormalized) weights into an ImpactScore, and assigns roles:

  chorus = highest ImpactScore section
  bridge = nearest neighbor to the chorus in the 7-dim sub-score space that
           is not the chorus (only when >= 3 sections exist — with 2 sections
           the runner-up is needed as the verse; the -2 st pitch contrast is
           applied later in arrangement, not here)
  verse  = everything else, in original order

Only stdlib + numpy + enjoi.core at module level.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..core import config

_SECTION_MAX_SEC = 25.0     # break sections at 25 s (target 8–25 s)
_SECTION_GAP_SEC = 1.5      # break sections at silences longer than this
_CONTOUR_POINTS = 8         # melodic contours resampled to this length
_EPS = 1e-9

_SUBSCORE_KEYS = ("energy", "pitch_range", "pitch_height", "vibrato",
                  "repetition", "brightness", "hookiness")


def score_sections(analysis: dict, weights: dict | None = None) -> dict:
    phrases = sorted(analysis.get("phrases") or [], key=lambda p: p["start"])
    used_weights = _merge_weights(analysis.get("weights"), weights)
    analysis["weights"] = used_weights

    if not phrases:
        analysis["sections"] = []
        return analysis

    groups = _group_phrases(phrases)

    # raw sub-scores per section
    raw = {k: [] for k in _SUBSCORE_KEYS}
    contours = [_section_contour(g) for g in groups]
    token_lists = [_tokens(" ".join(p.get("text", "") for p in g)) for g in groups]
    take_counts = Counter(t for toks in token_lists for t in toks)
    any_lyrics = bool(take_counts)

    for i, g in enumerate(groups):
        feats = [p["features"] for p in g]
        raw["energy"].append(_mean(feats, "rms"))
        raw["pitch_range"].append(_mean(feats, "f0_range_semitones"))
        raw["pitch_height"].append(_mean(feats, "pitch_height"))
        raw["vibrato"].append(_mean(feats, "vibrato"))
        raw["brightness"].append(_mean(feats, "brightness"))

        contour_rep = _max_contour_similarity(i, contours)
        if any_lyrics:
            lyric_rep = _max_lyric_overlap(i, token_lists)
            raw["repetition"].append(0.6 * contour_rep + 0.4 * lyric_rep)
            raw["hookiness"].append(_hookiness(g, token_lists[i], take_counts))
        else:
            raw["repetition"].append(contour_rep)
            raw["hookiness"].append(contour_rep)  # melodic-repetition fallback

    # min-max normalize each sub-score across sections (epsilon-guarded)
    norm = {k: _minmax(np.asarray(v, dtype=np.float64)) for k, v in raw.items()}

    impact = np.zeros(len(groups), dtype=np.float64)
    for k in _SUBSCORE_KEYS:
        impact += used_weights[k] * norm[k]

    # ---- role assignment ----------------------------------------------------
    chorus_i = int(np.argmax(impact))  # ties → earliest section (argmax rule)
    bridge_i = None
    if len(groups) >= 3:
        vectors = np.stack([np.array([norm[k][i] for k in _SUBSCORE_KEYS])
                            for i in range(len(groups))])
        dists = np.linalg.norm(vectors - vectors[chorus_i], axis=1)
        dists[chorus_i] = np.inf
        bridge_i = int(np.argmin(dists))  # ties → earliest

    sections = []
    for i, g in enumerate(groups):
        if i == chorus_i:
            role = "chorus"
        elif bridge_i is not None and i == bridge_i:
            role = "bridge"
        else:
            role = "verse"
        text = " ".join(t for t in (p.get("text", "").strip() for p in g) if t)
        sections.append({
            "id": i,
            "phrase_ids": [p["id"] for p in g],
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": text,
            "impact_score": round(float(impact[i]), 3),
            "scores": {k: round(float(norm[k][i]), 3) for k in _SUBSCORE_KEYS},
            "role": role,
        })

    analysis["sections"] = sections  # groups built in start order already
    return analysis


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def _group_phrases(phrases: list[dict]) -> list[list[dict]]:
    """Consecutive phrases → candidate sections (target 8–25 s; break at
    gaps > 1.5 s or when adding a phrase would exceed 25 s; min 1 phrase)."""
    groups: list[list[dict]] = [[phrases[0]]]
    for p in phrases[1:]:
        cur = groups[-1]
        gap = p["start"] - cur[-1]["end"]
        dur_if_added = p["end"] - cur[0]["start"]
        if gap > _SECTION_GAP_SEC or dur_if_added > _SECTION_MAX_SEC:
            groups.append([p])
        else:
            cur.append(p)

    # 1-section take: split at the phrase boundary nearest the midpoint so we
    # always have at least chorus + verse material to work with.
    if len(groups) == 1 and len(groups[0]) >= 2:
        g = groups[0]
        mid = (g[0]["start"] + g[-1]["end"]) / 2.0
        best = min(range(1, len(g)), key=lambda j: abs(g[j]["start"] - mid))
        groups = [g[:best], g[best:]]
    return groups


# ---------------------------------------------------------------------------
# sub-score helpers
# ---------------------------------------------------------------------------

def _mean(feats: list[dict], key: str) -> float:
    vals = [float(f.get(key, 0.0)) for f in feats]
    return float(np.mean(vals)) if vals else 0.0


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = float(x.max() - x.min()) if x.size else 0.0
    if rng < _EPS:
        return np.full_like(x, 0.5)  # indistinguishable sections → neutral
    return (x - x.min()) / (rng + _EPS)


def _section_contour(group: list[dict]) -> np.ndarray:
    """Melodic contour proxy: phrase mean-f0 sequence in semitones, resampled
    to a fixed length and zero-meaned."""
    f0s = [float(p["features"].get("f0_mean_hz", 0.0)) for p in group]
    st = np.array([12.0 * math.log2(max(f, 1e-6) / 440.0) if f > 0 else np.nan
                   for f in f0s], dtype=np.float64)
    if np.all(np.isnan(st)):
        return np.zeros(_CONTOUR_POINTS)
    # fill unvoiced phrases with the section mean so interp stays defined
    fill = float(np.nanmean(st))
    st = np.where(np.isnan(st), fill, st)
    if len(st) == 1:
        resampled = np.full(_CONTOUR_POINTS, st[0])
    else:
        xs = np.linspace(0.0, 1.0, len(st))
        resampled = np.interp(np.linspace(0.0, 1.0, _CONTOUR_POINTS), xs, st)
    return resampled - resampled.mean()


def _contour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 and nb < 1e-6:
        return 1.0  # both flat (monotone) → identical shape
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return (cos + 1.0) / 2.0


def _max_contour_similarity(i: int, contours: list[np.ndarray]) -> float:
    sims = [_contour_similarity(contours[i], c)
            for j, c in enumerate(contours) if j != i]
    return max(sims) if sims else 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", (text or "").lower())


def _max_lyric_overlap(i: int, token_lists: list[list[str]]) -> float:
    a = set(token_lists[i])
    if not a:
        return 0.0
    best = 0.0
    for j, toks in enumerate(token_lists):
        if j == i:
            continue
        b = set(toks)
        if not b:
            continue
        jac = len(a & b) / max(len(a | b), 1)
        best = max(best, jac)
    return best


def _hookiness(group: list[dict], tokens: list[str], take_counts: Counter) -> float:
    """Lyric hookiness: short lines + repeated bigrams within the section +
    title-likeness (words that recur across the whole take)."""
    if not tokens:
        return 0.0
    # short lines: phrases averaging ~3 words score highest
    per_phrase = [len(_tokens(p.get("text", ""))) for p in group]
    sung = [n for n in per_phrase if n > 0]
    avg_words = float(np.mean(sung)) if sung else 0.0
    shortness = 1.0 / (1.0 + max(0.0, avg_words - 3.0) / 4.0)
    # repeated bigrams within the section
    bigrams = list(zip(tokens, tokens[1:]))
    rep_bigram = 1.0 - len(set(bigrams)) / len(bigrams) if len(bigrams) >= 2 else 0.0
    # title-likeness: fraction of tokens repeated across the take
    thr = 3 if sum(take_counts.values()) >= 30 else 2
    title = float(np.mean([1.0 if take_counts[t] >= thr else 0.0 for t in tokens]))
    return (shortness + rep_bigram + title) / 3.0


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def _merge_weights(stored: dict | None, override: dict | None) -> dict:
    """defaults ← analysis weights ← caller override; keep only the 7 known
    keys, drop invalid values, renormalize to sum 1."""
    w = dict(config.IMPACT_WEIGHTS_DEFAULT)
    for src in (stored, override):
        if not isinstance(src, dict):
            continue
        for k in w:
            if k in src:
                try:
                    v = float(src[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v >= 0.0:
                    w[k] = v
    total = sum(w.values())
    if total < _EPS:
        w = dict(config.IMPACT_WEIGHTS_DEFAULT)
        total = sum(w.values())
    return {k: round(v / total, 6) for k, v in w.items()}
