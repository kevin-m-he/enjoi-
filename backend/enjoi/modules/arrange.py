"""Auto-placement / arrangement (Module: arrange, spec §4.7).

build_arrangement maps the scored vocal sections onto the instrumental grid's
section slots:
  * chorus slots all receive the chorus section (repeated each chorus);
  * verse (and prechorus) slots consume verse sections in original order —
    extra verse slots stay instrumental, leftover verses go unused;
  * the bridge slot receives the bridge section, flagged "bridge_fx" so the
    later tune/mix stages apply the −2 st shift + thinning contrast;
  * intro/outro/inst slots stay vocal-free (gap policy).

Each placement is onset-aligned to the slot's first downbeat, time-stretch is
clamped to ±config.MAX_VOCAL_STRETCH (6%), and overflow is trimmed at the last
phrase (breath) boundary that fits. Chop WAVs are cut from vocal_raw with
10 ms edge fades into vocal_chops/p{id:03d}.wav.

apply_overrides applies user timeline edits (move / re-slot / gain / remove),
re-snaps to the nearest half-bar, validates bounds + overlaps, and persists.

Only stdlib + numpy + enjoi.core at module level.
"""
from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config
from ..core.errors import PipelineError
from ..core.storage import read_json, write_json

SR = config.SAMPLE_RATE
_MAX_STRETCH = config.MAX_VOCAL_STRETCH    # ±6%
_MIN_SLOT_ROOM_SEC = 0.25                  # slots smaller than this stay empty
_GAIN_DB_LIMIT = 24.0

_VERSE_SLOT_LABELS = {"verse", "prechorus"}

ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# build_arrangement
# ---------------------------------------------------------------------------

def build_arrangement(project, grid: dict, analysis: dict, progress: ProgressFn) -> dict:
    progress(0.05, "Mapping vocal sections to song slots…")

    slots = [
        {
            "label": str(s.get("label", "inst")),
            "index": i,
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "filled": False,
        }
        for i, s in enumerate(grid.get("sections") or [])
    ]
    downbeats = [float(d) for d in (grid.get("downbeats") or [])]

    sections = sorted(analysis.get("sections") or [], key=lambda s: s["start"])
    phrases_by_id = {p["id"]: p for p in analysis.get("phrases") or []}
    chorus = next((s for s in sections if s.get("role") == "chorus"), None)
    bridge = next((s for s in sections if s.get("role") == "bridge"), None)
    verses = [s for s in sections if s.get("role") == "verse"]

    # ---- choose a section for each slot ------------------------------------
    plan: list[tuple[dict, dict, str, bool]] = []  # (slot, section, role, bridge_fx)
    verse_iter = iter(verses)
    for slot in slots:
        label = slot["label"]
        if label == "chorus" and chorus is not None:
            plan.append((slot, chorus, "chorus", False))
        elif label in _VERSE_SLOT_LABELS:
            sec = next(verse_iter, None)
            if sec is not None:  # fewer verses than slots → leave instrumental
                plan.append((slot, sec, "verse", False))
        elif label == "bridge" and bridge is not None:
            plan.append((slot, bridge, "bridge", True))
        # intro / outro / inst (and unmatched labels) stay vocal-free

    # ---- timing: onset-align, stretch budget, breath-boundary trimming -----
    progress(0.25, "Aligning phrases to the beat grid…")
    placements: list[dict] = []
    for slot, sec, role, bridge_fx in plan:
        timing = _fit_section_to_slot(sec, slot, downbeats, phrases_by_id)
        if timing is None:
            continue
        source_start, source_end, target_start, stretch = timing
        pid = len(placements)
        placement = {
            "id": pid,
            "role": role,
            "slot_label": slot["label"],
            "slot_index": slot["index"],
            "section_id": sec["id"],
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "target_start": round(target_start, 3),
            "stretch": round(stretch, 4),
            "gain_db": 0.0,
            "chop_file": f"vocal_chops/p{pid:03d}.wav",
            "tuned_file": None,
        }
        if bridge_fx:
            placement["bridge_fx"] = True  # −2 st + thinned mix applied downstream
        placements.append(placement)
        slot["filled"] = True

    # ---- cut chop WAVs -------------------------------------------------------
    if placements:
        if not project.vocal_raw_path.exists():
            raise PipelineError("Processed vocal not found — upload the vocal take again.")
        audio, sr = core_audio.load_audio(project.vocal_raw_path, sr=SR, mono=True)
        n = len(placements)
        for k, p in enumerate(placements):
            progress(0.35 + 0.55 * (k + 1) / n, f"Cutting vocal chop {k + 1}/{n}…")
            i0 = max(0, int(p["source_start"] * sr))
            i1 = min(len(audio), int(p["source_end"] * sr))
            if i1 - i0 < int(0.05 * sr):
                continue
            seg = core_audio.fade_edges(audio[i0:i1], 0.01, 0.01, sr=sr)
            core_audio.save_wav(project.chops_dir / f"p{p['id']:03d}.wav", seg, sr=sr, subtype="PCM_16")

    arrangement = {
        "placements": placements,
        "slots": slots,
        "summary": {
            "chorus_section_id": chorus["id"] if chorus else None,
            "bridge_section_id": bridge["id"] if bridge else None,
        },
    }
    write_json(project.arrangement_path, arrangement)
    progress(1.0, f"Arrangement ready — {len(placements)} placements")
    return arrangement


def _fit_section_to_slot(
    sec: dict, slot: dict, downbeats: list[float], phrases_by_id: dict
) -> tuple[float, float, float, float] | None:
    """Returns (source_start, source_end, target_start, stretch) or None
    when the slot has no usable room."""
    slot_start, slot_end = slot["start"], slot["end"]
    target_start = _first_downbeat_in(downbeats, slot_start, slot_end)
    avail = slot_end - target_start
    if avail < _MIN_SLOT_ROOM_SEC:
        return None

    source_start = float(sec["start"])
    source_end = float(sec["end"])
    src_dur = source_end - source_start
    if src_dur <= 0:
        return None

    if src_dur <= avail:
        return source_start, source_end, target_start, 1.0

    rate = src_dur / avail
    if rate <= 1.0 + _MAX_STRETCH:
        return source_start, source_end, target_start, rate

    # Section too long even at +6% → trim at the last phrase (breath) boundary
    # that fits inside the stretch budget.
    limit = avail * (1.0 + _MAX_STRETCH)
    boundaries = []
    for ph_id in sec.get("phrase_ids", []):
        ph = phrases_by_id.get(ph_id)
        if ph is not None:
            b = float(ph["end"]) - source_start
            if 0.2 < b <= limit + 1e-6:
                boundaries.append(b)
    new_dur = max(boundaries) if boundaries else limit  # mid-phrase hard trim as last resort
    source_end = source_start + new_dur
    stretch = min(max(new_dur / avail, 1.0), 1.0 + _MAX_STRETCH)
    return source_start, source_end, target_start, stretch


def _first_downbeat_in(downbeats: list[float], start: float, end: float) -> float:
    for d in downbeats:
        if start - 0.02 <= d < end:
            return max(d, 0.0)
    return start


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

def apply_overrides(project, arrangement: dict, placements_patch: list[dict]) -> dict:
    """Apply user timeline edits, re-snap to half-bars, validate, persist."""
    if not project.grid_path.exists():
        raise PipelineError("No instrumental grid found — generate an instrumental first.")
    grid = read_json(project.grid_path)
    snap_points = _half_bar_points(grid)
    song_duration = float(grid.get("duration_sec") or 0.0)
    if song_duration <= 0:
        song_duration = max(
            (float(s.get("end", 0.0)) for s in grid.get("sections") or []), default=0.0
        )

    arr = copy.deepcopy(arrangement)
    placements: list[dict] = arr.get("placements") or []
    slots: list[dict] = arr.get("slots") or []
    by_id = {p["id"]: p for p in placements}

    for patch in placements_patch or []:
        if not isinstance(patch, dict) or "id" not in patch:
            raise PipelineError("Each placement override needs an 'id'.")
        pid = patch["id"]
        p = by_id.get(pid)
        if p is None:
            raise PipelineError(f"Unknown placement id: {pid}")

        if patch.get("enabled") is False:
            placements.remove(p)
            del by_id[pid]
            continue

        if "slot_index" in patch or "slot_label" in patch:
            slot = _resolve_slot(slots, patch, p)
            p["slot_index"] = slot["index"]
            p["slot_label"] = slot["label"]
            if "target_start" not in patch:
                p["target_start"] = float(slot["start"])

        if "target_start" in patch:
            try:
                t = float(patch["target_start"])
            except (TypeError, ValueError):
                raise PipelineError(f"Invalid target_start for placement {pid}.")
            if not math.isfinite(t) or t < 0.0 or (song_duration > 0 and t > song_duration):
                raise PipelineError(
                    f"Placement {pid} would start outside the song "
                    f"({t:.2f} s of {song_duration:.2f} s)."
                )
            p["target_start"] = t

        if "gain_db" in patch:
            try:
                g = float(patch["gain_db"])
            except (TypeError, ValueError):
                raise PipelineError(f"Invalid gain_db for placement {pid}.")
            if not math.isfinite(g):
                raise PipelineError(f"Invalid gain_db for placement {pid}.")
            p["gain_db"] = round(min(max(g, -_GAIN_DB_LIMIT), _GAIN_DB_LIMIT), 2)

        # re-snap to the nearest half-bar, then bounds-check
        p["target_start"] = round(_snap(p["target_start"], snap_points), 3)
        if p["target_start"] < 0.0 or (
            song_duration > 0 and p["target_start"] > song_duration - 0.05
        ):
            raise PipelineError(
                f"Placement {pid} would start outside the song "
                f"({p['target_start']:.2f} s of {song_duration:.2f} s)."
            )

    # no overlapping placements after the change
    ordered = sorted(placements, key=lambda p: (p["target_start"], p["id"]))
    for a, b in zip(ordered, ordered[1:]):
        a_end = a["target_start"] + _placed_duration(a)
        if a_end > b["target_start"] + 0.01:
            raise PipelineError(
                f"Placements {a['id']} ({a['slot_label']}) and {b['id']} ({b['slot_label']}) "
                f"would overlap — move one later or remove it."
            )

    filled = {p["slot_index"] for p in placements}
    for s in slots:
        s["filled"] = s["index"] in filled

    write_json(project.arrangement_path, arr)
    return arr


def _placed_duration(p: dict) -> float:
    src = float(p.get("source_end", 0.0)) - float(p.get("source_start", 0.0))
    stretch = float(p.get("stretch", 1.0)) or 1.0
    return max(src, 0.0) / max(stretch, 1e-6)


def _resolve_slot(slots: list[dict], patch: dict, placement: dict) -> dict:
    if "slot_index" in patch:
        for s in slots:
            if s["index"] == patch["slot_index"]:
                return s
        raise PipelineError(f"No slot with index {patch['slot_index']}.")
    label = patch["slot_label"]
    candidates = [s for s in slots if s["label"] == label]
    if not candidates:
        raise PipelineError(f"No slot labeled '{label}' in this song.")
    # several slots share the label (e.g. chorus) → pick the one closest to
    # where the placement currently sits (deterministic).
    cur = float(placement.get("target_start", 0.0))
    return min(candidates, key=lambda s: (abs(float(s["start"]) - cur), s["index"]))


def _half_bar_points(grid: dict) -> np.ndarray:
    downbeats = [float(d) for d in (grid.get("downbeats") or [])]
    points: list[float] = []
    if len(downbeats) >= 2:
        for i, d in enumerate(downbeats):
            points.append(d)
            bar = (downbeats[i + 1] - d) if i + 1 < len(downbeats) else (d - downbeats[i - 1])
            if bar > 0:
                points.append(d + bar / 2.0)
    elif downbeats:
        points = downbeats
    else:
        points = [float(b) for b in (grid.get("beat_times") or [])]
    return np.asarray(sorted(points), dtype=np.float64)


def _snap(t: float, points: np.ndarray) -> float:
    if points.size == 0:
        return float(t)
    return float(points[int(np.argmin(np.abs(points - t)))])
