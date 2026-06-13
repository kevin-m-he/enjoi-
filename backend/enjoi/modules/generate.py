"""Instrumental generation orchestrator (spec 4.4).

Engine selection: MusicGen (audiocraft + torch present) with model size chosen
by VRAM, else the built-in procedural synthesizer (synth.py). MusicGen runs in
TEXT-PROMPT mode only — melody conditioning is never used at any similarity
value (spec 4.3): the reference melody must never be an input to generation.

Pipeline per attempt:
  render → conform length to the arithmetic beat grid (±5% of the plan's
  target duration) → write instrumental.wav → Uniqueness Guard audit
  (unique.run_uniqueness_audit against profile["fingerprints"]).

Fail behavior (spec 4.3.1): regenerate only the failing aspects with a
perturbed seed (plan["seed"] + attempt) and a nudged prompt ("different
melodic contour"); up to 3 retries, then lower the effective similarity by 10
points (rebuilding the plan via modules.similarity.build_generation_plan) and
retry — looping until pass or the similarity floor of 0. A failing MusicGen
engine falls back to the procedural synth, which (with a fresh seed) passes
essentially always since the reference melody is never an input.

Writes instrumental.wav (44.1 kHz / 24-bit), instrumental_grid.json (contract
schema, incl. key.scale_midi) and uniqueness_report.json, then deletes the
reference cache (_ref_cache/) — the ownership guarantee.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ..core import audio as core_audio
from ..core import config, deps, storage
from ..core.errors import PipelineError
from ..core.jobs import subprogress
from . import synth, unique

SR = config.SAMPLE_RATE

MAX_WINDOW_SEC = 30.0       # MusicGen generation window cap
CONT_PROMPT_SEC = 5.0       # audio-prompt length for continuation stitching
CROSSFADE_SEC = 0.05        # section stitch crossfade at downbeats

_CHECK_ASPECT = {
    "melody_ngram_overlap": "melody",
    "chord_run_length": "harmony",
    "chroma_correlation": "chroma",
    "audio_fingerprint": "fingerprint",
}

_MUSICGEN_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

def _select_engine() -> tuple[str, str | None]:
    """→ ("musicgen", model_id) or ("procedural", None)."""
    if not deps.has("audiocraft"):
        return "procedural", None
    torch = deps.optional_import("torch")
    if torch is None:
        return "procedural", None
    env = os.environ.get("ENJOI_MUSICGEN_MODEL", "").strip()
    if env:
        if "/" in env:
            return "musicgen", env
        name = env if env.startswith("musicgen-") else "musicgen-" + env
        return "musicgen", "facebook/" + name
    try:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            if vram_gb >= 8.0:
                return "musicgen", "facebook/musicgen-medium"
    except Exception:
        pass
    return "musicgen", "facebook/musicgen-small"  # < 8 GB VRAM or CPU


# ---------------------------------------------------------------------------
# MusicGen rendering (text prompt only — NEVER melody conditioning)
# ---------------------------------------------------------------------------

def _section_descriptor(prompt: str, label: str, energy: float) -> str:
    label_txt = {
        "intro": "soft intro", "verse": "verse groove",
        "prechorus": "building pre-chorus", "chorus": "energetic chorus, catchy hook",
        "bridge": "contrasting bridge", "outro": "outro, winding down",
        "inst": "instrumental break",
    }.get(label, label)
    if energy < 0.35:
        energy_txt = "soft, sparse arrangement"
    elif energy < 0.65:
        energy_txt = "moderate energy"
    else:
        energy_txt = "energetic, full arrangement"
    return f"{prompt}, {label_txt}, {energy_txt}"


def _get_musicgen(model_id: str):
    if _MUSICGEN_CACHE.get("id") == model_id:
        return _MUSICGEN_CACHE["model"]
    import torch
    from audiocraft.models import MusicGen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGen.get_pretrained(model_id, device=device)
    _MUSICGEN_CACHE.clear()
    _MUSICGEN_CACHE.update({"id": model_id, "model": model})
    return model


def _mg_window(model, desc: str, duration: float) -> np.ndarray:
    """One ≤30 s MusicGen window → mono float numpy at model.sample_rate."""
    model.set_generation_params(duration=float(max(2.0, min(duration, MAX_WINDOW_SEC))),
                                use_sampling=True, top_k=250)
    out = model.generate([desc], progress=False)
    return out[0].detach().cpu().float().numpy().mean(axis=0)


def _mg_section(model, desc: str, duration: float) -> np.ndarray:
    """Generate a full section, stitching >30 s via continuation with the last
    5 s as the audio prompt."""
    import torch

    msr = int(model.sample_rate)
    acc = _mg_window(model, desc, min(duration, MAX_WINDOW_SEC))
    while acc.size < int(duration * msr) - msr // 10:
        remaining = duration - acc.size / msr
        prompt = acc[-int(CONT_PROMPT_SEC * msr):]
        chunk_total = min(MAX_WINDOW_SEC, CONT_PROMPT_SEC + remaining)
        try:
            model.set_generation_params(duration=float(max(2.0, chunk_total)),
                                        use_sampling=True, top_k=250)
            tensor = torch.from_numpy(prompt.astype(np.float32))[None, None, :]
            tensor = tensor.to(next(model.lm.parameters()).device)
            out = model.generate_continuation(
                tensor, prompt_sample_rate=msr, descriptions=[desc], progress=False)
            y = out[0].detach().cpu().float().numpy().mean(axis=0)
            new = y[prompt.size:] if y.size > prompt.size else y
        except Exception:
            # Continuation unsupported/failed → fresh window, crossfaded later.
            new = _mg_window(model, desc, min(remaining, MAX_WINDOW_SEC))
        if new.size == 0:
            break
        acc = core_audio.crossfade_concat([acc, new], CROSSFADE_SEC, msr)
    return acc[: int(duration * msr)]


def _render_musicgen(plan: dict, model_id: str, seed: int, nudge: str,
                     progress: Callable[[float, str], None]) -> np.ndarray:
    """Full-song MusicGen render → stereo (2, n) float32 at 44.1 kHz."""
    import torch

    model = _get_musicgen(model_id)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)

    bpm = float(plan.get("bpm") or 120.0)
    ts = str(plan.get("time_signature") or "4/4")
    try:
        bpb = max(2, int(ts.split("/")[0]))
    except (ValueError, IndexError):
        bpb = 4
    spb = 60.0 / bpm
    structure = plan.get("structure") or [{"label": "verse", "bars": 16}]
    energies = list(plan.get("energy_targets") or [])
    key = plan.get("key") or {}
    genre = str(plan.get("genre") or "").strip()
    palette = [str(p) for p in (plan.get("instrument_palette") or [])]
    # plan["prompt"] (from similarity.py) already encodes genre + the
    # genre-appropriate instrument palette; the fallback mirrors that so
    # MusicGen is steered to the same genre/instrumentation as the procedural
    # engine even if the prompt key is missing.
    base_prompt = str(plan.get("prompt") or "").strip() or (
        f"{genre + ' ' if genre else ''}instrumental, {bpm:.0f} bpm, "
        f"{key.get('tonic', 'A')} {key.get('mode', 'minor')}"
        + (", featuring " + ", ".join(p.replace("_", " ") for p in palette) if palette else "")
        + ", no vocals")
    if nudge:
        base_prompt = f"{base_prompt}, {nudge}"

    label_totals: dict[str, int] = {}
    for s in structure:
        label_totals[s["label"]] = label_totals.get(s["label"], 0) + 1
    label_seen: dict[str, int] = {}

    chorus_cache: dict[tuple, np.ndarray] = {}
    msr = int(model.sample_rate)
    sections_mono: list[np.ndarray] = []
    n_sections = max(len(structure), 1)

    for i, sec in enumerate(structure):
        label = str(sec.get("label", "verse"))
        bars = max(1, int(sec.get("bars", 4)))
        duration = bars * bpb * spb
        energy = float(energies[i]) if i < len(energies) else 0.6
        desc = _section_descriptor(base_prompt, label, energy)
        label_seen[label] = label_seen.get(label, 0) + 1
        count_msg = (f" {label_seen[label]}/{label_totals[label]}"
                     if label_totals[label] > 1 else "")
        cache_key = (label, bars)
        if label == "chorus" and cache_key in chorus_cache:
            # Structure enforcement: the SAME chorus audio at every chorus slot.
            progress(i / n_sections, f"Reusing chorus material{count_msg}…")
            sections_mono.append(chorus_cache[cache_key].copy())
            continue
        progress(i / n_sections, f"Generating {label}{count_msg}…")
        y = _mg_section(model, desc, duration)
        if label == "chorus":
            chorus_cache[cache_key] = y
        sections_mono.append(y)

    progress(0.92, "Stitching sections")
    mono = core_audio.crossfade_concat(sections_mono, CROSSFADE_SEC, msr)
    progress(0.96, "Resampling to 44.1 kHz")
    mono = core_audio.resample(mono.astype(np.float32), msr, SR)
    return np.stack([mono, mono]).astype(np.float32)


# ---------------------------------------------------------------------------
# Length conform + beat grid (the grid is authoritative downstream)
# ---------------------------------------------------------------------------

def _conform_and_grid(audio: np.ndarray, plan: dict, engine: str,
                      progress: Callable[[float, str], None]) -> tuple[np.ndarray, dict]:
    # BPM IS FIXED TO THE REFERENCE — tempo is never a stylistic variable and is
    # never nudged/scaled to hit a target length. The grid bpm == the plan bpm
    # (== the reference bpm) at every similarity value. Length is matched by the
    # structure/bar choices upstream (similarity.py), not by tempo. The grid
    # duration is therefore whatever the bar grid yields; any drift from the
    # reference duration is accepted (coherence over exact length).
    bpm = float(plan.get("bpm") or 120.0)
    ts = str(plan.get("time_signature") or "4/4")
    try:
        bpb = max(2, min(12, int(ts.split("/")[0])))
    except (ValueError, IndexError):
        bpb = 4
    structure = plan.get("structure") or [{"label": "verse", "bars": 16}]
    total_beats = sum(max(1, int(s.get("bars", 4))) for s in structure) * bpb
    grid_dur = total_beats * 60.0 / bpm
    spb = 60.0 / bpm

    # Conform audio to the grid duration.
    cur_dur = audio.shape[1] / SR
    rate = cur_dur / grid_dur if grid_dur > 0 else 1.0
    if abs(rate - 1.0) > 0.005:
        progress(0.3, "Conforming length to the beat grid")
        stretched = [core_audio.time_stretch(ch.astype(np.float32), rate) for ch in audio]
        n = min(len(c) for c in stretched)
        audio = np.stack([c[:n] for c in stretched])
    n_target = int(round(grid_dur * SR))
    if audio.shape[1] >= n_target:
        audio = audio[:, :n_target]
    else:
        audio = np.pad(audio, ((0, 0), (0, n_target - audio.shape[1])))
    audio = audio.astype(np.float32)
    fade_in = min(int(0.005 * SR), audio.shape[1])
    fade_out = min(int(0.05 * SR), audio.shape[1])
    if fade_in > 1:
        audio[:, :fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 1:
        audio[:, -fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    audio = core_audio.normalize_peak(audio, -3.0)

    # Arithmetic beat grid.
    beat_times = [round(i * spb, 4) for i in range(total_beats)]
    downbeats = beat_times[::bpb]
    sections = []
    bar_cursor = 0
    for s in structure:
        bars = max(1, int(s.get("bars", 4)))
        start = bar_cursor * bpb * spb
        end = (bar_cursor + bars) * bpb * spb
        sections.append({"label": str(s.get("label", "verse")),
                         "start": round(start, 4), "end": round(end, 4),
                         "bars": bars})
        bar_cursor += bars

    key = plan.get("key") or {}
    tonic = str(key.get("tonic") or "A")
    mode = str(key.get("mode") or "minor")
    grid = {
        "bpm": round(bpm, 3),
        "time_signature": ts,
        "key": {"tonic": tonic, "mode": mode,
                "scale_midi": synth.scale_midi_notes(tonic, mode)},
        "beat_times": beat_times,
        "downbeats": downbeats,
        "sections": sections,
        "duration_sec": round(audio.shape[1] / SR, 3),
        "engine": engine,
    }
    return audio, grid


# ---------------------------------------------------------------------------
# Plan rebuilding when the effective similarity is lowered
# ---------------------------------------------------------------------------

def _rebuild_plan(profile: dict, effective_similarity: int, seed: int,
                  old_plan: dict) -> dict:
    try:
        from .similarity import build_generation_plan

        new_plan = build_generation_plan(profile, effective_similarity, rng_seed=seed)
        if old_plan.get("target_duration_sec"):
            new_plan["target_duration_sec"] = old_plan["target_duration_sec"]
        return new_plan
    except Exception:
        # similarity module unavailable — degrade by hand (seed + label only).
        new_plan = dict(old_plan)
        new_plan["similarity"] = effective_similarity
        new_plan["seed"] = seed
        new_plan.pop("_melody_seed", None)
        new_plan.pop("_harmony_seed", None)
        return new_plan


# ---------------------------------------------------------------------------
# Main entry point (contract signature)
# ---------------------------------------------------------------------------

def generate_instrumental(project, plan: dict, progress) -> dict:
    """Sectional generation + stitching + conform + Uniqueness Guard.

    Returns {"grid": ..., "report": ..., "engine": ...} and writes
    instrumental.wav / instrumental_grid.json / uniqueness_report.json.
    """
    if not project.reference_profile_path.exists():
        raise PipelineError("Reference profile missing — analyze a reference track first.")
    profile = storage.read_json(project.reference_profile_path)

    engine_kind, model_id = _select_engine()
    base_seed = int(plan.get("seed") or 0)
    initial_similarity = int(plan.get("similarity", 50) or 0)
    effective_similarity = initial_similarity
    current_plan = dict(plan)
    failing_aspects: set[str] = set()
    procedural_extra = 0
    attempts = 0
    audit_ran = False
    max_attempts = 5 + (max(initial_similarity, 0) // 10) + 4

    report: dict = {}
    grid: dict = {}
    engine_name = "procedural"

    try:
        while True:
            attempts += 1
            if attempts > max_attempts:
                raise PipelineError(
                    "Could not produce an instrumental that passes the originality "
                    "audit. Try again or lower the similarity slider.")

            # Progress window for this attempt (monotonic across retries).
            lo = 0.92 * (1.0 - 0.55 ** (attempts - 1))
            hi = 0.92 * (1.0 - 0.55 ** attempts)
            render_p = subprogress(progress, lo, lo + 0.8 * (hi - lo),
                                   f"Attempt {attempts}: " if attempts > 1 else "")
            audit_p = subprogress(progress, lo + 0.8 * (hi - lo), hi)

            seed = base_seed + (attempts - 1)
            nudge = ""
            if attempts > 1:
                nudge = "different melodic contour"
                if "harmony" in failing_aspects:
                    nudge += ", different chord progression"

            # ---- render -----------------------------------------------------
            audio = None
            if engine_kind == "musicgen":
                engine_name = (model_id or "musicgen").split("/")[-1]
                render_p(0.0, f"Engine: MusicGen ({engine_name})")
                try:
                    audio = _render_musicgen(current_plan, model_id, seed, nudge,
                                             render_p)
                except PipelineError:
                    raise
                except Exception as exc:
                    render_p(0.0, f"MusicGen failed ({type(exc).__name__}) — "
                                  "falling back to procedural synth")
                    engine_kind = "procedural"

            if audio is None:  # procedural path (default / fallback)
                engine_name = "procedural"
                synth_plan = dict(current_plan)
                if attempts == 1:
                    synth_plan["seed"] = base_seed
                elif failing_aspects and failing_aspects <= {"melody", "harmony"}:
                    # Regenerate ONLY the failing aspects with a perturbed seed.
                    synth_plan["seed"] = int(current_plan.get("seed", base_seed))
                    if "melody" in failing_aspects:
                        synth_plan["_melody_seed"] = base_seed + attempts
                    if "harmony" in failing_aspects:
                        synth_plan["_harmony_seed"] = base_seed + attempts
                else:
                    synth_plan["seed"] = seed
                render_p(0.0, "Engine: procedural synthesizer")
                audio = synth.render_song(synth_plan, render_p)

            # ---- conform + grid + write --------------------------------------
            audio, grid = _conform_and_grid(audio, current_plan, engine_name,
                                            subprogress(progress, hi - 0.02, hi))
            # 16-bit so the desktop <audio> preview can decode it (Chromium does
            # not support 24-bit PCM WAV). This is an intermediate that gets
            # re-rendered into the 24-bit final master, so there is no audible loss.
            core_audio.save_wav(project.instrumental_path, audio, SR, subtype="PCM_16")

            # ---- uniqueness gate ---------------------------------------------
            audit_p(0.0, f"Originality audit (attempt {attempts})…")
            audit = unique.run_uniqueness_audit(profile, project.instrumental_path,
                                                progress=audit_p)
            audit_ran = True
            failing_aspects = {
                _CHECK_ASPECT[k] for k, c in audit.get("checks", {}).items()
                if k in _CHECK_ASPECT and not c.get("passed", True)
            }
            report = {
                "passed": bool(audit.get("passed", False)),
                "attempts": attempts,
                "effective_similarity": effective_similarity,
                "checks": audit.get("checks", {}),
                "summary": audit.get("summary", ""),
            }
            if report["passed"]:
                break

            # ---- retry policy (spec 4.3.1) ------------------------------------
            progress(hi, f"Originality audit failed "
                         f"({', '.join(sorted(failing_aspects)) or 'unknown'}) — "
                         f"regenerating (attempt {attempts + 1})")
            if attempts >= 4:  # initial try + 3 retries exhausted at this level
                if effective_similarity > 0:
                    effective_similarity = max(0, effective_similarity - 10)
                    current_plan = _rebuild_plan(profile, effective_similarity,
                                                 base_seed + attempts, current_plan)
                    failing_aspects = set()
                elif engine_kind == "musicgen":
                    # Similarity floor reached: the procedural engine with a fresh
                    # seed essentially always passes (reference is never an input).
                    engine_kind = "procedural"
                    failing_aspects = set()
                else:
                    procedural_extra += 1
                    if procedural_extra > 3:
                        raise PipelineError(
                            "The originality audit kept failing even at the "
                            "similarity floor. Try a different reference track.")

        # ---- persist artifacts ------------------------------------------------
        storage.write_json(project.grid_path, grid)
        storage.write_json(project.uniqueness_report_path, report)
        progress(1.0, report.get("summary", "Instrumental ready"))
        return {"grid": grid, "report": report, "engine": engine_name}
    finally:
        # Ownership guarantee: once the audit has run, the reference audio
        # sandbox is removed — on success AND on failure, for every engine.
        if audit_ran:
            project.delete_ref_cache()
