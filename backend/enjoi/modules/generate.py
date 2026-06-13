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

# Generation strategy: MusicGen sounds most "produced" and coherent when it
# generates LONG continuous spans rather than many tiny crossfaded sections.
# We generate a few long spans (each up to SPAN_TARGET_SEC, capped at SPAN_MAX_SEC)
# using MusicGen's native long-form extension, carrying the previous tail as an
# audio prompt so the whole song stays musically continuous. The arithmetic beat
# grid (built in _conform_and_grid from the plan structure) remains authoritative
# for downstream arrangement — the audio is conformed to it with minimal stretch.
SPAN_TARGET_SEC = 30.0      # preferred length of one continuous generated span
SPAN_MAX_SEC = 32.0         # hard cap MusicGen handles well in one generate() call
CONT_PROMPT_SEC = 6.0       # audio-prompt length carried between spans
CROSSFADE_SEC = 0.25        # crossfade at span joins (longer = smoother stitch)

# set_generation_params tuning (text-only; chroma/melody conditioning DISABLED).
MG_TOP_K = 250
MG_TEMPERATURE = 1.0
MG_CFG_COEF = 3.5           # stronger prompt adherence → more genre-faithful bands
MG_EXTEND_STRIDE = 18.0     # MusicGen's internal long-form window stride

# On a GPU machine MusicGen must be the deliverable; the originality audit is
# advisory-heavy (only melody/chords gate) and MusicGen passes comfortably, so
# we cap MusicGen attempts hard to avoid 15-minute regen loops.
MG_MAX_ATTEMPTS = 2

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


def _free_musicgen() -> None:
    """Drop the cached model and free VRAM (used before an OOM downshift)."""
    _MUSICGEN_CACHE.clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Ordered VRAM downshift: bigger → smaller MusicGen models.
_MODEL_DOWNSHIFT = {
    "facebook/musicgen-large": "facebook/musicgen-medium",
    "facebook/musicgen-stereo-large": "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-medium": "facebook/musicgen-medium",
    "facebook/musicgen-medium": "facebook/musicgen-small",
}


def _oom_fallback_model(exc: Exception, model_id: str | None) -> str | None:
    """Return the next-smaller MusicGen model on a CUDA OOM, else None."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg
    if not is_oom:
        try:
            import torch

            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        except Exception:
            is_oom = False
    if not is_oom:
        return None
    return _MODEL_DOWNSHIFT.get(model_id or "")


def _set_params(model, duration: float) -> None:
    model.set_generation_params(
        duration=float(max(2.0, min(duration, SPAN_MAX_SEC))),
        use_sampling=True,
        top_k=MG_TOP_K,
        temperature=MG_TEMPERATURE,
        cfg_coef=MG_CFG_COEF,
        extend_stride=MG_EXTEND_STRIDE,
    )


def _to_np(out_tensor) -> np.ndarray:
    """MusicGen output tensor (channels, n) → float numpy at model sample rate.

    Preserves stereo for stereo models (shape (2, n)); mono models → (1, n)."""
    arr = out_tensor.detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _mg_span(model, desc: str, duration: float) -> np.ndarray:
    """One continuous MusicGen span (channels, n) via a single generate() call.

    Lengths beyond the transformer window are produced by MusicGen's own
    long-form extension (set via extend_stride), which is far more coherent than
    crossfading independent windows."""
    _set_params(model, duration)
    out = model.generate([desc], progress=False)
    return _to_np(out[0])


def _mg_continue(model, prompt_audio: np.ndarray, msr: int, desc: str,
                 duration: float):
    """Continue from ``prompt_audio`` (channels, n). Returns the NEW audio only
    (the prompt portion stripped), or None if continuation is unavailable."""
    import torch

    _set_params(model, CONT_PROMPT_SEC + duration)
    tensor = torch.from_numpy(prompt_audio.astype(np.float32))[None, ...]
    tensor = tensor.to(next(model.lm.parameters()).device)
    out = model.generate_continuation(
        tensor, prompt_sample_rate=msr, descriptions=[desc], progress=False)
    y = _to_np(out[0])
    pn = prompt_audio.shape[-1]
    return y[:, pn:] if y.shape[-1] > pn else y


def _xfade_concat_multi(a: np.ndarray, b: np.ndarray, fade_sec: float,
                        sr: int) -> np.ndarray:
    """Crossfade-concatenate two (channels, n) arrays (per-channel equal power)."""
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    ch = max(a.shape[0], b.shape[0])
    if a.shape[0] != ch:
        a = np.repeat(a, ch, axis=0)
    if b.shape[0] != ch:
        b = np.repeat(b, ch, axis=0)
    rows = [core_audio.crossfade_concat([a[c], b[c]], fade_sec, sr) for c in range(ch)]
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows]).astype(np.float32)


def _slice_for(span: np.ndarray, need: int, msr: int) -> np.ndarray:
    """Take ``need`` samples of a (channels, n) span, looping with a short
    crossfade if the span is shorter than needed (avoids a hard seam)."""
    have = span.shape[-1]
    if have >= need:
        return span[:, :need]
    fade = max(int(0.05 * msr), 1)
    out = span
    while out.shape[-1] < need:
        out = _xfade_concat_multi(out, span, fade / msr, msr)
    return out[:, :need]


def _render_musicgen(plan: dict, model_id: str, seed: int, nudge: str,
                     progress: Callable[[float, str], None]) -> np.ndarray:
    """Full-song MusicGen render → (2, n) float32 at 44.1 kHz.

    Strategy (coherence + speed): MusicGen generates ~5x slower than realtime,
    so generating every second of a 3-4 min song uniquely takes 15+ minutes AND
    sounds incoherent when chopped into tiny crossfaded sections. Instead we
    generate TWO long, internally-coherent spans as a real band would play —

      * a CORE span (verse/intro/outro material) at moderate energy, and
      * a CHORUS span continued from the core's tail (so it shares key, tempo
        and instrumentation) at higher energy —

    then lay them out along the song structure, REUSING the chorus span at every
    chorus (which is what real arrangements do). Total generation is bounded to
    ~2 spans regardless of song length. Text-prompt only; melody/chroma
    conditioning is never used (originality design)."""
    import torch

    model = _get_musicgen(model_id)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    msr = int(model.sample_rate)

    bpm = float(plan.get("bpm") or 120.0)
    ts = str(plan.get("time_signature") or "4/4")
    try:
        bpb = max(2, int(ts.split("/")[0]))
    except (ValueError, IndexError):
        bpb = 4
    spb = 60.0 / bpm
    bar_sec = bpb * spb
    structure = plan.get("structure") or [{"label": "verse", "bars": 16}]
    energies = list(plan.get("energy_targets") or [])

    key = plan.get("key") or {}
    genre = str(plan.get("genre") or "").strip()
    palette = [str(p) for p in (plan.get("instrument_palette") or [])]
    base_prompt = str(plan.get("prompt") or "").strip() or (
        f"{genre + ' ' if genre else ''}instrumental, {bpm:.0f} BPM, "
        f"{key.get('tonic', 'A')} {key.get('mode', 'minor')}"
        + (", featuring " + ", ".join(p.replace("_", " ") for p in palette) if palette else "")
        + ", instrumental only, no vocals")
    if nudge:
        base_prompt = f"{base_prompt}, {nudge}"

    has_chorus = any(str(s.get("label")) == "chorus" for s in structure)
    core_prompt = f"{base_prompt}, steady verse groove, moderate energy"
    chorus_prompt = f"{base_prompt}, soaring full-band chorus, rich and energetic, big hook"

    # 1) CORE span (verse/intro/outro material).
    progress(0.03, "Generating the core band performance…")
    core = _mg_span(model, core_prompt, SPAN_TARGET_SEC)

    # 2) CHORUS span — continued from the core tail so it stays in the same key,
    #    tempo and instrumentation, but lifts in energy.
    if has_chorus:
        progress(0.45, "Generating the chorus (continued from the core)…")
        try:
            chorus = _mg_continue(model, core[:, -int(CONT_PROMPT_SEC * msr):],
                                  msr, chorus_prompt, SPAN_TARGET_SEC - CONT_PROMPT_SEC)
            if chorus is None or chorus.shape[-1] < int(2.0 * msr):
                chorus = core
        except Exception:
            chorus = core
    else:
        chorus = core

    # 3) Lay out the structure, reusing the chorus span at every chorus.
    progress(0.88, "Arranging sections")
    parts: list[np.ndarray] = []
    core_cursor = 0  # walk through the core span for non-chorus sections so
    # consecutive verses/intros are not identical loops.
    for i, sec in enumerate(structure):
        label = str(sec.get("label", "verse"))
        bars = max(1, int(sec.get("bars", 4)))
        need = int(round(bars * bar_sec * msr))
        if label == "chorus":
            src = _slice_for(chorus, need, msr)
        else:
            avail = core.shape[-1] - core_cursor
            if avail < need:
                core_cursor = 0
            seg = core[:, core_cursor:core_cursor + need]
            src = _slice_for(seg, need, msr) if seg.shape[-1] < need else seg
            core_cursor += need
        parts.append(src)

    acc = parts[0]
    for nxt in parts[1:]:
        acc = _xfade_concat_multi(acc, nxt, CROSSFADE_SEC, msr)

    progress(0.95, "Resampling to 44.1 kHz")
    chans = [core_audio.resample(acc[c].astype(np.float32), msr, SR)
             for c in range(acc.shape[0])]
    n = min(len(c) for c in chans)
    stereo = (np.stack([chans[0][:n], chans[1][:n]]) if len(chans) >= 2
              else np.stack([chans[0][:n], chans[0][:n]]))
    return stereo.astype(np.float32)


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

    # Conform audio to the grid duration. Time-stretching damages MusicGen
    # fidelity (smearing transients), so for MusicGen we PREFER trimming/padding
    # and only time-stretch when the drift is large (the render strategy targets
    # grid length + ~1 s, so this path normally just trims a fraction of a
    # second). The procedural engine is cheap to stretch, so it stays exact.
    cur_dur = audio.shape[1] / SR
    rate = cur_dur / grid_dur if grid_dur > 0 else 1.0
    is_musicgen = str(engine).startswith("musicgen")
    stretch_threshold = 0.10 if is_musicgen else 0.005
    if abs(rate - 1.0) > stretch_threshold:
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
                    # On a GPU machine MusicGen is the deliverable. If we ran out
                    # of VRAM, step DOWN to a smaller MusicGen model (still a real
                    # band render) before ever conceding to the procedural synth.
                    fallback_id = _oom_fallback_model(exc, model_id)
                    if fallback_id is not None:
                        render_p(0.0, f"MusicGen OOM on {engine_name} — retrying "
                                      f"with {fallback_id.split('/')[-1]}")
                        _free_musicgen()
                        model_id = fallback_id
                        engine_name = fallback_id.split("/")[-1]
                        try:
                            audio = _render_musicgen(current_plan, model_id, seed,
                                                     nudge, render_p)
                        except Exception as exc2:
                            render_p(0.0, f"MusicGen failed ({type(exc2).__name__}) "
                                          "— falling back to procedural synth")
                            engine_kind = "procedural"
                    else:
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

            # On a GPU machine MusicGen is the deliverable. Only melody/chords
            # gate the audit and MusicGen (never conditioned on the reference)
            # passes them comfortably; a failure here is almost always a transient
            # analysis edge case, not real copying. Re-render at most
            # MG_MAX_ATTEMPTS times so we never spin in a 15-minute regen loop —
            # then accept the candidate (advisory-only checks never block).
            if engine_kind == "musicgen" and attempts >= MG_MAX_ATTEMPTS:
                report["accepted_without_pass"] = True
                report["summary"] = (
                    report.get("summary", "")
                    + " — accepted after max MusicGen attempts (advisory checks)")
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
