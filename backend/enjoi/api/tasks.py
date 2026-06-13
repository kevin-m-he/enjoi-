"""Pipeline task functions — the integration spine between routes and modules.

Each function runs inside a job thread (see core.jobs) and receives a
`progress(frac, msg)` callback. Module imports are lazy per contract rule 1.
"""
from __future__ import annotations

from pathlib import Path

from ..core import config
from ..core.errors import NotReadyError
from ..core.jobs import ProgressFn, subprogress
from ..core.storage import Project, read_json, write_json


def task_reference(project: Project, url: str, progress: ProgressFn) -> dict:
    from ..modules.reference import acquire_and_analyze

    profile = acquire_and_analyze(project, url, progress)
    src = profile.get("source", {})
    project.update_state(
        reference={
            "url": src.get("url", url),
            "video_id": src.get("video_id", ""),
            "title": src.get("title", ""),
            "channel": src.get("channel", ""),
            "duration_sec": profile.get("duration_sec", 0.0),
            "thumbnail_url": src.get("thumbnail_url", ""),
            "analyzed": True,
        }
    )
    progress(1.0, "Reference analyzed")
    return {"profile": profile}


def task_generate(project: Project, similarity: int, progress: ProgressFn) -> dict:
    if not project.reference_profile_path.exists():
        raise NotReadyError("Analyze a reference track before generating.")
    profile = read_json(project.reference_profile_path)

    from ..modules.generate import generate_instrumental
    from ..modules.similarity import build_generation_plan

    progress(0.02, "Building generation plan")
    plan = build_generation_plan(profile, similarity)
    out = generate_instrumental(project, plan, subprogress(progress, 0.05, 0.97))

    project.update_state(
        similarity=similarity,
        instrumental={
            "file": "instrumental.wav",
            "grid": "instrumental_grid.json",
            "engine": out.get("engine", "unknown"),
            "uniqueness_passed": bool(out.get("report", {}).get("passed", False)),
        },
    )

    # If a vocal was already processed, (re)build the arrangement on the new grid.
    if project.vocal_analysis_path.exists():
        _arrange(project, subprogress(progress, 0.97, 1.0))
    progress(1.0, "Instrumental ready")
    return {"plan_summary": plan.get("summary", ""), "report": out.get("report"), "engine": out.get("engine")}


def task_vocal(project: Project, uploaded_path: Path, progress: ProgressFn) -> dict:
    from ..modules.score import score_sections
    from ..modules.vocal import process_vocal

    analysis = process_vocal(project, uploaded_path, subprogress(progress, 0.0, 0.75))
    progress(0.78, "Scoring sections (chorus detection)")
    analysis = score_sections(analysis)
    write_json(project.vocal_analysis_path, analysis)
    project.update_state(
        vocal={
            "file": "vocal_raw.wav",
            "analysis": "vocal_analysis.json",
            "lyrics_available": bool(analysis.get("lyrics")),
        }
    )

    if project.grid_path.exists():
        _arrange(project, subprogress(progress, 0.85, 1.0))
    progress(1.0, "Vocal processed")
    return {"analysis": analysis}


def task_rearrange(project: Project, weights: dict | None, progress: ProgressFn) -> dict:
    if not project.vocal_analysis_path.exists():
        raise NotReadyError("Upload a vocal take first.")
    from ..modules.score import score_sections

    analysis = read_json(project.vocal_analysis_path)
    progress(0.1, "Re-scoring sections")
    analysis = score_sections(analysis, weights=weights or None)
    write_json(project.vocal_analysis_path, analysis)
    arrangement = _arrange(project, subprogress(progress, 0.3, 1.0))
    return {"analysis": analysis, "arrangement": arrangement}


def _arrange(project: Project, progress: ProgressFn) -> dict:
    if not project.grid_path.exists():
        raise NotReadyError("Generate an instrumental before arranging.")
    from ..modules.arrange import build_arrangement

    grid = read_json(project.grid_path)
    analysis = read_json(project.vocal_analysis_path)
    arrangement = build_arrangement(project, grid, analysis, progress)
    project.update_state(arrangement_ready=True)
    return arrangement


def task_render(
    project: Project,
    retune_speed: int,
    preset: str,
    loudness_preset: str,
    title: str,
    artist: str,
    include_stems: bool,
    progress: ProgressFn,
) -> dict:
    if not project.arrangement_path.exists():
        raise NotReadyError("Nothing to render — arrange vocals on an instrumental first.")
    grid = read_json(project.grid_path)
    arrangement = read_json(project.arrangement_path)

    from ..modules.export import export_song
    from ..modules.mix import mix_and_master
    from ..modules.tune import tune_vocals

    progress(0.02, "Tuning vocals")
    arrangement = tune_vocals(
        project, arrangement, grid, retune_speed, subprogress(progress, 0.02, 0.40, "Autotune: ")
    )
    write_json(project.arrangement_path, arrangement)

    master = mix_and_master(
        project, arrangement, grid, preset, loudness_preset, subprogress(progress, 0.40, 0.85, "Mix: ")
    )

    key = grid.get("key", {})
    metadata = {
        "title": title or project.read_state().get("name", "Untitled song"),
        "artist": artist or "",
        "bpm": grid.get("bpm", 0.0),
        "key": f"{key.get('tonic', '?')} {key.get('mode', '')}".strip(),
        "loudness_preset": loudness_preset,
        "preset": preset,
    }
    exports = export_song(
        project, Path(master), metadata, include_stems, subprogress(progress, 0.85, 1.0, "Export: ")
    )
    project.update_state(exports=exports)
    progress(1.0, "Song exported")
    return {"exports": exports}
