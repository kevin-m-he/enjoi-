"""REST + WebSocket routes (see docs/API_CONTRACT.md)."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .. import __version__
from ..core import config, deps
from ..core.errors import PipelineError
from ..core.jobs import manager
from ..core.storage import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    read_json,
)
from . import tasks

router = APIRouter()


def _project_or_404(pid: str):
    project = get_project(pid)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


# ---- health / search ---------------------------------------------------------

@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "capabilities": deps.capabilities()}


@router.get("/api/search")
def search(q: str, limit: int = config.SEARCH_RESULT_LIMIT) -> dict:
    if not q.strip():
        return {"results": []}
    from ..modules.search import search_youtube

    try:
        return {"results": search_youtube(q.strip(), limit=min(max(limit, 1), 25))}
    except PipelineError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---- projects ------------------------------------------------------------------

class CreateProjectBody(BaseModel):
    name: str | None = None


@router.get("/api/projects")
def projects_list() -> dict:
    return {"projects": list_projects()}


@router.post("/api/projects")
def projects_create(body: CreateProjectBody | None = None) -> dict:
    project = create_project((body.name if body else None) or None)
    return project.read_state()


@router.get("/api/projects/{pid}")
def project_get(pid: str) -> dict:
    return _project_or_404(pid).read_state()


@router.delete("/api/projects/{pid}")
def project_delete(pid: str) -> dict:
    if not delete_project(pid):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


# ---- pipeline stage triggers ----------------------------------------------------

class ReferenceBody(BaseModel):
    url: str


@router.post("/api/projects/{pid}/reference")
def start_reference(pid: str, body: ReferenceBody) -> dict:
    project = _project_or_404(pid)
    job = manager.start("reference", pid, tasks.task_reference, project, body.url)
    return {"job_id": job.id}


class GenerateBody(BaseModel):
    similarity: int = Field(ge=0, le=100)


@router.post("/api/projects/{pid}/generate")
def start_generate(pid: str, body: GenerateBody) -> dict:
    project = _project_or_404(pid)
    job = manager.start("generate", pid, tasks.task_generate, project, body.similarity)
    return {"job_id": job.id}


@router.post("/api/projects/{pid}/vocal")
async def start_vocal(pid: str, file: UploadFile = File(...)) -> dict:
    project = _project_or_404(pid)
    suffix = Path(file.filename or "take.wav").suffix.lower()
    if suffix not in (".wav", ".mp3"):
        raise HTTPException(status_code=400, detail="Please upload a .wav or .mp3 file")
    fd, tmp_name = tempfile.mkstemp(prefix="enjoi_vocal_", suffix=suffix)
    tmp = Path(tmp_name)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job = manager.start("vocal", pid, tasks.task_vocal, project, tmp)
    return {"job_id": job.id}


class RearrangeBody(BaseModel):
    weights: dict[str, float] | None = None


@router.post("/api/projects/{pid}/rearrange")
def start_rearrange(pid: str, body: RearrangeBody | None = None) -> dict:
    project = _project_or_404(pid)
    job = manager.start("rearrange", pid, tasks.task_rearrange, project, body.weights if body else None)
    return {"job_id": job.id}


class RenderBody(BaseModel):
    retune_speed: int = Field(default=config.DEFAULT_RETUNE_SPEED, ge=0, le=100)
    preset: str = "pop"
    loudness_preset: str = "streaming"
    title: str = ""
    artist: str = ""
    include_stems: bool = False


@router.post("/api/projects/{pid}/render")
def start_render(pid: str, body: RenderBody) -> dict:
    project = _project_or_404(pid)
    if body.preset not in config.MIX_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {body.preset}")
    if body.loudness_preset not in config.LOUDNESS_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown loudness preset: {body.loudness_preset}")
    job = manager.start(
        "render", pid, tasks.task_render, project,
        body.retune_speed, body.preset, body.loudness_preset,
        body.title, body.artist, body.include_stems,
    )
    return {"job_id": job.id}


# ---- arrangement read/override ---------------------------------------------------

@router.get("/api/projects/{pid}/arrangement")
def arrangement_get(pid: str) -> dict:
    project = _project_or_404(pid)
    if not project.arrangement_path.exists():
        raise HTTPException(status_code=404, detail="No arrangement yet")
    return read_json(project.arrangement_path)


class ArrangementBody(BaseModel):
    placements: list[dict]


@router.put("/api/projects/{pid}/arrangement")
def arrangement_put(pid: str, body: ArrangementBody) -> dict:
    project = _project_or_404(pid)
    if not project.arrangement_path.exists():
        raise HTTPException(status_code=404, detail="No arrangement yet")
    from ..modules.arrange import apply_overrides

    arrangement = read_json(project.arrangement_path)
    try:
        updated = apply_overrides(project, arrangement, body.placements)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return updated


# ---- vocal analysis / similarity preview -------------------------------------------

@router.get("/api/projects/{pid}/vocal-analysis")
def vocal_analysis_get(pid: str) -> dict:
    project = _project_or_404(pid)
    if not project.vocal_analysis_path.exists():
        raise HTTPException(status_code=404, detail="No vocal analysis yet")
    return read_json(project.vocal_analysis_path)


@router.get("/api/projects/{pid}/reference-profile")
def reference_profile_get(pid: str) -> dict:
    project = _project_or_404(pid)
    if not project.reference_profile_path.exists():
        raise HTTPException(status_code=404, detail="No reference profile yet")
    profile = read_json(project.reference_profile_path)
    profile.pop("ref_audio", None)  # never expose the sandbox path
    return profile


@router.get("/api/similarity/preview")
def similarity_preview(pid: str, value: int) -> dict:
    project = _project_or_404(pid)
    if not project.reference_profile_path.exists():
        raise HTTPException(status_code=404, detail="No reference profile yet")
    from ..modules.similarity import similarity_summary

    profile = read_json(project.reference_profile_path)
    return {"summary": similarity_summary(profile, min(max(value, 0), 100))}


# ---- jobs ----------------------------------------------------------------------

@router.get("/api/jobs/{job_id}")
def job_get(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    q = manager.subscribe()
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=20.0)
            except asyncio.TimeoutError:
                payload = {"type": "ping"}
            await ws.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.unsubscribe(q)
