"""Project storage: one directory per project under %APPDATA%/enjoi/projects.

Layout per docs/API_CONTRACT.md. All JSON I/O goes through read_json/write_json
(UTF-8, no BOM).
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config

_state_lock = threading.Lock()

_ID_RE = re.compile(r"^p_[0-9a-f]{10}$")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


class Project:
    def __init__(self, project_id: str):
        if not _ID_RE.match(project_id):
            raise ValueError(f"invalid project id: {project_id!r}")
        self.id = project_id
        self.dir = config.projects_dir() / project_id

    # ---- well-known paths -------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.dir / "project.json"

    @property
    def reference_profile_path(self) -> Path:
        return self.dir / "reference_profile.json"

    @property
    def ref_cache_dir(self) -> Path:
        return self.dir / "_ref_cache"

    @property
    def instrumental_path(self) -> Path:
        return self.dir / "instrumental.wav"

    @property
    def grid_path(self) -> Path:
        return self.dir / "instrumental_grid.json"

    @property
    def uniqueness_report_path(self) -> Path:
        return self.dir / "uniqueness_report.json"

    @property
    def vocal_raw_path(self) -> Path:
        return self.dir / "vocal_raw.wav"

    @property
    def vocal_analysis_path(self) -> Path:
        return self.dir / "vocal_analysis.json"

    @property
    def arrangement_path(self) -> Path:
        return self.dir / "arrangement.json"

    @property
    def chops_dir(self) -> Path:
        d = self.dir / "vocal_chops"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def tuned_dir(self) -> Path:
        d = self.dir / "vocal_tuned"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def exports_dir(self) -> Path:
        d = self.dir / "exports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def manifest_path(self) -> Path:
        return self.exports_dir / "song_manifest.json"

    # ---- state ------------------------------------------------------------
    def exists(self) -> bool:
        return self.state_path.exists()

    def read_state(self) -> dict:
        with _state_lock:
            return read_json(self.state_path)

    def update_state(self, **fields) -> dict:
        with _state_lock:
            state = read_json(self.state_path)
            state.update(fields)
            write_json(self.state_path, state)
            return state

    def delete_ref_cache(self) -> None:
        """Remove the analysis-only reference audio sandbox (ownership guarantee)."""
        shutil.rmtree(self.ref_cache_dir, ignore_errors=True)


def create_project(name: str | None = None) -> Project:
    pid = "p_" + uuid.uuid4().hex[:10]
    project = Project(pid)
    project.dir.mkdir(parents=True, exist_ok=True)
    write_json(
        project.state_path,
        {
            "id": pid,
            "name": name or "Untitled song",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference": None,
            "similarity": None,
            "instrumental": None,
            "vocal": None,
            "arrangement_ready": False,
            "exports": [],
        },
    )
    return project


def get_project(project_id: str) -> Project | None:
    try:
        project = Project(project_id)
    except ValueError:
        return None
    return project if project.exists() else None


def list_projects() -> list[dict]:
    out = []
    for d in sorted(config.projects_dir().iterdir()):
        if d.is_dir() and (d / "project.json").exists() and _ID_RE.match(d.name):
            try:
                out.append(read_json(d / "project.json"))
            except Exception:
                continue
    out.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return out


def delete_project(project_id: str) -> bool:
    project = get_project(project_id)
    if project is None:
        return False
    shutil.rmtree(project.dir, ignore_errors=True)
    return True
