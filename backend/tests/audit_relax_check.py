"""Verify the relaxed Uniqueness Guard lets generation pass quickly.

Reuses a real reference_profile.json (real fingerprints from an actual
download) so the audit runs against genuine reference data, at similarity 100
(worst case for chroma correlation). Runs in a throwaway data dir.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SRC_PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
WORK = Path(tempfile.mkdtemp(prefix="enjoi_audit_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enjoi.core import storage  # noqa: E402
from enjoi.modules.similarity import build_generation_plan  # noqa: E402
from enjoi.modules.generate import generate_instrumental  # noqa: E402

if SRC_PROFILE is None or not SRC_PROFILE.exists():
    print("usage: audit_relax_check.py <path-to-reference_profile.json>")
    sys.exit(2)

profile = json.loads(SRC_PROFILE.read_text(encoding="utf-8"))
project = storage.create_project("audit relax test")
storage.write_json(project.reference_profile_path, profile)
# Simulate the post-analysis state (ref cache already gone; audit uses fingerprints).
project.ref_cache_dir.mkdir(parents=True, exist_ok=True)

def progress(frac: float, msg: str) -> None:
    if "attempt" in msg.lower() or "originality check" in msg.lower():
        print(f"  {frac * 100:5.1f}%  {msg}")

for similarity in (100, 72, 50):
    print(f"\n=== similarity {similarity} ===")
    plan = build_generation_plan(profile, similarity)
    out = generate_instrumental(project, plan, progress)
    rep = out["report"]
    checks = {k: (v["value"], v["passed"]) for k, v in rep["checks"].items()}
    print(f"  -> engine={out['engine']} passed={rep['passed']} "
          f"attempts={rep['attempts']} eff_sim={rep['effective_similarity']}")
    print(f"  -> checks={checks}")
    assert rep["passed"], f"similarity {similarity} FAILED to pass audit"
    assert rep["attempts"] <= 2, f"similarity {similarity} took {rep['attempts']} attempts"
    # restore ref cache for the next iteration (generate deletes it)
    project.ref_cache_dir.mkdir(parents=True, exist_ok=True)

print("\nALL PASSED on first or second attempt — generation no longer stalls.")
shutil.rmtree(WORK, ignore_errors=True)
