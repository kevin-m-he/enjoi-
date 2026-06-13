"""Verify the APP's MusicGen path (generate.generate_instrumental) end to end on
GPU, using a real reference profile but a tiny structure so it's quick."""
import json, os, sys, tempfile, shutil
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="enjoi_mg_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enjoi.core import storage, audio as core_audio
from enjoi.modules.similarity import build_generation_plan
from enjoi.modules.generate import generate_instrumental, _select_engine

SRC = sys.argv[1]
profile = json.loads(Path(SRC).read_text(encoding="utf-8"))

print("engine:", _select_engine(), flush=True)

project = storage.create_project("musicgen app check")
storage.write_json(project.reference_profile_path, profile)
project.ref_cache_dir.mkdir(parents=True, exist_ok=True)

plan = build_generation_plan(profile, 72)
# shrink to ~12s so MusicGen generation is fast for the test
plan["structure"] = [{"label": "intro", "bars": 2}, {"label": "chorus", "bars": 4}]
plan["energy_targets"] = [0.4, 0.85]
plan.pop("energy_per_bar", None)
bpm = plan["bpm"]
plan["target_duration_sec"] = round(6 * 4 * 60.0 / bpm, 2)  # 6 bars

def progress(frac, msg):
    print(f"  {frac*100:5.1f}%  {msg}", flush=True)

out = generate_instrumental(project, plan, progress)
inst = project.instrumental_path
dur = core_audio.duration_sec(inst)
y, sr = core_audio.load_audio(inst)
import numpy as np
print("RESULT engine=", out["engine"], "passed=", out["report"]["passed"],
      "dur=", round(dur, 1), "finite=", bool(np.isfinite(y).all()),
      "nonsilent=", float(np.abs(y).max()) > 0.05, flush=True)
assert out["engine"].startswith("musicgen"), "expected MusicGen engine"
assert inst.exists() and dur > 3, "instrumental too short/missing"
assert np.isfinite(y).all() and float(np.abs(y).max()) > 0.05, "silent/NaN output"
print("APP_MUSICGEN_OK", flush=True)
shutil.rmtree(WORK, ignore_errors=True)
