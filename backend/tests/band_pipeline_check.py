"""Confirm the app's default engine is the real-instrument band, end to end."""
import json, os, sys, tempfile, time, shutil
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="enjoi_band_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from enjoi.core import storage, audio as core_audio
from enjoi.modules.similarity import build_generation_plan
from enjoi.modules.generate import generate_instrumental, _select_engine

print("engine:", _select_engine())

# a folk-ish profile (genre drives the band arrangement)
profile = {
    "duration_sec": 120.0, "bpm": 116.0, "time_signature": "4/4",
    "key": {"tonic": "E", "mode": "minor"},
    "structure": [{"label": "intro", "bars": 2, "start": 0, "end": 4},
                  {"label": "verse", "bars": 8, "start": 4, "end": 20},
                  {"label": "chorus", "bars": 8, "start": 20, "end": 36},
                  {"label": "outro", "bars": 4, "start": 36, "end": 44}],
    "energy_curve": {"per_bar_rms": [0.5] * 22, "per_bar_flux": [0.5] * 22},
    "instrumentation": {"drums": 0.4, "bass": 0.3, "guitar": 0.7, "piano": 0.1, "vocals": 0.9},
    "groove": {"swing": 0.1, "pattern_class": "backbeat", "onset_histogram": [0.5] * 16},
    "genre_tags": ["folk"], "mood_tags": ["melancholic"],
    "fingerprints": {"melody_interval_ngrams": [], "chord_sequence": [],
                     "chroma_downbeat": [], "fp_hashes": []},
}
project = storage.create_project("band pipeline test")
storage.write_json(project.reference_profile_path, profile)
project.ref_cache_dir.mkdir(parents=True, exist_ok=True)

plan = build_generation_plan(profile, 80)
msgs = []
def prog(f, m):
    msgs.append((round(f, 3), m))

t = time.time()
out = generate_instrumental(project, plan, prog)
dt = time.time() - t

grid = storage.read_json(project.grid_path)
y, sr = core_audio.load_audio(project.instrumental_path)
dur = core_audio.duration_sec(project.instrumental_path)
print(f"engine={out['engine']} attempts={out['report'].get('attempts')} time={dt:.1f}s dur={dur:.1f}s")
print(f"grid.engine={grid.get('engine')} peak={float(np.abs(y).max()):.3f} rms={core_audio.rms(y):.3f}")
print("max progress frac reached:", max(f for f, _ in msgs))
shutil.copy(project.instrumental_path, r"C:\Users\kevin\Desktop\enjoi_band_pipeline.wav")

# With a local sample_library present, the primary loop engine renders (real
# warped commercial loops); band-soundfont is only the no-library fallback.
assert out["engine"] == "band-loops", \
    f"sample_library present — expected band-loops, got {out['engine']} " \
    "(loop engine regressed back to the SoundFont fallback)"
assert out["report"].get("attempts") == 1, "should be a single render"
assert np.isfinite(y).all() and float(np.abs(y).max()) > 0.05, "silent/NaN"
assert dt < 60, f"too slow: {dt}s"
print("BAND_PIPELINE_OK")
shutil.rmtree(WORK, ignore_errors=True)
