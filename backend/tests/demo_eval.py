"""End-to-end demo + objective scorer for the Drake/Kestral test case.

Runs the REAL pipeline (analyze upload -> generate -> vocal -> arrange -> tune ->
mix/master) on the two provided files, writes the song to the Desktop, and prints
a scorecard against industry-ish targets so we can iterate until it passes.

Usage:  .venv/Scripts/python.exe tests/demo_eval.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WORK = Path(tempfile.mkdtemp(prefix="enjoi_demo_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enjoi.core import storage  # noqa: E402
from enjoi.modules.reference import analyze_uploaded  # noqa: E402
from enjoi.modules.similarity import build_generation_plan  # noqa: E402
from enjoi.modules.generate import generate_instrumental  # noqa: E402
from enjoi.modules.vocal import process_vocal  # noqa: E402
from enjoi.modules.score import score_sections  # noqa: E402
from enjoi.modules.arrange import build_arrangement  # noqa: E402
from enjoi.modules.tune import tune_vocals  # noqa: E402
from enjoi.modules.mix import mix_and_master  # noqa: E402
from enjoi.core.storage import read_json, write_json  # noqa: E402

REF = Path(r"C:\Users\kevin\Downloads\Drake - Find Your Love (HQ).mp3")
VOX = Path(r"C:\Users\kevin\Downloads\Kestral Cir 37.mp3")
OUT = Path.home() / "Desktop" / "enjoi_demo_out.wav"

# Ground-truth-ish targets for the Drake reference (real song ~96 BPM, B/E-ish).
REF_TRUE_BPM = 96.0


def _quiet(*a):
    pass


def _grid_align_score(y, sr, bpm):
    """Fraction of onset energy that lands within ±40 ms of the beat grid —
    a proxy for 'is it on the beat'. 1.0 = perfectly locked, ~0.25 = random."""
    import librosa
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    if bpm <= 0:
        return 0.0
    beat = 60.0 / bpm
    phase = np.abs(((times % beat) / beat) - 0.0)
    phase = np.minimum(phase, 1.0 - phase)  # distance to nearest beat (in beats)
    on = phase < (0.04 / beat)              # within ±40 ms
    tot = float(onset_env.sum()) + 1e-9
    return float(onset_env[on].sum() / tot)


def _lufs(y, sr):
    import pyloudnorm as pyln
    try:
        yy = y.T if y.ndim == 2 else y
        return float(pyln.Meter(sr).integrated_loudness(yy.astype(np.float64)))
    except Exception:
        return float("nan")


def run():
    t0 = time.time()
    proj = storage.create_project("demo")
    proj.ref_cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] analyze reference …")
    profile = analyze_uploaded(proj, REF, "Drake - Find Your Love", _quiet)
    write_json(proj.reference_profile_path, profile)
    det_bpm = profile.get("bpm")
    det_genre = profile.get("genre_tags")
    det_key = profile.get("key", {})

    print("[2/6] generate instrumental …")
    plan = build_generation_plan(profile, 100, salt=proj.id)
    generate_instrumental(proj, plan, _quiet)
    grid = read_json(proj.grid_path)

    print("[3/6] process vocal …")
    process_vocal(proj, VOX, _quiet)
    analysis = read_json(proj.vocal_analysis_path)
    analysis = score_sections(analysis)
    write_json(proj.vocal_analysis_path, analysis)

    print("[4/6] arrange …")
    build_arrangement(proj, grid, analysis, _quiet)

    print("[5/6] tune + mix/master …")
    arrangement = read_json(proj.arrangement_path)
    arrangement = tune_vocals(proj, arrangement, grid, 35, _quiet)
    write_json(proj.arrangement_path, arrangement)
    master = mix_and_master(proj, arrangement, grid, "rnb", "loud", _quiet)

    print("[6/6] score …")
    import soundfile as sf
    import shutil
    y, sr = sf.read(str(master))
    if y.ndim == 2:
        y = y.T
    mono = y.mean(axis=0) if y.ndim == 2 else y
    shutil.copy(master, OUT)

    import librosa
    obs_tempo = float(np.atleast_1d(librosa.beat.tempo(y=mono, sr=sr)[0]))
    lufs = _lufs(y, sr)
    peak = float(np.max(np.abs(y)))
    align = _grid_align_score(mono, sr, grid.get("bpm", det_bpm))
    has_vocals = bool(np.any(np.abs(_stem(proj, "_stem_vocals.wav")) > 1e-4))
    n_place = len((arrangement or {}).get("placements") or [])
    lyr = bool(analysis.get("lyrics"))

    dt = time.time() - t0
    print("\n" + "=" * 64)
    print("ENJOI DEMO SCORECARD  (Drake ref @100% + Kestral vox)")
    print("=" * 64)
    bpm_err = abs((grid.get("bpm") or det_bpm) - REF_TRUE_BPM)
    print(f"detected genre     : {det_genre}      (Drake = R&B/pop)")
    print(f"detected key       : {det_key.get('tonic')} {det_key.get('mode')}")
    print(f"plan/grid BPM      : {grid.get('bpm'):.1f}   target~{REF_TRUE_BPM:.0f}   "
          f"err={bpm_err:.1f}  {'OK' if bpm_err <= 6 else 'FAIL'}")
    print(f"LUFS               : {lufs:.1f}   target -8..-11   "
          f"{'OK' if -12 <= lufs <= -7 else 'FAIL'}")
    print(f"true peak (approx) : {20*np.log10(peak+1e-9):.2f} dBFS  "
          f"{'OK' if peak <= 0.995 else 'HOT'}")
    print(f"beat-grid lock     : {align*100:.0f}%   (>=45% = tight, ~25% = random)  "
          f"{'OK' if align >= 0.45 else 'FAIL'}")
    print(f"vocals present     : {has_vocals}   placements={n_place}   lyrics(whisper)={lyr}")
    print(f"render time        : {dt:.0f}s")
    print(f"\nwrote {OUT}")
    shutil.rmtree(WORK, ignore_errors=True)


def _stem(proj, name):
    import soundfile as sf
    p = proj.exports_dir / name
    if not p.exists():
        return np.zeros(1)
    y, _ = sf.read(str(p))
    return y


if __name__ == "__main__":
    run()
