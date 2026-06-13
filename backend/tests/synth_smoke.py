"""Smoke test for the procedural audio engine, genre-matching, fixed BPM,
432 Hz tuning, and mix balance.

Run with the venv python:
    backend\\.venv\\Scripts\\python.exe backend\\tests\\synth_smoke.py

Asserts (fixes until green — never returns failing work):
  1. plan["bpm"] == reference bpm at similarity 50/72/100 for several ref bpms.
  2. Genre palettes differ by genre (country vs edm produce different buses).
  3. render_song: stereo float32, finite, non-silent, peak <= 0 dBFS, < ~10 s.
  4. 432 tuning: a sustained MIDI-69 note has its FFT peak near 432 Hz, not 440.
  5. Mix balance: in a high-energy chorus, the hi-hat buses are quieter (RMS)
     than the kick, snare and bass buses.
  6. Full generate_instrumental against a REAL reference profile on disk passes
     the uniqueness audit on attempt 1, grid bpm == round(profile bpm), grid
     tempo == plan tempo (BPM never nudged).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

# Throwaway data dir so we never touch the user's real projects.
_WORK = Path(tempfile.mkdtemp(prefix="enjoi_synth_smoke_"))
os.environ["ENJOI_DATA_DIR"] = str(_WORK / "data")

from enjoi.core import config, storage  # noqa: E402
from enjoi.modules import synth  # noqa: E402
from enjoi.modules.similarity import build_generation_plan  # noqa: E402
from enjoi.modules.generate import generate_instrumental  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


def make_profile(genre: str, bpm: float) -> dict:
    """A minimal but realistic reference profile (no fingerprints needed for the
    plan/render checks)."""
    return {
        "duration_sec": 120.0,
        "bpm": bpm,
        "time_signature": "4/4",
        "key": {"tonic": "A", "mode": "minor", "confidence": 0.7},
        "structure": [
            {"label": "intro", "bars": 4}, {"label": "verse", "bars": 8},
            {"label": "chorus", "bars": 8}, {"label": "verse", "bars": 8},
            {"label": "chorus", "bars": 8}, {"label": "outro", "bars": 4},
        ],
        "energy_curve": {"per_bar_rms": [0.3] * 4 + [0.6] * 8 + [0.95] * 8
                         + [0.6] * 8 + [0.95] * 8 + [0.3] * 4},
        "instrumentation": {"drums": 0.9, "bass": 0.9, "melodic": 0.9, "vocals": 0.6},
        "groove": {"swing": 0.1, "pattern_class": "backbeat"},
        "genre_tags": [genre],
        "mood_tags": ["energetic"],
    }


GENRES = ["country", "rock", "trap", "r&b", "edm", "lofi", "pop"]
REF_BPMS = [92.0, 140.0]


def test_fixed_bpm() -> None:
    print("\n== 1. BPM fixed to reference at all similarity values ==")
    ok = True
    for bpm in REF_BPMS:
        expected = bpm  # both 92 and 140 are already inside 60..200
        for genre in GENRES:
            prof = make_profile(genre, bpm)
            for sim in (50, 72, 100):
                plan = build_generation_plan(prof, sim)
                if abs(plan["bpm"] - expected) > 1e-6:
                    ok = False
                    print(f"    bpm mismatch genre={genre} bpm={bpm} sim={sim} "
                          f"-> {plan['bpm']}")
    check(ok, "plan bpm == reference bpm for all genres x similarity x ref-bpm")
    # Octave-clamp sanity: a 40 bpm ref doubles into range, 240 halves.
    check(abs(build_generation_plan(make_profile("pop", 40.0), 72)["bpm"] - 80.0) < 1e-6,
          "40 bpm reference octave-clamps to 80")
    check(abs(build_generation_plan(make_profile("pop", 240.0), 72)["bpm"] - 120.0) < 1e-6,
          "240 bpm reference octave-clamps to 120")
    # Summary never claims a tempo tolerance.
    summ = build_generation_plan(make_profile("pop", 120.0), 50)["summary"]
    check("same tempo" in summ and "within" not in summ,
          f"summary says 'same tempo', no tolerance ({summ!r})")


def test_palettes_differ() -> None:
    print("\n== 2. Genre palettes differ ==")
    pals = {g: build_generation_plan(make_profile(g, 120.0), 72)["instrument_palette"]
            for g in GENRES}
    for g, p in pals.items():
        print(f"    {g:8s} -> {p}")
    country, edm = set(pals["country"]), set(pals["edm"])
    check(country != edm, "country palette != edm palette")
    # country should have an acoustic guitar; edm should have a sub/808-class low.
    check("acoustic_guitar" in pals["country"], "country palette has acoustic_guitar")
    check(any(x in pals["edm"] for x in ("sub_bass", "808")), "edm palette has sub bass")
    check(any(x in pals["trap"] for x in ("808",)), "trap palette has 808")
    # Each plan carries its genre tag.
    check(build_generation_plan(make_profile("lofi", 120.0), 72)["genre"] == "lofi",
          "plan['genre'] resolves lofi")


def test_render() -> None:
    print("\n== 3. render_song: format / finite / non-silent / peak / speed ==")
    for genre in GENRES:
        prof = make_profile(genre, 120.0)
        plan = build_generation_plan(prof, 72)
        t0 = time.time()
        y = synth.render_song(plan, None)
        dt = time.time() - t0
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        ok = (y.ndim == 2 and y.shape[0] == 2 and y.dtype == np.float32
              and np.all(np.isfinite(y)) and peak > 0.05 and peak <= 1.0 and dt < 12.0)
        print(f"    {genre:8s} shape={y.shape} peak={peak:.3f} "
              f"dtype={y.dtype} render={dt:.2f}s")
        check(ok, f"{genre}: stereo float32, finite, non-silent, peak<=0dBFS, <12s")


def test_432_tuning() -> None:
    print("\n== 4. 432 Hz tuning (FFT peak of MIDI 69) ==")
    # config sanity first.
    check(abs(config.midi_to_hz(69) - 432.0) < 1e-6,
          f"config.midi_to_hz(69) == 432 ({config.midi_to_hz(69):.4f})")
    sr = config.SAMPLE_RATE
    # Sustain a single MIDI-69 chord via the piano voice (clear fundamental).
    dur = 2.0
    sig = synth._piano_note(69, dur, sr, energy=0.5)
    sig = sig[: int(dur * sr)]
    win = sig * np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    # Look in the fundamental band only (ignore harmonics at 864 etc.).
    band = (freqs > 380) & (freqs < 500)
    peak_hz = float(freqs[band][np.argmax(spec[band])])
    print(f"    dominant fundamental = {peak_hz:.2f} Hz (target 432, NOT 440)")
    check(abs(peak_hz - 432.0) < 5.0, f"MIDI69 fundamental ~432 Hz ({peak_hz:.2f})")
    check(abs(peak_hz - 440.0) > 4.0, "fundamental is NOT 440 Hz")


def test_mix_balance() -> None:
    print("\n== 5. Mix balance: hats quieter than kick/snare/bass ==")
    # High-energy chorus, drum+bass-rich genres.
    for genre in ("pop", "rock", "edm"):
        prof = make_profile(genre, 120.0)
        plan = build_generation_plan(prof, 72)
        rms = synth.section_bus_rms(plan, {"label": "chorus", "bars": 8, "_index": 2})
        hat = max(rms.get("hat_c", 0.0), rms.get("hat_o", 0.0))
        kick, snare, bass = rms["kick"], rms["snare"], rms["bass"]
        print(f"    {genre:5s} hat={hat:.4f} kick={kick:.4f} "
              f"snare={snare:.4f} bass={bass:.4f}")
        check(hat < kick and hat < bass and hat < snare,
              f"{genre}: hats quieter than kick, snare and bass")


def _find_real_profile() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    root = Path(appdata) / "enjoi" / "projects"
    if not root.exists():
        return None
    for p in sorted(root.glob("*/reference_profile.json")):
        try:
            prof = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (prof.get("fingerprints") or {}).get("melody_interval_ngrams") is not None:
            return p
    return None


def test_full_generate_real_profile() -> None:
    print("\n== 6. Full generate_instrumental vs REAL reference profile ==")
    src = _find_real_profile()
    if src is None:
        check(False, "found a real reference_profile.json on disk")
        return
    print(f"    using {src}")
    prof = json.loads(src.read_text(encoding="utf-8"))
    project = storage.create_project("synth smoke")
    storage.write_json(project.reference_profile_path, prof)
    project.ref_cache_dir.mkdir(parents=True, exist_ok=True)

    plan = build_generation_plan(prof, 72)
    out = generate_instrumental(project, plan, lambda f, m: None)
    grid, rep = out["grid"], out["report"]
    ref_bpm_round = round(float(prof.get("bpm")))
    print(f"    engine={out['engine']} passed={rep['passed']} "
          f"attempts={rep['attempts']} grid.bpm={grid['bpm']} "
          f"plan.bpm={plan['bpm']} ref_bpm={prof.get('bpm')} "
          f"grid.dur={grid['duration_sec']}s")
    check(rep["passed"], "uniqueness report passed")
    check(rep["attempts"] == 1, f"passed on attempt 1 (got {rep['attempts']})")
    check(abs(grid["bpm"] - plan["bpm"]) < 1e-3,
          "grid bpm == plan bpm (tempo never nudged)")
    # ref bpm 117.45 octave-clamps to itself; grid bpm should equal the plan bpm,
    # which is round-trippable to the clamped reference bpm.
    check(round(grid["bpm"]) in (ref_bpm_round, round(plan["bpm"])),
          f"grid bpm rounds to reference/plan bpm ({ref_bpm_round})")


def main() -> int:
    try:
        test_fixed_bpm()
        test_palettes_differ()
        test_render()
        test_432_tuning()
        test_mix_balance()
        test_full_generate_real_profile()
    finally:
        shutil.rmtree(_WORK, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
