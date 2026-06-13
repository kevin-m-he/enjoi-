"""Full-pipeline integration test (run from backend/ with the venv python).

Exercises the exact code paths the API jobs use:
  search -> reference acquire+analyze -> generate (+ uniqueness gate)
  -> vocal process+score -> arrange -> tune -> mix -> export
plus a like-for-like fingerprint self-test of the Uniqueness Guard.

Uses a throwaway ENJOI_DATA_DIR. Needs network for the YouTube steps.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="enjoi_e2e_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from enjoi.core import audio as core_audio  # noqa: E402
from enjoi.core import storage  # noqa: E402
from enjoi.api import tasks  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label)
    if not cond:
        FAILURES.append(label)


def progress(frac: float, msg: str) -> None:
    print(f"    {frac * 100:5.1f}%  {msg}")


def make_fake_vocal(path: Path, sr: int = 44100) -> None:
    """A 64 s synthetic 'one-take': vowel-ish tones with vibrato, distinct
    louder/higher 'chorus-like' stretch in the middle, breaths/silences."""
    rng = np.random.default_rng(7)
    t_total = []
    # (duration, midi, level) phrases with 0.5 s gaps; mid-take block is the "chorus"
    plan = [
        (3.5, 57, 0.30), (3.0, 59, 0.30), (3.5, 57, 0.28), (3.0, 55, 0.27),
        (3.5, 60, 0.34), (3.0, 62, 0.33),
        (4.0, 64, 0.55), (4.0, 65, 0.58), (4.0, 64, 0.56), (4.0, 67, 0.60),  # chorus-ish
        (3.5, 57, 0.30), (3.0, 59, 0.29), (3.5, 60, 0.31), (3.0, 55, 0.27),
        (4.0, 64, 0.52), (4.0, 65, 0.55),
    ]
    for dur, midi, level in plan:
        n = int(dur * sr)
        tt = np.arange(n) / sr
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        vib = 0.30 * np.sin(2 * np.pi * 5.5 * tt)            # ~0.3 st vibrato
        freq = f0 * 2 ** (vib / 12)
        phase = np.cumsum(2 * np.pi * freq / sr)
        tone = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        env = np.minimum(1, tt / 0.08) * np.minimum(1, (dur - tt) / 0.15)
        breathy = 0.01 * rng.standard_normal(n)
        t_total.append((level * env * (tone / np.max(np.abs(tone)) + breathy)).astype(np.float32))
        t_total.append(np.zeros(int(0.5 * sr), dtype=np.float32))
    sig = np.concatenate(t_total)
    core_audio.save_wav(path, sig, sr, subtype="PCM_16")


def main() -> int:
    print("== enjoi end-to-end pipeline test ==")
    print("work dir:", WORK)

    # 1. search ---------------------------------------------------------------
    print("\n[1/7] YouTube search")
    from enjoi.modules.search import search_youtube

    results = search_youtube("creative commons instrumental music 2 minutes", limit=8)
    check(len(results) >= 1, f"search returned results ({len(results)})")
    for r in results[:3]:
        print(f"    - {r['title'][:60]!r} {r['duration_sec']}s")
    # shortest result -> fastest test
    pick = min(results, key=lambda r: r.get("duration_sec") or 9e9)
    check(0 < (pick.get("duration_sec") or 0) <= 600, "picked result within duration cap")

    # 2. reference acquire + analyze -------------------------------------------
    print(f"\n[2/7] Reference acquire+analyze: {pick['title'][:60]!r}")
    project = storage.create_project("E2E test song")
    out = tasks.task_reference(project, pick["url"], progress)
    profile = out["profile"]
    fp = profile.get("fingerprints", {})
    check(project.reference_profile_path.exists(), "reference_profile.json written")
    check(60 <= profile["bpm"] <= 200, f"BPM sane ({profile['bpm']})")
    check(profile["key"].get("tonic") in list("CDEFGAB") or "#" in profile["key"].get("tonic", ""),
          f"key detected ({profile['key']})")
    check(len(profile["structure"]) >= 3, f"structure has sections ({len(profile['structure'])})")
    check(len(fp.get("melody_interval_ngrams", [])) > 0, "melody n-grams present")
    check(len(fp.get("chord_sequence", [])) > 0, "chord sequence present")
    check(len(fp.get("fp_hashes", [])) > 100, f"fp hashes present ({len(fp.get('fp_hashes', []))})")
    ref_wav = project.ref_cache_dir / "reference.wav"
    check(ref_wav.exists(), "reference audio in _ref_cache (pre-generation)")
    ref_copy = WORK / "ref_copy.wav"
    shutil.copy(ref_wav, ref_copy)

    # 3. Uniqueness Guard self-test --------------------------------------------
    print("\n[3/7] Uniqueness Guard self-test (reference vs itself MUST fail)")
    from enjoi.modules.unique import run_uniqueness_audit

    self_report = run_uniqueness_audit(profile, ref_copy)
    print("    checks:", {k: (v["value"], v["passed"]) for k, v in self_report["checks"].items()})
    check(self_report["passed"] is False, "self-audit fails (fingerprints are like-for-like)")
    check(self_report["checks"]["audio_fingerprint"]["value"] > 0
          or self_report["checks"]["melody_ngram_overlap"]["value"] > 0.25
          or self_report["checks"]["chroma_correlation"]["value"] >= 0.80,
          "at least one strong self-match signal")

    # 4. generate ---------------------------------------------------------------
    print("\n[4/7] Instrumental generation (similarity 72)")
    gen = tasks.task_generate(project, 72, progress)
    check(project.instrumental_path.exists(), "instrumental.wav written")
    check(project.grid_path.exists(), "instrumental_grid.json written")
    check(project.uniqueness_report_path.exists(), "uniqueness_report.json written")
    check(gen["report"]["passed"] is True, "generated instrumental passed originality audit")
    check(not project.ref_cache_dir.exists() or not any(project.ref_cache_dir.iterdir()),
          "_ref_cache deleted after generation (ownership guarantee)")
    grid = storage.read_json(project.grid_path)
    inst_dur = core_audio.duration_sec(project.instrumental_path)
    target = profile["duration_sec"]
    check(abs(inst_dur - target) / target <= 0.07,
          f"length matched (instrumental {inst_dur:.1f}s vs ref {target:.1f}s)")
    check(len(grid["key"].get("scale_midi", [])) == 7, "grid has scale_midi")

    # 5. vocal -------------------------------------------------------------------
    print("\n[5/7] Vocal processing + scoring + arrangement")
    fake_vocal = WORK / "take.wav"
    make_fake_vocal(fake_vocal)
    vout = tasks.task_vocal(project, fake_vocal, progress)
    analysis = vout["analysis"]
    sections = analysis["sections"]
    roles = [s["role"] for s in sections]
    check(len(analysis["phrases"]) >= 8, f"phrases segmented ({len(analysis['phrases'])})")
    check(roles.count("chorus") == 1, f"exactly one chorus ({roles})")
    chorus = next(s for s in sections if s["role"] == "chorus")
    check(chorus["impact_score"] == max(s["impact_score"] for s in sections),
          "chorus has top ImpactScore")
    check(project.arrangement_path.exists(), "arrangement.json written")
    arrangement = storage.read_json(project.arrangement_path)
    placements = arrangement["placements"]
    check(len(placements) >= 2, f"placements created ({len(placements)})")
    chorus_slots = [p for p in placements if p["role"] == "chorus"]
    check(all(0.94 <= p["stretch"] <= 1.06 for p in placements), "stretch within ±6%")
    check(all((project.dir / p["chop_file"]).exists() for p in placements), "all chop files exist")

    # 6. render: tune -> mix -> export ---------------------------------------------
    print("\n[6/7] Render (tune 35 / pop / streaming)")
    rout = tasks.task_render(project, 35, "pop", "streaming", "E2E Song", "Kevin", True, progress)
    exports = rout["exports"]
    fmts = {e["format"] for e in exports}
    check("wav" in fmts and "mp3" in fmts, f"wav+mp3 exported ({fmts})")
    for e in exports:
        fpath = project.dir / e["file"]
        check(fpath.exists() and fpath.stat().st_size > 1000, f"export exists: {e['file']}")
    song_wav = project.dir / "exports" / "song.wav"
    mix_dur = core_audio.duration_sec(song_wav)
    check(abs(mix_dur - inst_dur) < 2.0, f"master duration matches instrumental ({mix_dur:.1f}s)")
    y, _ = core_audio.load_audio(song_wav)
    check(np.isfinite(y).all() and float(np.abs(y).max()) > 0.05, "master is finite and non-silent")
    manifest = storage.read_json(project.manifest_path)
    check(manifest["reference_audio_in_output"] is False, "manifest: no reference audio in output")
    check(manifest.get("uniqueness_report", {}).get("passed") is True, "manifest embeds passing audit")

    # 7. state ------------------------------------------------------------------------
    print("\n[7/7] Project state")
    state = project.read_state()
    check(state["instrumental"]["uniqueness_passed"] is True, "state: uniqueness_passed")
    check(state["arrangement_ready"] is True, "state: arrangement_ready")
    check(len(state["exports"]) >= 2, "state: exports recorded")

    print("\n== RESULT ==")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
