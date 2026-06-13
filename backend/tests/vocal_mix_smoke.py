"""Offline smoke test for the vocal-production chain (vocal->score->arrange->
tune->mix->export-render), using ONLY numpy-synthesized audio — no network,
no heavy/optional models required (runs on requirements-core.txt).

Proves the three product goals:
  1. GAIN-STAGING: against a LOUD instrumental the lead vocal still sits on top
     (vocal-stem RMS is in a lead-forward window vs instrumental-stem RMS), and
     the master still hits the streaming LUFS target.
  2. CHOP BY MUSICAL SENSE: the chorus section repeats across every chorus slot.
  3. AUTOTUNE @ 432 Hz: a held tuned note lands on a 432-based scale frequency
     (within ~20 cents) and is clearly NOT the 440-based equivalent.

Run from backend/ with the venv python:
    .venv\\Scripts\\python.exe tests\\vocal_mix_smoke.py
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="enjoi_vmsmoke_"))
os.environ["ENJOI_DATA_DIR"] = str(WORK / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from enjoi.core import audio as core_audio  # noqa: E402
from enjoi.core import config, storage  # noqa: E402
from enjoi.modules.arrange import build_arrangement  # noqa: E402
from enjoi.modules.mix import mix_and_master  # noqa: E402
from enjoi.modules.score import score_sections  # noqa: E402
from enjoi.modules.tune import tune_vocals  # noqa: E402
from enjoi.modules.vocal import process_vocal  # noqa: E402

SR = config.SAMPLE_RATE
FAILURES: list[str] = []

# A-minor scale (matches the grid we synthesize below).
A_MINOR_MIDI = [57, 59, 60, 62, 64, 65, 67]  # A B C D E F G


def check(cond: bool, label: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + label)
    if not cond:
        FAILURES.append(label)


def progress(frac: float, msg: str) -> None:
    pass  # quiet — this is a smoke test


# ---------------------------------------------------------------------------
# synthetic audio
# ---------------------------------------------------------------------------

def make_fake_vocal(path: Path, sr: int = SR) -> None:
    """~64 s synthetic one-take: vowel tones w/ vibrato, a distinct louder/higher
    'chorus' block, breaths + silences between phrases (mirrors e2e_pipeline)."""
    rng = np.random.default_rng(7)
    parts = []
    # (duration, midi, level); the loud/high block in the middle is the chorus.
    plan = [
        (3.5, 57, 0.30), (3.0, 59, 0.30), (3.5, 57, 0.28), (3.0, 55, 0.27),
        (3.5, 60, 0.34), (3.0, 62, 0.33),
        (4.0, 64, 0.55), (4.0, 65, 0.58), (4.0, 64, 0.56), (4.0, 67, 0.60),  # chorus
        (3.5, 57, 0.30), (3.0, 59, 0.29), (3.5, 60, 0.31), (3.0, 55, 0.27),
        (4.0, 64, 0.52), (4.0, 65, 0.55),
    ]
    for dur, midi, level in plan:
        n = int(dur * sr)
        tt = np.arange(n) / sr
        f0 = 440.0 * 2 ** ((midi - 69) / 12)        # sung "naturally" at 440
        vib = 0.30 * np.sin(2 * np.pi * 5.5 * tt)    # ~0.3 st vibrato
        freq = f0 * 2 ** (vib / 12)
        phase = np.cumsum(2 * np.pi * freq / sr)
        tone = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        env = np.minimum(1, tt / 0.08) * np.minimum(1, (dur - tt) / 0.15)
        breathy = 0.01 * rng.standard_normal(n)
        parts.append((level * env * (tone / np.max(np.abs(tone)) + breathy)).astype(np.float32))
        parts.append(np.zeros(int(0.5 * sr), dtype=np.float32))
    core_audio.save_wav(path, np.concatenate(parts), sr, subtype="PCM_16")


def make_fake_instrumental(project, sr: int = SR) -> dict:
    """LOUD A-minor instrumental at the 432 Hz reference + matching grid with
    several sections (>=2 chorus). Returns the grid dict (also written to disk)."""
    bpm = 96.0
    beat = 60.0 / bpm
    bar = 4 * beat
    # Section layout (bars): intro, verse, chorus, verse, chorus, outro.
    layout = [("intro", 4), ("verse", 8), ("chorus", 8),
              ("verse", 8), ("chorus", 8), ("outro", 4)]
    sections = []
    downbeats = []
    t = 0.0
    for label, bars in layout:
        sections.append({"label": label, "start": round(t, 3),
                         "end": round(t + bars * bar, 3), "bars": bars})
        for b in range(bars):
            downbeats.append(round(t + b * bar, 3))
        t += bars * bar
    duration = t
    beat_times = [round(b * beat, 3) for b in range(int(duration / beat) + 1)]

    # --- synthesize a dense, LOUD backing track tuned to 432 ---------------
    n = int(duration * sr)
    tt = np.arange(n) / sr
    mix = np.zeros(n, dtype=np.float64)
    # sustained A-minor triad pad (root A2, C4, E4) at 432 reference
    for midi in (45, 60, 64):
        f = config.midi_to_hz(midi)  # 432-based
        mix += 0.5 * np.sin(2 * np.pi * f * tt)
    # four-on-the-floor kick-ish thumps on every beat (broadband transients)
    rng = np.random.default_rng(3)
    for bt in beat_times:
        i0 = int(bt * sr)
        i1 = min(i0 + int(0.12 * sr), n)
        if i1 <= i0:
            continue
        env = np.exp(-np.linspace(0, 12, i1 - i0))
        mix[i0:i1] += 1.2 * env * np.sin(2 * np.pi * 60 * np.arange(i1 - i0) / sr)
        mix[i0:i1] += 0.3 * env * rng.standard_normal(i1 - i0)
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.97  # hot, near full-scale
    core_audio.save_wav(project.instrumental_path, mix.astype(np.float32), sr, subtype="PCM_24")

    grid = {
        "bpm": bpm, "time_signature": "4/4",
        "key": {"tonic": "A", "mode": "minor", "scale_midi": A_MINOR_MIDI},
        "beat_times": beat_times, "downbeats": downbeats,
        "sections": sections, "duration_sec": round(duration, 3),
        "engine": "procedural",
    }
    storage.write_json(project.grid_path, grid)
    return grid


# ---------------------------------------------------------------------------
# pitch measurement
# ---------------------------------------------------------------------------

def dominant_hz(y: np.ndarray, sr: int = SR) -> float:
    """FFT peak frequency of the loudest sustained window of a mono signal."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.size < sr // 2:
        win = y
    else:
        # pick the 0.7 s window with the most energy (a held note)
        w = int(0.7 * sr)
        hop = int(0.1 * sr)
        best_e, best_i = -1.0, 0
        for i in range(0, len(y) - w, hop):
            e = float(np.sum(y[i:i + w] ** 2))
            if e > best_e:
                best_e, best_i = e, i
        win = y[best_i:best_i + w]
    win = win * np.hanning(len(win))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    band = (freqs > 60) & (freqs < 1200)
    spec = np.where(band, spec, 0.0)
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:  # parabolic interpolation for sub-bin accuracy
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return float((k + delta) * sr / len(win))
    return float(freqs[k])


def cents_to(freq: float, target: float) -> float:
    return 1200.0 * math.log2(freq / target)


def nearest_scale_cents(freq: float, tuning: float) -> tuple[float, int]:
    """Cents from `freq` to the nearest A-minor scale note at the given tuning."""
    best_c, best_m = 1e9, 0
    for octave in range(2, 7):
        for pc in A_MINOR_MIDI:
            m = pc + 12 * (octave - 4)
            f = config.midi_to_hz(m, tuning)
            c = cents_to(freq, f)
            if abs(c) < abs(best_c):
                best_c, best_m = c, m
    return best_c, best_m


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_chain(retune_speed: int, label: str):
    project = storage.create_project(f"smoke {label}")
    grid = make_fake_instrumental(project)
    make_fake_vocal(WORK / f"take_{label}.wav")
    analysis = process_vocal(project, WORK / f"take_{label}.wav", progress)
    analysis = score_sections(analysis)
    storage.write_json(project.vocal_analysis_path, analysis)
    arrangement = build_arrangement(project, grid, analysis, progress)
    arrangement = tune_vocals(project, arrangement, grid, retune_speed, progress)
    storage.write_json(project.arrangement_path, arrangement)
    master = mix_and_master(project, arrangement, grid, "pop", "streaming", progress)
    return project, grid, analysis, arrangement, master


def main() -> int:
    print("== enjoi vocal+mix smoke test ==")
    print("work dir:", WORK)

    # === main run: default-ish retune 35 ===================================
    print("\n[1/5] Full chain (retune 35, pop, streaming)")
    project, grid, analysis, arrangement, master = run_chain(35, "main")
    placements = arrangement["placements"]
    check(len(placements) >= 2, f"placements created ({len(placements)})")

    # --- GOAL 2: chorus repeats across every chorus slot -------------------
    print("\n[2/5] Chop-by-sense: chorus repetition")
    grid_chorus_slots = sum(1 for s in grid["sections"] if s["label"] == "chorus")
    chorus_pl = [p for p in placements if p["role"] == "chorus"]
    chorus_secs = {p["section_id"] for p in chorus_pl}
    check(grid_chorus_slots >= 2, f"grid has >=2 chorus slots ({grid_chorus_slots})")
    check(len(chorus_pl) == grid_chorus_slots,
          f"chorus placed in every chorus slot ({len(chorus_pl)}/{grid_chorus_slots})")
    check(len(chorus_secs) == 1,
          f"same chorus section repeated (section ids {chorus_secs})")
    verse_pl = [p for p in placements if p["role"] == "verse"]
    check(len(verse_pl) >= 1, f"verse material placed ({len(verse_pl)})")
    check(all(0.94 <= p["stretch"] <= 1.06 for p in placements), "stretch within +/-6%")

    # --- every placement has a tuned file that exists ----------------------
    print("\n[3/5] Tuned files present")
    check(all(p.get("tuned_file") for p in placements), "every placement has tuned_file")
    check(all((project.dir / p["tuned_file"]).exists() for p in placements),
          "every tuned_file exists on disk")

    # --- GOAL 3: a held tuned note is 432-referenced -----------------------
    # At the default retune (35) the snap is intentionally PARTIAL — the note is
    # only pulled ~35% toward target — so we assert it moved AWAY from the raw
    # 440-scale pitch toward the 432 reference. The strict "snaps to 432, not
    # 440" proof is on the hard-tune (100) run in [extra], where the snap is
    # essentially complete.
    print("\n[4/5] Autotune @ 432 Hz (default retune 35 = partial pull)")
    sample = next(p for p in chorus_pl if not p.get("bridge_fx"))
    raw_y, _ = core_audio.load_audio(project.dir / sample["chop_file"], mono=True)
    ty, _ = core_audio.load_audio(project.dir / sample["tuned_file"], mono=True)
    f_raw = dominant_hz(raw_y)
    f_meas = dominant_hz(ty)
    raw_c432, _ = nearest_scale_cents(f_raw, config.TUNING_HZ)
    raw_c440, _ = nearest_scale_cents(f_raw, config.STANDARD_TUNING_HZ)
    c432, m432 = nearest_scale_cents(f_meas, config.TUNING_HZ)
    c440, _ = nearest_scale_cents(f_meas, config.STANDARD_TUNING_HZ)
    print(f"    raw held pitch:   {f_raw:.2f} Hz  (432:{raw_c432:+.1f}c  440:{raw_c440:+.1f}c)")
    print(f"    tuned held pitch: {f_meas:.2f} Hz  (432:{c432:+.1f}c  440:{c440:+.1f}c)")
    # raw sits on a 440 scale note; tuning must pull it toward 432 (|cents to 432|
    # shrinks, |cents to 440| grows).
    check(abs(c432) < abs(raw_c432) - 3.0,
          f"tuning pulled the note TOWARD 432 ({abs(raw_c432):.1f}c -> {abs(c432):.1f}c)")
    check(abs(c440) > abs(raw_c440) + 3.0,
          f"...and AWAY from 440 ({abs(raw_c440):.1f}c -> {abs(c440):.1f}c)")

    # --- GOAL 1: gain-staging — vocal sits ON TOP of a LOUD instrumental ----
    print("\n[5/5] Gain-staging: vocal on top + master LUFS")
    inst_stem, _ = core_audio.load_audio(project.exports_dir / "_stem_instrumental.wav", mono=False)
    voc_stem, _ = core_audio.load_audio(project.exports_dir / "_stem_vocals.wav", mono=False)

    def active_rms_db(x: np.ndarray) -> float:
        n = x.shape[-1]
        mono = np.mean(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
        hop = 512
        nf = max(len(mono) // hop, 1)
        fr = mono[: nf * hop].reshape(nf, hop)
        r = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-20)
        rdb = 20 * np.log10(r + 1e-12)
        gate = max(float(rdb.max()) - 25.0, -55.0)
        frame_mask = rdb >= gate
        samp_mask = np.repeat(frame_mask, hop)
        if samp_mask.size < n:
            samp_mask = np.pad(samp_mask, (0, n - samp_mask.size))
        else:
            samp_mask = samp_mask[:n]
        sel = x[:, samp_mask] if x.ndim == 2 else x[samp_mask]
        return core_audio.lin_to_db(core_audio.rms(sel))

    inst_db = active_rms_db(inst_stem)
    voc_db = active_rms_db(voc_stem)
    delta = voc_db - inst_db
    print(f"    instrumental-stem active RMS: {inst_db:.2f} dBFS")
    print(f"    vocal-stem        active RMS: {voc_db:.2f} dBFS")
    print(f"    vocal - instrumental: {delta:+.2f} dB (lead-forward window: not < -6, ideally +2..+4)")
    check(delta > -6.0, f"vocal NOT drowned (>= -6 dB vs instrumental); got {delta:+.2f} dB")
    check(delta >= 0.0, f"vocal sits AT/ABOVE the instrumental; got {delta:+.2f} dB")

    meta = storage.read_json(project.exports_dir / "_master_meta.json")
    y, _ = core_audio.load_audio(project.exports_dir / "_master_tmp.wav")
    peak_db = core_audio.lin_to_db(float(np.max(np.abs(y))))
    print(f"    master LUFS={meta['lufs']}  true_peak={meta['true_peak_db']} dB  peak={peak_db:.2f} dBFS")
    check(np.isfinite(y).all(), "master is finite (no NaN/inf)")
    check(float(np.abs(y).max()) > 0.05, "master is non-silent")
    check(peak_db <= 0.01, "master <= 0 dBFS (clip-guarded)")
    check(meta["lufs"] is not None and abs(meta["lufs"] - (-14.0)) <= 1.5,
          f"master LUFS ~ streaming target -14 ({meta['lufs']})")

    # === retune extremes: 0 ~ passthrough, 100 hard-snap + 432 =============
    print("\n[extra] Retune extremes")
    proj0, _, _, arr0, _ = run_chain(0, "soft")
    p0 = next(p for p in arr0["placements"] if p["role"] == "chorus" and not p.get("bridge_fx"))
    raw, _ = core_audio.load_audio(proj0.dir / p0["chop_file"], mono=True)
    t0, _ = core_audio.load_audio(proj0.dir / p0["tuned_file"], mono=True)
    m = min(len(raw), len(t0))
    diff = float(np.sqrt(np.mean((raw[:m] - t0[:m]) ** 2)))
    rawrms = float(np.sqrt(np.mean(raw[:m] ** 2)) + 1e-9)
    print(f"    retune=0 residual vs raw: {diff / rawrms * 100:.2f}% RMS")
    check(diff / rawrms < 0.20, "retune=0 is ~passthrough (tiny residual vs raw chop)")

    proj100, _, _, arr100, _ = run_chain(100, "hard")
    p100 = next(p for p in arr100["placements"] if p["role"] == "chorus" and not p.get("bridge_fx"))
    raw100, _ = core_audio.load_audio(proj100.dir / p100["chop_file"], mono=True)
    th, _ = core_audio.load_audio(proj100.dir / p100["tuned_file"], mono=True)
    check(len(th) == len(raw100), "hard-tune preserves length exactly (no drift)")
    check(np.isfinite(th).all(), "hard-tune output finite (no NaN/inf)")
    fh = dominant_hz(th)
    ch432, _ = nearest_scale_cents(fh, config.TUNING_HZ)
    ch440, _ = nearest_scale_cents(fh, config.STANDARD_TUNING_HZ)
    print(f"    retune=100 held pitch {fh:.2f} Hz -> 432 scale {ch432:+.1f}c (440 {abs(ch440):.1f}c)")
    check(abs(ch432) <= 20.0, f"hard-tune note within 20c of 432 scale ({ch432:+.1f}c)")
    check(abs(ch432) + 10.0 < abs(ch440), "hard-tune clearly 432-referenced, not 440")

    # vibrato/drift flatten: measure INTRA-NOTE pitch wobble over the single
    # loudest sustained window (one held note), so note-to-note jumps don't
    # pollute the measurement. Hard-tune should collapse the wobble.
    def held_window(y: np.ndarray) -> np.ndarray:
        w, hop = int(0.7 * SR), int(0.1 * SR)
        if len(y) <= w:
            return y
        best_e, best_i = -1.0, 0
        for i in range(0, len(y) - w, hop):
            e = float(np.sum(y[i:i + w] ** 2))
            if e > best_e:
                best_e, best_i = e, i
        return y[best_i:best_i + w]

    def intranote_wobble_cents(y: np.ndarray) -> float:
        import librosa
        f0, _vf, _vp = librosa.pyin(y.astype(np.float32), fmin=80.0, fmax=800.0,
                                    sr=SR, frame_length=2048, hop_length=256)
        v = f0[np.isfinite(f0)]
        if v.size < 8:
            return 0.0
        # zero-mean in cents around the note median; std captures vibrato + drift
        return float(np.std(1200.0 * np.log2(v / np.median(v))))

    raw_w = intranote_wobble_cents(held_window(raw100))
    hard_w = intranote_wobble_cents(held_window(th))
    print(f"    intra-note wobble (std): raw {raw_w:.1f}c -> hard-tuned {hard_w:.1f}c")
    check(hard_w < raw_w * 0.7 + 2.0,
          f"hard-tune flattens vibrato/drift ({raw_w:.1f}c -> {hard_w:.1f}c)")

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
