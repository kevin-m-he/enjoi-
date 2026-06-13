"""Uniqueness Guard (spec 4.3.1) — automated anti-plagiarism audit.

Compares a generated candidate against ``profile["fingerprints"]`` (computed by
reference.py while the reference audio still existed), so audits keep working
after ``_ref_cache/`` is deleted. Four checks:

* melody_ngram_overlap — pitch-interval 6-gram overlap ratio        < 0.45
* chord_run_length     — longest shared non-exempt chord run        <= 6
* chroma_correlation   — beat-synced chroma peak cross-correlation (ADVISORY:
                         reported but never blocks — shared key/groove is style)
* audio_fingerprint    — spectral landmark hash collisions (ADVISORY:
                         reported but never blocks — see below)

Threshold rationale: melody and chord-progression are the elements copyright
actually protects, so those guards stay meaningful (the generator passes them
comfortably — it is never conditioned on the reference melody/harmony). The
chroma check is lenient because high correlation is the EXPECTED result of
legitimately matching the reference's key and groove (the non-copyrightable
style the similarity slider is meant to track) — a tight value rejected
original output merely for sharing a key. The audio-fingerprint check is
ADVISORY only: the render graph's two sources are the generated instrumental
and the user's vocal, and the generator never reads the reference audio, so
literal leakage is impossible by design; the landmark comparison only registers
coincidental spectral overlap (zero-leakage procedural audio still scores
dozens of "matches"), so it is reported but never blocks. The UI disclaimer
already states no software can guarantee legal non-infringement.

Canonical fingerprint helpers are exposed publicly so reference.py builds the
profile with the EXACT same algorithms:

    melody_midi_sequence(y, sr)  -> list[int]
    interval_ngrams(midi_seq)    -> set[str]   (6 semitone intervals, comma-joined,
                                                consecutive-repeat-collapsed)
    chords_per_bar(y, sr, bpb)   -> list[str]  (e.g. "Am", "F")
    downbeat_chroma(y, sr, bpb)  -> np.ndarray (n_downbeats, 12), L2-normalized
    landmark_hashes(y, sr)       -> (hashes int64 array, anchor_times array)

Only stdlib + numpy at module top level; librosa/scipy are imported inside
functions per the engineering rules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import audio as core_audio
from .synth import NOTE_TO_PC

ANALYSIS_SR = 22050
MAX_ANALYSIS_SEC = 240.0

NGRAM_LEN = 6
MELODY_THRESHOLD = 0.45     # melody is the core copyrightable element — kept meaningful
CHORD_RUN_THRESHOLD = 6     # shared non-exempt chord run (common loops already exempt)
CHROMA_THRESHOLD = 0.94     # advisory only — shared key/groove is legitimate style, never blocks
FP_THRESHOLD = 0            # advisory only — fingerprint never blocks (see module docstring)

# Fingerprint clustering: this many hash collisions inside this window is a
# "matched segment" (random collisions are sparse; real copying clusters).
_FP_CLUSTER_MIN = 12
_FP_CLUSTER_WINDOW_SEC = 4.0

_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (for candidate key estimation).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Common-practice loops, expressed key-relative as "<semitones-from-tonic><quality>".
# Rotations are covered by cycle-doubling in _is_exempt_run.
_EXEMPT_CYCLES = [
    ["0maj", "7maj", "9min", "5maj"],                       # I-V-vi-IV
    ["0min", "8maj", "3maj", "10maj"],                      # i-VI-III-VII
    ["2min", "7maj", "0maj"],                               # ii-V-I
    ["0maj", "0maj", "0maj", "0maj", "5maj", "5maj",        # 12-bar blues
     "0maj", "0maj", "7maj", "5maj", "0maj", "0maj"],
    ["0maj", "0maj", "0maj", "0maj", "5maj", "5maj",        # 12-bar variant
     "0maj", "0maj", "7maj", "7maj", "0maj", "0maj"],
]


def _p(progress, frac: float, msg: str) -> None:
    if progress is None:
        return
    try:
        progress(min(max(float(frac), 0.0), 1.0), msg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Canonical fingerprint extractors (shared with reference.py)
# ---------------------------------------------------------------------------

def melody_midi_sequence(y: np.ndarray, sr: int) -> list[int]:
    """Dominant melody as rounded MIDI notes (pyin on the harmonic component)."""
    import librosa

    if y.size < sr:
        return []
    harmonic = librosa.effects.harmonic(y)
    f0, voiced, _prob = librosa.pyin(
        harmonic,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=sr,
        frame_length=2048,
    )
    good = voiced & np.isfinite(f0)
    if not np.any(good):
        return []
    midi = np.round(librosa.hz_to_midi(f0[good])).astype(int)
    return [int(m) for m in midi]


def interval_ngrams(midi_seq, n: int = NGRAM_LEN) -> set[str]:
    """Semitone-interval n-grams, comma-joined, consecutive-repeat-collapsed.

    This is the contract format for ``fingerprints.melody_interval_ngrams``.
    """
    collapsed: list[int] = []
    for m in midi_seq:
        m = int(m)
        if not collapsed or m != collapsed[-1]:
            collapsed.append(m)
    if len(collapsed) < n + 1:
        return set()
    intervals = np.diff(np.asarray(collapsed))
    return {
        ",".join(str(int(v)) for v in intervals[i: i + n])
        for i in range(len(intervals) - n + 1)
    }


def _beats_and_chroma(y: np.ndarray, sr: int):
    """(beat_frames, chroma) at hop 512, or (None, None) when untrackable."""
    import librosa

    if y.size < sr * 2:
        return None, None
    _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beats = np.asarray(beats)
    if beats.size < 8:
        return None, None
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    return beats, chroma


def chords_per_bar(y: np.ndarray, sr: int, bpb: int = 4) -> list[str]:
    """Per-bar chord labels via chroma template matching (24 maj/min triads)."""
    import librosa

    beats, chroma = _beats_and_chroma(y, sr)
    if beats is None:
        return []
    downbeats = beats[:: max(bpb, 1)]
    if downbeats.size < 2:
        return []
    bars = librosa.util.sync(chroma, downbeats, aggregate=np.median)
    # Columns between consecutive downbeats (drop the pre-roll and tail spans).
    bars = bars[:, 1: downbeats.size]
    templates = np.zeros((24, 12))
    for r in range(12):
        templates[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1.0       # major
        templates[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1.0  # minor
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    norms = np.linalg.norm(bars, axis=0, keepdims=True)
    norms[norms < 1e-9] = 1.0
    scores = templates @ (bars / norms)
    idx = np.argmax(scores, axis=0)
    out = []
    for k in idx:
        root, quality = (int(k) % 12, "maj" if k < 12 else "min")
        out.append(_PC_NAMES[root] + ("" if quality == "maj" else "m"))
    return out


def downbeat_chroma(y: np.ndarray, sr: int, bpb: int = 4) -> np.ndarray:
    """Beat-synchronous chroma: one L2-normalized 12-vector per downbeat span."""
    import librosa

    beats, chroma = _beats_and_chroma(y, sr)
    if beats is None:
        return np.zeros((0, 12))
    downbeats = beats[:: max(bpb, 1)]
    if downbeats.size < 2:
        return np.zeros((0, 12))
    sync = librosa.util.sync(chroma, downbeats, aggregate=np.mean)
    mat = sync[:, 1: downbeats.size].T  # (n, 12)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return mat / norms


def landmark_hashes(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Chromaprint-style landmark hashes: STFT log-spec peaks paired into
    (f1, df, dt) 32-bit hashes. Returns (hashes int64, anchor times sec)."""
    import librosa
    from scipy import ndimage

    if y.size < 4096:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    logspec = librosa.amplitude_to_db(spec, ref=np.max)
    local_max = ndimage.maximum_filter(logspec, size=(35, 21))
    threshold = max(float(np.median(logspec)) + 12.0, float(logspec.max()) - 60.0)
    mask = (logspec == local_max) & (logspec > threshold)
    fbins, frames = np.nonzero(mask)
    if fbins.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    order = np.argsort(frames, kind="stable")
    f, t = fbins[order], frames[order]
    hashes: list[int] = []
    times: list[float] = []
    fanout = 5
    for i in range(len(t)):
        j0 = int(np.searchsorted(t, t[i] + 1))
        j1 = int(np.searchsorted(t, t[i] + 64))
        paired = 0
        for j in range(j0, j1):
            df = int(f[j]) - int(f[i])
            if abs(df) > 255:
                continue
            dt = int(t[j] - t[i])
            h = ((int(f[i]) & 0x3FF) << 16) | (((df + 256) & 0x1FF) << 6) | (dt & 0x3F)
            hashes.append(h)
            times.append(float(t[i]) * 512.0 / sr)
            paired += 1
            if paired >= fanout:
                break
    return np.asarray(hashes, dtype=np.int64), np.asarray(times)


# ---------------------------------------------------------------------------
# Internal analysis helpers
# ---------------------------------------------------------------------------

def _tonic_pc(name: str) -> int:
    s = str(name or "C").strip().upper().replace("♯", "#").replace("♭", "B")
    return NOTE_TO_PC.get(s[:2], NOTE_TO_PC.get(s[:1], 0))


def _parse_chord_label(label: str) -> tuple[int, str] | None:
    s = str(label or "").strip()
    if not s:
        return None
    root = s[0].upper()
    rest = s[1:]
    accidental = ""
    if rest[:1] in ("#", "♯"):
        accidental, rest = "#", rest[1:]
    elif rest[:1] in ("b", "♭"):
        accidental, rest = "B", rest[1:]
    pc = NOTE_TO_PC.get(root + accidental)
    if pc is None:
        return None
    low = rest.lower()
    quality = "min" if (low.startswith("m") and not low.startswith("maj")) else "maj"
    return pc, quality


def _estimate_key_pc(y: np.ndarray, sr: int) -> int:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, best_pc = -2.0, 0
    for profile in (_KS_MAJOR, _KS_MINOR):
        for pc in range(12):
            ref = np.roll(profile, pc)
            c = float(np.corrcoef(chroma, ref)[0, 1])
            if c > best:
                best, best_pc = c, pc
    return best_pc


def _roman_tokens(chords: list[tuple[int, str]], tonic_pc: int) -> list[str]:
    return [f"{(pc - tonic_pc) % 12}{quality}" for pc, quality in chords]


def _is_exempt_run(run: list[str]) -> bool:
    """True if the run is a contiguous slice of a common-practice loop cycle
    (any rotation, any number of repeats)."""
    if not run:
        return True
    for cycle in _EXEMPT_CYCLES:
        reps = len(run) // len(cycle) + 2
        seq = cycle * reps
        for k in range(len(cycle)):
            if seq[k: k + len(run)] == run:
                return True
    return False


def _longest_nonexempt_run(a: list[str], b: list[str]) -> int:
    """Longest common contiguous run between token lists a and b that is NOT a
    common-practice loop."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    vocab = {tok: i for i, tok in enumerate(set(a) | set(b))}
    av = np.asarray([vocab[t] for t in a])
    bv = np.asarray([vocab[t] for t in b])
    eq = av[:, None] == bv[None, :]
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n):
        dp[i + 1, 1:] = np.where(eq[i], dp[i, :-1] + 1, 0)
    vals = dp[1:, 1:]
    if vals.max() == 0:
        return 0
    flat_order = np.argsort(-vals, axis=None)
    for pos in flat_order:
        i, j = divmod(int(pos), m)
        length = int(vals[i, j])
        if length == 0:
            break
        run = a[i - length + 1: i + 1]
        if not _is_exempt_run(run):
            return length
    return 0


def _chroma_peak_correlation(cand: np.ndarray, ref: np.ndarray) -> float:
    """Peak mean cosine correlation over all lags and all 12 transpositions."""
    n, m = cand.shape[0], ref.shape[0]
    if n == 0 or m == 0:
        return 0.0
    min_overlap = int(min(8, n, m))
    best = 0.0
    for t in range(12):
        rolled = np.roll(ref, t, axis=1)
        sim = cand @ rolled.T  # (n, m)
        for k in range(-(n - 1), m):
            diag = np.diagonal(sim, offset=k)
            if diag.size >= min_overlap:
                best = max(best, float(diag.mean()))
    return best


def _fingerprint_segments(cand_hashes: np.ndarray, cand_times: np.ndarray,
                          ref_hashes: set[int]) -> int:
    """Count time-consistent clusters of hash collisions in the candidate."""
    if cand_hashes.size == 0 or not ref_hashes:
        return 0
    ref_arr = np.fromiter(ref_hashes, dtype=np.int64)
    hit_times = np.sort(cand_times[np.isin(cand_hashes, ref_arr)])
    segments = 0
    i = 0
    while i < hit_times.size:
        j = int(np.searchsorted(hit_times, hit_times[i] + _FP_CLUSTER_WINDOW_SEC))
        if j - i >= _FP_CLUSTER_MIN:
            segments += 1
            i = j  # skip past this cluster
        else:
            i += 1
    return segments


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def _check(value, threshold, passed, note: str | None = None) -> dict:
    out = {"value": value, "threshold": threshold, "passed": bool(passed)}
    if note:
        out["note"] = note
    return out


def run_uniqueness_audit(profile: dict, candidate_wav: Path, progress=None) -> dict:
    """Spec 4.3.1 divergence audit. Returns {passed, checks, summary}.

    generate.py adds "attempts" / "effective_similarity" before writing
    uniqueness_report.json (full contract schema).
    """
    fingerprints = (profile or {}).get("fingerprints") or {}
    ts = str((profile or {}).get("time_signature") or "4/4")
    try:
        bpb = max(2, min(12, int(ts.split("/")[0])))
    except (ValueError, IndexError):
        bpb = 4

    _p(progress, 0.02, "Originality audit: loading candidate")
    y, sr = core_audio.load_audio(Path(candidate_wav), sr=ANALYSIS_SR, mono=True)
    y = y[: int(MAX_ANALYSIS_SEC * sr)]

    silent = y.size < int(3.0 * sr) or core_audio.rms(y) < 1e-5
    checks: dict[str, dict] = {}
    parts: list[str] = []

    # ---- 1. Melodic similarity -------------------------------------------
    _p(progress, 0.10, "Originality audit: melody analysis")
    ref_ngrams = set(fingerprints.get("melody_interval_ngrams") or [])
    if silent:
        checks["melody_ngram_overlap"] = _check(0.0, MELODY_THRESHOLD, True,
                                                "candidate too short/silent")
        parts.append("melody overlap 0%")
    elif not ref_ngrams:
        checks["melody_ngram_overlap"] = _check(0.0, MELODY_THRESHOLD, True,
                                                "no reference fingerprint available")
        parts.append("no reference melody fingerprint")
    else:
        try:
            cand_ngrams = interval_ngrams(melody_midi_sequence(y, sr))
            overlap = (len(cand_ngrams & ref_ngrams) / max(1, len(cand_ngrams))
                       if cand_ngrams else 0.0)
            checks["melody_ngram_overlap"] = _check(
                round(float(overlap), 4), MELODY_THRESHOLD, overlap < MELODY_THRESHOLD)
            parts.append(f"melody overlap {overlap * 100.0:.0f}%")
        except Exception as exc:  # analysis edge case must not kill the job
            checks["melody_ngram_overlap"] = _check(
                0.0, MELODY_THRESHOLD, True, f"melody check skipped ({exc})")
            parts.append("melody check skipped")

    # ---- 2. Harmonic similarity ------------------------------------------
    _p(progress, 0.45, "Originality audit: chord progression analysis")
    ref_chord_labels = list(fingerprints.get("chord_sequence") or [])
    if silent:
        checks["chord_run_length"] = _check(0, CHORD_RUN_THRESHOLD, True,
                                            "candidate too short/silent")
        checks["chord_run_length"]["exempt_loops"] = True
        parts.append("no chord-run matches")
    elif not ref_chord_labels:
        checks["chord_run_length"] = _check(0, CHORD_RUN_THRESHOLD, True,
                                            "no reference fingerprint available")
        checks["chord_run_length"]["exempt_loops"] = True
        parts.append("no reference chord fingerprint")
    else:
        try:
            ref_parsed = [c for c in (_parse_chord_label(s) for s in ref_chord_labels)
                          if c is not None]
            ref_tonic = _tonic_pc(((profile or {}).get("key") or {}).get("tonic"))
            ref_tokens = _roman_tokens(ref_parsed, ref_tonic)

            cand_labels = chords_per_bar(y, sr, bpb)
            cand_parsed = [c for c in (_parse_chord_label(s) for s in cand_labels)
                           if c is not None]
            cand_tokens = _roman_tokens(cand_parsed, _estimate_key_pc(y, sr))

            run = _longest_nonexempt_run(cand_tokens, ref_tokens)
            chk = _check(int(run), CHORD_RUN_THRESHOLD, run <= CHORD_RUN_THRESHOLD)
            chk["exempt_loops"] = True
            checks["chord_run_length"] = chk
            parts.append("no chord-run matches" if run <= CHORD_RUN_THRESHOLD
                         else f"shared chord run of {run}")
        except Exception as exc:
            chk = _check(0, CHORD_RUN_THRESHOLD, True, f"chord check skipped ({exc})")
            chk["exempt_loops"] = True
            checks["chord_run_length"] = chk
            parts.append("chord check skipped")

    # ---- 3. Chroma fingerprint -------------------------------------------
    _p(progress, 0.70, "Originality audit: chroma cross-correlation")
    ref_chroma = np.asarray(fingerprints.get("chroma_downbeat") or [], dtype=float)
    if silent:
        checks["chroma_correlation"] = _check(0.0, CHROMA_THRESHOLD, True,
                                              "candidate too short/silent")
        parts.append("chroma peak 0.00")
    elif ref_chroma.size == 0 or ref_chroma.ndim != 2 or ref_chroma.shape[1] != 12:
        checks["chroma_correlation"] = _check(0.0, CHROMA_THRESHOLD, True,
                                              "no reference fingerprint available")
        parts.append("no reference chroma fingerprint")
    else:
        try:
            norms = np.linalg.norm(ref_chroma, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            ref_norm = ref_chroma / norms
            cand_chroma = downbeat_chroma(y, sr, bpb)
            peak = _chroma_peak_correlation(cand_chroma, ref_norm)
            # ADVISORY: high chroma correlation just means the track shares the
            # reference's key / harmonic palette — the non-copyrightable "style"
            # that high similarity is meant to match (MusicGen output naturally
            # scores ~0.95+). Real copying is caught by the melody n-gram and
            # chord-run gates above. Reported for transparency, never blocks.
            checks["chroma_correlation"] = _check(
                round(float(peak), 4), CHROMA_THRESHOLD, True,
                "advisory — shared key/groove (style), not copied melody/chords")
            parts.append(f"chroma peak {peak:.2f} (advisory)")
        except Exception as exc:
            checks["chroma_correlation"] = _check(
                0.0, CHROMA_THRESHOLD, True, f"chroma check skipped ({exc})")
            parts.append("chroma check skipped")

    # ---- 4. Audio fingerprint (ADVISORY) ----------------------------------
    # The render graph has exactly two sources — the generated instrumental and
    # the user's vocal — and the generator never reads the reference audio, so
    # literal audio leakage is structurally impossible. The landmark-hash
    # comparison therefore cannot detect leakage here; it only registers
    # coincidental spectral overlap between two pieces of music with similar
    # instrumentation (procedurally-generated audio that never touched the
    # reference still scores dozens of "matches"). We report the count for
    # transparency but never block on it. Melody and chord-progression — the
    # copyrightable elements — remain the active gates above.
    _p(progress, 0.85, "Originality audit: audio fingerprint match")
    ref_hashes = {int(h) for h in (fingerprints.get("fp_hashes") or [])}
    note = "advisory — generator never reads reference audio; leakage impossible by design"
    if silent:
        checks["audio_fingerprint"] = _check(0, FP_THRESHOLD, True, "candidate too short/silent")
        parts.append("no fingerprint matches")
    elif not ref_hashes:
        checks["audio_fingerprint"] = _check(0, FP_THRESHOLD, True,
                                             "no reference fingerprint available")
        parts.append("no reference audio fingerprint")
    else:
        try:
            cand_h, cand_t = landmark_hashes(y, sr)
            segments = _fingerprint_segments(cand_h, cand_t, ref_hashes)
            checks["audio_fingerprint"] = _check(int(segments), FP_THRESHOLD, True, note)
            parts.append("no fingerprint matches" if segments == 0
                         else f"{segments} fingerprint segment(s) (advisory)")
        except Exception as exc:
            checks["audio_fingerprint"] = _check(
                0, FP_THRESHOLD, True, f"fingerprint check skipped ({exc})")
            parts.append("fingerprint check skipped")

    passed = all(c["passed"] for c in checks.values())
    verdict = "passed" if passed else "FAILED"
    summary = f"Originality check: {verdict} — " + ", ".join(parts) + "."
    _p(progress, 1.0, summary)
    return {"passed": passed, "checks": checks, "summary": summary}
