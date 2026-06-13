# enjoi享受 — Internal API & Module Contract (v1)

This document is the **single source of truth** for how the frontend, the FastAPI
spine, and the pipeline modules talk to each other. All code MUST conform to it.

## Engineering rules (apply to every backend module)

1. **No heavy imports at module top level.** `librosa`, `torch`, `audiocraft`,
   `faster_whisper`, `demucs`, `crepe`, `pedalboard`, `yt_dlp` etc. must be imported
   *inside* the functions that use them, or via `enjoi.core.deps.optional_import`.
   Top level may import only stdlib, `numpy`, and `enjoi.core.*`.
2. **Graceful degradation.** Every optional dependency has a documented fallback:
   - madmom/Essentia → librosa equivalents (librosa is the required baseline).
   - audiocraft/MusicGen → built-in procedural synthesis engine (`modules/synth.py`).
   - faster-whisper → energy-only segmentation, lyrics = "" (still works).
   - pyrubberband/rubberband CLI → `librosa.effects.time_stretch` / `pitch_shift`.
   - pedalboard → scipy-based minimal chain (HPF, compressor approximation, limiter).
   - crepe/torch → `librosa.pyin` for f0.
   The app must run end-to-end with ONLY `requirements-core.txt` installed.
3. **Progress callbacks.** Long functions accept `progress: Callable[[float, str], None]`
   (fraction 0..1, human message) and call it at sensible points. Never raise from it.
4. **Paths are `pathlib.Path`**, UTF-8 JSON via `enjoi.core.storage.write_json/read_json`.
5. **Audio I/O** only through `enjoi.core.audio` helpers (44.1 kHz float32 numpy).
6. **The reference cache (`_ref_cache/`) is the ONLY place reference audio may exist.**
   `modules/export.py` and `modules/mix.py` MUST NOT read from it. It is deleted at the
   end of the generation task (after the Uniqueness Guard ran).
7. Windows-first: no POSIX-only calls, no shelling out except via resolved
   `enjoi.core.deps.ffmpeg_path()`.
8. Raise `enjoi.core.errors.PipelineError("user-readable message")` for expected
   failures; they surface in the job's `error` field.

## Storage layout (per project)

`%APPDATA%/enjoi/projects/<project-id>/`
```
project.json               # project state (see below)
reference_profile.json     # analysis output (descriptors only, kept)
_ref_cache/                # reference audio sandbox (auto-deleted)
instrumental.wav           # 44.1 kHz / 24-bit
instrumental_grid.json
uniqueness_report.json
vocal_raw.wav              # resampled upload
vocal_analysis.json
vocal_chops/               # per-placement chop wavs
vocal_tuned/               # tuned chop wavs
arrangement.json
exports/                   # final .wav/.mp3 + stems + song_manifest.json
```

## REST API (FastAPI, `http://127.0.0.1:8723`)

| Method & path | Body | Returns |
|---|---|---|
| `GET /api/health` | – | `{status:"ok", version, capabilities:{ffmpeg,gpu,musicgen,whisper,demucs,pedalboard,rubberband,madmom,essentia,crepe}}` |
| `GET /api/search?q=...&limit=12` | – | `{results:[{video_id,title,channel,duration_sec,thumbnail_url,view_count,url}]}` |
| `GET /api/projects` | – | `{projects:[ProjectState]}` |
| `POST /api/projects` | `{name?}` | `ProjectState` |
| `GET /api/projects/{pid}` | – | `ProjectState` |
| `DELETE /api/projects/{pid}` | – | `{ok:true}` |
| `POST /api/projects/{pid}/reference` | `{url}` | `{job_id}` (acquire + analyze) |
| `POST /api/projects/{pid}/generate` | `{similarity:0..100}` | `{job_id}` |
| `POST /api/projects/{pid}/vocal` | multipart `file` | `{job_id}` (process + score) |
| `GET /api/projects/{pid}/arrangement` | – | arrangement.json content |
| `PUT /api/projects/{pid}/arrangement` | `{placements:[...]}` | updated arrangement |
| `POST /api/projects/{pid}/rearrange` | `{weights?:{...}}` | `{job_id}` (re-score + re-arrange) |
| `POST /api/projects/{pid}/render` | `{retune_speed:0..100, preset, loudness_preset, title?, artist?}` | `{job_id}` |
| `GET /api/jobs/{job_id}` | – | `Job` |
| `GET /api/similarity/preview?pid=...&value=72` | – | `{summary:"72% — same key, ..."}` |
| `GET /media/{pid}/<relpath>` | – | static file (audio preview, thumbnails) |
| `WS /ws` | – | server pushes `{type:"job", job:Job}` on every job update |

`Job = {id, type:"reference"|"generate"|"vocal"|"rearrange"|"render", project_id,
status:"queued"|"running"|"done"|"error", progress:0..1, message, result?, error?}`

`ProjectState (project.json)`:
```json
{
  "id":"p_ab12cd", "name":"Untitled", "created_at":"ISO8601",
  "reference": {"url","video_id","title","channel","duration_sec","thumbnail_url","analyzed":true} ,
  "similarity": 72,
  "instrumental": {"file":"instrumental.wav","grid":"instrumental_grid.json","engine":"musicgen-small|procedural","uniqueness_passed":true},
  "vocal": {"file":"vocal_raw.wav","analysis":"vocal_analysis.json","lyrics_available":true},
  "arrangement_ready": true,
  "exports": [{"file":"exports/song.wav","format":"wav"},{"file":"exports/song.mp3","format":"mp3"}]
}
```
(null for stages not reached yet)

## Module function signatures (exact — the spine `tasks.py` calls these)

```python
# enjoi/modules/search.py
def search_youtube(query: str, limit: int = 12) -> list[dict]   # SearchResult dicts as above

# enjoi/modules/reference.py
def acquire_and_analyze(project: "Project", url: str, progress) -> dict
# downloads audio into project.ref_cache_dir via yt-dlp+ffmpeg, runs full MIR
# analysis, writes reference_profile.json, returns the profile dict.

# enjoi/modules/similarity.py
def build_generation_plan(profile: dict, similarity: int, rng_seed: int | None = None) -> dict
# pure function: maps slider per spec table 4.3 → plan dict (below)
def similarity_summary(profile: dict, similarity: int) -> str   # human-readable live label

# enjoi/modules/generate.py
def generate_instrumental(project: "Project", plan: dict, progress) -> dict
# sectional generation + stitching + conform; runs uniqueness guard via unique.py
# (reference audio still in _ref_cache at this point); auto-retry per spec 4.3.1;
# deletes _ref_cache when done; writes instrumental.wav, instrumental_grid.json,
# uniqueness_report.json; returns {"grid":..., "report":..., "engine":...}

# enjoi/modules/unique.py
def run_uniqueness_audit(profile: dict, candidate_wav: Path, progress=None) -> dict
# Compares the candidate against profile["fingerprints"] (computed by reference.py
# while the reference audio still existed — melody interval n-grams, chord sequence,
# beat-synced chroma, fingerprint hashes). Works even after _ref_cache is deleted.
# Returns uniqueness_report dict {passed:bool, checks:{...}} per spec 4.3.1.

# enjoi/modules/vocal.py
def process_vocal(project: "Project", uploaded_path: Path, progress) -> dict
# resample→clean→transcribe→phrase segmentation; writes vocal_raw.wav +
# vocal_analysis.json (phrases, words, lyrics); returns the analysis dict.

# enjoi/modules/score.py
def score_sections(analysis: dict, weights: dict | None = None) -> dict
# pure: groups phrases into candidate sections, computes ImpactScore per spec 4.6,
# assigns roles chorus/verse/bridge; returns analysis dict with "sections" filled.

# enjoi/modules/arrange.py
def build_arrangement(project: "Project", grid: dict, analysis: dict, progress) -> dict
# maps roles onto grid slots, onset-aligns to downbeats, ±6% stretch budget,
# cuts chop wavs into vocal_chops/, writes arrangement.json, returns it.

# enjoi/modules/tune.py
def tune_vocals(project: "Project", arrangement: dict, grid: dict,
                retune_speed: int, progress) -> dict
# pitch-corrects every chop to the grid key scale, writes vocal_tuned/, returns
# arrangement with placements pointing at tuned files ("tuned_file" key added).

# enjoi/modules/mix.py
def mix_and_master(project: "Project", arrangement: dict, grid: dict,
                   preset: str, loudness_preset: str, progress) -> Path
# builds the two-source render graph (instrumental.wav + vocal_tuned chops ONLY),
# applies bus chains per spec 4.9, masters to target LUFS, returns path to
# exports/_master_tmp.wav (44.1k/24-bit float render).

# enjoi/modules/export.py
def export_song(project: "Project", master_wav: Path, metadata: dict,
                include_stems: bool, progress) -> list[dict]
# encodes WAV 24-bit + MP3 320 (ffmpeg), writes ID3/BWF metadata, song_manifest.json,
# optional stems; returns exports list for ProjectState.
```

`Project` is `enjoi.core.storage.Project` (has `.dir`, `.ref_cache_dir`,
`.instrumental_path`, `.grid_path`, `.vocal_raw_path`, `.vocal_analysis_path`,
`.arrangement_path`, `.chops_dir`, `.tuned_dir`, `.exports_dir`,
`.uniqueness_report_path`, `.reference_profile_path`, plus `read_state()/update_state(**kw)`).

## JSON artifact schemas

### reference_profile.json
```json
{
  "source": {"title":"","channel":"","url":"","video_id":"","duration_sec":0.0},
  "duration_sec": 212.4,
  "bpm": 120.2, "beat_times": [0.42, ...], "downbeats": [0.42, ...],
  "time_signature": "4/4",
  "key": {"tonic":"A","mode":"minor","confidence":0.81},
  "structure": [{"label":"intro","start":0.0,"end":12.3,"bars":8}],
  "energy_curve": {"per_bar_rms":[...], "per_bar_flux":[...]},
  "instrumentation": {"drums":0.9,"bass":0.8,"melodic":0.7,"vocals":0.6},
  "groove": {"swing":0.12,"pattern_class":"backbeat","onset_histogram":[16 floats]},
  "genre_tags": ["pop"], "mood_tags": ["energetic"],
  "ref_audio": "_ref_cache/reference.wav",
  "fingerprints": {
    "melody_interval_ngrams": ["2,-1,3,0,-2,1", "..."],
    "chord_sequence": ["Am","F","C","G", "..."],
    "chroma_downbeat": [[0.1, "... 12 floats per downbeat ..."]],
    "fp_hashes": [123456789, "..."]
  }
}
```
`fingerprints` is computed by `reference.py` while the reference audio exists; the
Uniqueness Guard (`unique.py`) compares candidates against it so audits work even
after `_ref_cache/` is deleted. `melody_interval_ngrams` are 6-note pitch-interval
sequences (semitones, comma-joined). `chord_sequence` is one chord per bar.
`chroma_downbeat` is the beat-synchronous chroma matrix (one 12-vector per downbeat,
L2-normalized). `fp_hashes` are 32-bit spectral landmark hashes (Chromaprint-style).
Structure labels: `intro|verse|prechorus|chorus|bridge|outro|inst`.

### generation plan (in-memory, from similarity.py)
```json
{
  "similarity": 72, "seed": 1234,
  "target_duration_sec": 212.4, "bpm": 118.0,
  "key": {"tonic":"A","mode":"minor"}, "time_signature":"4/4",
  "structure": [{"label":"intro","bars":8}, ...],
  "energy_targets": [0.4, ...],            // per section 0..1
  "instrument_palette": ["drums","bass","piano","pad"],
  "groove": {"swing":0.1,"pattern_class":"backbeat"},
  "prompt": "energetic pop, 118 bpm, A minor, ...",   // for MusicGen
  "summary": "72% — same key, same structure, tempo within 3% ..."
}
```

### instrumental_grid.json
```json
{
  "bpm":118.0, "time_signature":"4/4",
  "key":{"tonic":"A","mode":"minor","scale_midi":[57,59,60,62,64,65,67]},
  "beat_times":[...], "downbeats":[...],
  "sections":[{"label":"verse","start":12.3,"end":40.1,"bars":16}],
  "duration_sec":212.0, "engine":"musicgen-small"
}
```

### vocal_analysis.json
```json
{
  "file":"vocal_raw.wav","duration_sec":95.2,
  "lyrics":"full transcript ...",
  "words":[{"w":"hello","start":1.2,"end":1.5}],
  "phrases":[{"id":0,"start":1.2,"end":4.8,"text":"...",
              "features":{"rms":0.31,"f0_mean_hz":220.0,"f0_range_semitones":7.2,
                          "pitch_height":0.6,"vibrato":0.2,"brightness":0.5}}],
  "sections":[{"id":0,"phrase_ids":[0,1,2],"start":1.2,"end":14.0,"text":"...",
               "impact_score":0.74,
               "scores":{"energy":0.8,"pitch_range":0.6,"pitch_height":0.7,
                         "vibrato":0.5,"repetition":0.9,"brightness":0.6,"hookiness":0.8},
               "role":"chorus"}],
  "weights": {"energy":0.20,"pitch_range":0.15,"pitch_height":0.10,"vibrato":0.15,
              "repetition":0.20,"brightness":0.10,"hookiness":0.10}
}
```

### arrangement.json
```json
{
  "placements":[{"id":0,"role":"chorus","slot_label":"chorus","slot_index":1,
                 "section_id":2,"source_start":14.2,"source_end":28.0,
                 "target_start":42.1,"stretch":1.02,"gain_db":0.0,
                 "chop_file":"vocal_chops/p000.wav","tuned_file":null}],
  "slots":[{"label":"chorus","index":1,"start":42.1,"end":70.0,"filled":true}],
  "summary":{"chorus_section_id":2,"bridge_section_id":4}
}
```

### uniqueness_report.json
```json
{
  "passed": true, "attempts": 1, "effective_similarity": 72,
  "checks": {
    "melody_ngram_overlap": {"value":0.07,"threshold":0.25,"passed":true},
    "chord_run_length":     {"value":3,"threshold":4,"passed":true,"exempt_loops":true},
    "chroma_correlation":   {"value":0.41,"threshold":0.80,"passed":true},
    "audio_fingerprint":    {"value":0,"threshold":0,"passed":true}
  },
  "summary": "Originality check: passed — melody overlap 7%, no chord-run matches, no fingerprint matches."
}
```

### song_manifest.json
```json
{
  "title":"","artist":"","bpm":118.0,"key":"A minor","created_at":"ISO8601",
  "sources":["generated instrumental (local MusicGen/procedural)","user one-take vocal"],
  "reference_audio_in_output": false,
  "uniqueness_report": {...},
  "exports":[{"file":"song.wav","format":"wav","lufs":-14.0,"true_peak_db":-1.1}]
}
```

## Frontend contract
- Vite + React 18 + TypeScript + Tailwind, Electron shell (electron-builder).
- Backend base URL `http://127.0.0.1:8723`; WS `ws://127.0.0.1:8723/ws`.
- Six screens (stepper): Search → Analysis → Similarity → Vocal → Arrange → Mix&Export,
  matching spec §3. Audio previews use `<audio>` with `/media/...` URLs.
- Electron main spawns the backend (`scripts/run_backend`) in production, and assumes
  it is already running in dev (`npm run dev`).
