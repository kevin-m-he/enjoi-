# enjoi享受

**Reference-to-Release song maker.** A local, offline-first Windows desktop app that turns a *reference track* plus a *single one-take vocal recording* into a fully built, mixed, release-ready song. Search YouTube for a reference, dial in how close the vibe should be with a 0–100% Similarity Slider, generate an original instrumental of matching length, drop in one continuous vocal take — and the app finds your chorus, arranges verses and a bridge, autotunes, mixes, and exports a streaming-ready WAV/MP3. Everything runs on your machine; the internet is only used for YouTube search and analysis.

## Features

- **YouTube reference search** built in — pick any music video as a style reference (no API key needed).
- **Reference analysis** — BPM, key, time signature, song structure, energy curve, groove, and instrumentation profile extracted locally.
- **Similarity Slider (0–100%)** — control how closely the generated instrumental tracks the reference's *style*. 0% keeps only the song length; 100% matches tempo, key, structure, groove, and palette.
- **Original instrumental generation** — MusicGen (local) on capable hardware, with a procedural synthesis fallback so the app works on any PC.
- **One-take vocal workflow** — upload a single WAV/MP3 take; the app transcribes lyrics, detects your best section as the chorus, maps verses, and builds a bridge.
- **Auto-arrangement** — vocal phrases snapped to the instrumental's beat grid, with a timeline you can nudge.
- **Autotune & pro mix chain** — adjustable retune speed, genre mix presets, mastering to streaming loudness (−14 LUFS default).
- **Originality Check** — every instrumental passes an automated uniqueness audit before you ever hear it, and the report ships with your export.

## The ownership guarantee — "style, never substance"

The Similarity Slider only ever controls **non-copyrightable style descriptors**: tempo, key, groove feel, energy shape, section structure, and instrument palette. The elements copyright actually protects — the **melody**, the **chord progression**, the **lyrics**, and the **sound recording itself** — are **never copied or conditioned on at any slider value, including 100%**. 100% means "same vibe," never "same song."

Two mechanisms enforce this:

1. **Hard source isolation.** The final render graph has exactly two inputs: the freshly generated instrumental and audio derived from *your* uploaded vocal. The reference audio lives in an analysis-only sandbox the render engine cannot read, and is deleted right after analysis. No reference audio is ever present in the output.
2. **The Uniqueness Guard.** Every generated instrumental must pass an automated divergence audit against the reference before it reaches you: melody interval n-gram overlap (< 25%), longest shared chord-progression run (≤ 4 chords, common loops exempt), beat-synced chroma correlation (< 0.80), and an audio-fingerprint match check (zero matches). Failing sections are quietly regenerated. The results are written to `uniqueness_report.json` and summarized on the export screen as publishing evidence.

**Honest caveats:**

- No software can *guarantee* legal non-infringement. The guard enforces measurable divergence on melody, harmony, and audio — which removes the practical copying risk — but it is not legal advice.
- Downloading YouTube audio, even for analysis-only use that never reaches the output, may violate YouTube's Terms of Service. The app uses an auto-deleted analysis sandbox and documents this; if you want a ToS-clean path, use your own local audio file as the reference where supported.

## Quickstart

Prerequisites: Windows 10/11, [Python 3.11](https://www.python.org/downloads/) (`winget install Python.Python.3.11`), Node.js 18+ (`winget install OpenJS.NodeJS.LTS`), FFmpeg (`winget install Gyan.FFmpeg`).

```powershell
# 1. One-time setup (creates backend\.venv, installs deps, npm install)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1

#    ...or with the full ML stack (MusicGen, Whisper, Demucs - GPU recommended):
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1 -Full

# 2. Optional: prefetch model weights so the first song doesn't stall on downloads
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\download_models.ps1

# 3. Run in development (backend window + Vite/Electron)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dev.ps1

# Production-ish run against built frontend files:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start.ps1
```

The app works end-to-end with only the core install — heavier ML packages are quality upgrades, each with a graceful fallback. See the [User Guide](docs/USER_GUIDE.md) for the screen-by-screen walkthrough.

## Hardware tiers

| Tier | Spec | Experience |
|---|---|---|
| Minimum | 4-core CPU, 16 GB RAM, no GPU | works; generation 10–25 min via musicgen-small (or instant procedural fallback) |
| Recommended | NVIDIA GPU 8 GB+ VRAM (RTX 3060+), 16 GB RAM | generation 1–3 min, full pipeline < 10 min |
| Disk | ~12 GB (models) + ~1 GB per project | one-time model download on first run |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Frontend: Electron + React + Tailwind (desktop UI)    │
│  - Search, slider, timeline, players, progress         │
└───────────────▲────────────────────────────────────────┘
                │ HTTP/WebSocket (localhost only)
┌───────────────┴────────────────────────────────────────┐
│  Backend: Python 3.11 + FastAPI (local server)         │
│  ├─ search/      yt-dlp search & metadata              │
│  ├─ analyze/     librosa, madmom, Essentia, Demucs     │
│  ├─ generate/    MusicGen (PyTorch, CUDA/CPU)          │
│  ├─ vocal/       faster-whisper, noisereduce           │
│  ├─ score/       impact scoring, chorus/bridge picker  │
│  ├─ arrange/     grid alignment, Rubber Band stretch   │
│  ├─ tune/        CREPE/pYIN + PSOLA autotune           │
│  ├─ mix/         Pedalboard chains, LUFS metering      │
│  └─ export/      FFmpeg encode, metadata, manifest     │
│  Job queue: long tasks run as background jobs with     │
│  WebSocket progress events (generation, mixing)        │
└────────────────────────────────────────────────────────┘
Storage:  %APPDATA%/enjoi/projects/<project-id>/
          ├─ reference_profile.json   (kept)
          ├─ _ref_cache/              (analysis sandbox, auto-deleted)
          ├─ instrumental.wav + instrumental_grid.json
          ├─ vocal_raw.wav  vocal_chops/  vocal_tuned/
          └─ exports/  song_manifest.json
```

The backend listens on `http://127.0.0.1:8723` (localhost only). Full REST/WS contract: [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Desktop shell | Electron (+ React, Tailwind) | fast UI iteration, native file dialogs |
| Backend | Python 3.11, FastAPI, Uvicorn | entire audio/ML ecosystem is Python |
| YouTube | yt-dlp + FFmpeg | search + audio extraction, no API key |
| Beat/key analysis | librosa (madmom/Essentia if installed) | best-in-class open MIR tools |
| Stem profiling | Demucs v4 (htdemucs) | instrumentation fingerprint |
| Music generation | MusicGen (audiocraft) | local and free; procedural fallback built in |
| Transcription | faster-whisper (small/medium) | local word-level lyric timestamps |
| Pitch | CREPE / pYIN, psola | accurate f0 + natural retune |
| Time-stretch | Rubber Band Library (pyrubberband) | formant-safe, high quality |
| DSP/mixing | Spotify Pedalboard, pyloudnorm | pro-grade FX + LUFS compliance |
| Packaging | electron-builder + PyInstaller backend | single Windows installer (see [docs/PACKAGING.md](docs/PACKAGING.md)) |

## Troubleshooting

**"ffmpeg not found"** — Install it: `winget install Gyan.FFmpeg`, then open a *new* terminal (winget edits PATH for future shells only). The launch scripts also probe `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*` directly and patch the session PATH, so a re-run after install usually just works.

**No GPU / "generation is slow"** — Without an NVIDIA GPU the app uses CPU MusicGen (slow: 10–25 min) or the instant procedural synth engine if audiocraft isn't installed. This is expected; see the hardware tiers above. `setup.ps1 -Full` auto-detects your GPU via `nvidia-smi` and picks CUDA (cu121) or CPU PyTorch wheels accordingly.

**Port 8723 is busy** — Find and stop the occupant:
```powershell
Get-NetTCPConnection -LocalPort 8723 | Select-Object OwningProcess
Stop-Process -Id <pid>
```
(Often a leftover backend window — just close it.)

**`python` opens the Microsoft Store** — That's the Windows Store *alias*, not a real Python. Either install real Python (`winget install Python.Python.3.11`) or disable the alias under *Settings → Apps → Advanced app settings → App execution aliases*. `setup.ps1` deliberately ignores the Store alias when searching for Python.

**`running scripts is disabled on this system`** — Launch the scripts exactly as shown in Quickstart (`powershell -ExecutionPolicy Bypass -File ...`), which bypasses the machine policy for that one invocation.

## Where your data lives

All projects, caches, and downloaded models are stored under:

```
%APPDATA%\enjoi\        (e.g. C:\Users\<you>\AppData\Roaming\enjoi)
  ├─ projects\<project-id>\   one folder per song project
  ├─ models\                  app-managed model files
  └─ cache\                   thumbnails and other transient data
```

ML weights fetched by Hugging Face libraries live in `%USERPROFILE%\.cache\huggingface`. Set the `ENJOI_DATA_DIR` environment variable to relocate the data directory.

## Documentation

- [User Guide](docs/USER_GUIDE.md) — screen-by-screen walkthrough, recording tips, what the Originality Check means.
- [API Contract](docs/API_CONTRACT.md) — internal REST/WS API and module contract.
- [Packaging](docs/PACKAGING.md) — how the Windows installer will be produced.
- [Project Build Document](PROJECT_BUILD_DOCUMENT.md) — full product specification.

## License & fair use

For personal use. Your exported songs contain only the generated instrumental and your own voice — you own them. Respect YouTube's Terms of Service when using the reference-search feature, and remember the Originality Check is an engineering safeguard, not legal advice.
