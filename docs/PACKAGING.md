# Packaging enjoi享受 for Windows

Goal (Phase 6): a single Windows installer produced by **electron-builder**, embedding a **PyInstaller-frozen backend** as an extra resource. This document is the plan and the known gotchas; it is not yet wired into CI.

## Big picture

```
installer (NSIS, via electron-builder)
└─ enjoi.exe (Electron shell)
   └─ resources/
      └─ backend/                  <- extraResources: PyInstaller dist
         └─ enjoi-backend.exe      <- frozen FastAPI server (core deps only)
```

- Electron's main process spawns `resources/backend/enjoi-backend.exe` on launch (in dev it assumes the backend is already running — see `scripts/dev.ps1`), waits for `http://127.0.0.1:8723/api/health`, then opens the window.
- **The frozen backend contains only `requirements-core.txt`.** The heavy ML stack stays pip-installed (see "ML extras stay out of the bundle" below).

## 1. PyInstaller spec for the backend

Freeze from `backend\` using the venv that has *core* requirements only (a `-Full` venv would balloon the bundle with torch).

Entry point: `main.py`. Because uvicorn is started programmatically (`uvicorn.run(app, ...)` in `main.py`), freezing the entry script directly works — but several imports are dynamic and need to be declared.

Outline of `backend/enjoi-backend.spec` (to be created in Phase 6):

```python
# enjoi-backend.spec (outline)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = (
    # uvicorn loads its event loop / protocol classes by string name:
    [
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan.on",
        "anyio._backends._asyncio",
    ]
    # librosa/scipy do lazy submodule imports:
    + collect_submodules("librosa")
    + collect_submodules("scipy.signal")
    + ["soxr", "soundfile", "noisereduce", "pyloudnorm", "pedalboard", "mutagen"]
    # our own modules are imported via the package, keep them explicit:
    + collect_submodules("enjoi")
)

datas = (
    collect_data_files("librosa")        # librosa/util/example_data, registry files
    + collect_data_files("soundfile")    # bundled libsndfile DLL
    + collect_data_files("yt_dlp")
    + collect_data_files("imageio_ffmpeg")  # fallback ffmpeg binary
)

a = Analysis(["main.py"], pathex=["."], hiddenimports=hiddenimports, datas=datas)
# ... standard PYZ/EXE/COLLECT; build as a one-FOLDER app (onedir), console=False
```

Known gotchas to verify during Phase 6:

- **onedir, not onefile.** Onefile unpacks to temp on every launch (slow with numpy/librosa) and breaks relative data lookups; electron-builder ships a folder anyway.
- **librosa data files** (`collect_data_files("librosa")`) are mandatory — missing them fails at first `librosa` call, not at startup.
- `soundfile` needs its bundled `libsndfile` DLL collected; pedalboard ships compiled extensions that PyInstaller picks up automatically but should be smoke-tested.
- `yt-dlp` self-update must be disabled in frozen mode (it already detects freezing, but verify).
- Smoke test the frozen exe standalone: run it, hit `/api/health`, run a core-only end-to-end render.

Build command (Phase 6):

```powershell
backend\.venv\Scripts\python.exe -m PyInstaller backend\enjoi-backend.spec --noconfirm --distpath backend\dist
```

## 2. electron-builder configuration notes

In `frontend/package.json` (or `electron-builder.yml`):

```jsonc
{
  "build": {
    "appId": "app.enjoi",
    "productName": "enjoi",
    "win": { "target": "nsis" },
    "files": ["dist/**", "electron/**"],
    "extraResources": [
      {
        "from": "../backend/dist/enjoi-backend",   // PyInstaller onedir output
        "to": "backend"                            // -> resources/backend/
      }
    ]
  }
}
```

- Electron main resolves the backend at `path.join(process.resourcesPath, "backend", "enjoi-backend.exe")` in production, and skips spawning in dev (`scripts/dev.ps1` owns the backend there).
- Kill the spawned backend on `app.quit` (and on second-instance lock) or port 8723 stays occupied.
- Keep `asar` enabled for the frontend; the backend lives outside the asar via `extraResources` so its DLLs load normally.
- Product name caution: keep the install path ASCII (`enjoi`) — some audio libraries mishandle non-ASCII install paths, so don't use 享受 in `productName`/install dir.

## 3. ML extras stay pip-installed (deliberate)

The optional stack (`requirements-full.txt`: torch, audiocraft/MusicGen, faster-whisper, demucs, torchcrepe, pyrubberband) is **not** frozen into the installer:

- Torch + CUDA wheels alone exceed 2.5 GB and are hardware-specific (cu121 vs CPU) — bundling one flavor would be wrong for half the users.
- PyInstaller and torch/audiocraft interact badly (dynamic CUDA DLL loading, JIT kernels).
- Model weights (~12 GB) are downloaded at runtime anyway.

Instead, the installed app detects capabilities at startup (`/api/health` capability flags) and offers an "Enable full quality" path that runs the equivalent of:

```powershell
scripts\setup.ps1 -Full          # GPU-aware torch install + requirements-full.txt
scripts\download_models.ps1      # optional weight prefetch
```

against a user-writable venv (e.g. `%APPDATA%\enjoi\venv-full`) — the frozen core backend can then relaunch with that interpreter, or simply keep using its graceful fallbacks. The core-only frozen app remains fully functional either way (procedural synth engine, energy-only segmentation, scipy mix chain).

## 4. Release checklist (Phase 6)

1. `npm run build` in `frontend\` (Vite production bundle).
2. PyInstaller build from a **clean core-only venv**; smoke test the exe.
3. `npx electron-builder --win` in `frontend\`.
4. Install on a clean Windows VM with no Python/Node: app must reach the Search screen and complete a core-only song.
5. Verify `%APPDATA%\enjoi` is created and used (config: `backend/enjoi/core/config.py`), and that uninstall leaves user projects in place.
