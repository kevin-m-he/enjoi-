# Sample library & cloud hosting

The "band" engine (`backend/enjoi/modules/band.py`) builds instrumentals from a
library of **real commercial loops/one-shots** (guitars, pianos, trap drums,
808s, synths/pads). Filenames encode BPM and key, which the engine parses to
warp each loop to the target song's tempo and key.

## Licensing — why the audio is NOT in the repo
The loops are licensed commercial packs. Committing them to a public repo would
violate those licenses, so **`backend/sample_library/` is gitignored** and never
pushed. Only `backend/sample_manifest.json` (metadata: name/category/bpm/key/
duration — no audio) is committed, so the code ships with its index.

## Two ways the engine finds samples
1. **Local (current setup):** drop `.wav` loops into `backend/sample_library/`
   (BPM and key in the filename, e.g. `Spanish Guitar_82bpm_Gm.wav`). The engine
   indexes them directly. Regenerate the manifest after adding files:
   ```
   .venv\Scripts\python -c "import sys;sys.path.insert(0,'.');from enjoi.modules import band;print(band.write_manifest())"
   ```
2. **Cloud (for the public website — not launched yet):** host the same `.wav`
   files on a Cloudflare Pages/R2 bucket and set the env var
   `ENJOI_SAMPLE_CDN=https://your-bucket.example.com`. With no local library
   present, the engine reads the committed manifest and **downloads each needed
   sample on demand, caching it** under `%APPDATA%/enjoi/cache/samples/`. This
   keeps the licensed audio off GitHub while letting the free website stream it.

`ENJOI_SAMPLE_LIB=<path>` overrides the local library location.
The General-MIDI SoundFont engine remains a fallback when no library is present.
