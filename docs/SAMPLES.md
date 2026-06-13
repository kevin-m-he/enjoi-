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

## Deploying the samples to Cloudflare (ready, not launched)
A token-gated Worker + R2 bucket are prepared in `cloudflare/sample-worker/`
(wired to account `ed44bf6a…`). The Worker serves the loops ONLY to callers that
present a shared secret — so the raw `.wav` files are never publicly
downloadable. Steps (you run these with your own Cloudflare login):

```
npm i -g wrangler
wrangler login
wrangler r2 bucket create enjoi-samples
cd cloudflare/sample-worker
wrangler secret put SAMPLE_TOKEN        # paste a long random string
wrangler deploy
cd ../..
powershell -ExecutionPolicy Bypass -File scripts/upload-samples-r2.ps1
```
Then point the app at it:
```
$env:ENJOI_SAMPLE_CDN       = "https://enjoi-samples.<your-subdomain>.workers.dev"
$env:ENJOI_SAMPLE_CDN_TOKEN = "<the SAMPLE_TOKEN you set>"
```
With no local `sample_library/`, the backend reads `sample_manifest.json` and
downloads each needed loop from the Worker (sending the token), caching it.

## Licensing for the public free website
Serving the raw loops to end users would violate the pack licenses. The safe
model: keep the loops **private** (this token-gated Worker), run the generation
**on a server** that holds the token, and deliver only the **finished
instrumental** (a derivative work you're licensed to create) to the public —
never the raw loops. The token lives only in the server's env, never in the
browser. (Cloudflare Workers can't run the Python/librosa generation itself, so
the generation backend needs a real server/container; the Worker here is purely
the private sample store.)
