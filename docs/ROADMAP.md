# enjoi享受 — What's left to do

A living checklist to take enjoi from "working desktop app" to "free public
website." Grouped by theme; each item is tagged:

- **[me]** I can build it (code/config) in this repo.
- **[you]** needs your account, credentials, money, or a product decision.
- **[both]** I build it; you provide a secret/account/approval to finish.

---

## ✅ Where we are now
- Full local pipeline works: Search → Analyze → Similarity → Vocal → Arrange →
  Mix/Export, in the Electron desktop app with a local FastAPI backend.
- **Generation = real sampled instruments** (`band.py` loop engine): 144 licensed
  loops, warped to the song's tempo/key, one-shot programmed drums, "less is
  more" arrangement, vocal pocket + leveling. MusicGen and a GM SoundFont are
  fallbacks. Generation is one pass (audit is advisory).
- GPU stack installed (torch/MusicGen/Whisper/demucs).
- UI: neo-brutalist Great Wave theme, animated bubbles, wave loading bar, verse.
- Cloudflare sample host (private, token-gated R2 Worker) is **prepared but not
  deployed**; loops are gitignored, only the metadata manifest is committed.

---

## 1. Launch the sample hosting  (prepared — just needs you)
- [ ] **[you]** Deploy the Worker + R2 bucket and upload loops (≈5 commands in
      `docs/SAMPLES.md`): `wrangler login` → `r2 bucket create` →
      `secret put SAMPLE_TOKEN` → `wrangler deploy` → `scripts/upload-samples-r2.ps1`.
- [ ] **[you]** Set `ENJOI_SAMPLE_CDN` + `ENJOI_SAMPLE_CDN_TOKEN` wherever the
      backend runs.
- [ ] **[both]** (optional) Decide whether to keep the Cloudflare account id in
      the committed `wrangler.toml` or move it to a local-only file.

## 2. Put the app on the web  (the big one)
The frontend is a static SPA (easy); the backend is heavy Python (the hard part).
- [ ] **[both]** **Frontend → Cloudflare Pages.** Build is already static; point
      it at the hosted backend URL instead of `127.0.0.1:8723` (make the API base
      configurable via an env/build var). I can wire this; you create the Pages
      project + domain.
- [ ] **[you]** **Backend host.** librosa/demucs/MusicGen/pedalboard can't run on
      Cloudflare Workers — they need a real server/container (Fly.io, Render,
      Railway, a VPS, or a GPU box if you keep MusicGen). Pick a host + budget.
- [ ] **[me]** Containerize the backend (Dockerfile + start script) and make the
      API CORS/host/HTTPS-ready for a public origin.
- [ ] **[me]** Make the reference path default to **"upload your own audio"** for
      the public build (the yt-dlp/YouTube path is a ToS + IP-block risk at
      scale; keep it as an optional/local-only feature).

## 3. Make it multi-user & safe to run publicly
Currently single-user (one `%APPDATA%` folder, in-process job threads).
- [ ] **[me]** Per-session/project isolation in cloud storage (R2/S3) instead of
      one local dir; signed URLs for previews/exports.
- [ ] **[me]** A real job queue + worker pool for concurrent generations (today
      it's in-process threads — fine for one user, not for many), with limits on
      concurrent jobs.
- [ ] **[me]** Storage TTL/cleanup (auto-delete old projects/exports) so it
      doesn't grow forever.
- [ ] **[both]** Rate limiting / basic abuse protection (per-IP caps; Cloudflare
      can help). You decide policy.
- [ ] **[both]** Cost guardrails (generation is CPU/GPU heavy — cap usage or
      queue length per the host budget).

## 4. Legal / licensing  (must be right before public launch)
- [ ] **[me]** Enforce the licensing-safe model in the deployed build: loops stay
      private (token Worker), generation is **server-side only**, the public
      receives **only the finished song**, never raw loops. (Architecture is
      documented in `docs/SAMPLES.md`; needs to be the actual deployed shape.)
- [ ] **[you]** Confirm each sample pack's license permits this use (using loops
      to produce/distribute derivative instrumentals). Keep proof.
- [ ] **[me]** Add a short Terms / disclaimer page (ownership of output, the
      originality-guard "not legal advice" note already in-app, YouTube caveat).
- [ ] **[you]** Decide the ownership/usage terms shown to end users.

## 5. Music quality backlog
- [ ] **[both]** Broaden the sample library — it's currently heavy on
      Latin/Spanish guitar + trap + synth; **folk/country/rock/acoustic-drums are
      thin**. Add loops in those styles (you drop files in; I index them).
- [ ] **[me]** Per-genre mix tuning + the "tight waveform like mainstream"
      target: measure a reference song's loudness/dynamics and match generated
      instrumentals within a tolerance (build an analysis→target loop).
- [ ] **[me]** Vocal pipeline quality pass (chop/auto-tune/placement/mix) for the
      public bar, now that the instrumental is sample-based.
- [ ] **[me]** Smarter loop selection (chord-progression aware, energy-curve
      aware) and section-to-section variation so songs feel less loop-y.

## 6. Reliability & docs
- [ ] **[me]** End-to-end automated test for the full web flow (reference →
      generate → vocal → render) in CI.
- [ ] **[me]** Fresh-clone setup guide (one script) for a new machine/server.
- [ ] **[me]** Remove dev-only artifacts (`backend/.wheels/` scratch scripts) and
      tidy before launch.

## 7. Nice-to-haves (post-launch)
- [ ] Higher-fidelity generation option (musicgen-stereo / a paid model) behind a
      flag.
- [ ] Project save/recall + A/B compare, harmony stacking from the user's vocal,
      distributor export presets — the original spec's "future enhancements."
- [ ] Accounts / save history (only if you want persistence per user).

---

### Suggested next 3 steps
1. **[you]** Deploy the Cloudflare sample host (§1) — it's ready.
2. **[you+me]** Choose a backend host (§2) — then I containerize it and wire the
   frontend's API base + deploy the SPA to Pages.
3. **[me]** Switch the public build's reference to "upload your own audio" and add
   the Terms/disclaimer page (§2, §4).
