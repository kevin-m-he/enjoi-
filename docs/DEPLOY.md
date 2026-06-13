# Deploying enjoi to the web

Three free/cheap pieces + one ~$4.50/mo box:

| Piece | Where | Cost |
|---|---|---|
| Frontend (SPA) | Cloudflare Pages → `enjoi.dev` | free |
| Sample library | token-gated R2 Worker (`enjoi-samples`) | ~free |
| Backend (Python generator) | **Hetzner CX22**, Docker, `api.enjoi.dev` | ~$4.50/mo |
| DNS + HTTPS | Cloudflare (registrar + DNS) | ~$12/yr domain |

The backend is a **lean CPU image** — the loop instrumental engine, mixing, and
vocal processing all run on `requirements-core` (no GPU, no torch/demucs). One
generation at a time on a CX22; fine for launch traffic. Scale up (CX32, 8 GB)
or add a job queue (ROADMAP §3) when demand grows.

---

## 0. Prereqs (once)
- A Cloudflare account (you already have one: `ed44bf6a…`).
- Register **`enjoi.dev`**: Cloudflare dash → Domain Registration → Register.
- `npm i -g wrangler` and `wrangler login` on your machine.

## 1. Deploy the private sample host (R2 Worker)
From the repo root:
```bash
wrangler r2 bucket create enjoi-samples
cd cloudflare/sample-worker
wrangler secret put SAMPLE_TOKEN        # paste a long random string — SAVE IT
wrangler deploy                          # prints https://enjoi-samples.<sub>.workers.dev
cd ../..
powershell -ExecutionPolicy Bypass -File scripts/upload-samples-r2.ps1   # uploads 144 loops
```

## 2. Stand up the backend on Hetzner
1. Hetzner Cloud Console → create a **CX22** (Ubuntu 24.04), add your SSH key.
2. Point DNS at it: Cloudflare → DNS → add **A record** `api` → `<box IP>`,
   **DNS only (grey cloud)** so Caddy can issue a Let's Encrypt cert directly.
3. SSH in and install Docker:
   ```bash
   ssh root@<box IP>
   curl -fsSL https://get.docker.com | sh
   ```
4. Get the code + secrets onto the box:
   ```bash
   git clone https://github.com/kevin-m-he/enjoi-.git enjoi && cd enjoi
   cp .env.example .env
   nano .env        # paste ENJOI_SAMPLE_CDN + ENJOI_SAMPLE_CDN_TOKEN from step 1
   ```
5. Launch:
   ```bash
   docker compose up -d --build
   docker compose logs -f api      # watch it boot; Ctrl-C to stop tailing
   ```
6. Verify (after ~1 min for the cert):
   ```bash
   curl https://api.enjoi.dev/api/health
   ```
   Expect a JSON health blob with capability flags.

## 3. Deploy the frontend to Cloudflare Pages (like bonk-market)
**Dashboard:** Workers & Pages → Create → Pages → Connect to Git →
`kevin-m-he/enjoi-`. Settings:
- Build command: `cd frontend && npm install && npm run build`
- Output directory: `frontend/dist`
- Environment variable: `VITE_API_BASE = https://api.enjoi.dev`

Then Pages → Custom domains → add `enjoi.dev`.

**Or CLI:**
```bash
cd frontend
VITE_API_BASE=https://api.enjoi.dev npm run build
npx wrangler pages deploy dist --project-name=enjoi
```

## 4. Smoke-test the live flow
Open `https://enjoi.dev` → search a reference → generate → upload a vocal →
arrange → export. Watch `docker compose logs -f api` on the box for job progress.

---

## Operating notes
- **Update after a push:** `cd enjoi && git pull && docker compose up -d --build`
- **Single worker is intentional** — jobs and WebSockets are in-process. Don't
  raise `--workers` without an external queue (ROADMAP §3).
- **Storage** lives in the `enjoi-data` Docker volume (projects, exports, cached
  samples, the soundfont). Add a TTL cleanup before heavy public use (ROADMAP §3).
- **Reference path:** the YouTube search works, but at public scale YouTube may
  rate-limit/IP-block the box. Before a big launch, switch the public build's
  reference to **"upload your own audio"** (ROADMAP §2/§4) — a small code change.
- **If the image build fails on a missing wheel**, add `build-essential` to the
  `apt-get install` line in `backend/Dockerfile` and rebuild.
- **CORS** is currently `*`. Fine for launch (no browser-side secrets); tighten
  to `https://enjoi.dev` in `backend/main.py` when you want to lock it down.
