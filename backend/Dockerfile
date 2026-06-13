# enjoi backend — lean CPU image.
#
# Runs the FULL pipeline (loop instrumental engine + mix/master + vocal) on
# requirements-core alone. torch / demucs / MusicGen are OPTIONAL quality
# upgrades, intentionally omitted here to keep the image small (~1 GB vs ~4 GB)
# and safe on a 4 GB box. Every heavy dep is optional_import-guarded with a
# librosa fallback, so nothing in the request path breaks without them.
#
# Build/run from the REPO ROOT via docker-compose.yml (context = repo root).
FROM python:3.11-slim

# System libs the audio stack needs: ffmpeg (yt-dlp + decode), libsndfile (soundfile).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENJOI_HOST=0.0.0.0 \
    ENJOI_PORT=8723 \
    ENJOI_DATA_DIR=/data

WORKDIR /app/backend

# Deps first for layer caching. Core runs everything; tinysoundfont adds the
# General-MIDI fallback engine (pure wheel, tiny).
COPY backend/requirements-core.txt ./requirements-core.txt
RUN pip install --upgrade pip \
    && pip install -r requirements-core.txt \
    && pip install --no-deps "tinysoundfont>=0.3"
# --no-deps: tinysoundfont's only runtime dep is pyaudio, which it imports lazily
# ONLY for live audio playback (synth.py). The server renders offline, never plays
# live, so pyaudio is unused — and it has no Linux wheel (would force a source
# build + PortAudio). Skipping it keeps the SoundFont fallback working and the
# image lean. (tinysoundfont itself ships a manylinux x86_64 wheel.)

# YouTube changes its streaming API constantly; pull the newest yt-dlp so
# reference downloads keep working (and the player_client args in reference.py
# are recognized). Re-runs on each rebuild because YouTube breakage is frequent.
RUN pip install -U --no-cache-dir yt-dlp

# App code. NOTE: backend/sample_library/ is excluded by .dockerignore (licensed
# loops never go in the image); samples are fetched at runtime from the R2 CDN
# via ENJOI_SAMPLE_CDN, indexed by the committed sample_manifest.json.
COPY backend/ ./

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8723

# SINGLE worker is REQUIRED: job state + WebSocket connections live in-process on
# one event loop. Do NOT raise --workers without an external queue (ROADMAP §3).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8723", "--workers", "1"]
