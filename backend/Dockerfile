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
    && pip install "tinysoundfont>=0.3"

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
