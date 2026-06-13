"""enjoi享受 backend entry point — FastAPI on 127.0.0.1:8723 (localhost only)."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from enjoi import __version__
from enjoi.api.routes import router
from enjoi.core import config
from enjoi.core.jobs import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.set_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="enjoi", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "app://."],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# /media/<project-id>/<relpath> → file previews (audio players, exports)
app.mount("/media", StaticFiles(directory=str(config.projects_dir())), name="media")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
