"""
main.py — FastAPI application, IP-restriction middleware, and routes.

This is the single entry-point for the GPU microservice.  It wires up:
1. IP whitelisting (security middleware).
2. A /health endpoint for monitoring.
3. A /v1/embeddings endpoint that mirrors the OpenAI embedding format.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ml import compute_embeddings, is_gpu_available, load_model

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------
# Load .env file if present (no-op if it doesn't exist).
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan handler — runs once on startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Load the ML model when the application starts.

    Using the lifespan context manager (recommended since FastAPI 0.93+)
    instead of the deprecated @app.on_event("startup") decorator.
    """
    logger.info("Starting up — loading ML model …")
    load_model()
    logger.info("Model ready. Service is accepting requests.")
    yield
    # Nothing special to clean up on shutdown; Python's GC handles it.
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GPU Microservice",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Security: IP restriction middleware
# ---------------------------------------------------------------------------
# IPs that are ALWAYS allowed (localhost and common Docker bridge ranges).
# These make local development and Docker-internal health-checks work
# out of the box without touching the .env file.
DEFAULT_ALLOWED = {"127.0.0.1", "::1"}


def _get_allowed_ips() -> set[str]:
    """
    Parse ALLOWED_IPS from the environment.

    Format: comma-separated list, e.g. "10.0.0.5,192.168.1.100"
    If the variable is empty or unset, only DEFAULT_ALLOWED IPs are permitted.
    """
    raw = os.getenv("ALLOWED_IPS", "")
    extra = {ip.strip() for ip in raw.split(",") if ip.strip()}
    return DEFAULT_ALLOWED | extra


@app.middleware("http")
async def ip_whitelist_middleware(request: Request, call_next):
    """
    Reject requests from IPs that are not in the allow-list.

    Why middleware instead of a dependency?
    → It runs before any route logic, so even unknown paths are protected.
    → Keeps route functions clean.

    The client IP is read from request.client.host, which Uvicorn
    populates from the socket's peer address.  If you put a reverse
    proxy in front, make sure to forward the real IP via
    X-Forwarded-For and configure Uvicorn's --proxy-headers flag.
    """
    allowed = _get_allowed_ips()

    # If no extra IPs are configured (only defaults), we allow all traffic.
    # This makes first-run / local dev frictionless.  In production you
    # MUST set ALLOWED_IPS to lock the service down.
    if allowed != DEFAULT_ALLOWED:
        client_ip = request.client.host if request.client else None
        if client_ip not in allowed:
            logger.warning("Blocked request from %s", client_ip)
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: your IP is not allowed."},
            )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    """
    Mirrors the OpenAI /v1/embeddings request format.

    `input` can be a single string or a list of strings.
    """

    input: str | list[str]


class EmbeddingObject(BaseModel):
    """A single embedding vector inside the response."""

    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Top-level response matching the OpenAI format (simplified)."""

    data: list[EmbeddingObject]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Lightweight health-check endpoint.

    Returns GPU availability so monitoring dashboards can alert if the
    service loses access to the GPU after a driver update, etc.
    """
    return {"status": "ok", "gpu_available": is_gpu_available()}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(payload: EmbeddingRequest):
    """
    Compute text embeddings.

    Accepts a single string or a list of strings and returns one
    embedding vector per input text.
    """
    # Normalise input to a list so downstream code is uniform.
    texts = payload.input if isinstance(payload.input, list) else [payload.input]

    vectors = compute_embeddings(texts)

    return EmbeddingResponse(
        data=[EmbeddingObject(embedding=v) for v in vectors],
    )
