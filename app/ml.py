"""
ml.py — Model loading and inference logic.

This module handles:
1. Loading a sentence-transformer model at startup.
2. Pushing it to CUDA (GPU) if available, otherwise falling back to CPU.
3. Computing text embeddings and returning them as Python lists.

We keep this separate from main.py so that the FastAPI app code stays clean
and the ML logic can be tested or swapped independently.
"""

import logging
import os

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — populated once at startup via load_model().
# Using module-level globals keeps things simple; there is exactly one model
# instance shared across all request-handling threads / async tasks.
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_device: str = "cpu"


def load_model() -> None:
    """
    Load the embedding model into memory and move it to the best
    available device (CUDA → CPU fallback).

    Called once from the FastAPI lifespan handler so that the first
    request doesn't pay the model-loading cost.
    """
    global _model, _device

    model_name = os.getenv(
        "MODEL_NAME",
        # Default: a small, multilingual model that works well for
        # Finnish and other languages — matches the main app's needs.
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # --- Device selection ---------------------------------------------------
    # torch.cuda.is_available() returns True only when:
    #   • NVIDIA drivers are installed on the host, AND
    #   • The container was started with GPU access (--gpus / deploy block).
    # On a CPU-only dev machine this will simply be False.
    if torch.cuda.is_available():
        _device = "cuda"
        logger.info("CUDA is available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        _device = "cpu"
        logger.info("CUDA is NOT available — falling back to CPU.")

    logger.info("Loading model '%s' onto device '%s' …", model_name, _device)
    _model = SentenceTransformer(model_name, device=_device)
    logger.info("Model loaded successfully.")


def is_gpu_available() -> bool:
    """Check whether the model is currently running on a CUDA device."""
    return _device == "cuda"


def compute_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings for a list of text strings.

    Returns a list of embedding vectors (each vector is a plain Python
    list of floats so it can be directly JSON-serialised).
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded yet. Call load_model() first.")

    # SentenceTransformer.encode() handles batching internally.
    # convert_to_numpy=True gives us an ndarray which is faster to
    # convert to Python lists than a PyTorch tensor.
    embeddings = _model.encode(texts, convert_to_numpy=True)

    # .tolist() converts the numpy ndarray → nested Python lists.
    return embeddings.tolist()
