# ---------------------------------------------------------------------------
# Dockerfile — GPU-accelerated Python microservice
# ---------------------------------------------------------------------------
# Base image: official NVIDIA CUDA 12.1.1 runtime on Ubuntu 22.04.
# We install Python on top of it so PyTorch can find the CUDA libraries
# at runtime.  Using the "runtime" (not "devel") image keeps the final
# image smaller — we don't need nvcc or CUDA headers at runtime.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default "python" / "python3" command.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /service

# Install Python dependencies first (layer caching: deps change less often
# than application code).
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Copy application source.  We preserve the `app/` package structure so
# that "from app.ml import ..." works correctly.
COPY app/ ./app/

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Uvicorn will listen on all interfaces inside the container.
EXPOSE 8000

# Start the FastAPI app with Uvicorn.
# --host 0.0.0.0  → listen on all container interfaces
# --workers 1     → one worker is enough for a single-GPU service;
#                   the model lives in GPU memory so multiple workers
#                   would just duplicate it.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
