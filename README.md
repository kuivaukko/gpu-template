# GPU Microservice — Text Embeddings

A minimal, production-ready GPU-accelerated microservice that exposes an OpenAI-compatible `/v1/embeddings` endpoint. Designed to be called exclusively by a trusted backend over a private network.

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Web Framework | FastAPI + Uvicorn |
| ML Backend | PyTorch + sentence-transformers |
| Infrastructure | Docker, NVIDIA Container Toolkit |

---

## Prerequisites

| Requirement | Needed for |
|-------------|-----------|
| Docker & Docker Compose | Always |
| NVIDIA GPU drivers | GPU inference |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Passing the GPU into Docker |

> **CPU-only machines:** The service starts and works without a GPU — PyTorch falls back to CPU automatically. You just won't get GPU-accelerated performance.

---

## Quick Start (Local Testing)

```bash
# 1. Clone the repo
git clone <repo-url> && cd gpu-service

# 2. Create your .env file (defaults are fine for local dev)
cp .env.example .env

# 3. Build and start the service
docker compose up --build
```

The service will be available at `http://localhost:8000`.

---

## Configuration

All configuration is done via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_IPS` | *(empty — allow all)* | Comma-separated list of IPs allowed to access the service. **Set this in production.** |
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace model to load for embeddings. |

---

## API Endpoints

### `GET /health`

Health check — useful for monitoring and load balancers.

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok", "gpu_available": true}
```

### `POST /v1/embeddings`

Compute text embeddings (OpenAI-compatible format).

**Single text:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
```

**Multiple texts:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello, world!", "Hei maailma!"]}'
```

Response:

```json
{
  "data": [
    {"embedding": [0.0123, -0.0456, ...]},
    {"embedding": [0.0789, -0.0012, ...]}
  ]
}
```

---

## Production / UpCloud Operations

### Setting up IP restriction

Edit your `.env` file on the server:

```bash
# Only allow your Node.js backend's IP
ALLOWED_IPS=10.0.0.5
```

Multiple IPs:

```bash
ALLOWED_IPS=10.0.0.5,10.0.0.6,192.168.1.100
```

> With `ALLOWED_IPS` set, requests from any other IP get a `403 Forbidden` response. Localhost (`127.0.0.1`, `::1`) is always allowed.

### Starting the service

```bash
# SSH into the GPU instance
ssh user@gpu-instance

# Start (or restart) the service
cd /path/to/gpu-service
docker compose up -d --build
```

The service will auto-restart on crashes (`restart: unless-stopped`).

### Stopping the service (to save GPU costs)

```bash
# Gracefully stop the service — frees GPU memory
docker compose down
```

> **Cost note:** GPU instances are billed while running. Always `docker compose down` when the service is not needed to avoid unnecessary charges.

### Checking logs

```bash
# Follow live logs
docker compose logs -f gpu-service

# Last 100 lines
docker compose logs --tail=100 gpu-service
```

---

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI app, IP middleware, routes
│   ├── ml.py             # Model loading & inference
│   └── requirements.txt  # Pinned Python dependencies
├── .env.example          # Environment variable template
├── Dockerfile            # NVIDIA CUDA-based container
├── docker-compose.yml    # GPU passthrough & orchestration
└── README.md             # This file
```

---

## Security Notes

- **IP whitelisting** is the primary access control. The service is designed to sit on a private network, called only by your backend.
- There is **no authentication layer** (API keys, tokens) — add one if the service will be exposed to a broader network.
- Always keep `ALLOWED_IPS` configured in production.
