# Tongue Diagnosis

Realtime tongue image analysis using object detection, recognition, and classification.

## Architecture

UV workspace monorepo with three packages:

```
packages/
├── ai/        → tongue-ai       (PyTorch, MONAI, OpenCV, NumPy, Pandas, Label Studio SDK)
├── backend/   → tongue-backend   (FastAPI, Uvicorn, tongue-ai)
└── frontend/  → tongue-frontend  (Streamlit, httpx)
```

The frontend communicates with the backend exclusively via HTTP/SSE — no direct Python imports between frontend and backend/ai.

```
┌──────────┐   HTTP / SSE   ┌──────────┐   Python   ┌──────────┐
│ Frontend │ ──────────────► │ Backend  │ ─────────► │    AI    │
│ Streamlit│                 │ FastAPI  │            │ PyTorch  │
└──────────┘                 └──────────┘            │  MONAI   │
                                                     └──────────┘
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Astral package manager)
- Python 3.11+

## Quick Start

```bash
# Install all workspace packages (CPU PyTorch by default)
uv sync --all-packages

# Run the backend (http://localhost:8000)
uv run uvicorn tongue_backend.app:app --reload --port 8000

# Run the frontend (http://localhost:8501) — in another terminal
uv run streamlit run packages/frontend/src/tongue_frontend/app.py --server.port 8501
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + AI version |
| POST | `/api/detect` | Detect tongue region (upload image) |
| POST | `/api/recognize` | Recognize tongue features (upload image) |
| POST | `/api/classify` | Classify tongue condition (upload image) |
| POST | `/api/analyze` | Full pipeline: detect → recognize → classify |
| POST | `/api/analyze/stream` | Full pipeline with SSE streaming |

```bash
# Example: full analysis
curl -X POST http://localhost:8000/api/analyze -F "file=@tongue.jpg"

# Example: streaming analysis (SSE)
curl -N -X POST http://localhost:8000/api/analyze/stream -F "file=@tongue.jpg"
```

## GPU Support

Replace the PyTorch index URL in the root `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu124"   # CUDA 12.4
explicit = true
```

Then re-sync:

```bash
uv sync --all-packages --reinstall-package torch --reinstall-package torchvision
```

## Label Studio

The AI package includes `label-studio-sdk` for pulling labeled tongue images. Configure your Label Studio instance:

```bash
export LABEL_STUDIO_URL="http://localhost:8080"
export LABEL_STUDIO_API_KEY="your-api-key"
```

## Development

```bash
# Run a single package
uv run --package tongue-ai python -c "import tongue_ai"

# Add a dependency to a package
uv add --package tongue-ai <dependency>
```
