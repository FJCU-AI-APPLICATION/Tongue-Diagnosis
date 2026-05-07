# Tongue Diagnosis

End-to-end POC for AI tongue diagnosis: webcam/upload → PyTorch ResNet50 inference → locked-prompt Gemini doctor comment.

## Architecture

```
┌──────────┐   HTTP      ┌──────────┐   Python    ┌──────────────┐
│ Frontend │ ──────────► │ Backend  │ ──────────► │     AI       │
│ Gradio   │             │ FastAPI  │             │ PyTorch+ADK  │
└──────────┘             └──────────┘             └──────────────┘
```

UV workspace monorepo:

```
packages/
├── ai/        → tongue-ai       (PyTorch, torchvision, huggingface_hub, opencv)
├── backend/   → tongue-backend  (FastAPI, google-adk, pyyaml, tongue-ai)
└── frontend/  → tongue-frontend (gradio>=4, httpx)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Astral package manager)
- Python 3.11+
- A Hugging Face Hub account with read access to the model weights repo (private)
- Google Cloud service-account credentials with Vertex AI access (for Gemini via ADK)

## Quick Start

```bash
# Install all workspace packages (CPU PyTorch by default)
uv sync --all-packages

# Configure environment (copy and edit)
cp packages/backend/.env.example packages/backend/.env
# fill in HF_TOKEN and GOOGLE_APPLICATION_CREDENTIALS

# Run the backend (http://localhost:8000)
export $(cat packages/backend/.env | xargs)
uv run uvicorn tongue_backend.app:app --reload --port 8000

# Run the frontend (http://localhost:7860) — in another terminal
uv run --package tongue-frontend python -m tongue_frontend.app
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/health` | Service status + AI version + registry summary |
| POST   | `/api/analyze` | Full pipeline (multipart upload) |
| GET    | `/api/config/{section}` | Read prompt / llm / registry section |
| PUT    | `/api/config/{section}` | Save section (validates) |
| POST   | `/api/config/{section}/reset` | Restore section from default |
| POST   | `/api/config/registry/reload` | Re-init ONNX/PyTorch sessions |

## GPU Support

Replace the PyTorch index URL in the root `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

Then re-sync:

```bash
uv sync --all-packages --reinstall-package torch --reinstall-package torchvision
```

## Training (optional)

```bash
uv sync --all-packages --extra training

uv run --package tongue-ai python -m tongue_ai.training.train_front \
    --labels-json data/labels.json \
    --img-dir data/images/ \
    --weights-out weights/best_resnet50_front.pth \
    --epochs 15

uv run --package tongue-ai python -m tongue_ai.training.train_sublingual \
    --labels-json data/labels.json \
    --img-dir data/images/ \
    --weights-out weights/best_resnet50_sublingual.pth
```

## Manual smoke test

See [docs/smoke-test.md](docs/smoke-test.md).

## Design + Plan

- [Merge spec (2026-05-07)](docs/superpowers/specs/2026-05-07-merge-amanda-resnet50-design.md)
- [Implementation plan (2026-05-07)](docs/superpowers/plans/2026-05-07-merge-amanda-resnet50.md)
- [POC design (2026-05-06)](docs/superpowers/specs/2026-05-06-tongue-poc-design.md) — superseded sections noted in the merge spec
