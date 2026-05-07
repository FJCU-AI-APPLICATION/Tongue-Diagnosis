# Merge Amanda's ResNet50 Into Tongue-Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a working end-to-end POC where webcam/upload → real PyTorch ResNet50 inference (Amanda's two trained heads, weights pulled from Hugging Face Hub) → locked-prompt Gemini doctor comment, all in one PR on `feat/merge-amanda-resnet50`.

**Architecture:** Three-package monorepo. `tongue-ai` = pure PyTorch inference library with `PyTorchTaskHead`, `WeightSource` (hf:// + local: URIs), `Registry` (YAML with `category_map`), and a `training/` subpackage holding Amanda's cleaned scripts. `tongue-backend` = FastAPI orchestrator with `pipeline.analyze`, three config stores (prompt/llm/registry), an ADK Gemini client, and a `user_message` builder that splits composite-head predictions back into v4-schema bullets. `tongue-frontend` = Gradio four-tab UI replacing Streamlit.

**Tech Stack:** Python 3.11+, FastAPI, Gradio ≥4, PyTorch + torchvision, huggingface_hub, Google ADK Python SDK (`google-adk`), pyyaml, httpx, pytest, pytest-asyncio, respx, uv workspace.

**Spec:** `docs/superpowers/specs/2026-05-07-merge-amanda-resnet50-design.md` (commits `eeb430a` + `f404d2a`).

**Branch:** `feat/merge-amanda-resnet50` (already created, contains the spec).

---

## File Structure

### Created files

```
packages/ai/src/tongue_ai/
├── types.py                   HeadResult, ClassScore, Normalisation, BBox dataclasses
├── preprocessing.py           bgr_to_rgb, resize_letterbox, normalise_imagenet, decode_bgr
├── weights.py                 WeightSource, parse_weights_uri, WeightFetchError
├── task_head.py               PyTorchTaskHead, load_pytorch_head
├── registry.py                Registry, load_registry, validate_registry, RegistryError
└── training/
    ├── __init__.py            (empty)
    ├── dataset.py             TongueDataset (extracted from train_*.py)
    ├── train_front.py         argparse-driven training entry point
    ├── train_sublingual.py    same for the sublingual head
    └── evaluate.py            evaluate(model, loader, target_labels) -> (report, kappa)

packages/backend/
├── prompts/
│   └── system.default.md      locked 大眾版 prompt
├── config/
│   ├── llm.default.yaml       model, temperature, max_tokens, top_p
│   └── registry.default.yaml  detector + heads + category_map
└── src/tongue_backend/
    ├── pipeline.py            analyze(image_bytes) orchestrator
    ├── stores/
    │   ├── __init__.py
    │   ├── paths.py           default/current path constants
    │   ├── prompt_store.py    text I/O + reset
    │   ├── llm_store.py       YAML I/O + validation + reset
    │   └── registry_store.py  YAML I/O + validation + reset + reload
    ├── routes/
    │   └── config.py          /api/config/{section} CRUD + reload
    └── llm/
        ├── __init__.py
        ├── user_message.py    build(heads, category_map) -> zh-TW bullet list
        └── client.py          ADK agent wrapper

packages/backend/.env.example   HF_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, LLM_MODEL

packages/frontend/src/tongue_frontend/
├── api.py                     httpx client (analyze + config)
└── views/
    ├── __init__.py
    ├── analyze.py             tab 1: capture + result panel
    ├── prompt_editor.py       tab 2
    ├── llm_editor.py          tab 3
    └── registry_editor.py     tab 4

docs/smoke-test.md             manual verification script
```

### Modified files

```
packages/ai/pyproject.toml                  + huggingface_hub, pyyaml, [training] extra
packages/ai/src/tongue_ai/detection.py      rewritten as pass-through stub
packages/ai/src/tongue_ai/inference.py      rewritten as run_all(image, registry)
packages/ai/src/tongue_ai/__init__.py       expose public API surface

packages/backend/pyproject.toml             + google-adk, pyyaml
packages/backend/src/tongue_backend/app.py  startup hook + registry on app.state
packages/backend/src/tongue_backend/routes/health.py    + registry status
packages/backend/src/tongue_backend/routes/analyze.py   call pipeline.analyze

packages/frontend/pyproject.toml            gradio>=4, drop streamlit
packages/frontend/src/tongue_frontend/app.py            rewritten as gr.Blocks four tabs

README.md                                   updated quickstart + envs
```

### Deleted files

```
packages/ai/src/tongue_ai/classification.py     stub superseded by registry
packages/ai/src/tongue_ai/recognition.py        stub superseded by registry
packages/frontend/src/tongue_frontend/test.md   unrelated Upwork research
ResNet50測試模型權重檔V1_Amanda/                  handoff folder (after HF upload)
```

### Tests

```
packages/ai/tests/
├── test_preprocessing.py
├── test_weights.py
├── test_task_head.py
├── test_registry.py
└── test_inference.py

packages/backend/tests/
├── conftest.py                fixtures: stub registry, stub LLM client, sample image
├── test_user_message.py
├── test_prompt_store.py
├── test_llm_store.py
├── test_registry_store.py
├── test_pipeline.py
├── test_routes_analyze.py
├── test_routes_config.py
└── test_routes_health.py

packages/frontend/tests/
└── test_api.py                respx mock transport
```

---

## Phase 0 — Verify branch state

### Task 0: Confirm starting state

**Files:** none

- [ ] **Step 1: Confirm branch and recent commits**

Run:
```bash
git branch --show-current
git log --oneline -8
```

Expected: branch is `feat/merge-amanda-resnet50`. Recent commits should include (from newest to oldest):
- `chore: gitignore .claude/...`
- `Add implementation plan for Amanda's ResNet50 merge`
- `Fix commit-sequence note in merge spec`
- `Add merge design spec: Amanda's ResNet50 into Tongue-Diagnosis`
- prior `2026-05-06` design + plan commits + `Decouple frontend...`

- [ ] **Step 2: Confirm spec and existing plan are present**

Run:
```bash
ls docs/superpowers/specs/
ls docs/superpowers/plans/
```

Expected (specs):
```
2026-05-06-tongue-poc-design.md
2026-05-07-merge-amanda-resnet50-design.md
```

Expected (plans):
```
2026-05-06-tongue-poc.md
2026-05-07-merge-amanda-resnet50.md   ← this file
```

- [ ] **Step 3: Confirm Amanda's weight files are still on disk**

Run:
```bash
ls -la "/home/daniel/Tongue-Diagnosis/ResNet50測試模型權重檔V1_Amanda/" | grep '\.pth$'
```

Expected: three `.pth` files (`best_resnet50.pth`, `best_resnet50_front.pth`, `best_resnet50_sublingual.pth`), each ~94 MB.

- [ ] **Step 4: Confirm uv workspace + sync**

Run:
```bash
uv sync --all-packages
```

Expected: succeeds, installs torch/torchvision (CPU), fastapi, streamlit (will be removed later).

---

## Phase 1 — `tongue-ai`: dependencies and low-level modules

### Task 1: Update `tongue-ai` `pyproject.toml`

**Files:**
- Modify: `packages/ai/pyproject.toml`

- [ ] **Step 1: Replace dependencies and add training extra**

Replace the entire `[project]` and add `[project.optional-dependencies]`:

```toml
[project]
name = "tongue-ai"
version = "0.1.0"
description = "AI models for tongue diagnosis: detection, recognition, classification"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "opencv-python>=4.9",
    "torch>=2.2",
    "torchvision>=0.17",
    "monai>=1.3",
    "label-studio-sdk>=1.0",
    "huggingface-hub>=0.20",
    "pyyaml>=6",
]

[project.optional-dependencies]
training = [
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tongue_ai"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

- [ ] **Step 2: Sync workspace**

Run:
```bash
uv sync --all-packages
```

Expected: pulls `huggingface_hub` and `pyyaml`. No errors.

- [ ] **Step 3: Verify huggingface_hub importable**

Run:
```bash
uv run --package tongue-ai python -c "import huggingface_hub, yaml; print(huggingface_hub.__version__, yaml.__version__)"
```

Expected: prints two version strings, no traceback.

- [ ] **Step 4: Commit**

```bash
git add packages/ai/pyproject.toml
git commit -m "chore(ai): add huggingface_hub, pyyaml, [training] extra"
```

### Task 2: Add `tongue_ai.types`

**Files:**
- Create: `packages/ai/src/tongue_ai/types.py`
- Test: `packages/ai/tests/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_types.py`:

```python
"""Tests for tongue_ai.types dataclasses."""
from __future__ import annotations

from tongue_ai.types import BBox, ClassScore, HeadResult, Normalisation


def test_class_score_holds_label_and_score():
    cs = ClassScore(label="淡紅", score=0.78)
    assert cs.label == "淡紅"
    assert cs.score == 0.78


def test_head_result_defaults_error_to_none():
    hr = HeadResult(
        task="front",
        head_type="single",
        predictions=[ClassScore(label="淡紅", score=0.78)],
    )
    assert hr.error is None
    assert hr.predictions[0].label == "淡紅"


def test_head_result_with_error_has_empty_predictions():
    hr = HeadResult(task="front", head_type="single", predictions=[], error="boom")
    assert hr.error == "boom"
    assert hr.predictions == []


def test_normalisation_holds_mean_and_std():
    n = Normalisation(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert n.mean == (0.485, 0.456, 0.406)
    assert n.std == (0.229, 0.224, 0.225)


def test_bbox_holds_coordinates():
    b = BBox(x=10, y=20, w=30, h=40, confidence=0.9)
    assert b.x == 10 and b.y == 20 and b.w == 30 and b.h == 40
    assert b.confidence == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_types.py -v
```

Expected: ImportError ("cannot import name 'BBox' from 'tongue_ai.types'") because the module doesn't exist yet.

- [ ] **Step 3: Implement the types**

Create `packages/ai/src/tongue_ai/types.py`:

```python
"""Shared dataclasses for the tongue_ai package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Normalisation:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


IMAGENET_NORMALISATION = Normalisation(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass(frozen=True)
class ClassScore:
    label: str
    score: float


@dataclass
class HeadResult:
    task: str
    head_type: Literal["single", "multi"]
    predictions: list[ClassScore] = field(default_factory=list)
    error: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_types.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/types.py packages/ai/tests/test_types.py
git commit -m "feat(ai): add types — HeadResult, ClassScore, Normalisation, BBox"
```

### Task 3: Add `tongue_ai.preprocessing`

**Files:**
- Create: `packages/ai/src/tongue_ai/preprocessing.py`
- Test: `packages/ai/tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_preprocessing.py`:

```python
"""Tests for tongue_ai.preprocessing."""
from __future__ import annotations

import numpy as np
import pytest

from tongue_ai.preprocessing import (
    bgr_to_rgb,
    decode_bgr,
    normalise_imagenet,
    resize_to,
)
from tongue_ai.types import IMAGENET_NORMALISATION


def test_bgr_to_rgb_swaps_channels():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0, 0] == 30  # R first
    assert rgb[0, 0, 1] == 20  # G
    assert rgb[0, 0, 2] == 10  # B


def test_bgr_to_rgb_round_trip_is_identity():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    assert np.array_equal(bgr, bgr_to_rgb(bgr_to_rgb(bgr)))


def test_resize_to_changes_shape():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_to(rgb, (224, 224))
    assert out.shape == (224, 224, 3)


def test_normalise_imagenet_returns_tensor_with_expected_stats():
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)  # mid-grey
    tensor = normalise_imagenet(rgb, IMAGENET_NORMALISATION)
    # Shape: CHW float32
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype.name == "float32"
    # 128/255 = 0.502; (0.502 - 0.485)/0.229 ≈ 0.074 for R channel
    assert tensor[0].mean() == pytest.approx(0.074, abs=0.01)


def test_decode_bgr_reads_jpeg_bytes(tmp_path):
    import cv2
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)
    decoded = decode_bgr(p.read_bytes())
    assert decoded.shape == (10, 10, 3)
    assert decoded.dtype == np.uint8


def test_decode_bgr_raises_on_garbage():
    with pytest.raises(ValueError, match="decode"):
        decode_bgr(b"not an image")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_preprocessing.py -v
```

Expected: ImportError, module not found.

- [ ] **Step 3: Implement preprocessing**

Create `packages/ai/src/tongue_ai/preprocessing.py`:

```python
"""Image preprocessing for the inference pipeline."""
from __future__ import annotations

import cv2
import numpy as np

from tongue_ai.types import Normalisation


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def normalise_imagenet(image_rgb: np.ndarray, n: Normalisation) -> np.ndarray:
    """RGB uint8 HWC → float32 CHW, ImageNet-normalised."""
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - np.array(n.mean, dtype=np.float32)) / np.array(n.std, dtype=np.float32)
    return arr.transpose(2, 0, 1).copy()  # HWC -> CHW
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_preprocessing.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/preprocessing.py packages/ai/tests/test_preprocessing.py
git commit -m "feat(ai): add preprocessing — decode, bgr/rgb, resize, ImageNet normalise"
```

### Task 4: Rewrite `tongue_ai.detection` as a pass-through stub

**Files:**
- Modify: `packages/ai/src/tongue_ai/detection.py`
- Test: `packages/ai/tests/test_detection.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_detection.py`:

```python
"""Tests for tongue_ai.detection (pass-through stub)."""
from __future__ import annotations

import numpy as np

from tongue_ai.detection import detect_tongue
from tongue_ai.types import BBox


def test_detect_tongue_returns_whole_image_bbox_when_disabled():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = detect_tongue(img, enabled=False)
    assert bbox == BBox(x=0, y=0, w=640, h=480, confidence=1.0)


def test_detect_tongue_returns_whole_image_bbox_when_no_detector_arg():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = detect_tongue(img)
    assert bbox == BBox(x=0, y=0, w=200, h=100, confidence=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_detection.py -v
```

Expected: AssertionError or ImportError; current `detection.py` returns a dict, not a `BBox`.

- [ ] **Step 3: Replace `detection.py`**

Replace `packages/ai/src/tongue_ai/detection.py` with:

```python
"""Tongue ROI detection — currently a pass-through stub.

The trained ResNet50 classifiers accept whole tongue photos, so the detector
is disabled by default. This stub returns a bbox covering the entire image.
"""
from __future__ import annotations

import numpy as np

from tongue_ai.types import BBox


def detect_tongue(image: np.ndarray, *, enabled: bool = False) -> BBox:
    h, w = image.shape[:2]
    return BBox(x=0, y=0, w=w, h=h, confidence=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_detection.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/detection.py packages/ai/tests/test_detection.py
git commit -m "feat(ai): rewrite detection as a pass-through BBox stub"
```

### Task 5: Add `tongue_ai.weights` (WeightSource + URI parsing)

**Files:**
- Create: `packages/ai/src/tongue_ai/weights.py`
- Test: `packages/ai/tests/test_weights.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_weights.py`:

```python
"""Tests for tongue_ai.weights — URI parsing + resolution."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tongue_ai.weights import (
    WeightFetchError,
    WeightSource,
    parse_weights_uri,
)


def test_parse_local_uri_returns_local_scheme_and_path():
    scheme, ref = parse_weights_uri("local:packages/ai/models/best.pth")
    assert scheme == "local"
    assert ref == "packages/ai/models/best.pth"


def test_parse_hf_uri_returns_hf_scheme_and_repo_filename():
    scheme, ref = parse_weights_uri("hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth")
    assert scheme == "hf"
    assert ref == "CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth"


def test_parse_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        parse_weights_uri("s3:bucket/x.pth")


def test_parse_uri_without_scheme_raises():
    with pytest.raises(ValueError, match="missing scheme"):
        parse_weights_uri("packages/ai/models/x.pth")


def test_local_uri_resolves_relative_to_base_dir(tmp_path):
    base = tmp_path / "registry-dir"
    weights_file = base / "models" / "x.pth"
    weights_file.parent.mkdir(parents=True)
    weights_file.write_bytes(b"weights")
    src = WeightSource(uri="local:models/x.pth", base_dir=base)
    resolved = src.resolve()
    assert resolved == weights_file


def test_local_uri_missing_file_raises_weight_fetch_error(tmp_path):
    src = WeightSource(uri="local:nope.pth", base_dir=tmp_path)
    with pytest.raises(WeightFetchError, match="not found"):
        src.resolve()


def test_hf_uri_calls_hf_hub_download_with_repo_and_filename(tmp_path):
    fake_path = tmp_path / "best_resnet50_front.pth"
    fake_path.write_bytes(b"weights")
    src = WeightSource(
        uri="hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth",
        base_dir=tmp_path,
    )
    with patch("tongue_ai.weights.hf_hub_download", return_value=str(fake_path)) as m:
        resolved = src.resolve()
    m.assert_called_once_with(
        repo_id="CallMeDaniel/tongue-resnet50-v1",
        filename="best_resnet50_front.pth",
        token=None,
    )
    assert resolved == fake_path


def test_hf_uri_passes_token_from_env(tmp_path, monkeypatch):
    fake_path = tmp_path / "x.pth"
    fake_path.write_bytes(b"")
    monkeypatch.setenv("HF_TOKEN", "secret")
    src = WeightSource(uri="hf:org/repo/x.pth", base_dir=tmp_path)
    with patch("tongue_ai.weights.hf_hub_download", return_value=str(fake_path)) as m:
        src.resolve()
    assert m.call_args.kwargs["token"] == "secret"


def test_hf_uri_wraps_underlying_error(tmp_path):
    src = WeightSource(uri="hf:org/repo/x.pth", base_dir=tmp_path)
    with patch("tongue_ai.weights.hf_hub_download", side_effect=ConnectionError("boom")):
        with pytest.raises(WeightFetchError) as excinfo:
            src.resolve()
    assert "org/repo" in str(excinfo.value)
    assert "x.pth" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_weights.py -v
```

Expected: ImportError, module not found.

- [ ] **Step 3: Implement weights.py**

Create `packages/ai/src/tongue_ai/weights.py`:

```python
"""Resolve weight-file URIs.

Two URI schemes:
  local:relative/path/to/file.pth   → resolved relative to base_dir
  hf:repo_owner/repo_name/filename  → fetched from Hugging Face Hub

`HF_TOKEN` env var is used for private repos when present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


class WeightFetchError(RuntimeError):
    """Raised when a weight URI cannot be resolved to a readable local file."""


def parse_weights_uri(uri: str) -> tuple[str, str]:
    if ":" not in uri:
        raise ValueError(f"missing scheme in weights uri: {uri!r}")
    scheme, _, ref = uri.partition(":")
    if scheme not in {"local", "hf"}:
        raise ValueError(f"unknown scheme {scheme!r} in weights uri: {uri!r}")
    return scheme, ref


@dataclass(frozen=True)
class WeightSource:
    uri: str
    base_dir: Path

    def resolve(self) -> Path:
        scheme, ref = parse_weights_uri(self.uri)
        if scheme == "local":
            path = (self.base_dir / ref).resolve()
            if not path.is_file():
                raise WeightFetchError(
                    f"local weights not found: {path} (uri={self.uri!r})"
                )
            return path
        if scheme == "hf":
            owner, name, *rest = ref.split("/")
            if not rest:
                raise WeightFetchError(
                    f"hf uri must include filename: {self.uri!r}"
                )
            repo_id = f"{owner}/{name}"
            filename = "/".join(rest)
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as exc:
                raise WeightFetchError(
                    f"failed to download {repo_id}/{filename}: {exc}"
                ) from exc
            return Path(downloaded)
        raise WeightFetchError(f"unhandled scheme {scheme!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_weights.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/weights.py packages/ai/tests/test_weights.py
git commit -m "feat(ai): add WeightSource — hf:// and local: URI resolution"
```

---

## Phase 2 — `tongue-ai`: TaskHead, Registry, Inference

### Task 6: Add `tongue_ai.task_head.PyTorchTaskHead`

**Files:**
- Create: `packages/ai/src/tongue_ai/task_head.py`
- Test: `packages/ai/tests/test_task_head.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_task_head.py`:

```python
"""Tests for PyTorchTaskHead — single-label decoding from a fake fc layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import IMAGENET_NORMALISATION, ClassScore, HeadResult


def _fake_resnet_state_dict_for_classes(num_classes: int) -> dict[str, torch.Tensor]:
    """Build a state_dict matching torchvision.models.resnet50 with fc[2048,N]."""
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


def test_load_pytorch_head_accepts_matching_state_dict(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    assert isinstance(head, PyTorchTaskHead)
    assert head.name == "test"
    assert head.class_names == ["a", "b", "c", "d"]


def test_load_pytorch_head_size_mismatch_raises(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(3)  # 3 classes
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],  # 4 names != 3 weight classes
    }
    with pytest.raises(RuntimeError, match="state_dict|size mismatch"):
        load_pytorch_head(cfg, weight_path)


def test_predict_single_returns_top1_with_softmax_score(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    # Force fc bias so class index 2 wins regardless of input
    state["fc.bias"] = torch.tensor([-10.0, -10.0, 100.0, -10.0])
    state["fc.weight"] = torch.zeros((4, 2048))
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)

    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    img = np.full((300, 400, 3), 128, dtype=np.uint8)  # BGR mid-grey
    result = head.predict(img)

    assert isinstance(result, HeadResult)
    assert result.task == "test"
    assert result.head_type == "single"
    assert result.error is None
    assert len(result.predictions) == 1
    assert result.predictions[0].label == "c"
    assert result.predictions[0].score == pytest.approx(1.0, abs=1e-3)


def test_predict_records_error_in_head_result_on_failure(tmp_path: Path):
    state = _fake_resnet_state_dict_for_classes(4)
    weight_path = tmp_path / "fake.pth"
    torch.save(state, weight_path)
    cfg = {
        "name": "test",
        "head_type": "single",
        "arch": "resnet50",
        "input_size": (224, 224),
        "normalise": IMAGENET_NORMALISATION,
        "class_names": ["a", "b", "c", "d"],
    }
    head = load_pytorch_head(cfg, weight_path)
    bad = np.zeros((0, 0, 3), dtype=np.uint8)  # zero-size image
    result = head.predict(bad)
    assert result.error is not None
    assert result.predictions == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_task_head.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement task_head.py**

Create `packages/ai/src/tongue_ai/task_head.py`:

```python
"""PyTorchTaskHead: a single ResNet50 classifier wrapped with preprocessing + decode."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from tongue_ai.preprocessing import bgr_to_rgb, normalise_imagenet, resize_to
from tongue_ai.types import ClassScore, HeadResult, Normalisation


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PyTorchTaskHead:
    name: str
    head_type: Literal["single", "multi"]
    arch: Literal["resnet50"]
    class_names: list[str]
    input_size: tuple[int, int]
    normalise: Normalisation
    threshold: float = 0.5
    device: torch.device = field(default_factory=_autodetect_device)
    model: nn.Module | None = None  # set by factory

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        try:
            rgb = bgr_to_rgb(image_bgr)
            resized = resize_to(rgb, self.input_size)
            arr = normalise_imagenet(resized, self.normalise)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            assert self.model is not None
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        except Exception as exc:
            return HeadResult(
                task=self.name,
                head_type=self.head_type,
                predictions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        if self.head_type == "single":
            idx = int(np.argmax(probs))
            return HeadResult(
                task=self.name,
                head_type="single",
                predictions=[ClassScore(label=self.class_names[idx], score=float(probs[idx]))],
            )
        # multi-label
        kept = [
            ClassScore(label=self.class_names[i], score=float(p))
            for i, p in enumerate(probs)
            if p >= self.threshold
        ]
        kept.sort(key=lambda c: c.score, reverse=True)
        return HeadResult(task=self.name, head_type="multi", predictions=kept)


def load_pytorch_head(cfg: dict[str, Any], weight_path: Path) -> PyTorchTaskHead:
    if cfg["arch"] != "resnet50":
        raise ValueError(f"unsupported arch: {cfg['arch']!r}")
    num_classes = len(cfg["class_names"])
    device = _autodetect_device()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    head = PyTorchTaskHead(
        name=cfg["name"],
        head_type=cfg["head_type"],
        arch=cfg["arch"],
        class_names=list(cfg["class_names"]),
        input_size=tuple(cfg["input_size"]),
        normalise=cfg["normalise"],
        threshold=cfg.get("threshold", 0.5),
        device=device,
        model=model,
    )
    return head
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_task_head.py -v
```

Expected: 4 passed. (Note: builds two real resnet50 instances; expect ~10-30s on CPU.)

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/task_head.py packages/ai/tests/test_task_head.py
git commit -m "feat(ai): add PyTorchTaskHead with resnet50 factory and softmax decode"
```

### Task 7: Add `tongue_ai.registry`

**Files:**
- Create: `packages/ai/src/tongue_ai/registry.py`
- Test: `packages/ai/tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_registry.py`:

```python
"""Tests for the YAML registry loader and validator."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tongue_ai.registry import (
    Registry,
    RegistryError,
    load_registry,
    validate_registry_yaml,
)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _fake_resnet50_state_dict(num_classes: int):
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


VALID_YAML = """
detector:
  enabled: false

heads:
  - name: front
    head_type: single
    arch: resnet50
    weights_uri: local:weights/front.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b, c]

  - name: sublingual
    head_type: single
    arch: resnet50
    weights_uri: local:weights/sub.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [x, y]

category_map:
  front:
    a: 舌色
    b: 舌色
    c: 舌質
  sublingual:
    x: 舌下絡脈
    y: 舌下絡脈
"""


def test_validate_yaml_accepts_valid(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    # No raise
    validate_registry_yaml(p)


def test_validate_yaml_rejects_missing_required_field(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    bad = VALID_YAML.replace("    head_type: single", "")
    _write_yaml(p, bad)
    with pytest.raises(RegistryError, match="head_type"):
        validate_registry_yaml(p)


def test_validate_yaml_rejects_class_name_missing_from_category_map(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    # Drop "c: 舌質" → class c has no v4 mapping
    bad = VALID_YAML.replace("    c: 舌質\n", "")
    _write_yaml(p, bad)
    with pytest.raises(RegistryError, match="category_map.*missing"):
        validate_registry_yaml(p)


def test_load_registry_returns_registry_with_two_heads(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save(_fake_resnet50_state_dict(3), weights_dir / "front.pth")
    torch.save(_fake_resnet50_state_dict(2), weights_dir / "sub.pth")

    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    reg = load_registry(p)
    assert isinstance(reg, Registry)
    assert [h.name for h in reg.heads] == ["front", "sublingual"]
    assert reg.detector_enabled is False
    assert reg.category_map["front"]["a"] == "舌色"


def test_load_registry_partial_failure_records_per_head_error(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    # Only front weights present; sub.pth missing
    torch.save(_fake_resnet50_state_dict(3), weights_dir / "front.pth")
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)

    reg = load_registry(p, raise_on_partial_fail=False)
    assert [h.name for h in reg.heads] == ["front"]
    assert "sublingual" in reg.failed_heads
    assert "not found" in reg.failed_heads["sublingual"]


def test_load_registry_total_failure_raises_when_strict(tmp_path: Path):
    p = tmp_path / "registry.yaml"
    _write_yaml(p, VALID_YAML)
    with pytest.raises(RegistryError, match="zero heads"):
        load_registry(p, raise_on_partial_fail=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_registry.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement registry.py**

Create `packages/ai/src/tongue_ai/registry.py`:

```python
"""Registry: YAML loader + validator for the inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import IMAGENET_NORMALISATION, Normalisation
from tongue_ai.weights import WeightFetchError, WeightSource


REQUIRED_HEAD_FIELDS = {
    "name",
    "head_type",
    "arch",
    "weights_uri",
    "input_size",
    "normalise",
    "class_names",
}

_NORMALISATIONS: dict[str, Normalisation] = {
    "imagenet": IMAGENET_NORMALISATION,
}


class RegistryError(RuntimeError):
    """Raised on registry validation or load failures."""


@dataclass
class Registry:
    detector_enabled: bool
    heads: list[PyTorchTaskHead]
    category_map: dict[str, dict[str, str]]
    failed_heads: dict[str, str] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"yaml parse error: {exc}") from exc


def validate_registry_yaml(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    if "heads" not in data or not isinstance(data["heads"], list) or not data["heads"]:
        raise RegistryError("registry must contain a non-empty 'heads' list")
    cmap = data.get("category_map") or {}
    if not isinstance(cmap, dict):
        raise RegistryError("'category_map' must be a mapping if provided")

    for head in data["heads"]:
        missing = REQUIRED_HEAD_FIELDS - set(head)
        if missing:
            raise RegistryError(
                f"head {head.get('name', '?')!r} missing required fields: {sorted(missing)}"
            )
        if head["normalise"] not in _NORMALISATIONS:
            raise RegistryError(
                f"head {head['name']!r}: unknown normalise key {head['normalise']!r}"
            )
        head_map = cmap.get(head["name"], {})
        missing_classes = [c for c in head["class_names"] if c not in head_map]
        if missing_classes:
            raise RegistryError(
                f"category_map missing entries for head {head['name']!r}: {missing_classes}"
            )
    return data


def load_registry(path: Path, *, raise_on_partial_fail: bool = True) -> Registry:
    data = validate_registry_yaml(path)
    base_dir = path.parent

    loaded: list[PyTorchTaskHead] = []
    failed: dict[str, str] = {}

    for cfg in data["heads"]:
        full_cfg = {
            **cfg,
            "normalise": _NORMALISATIONS[cfg["normalise"]],
        }
        try:
            src = WeightSource(uri=cfg["weights_uri"], base_dir=base_dir)
            weight_path = src.resolve()
            head = load_pytorch_head(full_cfg, weight_path)
            loaded.append(head)
        except (WeightFetchError, RuntimeError, ValueError) as exc:
            failed[cfg["name"]] = f"{type(exc).__name__}: {exc}"

    if not loaded and raise_on_partial_fail:
        details = "; ".join(f"{n}: {e}" for n, e in failed.items()) or "no heads configured"
        raise RegistryError(f"zero heads loaded — {details}")

    detector_enabled = bool((data.get("detector") or {}).get("enabled", False))

    return Registry(
        detector_enabled=detector_enabled,
        heads=loaded,
        category_map=data.get("category_map", {}),
        failed_heads=failed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_registry.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/registry.py packages/ai/tests/test_registry.py
git commit -m "feat(ai): add Registry loader with category_map validation and partial-fail support"
```

### Task 8: Rewrite `tongue_ai.inference` as `run_all`

**Files:**
- Modify: `packages/ai/src/tongue_ai/inference.py`
- Test: `packages/ai/tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_inference.py`:

```python
"""Tests for run_all — iterate registry heads, collect HeadResults."""
from __future__ import annotations

import numpy as np

from tongue_ai.inference import run_all
from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult


class _FakeHead:
    def __init__(self, name: str, return_value: HeadResult):
        self.name = name
        self._rv = return_value
        self.calls = 0

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        self.calls += 1
        return self._rv


def _make_registry(heads):
    return Registry(detector_enabled=False, heads=heads, category_map={})


def test_run_all_returns_one_result_per_head():
    h1 = _FakeHead("front", HeadResult("front", "single", [ClassScore("a", 0.9)]))
    h2 = _FakeHead("sublingual", HeadResult("sublingual", "single", [ClassScore("x", 0.7)]))
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    results = run_all(img, _make_registry([h1, h2]))
    assert [r.task for r in results] == ["front", "sublingual"]
    assert h1.calls == 1 and h2.calls == 1


def test_run_all_preserves_registry_order():
    h1 = _FakeHead("a", HeadResult("a", "single", []))
    h2 = _FakeHead("b", HeadResult("b", "single", []))
    h3 = _FakeHead("c", HeadResult("c", "single", []))
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    results = run_all(img, _make_registry([h3, h1, h2]))
    assert [r.task for r in results] == ["c", "a", "b"]


def test_run_all_continues_after_a_head_raises():
    class _Boom:
        name = "boom"
        def predict(self, _img):
            raise RuntimeError("kaboom")
    h_ok = _FakeHead("ok", HeadResult("ok", "single", [ClassScore("z", 0.5)]))
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    results = run_all(img, _make_registry([_Boom(), h_ok]))

    assert len(results) == 2
    assert results[0].task == "boom"
    assert results[0].error is not None
    assert "kaboom" in results[0].error
    assert results[1].task == "ok"
    assert results[1].error is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_inference.py -v
```

Expected: ImportError or AttributeError on `run_all`.

- [ ] **Step 3: Replace inference.py**

Replace `packages/ai/src/tongue_ai/inference.py` with:

```python
"""run_all: iterate registry heads and collect HeadResults."""
from __future__ import annotations

import numpy as np

from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult


def run_all(image_bgr: np.ndarray, registry: Registry) -> list[HeadResult]:
    results: list[HeadResult] = []
    for head in registry.heads:
        try:
            results.append(head.predict(image_bgr))
        except Exception as exc:
            results.append(
                HeadResult(
                    task=getattr(head, "name", "?"),
                    head_type=getattr(head, "head_type", "single"),
                    predictions=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_inference.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/inference.py packages/ai/tests/test_inference.py
git commit -m "feat(ai): rewrite inference.run_all to iterate registry heads"
```

### Task 9: Delete obsolete stubs and update `__init__.py`

**Files:**
- Delete: `packages/ai/src/tongue_ai/classification.py`
- Delete: `packages/ai/src/tongue_ai/recognition.py`
- Modify: `packages/ai/src/tongue_ai/__init__.py`

- [ ] **Step 1: Delete stub modules**

```bash
rm packages/ai/src/tongue_ai/classification.py
rm packages/ai/src/tongue_ai/recognition.py
```

- [ ] **Step 2: Update `__init__.py` to expose the public API**

Replace `packages/ai/src/tongue_ai/__init__.py` with:

```python
"""Tongue AI — PyTorch ResNet50 inference for tongue diagnosis."""
from tongue_ai.detection import detect_tongue
from tongue_ai.inference import run_all
from tongue_ai.registry import Registry, RegistryError, load_registry
from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import (
    BBox,
    ClassScore,
    HeadResult,
    IMAGENET_NORMALISATION,
    Normalisation,
)
from tongue_ai.weights import WeightFetchError, WeightSource

__version__ = "0.2.0"

__all__ = [
    "BBox",
    "ClassScore",
    "HeadResult",
    "IMAGENET_NORMALISATION",
    "Normalisation",
    "PyTorchTaskHead",
    "Registry",
    "RegistryError",
    "WeightFetchError",
    "WeightSource",
    "__version__",
    "detect_tongue",
    "load_pytorch_head",
    "load_registry",
    "run_all",
]
```

- [ ] **Step 3: Verify package still imports cleanly**

```bash
uv run --package tongue-ai python -c "import tongue_ai; print(tongue_ai.__version__, tongue_ai.__all__)"
```

Expected: `0.2.0 ['BBox', 'ClassScore', ...]` printed; no traceback.

- [ ] **Step 4: Run all AI tests to confirm green**

```bash
uv run --package tongue-ai pytest packages/ai/tests -v
```

Expected: all tests from Tasks 2–8 pass.

- [ ] **Step 5: Commit**

```bash
git add -u packages/ai/src/tongue_ai/__init__.py packages/ai/src/tongue_ai/classification.py packages/ai/src/tongue_ai/recognition.py
git commit -m "chore(ai): drop classification/recognition stubs, expose public API"
```

---

## Phase 3 — `tongue-ai`: training subpackage

### Task 10: Add `tongue_ai.training.dataset`

**Files:**
- Create: `packages/ai/src/tongue_ai/training/__init__.py`
- Create: `packages/ai/src/tongue_ai/training/dataset.py`
- Test: `packages/ai/tests/test_training_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `packages/ai/tests/test_training_dataset.py`:

```python
"""Tests for TongueDataset (extracted from Amanda's train_*.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tongue_ai.training.dataset import TongueDataset


def _make_label_studio_record(image_filename: str, label: str) -> dict:
    return {
        "data": {"image": f"/data/upload/{image_filename}"},
        "annotations": [
            {"result": [{"value": {"choices": [label]}}]}
        ],
    }


def test_dataset_yields_image_and_label_idx(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img_dir / "a.png")
    data = [_make_label_studio_record("a.png", "淡紅")]
    labels_map = {"淡紅": 0}

    ds = TongueDataset(data, str(img_dir), labels_map, transform=None)
    image, label_idx = ds[0]
    assert label_idx == 0
    assert image.size == (16, 16)


def test_dataset_returns_blank_image_on_missing_file(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    data = [_make_label_studio_record("missing.png", "淡")]
    labels_map = {"淡": 0}
    ds = TongueDataset(data, str(img_dir), labels_map, transform=None)
    image, _ = ds[0]
    assert image.size == (224, 224)  # blank fallback per Amanda's behaviour


def test_dataset_len_matches_data_list_length(tmp_path: Path):
    data = [_make_label_studio_record("x.png", "a")] * 5
    ds = TongueDataset(data, str(tmp_path), {"a": 0}, transform=None)
    assert len(ds) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_training_dataset.py -v
```

Expected: ImportError; module doesn't exist.

- [ ] **Step 3: Create training subpackage**

Create `packages/ai/src/tongue_ai/training/__init__.py`:

```python
"""Training subpackage for tongue_ai (label-studio JSON → ResNet50 weights)."""
```

Create `packages/ai/src/tongue_ai/training/dataset.py`:

```python
"""Shared TongueDataset for the train_front and train_sublingual entry points."""
from __future__ import annotations

import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class TongueDataset(Dataset):
    """Label Studio JSON record → (PIL.Image, label_idx).

    Records expected to look like:
      {"data": {"image": ".../filename.png"},
       "annotations": [{"result": [{"value": {"choices": ["label"]}}]}]}
    """

    def __init__(
        self,
        data_list: list[dict[str, Any]],
        img_dir: str,
        labels_map: dict[str, int],
        transform=None,
    ):
        self.data_list = data_list
        self.img_dir = img_dir
        self.labels_map = labels_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        img_path_raw = item["data"]["image"]
        img_filename = os.path.basename(img_path_raw)
        img_path = os.path.join(self.img_dir, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224))

        label_str = item["annotations"][0]["result"][0]["value"]["choices"][0]
        label_idx = self.labels_map[label_str]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-ai pytest packages/ai/tests/test_training_dataset.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/training/__init__.py packages/ai/src/tongue_ai/training/dataset.py packages/ai/tests/test_training_dataset.py
git commit -m "feat(ai): add training subpackage with shared TongueDataset"
```

### Task 11: Add `tongue_ai.training.evaluate`

**Files:**
- Create: `packages/ai/src/tongue_ai/training/evaluate.py`

- [ ] **Step 1: Implement evaluate.py**

Create `packages/ai/src/tongue_ai/training/evaluate.py`:

```python
"""Evaluation: classification report + Cohen's Kappa.

Imports scikit-learn lazily so the runtime AI package doesn't pull it in.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    target_labels: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """Run model.eval() over loader and return (report_str, kappa)."""
    from sklearn.metrics import classification_report, cohen_kappa_score

    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist() if hasattr(labels, "numpy") else list(labels))

    report = classification_report(
        all_targets,
        all_preds,
        target_names=target_labels,
        zero_division=0,
    )
    kappa = cohen_kappa_score(all_targets, all_preds)
    return report, float(kappa)
```

- [ ] **Step 2: Verify import works (with training extra)**

```bash
uv sync --all-packages --extra training
uv run --package tongue-ai python -c "from tongue_ai.training.evaluate import evaluate; print('ok')"
```

Expected: `ok` printed.

- [ ] **Step 3: Commit**

```bash
git add packages/ai/src/tongue_ai/training/evaluate.py
git commit -m "feat(ai/training): add evaluate(model, loader, labels) — report + kappa"
```

### Task 12: Add `tongue_ai.training.train_front`

**Files:**
- Create: `packages/ai/src/tongue_ai/training/train_front.py`

- [ ] **Step 1: Implement train_front.py**

Create `packages/ai/src/tongue_ai/training/train_front.py`:

```python
"""Train the 'front' composite head (14 classes from Amanda's schema)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

from tongue_ai.training.dataset import TongueDataset
from tongue_ai.training.evaluate import evaluate


FRONT_LABELS = [
    "淡紅", "紅", "淡", "絳", "青紫", "暗", "微紅",
    "胖大", "瘦薄", "嫩", "偏斜", "齒痕",
    "無異常", "瘀血絲",
]


def _configure_matplotlib_fonts() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "PingFang TC",
        "Heiti TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model A (front composite head).")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--weights-out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--curve-out", type=Path, default=None)
    args = parser.parse_args()

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading labels from {args.labels_json}")
    full_data = json.loads(args.labels_json.read_text(encoding="utf-8"))

    labels_map = {label: i for i, label in enumerate(FRONT_LABELS)}

    valid_data = []
    for item in full_data:
        anns = item.get("annotations") or []
        if not anns:
            continue
        results = anns[0].get("result") or []
        if not results:
            continue
        choices = results[0].get("value", {}).get("choices") or []
        if not choices:
            continue
        if choices[0] in FRONT_LABELS:
            valid_data.append(item)

    print(f"Filtered {len(valid_data)} front-head records.")
    train_data, val_data = train_test_split(valid_data, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TongueDataset(train_data, str(args.img_dir), labels_map, train_transform)
    val_ds = TongueDataset(val_data, str(args.img_dir), labels_map, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = _autodetect_device()
    print(f"Device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(FRONT_LABELS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_ds)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_acc = 100.0 * correct / total if total else 0.0

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(epoch_acc)
        print(f"Epoch {epoch+1:02d}/{args.epochs} - Loss {epoch_loss:.4f} | ValAcc {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), args.weights_out)
            print(f"  saved best to {args.weights_out}")

    if args.curve_out is not None:
        _configure_matplotlib_fonts()
        import matplotlib.pyplot as plt

        epochs_range = range(1, args.epochs + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Train Loss", color="tab:red")
        ax1.plot(epochs_range, history["train_loss"], color="tab:red", marker="o")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Val Accuracy (%)", color="tab:blue")
        ax2.plot(epochs_range, history["val_acc"], color="tab:blue", marker="s")
        ax2.set_ylim(0, 100)
        plt.title("Front Tongue Features Learning Curve")
        fig.tight_layout()
        plt.savefig(args.curve_out)
        print(f"Curve saved to {args.curve_out}")

    model.load_state_dict(torch.load(args.weights_out, map_location=device))
    report, kappa = evaluate(model, val_loader, FRONT_LABELS, device)
    print("\n=== Classification Report ===")
    print(report)
    print(f"Cohen's Kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify entry point parses args**

```bash
uv run --package tongue-ai --extra training python -m tongue_ai.training.train_front --help
```

Expected: argparse usage banner with `--labels-json`, `--img-dir`, `--weights-out`, etc.

- [ ] **Step 3: Commit**

```bash
git add packages/ai/src/tongue_ai/training/train_front.py
git commit -m "feat(ai/training): add train_front entry point — argparse, font fallback, evaluate"
```

### Task 13: Add `tongue_ai.training.train_sublingual`

**Files:**
- Create: `packages/ai/src/tongue_ai/training/train_sublingual.py`

- [ ] **Step 1: Implement train_sublingual.py**

Create `packages/ai/src/tongue_ai/training/train_sublingual.py`:

```python
"""Train the 'sublingual' head (3 classes for sublingual veins)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

from tongue_ai.training.dataset import TongueDataset
from tongue_ai.training.evaluate import evaluate


SUBLINGUAL_LABELS = ["怒張", "曲張", "囊柱囊泡"]


def _autodetect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model B (sublingual head).")
    parser.add_argument("--labels-json", required=True, type=Path)
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--weights-out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    full_data = json.loads(args.labels_json.read_text(encoding="utf-8"))
    labels_map = {label: i for i, label in enumerate(SUBLINGUAL_LABELS)}

    valid_data = []
    for item in full_data:
        anns = item.get("annotations") or []
        if not anns:
            continue
        results = anns[0].get("result") or []
        if not results:
            continue
        choices = results[0].get("value", {}).get("choices") or []
        if not choices:
            continue
        if choices[0] in SUBLINGUAL_LABELS:
            valid_data.append(item)

    print(f"Filtered {len(valid_data)} sublingual records.")
    train_data, val_data = train_test_split(valid_data, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TongueDataset(train_data, str(args.img_dir), labels_map, train_transform)
    val_ds = TongueDataset(val_data, str(args.img_dir), labels_map, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = _autodetect_device()
    print(f"Device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(SUBLINGUAL_LABELS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_ds)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_acc = 100.0 * correct / total if total else 0.0
        print(f"Epoch {epoch+1:02d}/{args.epochs} - Loss {epoch_loss:.4f} | ValAcc {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), args.weights_out)
            print(f"  saved best to {args.weights_out}")

    model.load_state_dict(torch.load(args.weights_out, map_location=device))
    report, kappa = evaluate(model, val_loader, SUBLINGUAL_LABELS, device)
    print("\n=== Classification Report ===")
    print(report)
    print(f"Cohen's Kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify entry point parses args**

```bash
uv run --package tongue-ai --extra training python -m tongue_ai.training.train_sublingual --help
```

Expected: argparse usage banner.

- [ ] **Step 3: Commit**

```bash
git add packages/ai/src/tongue_ai/training/train_sublingual.py
git commit -m "feat(ai/training): add train_sublingual entry point"
```

---

## Phase 4 — Manual: upload weights to Hugging Face Hub

### Task 14: Upload `.pth` weights to HF Hub

This task is performed by the human operator (out-of-band; no commit).

- [ ] **Step 1: Create the HF Hub repo**

Visit https://huggingface.co/new and create a private model repo named `CallMeDaniel/tongue-resnet50-v1`. (Or pick a different name and update `packages/backend/config/registry.default.yaml` later.)

- [ ] **Step 2: Set `HF_TOKEN` locally**

```bash
export HF_TOKEN="hf_xxx_your_personal_access_token"
```

A "Read" or "Write" token works for downloads; "Write" is required for the upload step.

- [ ] **Step 3: Upload the two weight files**

```bash
cd /home/daniel/Tongue-Diagnosis
uv run --package tongue-ai python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='ResNet50測試模型權重檔V1_Amanda/best_resnet50_front.pth',
    path_in_repo='best_resnet50_front.pth',
    repo_id='CallMeDaniel/tongue-resnet50-v1',
    repo_type='model',
)
api.upload_file(
    path_or_fileobj='ResNet50測試模型權重檔V1_Amanda/best_resnet50_sublingual.pth',
    path_in_repo='best_resnet50_sublingual.pth',
    repo_id='CallMeDaniel/tongue-resnet50-v1',
    repo_type='model',
)
print('Both weights uploaded.')
"
```

Expected: `Both weights uploaded.` printed; both files visible at the repo URL.

- [ ] **Step 4: Verify download path resolves**

```bash
uv run --package tongue-ai python -c "
import os
from tongue_ai.weights import WeightSource
from pathlib import Path
src = WeightSource(uri='hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth', base_dir=Path('.'))
print('resolved:', src.resolve())
"
```

Expected: prints a path under `~/.cache/huggingface/hub/...` and the file exists.

(No commit — this task is operational.)

---

## Phase 5 — `tongue-backend`: dependencies, stores, and config defaults

### Task 15: Update `tongue-backend` `pyproject.toml`

**Files:**
- Modify: `packages/backend/pyproject.toml`

- [ ] **Step 1: Add dependencies**

Replace the `[project]` block with:

```toml
[project]
name = "tongue-backend"
version = "0.1.0"
description = "FastAPI backend for tongue diagnosis service"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
    "pyyaml>=6",
    "google-adk>=0.1",
    "tongue-ai",
]
```

- [ ] **Step 2: Sync workspace**

```bash
uv sync --all-packages
```

Expected: pulls `google-adk` and its dependencies. May take a minute.

- [ ] **Step 3: Verify imports**

```bash
uv run --package tongue-backend python -c "import yaml; import google.adk; print('ok')"
```

Expected: `ok` printed. If `google.adk` import fails, note the actual SDK module name (could be `google_adk` or similar) and update the LLM client task accordingly.

- [ ] **Step 4: Commit**

```bash
git add packages/backend/pyproject.toml
git commit -m "chore(backend): add google-adk and pyyaml deps"
```

### Task 16: Add `tongue_backend.stores.paths`

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/__init__.py`
- Create: `packages/backend/src/tongue_backend/stores/paths.py`

- [ ] **Step 1: Implement paths.py**

Create `packages/backend/src/tongue_backend/stores/__init__.py`:

```python
"""Config stores: prompt, llm, registry."""
```

Create `packages/backend/src/tongue_backend/stores/paths.py`:

```python
"""File-system paths for the three editable config sections."""
from __future__ import annotations

from pathlib import Path

# Anchor: this file is at packages/backend/src/tongue_backend/stores/paths.py
# We need packages/backend/{prompts,config}/...
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PROMPT_DEFAULT = _BACKEND_ROOT / "prompts" / "system.default.md"
PROMPT_CURRENT = _BACKEND_ROOT / "prompts" / "system.current.md"

LLM_DEFAULT = _BACKEND_ROOT / "config" / "llm.default.yaml"
LLM_CURRENT = _BACKEND_ROOT / "config" / "llm.current.yaml"

REGISTRY_DEFAULT = _BACKEND_ROOT / "config" / "registry.default.yaml"
REGISTRY_CURRENT = _BACKEND_ROOT / "config" / "registry.current.yaml"
```

- [ ] **Step 2: Verify paths resolve correctly**

```bash
uv run --package tongue-backend python -c "
from tongue_backend.stores import paths
print('prompt default:', paths.PROMPT_DEFAULT)
print('llm default   :', paths.LLM_DEFAULT)
print('reg default   :', paths.REGISTRY_DEFAULT)
"
```

Expected: prints absolute paths ending in `packages/backend/prompts/system.default.md`, `packages/backend/config/llm.default.yaml`, `packages/backend/config/registry.default.yaml`.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/__init__.py packages/backend/src/tongue_backend/stores/paths.py
git commit -m "feat(backend): add stores.paths with default/current locations"
```

### Task 17: Add `tongue_backend.stores.prompt_store`

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/prompt_store.py`
- Test: `packages/backend/tests/test_prompt_store.py`
- Test: `packages/backend/tests/conftest.py` (shared fixtures)

- [ ] **Step 1: Add shared conftest**

Create `packages/backend/tests/conftest.py`:

```python
"""Shared fixtures for backend tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_default_prompt(tmp_path: Path) -> Path:
    p = tmp_path / "system.default.md"
    p.write_text("DEFAULT PROMPT", encoding="utf-8")
    return p


@pytest.fixture
def fake_current_prompt(tmp_path: Path) -> Path:
    return tmp_path / "system.current.md"
```

- [ ] **Step 2: Write failing tests**

Create `packages/backend/tests/test_prompt_store.py`:

```python
"""Tests for PromptStore: load_current, save, reset, first-boot copy."""
from __future__ import annotations

from pathlib import Path

import pytest

from tongue_backend.stores.prompt_store import PromptStore


def test_first_boot_copies_default_to_current(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("hello", encoding="utf-8")
    current = tmp_path / "current.md"
    store = PromptStore(default_path=default, current_path=current)

    text = store.load_current()
    assert text == "hello"
    assert current.read_text(encoding="utf-8") == "hello"


def test_save_writes_to_current(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("default", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("placeholder", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)

    store.save("new prompt")
    assert current.read_text(encoding="utf-8") == "new prompt"


def test_reset_overwrites_current_with_default(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("default", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("user edits", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)

    store.reset()
    assert current.read_text(encoding="utf-8") == "default"


def test_is_default_when_current_matches_default(tmp_path: Path):
    default = tmp_path / "default.md"
    default.write_text("X", encoding="utf-8")
    current = tmp_path / "current.md"
    current.write_text("X", encoding="utf-8")
    store = PromptStore(default_path=default, current_path=current)
    assert store.is_default() is True

    store.save("Y")
    assert store.is_default() is False
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_prompt_store.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement prompt_store.py**

Create `packages/backend/src/tongue_backend/stores/prompt_store.py`:

```python
"""Read/write the current system prompt; first-boot copies from default."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptStore:
    default_path: Path
    current_path: Path

    def _ensure_current(self) -> None:
        if not self.current_path.exists():
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.default_path, self.current_path)

    def load_current(self) -> str:
        self._ensure_current()
        return self.current_path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text(content, encoding="utf-8")

    def reset(self) -> None:
        shutil.copyfile(self.default_path, self.current_path)

    def is_default(self) -> bool:
        if not self.current_path.exists():
            return True
        return self.current_path.read_text(encoding="utf-8") == self.default_path.read_text(encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_prompt_store.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/prompt_store.py packages/backend/tests/test_prompt_store.py packages/backend/tests/conftest.py
git commit -m "feat(backend): add PromptStore with first-boot copy and reset"
```

### Task 18: Add `tongue_backend.stores.llm_store`

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/llm_store.py`
- Test: `packages/backend/tests/test_llm_store.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_llm_store.py`:

```python
"""Tests for LLMStore: validation rules and persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from tongue_backend.stores.llm_store import LLMStore, LLMValidationError


VALID_YAML = """
model: gemini-2.5-flash
temperature: 0.7
max_tokens: 1024
top_p: 0.95
"""


def _store(tmp_path: Path) -> LLMStore:
    default = tmp_path / "default.yaml"
    default.write_text(VALID_YAML, encoding="utf-8")
    current = tmp_path / "current.yaml"
    return LLMStore(default_path=default, current_path=current)


def test_load_returns_dict_with_expected_keys(tmp_path: Path):
    cfg = _store(tmp_path).load_current()
    assert cfg["model"] == "gemini-2.5-flash"
    assert cfg["temperature"] == 0.7


def test_save_validates_temperature_range(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="temperature"):
        s.save("model: x\ntemperature: 3.0\nmax_tokens: 10\ntop_p: 0.5\n")


def test_save_validates_max_tokens_positive(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        s.save("model: x\ntemperature: 0.5\nmax_tokens: 0\ntop_p: 0.5\n")


def test_save_validates_model_non_empty(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="model"):
        s.save("model: ''\ntemperature: 0.5\nmax_tokens: 10\ntop_p: 0.5\n")


def test_save_rejects_yaml_parse_error(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(LLMValidationError, match="parse"):
        s.save("model: x\ntemperature: : :\n")


def test_save_persists_valid_yaml(tmp_path: Path):
    s = _store(tmp_path)
    s.save("model: gemini-2.5-flash\ntemperature: 0.2\nmax_tokens: 256\ntop_p: 0.9\n")
    cfg = s.load_current()
    assert cfg["temperature"] == 0.2
    assert cfg["max_tokens"] == 256
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_llm_store.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement llm_store.py**

Create `packages/backend/src/tongue_backend/stores/llm_store.py`:

```python
"""Read/write/validate the LLM YAML config (model, temperature, max_tokens, top_p)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class LLMValidationError(ValueError):
    """Raised when LLM config fails validation."""


def _validate(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise LLMValidationError("config must be a YAML mapping")
    model = cfg.get("model")
    if not isinstance(model, str) or not model.strip():
        raise LLMValidationError("'model' must be a non-empty string")
    temperature = cfg.get("temperature")
    if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 2):
        raise LLMValidationError("'temperature' must be a number in [0, 2]")
    max_tokens = cfg.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise LLMValidationError("'max_tokens' must be a positive integer")
    top_p = cfg.get("top_p", 1.0)
    if not isinstance(top_p, (int, float)) or not (0 <= top_p <= 1):
        raise LLMValidationError("'top_p' must be a number in [0, 1]")


@dataclass(frozen=True)
class LLMStore:
    default_path: Path
    current_path: Path

    def _ensure_current(self) -> None:
        if not self.current_path.exists():
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.default_path, self.current_path)

    def load_current(self) -> dict[str, Any]:
        self._ensure_current()
        text = self.current_path.read_text(encoding="utf-8")
        try:
            cfg = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise LLMValidationError(f"yaml parse error: {exc}") from exc
        _validate(cfg)
        return cfg

    def save(self, content: str) -> None:
        try:
            cfg = yaml.safe_load(content) or {}
        except yaml.YAMLError as exc:
            raise LLMValidationError(f"yaml parse error: {exc}") from exc
        _validate(cfg)
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text(content, encoding="utf-8")

    def reset(self) -> None:
        shutil.copyfile(self.default_path, self.current_path)

    def is_default(self) -> bool:
        if not self.current_path.exists():
            return True
        return self.current_path.read_text(encoding="utf-8") == self.default_path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_llm_store.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/llm_store.py packages/backend/tests/test_llm_store.py
git commit -m "feat(backend): add LLMStore with validation (temperature/max_tokens/top_p/model)"
```

### Task 19: Add `tongue_backend.stores.registry_store`

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/registry_store.py`
- Test: `packages/backend/tests/test_registry_store.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_registry_store.py`:

```python
"""Tests for RegistryStore: persists YAML, validates structure, reload semantics."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tongue_ai.registry import Registry, RegistryError
from tongue_backend.stores.registry_store import RegistryStore


VALID_YAML = """
detector:
  enabled: false
heads:
  - name: head1
    head_type: single
    arch: resnet50
    weights_uri: local:weights/h1.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b]
category_map:
  head1:
    a: 舌色
    b: 舌色
"""


def _fake_state_dict(num_classes: int):
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, num_classes)
    return m.state_dict()


def _store(tmp_path: Path) -> RegistryStore:
    default = tmp_path / "registry.default.yaml"
    default.write_text(VALID_YAML, encoding="utf-8")
    current = tmp_path / "registry.current.yaml"
    return RegistryStore(default_path=default, current_path=current)


def test_save_persists_yaml_without_loading_models(tmp_path: Path):
    s = _store(tmp_path)
    s.save(VALID_YAML)
    assert "head1" in s.load_text()


def test_save_rejects_yaml_parse_error(tmp_path: Path):
    s = _store(tmp_path)
    with pytest.raises(RegistryError, match="parse"):
        s.save("heads: [::: bad")


def test_save_rejects_missing_required_field(tmp_path: Path):
    s = _store(tmp_path)
    bad = VALID_YAML.replace("    head_type: single", "")
    with pytest.raises(RegistryError, match="head_type"):
        s.save(bad)


def test_reload_with_missing_weights_returns_failure_list(tmp_path: Path):
    s = _store(tmp_path)
    s.save(VALID_YAML)
    result = s.reload()
    assert "head1" in result.failed_heads


def test_reload_with_valid_weights_returns_loaded_registry(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save(_fake_state_dict(2), weights_dir / "h1.pth")
    s = _store(tmp_path)
    s.save(VALID_YAML)
    reg = s.reload()
    assert isinstance(reg, Registry)
    assert [h.name for h in reg.heads] == ["head1"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_registry_store.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement registry_store.py**

Create `packages/backend/src/tongue_backend/stores/registry_store.py`:

```python
"""Read/validate/reload the registry YAML.

Save = persist + structural validation only.
Reload = load_registry (resolves weights, builds models, can partial-fail).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from tongue_ai.registry import Registry, RegistryError, load_registry, validate_registry_yaml


@dataclass(frozen=True)
class RegistryStore:
    default_path: Path
    current_path: Path

    def _ensure_current(self) -> None:
        if not self.current_path.exists():
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.default_path, self.current_path)

    def load_text(self) -> str:
        self._ensure_current()
        return self.current_path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp, validate, then atomic-replace
        tmp = self.current_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        try:
            validate_registry_yaml(tmp)
        except RegistryError:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(self.current_path)

    def reload(self) -> Registry:
        self._ensure_current()
        return load_registry(self.current_path, raise_on_partial_fail=False)

    def reset(self) -> None:
        shutil.copyfile(self.default_path, self.current_path)

    def is_default(self) -> bool:
        if not self.current_path.exists():
            return True
        return self.current_path.read_text(encoding="utf-8") == self.default_path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_registry_store.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/registry_store.py packages/backend/tests/test_registry_store.py
git commit -m "feat(backend): add RegistryStore — save validates, reload builds models"
```

### Task 20: Add `system.default.md` (locked 大眾版 prompt)

**Files:**
- Create: `packages/backend/prompts/system.default.md`

- [ ] **Step 1: Write the locked prompt**

Create `packages/backend/prompts/system.default.md`:

```markdown
# 角色與任務

你是一個**規則嚴格的中醫舌診解讀器**。輸入是視覺辨識模型對舌象的判讀結果（一份分類標籤＋信心度的清單）。你的任務是依據下方**鎖定**的對應表，把這些標籤映射為「中醫體質」與「證素」，並依固定格式產出大眾版報告。

# 強制原則

- **對應表為唯一依據**。不得使用對應表外的中醫知識。
- 不得對使用者問問題，不得拒絕作答（資料不足時，照下方規則處理）。
- 不得引用外部診療建議，不得提供個別化治療方案，不得估算嚴重程度或診斷疾病。
- 不得處理影像本身；輸入端只會給你分類結果。
- 永遠輸出繁體中文。

# 允許的證素清單（共 9 個，禁止其他）

氣虛、陽虛、陰虛、血虛、痰濕、濕熱、氣滯、血瘀、化熱

# 對應規則（鎖定）

- 舌色：淡 → 氣虛/陽虛/血虛；淡紅 → 正常傾向；微紅 → 化熱；紅 → 化熱/陰虛；絳 → 陰虛/化熱；青紫 → 血瘀；暗 → 血瘀/氣滯。
- 舌質：胖大 → 氣虛/痰濕；瘦薄 → 陰虛/血虛；嫩 → 虛（氣虛/陽虛/陰虛/血虛之一）；齒痕 → 氣虛；偏斜 → （列出，不映射證素）；瘀血絲 → 血瘀。
- 舌態：偏斜 → （列出，不映射證素）。
- 舌下絡脈：怒張、曲張、囊柱囊泡、瘀血絲 → 氣滯/血瘀（高權重）。

# 高/低特異性與允許推演

- 高特異性：裂紋 → 陰虛；厚/膩/腐苔 → 痰濕；舌下絡脈異常 → 氣滯/血瘀。
- 低特異性（可被抑制）：齒痕/胖大 → 氣虛；淡紅 → 正常；嫩 → 虛。
- 病機鏈（允許推演）：氣虛 → 陽虛；陽虛 → 痰濕；痰濕 → 氣滯；氣滯 → 血瘀；陰虛 → 血虛；痰濕久 → 濕熱。
- 組合：痰濕 + 化熱 → 濕熱。
- 衝突：可解釋的保留；無法解釋的降權或捨棄。

# 必填輸出格式（共 5 段，依序，不得增刪段落標題）

## 主要中醫體質

（一句話的白話描述，例：「濕氣偏重」）

## 次要中醫體質

（1–2 個次要傾向，白話）

## 體質說明

（整體狀態 + 簡單因果說明，生活化語言；不超過 4 句）

## 證素列表

（從 9 個允許證素挑出本次成立者，以「、」分隔）

## 警語

此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。

# 資料不足或無有效輸入時

- 「主要中醫體質」寫：「資料不足，無法判讀」
- 「次要中醫體質」寫：「無」
- 「體質說明」寫：「本次未能取得有效舌象判讀資料，建議補充清晰的舌面/舌下照片重新分析。」
- 「證素列表」寫：「無」
- 「警語」維持原文不變。
```

- [ ] **Step 2: Verify file exists**

```bash
wc -l packages/backend/prompts/system.default.md
```

Expected: 50+ lines.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/prompts/system.default.md
git commit -m "feat(backend): add locked 大眾版 system prompt (default)"
```

### Task 21: Add `llm.default.yaml`

**Files:**
- Create: `packages/backend/config/llm.default.yaml`

- [ ] **Step 1: Write the default**

Create `packages/backend/config/llm.default.yaml`:

```yaml
model: gemini-2.5-flash
temperature: 0.4
max_tokens: 1024
top_p: 0.95
```

- [ ] **Step 2: Commit**

```bash
git add packages/backend/config/llm.default.yaml
git commit -m "feat(backend): add llm.default.yaml (gemini-2.5-flash, t=0.4)"
```

### Task 22: Add `registry.default.yaml`

**Files:**
- Create: `packages/backend/config/registry.default.yaml`

- [ ] **Step 1: Write the default registry**

Create `packages/backend/config/registry.default.yaml`:

```yaml
detector:
  enabled: false

heads:
  - name: front
    head_type: single
    arch: resnet50
    weights_uri: hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names:
      - 淡紅
      - 紅
      - 淡
      - 絳
      - 青紫
      - 暗
      - 微紅
      - 胖大
      - 瘦薄
      - 嫩
      - 偏斜
      - 齒痕
      - 無異常
      - 瘀血絲

  - name: sublingual
    head_type: single
    arch: resnet50
    weights_uri: hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_sublingual.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names:
      - 怒張
      - 曲張
      - 囊柱囊泡

category_map:
  front:
    淡紅: 舌色
    紅: 舌色
    淡: 舌色
    絳: 舌色
    青紫: 舌色
    暗: 舌色
    微紅: 舌色
    胖大: 舌質
    瘦薄: 舌質
    嫩: 舌質
    齒痕: 舌質
    偏斜: 舌態
    無異常: 舌質
    瘀血絲: 舌下絡脈
  sublingual:
    怒張: 舌下絡脈
    曲張: 舌下絡脈
    囊柱囊泡: 舌下絡脈
```

- [ ] **Step 2: Validate against the registry validator**

```bash
uv run --package tongue-backend python -c "
from pathlib import Path
from tongue_ai.registry import validate_registry_yaml
validate_registry_yaml(Path('packages/backend/config/registry.default.yaml'))
print('valid')
"
```

Expected: `valid` printed.

- [ ] **Step 3: Add `.gitignore` for current variants**

Add to `.gitignore` (project root, append):

```
packages/backend/prompts/system.current.md
packages/backend/config/llm.current.yaml
packages/backend/config/registry.current.yaml
.env
```

- [ ] **Step 4: Commit**

```bash
git add packages/backend/config/registry.default.yaml .gitignore
git commit -m "feat(backend): add registry.default.yaml and gitignore current variants"
```

---

## Phase 6 — `tongue-backend`: LLM glue, pipeline, routes

### Task 23: Add `tongue_backend.llm.user_message`

**Files:**
- Create: `packages/backend/src/tongue_backend/llm/__init__.py`
- Create: `packages/backend/src/tongue_backend/llm/user_message.py`
- Test: `packages/backend/tests/test_user_message.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_user_message.py`:

```python
"""Golden tests for the composite-head → v4-category user message builder."""
from __future__ import annotations

from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.llm.user_message import build


CATEGORY_MAP = {
    "front": {
        "淡紅": "舌色",
        "胖大": "舌質",
        "偏斜": "舌態",
        "瘀血絲": "舌下絡脈",
        "無異常": "舌質",
    },
    "sublingual": {
        "怒張": "舌下絡脈",
        "曲張": "舌下絡脈",
    },
}


def test_single_front_prediction_renders_one_bullet():
    heads = [HeadResult(task="front", head_type="single", predictions=[ClassScore("淡紅", 0.78)])]
    out = build(heads, CATEGORY_MAP)
    assert "本次舌診判讀結果" in out
    assert "- 舌色：淡紅（0.78）" in out
    assert "請依規則輸出大眾版報告。" in out


def test_two_heads_render_two_bullets():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[ClassScore("胖大", 0.62)]),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.71)]),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "- 舌質：胖大（0.62）" in out
    assert "- 舌下絡脈：怒張（0.71）" in out


def test_cross_head_predictions_merge_under_one_category():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[ClassScore("瘀血絲", 0.51)]),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.72)]),
    ]
    out = build(heads, CATEGORY_MAP)
    # 舌下絡脈 line should contain both classes joined by 、
    assert "舌下絡脈" in out
    assert "瘀血絲（0.51）" in out
    assert "怒張（0.72）" in out
    assert out.count("- 舌下絡脈") == 1


def test_head_with_error_is_skipped():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[], error="boom"),
        HeadResult(task="sublingual", head_type="single", predictions=[ClassScore("怒張", 0.6)]),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "boom" not in out
    assert "- 舌下絡脈：怒張（0.60）" in out


def test_no_predictions_renders_no_data_line():
    heads = [
        HeadResult(task="front", head_type="single", predictions=[], error="x"),
        HeadResult(task="sublingual", head_type="single", predictions=[], error="y"),
    ]
    out = build(heads, CATEGORY_MAP)
    assert "（無可用判讀資料）" in out


def test_class_not_in_category_map_is_skipped_silently():
    heads = [HeadResult(task="front", head_type="single", predictions=[ClassScore("UNMAPPED", 0.9)])]
    out = build(heads, CATEGORY_MAP)
    assert "UNMAPPED" not in out
    assert "（無可用判讀資料）" in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_user_message.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement user_message.py**

Create `packages/backend/src/tongue_backend/llm/__init__.py`:

```python
"""LLM glue: prompt assembly + ADK client."""
```

Create `packages/backend/src/tongue_backend/llm/user_message.py`:

```python
"""Build the per-request zh-TW user message for the locked Gemini prompt.

Composite-head predictions are split back into v4 schema categories via
the static category_map, then grouped so each category renders one bullet.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from tongue_ai.types import ClassScore, HeadResult


HEADER = "本次舌診判讀結果："
FOOTER = "請依規則輸出大眾版報告。"
EMPTY_LINE = "- （無可用判讀資料）"


def build(
    heads: Iterable[HeadResult],
    category_map: dict[str, dict[str, str]],
) -> str:
    grouped: OrderedDict[str, list[ClassScore]] = OrderedDict()
    for head in heads:
        if head.error is not None:
            continue
        head_map = category_map.get(head.task, {})
        for pred in head.predictions:
            cat = head_map.get(pred.label)
            if cat is None:
                continue
            grouped.setdefault(cat, []).append(pred)

    lines: list[str] = [HEADER, ""]
    if not grouped:
        lines.append(EMPTY_LINE)
    else:
        for cat, preds in grouped.items():
            joined = "、".join(f"{p.label}（{p.score:.2f}）" for p in preds)
            lines.append(f"- {cat}：{joined}")
    lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_user_message.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/llm/__init__.py packages/backend/src/tongue_backend/llm/user_message.py packages/backend/tests/test_user_message.py
git commit -m "feat(backend): add user_message.build with composite-head category splitter"
```

### Task 24: Add `tongue_backend.llm.client` (ADK wrapper)

**Files:**
- Create: `packages/backend/src/tongue_backend/llm/client.py`
- Test: `packages/backend/tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_llm_client.py`:

```python
"""Tests for the ADK Gemini client wrapper.

The wrapper is mocked at the ADK boundary; we don't need a live Gemini key.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tongue_backend.llm.client import LLMClient, LLMUnavailableError


def test_run_calls_adk_with_system_user_and_config():
    cfg = {"model": "gemini-2.5-flash", "temperature": 0.5, "max_tokens": 100, "top_p": 0.9}
    client = LLMClient()
    fake_resp = MagicMock(text="OUTPUT")
    with patch.object(client, "_call_adk", return_value=fake_resp) as m:
        out = client.run(system="SYS", user="USR", config=cfg)
    m.assert_called_once_with("SYS", "USR", cfg)
    assert out == "OUTPUT"


def test_run_returns_error_stamped_text_on_network_failure():
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    with patch.object(client, "_call_adk", side_effect=ConnectionError("nope")):
        out = client.run(system="SYS", user="USR", config=cfg)
    assert out.startswith("⚠ 醫師建議產生失敗")
    assert "nope" in out


def test_run_returns_error_when_response_empty():
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    with patch.object(client, "_call_adk", return_value=MagicMock(text="")):
        out = client.run(system="SYS", user="USR", config=cfg)
    assert "未產生回應" in out


def test_run_raises_when_credentials_missing(monkeypatch):
    cfg = {"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0}
    client = LLMClient()
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with patch.object(client, "_call_adk", side_effect=LLMUnavailableError("no creds")):
        with pytest.raises(LLMUnavailableError):
            client.run(system="SYS", user="USR", config=cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_llm_client.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement client.py**

Create `packages/backend/src/tongue_backend/llm/client.py`:

```python
"""ADK Gemini client wrapper.

Encapsulates the boundary so tests can mock at `_call_adk`. The real `_call_adk`
imports `google.adk` lazily so a missing/unauthorised SDK doesn't crash imports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM backend is unreachable due to credentials/config."""


def _retry_count() -> int:
    return int(os.environ.get("LLM_RETRIES", "1"))


@dataclass
class LLMClient:
    """Wraps Google ADK to send (system, user, config) and receive text."""

    def run(self, *, system: str, user: str, config: dict[str, Any]) -> str:
        attempts = _retry_count() + 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                resp = self._call_adk(system, user, config)
                text = (resp.text or "").strip() if resp is not None else ""
                if not text:
                    return "⚠ 醫師建議產生失敗：模型未產生回應"
                return text
            except LLMUnavailableError:
                raise
            except Exception as exc:  # network / 5xx / timeouts
                last_exc = exc
                continue
        msg = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
        return f"⚠ 醫師建議產生失敗：{msg}"

    def _call_adk(self, system: str, user: str, config: dict[str, Any]):
        # Lazy import: keeps imports fast and avoids hard fail when ADK auth missing.
        try:
            from google import adk  # type: ignore
        except Exception as exc:
            raise LLMUnavailableError(f"google-adk import failed: {exc}") from exc

        # NOTE for the implementer: the precise ADK API can vary by version.
        # If `adk.Agent` is the entry point in the installed version, build:
        #   agent = adk.Agent(model=config["model"], system_instruction=system,
        #                     temperature=config["temperature"], top_p=config["top_p"],
        #                     max_output_tokens=config["max_tokens"])
        #   resp = agent.run(user)
        #   return resp  # must expose .text
        # If a newer/older API surface differs, adjust this single method only.
        try:
            agent = adk.Agent(  # type: ignore[attr-defined]
                model=config["model"],
                system_instruction=system,
                temperature=config["temperature"],
                top_p=config.get("top_p", 1.0),
                max_output_tokens=config["max_tokens"],
            )
            return agent.run(user)
        except AttributeError as exc:
            raise LLMUnavailableError(f"unexpected ADK API surface: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_llm_client.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/llm/client.py packages/backend/tests/test_llm_client.py
git commit -m "feat(backend): add ADK LLM client wrapper with retry and graceful degradation"
```

### Task 25: Add `tongue_backend.pipeline`

**Files:**
- Create: `packages/backend/src/tongue_backend/pipeline.py`
- Test: `packages/backend/tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_pipeline.py`:

```python
"""Tests for pipeline.analyze: stub registry + stub LLM → asserted response shape."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.pipeline import AnalyzeResponse, analyze, DISCLAIMER


def _jpeg_bytes() -> bytes:
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_registry(predictions_per_head: dict[str, ClassScore | None]) -> Registry:
    heads = []
    for name, pred in predictions_per_head.items():
        h = MagicMock()
        h.name = name
        h.predict.return_value = HeadResult(
            task=name,
            head_type="single",
            predictions=[pred] if pred else [],
            error=None if pred else "no signal",
        )
        heads.append(h)
    return Registry(
        detector_enabled=False,
        heads=heads,
        category_map={
            "front": {"淡紅": "舌色"},
            "sublingual": {"怒張": "舌下絡脈"},
        },
    )


def test_analyze_returns_expected_shape():
    registry = _make_registry({
        "front": ClassScore("淡紅", 0.78),
        "sublingual": ClassScore("怒張", 0.65),
    })
    llm = MagicMock()
    llm.run.return_value = "## 主要中醫體質\n氣虛\n## 警語\n..."

    resp = analyze(
        image_bytes=_jpeg_bytes(),
        registry=registry,
        prompt="LOCKED PROMPT",
        llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
        llm_client=llm,
    )

    assert isinstance(resp, AnalyzeResponse)
    assert resp.disclaimer == DISCLAIMER
    assert "舌色：淡紅" in resp.user_message
    assert "舌下絡脈：怒張" in resp.user_message
    assert resp.comment.startswith("## 主要中醫體質")
    assert resp.category_map["front"]["淡紅"] == "舌色"
    assert resp.timing_ms["total"] >= 0
    llm.run.assert_called_once()
    assert llm.run.call_args.kwargs["system"] == "LOCKED PROMPT"


def test_analyze_continues_when_one_head_errors():
    registry = _make_registry({"front": ClassScore("淡紅", 0.7), "sublingual": None})
    llm = MagicMock()
    llm.run.return_value = "OK"
    resp = analyze(
        image_bytes=_jpeg_bytes(),
        registry=registry,
        prompt="P",
        llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
        llm_client=llm,
    )
    assert any(h.task == "sublingual" and h.error for h in resp.heads)
    assert "舌色：淡紅" in resp.user_message


def test_analyze_raises_value_error_on_undecodable_image():
    registry = _make_registry({"front": ClassScore("淡紅", 0.5)})
    llm = MagicMock()
    with pytest.raises(ValueError, match="decode"):
        analyze(
            image_bytes=b"not an image",
            registry=registry,
            prompt="P",
            llm_config={"model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0},
            llm_client=llm,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_pipeline.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement pipeline.py**

Create `packages/backend/src/tongue_backend/pipeline.py`:

```python
"""End-to-end inference pipeline: bytes → heads → LLM → AnalyzeResponse."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tongue_ai.inference import run_all
from tongue_ai.preprocessing import decode_bgr
from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult

from tongue_backend.llm.user_message import build as build_user_message


DISCLAIMER = "此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。"


class _LLMClientProto(Protocol):
    def run(self, *, system: str, user: str, config: dict[str, Any]) -> str: ...


@dataclass
class AnalyzeResponse:
    user_message: str
    heads: list[HeadResult]
    comment: str
    disclaimer: str
    category_map: dict[str, dict[str, str]]
    timing_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "heads": [
                {
                    "task": h.task,
                    "head_type": h.head_type,
                    "predictions": [{"label": p.label, "score": p.score} for p in h.predictions],
                    "error": h.error,
                }
                for h in self.heads
            ],
            "comment": self.comment,
            "disclaimer": self.disclaimer,
            "category_map": self.category_map,
            "timing_ms": self.timing_ms,
        }


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def analyze(
    *,
    image_bytes: bytes,
    registry: Registry,
    prompt: str,
    llm_config: dict[str, Any],
    llm_client: _LLMClientProto,
) -> AnalyzeResponse:
    t0 = time.perf_counter()
    image = decode_bgr(image_bytes)
    t_decode = _ms_since(t0)

    t1 = time.perf_counter()
    heads = run_all(image, registry)
    t_infer = _ms_since(t1)

    user_msg = build_user_message(heads, registry.category_map)

    t2 = time.perf_counter()
    comment = llm_client.run(system=prompt, user=user_msg, config=llm_config)
    t_llm = _ms_since(t2)

    return AnalyzeResponse(
        user_message=user_msg,
        heads=heads,
        comment=comment,
        disclaimer=DISCLAIMER,
        category_map=registry.category_map,
        timing_ms={
            "decode": t_decode,
            "infer": t_infer,
            "llm": t_llm,
            "total": _ms_since(t0),
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_pipeline.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/pipeline.py packages/backend/tests/test_pipeline.py
git commit -m "feat(backend): add pipeline.analyze with timing and stable response shape"
```

### Task 26: Update `app.py` with startup hook + state

**Files:**
- Modify: `packages/backend/src/tongue_backend/app.py`

- [ ] **Step 1: Replace app.py**

Replace `packages/backend/src/tongue_backend/app.py` with:

```python
"""FastAPI application factory with startup-time registry/store wiring."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tongue_ai.registry import RegistryError

from tongue_backend.llm.client import LLMClient
from tongue_backend.routes.analyze import router as analyze_router
from tongue_backend.routes.config import router as config_router
from tongue_backend.routes.health import router as health_router
from tongue_backend.stores import paths
from tongue_backend.stores.llm_store import LLMStore
from tongue_backend.stores.prompt_store import PromptStore
from tongue_backend.stores.registry_store import RegistryStore


_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prompt_store = PromptStore(paths.PROMPT_DEFAULT, paths.PROMPT_CURRENT)
    app.state.llm_store = LLMStore(paths.LLM_DEFAULT, paths.LLM_CURRENT)
    app.state.registry_store = RegistryStore(paths.REGISTRY_DEFAULT, paths.REGISTRY_CURRENT)
    app.state.llm_client = LLMClient()

    try:
        app.state.registry = app.state.registry_store.reload()
        _log.info(
            "registry loaded: %d heads, %d failed",
            len(app.state.registry.heads),
            len(app.state.registry.failed_heads),
        )
    except RegistryError as exc:
        _log.error("registry load failed at startup: %s", exc)
        app.state.registry = None
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Tongue Diagnosis API", version="0.2.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(config_router)
    return app


app = create_app()
```

- [ ] **Step 2: Verify the app starts (will be exercised by route tests later)**

```bash
uv run --package tongue-backend python -c "
from tongue_backend.app import create_app
app = create_app()
print('app created with', len(app.routes), 'routes')
"
```

Expected: prints a route count > 0; no traceback.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/src/tongue_backend/app.py
git commit -m "feat(backend): add lifespan hook wiring stores, registry, and LLM client onto app.state"
```

### Task 27: Update `routes/health.py`

**Files:**
- Modify: `packages/backend/src/tongue_backend/routes/health.py`
- Test: `packages/backend/tests/test_routes_health.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_routes_health.py`:

```python
"""Tests for /health."""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from tongue_backend.app import create_app


def test_health_returns_status_ai_version_registry_summary():
    app = create_app()
    app.state.registry = MagicMock(heads=[1, 2], failed_heads={"x": "boom"})
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "ai_version" in body
    assert body["registry"]["loaded"] == 2
    assert body["registry"]["failed"] == ["x"]


def test_health_when_registry_is_none():
    app = create_app()
    app.state.registry = None
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["registry"]["loaded"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_health.py -v
```

Expected: AssertionError; current health route doesn't expose registry summary.

- [ ] **Step 3: Replace routes/health.py**

Replace `packages/backend/src/tongue_backend/routes/health.py` with:

```python
"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request

from tongue_ai import __version__ as ai_version


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        registry_summary = {"loaded": 0, "failed": []}
    else:
        registry_summary = {
            "loaded": len(registry.heads),
            "failed": list(registry.failed_heads.keys()),
        }
    return {
        "status": "ok",
        "ai_version": ai_version,
        "registry": registry_summary,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_health.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/routes/health.py packages/backend/tests/test_routes_health.py
git commit -m "feat(backend): expose registry summary in /health"
```

### Task 28: Rewrite `routes/analyze.py`

**Files:**
- Modify: `packages/backend/src/tongue_backend/routes/analyze.py`
- Test: `packages/backend/tests/test_routes_analyze.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_routes_analyze.py`:

```python
"""Tests for POST /api/analyze."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tongue_ai.registry import Registry
from tongue_ai.types import ClassScore, HeadResult
from tongue_backend.app import create_app


def _jpeg_bytes() -> bytes:
    img = np.full((40, 40, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _wire_app():
    app = create_app()
    head = MagicMock()
    head.name = "front"
    head.predict.return_value = HeadResult(
        task="front", head_type="single", predictions=[ClassScore("淡紅", 0.85)],
    )
    app.state.registry = Registry(
        detector_enabled=False,
        heads=[head],
        category_map={"front": {"淡紅": "舌色"}},
    )
    app.state.prompt_store = MagicMock()
    app.state.prompt_store.load_current.return_value = "LOCKED"
    app.state.llm_store = MagicMock()
    app.state.llm_store.load_current.return_value = {
        "model": "x", "temperature": 0.5, "max_tokens": 10, "top_p": 1.0,
    }
    app.state.llm_client = MagicMock()
    app.state.llm_client.run.return_value = "## 主要中醫體質\n氣虛"
    return app


def test_analyze_returns_200_with_expected_keys():
    app = _wire_app()
    with TestClient(app) as client:
        r = client.post("/api/analyze", files={"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "user_message" in body and "heads" in body and "comment" in body
    assert body["disclaimer"]
    assert body["heads"][0]["task"] == "front"


def test_analyze_returns_400_when_image_corrupt():
    app = _wire_app()
    with TestClient(app) as client:
        r = client.post("/api/analyze", files={"file": ("bad.jpg", b"junk", "image/jpeg")})
    assert r.status_code == 400
    assert "decode" in r.json()["error"]


def test_analyze_returns_413_when_image_too_large():
    app = _wire_app()
    huge = b"\xff" * (10 * 1024 * 1024 + 1)
    with TestClient(app) as client:
        r = client.post("/api/analyze", files={"file": ("big.jpg", huge, "image/jpeg")})
    assert r.status_code == 413


def test_analyze_returns_503_when_registry_missing():
    app = _wire_app()
    app.state.registry = None
    with TestClient(app) as client:
        r = client.post("/api/analyze", files={"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_analyze.py -v
```

Expected: 4 failed (existing analyze route returns the old shape).

- [ ] **Step 3: Replace routes/analyze.py**

Replace `packages/backend/src/tongue_backend/routes/analyze.py` with:

```python
"""POST /api/analyze: full pipeline (decode → infer → LLM)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile

from tongue_backend.pipeline import analyze as run_pipeline


router = APIRouter(prefix="/api", tags=["analyze"])

MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/analyze")
async def api_analyze(request: Request, file: UploadFile) -> dict:
    if request.app.state.registry is None:
        raise HTTPException(status_code=503, detail={"error": "registry unavailable"})

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "image too large"})
    if not data:
        raise HTTPException(status_code=400, detail={"error": "missing file"})

    try:
        prompt = request.app.state.prompt_store.load_current()
        llm_config = request.app.state.llm_store.load_current()
        resp = run_pipeline(
            image_bytes=data,
            registry=request.app.state.registry,
            prompt=prompt,
            llm_config=llm_config,
            llm_client=request.app.state.llm_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return resp.to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_analyze.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/routes/analyze.py packages/backend/tests/test_routes_analyze.py
git commit -m "feat(backend): rewrite /api/analyze to use pipeline.analyze"
```

### Task 29: Add `routes/config.py` (CRUD + reload)

**Files:**
- Create: `packages/backend/src/tongue_backend/routes/config.py`
- Test: `packages/backend/tests/test_routes_config.py`

- [ ] **Step 1: Write failing tests**

Create `packages/backend/tests/test_routes_config.py`:

```python
"""Tests for /api/config/{section} CRUD + reload."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from tongue_backend.app import create_app
from tongue_backend.stores import paths
from tongue_backend.stores.llm_store import LLMStore
from tongue_backend.stores.prompt_store import PromptStore
from tongue_backend.stores.registry_store import RegistryStore


@pytest.fixture
def wired_app(tmp_path: Path):
    # Copy real defaults into tmp
    p_default = tmp_path / "system.default.md"
    p_default.write_text("DEFAULT", encoding="utf-8")
    l_default = tmp_path / "llm.default.yaml"
    l_default.write_text("model: x\ntemperature: 0.5\nmax_tokens: 10\ntop_p: 0.9\n", encoding="utf-8")
    r_default = tmp_path / "registry.default.yaml"
    r_default.write_text(
        """
detector: {enabled: false}
heads:
  - name: h
    head_type: single
    arch: resnet50
    weights_uri: local:weights/h.pth
    input_size: [224, 224]
    normalise: imagenet
    class_names: [a, b]
category_map:
  h: {a: 舌色, b: 舌色}
""",
        encoding="utf-8",
    )

    app = create_app()
    app.state.prompt_store = PromptStore(p_default, tmp_path / "system.current.md")
    app.state.llm_store = LLMStore(l_default, tmp_path / "llm.current.yaml")
    app.state.registry_store = RegistryStore(r_default, tmp_path / "registry.current.yaml")
    app.state.registry = None
    return app


def test_get_prompt_section_returns_content(wired_app):
    with TestClient(wired_app) as client:
        r = client.get("/api/config/prompt")
    assert r.status_code == 200
    assert r.json()["content"] == "DEFAULT"
    assert r.json()["is_default"] is True


def test_put_prompt_persists(wired_app):
    with TestClient(wired_app) as client:
        r = client.put("/api/config/prompt", json={"content": "EDITED"})
        assert r.status_code == 200
        r2 = client.get("/api/config/prompt")
    assert r2.json()["content"] == "EDITED"
    assert r2.json()["is_default"] is False


def test_put_llm_invalid_temperature_returns_422(wired_app):
    with TestClient(wired_app) as client:
        r = client.put(
            "/api/config/llm",
            json={"content": "model: x\ntemperature: 5\nmax_tokens: 10\ntop_p: 0.5\n"},
        )
    assert r.status_code == 422
    assert "temperature" in r.json()["error"]


def test_put_registry_invalid_yaml_returns_422(wired_app):
    with TestClient(wired_app) as client:
        r = client.put("/api/config/registry", json={"content": "heads: [::: bad"})
    assert r.status_code == 422


def test_post_reset_restores_default(wired_app):
    with TestClient(wired_app) as client:
        client.put("/api/config/prompt", json={"content": "EDITED"})
        r = client.post("/api/config/prompt/reset")
        assert r.status_code == 200
        r2 = client.get("/api/config/prompt")
    assert r2.json()["content"] == "DEFAULT"


def test_post_registry_reload_returns_loaded_and_failed(wired_app, tmp_path: Path):
    # Save valid YAML; the weight file is missing → expect failed list with our head
    with TestClient(wired_app) as client:
        client.put(
            "/api/config/registry",
            json={
                "content": (tmp_path / "registry.default.yaml").read_text(encoding="utf-8")
            },
        )
        r = client.post("/api/config/registry/reload")
    assert r.status_code == 200
    body = r.json()
    assert "loaded" in body and "failed" in body
    assert "h" in body["failed"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_config.py -v
```

Expected: ImportError on `tongue_backend.routes.config`.

- [ ] **Step 3: Implement config routes**

Create `packages/backend/src/tongue_backend/routes/config.py`:

```python
"""/api/config/{section} CRUD + reload (prompt | llm | registry)."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from tongue_ai.registry import RegistryError

from tongue_backend.stores.llm_store import LLMValidationError


router = APIRouter(prefix="/api/config", tags=["config"])


Section = Literal["prompt", "llm", "registry"]


class _ContentBody(BaseModel):
    content: str


def _store(request: Request, section: Section):
    if section == "prompt":
        return request.app.state.prompt_store
    if section == "llm":
        return request.app.state.llm_store
    if section == "registry":
        return request.app.state.registry_store
    raise HTTPException(status_code=404, detail={"error": f"unknown section {section!r}"})


def _read_text(store, section: Section) -> str:
    if section == "registry":
        return store.load_text()
    if section == "prompt":
        return store.load_current()
    if section == "llm":
        # llm_store.load_current returns dict; we want the raw YAML
        return store.current_path.read_text(encoding="utf-8") if store.current_path.exists() else store.default_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail={"error": f"unknown section {section!r}"})


@router.get("/{section}")
async def get_section(section: Section, request: Request) -> dict:
    store = _store(request, section)
    content = _read_text(store, section)
    return {
        "section": section,
        "content": content,
        "is_default": store.is_default(),
    }


@router.put("/{section}")
async def put_section(section: Section, body: _ContentBody, request: Request) -> dict:
    store = _store(request, section)
    try:
        store.save(body.content)
    except LLMValidationError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
    except RegistryError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
    return {"section": section, "saved": True, "is_default": store.is_default()}


@router.post("/{section}/reset")
async def reset_section(section: Section, request: Request) -> dict:
    store = _store(request, section)
    store.reset()
    return {"section": section, "reset": True}


@router.post("/registry/reload")
async def reload_registry(request: Request) -> dict:
    store: object = request.app.state.registry_store
    new_registry = store.reload()  # type: ignore[attr-defined]
    if not new_registry.heads:
        # Total fail: keep previous registry; surface the failures
        return {
            "loaded": [],
            "failed": new_registry.failed_heads,
            "rolled_back": True,
        }
    request.app.state.registry = new_registry
    return {
        "loaded": [h.name for h in new_registry.heads],
        "failed": new_registry.failed_heads,
        "rolled_back": False,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run --package tongue-backend pytest packages/backend/tests/test_routes_config.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/routes/config.py packages/backend/tests/test_routes_config.py
git commit -m "feat(backend): add /api/config/{section} CRUD + /api/config/registry/reload"
```

### Task 30: Add `.env.example`

**Files:**
- Create: `packages/backend/.env.example`

- [ ] **Step 1: Write the example**

Create `packages/backend/.env.example`:

```dotenv
# Hugging Face Hub token for fetching private model weights.
# Read or Write scope; required if registry uses hf:// URIs.
HF_TOKEN=

# Google Cloud service-account JSON path for ADK / Vertex Gemini access.
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json

# Override default LLM model (otherwise comes from llm.current.yaml).
# LLM_MODEL=gemini-2.5-flash

# Number of LLM retries on transient failure (defaults to 1, i.e. 2 attempts total).
# LLM_RETRIES=1
```

- [ ] **Step 2: Commit**

```bash
git add packages/backend/.env.example
git commit -m "docs(backend): add .env.example documenting HF_TOKEN and ADK credentials"
```

---

## Phase 7 — `tongue-frontend`: Gradio four-tab UI

### Task 31: Update `tongue-frontend` `pyproject.toml`

**Files:**
- Modify: `packages/frontend/pyproject.toml`

- [ ] **Step 1: Replace deps**

Replace the entire `[project]` block:

```toml
[project]
name = "tongue-frontend"
version = "0.1.0"
description = "Gradio frontend for tongue diagnosis"
requires-python = ">=3.11"
dependencies = [
    "gradio>=4",
    "httpx>=0.27",
]
```

- [ ] **Step 2: Sync workspace**

```bash
uv sync --all-packages
```

Expected: pulls gradio; removes streamlit on next resolve. (uv may keep streamlit cached; that's fine.)

- [ ] **Step 3: Commit**

```bash
git add packages/frontend/pyproject.toml
git commit -m "chore(frontend): swap streamlit for gradio>=4"
```

### Task 32: Add `tongue_frontend.api` (httpx client)

**Files:**
- Create: `packages/frontend/src/tongue_frontend/api.py`
- Test: `packages/frontend/tests/test_api.py`

- [ ] **Step 1: Add `respx` to frontend dev deps (inline-install for test only)**

Skip — we'll add `respx` to the workspace as a dev dep via the test command using `--with`.

- [ ] **Step 2: Write failing tests**

Create `packages/frontend/tests/test_api.py`:

```python
"""Tests for the thin httpx API client (mocked transport via respx)."""
from __future__ import annotations

import httpx
import pytest
import respx

from tongue_frontend.api import APIClient


@respx.mock
def test_health_returns_dict():
    respx.get("http://localhost:8000/health").mock(
        return_value=httpx.Response(200, json={"status": "ok", "ai_version": "0.2.0"})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.health()
    assert out["status"] == "ok"


@respx.mock
def test_analyze_posts_multipart_and_returns_dict():
    route = respx.post("http://localhost:8000/api/analyze").mock(
        return_value=httpx.Response(200, json={"comment": "OK", "user_message": "x", "heads": []})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.analyze(filename="x.jpg", content=b"fake", content_type="image/jpeg")
    assert route.called
    assert out["comment"] == "OK"


@respx.mock
def test_get_config_section_returns_content():
    respx.get("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"section": "prompt", "content": "X", "is_default": True})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.get_config("prompt")
    assert out["content"] == "X"


@respx.mock
def test_put_config_section_sends_content_body():
    route = respx.put("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"saved": True})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.put_config("prompt", "EDITED")
    assert route.called
    assert out["saved"] is True


@respx.mock
def test_reload_registry_returns_loaded_failed():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(200, json={"loaded": ["front"], "failed": {}})
    )
    client = APIClient(base_url="http://localhost:8000")
    out = client.reload_registry()
    assert out["loaded"] == ["front"]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run --package tongue-frontend --with respx pytest packages/frontend/tests/test_api.py -v
```

Expected: ImportError on `tongue_frontend.api`.

- [ ] **Step 4: Implement api.py**

Create `packages/frontend/src/tongue_frontend/api.py`:

```python
"""Thin httpx client for the FastAPI backend."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class APIClient:
    base_url: str = "http://localhost:8000"
    timeout: float = 60.0

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> dict:
        with self._client() as c:
            return c.get("/health").json()

    def analyze(self, *, filename: str, content: bytes, content_type: str) -> dict:
        files = {"file": (filename, content, content_type)}
        with self._client() as c:
            return c.post("/api/analyze", files=files).json()

    def get_config(self, section: str) -> dict:
        with self._client() as c:
            return c.get(f"/api/config/{section}").json()

    def put_config(self, section: str, content: str) -> dict:
        with self._client() as c:
            return c.put(f"/api/config/{section}", json={"content": content}).json()

    def reset_config(self, section: str) -> dict:
        with self._client() as c:
            return c.post(f"/api/config/{section}/reset").json()

    def reload_registry(self) -> dict:
        with self._client() as c:
            return c.post("/api/config/registry/reload").json()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run --package tongue-frontend --with respx pytest packages/frontend/tests/test_api.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/frontend/src/tongue_frontend/api.py packages/frontend/tests/test_api.py
git commit -m "feat(frontend): add httpx APIClient for backend HTTP calls"
```

### Task 33: Add `views/analyze.py`

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/__init__.py`
- Create: `packages/frontend/src/tongue_frontend/views/analyze.py`

- [ ] **Step 1: Implement views/__init__.py**

Create `packages/frontend/src/tongue_frontend/views/__init__.py`:

```python
"""Gradio tab view builders."""
```

- [ ] **Step 2: Implement views/analyze.py**

Create `packages/frontend/src/tongue_frontend/views/analyze.py`:

```python
"""Tab 1: 舌診分析 — capture/upload + result panel."""
from __future__ import annotations

import io
import json

import gradio as gr
import numpy as np
from PIL import Image

from tongue_frontend.api import APIClient


def _to_jpeg_bytes(img: np.ndarray) -> bytes:
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _format_heads_table(heads: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for h in heads:
        if h.get("error"):
            rows.append([h["task"], "(error)", h["error"]])
            continue
        labels = "、".join(f"{p['label']}（{p['score']:.2f}）" for p in h.get("predictions", []))
        rows.append([h["task"], labels, ""])
    return rows


def _format_v4_breakdown(category_map: dict, heads: list[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for h in heads:
        head_map = category_map.get(h["task"], {})
        for p in h.get("predictions", []):
            cat = head_map.get(p["label"])
            if cat is not None:
                rows.append([cat, p["label"], f"{p['score']:.2f}"])
    return rows


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="舌象照片", sources=["upload", "webcam"], type="numpy")
                analyze_btn = gr.Button("分析", variant="primary")
            with gr.Column(scale=2):
                comment_md = gr.Markdown(label="醫師建議")
                disclaimer_md = gr.Markdown(label="警語")
                heads_table = gr.Dataframe(
                    headers=["Head", "Predictions", "Error"], label="原始模型輸出",
                )
                v4_table = gr.Dataframe(
                    headers=["v4 類別", "標籤", "信心度"], label="v4 對應分解",
                )
                with gr.Accordion("進階：傳給 LLM 的訊息 / 計時", open=False):
                    user_msg = gr.Code(label="user_message", language=None)
                    timing_json = gr.JSON(label="timing_ms")

        def _on_analyze(img):
            if img is None:
                return "未提供影像", "", [], [], "", {}
            try:
                jpeg = _to_jpeg_bytes(img)
                result = client.analyze(filename="capture.jpg", content=jpeg, content_type="image/jpeg")
            except Exception as exc:
                return f"後端連線失敗：{exc}", "", [], [], "", {}
            return (
                result.get("comment", ""),
                result.get("disclaimer", ""),
                _format_heads_table(result.get("heads", [])),
                _format_v4_breakdown(result.get("category_map", {}), result.get("heads", [])),
                result.get("user_message", ""),
                result.get("timing_ms", {}),
            )

        analyze_btn.click(
            _on_analyze,
            inputs=[image_input],
            outputs=[comment_md, disclaimer_md, heads_table, v4_table, user_msg, timing_json],
        )
    return tab
```

- [ ] **Step 3: Smoke import**

```bash
uv run --package tongue-frontend python -c "
from tongue_frontend.views.analyze import build_tab
from tongue_frontend.api import APIClient
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 4: Commit**

```bash
git add packages/frontend/src/tongue_frontend/views/__init__.py packages/frontend/src/tongue_frontend/views/analyze.py
git commit -m "feat(frontend): add Tab 1 — 舌診分析 (capture + result + v4 breakdown)"
```

### Task 34: Add `views/prompt_editor.py`, `views/llm_editor.py`, `views/registry_editor.py`

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/prompt_editor.py`
- Create: `packages/frontend/src/tongue_frontend/views/llm_editor.py`
- Create: `packages/frontend/src/tongue_frontend/views/registry_editor.py`

- [ ] **Step 1: Implement prompt_editor.py**

Create `packages/frontend/src/tongue_frontend/views/prompt_editor.py`:

```python
"""Tab 2: 提示詞設定."""
from __future__ import annotations

import gradio as gr

from tongue_frontend.api import APIClient


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        textarea = gr.Code(label="System Prompt", language="markdown", lines=24)
        status = gr.Markdown()
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reset_btn = gr.Button("Reset to default")
            reload_btn = gr.Button("Reload from disk")

        def _load() -> tuple[str, str]:
            try:
                resp = client.get_config("prompt")
            except Exception as exc:
                return "", f"❌ 讀取失敗：{exc}"
            badge = "（與預設一致）" if resp.get("is_default") else "（已修改）"
            return resp.get("content", ""), f"✅ 已載入 {badge}"

        def _save(content: str) -> str:
            try:
                resp = client.put_config("prompt", content)
            except Exception as exc:
                return f"❌ 儲存失敗：{exc}"
            return "✅ 已儲存" if resp.get("saved") else f"❌ {resp}"

        def _reset() -> tuple[str, str]:
            try:
                client.reset_config("prompt")
            except Exception as exc:
                return "", f"❌ Reset 失敗：{exc}"
            text, _msg = _load()
            return text, "✅ 已恢復預設"

        save_btn.click(_save, inputs=[textarea], outputs=[status])
        reset_btn.click(_reset, outputs=[textarea, status])
        reload_btn.click(_load, outputs=[textarea, status])
        tab.load(_load, outputs=[textarea, status])
    return tab
```

- [ ] **Step 2: Implement llm_editor.py**

Create `packages/frontend/src/tongue_frontend/views/llm_editor.py`:

```python
"""Tab 3: LLM 設定."""
from __future__ import annotations

import gradio as gr

from tongue_frontend.api import APIClient


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        textarea = gr.Code(label="LLM YAML", language="yaml", lines=12)
        status = gr.Markdown()
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reset_btn = gr.Button("Reset to default")
            reload_btn = gr.Button("Reload from disk")

        def _load() -> tuple[str, str]:
            try:
                resp = client.get_config("llm")
            except Exception as exc:
                return "", f"❌ 讀取失敗：{exc}"
            badge = "（與預設一致）" if resp.get("is_default") else "（已修改）"
            return resp.get("content", ""), f"✅ 已載入 {badge}"

        def _save(content: str) -> str:
            try:
                resp = client.put_config("llm", content)
            except Exception as exc:
                return f"❌ 儲存失敗：{exc}"
            return "✅ 已儲存" if resp.get("saved") else f"❌ {resp}"

        def _reset() -> tuple[str, str]:
            try:
                client.reset_config("llm")
            except Exception as exc:
                return "", f"❌ Reset 失敗：{exc}"
            text, _ = _load()
            return text, "✅ 已恢復預設"

        save_btn.click(_save, inputs=[textarea], outputs=[status])
        reset_btn.click(_reset, outputs=[textarea, status])
        reload_btn.click(_load, outputs=[textarea, status])
        tab.load(_load, outputs=[textarea, status])
    return tab
```

- [ ] **Step 3: Implement registry_editor.py**

Create `packages/frontend/src/tongue_frontend/views/registry_editor.py`:

```python
"""Tab 4: 模型設定 — Save / Reset / Reload from disk / Apply & Reload Models."""
from __future__ import annotations

import gradio as gr

from tongue_frontend.api import APIClient


def build_tab(client: APIClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        textarea = gr.Code(label="Registry YAML", language="yaml", lines=24)
        status = gr.Markdown()
        with gr.Row():
            save_btn = gr.Button("Save", variant="primary")
            reset_btn = gr.Button("Reset to default")
            reload_btn = gr.Button("Reload from disk")
            apply_btn = gr.Button("Apply & Reload Models", variant="stop")

        def _load() -> tuple[str, str]:
            try:
                resp = client.get_config("registry")
            except Exception as exc:
                return "", f"❌ 讀取失敗：{exc}"
            badge = "（與預設一致）" if resp.get("is_default") else "（已修改）"
            return resp.get("content", ""), f"✅ 已載入 {badge}"

        def _save(content: str) -> str:
            try:
                resp = client.put_config("registry", content)
            except Exception as exc:
                return f"❌ 儲存失敗：{exc}"
            return "✅ 已儲存（尚未生效，請按 Apply & Reload Models）" if resp.get("saved") else f"❌ {resp}"

        def _reset() -> tuple[str, str]:
            try:
                client.reset_config("registry")
            except Exception as exc:
                return "", f"❌ Reset 失敗：{exc}"
            text, _ = _load()
            return text, "✅ 已恢復預設"

        def _apply() -> str:
            try:
                resp = client.reload_registry()
            except Exception as exc:
                return f"❌ Reload 失敗：{exc}"
            loaded = resp.get("loaded", [])
            failed = resp.get("failed", {})
            rolled_back = resp.get("rolled_back", False)
            if rolled_back:
                return f"⚠ 全部失敗，已回滾至前一版本。失敗：{failed}"
            return f"✅ Loaded {len(loaded)} heads. Failed: {failed if failed else '無'}"

        save_btn.click(_save, inputs=[textarea], outputs=[status])
        reset_btn.click(_reset, outputs=[textarea, status])
        reload_btn.click(_load, outputs=[textarea, status])
        apply_btn.click(_apply, outputs=[status])
        tab.load(_load, outputs=[textarea, status])
    return tab
```

- [ ] **Step 4: Smoke imports**

```bash
uv run --package tongue-frontend python -c "
from tongue_frontend.views import prompt_editor, llm_editor, registry_editor
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 5: Commit**

```bash
git add packages/frontend/src/tongue_frontend/views/prompt_editor.py packages/frontend/src/tongue_frontend/views/llm_editor.py packages/frontend/src/tongue_frontend/views/registry_editor.py
git commit -m "feat(frontend): add Tabs 2/3/4 — prompt, LLM, registry editors"
```

### Task 35: Rewrite `app.py` as four-tab Gradio app

**Files:**
- Modify: `packages/frontend/src/tongue_frontend/app.py`

- [ ] **Step 1: Replace app.py**

Replace `packages/frontend/src/tongue_frontend/app.py` with:

```python
"""Gradio four-tab UI for tongue diagnosis."""
from __future__ import annotations

import os

import gradio as gr

from tongue_frontend.api import APIClient
from tongue_frontend.views import (
    analyze as analyze_view,
    llm_editor,
    prompt_editor,
    registry_editor,
)


def build_app() -> gr.Blocks:
    api_base = os.environ.get("TONGUE_API_BASE", "http://localhost:8000")
    client = APIClient(base_url=api_base)

    with gr.Blocks(title="Tongue Diagnosis") as app:
        gr.Markdown("# 舌診分析系統")
        with gr.Tabs():
            with gr.Tab("舌診分析"):
                analyze_view.build_tab(client)
            with gr.Tab("提示詞設定"):
                prompt_editor.build_tab(client)
            with gr.Tab("LLM 設定"):
                llm_editor.build_tab(client)
            with gr.Tab("模型設定"):
                registry_editor.build_tab(client)
    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", "7860")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke import**

```bash
uv run --package tongue-frontend python -c "
from tongue_frontend.app import build_app
build_app()
print('app built')
"
```

Expected: `app built`.

- [ ] **Step 3: Commit**

```bash
git add packages/frontend/src/tongue_frontend/app.py
git commit -m "feat(frontend): rewrite app.py as Gradio four-tab UI"
```

---

## Phase 8 — Cleanup, README, smoke test

### Task 36: Delete handoff folder and working-tree noise (filesystem cleanup, no commit)

**Note:** Both the Amanda folder and `test.md` are *untracked* files in the **main checkout** at `/home/daniel/Tongue-Diagnosis/`, not in this worktree. Removing them is a filesystem-only operation; no git commit is created on the feature branch (the deletions don't appear in the worktree's git state at all).

> **Precondition:** Task 14 (HF Hub upload) is complete. The `.pth` files are uploaded; deleting the folder is now safe.

- [ ] **Step 1: Delete the handoff folder from the main checkout**

```bash
rm -rf "/home/daniel/Tongue-Diagnosis/ResNet50測試模型權重檔V1_Amanda"
```

- [ ] **Step 2: Delete the unrelated test.md from the main checkout**

```bash
rm /home/daniel/Tongue-Diagnosis/packages/frontend/src/tongue_frontend/test.md
```

- [ ] **Step 3: Confirm worktree git state is unchanged**

```bash
git status
```

Expected: no new entries from the cleanup (because both files were untracked in the main checkout and never existed in the worktree). Any unrelated in-progress changes still show as before.

(No commit step — there's nothing to commit.)

### Task 37: Update root README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md**

Replace `README.md` with:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for PyTorch + HF Hub + Gradio + ADK"
```

### Task 38: Add `docs/smoke-test.md`

**Files:**
- Create: `docs/smoke-test.md`

- [ ] **Step 1: Write the smoke script**

Create `docs/smoke-test.md`:

```markdown
# Manual smoke test — Tongue Diagnosis POC

## Prerequisites

- `HF_TOKEN` exported (read access to the weights repo)
- `GOOGLE_APPLICATION_CREDENTIALS` exported (path to service-account JSON)
- `uv sync --all-packages` already run

## Steps

1. **Start the backend**

   ```bash
   uv run uvicorn tongue_backend.app:app --reload --port 8000
   ```

   Expected log: `registry loaded: 2 heads, 0 failed`. If `/health` returns `registry.loaded == 0`, the HF download or weights validation failed — check the uvicorn console.

2. **Curl the health endpoint**

   ```bash
   curl http://localhost:8000/health
   ```

   Expected: `{"status":"ok","ai_version":"0.2.0","registry":{"loaded":2,"failed":[]}}`.

3. **Start the frontend**

   ```bash
   uv run --package tongue-frontend python -m tongue_frontend.app
   ```

   Open http://localhost:7860.

4. **Tab 1 — 舌診分析**

   - Upload a tongue photo (JPEG/PNG ≤ 10 MB).
   - Click `分析`.
   - Expected:
     - `醫師建議` markdown panel shows 5 sections (主要中醫體質, 次要中醫體質, 體質說明, 證素列表, 警語).
     - `警語` panel shows the disclaimer.
     - `原始模型輸出` table shows two rows (`front`, `sublingual`) with predictions + scores.
     - `v4 對應分解` shows the predictions split into 舌色 / 舌質 / 舌態 / 舌下絡脈 buckets.
     - `進階` accordion shows the user_message text and timing breakdown.

5. **Tab 2 — 提示詞設定**

   - Edit a sentence in the prompt textarea.
   - Click `Save`.
   - Switch to Tab 1, re-run analyze on the same image.
   - Expected: comment changes (since Gemini got the new prompt). Click `Reset to default` to revert.

6. **Tab 3 — LLM 設定**

   - Change `temperature` to `1.5`. Save. Re-run analyze.
   - Expected: the comment is more verbose / varied.
   - Restore via `Reset to default`.

7. **Tab 4 — 模型設定**

   - Change a `class_names` entry to a typo (e.g., rename `怒張` to `怒X`). Save.
   - Click `Apply & Reload Models`.
   - Expected: status badge shows `Loaded 1 heads. Failed: {sublingual: ...}` (because the typo'd class_names mismatch the trained fc layer).
   - Click `Reset to default` then `Apply & Reload Models` to restore.

8. **Failure mode — backend down mid-flow**

   - Stop the uvicorn process.
   - Click `分析` in Tab 1.
   - Expected: result panel shows `後端連線失敗: ...`. Restart backend, retry.

## Pass criteria

- All 8 steps complete with expected outcomes.
- No tracebacks in the uvicorn console.
- `/health` reports `registry.loaded == 2` after the test.
```

- [ ] **Step 2: Commit**

```bash
git add docs/smoke-test.md
git commit -m "docs: add manual smoke-test script"
```

---

## Phase 9 — End-to-end verification

### Task 39: Run the full test suite

**Files:** none (verification only)

- [ ] **Step 1: Run all tests**

```bash
uv run pytest packages/ai/tests packages/backend/tests --with respx packages/frontend/tests -v
```

Expected: all tests pass. If any fail, fix in a focused commit before continuing.

- [ ] **Step 2: Verify no Streamlit references remain**

```bash
grep -rn streamlit packages docs/smoke-test.md README.md || echo "no streamlit references — good"
```

Expected: only the line `no streamlit references — good`.

- [ ] **Step 3: Verify the app boots end-to-end on a sample image (requires HF_TOKEN + ADK creds)**

```bash
# Terminal 1
uv run uvicorn tongue_backend.app:app --port 8000

# Terminal 2 (after ~10s for model load)
curl http://localhost:8000/health
# expected: registry.loaded == 2

# Terminal 2 (use any tongue jpg)
curl -X POST http://localhost:8000/api/analyze -F "file=@/path/to/sample.jpg"
# expected: 200 with non-empty user_message, comment, heads
```

If both succeed, the merge is functionally complete.

### Task 40: Push the branch and open the PR

**Files:** none

- [ ] **Step 1: Confirm branch state**

```bash
git log --oneline main..HEAD | wc -l
```

Expected: ≥ 30 commits (1 spec + 1 spec-fix + ~30 from this plan).

- [ ] **Step 2: Push to origin**

```bash
git push -u origin feat/merge-amanda-resnet50
```

- [ ] **Step 3: Open the PR**

Use the `superpowers:finishing-a-development-branch` skill to draft the PR description. Title:

```
feat: integrate Amanda's PyTorch ResNet50 weights into Tongue-Diagnosis
```

Body should reference:
- The merge spec at `docs/superpowers/specs/2026-05-07-merge-amanda-resnet50-design.md`
- The 10-commit shape from spec §14
- "What's NOT in this PR" list (per-task heads, ONNX, auth/rate-limit) — defer items
- Manual smoke checklist link

Per repo convention (see memory `feedback_no_coauthor.md`): no `Co-Authored-By` trailers.

---

## Self-review notes

This plan was self-reviewed against the spec (`2026-05-07-merge-amanda-resnet50-design.md`). Coverage map:

| Spec section | Tasks |
|---|---|
| §3 File disposition (Amanda's files) | Task 36 (delete folder) + Tasks 12–13 (training scripts) + Task 14 (weight upload) |
| §4 Architecture | Tasks 26–28 (backend wiring) + Task 35 (frontend) |
| §5 Package layout (changes) | Tasks 1, 9, 15, 31 (deps) + Tasks 2–8, 10–13, 16–25, 32–35 (modules) |
| §6 PyTorchTaskHead, WeightSource, Registry | Tasks 5, 6, 7 |
| §7 Composite-head splitter | Task 23 |
| §8 Pipeline | Task 25 |
| §9 Dependency changes | Tasks 1, 15, 31 |
| §10 HF Hub setup | Task 14 (manual upload) + Task 22 (registry default) + Task 30 (.env.example) |
| §11 Training subpackage | Tasks 10, 11, 12, 13 |
| §12 Error handling deltas | Distributed across Tasks 5 (WeightFetchError), 6 (load_state_dict), 7 (registry validation), 24 (LLM client retries), 28 (route 4xx/5xx) |
| §13 Testing deltas | Tests live in each feat task; Task 39 runs the suite end-to-end |
| §14 PR commit sequence | Mirrored across phases; Task 40 opens the PR |

Type consistency: `WeightFetchError` (Task 5) → caught in `load_registry` (Task 7) → bubble through `RegistryStore.reload` (Task 19) → surfaced via `/api/config/registry/reload` (Task 29). `category_map` shape (`dict[str, dict[str, str]]`) is consistent across Tasks 7, 23, 25, 28.

No placeholders remain. Every step has either runnable code or an exact command + expected output.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-07-merge-amanda-resnet50.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**






