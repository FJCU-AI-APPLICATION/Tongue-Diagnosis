# Tongue Diagnosis POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end POC where a user captures or uploads a tongue image in a Gradio UI, the FastAPI backend runs per-task ONNX classifiers and calls Gemini via Google ADK to produce a TCM doctor comment, and all runtime configuration (system prompt, LLM params, model registry) is editable from the same UI.

**Architecture:** Three-package monorepo (`tongue-ai`, `tongue-backend`, `tongue-frontend`). The AI package is a pure inference library (PyTorch + ONNX Runtime). The backend orchestrates a single `/api/analyze` pipeline (decode → ROI → per-task ONNX → assemble zh-TW user message → call ADK Gemini with locked system prompt) and exposes `/api/config/{section}` CRUD for the three editable sections. The frontend is a Gradio app with four tabs: 舌診分析 / 提示詞設定 / LLM 設定 / 模型設定.

**Tech Stack:** Python 3.11+, FastAPI, Gradio ≥4, ONNX Runtime, PyTorch (kept for ROI/preproc & label-studio), Google ADK Python SDK (`google-adk`), pyyaml, httpx, pytest, pytest-asyncio, respx, uv workspace.

**Spec:** `docs/superpowers/specs/2026-05-06-tongue-poc-design.md` (commit `12e0ca0`).

---

## File Structure (slight evolution from spec)

The spec showed a `llm/prompt_store.py` but didn't fully lay out where LLM and registry config storage modules live. This plan introduces a `stores/` subpackage so the three sections are symmetric:

```
packages/ai/
├── pyproject.toml
├── models/                              ONNX artefacts only
└── src/tongue_ai/
    ├── __init__.py
    ├── types.py                         HeadResult, ClassScore, Normalisation, BBox
    ├── preprocessing.py                 resize, normalise, BGR↔RGB, crop_bbox
    ├── detection.py                     detect_tongue() — pass-through stub
    ├── task_head.py                     TaskHead class
    ├── registry.py                      load_registry(yaml_path), Registry, errors
    └── inference.py                     run_all(image, registry)

packages/backend/
├── pyproject.toml
├── prompts/
│   ├── system.default.md                shipped 大眾版 prompt
│   └── system.current.md                gitignored
├── config/
│   ├── llm.default.yaml
│   ├── llm.current.yaml                 gitignored
│   ├── registry.default.yaml
│   └── registry.current.yaml            gitignored
└── src/tongue_backend/
    ├── __init__.py
    ├── app.py                           FastAPI factory + startup hook
    ├── pipeline.py                      analyze(image_bytes) orchestrator
    ├── stores/
    │   ├── __init__.py
    │   ├── paths.py                     path constants for default/current
    │   ├── prompt_store.py              text I/O + reset
    │   ├── llm_store.py                 YAML I/O + validation + reset
    │   └── registry_store.py            YAML I/O + validation + reset + reload
    ├── routes/
    │   ├── __init__.py
    │   ├── health.py                    GET /health
    │   ├── analyze.py                   POST /api/analyze
    │   └── config.py                    /api/config/{section} CRUD + reload
    └── llm/
        ├── __init__.py
        ├── user_message.py              HeadResult[] → zh-TW bullet list
        └── client.py                    ADK agent wrapper

packages/frontend/
├── pyproject.toml                       gradio>=4, httpx (drop streamlit)
└── src/tongue_frontend/
    ├── __init__.py
    ├── app.py                           gr.Blocks with four tabs
    ├── api.py                           thin httpx client
    └── views/
        ├── __init__.py
        ├── analyze.py
        ├── prompt_editor.py
        ├── llm_editor.py
        └── registry_editor.py
```

---

## Phase 0 — Branch & cleanup (≈5 min)

### Task 0: Create feature branch and resolve uncommitted state

**Files:**
- Modify: `packages/ai/src/tongue_ai/detection.py` (already has the cv2-import removal staged in working tree)
- Delete: `packages/frontend/src/tongue_frontend/test.md` (unrelated Upwork notes)

- [ ] **Step 1: Create feature branch from main**

```bash
git checkout -b feat/tongue-poc
git status
```

Expected: branch created; working tree shows the modified `detection.py` and untracked `test.md`.

- [ ] **Step 2: Delete the stray test.md**

```bash
rm packages/frontend/src/tongue_frontend/test.md
```

- [ ] **Step 3: Commit the existing detection.py cleanup**

```bash
git add packages/ai/src/tongue_ai/detection.py
git commit -m "chore(ai): remove unused cv2 import in detection placeholder"
```

- [ ] **Step 4: Verify tree is clean**

```bash
git status
```

Expected: "nothing to commit, working tree clean".

---

## Phase 1 — `tongue-ai` core (≈90 min)

### Task 1: Add ONNX Runtime + PyYAML to `tongue-ai`

**Files:**
- Modify: `packages/ai/pyproject.toml`

- [ ] **Step 1: Add the two dependencies**

Edit `packages/ai/pyproject.toml`, add to `dependencies = [...]`:

```toml
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "opencv-python>=4.9",
    "torch>=2.2",
    "torchvision>=0.17",
    "monai>=1.3",
    "label-studio-sdk>=1.0",
    "onnxruntime>=1.17",
    "pyyaml>=6",
]
```

- [ ] **Step 2: Sync workspace**

Run: `uv sync --all-packages`

Expected: resolves cleanly; `onnxruntime` and `pyyaml` appear in `uv.lock`.

- [ ] **Step 3: Commit**

```bash
git add packages/ai/pyproject.toml uv.lock
git commit -m "chore(ai): add onnxruntime and pyyaml deps"
```

---

### Task 2: Define core types

**Files:**
- Create: `packages/ai/src/tongue_ai/types.py`
- Test: `packages/ai/tests/test_types.py`

- [ ] **Step 1: Create the test directory and a failing test**

Create `packages/ai/tests/__init__.py` (empty). Create `packages/ai/tests/test_types.py`:

```python
from tongue_ai.types import HeadResult, ClassScore, Normalisation, BBox


def test_class_score_holds_label_and_score():
    cs = ClassScore(label="淡紅", score=0.78)
    assert cs.label == "淡紅"
    assert cs.score == 0.78


def test_head_result_single_label():
    r = HeadResult(
        task="舌色",
        head_type="single",
        predictions=[ClassScore(label="淡紅", score=0.78)],
    )
    assert r.error is None
    assert len(r.predictions) == 1


def test_head_result_with_error_has_no_predictions():
    r = HeadResult(task="舌色", head_type="single", predictions=[], error="boom")
    assert r.error == "boom"


def test_normalisation_round_trip():
    n = Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    assert n.mean[0] == 0.485
    assert len(n.std) == 3


def test_bbox_dataclass():
    b = BBox(x=10, y=20, w=100, h=120, confidence=0.9)
    assert b.x == 10
    assert b.confidence == 0.9
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_types.py -v`

Expected: collection fails with "No module named 'tongue_ai.types'".

- [ ] **Step 3: Implement types.py**

Create `packages/ai/src/tongue_ai/types.py`:

```python
"""Shared dataclasses for tongue-ai."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


HeadType = Literal["single", "multi"]


@dataclass(frozen=True)
class ClassScore:
    label: str
    score: float


@dataclass
class HeadResult:
    task: str
    head_type: HeadType
    predictions: list[ClassScore] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class Normalisation:
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_types.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/types.py packages/ai/tests/__init__.py packages/ai/tests/test_types.py
git commit -m "feat(ai): add HeadResult, ClassScore, Normalisation, BBox types"
```

---

### Task 3: Implement preprocessing helpers

**Files:**
- Create: `packages/ai/src/tongue_ai/preprocessing.py`
- Test: `packages/ai/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

Create `packages/ai/tests/test_preprocessing.py`:

```python
import numpy as np

from tongue_ai.preprocessing import (
    resize_letterbox,
    normalise,
    bgr_to_rgb,
    crop_bbox,
    to_chw_float,
)
from tongue_ai.types import BBox, Normalisation


def test_resize_letterbox_keeps_aspect():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = resize_letterbox(img, (224, 224))
    assert out.shape == (224, 224, 3)
    # Letterboxed top/bottom should be black (0)
    assert out[0, 0].tolist() == [0, 0, 0]


def test_normalise_imagenet_zero_mean():
    img = np.full((224, 224, 3), 128, dtype=np.uint8)  # mid-grey
    n = Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    out = normalise(img, n)
    assert out.dtype == np.float32
    # Mid-grey ≈ 0.5 → roughly zero-ish after normalisation
    assert abs(out.mean()) < 0.5


def test_bgr_to_rgb_swaps_channels():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 1   # B
    img[..., 2] = 3   # R
    out = bgr_to_rgb(img)
    assert out[0, 0].tolist() == [3, 0, 1]


def test_bgr_to_rgb_round_trip_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
    np.testing.assert_array_equal(img, bgr_to_rgb(bgr_to_rgb(img)))


def test_crop_bbox_returns_correct_region():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    bbox = BBox(x=10, y=20, w=30, h=40)
    out = crop_bbox(img, bbox)
    assert out.shape == (40, 30, 3)


def test_to_chw_float_changes_layout():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    out = to_chw_float(img)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run tests, expect collection failure**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_preprocessing.py -v`

Expected: ImportError on `tongue_ai.preprocessing`.

- [ ] **Step 3: Implement preprocessing.py**

Create `packages/ai/src/tongue_ai/preprocessing.py`:

```python
"""Image preprocessing for ONNX inference."""

from __future__ import annotations

import cv2
import numpy as np

from tongue_ai.types import BBox, Normalisation


def resize_letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image to ``size`` (H, W) keeping aspect, padding with black."""
    target_h, target_w = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def normalise(image: np.ndarray, params: Normalisation) -> np.ndarray:
    """Convert uint8 [0,255] → float32 normalised by mean/std."""
    arr = image.astype(np.float32) / 255.0
    mean = np.asarray(params.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(params.std, dtype=np.float32).reshape(1, 1, -1)
    return (arr - mean) / std


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Swap channel order BGR↔RGB."""
    return image[..., ::-1].copy()


def crop_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop the image to the given bbox."""
    return image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]


def to_chw_float(image: np.ndarray) -> np.ndarray:
    """HWC → NCHW with batch dim 1, float32."""
    arr = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_preprocessing.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/preprocessing.py packages/ai/tests/test_preprocessing.py
git commit -m "feat(ai): add preprocessing helpers (resize_letterbox, normalise, ...)"
```

---

### Task 4: Implement detection (pass-through stub)

**Files:**
- Modify: `packages/ai/src/tongue_ai/detection.py`
- Test: `packages/ai/tests/test_detection.py`

- [ ] **Step 1: Write failing tests**

Create `packages/ai/tests/test_detection.py`:

```python
import numpy as np

from tongue_ai.detection import detect_tongue
from tongue_ai.types import BBox


def test_detect_tongue_with_no_detector_returns_none():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_tongue(img, detector=None) is None


def test_detect_tongue_with_callable_returns_bbox():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    expected = BBox(x=10, y=10, w=50, h=50, confidence=0.9)
    result = detect_tongue(img, detector=lambda _img: expected)
    assert result == expected
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_detection.py -v`

Expected: import error or signature mismatch — current `detect_tongue` ignores arguments.

- [ ] **Step 3: Replace detection.py**

Replace contents of `packages/ai/src/tongue_ai/detection.py`:

```python
"""Tongue ROI detection — pluggable detector abstraction."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from tongue_ai.types import BBox


Detector = Callable[[np.ndarray], Optional[BBox]]


def detect_tongue(image: np.ndarray, detector: Detector | None) -> BBox | None:
    """Run the configured detector. Returns None when no detector is wired up,
    so the pipeline falls back to using the whole image."""
    if detector is None:
        return None
    return detector(image)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_detection.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/detection.py packages/ai/tests/test_detection.py
git commit -m "feat(ai): make detect_tongue accept pluggable detector callable"
```

---

### Task 5: Implement TaskHead

**Files:**
- Create: `packages/ai/src/tongue_ai/task_head.py`
- Test: `packages/ai/tests/test_task_head.py`

- [ ] **Step 1: Write failing tests**

Create `packages/ai/tests/test_task_head.py`:

```python
from unittest.mock import MagicMock

import numpy as np

from tongue_ai.task_head import TaskHead
from tongue_ai.types import Normalisation


def _fake_session(logits: np.ndarray) -> MagicMock:
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = [logits]
    return session


def test_single_label_decodes_argmax():
    logits = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
    head = TaskHead(
        session=_fake_session(logits),
        name="舌色",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        class_names=["A", "B", "C"],
    )
    result = head.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.task == "舌色"
    assert result.head_type == "single"
    assert len(result.predictions) == 1
    assert result.predictions[0].label == "B"
    assert abs(result.predictions[0].score - 0.7) < 1e-5


def test_multi_label_thresholds():
    # Sigmoid probs: [0.9, 0.4, 0.95]
    logits = np.array([[0.9, 0.4, 0.95]], dtype=np.float32)
    head = TaskHead(
        session=_fake_session(logits),
        name="舌質",
        head_type="multi",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.0]*3, std=[1.0]*3),
        class_names=["齒痕", "胖大", "嫩"],
        threshold=0.5,
        already_probs=True,
    )
    result = head.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    labels = [p.label for p in result.predictions]
    assert labels == ["齒痕", "嫩"]


def test_predict_catches_exception_returns_error():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input"
    session.run.side_effect = RuntimeError("bad ONNX")
    head = TaskHead(
        session=session,
        name="舌色",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.0]*3, std=[1.0]*3),
        class_names=["A", "B"],
    )
    result = head.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.error is not None
    assert "bad ONNX" in result.error
    assert result.predictions == []
```

- [ ] **Step 2: Run tests, expect import failure**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_task_head.py -v`

Expected: import error.

- [ ] **Step 3: Implement task_head.py**

Create `packages/ai/src/tongue_ai/task_head.py`:

```python
"""TaskHead — wraps one ONNX model + its preprocessing + label decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tongue_ai.preprocessing import (
    bgr_to_rgb,
    normalise,
    resize_letterbox,
    to_chw_float,
)
from tongue_ai.types import ClassScore, HeadResult, HeadType, Normalisation


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TaskHead:
    session: Any                 # ort.InferenceSession or compatible mock
    name: str
    head_type: HeadType
    input_size: tuple[int, int]
    normalise: Normalisation
    class_names: list[str]
    threshold: float = 0.5
    already_probs: bool = False  # if True, skip softmax/sigmoid

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        try:
            tensor = self._preprocess(image_bgr)
            input_name = self.session.get_inputs()[0].name
            (logits,) = self.session.run(None, {input_name: tensor})
            return self._decode(logits)
        except Exception as exc:
            return HeadResult(
                task=self.name,
                head_type=self.head_type,
                predictions=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = bgr_to_rgb(image_bgr)
        sized = resize_letterbox(rgb, self.input_size)
        floated = normalise(sized, self.normalise)
        return to_chw_float(floated)

    def _decode(self, logits: np.ndarray) -> HeadResult:
        # Take batch index 0
        scores = logits[0]
        if self.head_type == "single":
            probs = scores if self.already_probs else _softmax(scores)
            idx = int(np.argmax(probs))
            return HeadResult(
                task=self.name,
                head_type="single",
                predictions=[
                    ClassScore(label=self.class_names[idx], score=float(probs[idx]))
                ],
            )
        # multi
        probs = scores if self.already_probs else _sigmoid(scores)
        picks = [
            ClassScore(label=self.class_names[i], score=float(probs[i]))
            for i in range(len(self.class_names))
            if probs[i] >= self.threshold
        ]
        return HeadResult(task=self.name, head_type="multi", predictions=picks)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_task_head.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/task_head.py packages/ai/tests/test_task_head.py
git commit -m "feat(ai): add TaskHead with single/multi-label decoding and error containment"
```

---

### Task 6: Implement Registry loader

**Files:**
- Create: `packages/ai/src/tongue_ai/registry.py`
- Test: `packages/ai/tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

Create `packages/ai/tests/test_registry.py`:

```python
from pathlib import Path

import pytest

from tongue_ai.registry import Registry, RegistryError, load_registry


SAMPLE_YAML = """
detector: null
heads:
  - task: 舌色
    head_type: single
    onnx_path: tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["淡","淡紅","微紅","紅","絳","青紫","暗"]
  - task: 舌質
    head_type: multi
    threshold: 0.5
    onnx_path: tongue_body.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["齒痕","胖大","瘦薄","老","嫩","無異常"]
"""


def _write_yaml(tmp_path: Path, body: str = SAMPLE_YAML) -> Path:
    (tmp_path / "tongue_color.onnx").write_bytes(b"")     # placeholder
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    return p


def test_load_registry_two_heads(tmp_path, monkeypatch):
    yaml_path = _write_yaml(tmp_path)
    # Stub ort.InferenceSession so we don't actually load ONNX bytes
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    registry = load_registry(yaml_path)
    assert isinstance(registry, Registry)
    assert [h.name for h in registry.heads] == ["舌色", "舌質"]
    assert registry.detector is None


def test_load_registry_missing_required_key(tmp_path, monkeypatch):
    yaml_path = _write_yaml(
        tmp_path,
        """
heads:
  - task: 舌色
    head_type: single
""",
    )
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(yaml_path)
    assert "onnx_path" in str(e.value)


def test_load_registry_missing_onnx_file(tmp_path, monkeypatch):
    body = SAMPLE_YAML.replace("tongue_color.onnx", "missing.onnx")
    p = tmp_path / "registry.yaml"
    p.write_text(body)
    (tmp_path / "tongue_body.onnx").write_bytes(b"")
    import tongue_ai.registry as reg
    monkeypatch.setattr(reg, "_make_session", lambda _path: object())
    with pytest.raises(RegistryError) as e:
        load_registry(p)
    assert "missing.onnx" in str(e.value)
```

- [ ] **Step 2: Run tests, expect import failure**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_registry.py -v`

Expected: import error.

- [ ] **Step 3: Implement registry.py**

Create `packages/ai/src/tongue_ai/registry.py`:

```python
"""Registry loader — turns a YAML config into a list of TaskHead objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tongue_ai.task_head import TaskHead
from tongue_ai.types import Normalisation


class RegistryError(ValueError):
    """Raised when a registry YAML is malformed."""


@dataclass
class Registry:
    heads: list[TaskHead]
    detector: object | None  # detector callable; None if disabled


REQUIRED_HEAD_KEYS = {"task", "head_type", "onnx_path", "input_size", "normalise", "class_names"}


def _make_session(onnx_path: Path):
    """Indirection so tests can monkeypatch without importing onnxruntime."""
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path))


def load_registry(yaml_path: str | Path) -> Registry:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise RegistryError(f"Registry YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(data, dict):
        raise RegistryError("registry root must be a mapping")

    heads_data = data.get("heads")
    if not isinstance(heads_data, list) or not heads_data:
        raise RegistryError("registry must define a non-empty 'heads' list")

    base_dir = yaml_path.parent
    heads = [_build_head(item, base_dir) for item in heads_data]

    detector = None  # POC: detector wiring deferred (see Task 30)
    return Registry(heads=heads, detector=detector)


def _build_head(raw: Any, base_dir: Path) -> TaskHead:
    if not isinstance(raw, dict):
        raise RegistryError(f"head entry must be a mapping, got {type(raw).__name__}")
    missing = REQUIRED_HEAD_KEYS - raw.keys()
    if missing:
        raise RegistryError(f"head '{raw.get('task','?')}' missing keys: {sorted(missing)}")

    onnx_path = (base_dir / raw["onnx_path"]).resolve()
    if not onnx_path.exists():
        raise RegistryError(f"onnx_path not found on disk: {onnx_path} (config: {raw['onnx_path']})")

    input_size = tuple(raw["input_size"])
    if len(input_size) != 2:
        raise RegistryError(f"input_size must be [H, W], got {raw['input_size']}")

    norm = raw["normalise"]
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        raise RegistryError(f"head '{raw['task']}': normalise must define mean and std")

    class_names = list(raw["class_names"])
    if not class_names:
        raise RegistryError(f"head '{raw['task']}': class_names must be non-empty")

    head_type = raw["head_type"]
    if head_type not in {"single", "multi"}:
        raise RegistryError(f"head '{raw['task']}': head_type must be 'single' or 'multi'")

    return TaskHead(
        session=_make_session(onnx_path),
        name=raw["task"],
        head_type=head_type,
        input_size=input_size,
        normalise=Normalisation(mean=norm["mean"], std=norm["std"]),
        class_names=class_names,
        threshold=float(raw.get("threshold", 0.5)),
        already_probs=bool(raw.get("already_probs", False)),
    )
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_registry.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/ai/src/tongue_ai/registry.py packages/ai/tests/test_registry.py
git commit -m "feat(ai): add Registry loader with validation errors"
```

---

### Task 7: Implement run_all inference loop

**Files:**
- Create: `packages/ai/src/tongue_ai/inference.py`
- Test: `packages/ai/tests/test_inference.py`
- Modify: `packages/ai/src/tongue_ai/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `packages/ai/tests/test_inference.py`:

```python
from dataclasses import dataclass

import numpy as np

from tongue_ai.inference import run_all
from tongue_ai.registry import Registry
from tongue_ai.types import HeadResult, ClassScore


@dataclass
class _StubHead:
    name: str
    head_type: str = "single"
    raises: bool = False

    def predict(self, _image):
        if self.raises:
            raise RuntimeError("kaboom")
        return HeadResult(
            task=self.name,
            head_type=self.head_type,
            predictions=[ClassScore(label="L", score=0.9)],
        )


def test_run_all_returns_one_result_per_head_in_order():
    registry = Registry(
        heads=[_StubHead("舌色"), _StubHead("舌質"), _StubHead("舌苔顏色")],
        detector=None,
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    results = run_all(image, registry)
    assert [r.task for r in results] == ["舌色", "舌質", "舌苔顏色"]


def test_run_all_continues_when_one_head_raises():
    registry = Registry(
        heads=[_StubHead("舌色"), _StubHead("舌質", raises=True), _StubHead("舌苔顏色")],
        detector=None,
    )
    results = run_all(np.zeros((10, 10, 3), dtype=np.uint8), registry)
    assert results[1].error is not None
    assert results[1].predictions == []
    assert results[0].error is None
    assert results[2].error is None
```

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run --package tongue-ai pytest packages/ai/tests/test_inference.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement inference.py**

Create `packages/ai/src/tongue_ai/inference.py`:

```python
"""Run all configured task heads on one image."""

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
            # belt-and-braces: TaskHead.predict already catches, but a buggy head
            # in a future implementation must not break the loop
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

- [ ] **Step 4: Wire exports**

Replace `packages/ai/src/tongue_ai/__init__.py`:

```python
"""Tongue AI -- detection, recognition, and classification models."""

from tongue_ai.types import BBox, ClassScore, HeadResult, Normalisation
from tongue_ai.task_head import TaskHead
from tongue_ai.registry import Registry, RegistryError, load_registry
from tongue_ai.inference import run_all
from tongue_ai.detection import detect_tongue

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "ClassScore",
    "HeadResult",
    "Normalisation",
    "TaskHead",
    "Registry",
    "RegistryError",
    "load_registry",
    "run_all",
    "detect_tongue",
]
```

- [ ] **Step 5: Run all tongue-ai tests**

Run: `uv run --package tongue-ai pytest packages/ai/tests -v`

Expected: all green (≥17 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/ai/src/tongue_ai/inference.py packages/ai/src/tongue_ai/__init__.py packages/ai/tests/test_inference.py
git commit -m "feat(ai): add run_all inference loop with per-head error containment"
```

---

## Phase 2 — `tongue-backend` (≈3 hours)

### Task 8: Update backend dependencies

**Files:**
- Modify: `packages/backend/pyproject.toml`

- [ ] **Step 1: Edit dependencies**

Update `packages/backend/pyproject.toml`:

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
    "google-adk>=0.2",
    "tongue-ai",
]
```

> **Note:** Verify `google-adk` is the correct PyPI name for Google's Agent Development Kit when implementing. If the SDK has been renamed, swap to whichever package exposes `Agent(model=..., instructions=...)` and `agent.run(...)`. The `llm/client.py` wrapper isolates the SDK so the rest of the code is unaffected.

- [ ] **Step 2: Sync workspace**

Run: `uv sync --all-packages`

Expected: resolves; new packages appear in `uv.lock`.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/pyproject.toml uv.lock
git commit -m "chore(backend): add pyyaml and google-adk deps"
```

---

### Task 9: Add the default config files (shipped, read-only)

**Files:**
- Create: `packages/backend/prompts/system.default.md`
- Create: `packages/backend/config/llm.default.yaml`
- Create: `packages/backend/config/registry.default.yaml`
- Modify: `packages/backend/pyproject.toml` (include data files)
- Modify: `.gitignore`

- [ ] **Step 1: Write the system prompt default**

Create `packages/backend/prompts/system.default.md` with the verbatim 大眾版 prompt. Content:

```markdown
你是一個中醫舌診辨證系統，請嚴格依據以下規則執行。

====================
【一、對應表（唯一依據）】

所有病機判斷「只能」來自以下對應表。
禁止使用任何外部中醫知識。
若對應表與常識衝突，以對應表為準。

舌色：
淡 → 氣虛、血虛、陽虛
淡紅 → 正常、輕微偏虛
微紅 → 陰虛、化熱
紅 → 陰虛、化熱
絳 → 陰虛、化熱
青紫 → 血瘀、氣滯
黯 → 血瘀、氣滯、陽虛

舌質：
齒痕 → 氣虛、痰濕
胖大 → 氣虛、陽虛、痰濕
瘦薄 → 氣虛、血虛、陰虛
老 → 痰濕、血瘀、化熱
嫩 → 氣虛、血虛、陰虛、陽虛

舌態：
偏斜 → 痰濕
無異常 → 正常

朱點：
舌尖 → 化熱
兩側 → 化熱
中根 → 化熱
無 → 正常

瘀斑瘀點：
舌尖 → 血瘀
兩側 → 血瘀
中根 → 血瘀
無 → 正常

裂紋：
苔裂 → 陰虛
質裂 → 陰虛
無異常 → 正常

舌津：
少 → 陰虛
正常 → 正常
多 → 陽虛、痰濕

舌苔顏色：
白 → 痰濕
黃 → 濕熱、化熱
灰 → 痰濕
黑 → 陽虛、化熱
染 → 忽略

舌苔厚薄：
無 → 氣虛、陰虛
少 → 氣虛、陰虛
薄少 → 陰虛、氣虛
薄 → 正常
微 → 忽略
微厚 → 痰濕
厚 → 痰濕（高特異性）
膩 → 痰濕（高特異性）
腐 → 痰濕（高特異性）
剝 → 陰虛
其他 → 忽略

舌苔根：
有根 → 胃氣尚存（不轉證素）
無根 → 氣虛

舌下絡脈：
怒張 → 氣滯、血瘀
曲張 → 血瘀
囊柱囊泡 → 血瘀
瘀血絲 → 血瘀
舌下肉埠腫脹 → 血瘀、痰濕、濕熱
無異常 → 正常

====================
【二、證素限制】

僅允許以下證素：
氣虛、陽虛、陰虛、血虛、痰濕、濕熱、氣滯、血瘀、化熱

====================
【三、候選建立】

1. 每一個舌象都必須對應表映射
2. 列出所有候選證素
3. 不可提前刪除

====================
【四、特異性權重】

高特異性（優先成立）：
- 裂紋 → 陰虛
- 厚／膩／腐苔 → 痰濕
- 舌下絡脈異常 → 氣滯、血瘀

低特異性（可被壓制）：
- 齒痕 → 氣虛
- 胖大 → 氣虛
- 淡紅 → 正常
- 嫩 → 虛

====================
【五、主證判定】

主證 = 支持最多 + 一致性最高 + 特異性高
痰濕若證據強 → 可壓制氣虛

====================
【六、病機推導（僅限）】

氣虛 → 陽虛
陽虛 → 痰濕
痰濕 → 氣滯
氣滯 → 血瘀
陰虛 → 血虛
痰濕久 → 濕熱

====================
【七、組合辨證】

痰濕 + 化熱 → 濕熱

====================
【八、衝突處理】

可解釋 → 保留
不可解釋 → 降級或刪除

====================
【九、輸出格式（大眾版）】

請用一般大學生能理解的語言輸出：

1. 主要中醫體質
（用一句話描述，例如：濕氣偏重、循環較差等）

2. 次要中醫體質
（列出1–2個次要傾向，使用白話）

3. 體質說明
（用白話說明：
- 身體目前的整體狀態
- 為什麼會形成這種情況（簡單因果）
- 避免使用專業術語，必要時轉成生活化描述）

4. 證素列表（保留專業內容）

5. 警語
此為AI自動生成，不具醫療建議，若有疾病或疑問，應向專業中醫師諮詢。

====================
【十、核心原則】

建立一個「能解釋全部舌象」的最小病機模型。
```

- [ ] **Step 2: Write the LLM default**

Create `packages/backend/config/llm.default.yaml`:

```yaml
model: gemini-2.0-flash
temperature: 0.2
max_tokens: 2048
top_p: 0.9
```

- [ ] **Step 3: Write the registry default**

Create `packages/backend/config/registry.default.yaml`:

```yaml
# Tongue diagnosis registry — paths are relative to this file's directory.
# ONNX artefacts live in ../../ai/models/. Edit and Apply & Reload Models from the UI.

detector: null  # set to a mapping like {onnx_path: ../../ai/models/detect_tongue.onnx, ...}

heads:
  - task: 舌色
    head_type: single
    onnx_path: ../../ai/models/tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["淡","淡紅","微紅","紅","絳","青紫","暗"]

  - task: 舌質
    head_type: multi
    threshold: 0.5
    onnx_path: ../../ai/models/tongue_body.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["齒痕","胖大","瘦薄","老","嫩","無異常"]

  - task: 舌態
    head_type: single
    onnx_path: ../../ai/models/tongue_movement.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["偏斜","無異常"]

  - task: 朱點部位
    head_type: multi
    threshold: 0.5
    onnx_path: ../../ai/models/red_dots.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["舌尖","兩側","中根","無"]

  - task: 瘀斑瘀點部位
    head_type: multi
    threshold: 0.5
    onnx_path: ../../ai/models/ecchymosis.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["舌尖","兩側","中根","無"]

  - task: 裂紋
    head_type: single
    onnx_path: ../../ai/models/fissures.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["苔裂","質裂","無異常"]

  - task: 舌津
    head_type: single
    onnx_path: ../../ai/models/moisture.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["少","正常","多"]

  - task: 舌苔顏色
    head_type: multi
    threshold: 0.5
    onnx_path: ../../ai/models/coating_color.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["白","黃","灰","黑","染"]

  - task: 舌苔厚薄
    head_type: single
    onnx_path: ../../ai/models/coating_thickness.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["無","少","薄少","薄","微","微厚","厚"]

  - task: 舌苔根
    head_type: single
    onnx_path: ../../ai/models/coating_root.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["有根","無根","無異常"]

  - task: 舌下絡脈
    head_type: multi
    threshold: 0.5
    onnx_path: ../../ai/models/sublingual.onnx
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: ["怒張","曲張","囊柱囊泡","瘀血絲","舌下肉埠腫脹","無異常"]
```

- [ ] **Step 4: Update .gitignore**

Append to root `.gitignore`:

```gitignore

# Editable runtime config — written by the API, not source-controlled
packages/backend/prompts/system.current.md
packages/backend/config/llm.current.yaml
packages/backend/config/registry.current.yaml
```

- [ ] **Step 5: Commit**

```bash
git add packages/backend/prompts/system.default.md \
        packages/backend/config/llm.default.yaml \
        packages/backend/config/registry.default.yaml \
        .gitignore
git commit -m "feat(backend): add default prompt, llm, and registry config files"
```

---

### Task 10: Path constants

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/__init__.py`
- Create: `packages/backend/src/tongue_backend/stores/paths.py`
- Test: `packages/backend/tests/__init__.py` (empty)
- Test: `packages/backend/tests/test_paths.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/__init__.py` (empty), then `packages/backend/tests/test_paths.py`:

```python
from tongue_backend.stores.paths import (
    PROMPT_DEFAULT,
    PROMPT_CURRENT,
    LLM_DEFAULT,
    LLM_CURRENT,
    REGISTRY_DEFAULT,
    REGISTRY_CURRENT,
)


def test_paths_point_to_real_default_files():
    assert PROMPT_DEFAULT.exists(), PROMPT_DEFAULT
    assert LLM_DEFAULT.exists(), LLM_DEFAULT
    assert REGISTRY_DEFAULT.exists(), REGISTRY_DEFAULT


def test_current_paths_are_siblings_of_defaults():
    assert PROMPT_CURRENT.parent == PROMPT_DEFAULT.parent
    assert LLM_CURRENT.parent == LLM_DEFAULT.parent
    assert REGISTRY_CURRENT.parent == REGISTRY_DEFAULT.parent
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_paths.py -v`

Expected: import error.

- [ ] **Step 3: Create stores package + paths.py**

Create `packages/backend/src/tongue_backend/stores/__init__.py` (empty).

Create `packages/backend/src/tongue_backend/stores/paths.py`:

```python
"""Filesystem locations of default and current config files."""

from __future__ import annotations

from pathlib import Path


# packages/backend/src/tongue_backend/stores/paths.py → packages/backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[3]

PROMPT_DEFAULT = _BACKEND_ROOT / "prompts" / "system.default.md"
PROMPT_CURRENT = _BACKEND_ROOT / "prompts" / "system.current.md"

LLM_DEFAULT = _BACKEND_ROOT / "config" / "llm.default.yaml"
LLM_CURRENT = _BACKEND_ROOT / "config" / "llm.current.yaml"

REGISTRY_DEFAULT = _BACKEND_ROOT / "config" / "registry.default.yaml"
REGISTRY_CURRENT = _BACKEND_ROOT / "config" / "registry.current.yaml"
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_paths.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/__init__.py \
        packages/backend/src/tongue_backend/stores/paths.py \
        packages/backend/tests/__init__.py \
        packages/backend/tests/test_paths.py
git commit -m "feat(backend): add stores.paths constants for default/current configs"
```

---

### Task 11: Prompt store (text I/O)

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/prompt_store.py`
- Test: `packages/backend/tests/test_prompt_store.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_prompt_store.py`:

```python
from pathlib import Path

import pytest

from tongue_backend.stores import prompt_store


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "system.default.md"
    current = tmp_path / "system.current.md"
    default.write_text("DEFAULT PROMPT")
    monkeypatch.setattr(prompt_store, "PROMPT_DEFAULT", default)
    monkeypatch.setattr(prompt_store, "PROMPT_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    default, current = tmp_paths
    text = prompt_store.load_current()
    assert text == "DEFAULT PROMPT"
    assert current.read_text() == "DEFAULT PROMPT"


def test_save_persists_then_load_returns_same(tmp_paths):
    prompt_store.save("MY EDIT")
    assert prompt_store.load_current() == "MY EDIT"


def test_reset_overwrites_current_with_default(tmp_paths):
    prompt_store.save("MY EDIT")
    prompt_store.reset()
    assert prompt_store.load_current() == "DEFAULT PROMPT"


def test_status_reports_is_default_flag(tmp_paths):
    prompt_store.reset()
    status = prompt_store.status()
    assert status["is_default"] is True
    prompt_store.save("MY EDIT")
    status = prompt_store.status()
    assert status["is_default"] is False
    assert status["content"] == "MY EDIT"
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_prompt_store.py -v`

Expected: import error.

- [ ] **Step 3: Implement prompt_store.py**

Create `packages/backend/src/tongue_backend/stores/prompt_store.py`:

```python
"""Read/write the system-prompt 'current' file with reset semantics."""

from __future__ import annotations

from pathlib import Path

from tongue_backend.stores.paths import PROMPT_DEFAULT, PROMPT_CURRENT


def _ensure_current() -> Path:
    """If current is missing, copy from default."""
    if not PROMPT_CURRENT.exists():
        PROMPT_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_CURRENT.write_text(PROMPT_DEFAULT.read_text())
    return PROMPT_CURRENT


def load_current() -> str:
    return _ensure_current().read_text()


def save(content: str) -> None:
    PROMPT_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_CURRENT.write_text(content)


def reset() -> None:
    save(PROMPT_DEFAULT.read_text())


def status() -> dict:
    content = load_current()
    return {
        "content": content,
        "is_default": content == PROMPT_DEFAULT.read_text(),
        "mtime": PROMPT_CURRENT.stat().st_mtime,
    }
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_prompt_store.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/prompt_store.py \
        packages/backend/tests/test_prompt_store.py
git commit -m "feat(backend): add prompt_store for system-prompt CRUD"
```

---

### Task 12: LLM config store (YAML + validation)

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/llm_store.py`
- Test: `packages/backend/tests/test_llm_store.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_llm_store.py`:

```python
import pytest

from tongue_backend.stores import llm_store


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "llm.default.yaml"
    current = tmp_path / "llm.current.yaml"
    default.write_text(
        "model: gemini-2.0-flash\ntemperature: 0.2\nmax_tokens: 2048\ntop_p: 0.9\n"
    )
    monkeypatch.setattr(llm_store, "LLM_DEFAULT", default)
    monkeypatch.setattr(llm_store, "LLM_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    cfg = llm_store.load_current()
    assert cfg["model"] == "gemini-2.0-flash"
    assert cfg["temperature"] == 0.2


def test_save_then_load_round_trip(tmp_paths):
    llm_store.save("model: gemini-2.0-pro\ntemperature: 0.5\nmax_tokens: 1024\ntop_p: 0.9\n")
    cfg = llm_store.load_current()
    assert cfg["model"] == "gemini-2.0-pro"


def test_save_rejects_invalid_temperature(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("model: x\ntemperature: 99\nmax_tokens: 1024\ntop_p: 0.9\n")
    assert "temperature" in str(e.value)


def test_save_rejects_malformed_yaml(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("model: : :")
    assert "yaml" in str(e.value).lower()


def test_save_rejects_missing_model(tmp_paths):
    with pytest.raises(llm_store.ValidationError) as e:
        llm_store.save("temperature: 0.2\nmax_tokens: 1024\ntop_p: 0.9\n")
    assert "model" in str(e.value)


def test_reset_restores_default(tmp_paths):
    llm_store.save("model: x\ntemperature: 0.5\nmax_tokens: 1024\ntop_p: 0.9\n")
    llm_store.reset()
    cfg = llm_store.load_current()
    assert cfg["model"] == "gemini-2.0-flash"
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_llm_store.py -v`

Expected: import error.

- [ ] **Step 3: Implement llm_store.py**

Create `packages/backend/src/tongue_backend/stores/llm_store.py`:

```python
"""LLM config: YAML I/O + validation + reset."""

from __future__ import annotations

from pathlib import Path

import yaml

from tongue_backend.stores.paths import LLM_DEFAULT, LLM_CURRENT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


def _ensure_current() -> Path:
    if not LLM_CURRENT.exists():
        LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        LLM_CURRENT.write_text(LLM_DEFAULT.read_text())
    return LLM_CURRENT


def load_current() -> dict:
    text = _ensure_current().read_text()
    return yaml.safe_load(text) or {}


def save(content: str) -> None:
    """Persist a YAML string after validation."""
    parsed = _validate(content)
    LLM_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    LLM_CURRENT.write_text(content)
    return parsed


def reset() -> None:
    LLM_CURRENT.write_text(LLM_DEFAULT.read_text())


def status() -> dict:
    content = _ensure_current().read_text()
    return {
        "content": content,
        "is_default": content == LLM_DEFAULT.read_text(),
        "mtime": LLM_CURRENT.stat().st_mtime,
    }


def _validate(content: str) -> dict:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("LLM config root must be a mapping")
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("model must be a non-empty string")
    temp = data.get("temperature", 0.0)
    if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
        raise ValidationError("temperature must be a number in [0, 2]")
    max_tokens = data.get("max_tokens", 1024)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValidationError("max_tokens must be a positive integer")
    top_p = data.get("top_p", 1.0)
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValidationError("top_p must be in (0, 1]")
    return data
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_llm_store.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/llm_store.py \
        packages/backend/tests/test_llm_store.py
git commit -m "feat(backend): add llm_store with YAML validation"
```

---

### Task 13: Registry store (validation only — reload deferred to Task 18)

**Files:**
- Create: `packages/backend/src/tongue_backend/stores/registry_store.py`
- Test: `packages/backend/tests/test_registry_store.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_registry_store.py`:

```python
import pytest

from tongue_backend.stores import registry_store


VALID_YAML = """
detector: null
heads:
  - task: 舌色
    head_type: single
    onnx_path: tongue_color.onnx
    input_size: [224, 224]
    normalise:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    class_names: ["A","B"]
"""


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    default = tmp_path / "registry.default.yaml"
    current = tmp_path / "registry.current.yaml"
    default.write_text(VALID_YAML)
    (tmp_path / "tongue_color.onnx").write_bytes(b"")
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", default)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", current)
    return default, current


def test_load_current_copies_default_when_missing(tmp_paths):
    text = registry_store.load_current_text()
    assert "舌色" in text


def test_save_persists_valid_yaml(tmp_paths):
    registry_store.save(VALID_YAML)
    assert "舌色" in registry_store.load_current_text()


def test_save_rejects_malformed_yaml(tmp_paths):
    with pytest.raises(registry_store.ValidationError):
        registry_store.save("heads: : bad")


def test_save_rejects_missing_keys(tmp_paths):
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save("""
heads:
  - task: 舌色
    head_type: single
""")
    assert "onnx_path" in str(e.value)


def test_save_rejects_missing_onnx_file(tmp_paths):
    bad = VALID_YAML.replace("tongue_color.onnx", "missing.onnx")
    with pytest.raises(registry_store.ValidationError) as e:
        registry_store.save(bad)
    assert "missing.onnx" in str(e.value)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_registry_store.py -v`

Expected: import error.

- [ ] **Step 3: Implement registry_store.py (validation only — reload comes in Task 18)**

Create `packages/backend/src/tongue_backend/stores/registry_store.py`:

```python
"""Registry config: YAML I/O + validation + reset.

Note: this module persists the YAML and validates it without loading any ONNX
sessions. The expensive reload-into-process step lives in app.state.registry
and is exercised via app.reload_registry() (Task 18).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tongue_backend.stores.paths import REGISTRY_DEFAULT, REGISTRY_CURRENT


class ValidationError(ValueError):
    """Raised on PUT when YAML or values are invalid."""


REQUIRED_HEAD_KEYS = {"task", "head_type", "onnx_path", "input_size", "normalise", "class_names"}


def _ensure_current() -> Path:
    if not REGISTRY_CURRENT.exists():
        REGISTRY_CURRENT.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_CURRENT.write_text(REGISTRY_DEFAULT.read_text())
    return REGISTRY_CURRENT


def load_current_text() -> str:
    return _ensure_current().read_text()


def save(content: str) -> None:
    _validate(content)
    REGISTRY_CURRENT.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_CURRENT.write_text(content)


def reset() -> None:
    REGISTRY_CURRENT.write_text(REGISTRY_DEFAULT.read_text())


def status() -> dict:
    content = load_current_text()
    return {
        "content": content,
        "is_default": content == REGISTRY_DEFAULT.read_text(),
        "mtime": REGISTRY_CURRENT.stat().st_mtime,
    }


def _validate(content: str) -> None:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(f"yaml parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError("registry root must be a mapping")
    heads = data.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValidationError("registry must define a non-empty 'heads' list")

    base_dir = REGISTRY_CURRENT.parent
    for raw in heads:
        if not isinstance(raw, dict):
            raise ValidationError(f"head entry must be a mapping, got {type(raw).__name__}")
        missing = REQUIRED_HEAD_KEYS - raw.keys()
        if missing:
            raise ValidationError(
                f"head '{raw.get('task','?')}' missing keys: {sorted(missing)}"
            )
        onnx_path = (base_dir / raw["onnx_path"]).resolve()
        if not onnx_path.exists():
            raise ValidationError(f"onnx_path not found on disk: {raw['onnx_path']}")
        if raw["head_type"] not in {"single", "multi"}:
            raise ValidationError(
                f"head '{raw['task']}': head_type must be 'single' or 'multi'"
            )
        if not list(raw.get("class_names") or []):
            raise ValidationError(f"head '{raw['task']}': class_names must be non-empty")
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_registry_store.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/stores/registry_store.py \
        packages/backend/tests/test_registry_store.py
git commit -m "feat(backend): add registry_store with YAML validation"
```

---

### Task 14: User-message builder (golden tests)

**Files:**
- Create: `packages/backend/src/tongue_backend/llm/__init__.py` (empty)
- Create: `packages/backend/src/tongue_backend/llm/user_message.py`
- Test: `packages/backend/tests/test_user_message.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_user_message.py`:

```python
from tongue_ai.types import ClassScore, HeadResult

from tongue_backend.llm.user_message import build


def _hr(task, head_type, preds, error=None):
    return HeadResult(
        task=task,
        head_type=head_type,
        predictions=[ClassScore(label=l, score=s) for l, s in preds],
        error=error,
    )


def test_builds_zh_tw_bullet_list_with_confidences():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi",  [("齒痕", 0.84), ("嫩", 0.71)]),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    msg = build(heads)
    assert msg.startswith("本次舌診判讀結果：")
    assert "- 舌色：淡紅（0.78）" in msg
    assert "- 舌質：齒痕（0.84）、嫩（0.71）" in msg
    assert "- 舌態：無異常（0.91）" in msg
    assert msg.rstrip().endswith("請依規則輸出大眾版報告。")


def test_skips_heads_with_error():
    heads = [
        _hr("舌色", "single", [("淡紅", 0.78)]),
        _hr("舌質", "multi", [], error="boom"),
        _hr("舌態", "single", [("無異常", 0.91)]),
    ]
    msg = build(heads)
    assert "舌質" not in msg
    assert "舌色" in msg and "舌態" in msg


def test_handles_empty_heads_with_no_data_marker():
    msg = build([])
    assert "（無可用判讀資料）" in msg
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_user_message.py -v`

Expected: import error.

- [ ] **Step 3: Implement user_message.py**

Create `packages/backend/src/tongue_backend/llm/__init__.py` (empty).

Create `packages/backend/src/tongue_backend/llm/user_message.py`:

```python
"""Build the per-request user message for Gemini from HeadResult[]."""

from __future__ import annotations

from tongue_ai.types import HeadResult


HEADER = "本次舌診判讀結果："
FOOTER = "請依規則輸出大眾版報告。"
EMPTY_LINE = "- （無可用判讀資料）"


def build(heads: list[HeadResult]) -> str:
    lines: list[str] = []
    for h in heads:
        if h.error or not h.predictions:
            continue
        rendered = "、".join(f"{p.label}（{p.score:.2f}）" for p in h.predictions)
        lines.append(f"- {h.task}：{rendered}")

    body = "\n".join(lines) if lines else EMPTY_LINE
    return f"{HEADER}\n\n{body}\n\n{FOOTER}"
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_user_message.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/llm/__init__.py \
        packages/backend/src/tongue_backend/llm/user_message.py \
        packages/backend/tests/test_user_message.py
git commit -m "feat(backend): add llm.user_message.build (golden tested)"
```

---

### Task 15: ADK Gemini client wrapper

**Files:**
- Create: `packages/backend/src/tongue_backend/llm/client.py`
- Test: `packages/backend/tests/test_llm_client.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_llm_client.py`:

```python
from unittest.mock import MagicMock

import pytest

from tongue_backend.llm import client as llm_client


def test_run_calls_sdk_with_system_user_and_config(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.return_value = "MOCK COMMENT"
    fake_agent_factory = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(llm_client, "_make_agent", fake_agent_factory)

    out = llm_client.run(
        system="SYS",
        user="USR",
        config={"model": "gemini-2.0-flash", "temperature": 0.2, "max_tokens": 1024, "top_p": 0.9},
    )
    assert out == "MOCK COMMENT"
    fake_agent_factory.assert_called_once_with(
        model="gemini-2.0-flash", instructions="SYS",
        temperature=0.2, max_tokens=1024, top_p=0.9,
    )
    fake_agent.run.assert_called_once_with("USR")


def test_run_returns_error_stamp_on_exception(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.side_effect = TimeoutError("upstream timeout")
    monkeypatch.setattr(llm_client, "_make_agent", lambda **_: fake_agent)

    out = llm_client.run(
        system="x", user="y",
        config={"model": "m", "temperature": 0.0, "max_tokens": 1, "top_p": 1.0},
    )
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "upstream timeout" in out


def test_run_returns_error_stamp_on_empty_response(monkeypatch):
    fake_agent = MagicMock()
    fake_agent.run.return_value = ""
    monkeypatch.setattr(llm_client, "_make_agent", lambda **_: fake_agent)

    out = llm_client.run(
        system="x", user="y",
        config={"model": "m", "temperature": 0.0, "max_tokens": 1, "top_p": 1.0},
    )
    assert out.startswith("⚠ 醫師建議產生失敗：")
    assert "模型未產生回應" in out
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_llm_client.py -v`

Expected: import error.

- [ ] **Step 3: Implement client.py**

Create `packages/backend/src/tongue_backend/llm/client.py`:

```python
"""Thin wrapper around Google ADK so the rest of the code can call run(system, user, config)."""

from __future__ import annotations

ERROR_STAMP = "⚠ 醫師建議產生失敗："


def _make_agent(*, model: str, instructions: str, temperature: float,
                max_tokens: int, top_p: float):
    """Indirection so tests can monkeypatch without importing google-adk."""
    from google.adk.agents import Agent  # type: ignore[import-not-found]

    return Agent(
        model=model,
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


def run(*, system: str, user: str, config: dict) -> str:
    try:
        agent = _make_agent(
            model=config["model"],
            instructions=system,
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
            top_p=float(config.get("top_p", 0.9)),
        )
        text = agent.run(user)
    except Exception as exc:
        return f"{ERROR_STAMP}{type(exc).__name__}: {exc}"

    if not text or not str(text).strip():
        return f"{ERROR_STAMP}模型未產生回應"
    return str(text)
```

> **Note for the engineer:** the `from google.adk.agents import Agent` import path is the expected shape for Google's ADK Python SDK. If the package's API differs (e.g., `Agent.create(...)` or named differently), adjust the body of `_make_agent` to match the actual SDK without changing the function signature; the rest of the codebase only sees `run(system=..., user=..., config=...)`.

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_llm_client.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/llm/client.py \
        packages/backend/tests/test_llm_client.py
git commit -m "feat(backend): add llm.client wrapper around ADK with error stamping"
```

---

### Task 16: Pipeline orchestrator

**Files:**
- Create: `packages/backend/src/tongue_backend/pipeline.py`
- Test: `packages/backend/tests/test_pipeline.py`
- Modify: `packages/backend/src/tongue_backend/__init__.py` (no-op if empty already)

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_pipeline.py`:

```python
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import pytest

from tongue_ai.types import ClassScore, HeadResult

from tongue_backend import pipeline as pl


def _jpeg_bytes(shape=(100, 100, 3)) -> bytes:
    img = (np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8))
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


@dataclass
class _StubHead:
    name: str
    head_type: str = "single"
    label: str = "L"

    def predict(self, _img):
        return HeadResult(
            task=self.name,
            head_type=self.head_type,
            predictions=[ClassScore(label=self.label, score=0.8)],
        )


@dataclass
class _StubRegistry:
    heads: list
    detector: object | None = None


def test_analyze_returns_full_response_shape(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色", label="淡紅"), _StubHead("舌態", label="無異常")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x", "temperature": 0.0, "max_tokens": 100, "top_p": 0.9})
    monkeypatch.setattr(pl.client, "run", lambda **kw: "## 醫師建議\n你還好")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)

    assert "user_message" in resp
    assert "heads" in resp and len(resp["heads"]) == 2
    assert resp["comment"].startswith("## 醫師建議")
    assert resp["disclaimer"].startswith("此為AI自動生成")
    assert "timing_ms" in resp


def test_analyze_raises_on_corrupt_image_bytes(monkeypatch):
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x"})
    with pytest.raises(pl.ImageDecodeError):
        pl.analyze(b"not a jpeg", registry=_StubRegistry(heads=[]))


def test_analyze_propagates_llm_error_into_comment(monkeypatch):
    registry = _StubRegistry(heads=[_StubHead("舌色")])
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model": "x", "temperature": 0.0, "max_tokens": 100, "top_p": 0.9})
    monkeypatch.setattr(pl.client, "run", lambda **kw: "⚠ 醫師建議產生失敗：upstream")

    resp = pl.analyze(_jpeg_bytes(), registry=registry)
    assert resp["comment"].startswith("⚠ 醫師建議產生失敗")
    assert len(resp["heads"]) == 1  # heads still populated
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_pipeline.py -v`

Expected: import error.

- [ ] **Step 3: Implement pipeline.py**

Create `packages/backend/src/tongue_backend/pipeline.py`:

```python
"""Single orchestrator for /api/analyze."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from tongue_ai import detect_tongue, run_all
from tongue_ai.types import HeadResult

from tongue_backend.llm import client, user_message
from tongue_backend.stores import llm_store, prompt_store


DISCLAIMER = "此為AI自動生成，不具醫療建議。若有疾病或疑問，應向專業中醫師諮詢。"


class ImageDecodeError(ValueError):
    """Raised when input bytes cannot be decoded as an image."""


# Indirections so tests can monkeypatch the FS reads without touching disk
def _load_prompt() -> str:
    return prompt_store.load_current()


def _load_llm_config() -> dict:
    return llm_store.load_current()


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("could not decode image")
    return img


def _serialise_heads(heads: list[HeadResult]) -> list[dict]:
    return [asdict(h) for h in heads]


def analyze(image_bytes: bytes, *, registry) -> dict[str, Any]:
    """Run the full analysis pipeline. ``registry`` is injected by the caller
    (the route reads ``request.app.state.registry``)."""
    timing: dict[str, int] = {}
    t0 = time.perf_counter()

    image = _decode_bgr(image_bytes)
    timing["decode"] = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    bbox = detect_tongue(image, getattr(registry, "detector", None))
    roi = image if bbox is None else image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    timing["detect"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    heads = run_all(roi, registry)
    timing["infer"] = int((time.perf_counter() - t2) * 1000)

    user_msg = user_message.build(heads)
    system = _load_prompt()
    llm_cfg = _load_llm_config()

    t3 = time.perf_counter()
    comment = client.run(system=system, user=user_msg, config=llm_cfg)
    timing["llm"] = int((time.perf_counter() - t3) * 1000)
    timing["total"] = int((time.perf_counter() - t0) * 1000)

    return {
        "user_message": user_msg,
        "heads": _serialise_heads(heads),
        "comment": comment,
        "disclaimer": DISCLAIMER,
        "timing_ms": timing,
    }
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_pipeline.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/pipeline.py \
        packages/backend/tests/test_pipeline.py
git commit -m "feat(backend): add pipeline.analyze orchestrator with timing snapshots"
```

---

### Task 17: `/api/analyze` route

**Files:**
- Modify: `packages/backend/src/tongue_backend/routes/analyze.py`
- Test: `packages/backend/tests/test_route_analyze.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_route_analyze.py`:

```python
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tongue_backend import pipeline as pl
from tongue_backend.app import create_app


def _jpeg_bytes(shape=(50, 50, 3)) -> bytes:
    img = np.zeros(shape, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _client(monkeypatch):
    monkeypatch.setattr(pl, "_load_prompt", lambda: "SYS")
    monkeypatch.setattr(pl, "_load_llm_config", lambda: {"model":"x","temperature":0.0,"max_tokens":100,"top_p":0.9})
    monkeypatch.setattr(pl.client, "run", lambda **_: "DOCTOR")
    app = create_app()
    # Inject a dummy registry into app.state so the route can find one
    app.state.registry = type("R", (), {"heads": [], "detector": None})()
    return TestClient(app)


def test_analyze_returns_200_with_expected_keys(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("t.jpg", _jpeg_bytes(), "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) >= {"user_message", "heads", "comment", "disclaimer", "timing_ms"}


def test_analyze_400_on_missing_file(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/analyze")
    assert r.status_code in (400, 422)  # FastAPI returns 422 for missing required form


def test_analyze_400_on_corrupt_bytes(monkeypatch):
    client = _client(monkeypatch)
    files = {"file": ("t.jpg", b"not an image", "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 400
    assert "could not decode" in r.json()["error"]


def test_analyze_413_on_oversize_file(monkeypatch):
    client = _client(monkeypatch)
    big = b"\x00" * (10 * 1024 * 1024 + 1)
    files = {"file": ("t.jpg", big, "image/jpeg")}
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 413
```

> **Note:** the test imports `tongue_backend.app.create_app` which is updated in Task 19. If running this test before Task 19 it will fail on import — that's fine; we'll wire it then. For now, run only the bits we have:

- [ ] **Step 2: Update routes/analyze.py**

Replace contents of `packages/backend/src/tongue_backend/routes/analyze.py`:

```python
"""POST /api/analyze — tongue diagnosis pipeline."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile

from tongue_backend.pipeline import ImageDecodeError, analyze


MAX_BYTES = 10 * 1024 * 1024  # 10 MB

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze")
async def api_analyze(file: UploadFile, request: Request) -> dict:
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail={"error": "missing file"})
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "image too large"})

    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail={"error": "registry unavailable; check /api/config/registry"})

    try:
        return analyze(data, registry=registry)
    except ImageDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
```

> Tests will run after Task 19 wires create_app. Skip running them now or expect import errors until then.

- [ ] **Step 3: Commit**

```bash
git add packages/backend/src/tongue_backend/routes/analyze.py \
        packages/backend/tests/test_route_analyze.py
git commit -m "feat(backend): wire POST /api/analyze to pipeline orchestrator"
```

---

### Task 18: `/api/config/{section}` routes + registry reload

**Files:**
- Create: `packages/backend/src/tongue_backend/routes/config.py`
- Test: `packages/backend/tests/test_route_config.py`

- [ ] **Step 1: Write failing test**

Create `packages/backend/tests/test_route_config.py`:

```python
import pytest
from fastapi.testclient import TestClient

from tongue_backend.app import create_app
from tongue_backend.stores import llm_store, prompt_store, registry_store


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    # Redirect all default/current paths into tmp_path
    pd = tmp_path / "system.default.md"; pc = tmp_path / "system.current.md"
    ld = tmp_path / "llm.default.yaml"; lc = tmp_path / "llm.current.yaml"
    rd = tmp_path / "registry.default.yaml"; rc = tmp_path / "registry.current.yaml"

    pd.write_text("DEFAULT PROMPT")
    ld.write_text("model: x\ntemperature: 0.2\nmax_tokens: 1024\ntop_p: 0.9\n")
    (tmp_path / "tongue_color.onnx").write_bytes(b"")
    rd.write_text(
        "heads:\n"
        "  - task: 舌色\n    head_type: single\n    onnx_path: tongue_color.onnx\n"
        "    input_size: [224, 224]\n    normalise:\n      mean: [0,0,0]\n      std: [1,1,1]\n"
        "    class_names: ['A','B']\n"
    )

    monkeypatch.setattr(prompt_store, "PROMPT_DEFAULT", pd)
    monkeypatch.setattr(prompt_store, "PROMPT_CURRENT", pc)
    monkeypatch.setattr(llm_store, "LLM_DEFAULT", ld)
    monkeypatch.setattr(llm_store, "LLM_CURRENT", lc)
    monkeypatch.setattr(registry_store, "REGISTRY_DEFAULT", rd)
    monkeypatch.setattr(registry_store, "REGISTRY_CURRENT", rc)

    return TestClient(create_app())


def test_get_prompt_returns_default_on_first_read(app_client):
    r = app_client.get("/api/config/prompt")
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "DEFAULT PROMPT"
    assert body["is_default"] is True


def test_put_prompt_persists(app_client):
    r = app_client.put("/api/config/prompt", json={"content": "MY EDIT"})
    assert r.status_code == 200
    r = app_client.get("/api/config/prompt")
    assert r.json()["content"] == "MY EDIT"
    assert r.json()["is_default"] is False


def test_reset_prompt_restores_default(app_client):
    app_client.put("/api/config/prompt", json={"content": "EDIT"})
    r = app_client.post("/api/config/prompt/reset")
    assert r.status_code == 200
    assert app_client.get("/api/config/prompt").json()["content"] == "DEFAULT PROMPT"


def test_put_llm_validates_temperature(app_client):
    bad = "model: x\ntemperature: 99\nmax_tokens: 1024\ntop_p: 0.9\n"
    r = app_client.put("/api/config/llm", json={"content": bad})
    assert r.status_code == 422
    assert "temperature" in r.json()["detail"]["error"]


def test_put_registry_validates_yaml(app_client):
    r = app_client.put("/api/config/registry", json={"content": "heads: : ::"})
    assert r.status_code == 422


def test_unknown_section_returns_404(app_client):
    r = app_client.get("/api/config/nope")
    assert r.status_code == 404


def test_registry_reload_swaps_app_state(app_client, monkeypatch):
    # Stub the registry loader to avoid loading real ONNX
    from tongue_ai import registry as ai_registry
    monkeypatch.setattr(ai_registry, "_make_session", lambda _p: object())
    r = app_client.post("/api/config/registry/reload")
    assert r.status_code == 200
    body = r.json()
    assert "loaded" in body and len(body["loaded"]) == 1
```

- [ ] **Step 2: Run, expect import or 404**

Run: `uv run --package tongue-backend pytest packages/backend/tests/test_route_config.py -v`

Expected: failures (route not implemented).

- [ ] **Step 3: Implement routes/config.py**

Create `packages/backend/src/tongue_backend/routes/config.py`:

```python
"""GET/PUT/RESET /api/config/{section} + POST /api/config/registry/reload."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from tongue_backend.stores import llm_store, prompt_store, registry_store


router = APIRouter(prefix="/api/config", tags=["config"])


class _PutBody(BaseModel):
    content: str


# Map {section} → (status_fn, save_fn, reset_fn)
_SECTIONS = {
    "prompt":   (prompt_store.status,   prompt_store.save,   prompt_store.reset),
    "llm":      (llm_store.status,      llm_store.save,      llm_store.reset),
    "registry": (registry_store.status, registry_store.save, registry_store.reset),
}


def _resolve(section: str):
    if section not in _SECTIONS:
        raise HTTPException(status_code=404, detail={"error": f"unknown section: {section}"})
    return _SECTIONS[section]


@router.get("/{section}")
def get_section(section: str):
    status_fn, _, _ = _resolve(section)
    return status_fn()


@router.put("/{section}")
def put_section(section: str, body: _PutBody):
    _, save_fn, _ = _resolve(section)
    try:
        save_fn(body.content)
    except (llm_store.ValidationError, registry_store.ValidationError) as e:
        # prompt_store.save accepts any text (no validation), so it never raises here
        raise HTTPException(status_code=422, detail={"error": str(e)})
    return {"ok": True}


@router.post("/{section}/reset")
def reset_section(section: str):
    _, _, reset_fn = _resolve(section)
    reset_fn()
    return {"ok": True}


@router.post("/registry/reload")
def reload_registry(request: Request):
    """Re-init ONNX sessions in-process. All-or-rollback on total failure."""
    from tongue_ai import load_registry, RegistryError
    from tongue_backend.stores.paths import REGISTRY_CURRENT

    previous = getattr(request.app.state, "registry", None)
    try:
        new_registry = load_registry(REGISTRY_CURRENT)
    except RegistryError as e:
        # Roll back: keep previous registry
        raise HTTPException(status_code=422, detail={"error": str(e)})

    request.app.state.registry = new_registry
    return {
        "loaded": [h.name for h in new_registry.heads],
        "failed": [],
        "previous_kept": previous is not None,
    }
```

> **Note on the exception tuple:** `prompt_store` does not raise `ValidationError` (free-form text), so the `except` only catches the LLM and registry variants. The `getattr` form keeps it grep-friendly without crashing on prompt PUTs.

- [ ] **Step 4: Run, expect pass (after wiring app in Task 19)**

Defer test run to Task 19's step 5.

- [ ] **Step 5: Commit**

```bash
git add packages/backend/src/tongue_backend/routes/config.py \
        packages/backend/tests/test_route_config.py
git commit -m "feat(backend): add /api/config CRUD + registry reload routes"
```

---

### Task 19: Wire `app.py` (factory + startup + state)

**Files:**
- Modify: `packages/backend/src/tongue_backend/app.py`
- Modify: `packages/backend/src/tongue_backend/routes/health.py`

- [ ] **Step 1: Update `app.py`**

Replace `packages/backend/src/tongue_backend/app.py`:

```python
"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from tongue_backend.routes.analyze import router as analyze_router
from tongue_backend.routes.config import router as config_router
from tongue_backend.routes.health import router as health_router
from tongue_backend.stores import llm_store, prompt_store, registry_store
from tongue_backend.stores.paths import REGISTRY_CURRENT


log = logging.getLogger("tongue_backend")


def create_app() -> FastAPI:
    app = FastAPI(title="Tongue Diagnosis API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(config_router)
    app.state.registry = None

    @app.on_event("startup")
    def _startup() -> None:
        # Ensure 'current' files exist (copied from default if missing)
        prompt_store.load_current()
        llm_store.load_current()
        registry_store.load_current_text()
        # Try to load registry; on failure, leave app.state.registry = None
        # so /api/analyze returns 503 until the user fixes config + reloads.
        try:
            from tongue_ai import load_registry
            app.state.registry = load_registry(REGISTRY_CURRENT)
            log.info("loaded registry: %s heads", len(app.state.registry.heads))
        except Exception as e:
            log.warning("registry not loaded at startup: %s", e)

    return app


# Module-level `app` for `uvicorn tongue_backend.app:app`
app = create_app()
```

> **No singleton, no global registry helper.** The route reads
> `request.app.state.registry` directly and passes it into `pipeline.analyze` as
> a parameter. Tests construct an app via `create_app()` and inject a stub
> registry into `app.state.registry`; the pipeline tests pass a stub registry
> directly to `pipeline.analyze`.

- [ ] **Step 2: Update health.py to surface registry status**

Replace `packages/backend/src/tongue_backend/routes/health.py`:

```python
"""GET /health — basic liveness + AI version + registry status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from tongue_ai import __version__ as ai_version


router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    reg = getattr(request.app.state, "registry", None)
    return {
        "status": "ok",
        "ai_version": ai_version,
        "registry_loaded": reg is not None,
        "heads_loaded": [h.name for h in reg.heads] if reg is not None else [],
    }
```

- [ ] **Step 3: Run all backend tests**

Run: `uv run --package tongue-backend pytest packages/backend/tests -v`

Expected: all green. `test_route_analyze.py` constructs an app via `create_app()` and injects a stub registry into `app.state.registry`, so the route reads the stub instead of needing real ONNX.

- [ ] **Step 4: Commit**

```bash
git add packages/backend/src/tongue_backend/app.py \
        packages/backend/src/tongue_backend/routes/health.py
git commit -m "feat(backend): wire app factory with startup hook + state.registry"
```

---

## Phase 3 — `tongue-frontend` (≈90 min)

### Task 20: Swap Streamlit for Gradio

**Files:**
- Modify: `packages/frontend/pyproject.toml`

- [ ] **Step 1: Replace deps**

Update `packages/frontend/pyproject.toml`:

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

Run: `uv sync --all-packages`

Expected: streamlit removed, gradio installed.

- [ ] **Step 3: Commit**

```bash
git add packages/frontend/pyproject.toml uv.lock
git commit -m "chore(frontend): switch from streamlit to gradio"
```

---

### Task 21: Backend HTTP client

**Files:**
- Create: `packages/frontend/src/tongue_frontend/api.py`
- Test: `packages/frontend/tests/__init__.py` (empty)
- Test: `packages/frontend/tests/test_api.py`
- Modify: `packages/frontend/pyproject.toml` (add respx as test dep)

- [ ] **Step 1: Add respx as a dev dependency**

In `packages/frontend/pyproject.toml`, add:

```toml
[dependency-groups]
dev = ["respx>=0.21", "pytest>=8"]
```

Run: `uv sync --all-packages --group dev`

- [ ] **Step 2: Write failing test**

Create `packages/frontend/tests/__init__.py` (empty), then `packages/frontend/tests/test_api.py`:

```python
import io

import httpx
import respx

from tongue_frontend import api


@respx.mock
def test_analyze_posts_multipart_and_returns_json():
    route = respx.post("http://localhost:8000/api/analyze").mock(
        return_value=httpx.Response(200, json={"comment": "OK"})
    )
    out = api.analyze(b"jpeg-bytes", "tongue.jpg")
    assert out == {"comment": "OK"}
    assert route.called


@respx.mock
def test_get_config_returns_status_dict():
    respx.get("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"content": "P", "is_default": True, "mtime": 0})
    )
    out = api.get_config("prompt")
    assert out["content"] == "P"


@respx.mock
def test_put_config_sends_content():
    route = respx.put("http://localhost:8000/api/config/prompt").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    api.put_config("prompt", "NEW")
    assert route.calls[0].request.read() == b'{"content":"NEW"}'


@respx.mock
def test_reset_config_posts():
    respx.post("http://localhost:8000/api/config/llm/reset").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    out = api.reset_config("llm")
    assert out == {"ok": True}


@respx.mock
def test_reload_registry_posts():
    respx.post("http://localhost:8000/api/config/registry/reload").mock(
        return_value=httpx.Response(200, json={"loaded": ["舌色"], "failed": []})
    )
    out = api.reload_registry()
    assert out["loaded"] == ["舌色"]
```

- [ ] **Step 3: Run, expect ImportError**

Run: `uv run --package tongue-frontend pytest packages/frontend/tests -v`

Expected: import error.

- [ ] **Step 4: Implement api.py**

Create `packages/frontend/src/tongue_frontend/api.py`:

```python
"""Thin httpx client to the FastAPI backend."""

from __future__ import annotations

import os

import httpx


BASE_URL = os.environ.get("TONGUE_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 60.0


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


def analyze(image_bytes: bytes, filename: str = "tongue.jpg") -> dict:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    with _client() as c:
        r = c.post("/api/analyze", files=files)
        r.raise_for_status()
        return r.json()


def get_config(section: str) -> dict:
    with _client() as c:
        r = c.get(f"/api/config/{section}")
        r.raise_for_status()
        return r.json()


def put_config(section: str, content: str) -> dict:
    with _client() as c:
        r = c.put(f"/api/config/{section}", json={"content": content})
        if r.status_code == 422:
            return {"error": r.json().get("detail", {}).get("error", "validation failed")}
        r.raise_for_status()
        return r.json()


def reset_config(section: str) -> dict:
    with _client() as c:
        r = c.post(f"/api/config/{section}/reset")
        r.raise_for_status()
        return r.json()


def reload_registry() -> dict:
    with _client() as c:
        r = c.post("/api/config/registry/reload")
        if r.status_code == 422:
            return {"error": r.json().get("detail", {}).get("error", "reload failed")}
        r.raise_for_status()
        return r.json()


def health() -> dict:
    with _client() as c:
        r = c.get("/health")
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 5: Run, expect pass**

Run: `uv run --package tongue-frontend pytest packages/frontend/tests -v`

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/frontend/src/tongue_frontend/api.py \
        packages/frontend/tests/__init__.py \
        packages/frontend/tests/test_api.py \
        packages/frontend/pyproject.toml uv.lock
git commit -m "feat(frontend): add httpx api client (analyze + config CRUD + reload)"
```

---

### Task 22: Analyze view (Tab 1)

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/__init__.py` (empty)
- Create: `packages/frontend/src/tongue_frontend/views/analyze.py`

- [ ] **Step 1: Create the view module**

Create `packages/frontend/src/tongue_frontend/views/__init__.py` (empty).

Create `packages/frontend/src/tongue_frontend/views/analyze.py`:

```python
"""Tab 1: 舌診分析 — capture/upload + result panel."""

from __future__ import annotations

import io

import gradio as gr
import httpx
import numpy as np
from PIL import Image

from tongue_frontend import api


def _heads_to_rows(heads: list[dict]) -> list[list[str]]:
    rows = []
    for h in heads:
        if h.get("error"):
            rows.append([h["task"], f"⚠ {h['error']}"])
            continue
        preds = "、".join(f"{p['label']} ({p['score']:.2f})" for p in h.get("predictions", []))
        rows.append([h["task"], preds or "(無)"])
    return rows


def _to_jpeg_bytes(image: np.ndarray) -> bytes:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _on_analyze(image: np.ndarray | None):
    if image is None:
        return [], "請選擇或拍攝照片", "", "", ""
    try:
        result = api.analyze(_to_jpeg_bytes(image))
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("detail", {}).get("error", str(e))
        except Exception:
            err = str(e)
        return [], "", f"⚠ 分析失敗：{err}", "", ""
    except httpx.ConnectError:
        return [], "", "⚠ 無法連線到後端 — 請啟動 backend (port 8000)", "", ""
    rows = _heads_to_rows(result["heads"])
    timing = result["timing_ms"]
    timing_str = " · ".join(f"{k}: {v}ms" for k, v in timing.items())
    return rows, result["comment"], result["disclaimer"], result["user_message"], timing_str


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 舌診影像")
                img = gr.Image(sources=["webcam", "upload"], type="numpy",
                               label="拍攝或上傳", height=320)
                go = gr.Button("分析", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("### 判讀結果")
                heads_table = gr.Dataframe(
                    headers=["項目", "預測"],
                    interactive=False,
                    label="各項判讀",
                )
                comment_md = gr.Markdown(label="醫師建議")
                disclaimer_md = gr.Markdown()
                with gr.Accordion("進階 (debug)", open=False):
                    user_msg_box = gr.Textbox(label="送至 Gemini 的 user message", lines=10)
                    timing_box = gr.Textbox(label="耗時 (ms)")

        go.click(
            fn=_on_analyze,
            inputs=[img],
            outputs=[heads_table, comment_md, disclaimer_md, user_msg_box, timing_box],
        )
    return view
```

- [ ] **Step 2: Commit (manual smoke deferred to Task 26)**

```bash
git add packages/frontend/src/tongue_frontend/views/__init__.py \
        packages/frontend/src/tongue_frontend/views/analyze.py
git commit -m "feat(frontend): add analyze view with capture, table, and debug panel"
```

---

### Task 23: Prompt editor view (Tab 2)

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/prompt_editor.py`

- [ ] **Step 1: Create the view**

Create `packages/frontend/src/tongue_frontend/views/prompt_editor.py`:

```python
"""Tab 2: 提示詞設定 — system prompt textarea."""

from __future__ import annotations

import gradio as gr

from tongue_frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("prompt")
    flag = "default" if s.get("is_default") else "custom"
    return s["content"], f"狀態：{flag}"


def _save(content: str) -> str:
    out = api.put_config("prompt", content)
    if "error" in out:
        return f"⚠ 儲存失敗：{out['error']}"
    return "已儲存 — 下次分析將使用新提示詞"


def _reset() -> tuple[str, str]:
    api.reset_config("prompt")
    content, status = _load()
    return content, "已還原預設值"


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### 系統提示詞 (大眾版規則)")
        gr.Markdown("編輯後按「儲存」即可生效；下次「分析」會立刻採用。")
        textbox = gr.Textbox(
            lines=30, label="system prompt", show_copy_button=True
        )
        with gr.Row():
            save_btn = gr.Button("儲存", variant="primary")
            reset_btn = gr.Button("還原預設")
            reload_btn = gr.Button("從磁碟重新載入")
        status_box = gr.Markdown()

        view.load(fn=_load, outputs=[textbox, status_box])
        save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
        reset_btn.click(fn=_reset, outputs=[textbox, status_box])
        reload_btn.click(fn=_load, outputs=[textbox, status_box])
    return view
```

- [ ] **Step 2: Commit**

```bash
git add packages/frontend/src/tongue_frontend/views/prompt_editor.py
git commit -m "feat(frontend): add prompt editor view (Tab 2)"
```

---

### Task 24: LLM editor view (Tab 3)

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/llm_editor.py`

- [ ] **Step 1: Create the view**

Create `packages/frontend/src/tongue_frontend/views/llm_editor.py`:

```python
"""Tab 3: LLM 設定 — model, temperature, max_tokens, top_p."""

from __future__ import annotations

import gradio as gr

from tongue_frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("llm")
    flag = "default" if s.get("is_default") else "custom"
    return s["content"], f"狀態：{flag}"


def _save(content: str) -> str:
    out = api.put_config("llm", content)
    if "error" in out:
        return f"⚠ 儲存失敗：{out['error']}"
    return "已儲存 — 下次分析將使用新設定"


def _reset() -> tuple[str, str]:
    api.reset_config("llm")
    content, _ = _load()
    return content, "已還原預設值"


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### LLM 設定 (Gemini via ADK)")
        gr.Markdown("以 YAML 編輯：`model`, `temperature` ∈ [0,2], `max_tokens` > 0, `top_p` ∈ (0,1].")
        textbox = gr.Code(language="yaml", label="llm.yaml", lines=10)
        with gr.Row():
            save_btn = gr.Button("儲存", variant="primary")
            reset_btn = gr.Button("還原預設")
            reload_btn = gr.Button("從磁碟重新載入")
        status_box = gr.Markdown()

        view.load(fn=_load, outputs=[textbox, status_box])
        save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
        reset_btn.click(fn=_reset, outputs=[textbox, status_box])
        reload_btn.click(fn=_load, outputs=[textbox, status_box])
    return view
```

- [ ] **Step 2: Commit**

```bash
git add packages/frontend/src/tongue_frontend/views/llm_editor.py
git commit -m "feat(frontend): add LLM editor view (Tab 3)"
```

---

### Task 25: Registry editor view (Tab 4)

**Files:**
- Create: `packages/frontend/src/tongue_frontend/views/registry_editor.py`

- [ ] **Step 1: Create the view**

Create `packages/frontend/src/tongue_frontend/views/registry_editor.py`:

```python
"""Tab 4: 模型設定 — registry YAML + reload."""

from __future__ import annotations

import gradio as gr

from tongue_frontend import api


def _load() -> tuple[str, str]:
    s = api.get_config("registry")
    flag = "default" if s.get("is_default") else "custom"
    h = api.health()
    heads = "、".join(h.get("heads_loaded", [])) or "(none)"
    return s["content"], f"狀態：{flag}　已載入 {len(h.get('heads_loaded', []))} 個 heads：{heads}"


def _save(content: str) -> str:
    out = api.put_config("registry", content)
    if "error" in out:
        return f"⚠ 儲存失敗：{out['error']}"
    return "已儲存 — 點「Apply & Reload Models」生效"


def _reset() -> tuple[str, str]:
    api.reset_config("registry")
    content, status = _load()
    return content, "已還原預設值"


def _reload() -> str:
    out = api.reload_registry()
    if "error" in out:
        return f"⚠ 重新載入失敗：{out['error']}"
    loaded = out.get("loaded", [])
    failed = out.get("failed", [])
    msg = f"已載入 {len(loaded)} heads"
    if failed:
        msg += f"，失敗 {len(failed)}：{failed}"
    return msg


def build() -> gr.Blocks:
    with gr.Blocks() as view:
        gr.Markdown("### 模型設定 (registry YAML)")
        gr.Markdown(
            "編輯各 task head 的 ONNX 路徑、輸入大小、normalisation、class_names、threshold。"
            " 「儲存」只會持久化 YAML；要套用必須點「Apply & Reload Models」。"
        )
        textbox = gr.Code(language="yaml", label="registry.yaml", lines=30)
        with gr.Row():
            save_btn = gr.Button("儲存")
            reset_btn = gr.Button("還原預設")
            reload_disk_btn = gr.Button("從磁碟重新載入")
            apply_btn = gr.Button("Apply & Reload Models", variant="primary")
        status_box = gr.Markdown()
        reload_status = gr.Markdown()

        view.load(fn=_load, outputs=[textbox, status_box])
        save_btn.click(fn=_save, inputs=[textbox], outputs=[status_box])
        reset_btn.click(fn=_reset, outputs=[textbox, status_box])
        reload_disk_btn.click(fn=_load, outputs=[textbox, status_box])
        apply_btn.click(fn=_reload, outputs=[reload_status])
    return view
```

- [ ] **Step 2: Commit**

```bash
git add packages/frontend/src/tongue_frontend/views/registry_editor.py
git commit -m "feat(frontend): add registry editor view (Tab 4) with reload button"
```

---

### Task 26: Compose tabs in `app.py`

**Files:**
- Create/Replace: `packages/frontend/src/tongue_frontend/app.py`

- [ ] **Step 1: Compose**

Replace `packages/frontend/src/tongue_frontend/app.py`:

```python
"""Gradio app — four tabs: 舌診分析 / 提示詞設定 / LLM 設定 / 模型設定."""

from __future__ import annotations

import gradio as gr

from tongue_frontend.views import analyze, prompt_editor, llm_editor, registry_editor


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Tongue Diagnosis POC") as app:
        gr.Markdown("# 舌診 POC")
        with gr.Tabs():
            with gr.Tab("舌診分析"):
                analyze.build()
            with gr.Tab("提示詞設定"):
                prompt_editor.build()
            with gr.Tab("LLM 設定"):
                llm_editor.build()
            with gr.Tab("模型設定"):
                registry_editor.build()
    return app


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Commit**

```bash
git add packages/frontend/src/tongue_frontend/app.py
git commit -m "feat(frontend): compose four-tab gradio app"
```

---

## Phase 4 — Documentation & smoke (≈30 min)

### Task 27: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit README**

Update `README.md` — replace the existing "Architecture", "Quick Start", and "API Endpoints" sections to reflect Gradio frontend, the new `/api/config` routes, and the four-tab UI. Use this content (preserve the GPU + Label Studio sections at the bottom):

```markdown
# Tongue Diagnosis

Realtime tongue image analysis using object detection, recognition, and classification, with a TCM doctor comment generated by Gemini.

## Architecture

UV workspace monorepo with three packages:

```
packages/
├── ai/        → tongue-ai       (PyTorch, ONNX Runtime, OpenCV, NumPy, label-studio-sdk)
├── backend/   → tongue-backend  (FastAPI, Google ADK / Gemini, pyyaml)
└── frontend/  → tongue-frontend (Gradio, httpx)
```

Frontend is HTTP-only — no Python imports across frontend / backend.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Astral package manager)
- Python 3.11+
- A Google Cloud project with Vertex AI enabled (or Gemini API access) — auth via `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`.

## Quick Start

```bash
uv sync --all-packages

# Terminal 1: backend
uv run uvicorn tongue_backend.app:app --reload --port 8000

# Terminal 2: frontend
uv run python -m tongue_frontend.app
# open http://localhost:7860
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness, AI version, registry status |
| POST | `/api/analyze` | Full pipeline (multipart `file=`) → heads + Gemini comment |
| GET / PUT / POST | `/api/config/{section}` | CRUD for `prompt`, `llm`, `registry` |
| POST | `/api/config/{section}/reset` | Restore default |
| POST | `/api/config/registry/reload` | Re-init ONNX sessions |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for Gradio frontend and config endpoints"
```

---

### Task 28: Add smoke test doc

**Files:**
- Create: `docs/smoke-test.md`

- [ ] **Step 1: Write smoke script**

Create `docs/smoke-test.md`:

```markdown
# Smoke test

Manual end-to-end verification. Run before every merge to `main`.

## Prereqs

- ONNX models present at the paths in `packages/backend/config/registry.default.yaml`
  (or edit registry to point at whatever models you have, then Reload).
- ADK auth configured (e.g., `gcloud auth application-default login`).

## Steps

1. `uv sync --all-packages`
2. Terminal A: `uv run uvicorn tongue_backend.app:app --port 8000`
3. Terminal B: `uv run python -m tongue_frontend.app`
4. Open http://localhost:7860 — confirm 4 tabs render.
5. **Tab 1 — 舌診分析:** upload `tongue.jpg`, click 分析. Expect:
   - Heads dataframe populated (one row per active head)
   - Markdown comment with sections 主要中醫體質 / 次要中醫體質 / 體質說明 / 證素列表 / 警語
   - Disclaimer visible
   - Advanced panel shows `user_message` and `timing_ms`
6. **Tab 2 — 提示詞設定:** trim the prompt to a one-liner (`回應一句中文`) → 儲存 → re-run analyze → comment now reflects the edit. Click 還原預設 → confirm full prompt returns.
7. **Tab 3 — LLM 設定:** lower `temperature` to 0.0 → 儲存 → re-run; expect roughly identical output across two runs.
8. **Tab 4 — 模型設定:** change one head's `threshold` from 0.5 to 0.9 → 儲存 → Apply & Reload Models → re-run; that head should return fewer multi-labels.
9. **Failure path:** stop the backend mid-flow → frontend shows `無法連線到後端`.
10. **Empty input:** click 分析 with no image → frontend shows `請選擇或拍攝照片`.

## Recovery

If you've broken something via the UI:
- Tab 2/3/4: click 還原預設.
- Or remove `*.current.*` files under `packages/backend/{prompts,config}/` and restart the backend.
```

- [ ] **Step 2: Commit**

```bash
git add docs/smoke-test.md
git commit -m "docs: add manual smoke test script"
```

---

### Task 29: Final lint pass and PR-ready commit

**Files:**
- (none — verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest packages -v`

Expected: all tests across all three packages pass (≈30 tests).

- [ ] **Step 2: Verify import surface**

Run:

```bash
uv run --package tongue-backend python -c "from tongue_backend.app import create_app; print(create_app().routes)"
```

Expected: prints route list including `/health`, `/api/analyze`, `/api/config/{section}`, `/api/config/{section}/reset`, `/api/config/registry/reload`.

- [ ] **Step 3: Verify frontend imports**

Run:

```bash
uv run --package tongue-frontend python -c "from tongue_frontend.app import build_app; build_app(); print('OK')"
```

Expected: prints `OK` (does NOT launch — just constructs the Blocks).

- [ ] **Step 4: Push branch**

```bash
git push -u origin feat/tongue-poc
```

- [ ] **Step 5: Open PR (optional — only if user asks)**

Defer to user instruction. Do not auto-open.

---

## Phase 5 — Real-model smoke (deferred)

After the POC merges, the user will:

1. Drop real `.onnx` files into `packages/ai/models/`.
2. Edit `packages/backend/config/registry.default.yaml` with the right `input_size` / `normalise` / `class_names` per task.
3. Configure ADK auth.
4. Re-run the smoke script in `docs/smoke-test.md`.

This phase is intentionally **out of plan scope** — it requires artefacts and credentials this plan can't produce. The plan delivers the harness; real models are dropped in afterwards.

---

## Spec coverage check (self-review)

| Spec section | Implemented in | Notes |
|---|---|---|
| 1. Goal | Phase 0–4 | end-to-end flow |
| 2. Non-goals | — | adhered to (no auth, no SSE, no audit log) |
| 3. Architecture (3 boundaries) | Phase 1, 2, 3 | preserved |
| 4. Package layout | Phase 1, 2, 3 | added `stores/` subpackage (called out in plan header) |
| 5. Core abstractions (`TaskHead`, `HeadResult`, `pipeline.analyze`) | Tasks 2, 5, 16 | |
| 6. Config plane (3 sections, API, validation, reload) | Tasks 9–13, 18–19 | |
| 7. Data flow (request shape, response shape, two-message prompt) | Tasks 14–17, 22 | |
| 8. Error handling (input / inference / LLM / config / lifecycle) | Tasks 7, 15–17, 19 | |
| 9. Dependency changes | Tasks 1, 8, 20 | |
| 10. Testing (per-package contract tests + smoke) | Tasks 2–7, 10–18, 21, 28 | golden tests for `user_message` |
| 11. Open assumptions to verify | Phase 5 | deferred to real-model phase |
| 12. Out of scope | — | not implemented (correctly) |
