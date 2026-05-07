# Merge Amanda's ResNet50 Into Tongue-Diagnosis — Design

**Date:** 2026-05-07
**Status:** Design approved by user; spec committed; pending user review and implementation plan.
**Supersedes:** Sections 3 (Architecture), 4 (Package layout), 5 (Core abstractions), 7 (Data flow), 9 (Dependency changes), and parts of 10 (Testing) of [`2026-05-06-tongue-poc-design.md`](./2026-05-06-tongue-poc-design.md). Other sections of the prior spec (config plane, error handling shape, frontend tab structure, smoke script) are preserved unchanged.
**Stack:** Gradio · FastAPI · **PyTorch + torchvision** (was: ONNX Runtime) · Google ADK → Gemini · **Hugging Face Hub** for weight delivery.

---

## 1. Goal

Integrate the trained ResNet50 weights and training scripts delivered in `ResNet50測試模型權重檔V1_Amanda/` into the existing tongue-diagnosis monorepo, in a way that gives a working end-to-end POC: webcam/upload → real model inference → locked-prompt LLM doctor comment.

This document captures the **deltas** from the 2026-05-06 POC design caused by adopting the existing trained weights. Where it does not supersede the prior spec, the prior spec stands.

## 2. What changes vs the 2026-05-06 spec

| Area | 2026-05-06 spec | This merge spec |
|---|---|---|
| Inference runtime | ONNX Runtime sessions | PyTorch + torchvision (`resnet50` + `load_state_dict`) |
| Head structure | 11 per-task heads matching v4 schema | 2 composite heads (`front`, `sublingual`) as Amanda trained them |
| Weight delivery | ONNX files committed under `packages/ai/models/` | `.pth` files pulled from a Hugging Face Hub private repo at runtime |
| `tongue_ai` deps | add `onnxruntime`, `pyyaml` | add `huggingface_hub`, `pyyaml`; keep `torch`/`torchvision` |
| User-message construction | One bullet per v4 task head | Composite head predictions split back into v4 categories via static `category_map` |
| Detector | Optional ONNX detector | Disabled (Amanda's classifiers were trained on whole tongue photos) |

Unchanged from 2026-05-06 spec: Gradio frontend with four tabs · `/api/analyze` JSON shape · config plane (prompt / llm / registry CRUD + reload) · locked 大眾版 prompt with 9-證素 lookup · mandatory disclaimer · graceful-degradation per-head error handling · ADK as the Gemini client · always-call LLM (no Amanda-style confidence-gated trigger).

## 3. File disposition for the Amanda handoff

| Source file in `ResNet50測試模型權重檔V1_Amanda/` | Disposition |
|---|---|
| `best_resnet50.pth` (generic) | **Discard.** Superseded by the front/sublingual split. |
| `best_resnet50_front.pth` | **Adopt.** Uploaded to HF Hub, referenced by the `front` head's `weights_uri`. |
| `best_resnet50_sublingual.pth` | **Adopt.** Same flow for the `sublingual` head. |
| `inference.py` | **Reference only.** Its `TongueClassifier` shape inspires `tongue_ai.task_head.PyTorchTaskHead`, rewritten for BGR/cv2 input, no `print()` side-effects, no auto-`mps` on import, no top-3 hardcoding. |
| `analyze.py` | **Discard.** Hardcoded `/Users/chenyu/...` path, free-form prompt, conditional Gemini, sys.path hack — all violate the design contract. |
| `train.py`, `train_front.py`, `train_sublingual.py` | **Adopt with cleanup.** Move to `packages/ai/training/`; argparse-driven paths; matplotlib font fallback chain; `TongueDataset` extracted to a shared module; Kappa + classification-report eval preserved. |
| `*.pdf`, `.DS_Store`, `*:Zone.Identifier` | **Discard.** Not committed to git. |
| The folder itself | **Delete after merge.** CJK filename fights git on Windows; folder is a temporary handoff. |

**Working-tree noise to clean in the same PR:** `packages/frontend/src/tongue_frontend/test.md` (unrelated Upwork research) — delete. The stray `cv2` import removal in `detection.py` is moot once `detection.py` is rewritten as part of the AI package overhaul.

## 4. Architecture

```
┌──────────────────────┐  HTTP multipart   ┌──────────────────────────────────────┐
│  Gradio frontend     │ ─────────────────►│  FastAPI backend                     │
│  (tongue-frontend)   │   POST /api/      │  (tongue-backend)                    │
│                      │     analyze       │                                       │
│  • webcam capture    │                   │   1. Decode BGR                       │
│  • file upload       │ ◄─────────────────│   2. PyTorchTaskHead.predict() × N    │
│  • result panel      │   JSON {heads,    │   3. Split composite heads → v4 bullets│
│  • prompt editor     │   comment, ...}   │   4. Read prompt + LLM config         │
│  • registry editor   │                   │   5. ADK → Gemini → comment           │
└──────────────────────┘                   └──────────────────────────────────────┘
                                                        │             │
                                                        ▼             ▼
                                         ┌──────────────────┐ ┌────────────────┐
                                         │ tongue-ai        │ │ Google ADK SDK │
                                         │ PyTorch+torchvis │ │ → Gemini       │
                                         │ HF Hub weights   │ │                │
                                         └──────────────────┘ └────────────────┘
```

Three-package boundary as before:

- `tongue-frontend` (Gradio) — capture + display + config editing only.
- `tongue-backend` (FastAPI) — orchestration, config persistence, ADK call. Owns the composite-head → v4-category splitter.
- `tongue-ai` — pure inference library. Holds `PyTorchTaskHead`, `WeightSource`, `Registry`. No web, no LLM. Importable for tests and future training/eval.

## 5. Package layout (changes only)

```
packages/ai/
├── pyproject.toml              ← drop onnxruntime; add huggingface_hub; gain [training] extra
├── models/                     ← runtime cache only; gitignored except .gitkeep
└── src/tongue_ai/
    ├── types.py                ← unchanged: HeadResult, ClassScore, Normalisation, BBox
    ├── preprocessing.py        ← unchanged: BGR/cv2-based, ImageNet normalise
    ├── detection.py            ← unchanged stub (whole-image bbox)
    ├── weights.py              ← NEW: WeightSource — resolves hf:// or local: URIs
    ├── task_head.py            ← REVISED: PyTorchTaskHead (replaces ONNX flavor)
    ├── registry.py             ← REVISED: YAML schema gains weights_uri, arch, category_map
    ├── inference.py            ← unchanged contract: run_all(image, registry)
    └── training/               ← NEW subpackage (Amanda's scripts, cleaned)
        ├── __init__.py
        ├── dataset.py          ← TongueDataset extracted from train_*.py
        ├── train_front.py      ← argparse, no hardcoded paths
        ├── train_sublingual.py ← same treatment
        └── evaluate.py         ← classification_report + Cohen's Kappa, callable

packages/backend/
└── src/tongue_backend/
    ├── llm/
    │   └── user_message.py     ← REVISED: takes category_map, groups composite predictions
    └── …                       ← all other modules per 2026-05-06 spec
```

## 6. Core abstractions (PyTorch flavor)

### `tongue_ai.task_head.PyTorchTaskHead`

```python
class PyTorchTaskHead:
    name: str                              # "front" or "sublingual"
    head_type: Literal["single", "multi"]  # both composite heads = "single"
    arch: Literal["resnet50"]              # extension point for future archs
    class_names: list[str]                 # 14 (front) or 3 (sublingual), v4-style strings
    input_size: tuple[int, int]            # (224, 224)
    normalise: Normalisation               # ImageNet mean/std
    threshold: float = 0.5                 # unused for single-label; kept for symmetry
    device: torch.device                   # cuda > mps > cpu (autodetected)
    model: torch.nn.Module                 # eager-loaded resnet50 in eval mode

    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        # BGR → RGB → resize → ToTensor → normalise → forward → softmax → top-1 (single)
        # multi-label (future): predictions = [c for c in classes if score >= threshold]
```

A factory `load_pytorch_head(cfg, weight_path) -> PyTorchTaskHead` builds `torchvision.models.resnet50(weights=None)`, replaces `model.fc = nn.Linear(2048, len(class_names))`, calls `load_state_dict(torch.load(weight_path, map_location=device))`, sets `eval()`. This mirrors Amanda's `TongueClassifier.__init__` faithfully.

### `tongue_ai.weights.WeightSource`

A small URI resolver. Two schemes:

| URI | Resolution |
|---|---|
| `local:packages/ai/models/best_resnet50_front.pth` | resolved relative to the registry YAML's directory; just returns the path |
| `hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth` | calls `huggingface_hub.hf_hub_download(repo_id="CallMeDaniel/tongue-resnet50-v1", filename="best_resnet50_front.pth", token=os.environ.get("HF_TOKEN"))` and returns the cached path |

Cache locations:
- HF SDK manages its own cache under `~/.cache/huggingface/hub` (default).
- A second symlink/copy under `packages/ai/models/` is *not* introduced — single source of truth, less to go wrong.
- Override location via `HF_HOME` if needed (standard HF env var).

Failures raise a typed `WeightFetchError(repo, filename, cause)`. The registry-load step catches and surfaces these in the `/api/config/registry/reload` failure list, with the same partial-fail / rollback semantics as the prior spec §8.

### Registry YAML

```yaml
detector:
  enabled: false                            # Amanda's models accept whole tongue photos

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
    無異常: 舌質           # default-normal per v4 schema
    瘀血絲: 舌下絡脈
  sublingual:
    怒張: 舌下絡脈
    曲張: 舌下絡脈
    囊柱囊泡: 舌下絡脈
```

Class-name order matches Amanda's training scripts verbatim (the index→label decoding depends on it; the `fc` layer's dimensions are validated at load time: `len(class_names) == fc.out_features`, else load fails with a "expected N classes, got M" error).

## 7. Composite-head → v4 user-message splitter

Amanda's `front` model emits one of 14 classes; the locked Gemini prompt expects bullets per v4 category. The `category_map` (above) is the static lookup that bridges them.

`llm/user_message.build(heads, category_map)`:

1. For each `HeadResult` (skipping any with `error`), look up each prediction's `label` in `category_map[head_name]` to find its v4 category.
2. Group predictions by v4 category, merging across heads (e.g., if `front` outputs `瘀血絲` and `sublingual` outputs `怒張`, both end up under `舌下絡脈`).
3. Emit one bullet per v4 category that has ≥1 prediction. Categories with no signal are simply absent from the user message — the locked prompt already handles partial input.
4. Each bullet: `- {category}：{label1}（{score:.2f}）、{label2}（…）` (multi labels joined with `、`).

Worked example (single front + single sublingual):

```
本次舌診判讀結果：

- 舌色：淡紅（0.78）
- 舌下絡脈：怒張（0.72）

請依規則輸出大眾版報告。
```

Worked example (front predicts `瘀血絲` + sublingual predicts `怒張` — both map to 舌下絡脈):

```
本次舌診判讀結果：

- 舌下絡脈：瘀血絲（0.51）、怒張（0.72）

請依規則輸出大眾版報告。
```

The locked system prompt continues to apply its 9-證素 lookup verbatim; the splitter never touches the prompt, only the per-request user message.

## 8. Pipeline (signature unchanged, internals updated)

`tongue_backend.pipeline.analyze` keeps the same shape as the 2026-05-06 spec §5:

```python
def analyze(image_bytes: bytes) -> AnalyzeResponse:
    image     = decode_bgr(image_bytes)
    roi       = detect_tongue(image, registry.detector) or image       # detector disabled → image
    heads     = inference.run_all(roi, registry)                        # PyTorchTaskHead loop
    user_msg  = user_message.build(heads, registry.category_map)        # composite → v4 split
    system    = prompt_store.load_current()
    llm_cfg   = config_store.load_llm()
    comment   = llm.client.run(system, user_msg, llm_cfg)               # ADK → Gemini
    return AnalyzeResponse(
        user_message=user_msg,
        heads=heads,
        comment=comment,
        disclaimer=DISCLAIMER,
        category_map=registry.category_map,                              # echoed for frontend
        timing_ms=timing.snapshot(),
    )
```

`category_map` is echoed in the response so the frontend can render the v4-category breakdown alongside the raw head outputs without re-implementing the split.

## 9. Dependency changes

| Package | Add | Remove |
|---|---|---|
| `tongue-ai` | `huggingface_hub>=0.20`, `pyyaml>=6` | (no `onnxruntime` was ever landed; just don't add it) |
| `tongue-ai` `[training]` extra | `scikit-learn>=1.4`, `matplotlib>=3.8` | — |
| `tongue-backend` | `google-adk` (Vertex), `pyyaml>=6` | — |
| `tongue-frontend` | `gradio>=4` | `streamlit` |

Training-only deps live under `[project.optional-dependencies] training` so the runtime backend image stays slim. Install via `uv sync --all-packages --extra training` when training.

## 10. Hugging Face Hub setup

- **Repo:** `CallMeDaniel/tongue-resnet50-v1` (private). User to create before merging. Spec is agnostic to the repo id; it lives only in the registry YAML.
- **Auth:** `HF_TOKEN` env var (read by `huggingface_hub` automatically). `.env.example` documents this.
- **Files uploaded:** `best_resnet50_front.pth`, `best_resnet50_sublingual.pth`. The generic `best_resnet50.pth` is dropped per §3.
- **Backend startup:** registry-load eagerly resolves all `hf:` URIs. Failures land in the per-head failure list returned by `/api/config/registry/reload`.
- **Local dev override:** setting `weights_uri: local:relative/path` skips HF and reads from disk — useful when iterating on weights before re-uploading.

## 11. Training subpackage

- **Location:** `packages/ai/training/`.
- **Install:** `uv sync --extra training`.
- **Entry points:**
  - `python -m tongue_ai.training.train_front --labels-json … --img-dir … --weights-out …`
  - `python -m tongue_ai.training.train_sublingual --labels-json … --img-dir … --weights-out …`
- **Cleanups vs Amanda's originals:**
  - Argparse-driven paths (no hardcoded `packages/ai/data/labels.json`).
  - `matplotlib` font fallback chain (no `Heiti TC` Mac-only).
  - `dataset.py` extracted as a shared `TongueDataset` so both training entry points import it.
  - `evaluate.py` exposes a callable `evaluate(model, loader, target_labels)` returning `(report_str, kappa)` for use both at end-of-training and in standalone evaluation.

## 12. Error handling deltas (relative to 2026-05-06 spec §8)

| Case | Behaviour |
|---|---|
| `WeightFetchError` (HF download fails) | registry-load partial-fail; head not loaded; rest proceeds; if zero heads load, registry rolls back to previous good state per prior spec |
| `RuntimeError` from `load_state_dict` (size mismatch) | registry-load failure with explicit "expected N classes, got M" message |
| Per-head `predict()` exception | unchanged: head's entry becomes `{..., "error": "..."}`; user message excludes it; rest proceeds |
| HF API rate-limit / 5xx | bubble up as `WeightFetchError`; no automatic retry inside the registry-load (caller can retry the reload endpoint) |
| `category_map` references a class not in any head | registry validation 422 on `PUT` (warning, not block — orphan map entries are harmless but noisy) |
| Head class missing from `category_map` | registry validation 422 on `PUT` (block — would silently drop predictions at request time) |

## 13. Testing deltas (relative to 2026-05-06 spec §10)

The spec's test plan stands. Additions:

| Module | Tests |
|---|---|
| `task_head` | use a fake state_dict on a 4-class fc layer to assert decoding without needing real weights; CPU-only path covered; mps/cuda paths skipped in CI |
| `weights` | `local:` URI returns the resolved path; `hf:` URI mocked at the `huggingface_hub.hf_hub_download` boundary; `WeightFetchError` raised on simulated network failure |
| `registry` | `category_map` validation: every class in every head must appear in `category_map[head_name]`; missing entry → 422 |
| `user_message.build` | golden zh-TW snapshots for: front-only single prediction; sublingual-only; both heads; cross-head merge into 舌下絡脈; head with `error` skipped |
| `pipeline.analyze` (integration) | end-to-end with stubbed registry (no real `.pth`) + stubbed LLM client; asserts `category_map` is echoed in response |

## 14. PR commit sequence

The merge lands as one PR on branch `feat/merge-amanda-resnet50`, structured as a sequence of focused commits:

1. **chore: clean handoff folder + working-tree noise** — delete `ResNet50測試模型權重檔V1_Amanda/`, `frontend/.../test.md`; revert the stray `cv2` import edit in `detection.py`.
2. **docs: add merge spec + new implementation plan** — this document plus the plan derived from it via `superpowers:writing-plans`.
3. **chore(ai): drop onnxruntime, add huggingface_hub, add [training] extra** — `pyproject.toml` only.
4. **feat(ai): WeightSource (hf:// + local:) + types/preprocessing** — small, testable.
5. **feat(ai): PyTorchTaskHead + registry YAML loader + run_all** — replaces the stub modules.
6. **feat(ai): training subpackage** — Amanda's scripts cleaned + extracted dataset/evaluate.
7. **feat(backend): pipeline + stores + user_message with category_map splitter** — most of the spec's Phase 2.
8. **feat(backend): ADK client + locked prompt + config endpoints** — last backend piece.
9. **feat(frontend): Gradio four-tab UI** — Streamlit removed.
10. **chore: smoke script, README updates, .env.example** — final polish.

Implementation does not start until: (a) this spec is reviewed by the user, (b) the implementation plan is reviewed, and (c) the worktree is created via `superpowers:using-git-worktrees`. Per repo convention, no `Co-Authored-By` trailers on commits.

## 15. Open assumptions to verify during implementation

1. **HF Hub repo is private and you have admin access** — the repo `CallMeDaniel/tongue-resnet50-v1` (or whatever id you choose) must exist and accept the two `.pth` uploads before the registry-load succeeds end-to-end.
2. **Class-name order in registry matches Amanda's training** — `FRONT_LABELS` taken from `inference.py` line 58 verbatim; if Amanda re-trained with a different order, decoding will be silently wrong without updating the YAML.
3. **`map_location` works on the deployment machine** — Amanda's `inference.py` defaults to `mps` on Mac; backend autodetects `cuda > mps > cpu`.
4. **`google-adk` Vertex auth available** — backend assumes a service-account JSON via `GOOGLE_APPLICATION_CREDENTIALS`. If you'd rather use direct `google-genai` API key auth, swap the client wrapper; the rest is unchanged.
5. **`resnet50` `fc` topology is `2048 → N` linear, no extra heads** — if Amanda's `.pth` was trained with a different head topology, `load_state_dict` will fail at load with a clear error.

## 16. Out of scope (named, not sneaked in later)

Same as 2026-05-06 spec §12. In particular:

- Authentication / authorisation on any endpoint.
- Audit log of config changes.
- Rate limiting on `/api/analyze`.
- Persistent session history.
- Multimodal LLM fallback (sending image to Gemini if classifiers fail).
- Streaming partial results (SSE).
- Structured config forms (raw YAML/text only for POC).
- Per-task retraining of the composite heads into the spec's original 11-head structure (a follow-up project; the registry already supports it via additional `heads:` entries when those weights exist).
- ONNX export of Amanda's `.pth` (a follow-up optimisation if/when ORT becomes worth the export step).

---

**Approval:** This design has been walked through section by section with the user and approved. Implementation plan to follow via `superpowers:writing-plans`.
