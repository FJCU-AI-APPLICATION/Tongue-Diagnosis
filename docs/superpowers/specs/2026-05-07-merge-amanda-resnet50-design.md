# Merge Amanda's ResNet50 Into Tongue-Diagnosis — Design

**Date:** 2026-05-07
**Status:** Design approved by user; spec committed; pending review and implementation as a focused delta on top of `main`.
**Layers on top of:** the 2026-05-06 POC implementation that landed via PRs #1 and #2 — `tongue-ai` (ONNX `TaskHead`, `Registry`, `inference.run_all`, preprocessing) and `tongue-backend` (FastAPI app with stores, pipeline, ADK client, four-tab Gradio frontend, pydantic-settings + typed Pydantic responses). That implementation followed the 2026-05-06 spec at [`docs/superpowers/specs/2026-05-06-tongue-poc-design.md`](./2026-05-06-tongue-poc-design.md). The registry skeleton expects 11 per-task ONNX models that **do not yet exist on disk**; this PR makes the POC functionally end-to-end by introducing PyTorch heads.
**Stack additions:** `huggingface_hub` (weight delivery), `[training]` extra (`scikit-learn`, `matplotlib`).

---

## 1. Goal

Amanda delivered two trained PyTorch ResNet50 weights (`best_resnet50_front.pth`, `best_resnet50_sublingual.pth`) covering most of the v4 TCM tongue inspection schema as **composite heads** (one head emits 14 mixed-category classes, the other emits 3 sublingual-vessel classes). This PR integrates them so the existing POC infrastructure on `main` becomes functionally end-to-end.

The merge is **additive**, not replacement-style:

- The existing ONNX `TaskHead` and `Registry` keep working — `weights_uri` becomes a sibling of `onnx_path` in the registry YAML.
- A new `PyTorchSession` shim presents the ONNX `InferenceSession` interface (`get_inputs()`, `run()`) backed by a `torchvision.models.resnet50`. `TaskHead` is unchanged.
- A new `WeightSource` resolves `hf://repo/file` URIs via `huggingface_hub` (with optional `HF_TOKEN`) and `local://path` URIs against the YAML's directory. Weights live in HF Hub instead of git.
- The default `registry.default.yaml` switches from 11 nonexistent ONNX heads to Amanda's 2 working PyTorch heads.
- A new `category_map` field on the registry maps each composite-head class back to its v4 category, so the per-request user message keeps emitting v4-shaped bullets that the locked Gemini prompt expects.

## 2. Why this shape

The 2026-05-06 spec assumed 11 per-task ONNX heads. Reality: only Amanda's 2 PyTorch composite heads exist. Two options were considered:

1. **Replace the ONNX TaskHead with a PyTorch one.** Cleanest if ONNX is dead. But it forces a full refactor of `TaskHead`, `Registry`, and tests landed two days ago in PRs #1+#2.
2. **Add PyTorch as a sibling format.** PyTorchSession shim plugs into the existing TaskHead unchanged. `_build_head` dispatches on whether YAML supplies `onnx_path` or `weights_uri`. ONNX path stays available for future heads.

Option 2 wins because it preserves the prior PR's work, keeps the diff focused, and leaves a path open for future per-task ONNX heads if the team retrains.

## 3. Scope of this PR (what changes vs origin/main)

| Area | origin/main | This PR |
|---|---|---|
| Inference runtime | ONNX Runtime via `_make_session(onnx_path)` | ONNX **and** PyTorch via `PyTorchSession` shim — TaskHead unchanged |
| Registry head schema | required `onnx_path` | `onnx_path` **or** `weights_uri` |
| Weight delivery | implicit local files at `../../ai/models/*.onnx` | `WeightSource` resolves `hf:` (HF Hub via `huggingface_hub`) or `local:` URIs |
| `Registry` dataclass | `heads`, `detector` | + `category_map: dict[str, dict[str, str]]` |
| `user_message.build` | one bullet per head, keyed by `head.task` | when `category_map` is present, predictions are split back into v4 categories and grouped (cross-head) |
| `AnalyzeResponse` | unchanged keys | + `category_map` echoed back so the frontend can render the v4 breakdown |
| Default registry | 11 ONNX heads referencing nonexistent files | 2 PyTorch heads (`front` 14-class, `sublingual` 3-class) on HF Hub + `category_map` covering all 17 classes |
| `tongue-ai` deps | `onnxruntime`, `pyyaml`, … | + `huggingface_hub`; `[training]` extra adds `scikit-learn`, `matplotlib` |
| Training | none | `tongue_ai.training` subpackage (cleaned-up Amanda scripts) |

**Unchanged on `main`:** the `TaskHead` class itself, the FastAPI app factory and lifespan hooks, all three stores (prompt / llm / registry), all four config routes, the four-tab Gradio UI, `pydantic-settings` for env config, typed Pydantic response models, the locked 大眾版 prompt + 9 證素 contract, the `DISCLAIMER` constant, the per-head error containment in `run_all`.

## 4. Architecture (deltas only)

```
                                        ┌─────────────────────────────┐
                                        │  registry.default.yaml      │
                                        │  - heads:                   │
                                        │    - weights_uri: hf:…front │
                                        │      class_names: [14 …]    │
                                        │    - weights_uri: hf:…sub   │
                                        │      class_names: [3 …]     │
                                        │  - category_map:            │
                                        │      front: { ... → 舌色 …} │
                                        │      sublingual: { ...→舌下絡脈 } │
                                        └──────────────┬──────────────┘
                                                       │ load_registry()
                                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Registry(heads=[TaskHead, TaskHead], category_map=...)  │
            └──────────────────────────────────────────────────────────┘
                                                       │
            for each head config:                      │
              if "weights_uri" in raw:                 │
                src = WeightSource(uri, base_dir)      │
                path = src.resolve()  # hf_hub_download or local
                session = build_pytorch_session(path, num_classes)
              else:  # onnx_path
                session = _make_session(onnx_path)     # existing
              TaskHead(session=session, …)             │
                                                       ▼
                                       pipeline.analyze(image_bytes, registry)
                                       └─► run_all(image, registry)
                                       └─► user_message.build(heads, registry.category_map)
                                       └─► llm.client.run(system, user, llm_cfg)
                                       └─► AnalyzeResponse(..., category_map=...)
```

`PyTorchSession` exposes:

- `get_inputs() → [Input(name="input")]`
- `run(_output_names, {"input": chw_float32_array}) → [logits_array_shape_1xN]`

Inside, it does `torch.from_numpy(chw).to(device)`, `model.eval()` + `with torch.no_grad(): logits = model(tensor)`, returns `logits.cpu().numpy()`. The existing `TaskHead._decode` then applies softmax and picks top-1 (single) or threshold (multi).

## 5. Modules added / modified

### Added

```
docs/superpowers/specs/2026-05-07-merge-amanda-resnet50-design.md   ← this file
packages/ai/src/tongue_ai/weights.py                                 ← WeightSource, parse_weights_uri, WeightFetchError
packages/ai/src/tongue_ai/pytorch_session.py                         ← PyTorchSession + build_pytorch_session
packages/ai/src/tongue_ai/training/__init__.py
packages/ai/src/tongue_ai/training/dataset.py                        ← TongueDataset (Label-Studio JSON)
packages/ai/src/tongue_ai/training/evaluate.py                       ← classification_report + Cohen's Kappa
packages/ai/src/tongue_ai/training/train_front.py                    ← argparse-driven training entrypoint
packages/ai/src/tongue_ai/training/train_sublingual.py
packages/ai/tests/test_weights.py
packages/ai/tests/test_pytorch_session.py
packages/ai/tests/test_training_dataset.py
```

### Modified

```
packages/ai/pyproject.toml                       + huggingface_hub; + [training] extra
packages/ai/src/tongue_ai/registry.py            dispatch on weights_uri vs onnx_path; parse category_map
packages/ai/tests/test_registry.py               + cases for weights_uri / category_map
packages/backend/src/tongue_backend/llm/user_message.py    accept optional category_map; group across heads
packages/backend/src/tongue_backend/pipeline.py            pass category_map; include in AnalyzeResponse
packages/backend/src/tongue_backend/models.py              + category_map field on AnalyzeResponse
packages/backend/tests/test_user_message.py                + cross-head merge tests
packages/backend/tests/test_pipeline.py                    + category_map echo test
packages/backend/config/registry.default.yaml              full replace: 2 PyTorch heads + category_map
packages/backend/.env.example                              + HF_TOKEN
README.md                                                  + HF Hub + training notes
```

## 6. Composite-head → v4 user-message splitter (new behaviour)

`user_message.build(heads, category_map=None)`:

- If `category_map` is None or empty → existing behaviour (one bullet per `head.task`, unchanged).
- Otherwise:
  - For each non-error head, look up each prediction's `label` in `category_map[head.task]` to find its v4 category.
  - Group predictions by v4 category, merging across heads (e.g. if `front` outputs `瘀血絲` and `sublingual` outputs `怒張`, both end up under `舌下絡脈`).
  - Emit one bullet per v4 category that has ≥1 prediction. Categories with no signal are absent. Predictions whose label is not in the map are silently dropped (prevents accidental contract violations).
  - Heads with `error` are skipped silently (matches existing behaviour).
  - Maintains insertion order via `OrderedDict` so output is deterministic.

Worked example (Amanda's heads):

```
本次舌診判讀結果：

- 舌色：淡紅（0.78）
- 舌下絡脈：瘀血絲（0.51）、怒張（0.72）

請依規則輸出大眾版報告。
```

The locked system prompt continues to apply its 9-證素 lookup verbatim; this builder never touches the prompt, only the per-request user message.

## 7. Registry YAML

The new `registry.default.yaml`:

```yaml
detector: null                                # detector still wired but inert (Amanda's models accept whole tongue photos)

heads:
  - task: front
    head_type: single
    weights_uri: hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: [淡紅, 紅, 淡, 絳, 青紫, 暗, 微紅, 胖大, 瘦薄, 嫩, 偏斜, 齒痕, 無異常, 瘀血絲]

  - task: sublingual
    head_type: single
    weights_uri: hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_sublingual.pth
    input_size: [224, 224]
    normalise: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    class_names: [怒張, 曲張, 囊柱囊泡]

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
    無異常: 舌質                # default-normal per v4 schema
    瘀血絲: 舌下絡脈
  sublingual:
    怒張: 舌下絡脈
    曲張: 舌下絡脈
    囊柱囊泡: 舌下絡脈
```

`task` keeps its meaning ("the head's identity"); for composite heads it's the head NAME (`front`, `sublingual`) rather than a v4 category. The category_map is the bridge.

## 8. Hugging Face Hub setup

- **Repo:** `CallMeDaniel/tongue-resnet50-v1` (private). Already exists with both weight files uploaded as part of this work.
- **Auth:** `HF_TOKEN` env var (read by `huggingface_hub` automatically). `.env.example` documents this. Backend reads it at registry-load time.
- **Cache:** `huggingface_hub` manages its own cache under `~/.cache/huggingface/hub`. No second cache layer.
- **Local override:** setting `weights_uri: local:relative/path` skips HF and reads from disk — useful when iterating on weights before re-uploading.

## 9. Training subpackage

- **Location:** `packages/ai/src/tongue_ai/training/`
- **Install:** `uv sync --all-packages --extra training` (pulls scikit-learn, matplotlib).
- **Entry points:**
  - `python -m tongue_ai.training.train_front --labels-json … --img-dir … --weights-out …`
  - `python -m tongue_ai.training.train_sublingual --labels-json … --img-dir … --weights-out …`
- **Cleanups vs Amanda's originals:** argparse-driven paths (no hardcoded `/Users/chenyu/...`), matplotlib font fallback chain (no Mac-only `Heiti TC`), `TongueDataset` extracted to `dataset.py`, classification report + Cohen's Kappa preserved as `evaluate.py` (lazy sklearn import).

## 10. Error-handling deltas

| Case | Behaviour |
|---|---|
| `WeightFetchError` (HF download fails) | Currently `_build_head` raises and load_registry propagates. (Future: per-head partial-fail like the original POC spec; out of scope for this PR.) |
| `RuntimeError` from `load_state_dict` (size mismatch) | Same — propagates with explicit "expected N classes, got M" message. |
| Per-head `predict()` exception | Unchanged from origin/main: head's entry becomes `{..., "error": "..."}`; user message excludes it; rest proceeds. |
| `category_map` references a class not in any head | Tolerated (orphan entries are harmless). |
| Head class missing from `category_map` | Currently silently dropped from the user message (forgiving). Future enhancement: registry-load validation 422. |

## 11. Testing deltas (relative to origin/main's existing 63 tests)

The existing test suite stays green. Additions:

| Module | Tests |
|---|---|
| `weights` | URI parsing (local / hf / unknown / no scheme); local resolution; hf resolution with mocked `hf_hub_download`; HF_TOKEN env passthrough; error wrapping |
| `pytorch_session` | shim returns logits with the expected shape and dtype; `model.eval()` is called; CPU path covered (mps/cuda skipped in CI) |
| `registry` | YAML with `weights_uri` builds a TaskHead via PyTorchSession; YAML with `onnx_path` still works (regression); `category_map` parsed when present |
| `user_message` | when `category_map=None` behaviour matches today; with category_map, single-head + cross-head merging + skipped error head + orphan-class drop |
| `pipeline` | response includes `category_map` echoed from registry |
| `training/dataset` | record → (PIL.Image, label_idx); missing file falls back to blank 224×224; `__len__` matches |

## 12. Out of scope

- Per-head partial-fail registry-load (future; current behaviour: total fail if any head fails to build).
- `category_map` validation strictness (currently forgiving).
- ONNX export of Amanda's weights.
- Retraining the composite heads as the spec's original 11-head structure.
- Authentication, rate limiting, audit log, persistent session history, multimodal LLM fallback, SSE streaming.

## 13. PR commit sequence

8 focused commits on `feat/pytorch-amanda`:

1. `docs: add merge spec for Amanda's PyTorch ResNet50 integration`
2. `chore(ai): add huggingface_hub for HF Hub weight fetching`
3. `feat(ai): add WeightSource (hf:// + local: URI resolution)`
4. `feat(ai): add PyTorchSession shim — TaskHead-compatible PyTorch backend`
5. `feat(ai): registry dispatches on weights_uri; expose category_map`
6. `feat(backend): thread category_map through user_message + AnalyzeResponse`
7. `feat(backend): replace registry.default.yaml with Amanda's 2 composite heads`
8. `feat(ai): add training subpackage with cleaned-up Amanda scripts`

Per repo memory: no `Co-Authored-By` trailers on commits.

## 14. Open assumptions

1. HF Hub repo `CallMeDaniel/tongue-resnet50-v1` (private) already holds both weight files. Verified end-to-end during this work.
2. `FRONT_LABELS` order in registry YAML matches Amanda's training script (`inference.py` line 58) verbatim — required because `fc` layer index → label decoding depends on it. Validated: `len(class_names) == fc.out_features` enforced at `load_state_dict` time.
3. `_autodetect_device()` runs `cuda > mps > cpu`; backend autodetects.
4. Origin/main's `google-adk` integration works as-is; no changes to the LLM client.
5. `resnet50` `fc` topology is `2048 → N` linear; `load_state_dict` will raise a clear error at load if Amanda re-trained with a different topology.

---

**Approval:** This design has been walked through with the user. Implementation follows in 8 focused commits.
