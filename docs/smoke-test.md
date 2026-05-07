# Smoke test

Manual end-to-end verification. Run before every merge to `main`.

## Prereqs

- `HF_TOKEN` exported with read access to the private weights repo
  (`CallMeDaniel/tongue-resnet50-v1` by default — adjust the `weights_uri`
  values in `packages/backend/config/registry.default.yaml` if you forked
  or renamed it). Place it in `packages/backend/.env` (gitignored) or
  export inline.
- ADK auth configured (e.g., `gcloud auth application-default login`,
  or set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON).

## Steps

1. `uv sync --all-packages`
2. Terminal A: `uv run uvicorn tongue_backend.app:app --port 8000`
   - First boot fetches the two PyTorch ResNet50 weights from HF Hub
     (~190 MB; subsequent boots use the local HF cache).
   - Watch for `Loaded N heads` in the log — should be 2.
3. Terminal B: `uv run python -m tongue_frontend.app`
4. Open http://localhost:7860 — confirm 4 tabs render.
5. **Tab 1 — 舌診分析:** upload `tongue.jpg`, click 分析. Expect:
   - Two rows in the heads dataframe: `front` and `sublingual`
   - Markdown comment with sections 主要中醫體質 / 次要中醫體質 / 體質說明 / 證素列表 / 警語
   - Disclaimer visible
   - Advanced panel shows `user_message` (with **v4-category bullets** — 舌色, 舌質, 舌態, 舌下絡脈 — *not* the raw `front`/`sublingual` head names) and `timing_ms`
6. **Tab 2 — 提示詞設定:** trim the prompt to a one-liner (`回應一句中文`) → 儲存 → re-run analyze → comment now reflects the edit. Click 還原預設 → confirm full prompt returns.
7. **Tab 3 — LLM 設定:** lower `temperature` to 0.0 → 儲存 → re-run; expect roughly identical output across two runs.
8. **Tab 4 — 模型設定:** swap one head's `weights_uri` to a non-existent file (e.g. append `.bad` to the URI) → 儲存 → Apply & Reload Models → expect the response to list that head under `failed` and a non-empty `previous_kept`. Click 還原預設 → Apply & Reload Models to restore. (For per-task heads added later via `onnx_path`, the same flow works with a `threshold` change on a multi-label head.)
9. **Failure path:** stop the backend mid-flow → frontend shows `無法連線到後端`.
10. **Empty input:** click 分析 with no image → frontend shows `請選擇或拍攝照片`.

## Recovery

If you've broken something via the UI:
- Tab 2/3/4: click 還原預設.
- Or remove `*.current.*` files under `packages/backend/{prompts,config}/` and restart the backend.

## What changed since the original 2026-05-06 plan

Originally the registry shipped 11 per-task ONNX heads (舌色, 舌質, … 舌下絡脈). After PR #4 it ships **two PyTorch composite heads** (`front` covering 14 mixed-category classes, `sublingual` covering 3 sublingual-vessel classes), with a `category_map` that re-projects each prediction back to a v4 schema bullet so the locked Gemini prompt's lookup table still applies. The `onnx_path` registry key still works per-head — future per-task ONNX heads can be added alongside Amanda's PyTorch heads.
