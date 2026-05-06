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
