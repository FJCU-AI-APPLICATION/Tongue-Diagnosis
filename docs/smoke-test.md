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
