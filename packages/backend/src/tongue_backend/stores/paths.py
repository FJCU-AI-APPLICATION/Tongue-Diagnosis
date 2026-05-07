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

SECRETS_DIR = _BACKEND_ROOT / "secrets"
GEMINI_API_KEY_FILE = SECRETS_DIR / "gemini_api_key"
