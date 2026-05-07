"""Filesystem locations of default and current config files."""

from __future__ import annotations

from pathlib import Path


# parents: [0]=stores  [1]=backend  [2]=src  [3]=<repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSETS = _REPO_ROOT / "assets"

PROMPT_DEFAULT = _ASSETS / "prompts" / "system.default.md"
PROMPT_CURRENT = _ASSETS / "prompts" / "system.current.md"

LLM_DEFAULT = _ASSETS / "config" / "llm.default.yaml"
LLM_CURRENT = _ASSETS / "config" / "llm.current.yaml"

REGISTRY_DEFAULT = _ASSETS / "config" / "registry.default.yaml"
REGISTRY_CURRENT = _ASSETS / "config" / "registry.current.yaml"

SECRETS_DIR = _ASSETS / "secrets"
GEMINI_API_KEY_FILE = SECRETS_DIR / "gemini_api_key"
