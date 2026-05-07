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
