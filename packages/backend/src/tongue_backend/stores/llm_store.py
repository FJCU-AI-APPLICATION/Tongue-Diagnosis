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
