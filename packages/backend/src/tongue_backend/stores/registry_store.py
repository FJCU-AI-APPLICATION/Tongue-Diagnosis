"""Read/validate/reload the registry YAML.

Save = persist + structural validation only.
Reload = load_registry (resolves weights, builds models, can partial-fail).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from tongue_ai.registry import Registry, RegistryError, load_registry, validate_registry_yaml


@dataclass(frozen=True)
class RegistryStore:
    default_path: Path
    current_path: Path

    def _ensure_current(self) -> None:
        if not self.current_path.exists():
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.default_path, self.current_path)

    def load_text(self) -> str:
        self._ensure_current()
        return self.current_path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp, validate, then atomic-replace
        tmp = self.current_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        try:
            validate_registry_yaml(tmp)
        except RegistryError:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(self.current_path)

    def reload(self) -> Registry:
        self._ensure_current()
        return load_registry(self.current_path, raise_on_partial_fail=False)

    def reset(self) -> None:
        shutil.copyfile(self.default_path, self.current_path)

    def is_default(self) -> bool:
        if not self.current_path.exists():
            return True
        return self.current_path.read_text(encoding="utf-8") == self.default_path.read_text(encoding="utf-8")
