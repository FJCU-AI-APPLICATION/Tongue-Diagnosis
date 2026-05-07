"""Read/write the current system prompt; first-boot copies from default."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptStore:
    default_path: Path
    current_path: Path

    def _ensure_current(self) -> None:
        if not self.current_path.exists():
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.default_path, self.current_path)

    def load_current(self) -> str:
        self._ensure_current()
        return self.current_path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_path.write_text(content, encoding="utf-8")

    def reset(self) -> None:
        shutil.copyfile(self.default_path, self.current_path)

    def is_default(self) -> bool:
        if not self.current_path.exists():
            return True
        return self.current_path.read_text(encoding="utf-8") == self.default_path.read_text(encoding="utf-8")
