"""Thin httpx client for the FastAPI backend."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class APIClient:
    base_url: str = "http://localhost:8000"
    timeout: float = 60.0

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> dict:
        with self._client() as c:
            return c.get("/health").json()

    def analyze(self, *, filename: str, content: bytes, content_type: str) -> dict:
        files = {"file": (filename, content, content_type)}
        with self._client() as c:
            return c.post("/api/analyze", files=files).json()

    def get_config(self, section: str) -> dict:
        with self._client() as c:
            return c.get(f"/api/config/{section}").json()

    def put_config(self, section: str, content: str) -> dict:
        with self._client() as c:
            return c.put(f"/api/config/{section}", json={"content": content}).json()

    def reset_config(self, section: str) -> dict:
        with self._client() as c:
            return c.post(f"/api/config/{section}/reset").json()

    def reload_registry(self) -> dict:
        with self._client() as c:
            return c.post("/api/config/registry/reload").json()
