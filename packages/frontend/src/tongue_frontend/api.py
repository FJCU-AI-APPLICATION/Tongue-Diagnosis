"""Thin httpx client to the FastAPI backend."""

from __future__ import annotations

import os

import httpx


BASE_URL = os.environ.get("TONGUE_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 60.0


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


def analyze(image_bytes: bytes, filename: str = "tongue.jpg") -> dict:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    with _client() as c:
        r = c.post("/api/analyze", files=files)
        r.raise_for_status()
        return r.json()


def get_config(section: str) -> dict:
    with _client() as c:
        r = c.get(f"/api/config/{section}")
        r.raise_for_status()
        return r.json()


def put_config(section: str, content: str) -> dict:
    with _client() as c:
        r = c.put(f"/api/config/{section}", json={"content": content})
        if r.status_code == 422:
            return {"error": r.json().get("error", "validation failed")}
        r.raise_for_status()
        return r.json()


def reset_config(section: str) -> dict:
    with _client() as c:
        r = c.post(f"/api/config/{section}/reset")
        r.raise_for_status()
        return r.json()


def reload_registry() -> dict:
    with _client() as c:
        r = c.post("/api/config/registry/reload")
        if r.status_code == 422:
            return {"error": r.json().get("error", "reload failed")}
        r.raise_for_status()
        return r.json()


def health() -> dict:
    with _client() as c:
        r = c.get("/health")
        r.raise_for_status()
        return r.json()
