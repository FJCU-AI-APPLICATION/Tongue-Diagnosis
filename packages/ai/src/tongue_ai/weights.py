"""Resolve weight-file URIs.

Two URI schemes:
  local:relative/path/to/file.pth   → resolved relative to base_dir
  hf:repo_owner/repo_name/filename  → fetched from Hugging Face Hub

`HF_TOKEN` env var is used for private repos when present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


class WeightFetchError(RuntimeError):
    """Raised when a weight URI cannot be resolved to a readable local file."""


def parse_weights_uri(uri: str) -> tuple[str, str]:
    if ":" not in uri:
        raise ValueError(f"missing scheme in weights uri: {uri!r}")
    scheme, _, ref = uri.partition(":")
    if scheme not in {"local", "hf"}:
        raise ValueError(f"unknown scheme {scheme!r} in weights uri: {uri!r}")
    return scheme, ref


@dataclass(frozen=True)
class WeightSource:
    uri: str
    base_dir: Path

    def resolve(self) -> Path:
        scheme, ref = parse_weights_uri(self.uri)
        if scheme == "local":
            path = (self.base_dir / ref).resolve()
            if not path.is_file():
                raise WeightFetchError(
                    f"local weights not found: {path} (uri={self.uri!r})"
                )
            return path
        if scheme == "hf":
            owner, name, *rest = ref.split("/")
            if not rest:
                raise WeightFetchError(
                    f"hf uri must include filename: {self.uri!r}"
                )
            repo_id = f"{owner}/{name}"
            filename = "/".join(rest)
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as exc:
                raise WeightFetchError(
                    f"failed to download {repo_id}/{filename}: {exc}"
                ) from exc
            return Path(downloaded)
        raise WeightFetchError(f"unhandled scheme {scheme!r}")
