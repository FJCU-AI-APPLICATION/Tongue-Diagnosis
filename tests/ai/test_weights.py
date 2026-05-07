"""Tests for ai.weights — URI parsing + resolution."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai.weights import (
    WeightFetchError,
    WeightSource,
    parse_weights_uri,
)


def test_parse_local_uri_returns_local_scheme_and_path():
    scheme, ref = parse_weights_uri("local:packages/ai/models/best.pth")
    assert scheme == "local"
    assert ref == "packages/ai/models/best.pth"


def test_parse_hf_uri_returns_hf_scheme_and_repo_filename():
    scheme, ref = parse_weights_uri("hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth")
    assert scheme == "hf"
    assert ref == "CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth"


def test_parse_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        parse_weights_uri("s3:bucket/x.pth")


def test_parse_uri_without_scheme_raises():
    with pytest.raises(ValueError, match="missing scheme"):
        parse_weights_uri("packages/ai/models/x.pth")


def test_local_uri_resolves_relative_to_base_dir(tmp_path):
    base = tmp_path / "registry-dir"
    weights_file = base / "models" / "x.pth"
    weights_file.parent.mkdir(parents=True)
    weights_file.write_bytes(b"weights")
    src = WeightSource(uri="local:models/x.pth", base_dir=base)
    resolved = src.resolve()
    assert resolved == weights_file


def test_local_uri_missing_file_raises_weight_fetch_error(tmp_path):
    src = WeightSource(uri="local:nope.pth", base_dir=tmp_path)
    with pytest.raises(WeightFetchError, match="not found"):
        src.resolve()


def test_hf_uri_calls_hf_hub_download_with_repo_and_filename(tmp_path):
    fake_path = tmp_path / "best_resnet50_front.pth"
    fake_path.write_bytes(b"weights")
    src = WeightSource(
        uri="hf:CallMeDaniel/tongue-resnet50-v1/best_resnet50_front.pth",
        base_dir=tmp_path,
    )
    with patch("ai.weights.hf_hub_download", return_value=str(fake_path)) as m:
        resolved = src.resolve()
    m.assert_called_once_with(
        repo_id="CallMeDaniel/tongue-resnet50-v1",
        filename="best_resnet50_front.pth",
        token=None,
    )
    assert resolved == fake_path


def test_hf_uri_passes_token_from_env(tmp_path, monkeypatch):
    fake_path = tmp_path / "x.pth"
    fake_path.write_bytes(b"")
    monkeypatch.setenv("HF_TOKEN", "secret")
    src = WeightSource(uri="hf:org/repo/x.pth", base_dir=tmp_path)
    with patch("ai.weights.hf_hub_download", return_value=str(fake_path)) as m:
        src.resolve()
    assert m.call_args.kwargs["token"] == "secret"


def test_hf_uri_wraps_underlying_error(tmp_path):
    src = WeightSource(uri="hf:org/repo/x.pth", base_dir=tmp_path)
    with patch("ai.weights.hf_hub_download", side_effect=ConnectionError("boom")):
        with pytest.raises(WeightFetchError) as excinfo:
            src.resolve()
    assert "org/repo" in str(excinfo.value)
    assert "x.pth" in str(excinfo.value)
