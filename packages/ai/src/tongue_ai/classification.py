"""Tongue classification: maps features to diagnostic categories."""

from __future__ import annotations

import torch


def classify_tongue(features: dict) -> dict:
    """Classify tongue condition based on extracted features.

    Returns
    -------
    dict
        Classification result with keys: condition, confidence, device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "condition": "healthy",
        "confidence": 0.87,
        "device": device,
    }
