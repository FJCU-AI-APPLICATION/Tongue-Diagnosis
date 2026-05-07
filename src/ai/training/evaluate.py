"""Evaluation: classification report + Cohen's Kappa.

Imports scikit-learn lazily so the runtime AI package doesn't pull it in.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    target_labels: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """Run model.eval() over loader and return (report_str, kappa)."""
    from sklearn.metrics import classification_report, cohen_kappa_score

    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist() if hasattr(labels, "numpy") else list(labels))

    report = classification_report(
        all_targets,
        all_preds,
        target_names=target_labels,
        zero_division=0,
    )
    kappa = cohen_kappa_score(all_targets, all_preds)
    return report, float(kappa)
