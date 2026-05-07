from unittest.mock import MagicMock

import numpy as np

from ai.task_head import TaskHead
from ai.types import Normalisation


def _fake_session(logits: np.ndarray) -> MagicMock:
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.run.return_value = [logits]
    return session


def test_single_label_decodes_argmax():
    # Pre-computed probabilities; set already_probs=True so the score is
    # reported verbatim (no softmax). Mirrors models that emit softmax outputs.
    probs = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
    head = TaskHead(
        session=_fake_session(probs),
        name="舌色",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        class_names=["A", "B", "C"],
        already_probs=True,
    )
    result = head.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.task == "舌色"
    assert result.head_type == "single"
    assert len(result.predictions) == 1
    assert result.predictions[0].label == "B"
    assert abs(result.predictions[0].score - 0.7) < 1e-5


def test_single_label_softmaxes_when_not_already_probs():
    # Raw logits; softmax([0.1, 0.7, 0.2])[1] ≈ 0.464 — score lands in [0, 1].
    logits = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
    head = TaskHead(
        session=_fake_session(logits),
        name="舌色",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.0]*3, std=[1.0]*3),
        class_names=["A", "B", "C"],
    )
    result = head.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.predictions[0].label == "B"
    assert 0.4 < result.predictions[0].score < 0.5


def test_multi_label_thresholds():
    # Sigmoid probs: [0.9, 0.4, 0.95]
    logits = np.array([[0.9, 0.4, 0.95]], dtype=np.float32)
    head = TaskHead(
        session=_fake_session(logits),
        name="舌質",
        head_type="multi",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.0]*3, std=[1.0]*3),
        class_names=["齒痕", "胖大", "嫩"],
        threshold=0.5,
        already_probs=True,
    )
    result = head.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    labels = [p.label for p in result.predictions]
    assert labels == ["齒痕", "嫩"]


def test_predict_catches_exception_returns_error():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input"
    session.run.side_effect = RuntimeError("bad ONNX")
    head = TaskHead(
        session=session,
        name="舌色",
        head_type="single",
        input_size=(224, 224),
        normalise=Normalisation(mean=[0.0]*3, std=[1.0]*3),
        class_names=["A", "B"],
    )
    result = head.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.error is not None
    assert "bad ONNX" in result.error
    assert result.predictions == []
