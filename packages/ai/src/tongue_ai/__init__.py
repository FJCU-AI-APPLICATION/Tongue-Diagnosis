"""Tongue AI — PyTorch ResNet50 inference for tongue diagnosis."""
from tongue_ai.detection import detect_tongue
from tongue_ai.inference import run_all
from tongue_ai.registry import Registry, RegistryError, load_registry
from tongue_ai.task_head import PyTorchTaskHead, load_pytorch_head
from tongue_ai.types import (
    BBox,
    ClassScore,
    HeadResult,
    IMAGENET_NORMALISATION,
    Normalisation,
)
from tongue_ai.weights import WeightFetchError, WeightSource

__version__ = "0.2.0"

__all__ = [
    "BBox",
    "ClassScore",
    "HeadResult",
    "IMAGENET_NORMALISATION",
    "Normalisation",
    "PyTorchTaskHead",
    "Registry",
    "RegistryError",
    "WeightFetchError",
    "WeightSource",
    "__version__",
    "detect_tongue",
    "load_pytorch_head",
    "load_registry",
    "run_all",
]
