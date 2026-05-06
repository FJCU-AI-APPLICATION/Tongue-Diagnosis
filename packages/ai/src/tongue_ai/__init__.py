"""Tongue AI -- detection + registry-driven multi-head classification."""

from tongue_ai.types import BBox, ClassScore, HeadResult, Normalisation
from tongue_ai.task_head import TaskHead
from tongue_ai.registry import Registry, RegistryError, load_registry
from tongue_ai.inference import run_all
from tongue_ai.detection import detect_tongue

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "ClassScore",
    "HeadResult",
    "Normalisation",
    "TaskHead",
    "Registry",
    "RegistryError",
    "load_registry",
    "run_all",
    "detect_tongue",
]
