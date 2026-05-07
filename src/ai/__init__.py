"""Tongue AI -- detection + registry-driven multi-head classification."""

from ai.types import BBox, ClassScore, HeadResult, Normalisation
from ai.task_head import TaskHead
from ai.registry import Registry, RegistryError, load_registry
from ai.inference import run_all
from ai.detection import detect_tongue

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
