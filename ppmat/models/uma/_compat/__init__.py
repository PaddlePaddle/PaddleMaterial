"""Local compatibility shims for UMA without fairchem dependency."""

from . import gp_utils
from .base import HeadInterface
from .graph import generate_graph
from .inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    OutputSpec,
    Task,
    UMATask,
)
from .irreps import cg_change_mat, irreps_sum
from .registry import registry
from .utils import conditional_grad

__all__ = [
    "gp_utils",
    "HeadInterface",
    "generate_graph",
    "CHARGE_RANGE",
    "DEFAULT_CHARGE",
    "DEFAULT_SPIN",
    "DEFAULT_SPIN_OMOL",
    "SPIN_RANGE",
    "InferenceSettings",
    "OutputSpec",
    "Task",
    "UMATask",
    "cg_change_mat",
    "irreps_sum",
    "registry",
    "conditional_grad",
]
