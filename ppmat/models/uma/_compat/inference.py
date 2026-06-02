from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import paddle


class UMATask(str, Enum):
    OMOL = "omol"
    OMAT = "omat"
    ODAC = "odac"
    OC20 = "oc20"
    OC25 = "oc25"
    OMC = "omc"


CHARGE_RANGE = [-100, 100]
SPIN_RANGE = [0, 100]
DEFAULT_CHARGE = 0
DEFAULT_SPIN_OMOL = 1
DEFAULT_SPIN = 0


@dataclass
class InferenceSettings:
    tf32: bool = False
    activation_checkpointing: bool | None = True
    merge_mole: bool = False
    compile: bool = False
    external_graph_gen: bool | None = False
    internal_graph_gen_version: int | None = 2
    edge_chunk_size: int | None = None
    use_quaternion_wigner: bool | None = True
    base_precision_dtype: str | paddle.dtype = "float32"
    execution_mode: str | None = None
    predict_untrained_forces: set[str] = field(default_factory=set)
    predict_untrained_stress: set[str] = field(default_factory=set)
    predict_untrained_hessian: set[str] = field(default_factory=set)
    hessian_vmap: bool = True
    auto_add_default_untrained_tasks: bool = True


@dataclass
class OutputSpec:
    dim: list[int]
    dtype: Any


@dataclass
class Task:
    name: str
    level: str
    property: str
    out_spec: OutputSpec
    normalizer: Any
    datasets: list[str]
    loss_fn: Any = None
    element_references: Any = None
    metrics: list[str] = field(default_factory=list)
    train_on_free_atoms: bool = True
    eval_on_free_atoms: bool = True
    inference_only: bool = False
