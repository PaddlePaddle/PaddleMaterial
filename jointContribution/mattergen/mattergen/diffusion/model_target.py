from enum import Enum
from typing import Mapping, Union


class ModelTarget(Enum):
    """Specifies what the score model is trained to predict.
    Only relevant for fields that are corrupted with an SDE."""

    score_times_std = "score_times_std"
    logits = "logits"


ModelTargets = Mapping[str, Union[ModelTarget, str]]
