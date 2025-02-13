import copy

from ppmat.models.chgnet.model import CHGNet
from ppmat.models.chgnet_v2.model.model_v2 import CHGNet_v2
from ppmat.models.comformer.comformer import iComformer
from ppmat.models.diffcsp.diffcsp import CSPDiffusion
from ppmat.models.diffcsp.diffcsp_d3pm import CSPDiffusionWithD3PM
from ppmat.models.diffcsp.diffcsp_with_guidance import CSPDiffusionWithGuidance
from ppmat.models.diffcsp.diffcsp_with_type import CSPDiffusionWithType
from ppmat.models.dimenet.dimenet import DimeNetPlusPlusWrap
from ppmat.models.gemnet.gemnet import GemNetT
from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.megnet.megnet import MEGNetPlus

# from ppmat.models.digress.base_model import (
#     MolecularGraphTransformer,
#     ContrastGraphTransformer,
#     ConditionGraphTransformer
# )
from ppmat.utils import logger

__all__ = [
    "MEGNetPlus",
    "DimeNetPlusPlusWrap",
    "GemNetT",
    "CSPDiffusion",
    "CSPDiffusionWithType",
    "CSPDiffusionWithD3PM",
    "CSPDiffusionWithGuidance",
    "MatterGen",
    "iComformer",
    # "MolecularGraphTransformer",
    # "ContrastGraphTransformer",
    # "ConditionGraphTransformer",
    "CHGNet",
    "CHGNet_v2",
]


def build_model(cfg):
    """Build model

    Args:
        cfg (DictConfig): Arch config.

    Returns:
        nn.Layer: Model.
    """
    cfg = copy.deepcopy(cfg)
    arch_cls = cfg.pop("__name__")
    arch = eval(arch_cls)(**cfg)

    logger.debug(str(arch))

    return arch
