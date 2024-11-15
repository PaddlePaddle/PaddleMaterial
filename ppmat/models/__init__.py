import copy

from ppmat.models.dimenet.dimenet import DimeNetPlusPlusWrap
from ppmat.models.gemnet.gemnet import GemNetT
from ppmat.models.megnet.megnet import MEGNetPlus
from ppmat.utils import logger

__all__ = ["MEGNetPlus", "DimeNetPlusPlusWrap", "GemNetT"]


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
