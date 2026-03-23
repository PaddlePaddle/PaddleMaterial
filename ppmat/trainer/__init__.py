from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.trainer.ecformer_trainer import ECFormerTrainer

__all__ = ["BaseTrainer", "build_trainer", "ECFormerTrainer"]


def build_trainer(cfg, **kwargs):

    class_name = cfg.get("__class_name__", "BaseTrainer")
    init_params = cfg.get("__init_params__", {})
    trainer = eval(class_name)(**init_params, **kwargs)
    return trainer
