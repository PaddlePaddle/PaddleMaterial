
import hydra
import omegaconf
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main
from omegaconf import OmegaConf


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"),
    config_name="default",
    version_base="1.1",
)
def mattergen_main(cfg: omegaconf.DictConfig):
#   paddle.set_float32_matmul_precision("high")
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.set_readonly(config, True)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    main(config)


if __name__ == "__main__":
    mattergen_main()

#  PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python -m paddle.distributed.launch --gpus="6,7" scripts/run.py 
#  PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python scripts/run.py 