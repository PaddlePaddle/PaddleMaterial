import os
import os.path as osp
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))  # ruff: noqa

from ppmat.utils import logger
from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers


if __name__ == "__main__":
    
    # init logger
    logger_path = osp.join(sys.path[0], "test/create_dataset.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")
    logger.info(f"Test create new dataset.")

    set_signal_handlers()
    train_data_cfg = {
        "dataset": {
            "__class_name__": "MP2024Dataset",
            # "__class_name__": "MP2018Dataset",
            "__init_params__": {
                "path": osp.join(sys.path[0], "data/mp2024_train_130k/mp2024_train.txt"),
                # "path": osp.join(sys.path[0], "data/mp2018_train_60k/mp.2018.6.1_val.json"),
                "property_names": ["formation_energy_per_atom"],  # 这里改成你实际需要的标签名
                "build_structure_cfg": {
                    "format": "dict",
                    "num_cpus": 10,
                },
                "build_graph_cfg": {
                    "__class_name__": "FindPointsInSpheres",
                    "__init_params__": {
                        "cutoff": 4.0,
                        "num_cpus": 10,
                    },
                },
                "cache_path": osp.join(sys.path[0], "data/mp2024_train_130k_cache_find_points_in_spheres_cutoff_4/mp2024_train"),
                # "cache_path": osp.join(sys.path[0], "data/mp2018_train_60k_cache_find_points_in_spheres_cutoff_4/mp.2018.6.1_val"),
            },
            "num_workers": 4,
            "use_shared_memory": False,
        },
        "sampler": {
            "__class_name__": "BatchSampler",
            "__init_params__": {
                "shuffle": True,
                "drop_last": True,
                "batch_size": 10,
            },
        },
    }
    logger.info("Train data config:\n" + 
                "\n".join( f"{k}: {v}" for k, v in train_data_cfg.items() )
            )

    train_loader = build_dataloader(train_data_cfg)
