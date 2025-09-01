from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf

from ppmat.predictor import PPMatPredictor
from ppmat.predictor.structures import build_init_structures
from ppmat.utils import logger


@hydra.main(
    config_path="application_configs", 
    config_name="property.yaml", 
    version_base=None
    )
def main(cfg: DictConfig):
    # Save the loaded config
    OmegaConf.save(cfg, "config_saved.yaml")

    # Initialize logger
    log_file=cfg.logger.get("log_file", "out.log")
    logger.init_logger( log_file=log_file, log_level=cfg.get("log_level", "INFO") )
    logger.info("[PPMaterial] Logger initialized")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file path    : {os.path.abspath(log_file)}")

    # Initialize the model
    load_model = instantiate(cfg.model)
    predictor = PPMatPredictor(
        config_path = cfg.run.config_path,
        device = cfg.device,
        **load_model 
    )
    
    # Detect interface type and interface object
    if cfg.get("interface") is not None:
        # Read interface type
        interface_type = cfg.interface.get("type", None)  
        logger.info( f"Interface type is {interface_type}" )
        # Load inference model
        predictor.load_inference_model( interface_type=interface_type )
        # Initialize the interface object
        interface_obj = instantiate(cfg.interface, predictor=predictor)
        logger.info(f"Interface object is {interface_obj}")
    else:
        # Load inference model
        predictor.load_inference_model( interface_type=None )
    
    # Load structures
    files, structures = build_init_structures(cfg, predictor)

    if cfg.get("task") is not None:
        # Initialize the task
        task = instantiate(cfg.task)
        # Run the task
        task(interface_obj, structures)
    else:
        predictor.get_predict(files, structures)
    
    logger.info("All tasks finished successfully.")

if __name__ == "__main__":
    main()
    