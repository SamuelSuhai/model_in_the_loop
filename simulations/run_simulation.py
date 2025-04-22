import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import List, Dict, Any
# TODO: refactor by importing loop components based on config
from simulations.loop_components.recorders.file_copier import FileCopier
from simulations.loop_components.preprocessors.preprocessor1 import Preprocessor1
from time import sleep


#TODO: add a config schem
@dataclass
class mainconfig:
    pass


def create_loop_components(
    cfg: DictConfig,
    ):
    # retrieve config dicts 
    preprocessors_cfg: Dict[str, Any] = cfg.preprocessors
    models_cfg: dict = cfg.models
    recorders_cfg: Dict[str, Any] = cfg.recorders
    stimulators_cfg: dict = cfg.stimulators
    
    # create recorder  
    recorder = FileCopier(
        source_dir= recorders_cfg.source_dir,
        source_exp_dir=recorders_cfg.source_exp_dir,
        source_smp_file=recorders_cfg.source_smp_file,
        source_smh_file=recorders_cfg.source_smh_file,
        ini_file=recorders_cfg.ini_file,
        target_dir=recorders_cfg.target_dir,
        new_file_name_base=recorders_cfg.new_file_name_base,
    )

    # create preprocessor
    preprocessor = Preprocessor1(
                    username=preprocessors_cfg.username,
                    home_directory=preprocessors_cfg.home_directory,
                    path_to_djimaging_rel_to_home=preprocessors_cfg.path_to_djimaging_rel_to_home,
                    path_to_djconfig_rel_to_home=preprocessors_cfg.path_to_djconfig_rel_to_home,
                    userinfo=preprocessors_cfg.userinfo,
                    
                    # from overall configs 
                    stimulus_type=cfg.stimulus_type,
                    stimulus_file_path=cfg.stimulus_file_path,
                    )
  
    # create model
    model = None

    # create stimulator
    stimulator = None

    
    return recorder, preprocessor, model, stimulator



@hydra.main(version_base=None, config_path="../config/", config_name="config")
def run_simulation(cfg: DictConfig) -> None:


    recorder, preprocessor, model, stimulator = create_loop_components(cfg)

    # MAKE SURE TO ADD SLEEP TIME TO AVOID CRASHING SERVER
    
    


if __name__ == "__main__":
    run_simulation()