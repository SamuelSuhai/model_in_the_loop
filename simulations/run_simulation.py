import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import List, Dict, Any
# TODO: refactor by importing loop components based on config
from simulations.loop_components.stimulators.noise_file_copier import NoiseFileCopier
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

    # create stimulator
    stimulator = NoiseFileCopier(
        home_directory=stimulators_cfg.home_directory, # type: ignore
        source_path=stimulators_cfg.source_path, # type: ignore

        # from overall configs
        dir_where_new_stim_appear=cfg.dir_where_new_stim_appear, # type: ignore
        debug = cfg.debug, # type: ignore

    )
    


    # create recorder  
    recorder = FileCopier(
        source_dir= recorders_cfg.source_dir, # type: ignore
        source_exp_dir=recorders_cfg.source_exp_dir, # type: ignore
        source_smp_file=recorders_cfg.source_smp_file, # type: ignore
        source_smh_file=recorders_cfg.source_smh_file, # type: ignore
        ini_file=recorders_cfg.ini_file, # type: ignore
        target_dir=recorders_cfg.target_dir, # type: ignore
        new_file_name_base=recorders_cfg.new_file_name_base, # type: ignore

        # from overall configs
        stimulus_type = cfg.stimulus_type,
        debug = cfg.debug, # type: ignore
    )

    # create preprocessor
    preprocessor = Preprocessor1(
                    username=preprocessors_cfg.username, # type: ignore
                    home_directory=preprocessors_cfg.home_directory, # type: ignore
                    path_to_djimaging_rel_to_home=preprocessors_cfg.path_to_djimaging_rel_to_home, # type: ignore
                    path_to_djconfig_rel_to_home=preprocessors_cfg.path_to_djconfig_rel_to_home, # type: ignore
                    userinfo=preprocessors_cfg.userinfo, # type: ignore
                    
                    # from overall configs 
                    dir_where_new_stim_appear=cfg.dir_where_new_stim_appear,
                    openretina_processed_data_path=cfg.openretina_processed_data_path,
                    stimulus_shape=cfg.stimulus_shape,
                    stimulus_type=cfg.stimulus_type,
                    debug=cfg.debug, # type: ignore
                    
                    )
  
    # create model
    model = None

    
    return recorder, preprocessor, model, stimulator



@hydra.main(version_base=None, config_path="../config/", config_name="config")
def run_simulation(cfg: DictConfig) -> None:


    recorder, preprocessor, model, stimulator = create_loop_components(cfg)

    # MAKE SURE TO ADD SLEEP TIME TO AVOID CRASHING SERVER
    
    


if __name__ == "__main__":
    run_simulation()