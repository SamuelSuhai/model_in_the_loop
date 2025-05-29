import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import List, Dict, Any
# TODO: refactor by importing loop components based on config
from simulations.loop_components.stimulus_file_copier import StimulusFileCopier
from simulations.loop_components.recording_file_copier import RecordingFileCopier
from simulations.loop_components.DJWrappers import Preprocessor1
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
    stimulator = StimulusFileCopier(
        repo_directory=cfg.repo_directory, # type: ignore
        stimulus_type=cfg.stimulus_type, # type: ignore
        debug = cfg.debug, # type: ignore

    )
    


    # create recorder  
    recorder = RecordingFileCopier(
        
        # from overall configs        
        repo_directory=cfg.repo_directory, # type: ignore
        stimulus_type = cfg.stimulus_type,
        debug = cfg.debug, # type: ignore
    )

    # create preprocessor
    preprocessor = Preprocessor1(
                    username=preprocessors_cfg.username, # type: ignore
                    home_directory=cfg.home_directory, # type: ignore
                    repo_directory=cfg.repo_directory, # type: ignore
                    path_to_djimaging_rel_to_home=preprocessors_cfg.path_to_djimaging_rel_to_home, # type: ignore
                    path_to_djconfig_rel_to_home=preprocessors_cfg.path_to_djconfig_rel_to_home, # type: ignore
                    userinfo=preprocessors_cfg.userinfo, # type: ignore
                    
                    # from overall configs 
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


    
    


if __name__ == "__main__":
    run_simulation()