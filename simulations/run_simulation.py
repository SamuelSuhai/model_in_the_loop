import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import List, Dict, Any
# TODO: refactor by importing loop components based on config
# from simulations.loop_components.stimulus_file_copier import StimulusFileCopier
# from simulations.loop_components.recording_file_copier import RecordingFileCopier
from loop_components.dj_wrappers import OpenRetinaWrapper
from time import sleep


def create_loop_components(
    cfg: DictConfig,
    ):
    
    # # create stimulator
    # stimulator = StimulusFileCopier(
    #     repo_directory=cfg.paths.repo_directory, # type: ignore
    #     stimulus_type=cfg.paths.stimulus_type, # type: ignore
    #     debug = cfg.debug, # type: ignore

    # )
    
    # # create recorder  
    # recorder = RecordingFileCopier(
        
    #     # from overall configs        
    #     repo_directory=cfg.paths.repo_directory, # type: ignore
    #     stimulus_type = cfg.paths.stimulus_type,
    #     debug = cfg.debug, # type: ignore
    # )

    # create preprocessor
    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "True"
    
    openretinawrapper = OpenRetinaWrapper(
                    username=cfg.DJ.username, # type: ignore
                    home_directory=cfg.paths.home_directory, # type: ignore
                    repo_directory=cfg.paths.repo_directory, # type: ignore
                    dj_config_directory= cfg.paths.dj_config_directory, # type: ignore
                    rgc_output_directory= cfg.paths.rgc_output_directory, # type: ignore
                    userinfo= cfg.DJ.userinfo, # type: ignore
                    # from overall configs 
                    debug=cfg.debug, # type: ignore
                    
                    )
  
    # create model
    model = None

    return openretinawrapper
    # return recorder, openretinawrapper, model, stimulator



@hydra.main(version_base="1.3", config_path="../config/", config_name="config",)
def run_simulation(cfg: DictConfig) -> None:


    # recorder, openretinawrapper, model, stimulator = create_loop_components(cfg)
    openretinawrapper = create_loop_components(cfg)
    openretinawrapper.setup()
    openretinawrapper.process_iteration_data()

    
    


if __name__ == "__main__":
    run_simulation()