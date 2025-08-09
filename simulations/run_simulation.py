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
from simulations.loop_components.recording_file_copier import copy_rec_files,create_directory_structure
from loop_components.model_to_stimulus import from_data_to_mei_video
from time import sleep



def create_loop_components(
    cfg: DictConfig,
    ):

    # create preprocessor
    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

    openretinawrapper = OpenRetinaWrapper(
                    username=cfg.DJ.username, # type: ignore
                    
                    #paths
                    home_directory=cfg.paths.home_directory, # type: ignore
                    repo_directory=cfg.paths.repo_directory, # type: ignore
                    dj_config_directory= cfg.paths.dj_config_directory, # type: ignore
                    rgc_output_directory= cfg.paths.rgc_output_directory, # type: ignore
                    data_subfolders=cfg.data_subfolders, # type: ignore


                    userinfo= cfg.DJ.userinfo, # type: ignore

                    table_parameters=cfg.DJ.table_parameters, # type: ignore

                    # from overall configs
                    debug=cfg.debug, # type: ignore
                    plot_results=cfg.plot_results, # type: ignore

                    )

    # create model
    model = None

    return openretinawrapper
    # return recorder, openretinawrapper, model, stimulator





@hydra.main(version_base="1.3", config_path="../config/", config_name="config",)
def run_simulation(cfg: DictConfig) -> None:


    # create_directory_structure(cfg.DJ.userinfo.data_dir, cfg.data_subfolders.day, cfg.data_subfolders.experiment)
    # print("Copying recording files to repo ... ")
    # copy_rec_files(
    #     recording_files_dir=cfg.paths.recording_files_dir,  # type: ignore
    #     destination_base=cfg.DJ.userinfo.data_dir,  # type: ignore
    #     date=cfg.data_subfolders.day,  # type: ignore
    #     experiment=cfg.data_subfolders.experiment,  # type: ignore
    # )


    # ## The entire iteration
    openretinawrapper = create_loop_components(cfg)

    
    if cfg.clean_up_before: # type: ignore
        openretinawrapper.load_config()
        openretinawrapper.load_tables()
        openretinawrapper.clean_up(at_processing_stage="setup")


    # FULL LOOP
    
    openretinawrapper.setup()
    raw_neuron_data_dict = openretinawrapper.process_iteration_data()


    if raw_neuron_data_dict is not None: 
        from_data_to_mei_video(cfg, raw_neuron_data_dict,0)

    if cfg.clean_up_after:
        openretinawrapper.clean_up(at_processing_stage="setup")    



if __name__ == "__main__":
    run_simulation()