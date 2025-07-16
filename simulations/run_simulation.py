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
from loop_components.stimulus_file_copier import copy_stim_files
from loop_components.model_to_stimulus import from_data_to_mei_video
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
    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

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

    # ## The entire iteration
    # # recorder, openretinawrapper, model, stimulator = create_loop_components(cfg)
    openretinawrapper = create_loop_components(cfg)

    # openretinawrapper.setup()
    # raw_neuron_data_dict = openretinawrapper.process_iteration_data()


    # # FOR CLEANING UP
    # openretinawrapper.load_config()
    # openretinawrapper.load_tables()
    # openretinawrapper.clean_up(at_processing_stage="setup")


    # openretinawrapper.add_all_stimuli()
    # sleep(1)    
    # openretinawrapper.set_params_and_userinfo()

    # sleep(1)
    
    # files = ["M1_LR_GCL0_DN","M1_LR_GCL0_Chirp", "M1_LR_GCL0_MB","M1_LR_GCL0_MC18"]
    # for iter,file in enumerate(files):
    #     copy_stim_files(
    #         repo_directory=cfg.paths.repo_directory, # type: ignore
    #         stim_file=file, # type: ignore
    #         new_dir='/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/data/recordings/updated_loop_data/20200226/1/Raw',
    #         iter_nr=0,)
        
    # #
    # raw_neuron_data_dict = openretinawrapper.process_iteration_data()
    
    # openretinawrapper.load_config()
    # openretinawrapper.load_tables()
    # raw_neuron_data_dict = openretinawrapper.extract_data()

    # if raw_neuron_data_dict is not None: 
    #     from_data_to_mei_video(cfg, raw_neuron_data_dict,0)


    # FULL LOOP
    
    openretinawrapper.setup()
    raw_neuron_data_dict = openretinawrapper.process_iteration_data()


    if raw_neuron_data_dict is not None: 
        from_data_to_mei_video(cfg, raw_neuron_data_dict,0)

    openretinawrapper.clean_up(at_processing_stage="setup")    



if __name__ == "__main__":
    run_simulation()