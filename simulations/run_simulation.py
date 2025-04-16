import hydra
from hydra.utils import get_original_cwd
import os
from omegaconf import DictConfig, OmegaConf

# TODO: refactor by importing loop components based on config
from simulations.loop_components.recorders.file_copier import FileCopier





@hydra.main(version_base=None, config_path="../config/", config_name="config")
def run_simulation(cfg: DictConfig):

    # retrieve config dicts 
    preprocessors_cfg: dict = cfg.preprocessors
    models_cfg: dict = cfg.models
    recorders_cfg: dict = cfg.recorders
    stimulators_cfg: dict = cfg.stimulators
    
    # create preprocessor    
    preprocessor = FileCopier(
        source_dir= preprocessors_cfg.source_dir,
        source_exp_dir=preprocessors_cfg.source_exp_dir,
        source_smp_file=preprocessors_cfg.source_smp_file,
        source_smh_file=preprocessors_cfg.source_smh_file,
        source_stim_file=preprocessors_cfg.source_stim_file,
        ini_file=preprocessors_cfg.ini_file,
        target_dir=preprocessors_cfg.target_dir
    )

    for exp in range(1):
        preprocessor.record()


if __name__ == "__main__":
    main()