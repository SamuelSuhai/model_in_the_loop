
import os
from omegaconf import DictConfig
from hydra import compose, initialize



def set_env_vars(cfg: DictConfig) -> None:
    """Set environment variables based on the configuration."""
    
    
    if cfg.paths.set_cache_dir_openretina is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")
    
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.set_cache_dir_openretina

    # set repo dir
    repo_directory = cfg.paths.get("repo_directory", None)
    if repo_directory is not None:
        os.environ["MITL_REPO_DIRECTORY"] = repo_directory


    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"



def load_config() -> DictConfig:
    """Load the Hydra configuration for use in notebooks.
    assumes this function sits in model_in_the_loop/utils/hydra_utils.py
    Modifies cfg entries that cannot be run in notebooks"""

    config_path_relative_to_utils = "../config"

    # Initialize Hydra
    with initialize(version_base="1.3", config_path=config_path_relative_to_utils):
        # Compose the configuration
        cfg = compose(config_name="config")
    
    # changes entries to fixed outputs and logs dir
    output_dir = os.path.join(cfg.paths.repo_directory,"model_in_the_loop","outputs")
    cfg.paths.output_dir = output_dir
    cfg.paths.log_dir = os.path.join(output_dir,"logs")
    
    return cfg