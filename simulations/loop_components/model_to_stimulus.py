


from typing import Dict,Any,Protocol, List
import torch
import logging
import os
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from openretina.models.core_readout import BaseCoreReadout
from openretina.models.core_readout import load_core_readout_model
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit

import lightning.pytorch
import torch.utils.data as data

from openretina.data_io.base import compute_data_info
from openretina.data_io.cyclers import LongCycler, ShortCycler


from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.insilico.stimulus_optimization.objective import (
    AbstractObjective,
    IncreaseObjective,
    ResponseReducer,
    SliceMeanReducer,
)
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
)
from openretina.models.core_readout import load_core_readout_model
from openretina.utils.nnfabrik_model_loading import load_ensemble_model_from_remote
from openretina.utils.plotting import play_stimulus, plot_stimulus_composition

from .utils import time_it

log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: move to configs
stimulus_shape = (1, 2, 50, 18, 16)

STIMULUS_RANGE_CONSTRAINTS = {
    "norm": 5.0,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}

stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately(
    min_max_values=[
        (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
        (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
    ],
    norm=STIMULUS_RANGE_CONSTRAINTS["norm"],
)

response_reducer = SliceMeanReducer(axis=0, start=10, length=10)


def load_pretrained_model(checkpoint_path: str) -> BaseCoreReadout:
    is_gru_model = "gru" in checkpoint_path
    model = load_core_readout_model(checkpoint_path,device=DEVICE, is_gru_model=is_gru_model)
    return model

@time_it
def generate_stimulus(model: BaseCoreReadout,new_sessoin_id:str,neuron_id: List[int] | int = 0) -> torch.Tensor:

    # check if model params are on same device as stimulus
    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE)

    data_info = model.hparams["data_info"] # check if there is original stimulus stats in there mean sd???
    stimulus = torch.randn(stimulus_shape, requires_grad=True, device=DEVICE)
    stimulus.data = stimulus.data * 0.1

    objective = IncreaseObjective(
        model, neuron_indices=neuron_id, data_key=new_sessoin_id, response_reducer=response_reducer
    )
    optimization_stopper = OptimizationStopper(max_iterations=10)
    optimizer_init_fn = partial(torch.optim.SGD, lr=10.0)

    optimize_stimulus(
    stimulus,
    optimizer_init_fn,
    objective,
    optimization_stopper,
    stimulus_postprocessor=stimulus_postprocessor,
    stimulus_regularization_loss=None,
    )


    return stimulus[0] # return first batch

@time_it    
def create_avi_from_tensor(stimulus: torch.Tensor, filename: str, fps: int = 50, original_stimulus_stats: Dict[str,float] | Any = None) -> None:
    """Crates an AVI file from toch.Tensor stimulus and saves it at `filename`"""
    import numpy as np
    import cv2
    assert len(stimulus.shape) == 4, "Stimulus tensor must be of shape (C,T,H,W)"
    stimulus = stimulus.detach().cpu()

    # put back to same space as original stimulus
    if original_stimulus_stats is not None:
        stimulus = stimulus * original_stimulus_stats["std"] + original_stimulus_stats["mean"]
    else:
        # simply map to [0,255] 
        min_val = torch.min(stimulus)
        max_val = torch.max(stimulus)
        stimulus = 255 * (stimulus - min_val) / (max_val - min_val)
   
    # open cv2 expects the stimulus in (T, H, W, C) format
    stimulus_np = stimulus.numpy()
    stimulus_np = np.transpose(stimulus_np, (1, 2, 3, 0)).astype(np.uint8)

    frames, height, width, channels = stimulus_np.shape

    rgb_frames = np.zeros((frames, height, width, 3), dtype=np.uint8)
    # Map channel 0 to green and channel 1 to blue (UV)
    rgb_frames[:, :, :, 1] = stimulus_np[:, :, :, 0]  # Green channel
    rgb_frames[:, :, :, 2] = stimulus_np[:, :, :, 1]  # Blue channel (for UV)
    stimulus_np = rgb_frames
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type : ignore # Use XVID codec 
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)
    
    # Write frames
    for i in range(frames):
        frame = stimulus_np[i]
        # OpenCV uses BGR format
        if frame.shape[-1] == 3:  # RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore
        out.write(frame)  # type: ignore
     
    # Release resources
    out.release()

    log.info(f"AVI file saved as {filename}")

@time_it
def train_model_online(cfg: DictConfig,
                       neuron_data_dict:Dict[str,ResponsesTrainTestSplit],
                       movies_dict:Dict[str,MoviesTrainTestSplit] | MoviesTrainTestSplit) -> BaseCoreReadout:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

  
    if cfg.check_stimuli_responses_match:
        for session, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict[session]) # type: ignore

    dataloaders = hydra.utils.instantiate( # dict[str, dict[str, DataLoader]]
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )

    data_info = compute_data_info(neuron_data_dict, movies_dict)

    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    # ## Assign missing n_neurons_dict to model
    # cfg.model.n_neurons_dict = data_info["n_neurons_dict"]
    # log.info(f"Instantiating model <{cfg.model._target_}>")
    # model = hydra.utils.instantiate(cfg.model, data_info=data_info)

    ## Model init
    load_model_path = cfg.paths.get("load_model_path")
    # load_model_path = "/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/outputs/2025-05-30/17-34-53/checkpoints/epoch=08_val_correlation=0.333.ckpt"
    log.info(f"Loading model from <{load_model_path}>")
    is_gru_model = "gru" in cfg.model._target_.lower() if hasattr(cfg.model, "_target_") else False
    model = load_core_readout_model(load_model_path, DEVICE, is_gru_model=is_gru_model)

    
    # add new readouts and modify stored data in model
    model.readout.add_sessions(data_info["n_neurons_dict"])  # type: ignore
    model.update_model_data_info(data_info)

    if cfg.get("only_train_readout") is True:
        log.info("Only training readout, core model parameters will be frozen.")
        model.core.requires_grad_(False)

    ### Logging
    log.info("Instantiating loggers...")
    logger_array = []
    for _, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params)
        logger_array.append(logger)

    ### Callbacks
    log.info("Instantiating callbacks...")
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]

    ### Trainer init
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ### Testing
    log.info("Starting testing!")
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    dataloader_mapping = {f"DataLoader {i}": x[0] for i, x in enumerate(short_cyclers)}
    log.info(f"Dataloader mapping: {dataloader_mapping}")
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path="best")

    return model

@time_it
def preprocess_for_openretina(raw_neuron_data_dict:Dict[str,Dict[str,Any]]) -> Dict[str,ResponsesTrainTestSplit]:
    filt_neuron_data =  filter_responses(raw_neuron_data_dict)
    neuron_data_dict =  make_final_responses(filt_neuron_data) 
    return neuron_data_dict

@time_it
def save_new_stimulus_position(new_session_id: str ,full_path: str,raw_neuron_data_dict:Dict[str,Dict[str,Any]],neuron_id: int = 0) -> None:
    """Retrieves peak RF position and saves that info in a yaml file in the stimuli directory."""
    
    if isinstance(neuron_id,list):
        raise NotImplementedError("Only able to get peak from one neuron. not sure how to do this for many neurons yet.") 

    x = raw_neuron_data_dict[new_session_id].get("rf_peak_x_um",0)[neuron_id] 
    y = raw_neuron_data_dict[new_session_id].get("rf_peak_y_um",0)[neuron_id]

    # convert from array values to type yaml perser can handle
    if hasattr(x, "item"):
        x = float(x.item())
    if hasattr(y, "item"):
        y = float(y.item())

    stim_metadata = {"position":{"x": x, "y": y}}
    
    with open(full_path, "w") as f:
        yaml.dump(stim_metadata, f, default_flow_style=False)
    

def from_data_to_mei_video(cfg: DictConfig, raw_neuron_data_dict:Dict[str,Dict[str,Any]],neuron_ids = 0):

    openretina_cfg = cfg.model_configs
    movies_dict: MoviesTrainTestSplit = hydra.utils.call(openretina_cfg.data_io.stimuli) # are stmulus stats here???? ST WHAT PART IS CACHE_DIR NEEEDED???!!!

    neuron_data_dict = preprocess_for_openretina(raw_neuron_data_dict)
    
    # load and refine model
    model = train_model_online(openretina_cfg,neuron_data_dict,movies_dict)
    new_session_id = list(raw_neuron_data_dict.keys())[0]

    new_stimulus = generate_stimulus(model,new_session_id,neuron_id=neuron_ids)

    save_new_stimulus_position(new_session_id,
                               full_path = os.path.join(cfg.paths.repo_directory, "data/stimuli",f"mei_{new_session_id}.yaml"),
                               raw_neuron_data_dict=raw_neuron_data_dict,
                               neuron_id=neuron_ids)

    create_avi_from_tensor(
        new_stimulus,
        filename=os.path.join(cfg.paths.repo_directory, "data/stimuli",f"mei_{new_session_id}.avi"),
        original_stimulus_stats=None,
    )




@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main (cfg: DictConfig):

    import pickle
    with open("/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/data/model_input/20200226_GCL0_iter0.pickle", "rb") as f:
        raw_neuron_data_dict = pickle.load(f)

    from_data_to_mei_video(cfg, raw_neuron_data_dict,0)

if __name__ == "__main__":
    main()

    

