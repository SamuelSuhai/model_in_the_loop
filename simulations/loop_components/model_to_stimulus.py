


from typing import Dict,Any,Protocol, List,Tuple
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
from openretina.utils.video_analysis import decompose_kernel

from openretina.utils.nnfabrik_model_loading import Center


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

from .utils import time_it

log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: move to configs
STIMULUS_SHAPE = (1, 2, 50, 18, 16)

STIMULUS_RANGE_CONSTRAINTS = {
    "norm": 5.0,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}




def load_pretrained_model(checkpoint_path: str) -> BaseCoreReadout:
    is_gru_model = "gru" in checkpoint_path
    model = load_core_readout_model(checkpoint_path,device=DEVICE, is_gru_model=is_gru_model)
    return model



def generate_optimization_components(stimulus_range_constraints: Dict[str, float] = STIMULUS_RANGE_CONSTRAINTS,
                                     reducer_axis: int= 0,
                                     reducer_start: int = 10,
                                     reducer_length: int = 10):
    stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately(
    min_max_values=[
        (stimulus_range_constraints["x_min_green"], stimulus_range_constraints["x_max_green"]),
        (stimulus_range_constraints["x_min_uv"], stimulus_range_constraints["x_max_uv"]),
    ],
    norm=stimulus_range_constraints["norm"],
    )

    response_reducer = SliceMeanReducer(axis=reducer_axis, start=reducer_start, length=reducer_length)

    return stimulus_postprocessor, response_reducer


def get_model_gaussian_scaled_means(model: BaseCoreReadout, session: str) -> torch.Tensor:
    """Return the model gaussian spatial mean over the core output"""
    session_readout = model.readout[session]
    return session_readout.mask_mean * session_readout.gaussian_mean_scale

#@time_it
def generate_mei(model: BaseCoreReadout,
                      new_session_id:str,
                      stimulus_postprocessor,
                      response_reducer,
                      stimulus_shape: tuple = STIMULUS_SHAPE,
                      neuron_id: List[int] | int = 0, 
                      max_iterations: int = 10,
                      ) -> torch.Tensor:

    # check if model params are on same device as stimulus
    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE)

    data_info = model.hparams["data_info"] # check if there is original stimulus stats in there mean sd???
    stimulus = torch.randn(stimulus_shape, requires_grad=True, device=DEVICE)
    stimulus.data = stimulus.data * 0.1

    objective = IncreaseObjective(
        model, neuron_indices=neuron_id, data_key=new_session_id, response_reducer=response_reducer
    )
    optimization_stopper = OptimizationStopper(max_iterations=max_iterations)
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


def decompose_mei(stimulus: np.ndarray, frame_rate_model: float = 30.0) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    assert stimulus.ndim == 4, "Stimulus should be a 4D array (num_color_channels, time_steps, dim_y, dim_x)"
 
    num_color_channels, time_steps, dim_y, dim_x = stimulus.shape

    stimulus_time = np.linspace(0, time_steps / frame_rate_model, time_steps)


    temporal_kernels = []
    spatial_kernels = []
    for color_idx in range(num_color_channels):
        temporal, spatial, singular_values = decompose_kernel(stimulus[color_idx])
        temporal_kernels.append(temporal)
        spatial_kernels.append(spatial)

    return temporal_kernels, spatial_kernels, stimulus_time

def reconstruct_spatiotemporal_kernel(
        temporal_kernel: np.ndarray,
        spatial_kernel: np.ndarray,) -> np.ndarray:
    

    recon = temporal_kernel[:, None, None] * spatial_kernel[None, :, :]
    assert recon.shape == STIMULUS_SHAPE[2:]

    return recon

def reconstruct_mei_from_decomposed(
        temporal_kernels: List[np.ndarray],
        spatial_kernels: List[np.ndarray],
    ) -> np.ndarray:
    assert len(temporal_kernels) == len(spatial_kernels), "Number of temporal and spatial kernels must match"
    num_color_channels = len(temporal_kernels)
    dim_y, dim_x = spatial_kernels[0].shape
    time_steps = temporal_kernels[0].shape[0]
    
    
    reconstructed_mei = np.zeros((num_color_channels, time_steps, dim_y, dim_x))

    for color_idx in range(num_color_channels):
        reconstructed_mei[color_idx] = reconstruct_spatiotemporal_kernel(
            temporal_kernels[color_idx], spatial_kernels[color_idx]
        )
    assert reconstructed_mei.shape == STIMULUS_SHAPE[1:]

    return reconstructed_mei
    

def get_model_mei_response(model: BaseCoreReadout, mei: torch.Tensor, session_id: str, neuron_id: int) -> np.ndarray:
    
    # check if mei correct shape of b,c,t,h,w
    if mei.ndim == 4:
        mei = mei.unsqueeze(0)
    
    # check if on correct device
    if mei.device != DEVICE:
        mei = mei.to(DEVICE)

    # set model to eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        single_mei_response = model.forward(mei, data_key=session_id)[0, :, neuron_id].detach().cpu().numpy()
    
    return single_mei_response
            
        


def generate_meis_with_n_random_seeds(
    model: BaseCoreReadout,
    new_session_id: str,
    random_seeds: List = [42],
    neuron_ids_to_analyze: List[int] = [0], # NOTE: this will optimize each id individually 
    set_model_to_eval_mode: bool = False,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Generates a dictionary of MEIs for each neuron id and each random seed."""
    
    if set_model_to_eval_mode:
        model.eval()
    else:
        if not model.training:
            model.train()

    # generate optimization components
    stimulus_postprocessor, response_reducer = generate_optimization_components(
        stimulus_range_constraints=STIMULUS_RANGE_CONSTRAINTS,
        reducer_axis=0,
        reducer_start=10,
        reducer_length=10,
    )
    
    all_meis = {neuron_id: {} for neuron_id in neuron_ids_to_analyze}
    
    for i,seed in enumerate(random_seeds):
        lightning.pytorch.seed_everything(seed)

        for neuron_id in neuron_ids_to_analyze:
            
            # set the seed 
            single_neuron_seed_mei = generate_mei(model=model,
                        new_session_id = new_session_id,
                        stimulus_postprocessor = stimulus_postprocessor,
                        response_reducer = response_reducer,
                        stimulus_shape= STIMULUS_SHAPE,
                        neuron_id = neuron_id)
                        
            all_meis[neuron_id][seed] = single_neuron_seed_mei
    return all_meis



#@time_it
def train_model_online(cfg: DictConfig,
                       neuron_data_dict:Dict[str,ResponsesTrainTestSplit],
                       movies_dict:Dict[str,MoviesTrainTestSplit] | MoviesTrainTestSplit) -> Tuple[BaseCoreReadout,Dict[int, float]]:
    
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
    neuron_testset_correl_dict = get_single_neuron_test_correlations(dataloaders, model)
    assert len(neuron_testset_correl_dict) == 1, "Expected only one session in the test set for online training."
    session_name,neuron_testset_correl = neuron_testset_correl_dict.popitem()  # get first session correlations
    log.info(f"Test set neuron correlations statistics (mean,std,min,max) for session {session_name}: {[func(list(neuron_testset_correl.values())) for func in [np.mean, np.std, np.min, np.max]]}")

    return model,neuron_testset_correl



def get_single_neuron_test_correlations(dataloaders , model: BaseCoreReadout) -> Dict[str, Dict[int, float]]:
    """Calculate the correlation between model predictions and targets for each neuron in each session in the test session."""

    neuron_correlations = {}


    for session_id, session_dataloader in dataloaders["test"].items():
        all_preds, all_targets = [], []
        
        # Run model on all test batches
        with torch.no_grad():
            model.eval()
            model.to(DEVICE)
            for batch in session_dataloader:
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                predictions = model(inputs, data_key=session_id)
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate batch results and compute correlations
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Fast vectorized correlation calculation
        session_correlations = {}
        num_neurons = all_targets.shape[-1]

        conv_eats_n_frames = all_targets.shape[1] - all_preds.shape[1]
        for i in range(num_neurons):
            pred = all_preds[..., i].flatten()
            target = all_targets[:,conv_eats_n_frames:, i].flatten()
            corr = np.corrcoef(pred, target)[0, 1] if np.var(pred) > 0 and np.var(target) > 0 else 0
            session_correlations[i] = corr
            
        neuron_correlations[session_id] = session_correlations

    return neuron_correlations
        

#@time_it
def preprocess_for_openretina(raw_neuron_data_dict:Dict[str,Dict[str,Any]],model_condigs) -> Dict[str,ResponsesTrainTestSplit]:
    filt_neuron_data =  filter_responses(raw_neuron_data_dict, **model_condigs.quality_checks)
    neuron_data_dict =  make_final_responses(filt_neuron_data,response_type="natural") 
    return neuron_data_dict

#@time_it
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

def load_stimuli(or_config: DictConfig):

    # are stmulus stats here???? ST WHAT PART IS CACHE_DIR NEEEDED???!!!
    movies_dict: MoviesTrainTestSplit = hydra.utils.call(or_config.data_io.stimuli) 
    return movies_dict
   


def from_data_to_mei_video(cfg: DictConfig, raw_neuron_data_dict:Dict[str,Dict[str,Any]],neuron_ids = 0):

    movies_dict = load_stimuli(cfg.model_configs)

    neuron_data_dict = preprocess_for_openretina(raw_neuron_data_dict,cfg.model_configs)
    
    # load and refine model
    model = train_model_online(cfg.model_configs,neuron_data_dict,movies_dict)
    new_session_id = list(raw_neuron_data_dict.keys())[0]


    # generate optimization components
    stimulus_postprocessor, response_reducer = generate_optimization_components(
        stimulus_range_constraints=STIMULUS_RANGE_CONSTRAINTS,
        reducer_axis=0,
        reducer_start=10,
        reducer_length=10,
    )
    new_stimulus = generate_mei(model=model,
                      new_session_id = new_session_id,
                      stimulus_postprocessor = stimulus_postprocessor,
                      response_reducer = response_reducer,
                      stimulus_shape= STIMULUS_SHAPE,
                      neuron_id = neuron_ids, 
                      )
    save_new_stimulus_position(new_session_id,
                               full_path = os.path.join(cfg.paths.repo_directory, "data/stimuli",f"mei_{new_session_id}.yaml"),
                               raw_neuron_data_dict=raw_neuron_data_dict,
                               neuron_id=neuron_ids)





@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main (cfg: DictConfig):

    import pickle
    with open("/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/data/model_input/20200226_GCL0_iter0.pickle", "rb") as f:
        raw_neuron_data_dict = pickle.load(f)

    from_data_to_mei_video(cfg, raw_neuron_data_dict,0)

if __name__ == "__main__":
    main()

    

