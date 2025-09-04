


from typing import Dict,Any,Protocol, List,Tuple
import torch
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from openretina.insilico.stimulus_optimization.regularizer import StimulusPostprocessor,_gaussian_1d_kernel
import torch.nn.functional as F
import einops
from openretina.models.core_readout import BaseCoreReadout
from openretina.models.core_readout import load_core_readout_model
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
    TemporalGaussianLowPassFilterProcessor,
)
from openretina.models.core_readout import load_core_readout_model
from openretina.modules.layers.ensemble import EnsembleModel

from .utils import time_it

log = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_RATE_MODEL = 30.0  # Hz
STIMULUS_SHAPE = (1, 2, 50, 18, 16)



class IncreaseObjectiveFourierBasis(AbstractObjective):
    """
    In forward consrtucts a stimulus form the fourier components."""

    def __init__(self, model: BaseCoreReadout, neuron_indices, data_key: str, response_reducer: ResponseReducer):
        self.model = model
        self.neuron_indices = neuron_indices
        self.data_key = data_key
        self.response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        pass


class IncreaseObjectiveSeparable(AbstractObjective):
    """
    ."""

    def __init__(self, model: BaseCoreReadout, neuron_indices, data_key: str, response_reducer: ResponseReducer):
        super().__init__(model, data_key)
        self.model = model
        self._neuron_indices = [neuron_indices] if isinstance(neuron_indices, int) else neuron_indices
        self.data_key = data_key
        self._response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        """ assumes stimulus is shae (1,2 ,time + hight*width), will reshape the laset hight*width entries and mulitpy by time to get full stim"""
        spatial_kernel_ch0 = stimulus[0,0,50:].reshape(18,16)
        spatial_kernel_ch1 = stimulus[0,1,50:].reshape(18,16)
        temporal_kernel_ch0 = stimulus[0,0,:50]
        temporal_kernel_ch1 = stimulus[0,1,:50]

        full_stim_ch0 = torch.einsum("t,hw->thw",temporal_kernel_ch0,spatial_kernel_ch0)
        full_stim_ch1 = torch.einsum("t,hw->thw",temporal_kernel_ch1,spatial_kernel_ch1)
        full_stim = torch.stack([full_stim_ch0,full_stim_ch1],dim=0).unsqueeze(0)
        responses = self.model_forward(full_stim)
 
        # responses.shape = (time, neuron)
        selected_responses = responses[:, self._neuron_indices]
        mean_response = selected_responses.mean(dim=-1)
        # average over time dimension
        single_score = self._response_reducer.forward(mean_response)
        return single_score

class SpatialGaussianLowPassFilterProcessor(StimulusPostprocessor):
    """
    Separable spatial Gaussian LPF: 1D over H, then 1D over W.
    Keeps channels depthwise and leaves time dimension unchanged.
    Also keeps the norm unchanged.
    """
    def __init__(self, sigma: float, kernel_size: int, device: str = "cpu",reflect_pad =False):
        k = _gaussian_1d_kernel(sigma, kernel_size).to(device)
        self.kernel_h = k
        self.kernel_w = k
        self.reflect_pad = reflect_pad

    def process(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        dev = x.device

        kh = einops.repeat(self.kernel_h.to(dev), "s -> c 1 1 s 1", c=C)  # (C,1,1,kh,1)
        kw = einops.repeat(self.kernel_w.to(dev), "s -> c 1 1 1 s", c=C)  # (C,1,1,1,kw)
        
        if self.reflect_pad:
            pad_h = self.kernel_h.numel() // 2
            # pad tuple for 5D (N,C,D,H,W): (W_left, W_right, H_top, H_bottom, D_front, D_back)
            x = F.pad(x, (0, 0, pad_h, pad_h, 0, 0), mode="reflect")
            x = F.conv3d(x, kh, groups=C, padding=0)
            pad_w = self.kernel_w.numel() // 2
            x = F.pad(x, (pad_w, pad_w, 0, 0, 0, 0), mode="reflect")
            x = F.conv3d(x, kw, groups=C, padding=0)
        else:
            # zero-padding “same”
            x = F.conv3d(x, kh, groups=C, padding="same")
            x = F.conv3d(x, kw, groups=C, padding="same")

        return x


def load_pretrained_model(checkpoint_path: str) -> BaseCoreReadout:
    is_gru_model = "gru" in checkpoint_path
    model = load_core_readout_model(checkpoint_path,device=DEVICE, is_gru_model=is_gru_model)
    return model

def load_pretrained_ensemble_model(ckpt_dir: str,seeds: List[str],set_eval=True) -> EnsembleModel:
    models = []
    for seed in seeds:
        ckpt_path = os.path.join(ckpt_dir, f"seed_{seed}.ckpt")
        print(f"Loading model from {ckpt_path}")
        model = load_pretrained_model(ckpt_path)
        if set_eval:
            model.eval()
        models.append(model)

    ensemble_model = EnsembleModel(*models)
    return ensemble_model

def get_seed_and_path(ckpt_dir: str) -> Dict[int,str]:
    seed_and_path = {}
    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt") and "seed_" in file:
            seed_str = file.split("seed_")[-1].split(".ckpt")[0]
            try:
                seed = int(seed_str)
                seed_and_path[seed] = os.path.join(ckpt_dir, file)
            except ValueError:
                log.warning(f"Could not convert {seed_str} to int.")
    return seed_and_path


def generate_optimization_components(stimulus_range_constraints: Dict[str, float],
                                     reducer_axis: int= 0,
                                     reducer_start: int = 10,
                                     reducer_length: int = 10,
                                     temporal_gaussian_kwargs: Dict[str,float | int] | None= None,
                                     spatial_gaussian_kwargs: Dict[str,Any] | None= None,) -> Tuple[List[Any], ResponseReducer]:
    stimulus_postprocessor_list = []
    if temporal_gaussian_kwargs is not None:
        temp_sig = temporal_gaussian_kwargs["sigma"]
        temp_kernel_size = int(temporal_gaussian_kwargs["kernel_size"])
        stimulus_postprocessor_list.append(TemporalGaussianLowPassFilterProcessor(sigma=temp_sig,kernel_size=temp_kernel_size, device=DEVICE))
    
    if spatial_gaussian_kwargs is not None:
        spatial_sigma = spatial_gaussian_kwargs["sigma"]
        spatial_kernel_size = int(spatial_gaussian_kwargs["kernel_size"])
        reflect_pad = spatial_gaussian_kwargs["reflect_pad"]
        stimulus_postprocessor_list.append(SpatialGaussianLowPassFilterProcessor(sigma=spatial_sigma,kernel_size=spatial_kernel_size, reflect_pad=reflect_pad,device=DEVICE))
    
    stimulus_postprocessor_list.append(ChangeNormJointlyClipRangeSeparately(
            min_max_values=[
                (stimulus_range_constraints["x_min_green"], stimulus_range_constraints["x_max_green"]),
                (stimulus_range_constraints["x_min_uv"], stimulus_range_constraints["x_max_uv"]),
            ],
            norm=stimulus_range_constraints["norm"],
    ))


    response_reducer = SliceMeanReducer(axis=reducer_axis, start=reducer_start, length=reducer_length)

    return stimulus_postprocessor_list, response_reducer


def get_model_gaussian_scaled_means(model: BaseCoreReadout, session: str) -> torch.Tensor:
    """Return the model gaussian spatial mean over the core output"""
    session_readout = model.readout[session]
    return session_readout.mask_mean * session_readout.gaussian_mean_scale

#@time_it
def generate_mei(model: BaseCoreReadout | EnsembleModel,
                      new_session_id:str,
                      stimulus_postprocessor_list: List[Any],
                      response_reducer,
                      stimulus_shape: tuple = STIMULUS_SHAPE,
                      neuron_id: List[int] | int = 0, 
                      max_iterations: int = 10,
                      lr =10.0,
                      ) -> torch.Tensor:

    # check if model params are on same device as stimulus
    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE)

    stimulus = torch.randn(stimulus_shape, requires_grad=True, device=DEVICE)

    # scale and clip data once TODO: make this more elegants
    clipper = [processor for processor in stimulus_postprocessor_list if isinstance(processor, ChangeNormJointlyClipRangeSeparately)][0]
    stimulus.data = clipper.process(stimulus.data * 0.1)
    objective = IncreaseObjective(
        model, neuron_indices=neuron_id, data_key=new_session_id, response_reducer=response_reducer
    )

    
    optimization_stopper = OptimizationStopper(max_iterations=max_iterations)
    optimizer_init_fn = partial(torch.optim.SGD, lr=lr)

    optimize_stimulus(
    stimulus,
    optimizer_init_fn,
    objective,
    optimization_stopper,
    stimulus_postprocessor=stimulus_postprocessor_list,
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
    """reconstructs MEI with outer product of spatial and temporal kerenel."""

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
    

def get_model_mei_response(model: BaseCoreReadout | EnsembleModel, 
                           mei: torch.Tensor, 
                           session_id: str, 
                           neuron_id: List[int]) -> np.ndarray:
    """
    Get model response but stay flexible with mei shape (4D or 5D) and neuron_id (int or list of int) and return shape (b,c,t,h,w) or (c,t,h,w)"""
    
    # check if mei correct shape of b,c,t,h,w
    initial_ndim = mei.ndim
    if mei.ndim == 4:
        mei = mei.unsqueeze(0)
    
    if isinstance(neuron_id, int):
        neuron_id = [neuron_id]

    # check if on correct device
    if mei.device != DEVICE:
        mei = mei.to(DEVICE)

    # set model to eval mode
    if model.training:
        model.eval()
        model.freeze()

    with torch.no_grad():
        single_mei_response = model.forward(mei, data_key=session_id)[:, :, neuron_id].detach().cpu().numpy()
    if initial_ndim == 4:
        # only keep batch dim if inputas not batched
        single_mei_response = single_mei_response[0]
    
    return single_mei_response
            
        


def generate_meis_with_n_random_seeds(
    model: BaseCoreReadout | EnsembleModel,
    new_session_id: str, 
    mei_generation_params: Dict[str, Any], 
    random_seeds: List = [42],
    neuron_ids_to_analyze: List[int] = [0], # NOTE: this will optimize each id individually
    set_model_to_eval_mode: bool = False,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Generates a dictionary of MEIs for each neuron id and each random seed."""
    

    if set_model_to_eval_mode:
        model.eval()
        if isinstance(model, EnsembleModel):
            for member in model.members:
                member.eval()
                member.freeze()
        else:
            model.eval()
            model.freeze()
    else:
        if not model.training:
            model.train()

    # generate optimization components
    stimulus_postprocessor_list, response_reducer = generate_optimization_components(
        stimulus_range_constraints=mei_generation_params["stimulus_range_constraints"],
        reducer_axis=0,
        reducer_start=mei_generation_params["reducer_start"],
        reducer_length=mei_generation_params["reducer_length"],
        temporal_gaussian_kwargs=mei_generation_params["temporal_gaussian_kwargs"],
        spatial_gaussian_kwargs=mei_generation_params["spatial_gaussian_kwargs"],
    )
    
    all_meis = {neuron_id: {} for neuron_id in neuron_ids_to_analyze}
    
    for i,seed in enumerate(random_seeds):
        lightning.pytorch.seed_everything(seed)

        for neuron_id in neuron_ids_to_analyze:
            
            # set the seed 
            single_neuron_seed_mei = generate_mei(model=model,
                        new_session_id = new_session_id,
                        stimulus_postprocessor_list = stimulus_postprocessor_list,
                        response_reducer = response_reducer,
                        stimulus_shape= STIMULUS_SHAPE,
                        neuron_id = neuron_id,
                        max_iterations=mei_generation_params["max_iteration"],
                        lr=mei_generation_params["lr"],)
                        
            all_meis[neuron_id][seed] = single_neuron_seed_mei
    return all_meis


def single_training_loop(dataloaders,cfg,data_info,log,load_model_path=None,seed= None):
    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if seed is not None:
        lightning.pytorch.seed_everything(seed)
    elif cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ## Model init
    if load_model_path is None:
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


    # Get the checkpoint callback from your callbacks list
    checkpoint_callback = next(c for c in callbacks if isinstance(c, lightning.pytorch.callbacks.ModelCheckpoint))
    if checkpoint_callback is None:
        raise RuntimeError("No ModelCheckpoint callback found. Ensure save_top_k >= 1 and monitor are set.")


    # After training completes:
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("best_model_path is empty. Check ModelCheckpoint(monitor=..., save_top_k>=1).")


    # load best model for testing
    log.info(f"Loading best model from {best_model_path} for testing and inference.")

    #DEBUG:MAKE SURE SESSION DATA INFO AND NNEURONS DICT IS THERE 
    model = load_core_readout_model(best_model_path, DEVICE, is_gru_model=is_gru_model)

    log.info("All dataloaders...")
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    dataloader_mapping = {f"DataLoader {i}": x[0] for i, x in enumerate(short_cyclers)}
    log.info(f"Dataloader mapping: {dataloader_mapping}")
    # test with model passed
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path=None)


    ### Testing
    log.info("Testing individual neurons!")
    neuron_testset_correl = get_single_neuron_test_correlations(dataloaders, model)
    log.info(f"Test set neuron correlations statistics (mean,std,min,max): {[func(list(neuron_testset_correl.values())) for func in [np.mean, np.std, np.min, np.max]]}")
    
    return model,neuron_testset_correl, best_model_path


#@time_it
def train_model_online(cfg: DictConfig,
                       neuron_data_dict:Dict[str,ResponsesTrainTestSplit],
                       movies_dict: MoviesTrainTestSplit) -> \
                       Tuple[EnsembleModel | BaseCoreReadout, Dict[int,float], Dict[int,str] | str]:
    
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

  
    if cfg.check_stimuli_responses_match:
        for _, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict)

    dataloaders = hydra.utils.instantiate( # dict[str, dict[str, DataLoader]]
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )

    data_info = compute_data_info(neuron_data_dict, movies_dict)

    is_ensemble_model =cfg.get("is_ensemble_model", False)
    log.info(f"Is ensemble model: {is_ensemble_model}")
    if is_ensemble_model:
        assert os.path.isdir(cfg.paths.load_model_path), "For ensemble model, load_model_path should be a directory containing seed_x.ckpt files."
        seed_and_path = get_seed_and_path(cfg.paths.load_model_path)
        log.info(f"Found {len(seed_and_path)} seeds in {cfg.paths.load_model_path}: {list(seed_and_path.keys())}")

        all_models = []
        best_model_path = {}
        for seed, path in seed_and_path.items():
            log.info(f"Loading model for seed {seed} from {path}")

            # perform single training loop for each model
            model,_,best_path = single_training_loop(dataloaders, 
                                                        cfg, data_info, 
                                                        log, 
                                                        load_model_path=path,
                                                        seed=seed)
            all_models.append(model)
            best_model_path[seed] = best_path

        # bind to ensemble and test
        model = EnsembleModel(*all_models)
        neuron_testset_correl = get_single_neuron_test_correlations(dataloaders, model)
    else:

        model,neuron_testset_correl, best_model_path = single_training_loop(dataloaders, cfg, data_info, log)

    return model,neuron_testset_correl, best_model_path



def get_single_neuron_test_correlations(dataloaders , model: BaseCoreReadout | EnsembleModel) ->  Dict[int, float]:
    """Calculate the correlation between model predictions and targets for each neuron in each session in the test session."""

    neuron_correlations = {}
    if len(dataloaders["test"]) != 1:
        raise ValueError(f"Expected only one session in the test set for online training. but found {len(dataloaders['test'])} sessions.")


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
            
        neuron_correlations = session_correlations

    return neuron_correlations
        





def load_stimuli(or_config: DictConfig):

    movies_dict: MoviesTrainTestSplit = hydra.utils.call(or_config.data_io.stimuli) 
    return movies_dict
   
