


from typing import Dict,Any,Protocol, List,Tuple
import torch

from openretina.insilico.stimulus_optimization.regularizer import StimulusPostprocessor,_gaussian_1d_kernel
import torch.nn.functional as F
import einops
from openretina.models.core_readout import BaseCoreReadout
from openretina.utils.video_analysis import decompose_kernel

from openretina.utils.nnfabrik_model_loading import Center


import lightning.pytorch
import torch.utils.data as data



from functools import partial

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
from openretina.modules.layers.ensemble import EnsembleModel




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_RATE_MODEL = 30.0  # Hz
STIMULUS_SHAPE = (1, 2, 50, 18, 16)



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

