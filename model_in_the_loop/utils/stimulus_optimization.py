


from typing import Dict,Any,Protocol, List,Tuple,Callable
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
    RangeRegularizationLoss,
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


class DecreaseObjective(AbstractObjective):
    def __init__(self, model, neuron_indices: list[int] | int, data_key: str | None, response_reducer: ResponseReducer):
        super().__init__(model, data_key)
        self._neuron_indices = [neuron_indices] if isinstance(neuron_indices, int) else neuron_indices
        self._response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self.model_forward(stimulus)
        # responses.shape = (time, neuron)
        selected_responses = responses[:, self._neuron_indices]
        mean_response = selected_responses.mean(dim=-1)
        # average over time dimension AND FLIP TIME TO MAKE SUPPRESSIVE
        single_score =   - self._response_reducer.forward(mean_response)
        return single_score


class DiverselyIncreaseObjective(AbstractObjective):
    """
    Implements the diverse exciting stimuli objectife funciton by Ding 2025. 

    It takes: 
    - a list of stimuli (should be the MEI + some noise added)
    - the response to the MEI
    - d_weight: the hyperparameter weighing the diverity term
    - frac_max_response: at what activity, relative to MEI actiity, 
        the neural response should be increased again
    - rf_mask: spatio temporal rf mask

    
    """
    def __init__(self, 
                 model, 
                 neuron_indices: list[int] | int, 
                 data_key: str | None, 
                 response_reducer: ResponseReducer,
                 d_weight: float,
                 frac_max_response: float,
                 response_mei: float,
                 rf_mask: torch.Tensor,
                 ):
    
        super().__init__(model, data_key)
        self._neuron_indices = [neuron_indices] if isinstance(neuron_indices, int) else neuron_indices
        self._response_reducer = response_reducer
        self.d_weight = d_weight
        self.frac_max_response = frac_max_response
        self.response_mei = response_mei
        self.rf_mask = rf_mask # (1,channels, time, width, height)

    def forward(self, deis:torch.Tensor) -> torch.Tensor:
        """
        For how, individual DEIs are seen as one example from the batch of stimuli
        Each DEI is a stimulus of shape (channels,time,width,height),"""

        # ??? set stim parts outside rf to zero
        deis = deis * self.rf_mask # (batch,channels,time,width,height)
     
        ### part of the objective keeping responses high
        responses = self.model_forward(deis) # responses.shape = (dei,time, neuron)

        selected_responses = responses[:,:, self._neuron_indices]
        mean_response = selected_responses.mean(dim=-1)
        # average over time dimension
        single_score = self._response_reducer.forward(mean_response)
        # assert torch.all(single_score <= self.response_mei), "This should not happen... \
        #     maybe MEI has differet response reducer than DEIs now?"
        
        # only if the rsponse is below the threshold, we want to increase it
        diff_from_contribution_threshold = self.frac_max_response - single_score / self.response_mei
        increase_part = torch.sum(diff_from_contribution_threshold[diff_from_contribution_threshold > 0]) / diff_from_contribution_threshold.shape[0]

  

        ### diversity part: the negative of the minimum pariwise euclidian distance
        deis_flat = deis.view(deis.shape[0], -1)
        pw_dists = torch.cdist(deis_flat,deis_flat, p=2)
        
        # Create mask for non-diagonal elements
        non_diag_mask = ~torch.eye(pw_dists.shape[0], dtype=bool, device=pw_dists.device)
        non_diag_dists = pw_dists[non_diag_mask]
        min_pw_dist = torch.min(non_diag_dists)

        toal_loss = increase_part - self.d_weight * min_pw_dist 
        total_score = - toal_loss
        return total_score




def generate_optimization_components(stimulus_range_constraints: Dict[str, float],
                                     reducer_axis: int= 0,
                                     reducer_start: int = 10,
                                     reducer_length: int = 10,
                                     temporal_gaussian_kwargs: Dict[str,float | int] | None= None,
                                     spatial_gaussian_kwargs: Dict[str,Any] | None= None,
                                     range_regularization_kwargs: Dict[str,Any] | None= None,) -> Tuple[List[Any], 
                                                                                                        ResponseReducer,
                                                                                                        RangeRegularizationLoss | None]:
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
    

    min_max_values=[
                (stimulus_range_constraints["x_min_green"], stimulus_range_constraints["x_max_green"]),
                (stimulus_range_constraints["x_min_uv"], stimulus_range_constraints["x_max_uv"]),
            ]

    stimulus_postprocessor_list.append(ChangeNormJointlyClipRangeSeparately(
            min_max_values=min_max_values,
            norm=stimulus_range_constraints["norm"],
    ))


    response_reducer = SliceMeanReducer(axis=reducer_axis, start=reducer_start, length=reducer_length)

    if range_regularization_kwargs is not None:
        factor = range_regularization_kwargs.get("factor", 0.1)
        stimulus_regularizing_loss = RangeRegularizationLoss(
            min_max_values=min_max_values,
            max_norm=stimulus_range_constraints["norm"],
            factor=factor,
        )
    else:
        stimulus_regularizing_loss = None


    return stimulus_postprocessor_list, response_reducer,stimulus_regularizing_loss


def get_model_gaussian_scaled_means(model: BaseCoreReadout, session: str) -> torch.Tensor:
    """Return the model gaussian spatial mean over the core output"""
    session_readout = model.readout[session]
    return session_readout.mask_mean * session_readout.gaussian_mean_scale



def generate_deis(model: BaseCoreReadout | EnsembleModel,
                  mei: torch.Tensor, 
                  n_deis: int,
                  neuron_id: int,
                  session_id: str,
                  opt_stim_generation_params: Dict[str,Any]) -> torch.Tensor:
    """
    Generates deis
    """
    norm = opt_stim_generation_params["stimulus_range_constraints"]["norm"]
    
    detached_stimulus = mei.detach().clone()
    if detached_stimulus.ndim == 5:
        detached_stimulus = detached_stimulus.squeeze(0) # shape (C.T,H,W)
    c,t,h,w = detached_stimulus.shape
    # get rf mask: first spatial H,W then temporal then combine
    rf_mask_space = torch.any(torch.abs(detached_stimulus - torch.mean(detached_stimulus)) > 1.5 * torch.std(detached_stimulus), dim=(0,1)) # shape (H,W)
    rf_mask_time = torch.any(torch.abs(detached_stimulus - torch.mean(detached_stimulus)) > 1.5 * torch.std(detached_stimulus), dim=(0,2,3)) # shape (T,)
    rf_mask_thw = rf_mask_time[:,None,None] * rf_mask_space[None,:,:] # shape (T,H,W)
    rf_mask = rf_mask_thw[None].repeat(c, 1, 1, 1).unsqueeze(0) # shape (1,C,T,H,W)
    rf_mask = rf_mask.to(DEVICE)

    # apply mask to stimulus    
    detached_stimulus = detached_stimulus * rf_mask.squeeze(0)

    # 1) init deis
    deis = detached_stimulus.repeat(n_deis, 1, 1, 1, 1)
    noise = torch.randn_like(deis) * 0.1
    deis = deis + noise
    
    # apply mask and normalize
    for i in range(deis.shape[0]):
        d = deis[i] * rf_mask  # rf_mask: (C,T,H,W) or (1,C,T,H,W) with broadcasting
        n = torch.norm(d)
        if n > 0:
            d = d / n * norm
        deis[i] = d

    deis.requires_grad_(True)

    # 2) generate optimization components
    stimulus_postprocessor_list, response_reducer,stimulus_regularizing_loss = generate_optimization_components(
        stimulus_range_constraints=opt_stim_generation_params["stimulus_range_constraints"],
        reducer_axis=1,
        reducer_start=opt_stim_generation_params["reducer_start"],
        reducer_length=opt_stim_generation_params["reducer_length"],
        temporal_gaussian_kwargs=opt_stim_generation_params["temporal_gaussian_kwargs"],
        spatial_gaussian_kwargs=opt_stim_generation_params["spatial_gaussian_kwargs"],
        range_regularization_kwargs=opt_stim_generation_params["range_regularization_kwargs"],
    )
    
    # check if model params are on same device as stimulus
    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE)

    # scale and clip data once
    clipper = [processor for processor in stimulus_postprocessor_list if isinstance(processor, ChangeNormJointlyClipRangeSeparately)][0]
    for i in range(deis.shape[0]):
        deis[i].unsqueeze(0).data = clipper.process(deis[i].unsqueeze(0).data)
    
    # get model response to mei 
    response_mei = get_model_mei_response(model, detached_stimulus, session_id, [neuron_id])
    single_neuron_response = response_mei[...,0]
    # get mean ofer response reducer time
    mean_single_neuron_mei_response = single_neuron_response[
        opt_stim_generation_params["reducer_start"] : opt_stim_generation_params["reducer_start"] + opt_stim_generation_params["reducer_length"]
    ].mean()

    objective = DiverselyIncreaseObjective(
        model, 
        neuron_indices=neuron_id, 
        data_key=session_id, 
        response_reducer=response_reducer,
        d_weight = 1,
        frac_max_response = opt_stim_generation_params.get("frac_max_response",0.8),
        response_mei = mean_single_neuron_mei_response,
        rf_mask =rf_mask.detach()
    )
    optimization_stopper = OptimizationStopper(max_iterations=opt_stim_generation_params["max_iteration"])
    optimizer_init_fn = partial(torch.optim.SGD, lr=opt_stim_generation_params["lr"])

    optimize_stimulus(
    deis,
    optimizer_init_fn,
    objective,
    optimization_stopper,
    stimulus_postprocessor=stimulus_postprocessor_list,
    stimulus_regularization_loss=stimulus_regularizing_loss,
    )
    
    return deis.detach()
    

def generate_opt_stim(model: BaseCoreReadout | EnsembleModel,
                      new_session_id:str,
                      stimulus_postprocessor_list: List[Any],
                      response_reducer,
                      stimulus_regularizing_loss: RangeRegularizationLoss | None = None,
                      stimulus_shape: tuple = STIMULUS_SHAPE,
                      neuron_id: List[int] | int = 0, 
                      max_iterations: int = 10,
                      lr =10.0,
                      objective_name= "increase"
                      ) -> torch.Tensor:
    """
    1. generates objectives
    2. initializes stimulus, optimization stopper and optimizer
    3. optimizes stimulus
    """

    if objective_name == "increase":
        objective_class = IncreaseObjective
    elif objective_name == "decrease":
        objective_class = DecreaseObjective
    else:
        raise ValueError(f"Objective name {objective_name} not recognized.")

    # check if model params are on same device as stimulus
    if next(model.parameters()).device != DEVICE:
        model = model.to(DEVICE)

    stimulus = torch.randn(stimulus_shape, requires_grad=True, device=DEVICE)

    # scale and clip data once TODO: make this more elegants
    clipper = [processor for processor in stimulus_postprocessor_list if isinstance(processor, ChangeNormJointlyClipRangeSeparately)][0]
    stimulus.data = clipper.process(stimulus.data * 0.1)
    objective = objective_class(
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
    stimulus_regularization_loss=stimulus_regularizing_loss,
    )

    return stimulus[0] # return first batch

def center_member_or_ensemble_readouts(model: BaseCoreReadout | EnsembleModel, new_session_id: str) -> List[torch.Tensor]:
            ## center the readouts            
    scaled_means_before_centering = []
    if isinstance(model, EnsembleModel):
        for member in model.members:
            scaled_means_before_centering.append(get_model_gaussian_scaled_means(member,session= new_session_id)) # type: ignore
    elif isinstance(model, BaseCoreReadout):
        scaled_means_before_centering.append(get_model_gaussian_scaled_means(model,session= new_session_id)) # type: ignore
    else:
        raise ValueError("Model is neither ensemble nor BaseCoreReadout. Cannot center readouts.")
    # apply centering
    center = Center(target_mean = 0.0)
    center(model)

    return scaled_means_before_centering


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
        turn_to_tensor: bool = True
    ) -> np.ndarray | torch.Tensor:
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

    if turn_to_tensor:
        reconstructed_mei = torch.tensor(reconstructed_mei, dtype=torch.float32,device=DEVICE)

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
            
        


def generate_opt_stim_for_neuron_list(
    model: BaseCoreReadout | EnsembleModel,
    new_session_id: str, 
    opt_stim_generation_params: Dict[str, Any], 
    random_seeds: List[int] | None = None,
    seed_it_func: Callable | None = None,
    neuron_ids_to_analyze: List[int] = [0], # NOTE: this will optimize each id individually
    set_model_to_eval_mode: bool = True,
    objective_name: str = "increase",
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Generates a dictionary of MEIs for each neuron."""

    if random_seeds is not None and seed_it_func is None:
        print("Setting random seed function to torch.manual_seed")
        seed_it_func = torch.manual_seed
    elif random_seeds is None and seed_it_func is not None:
        raise ValueError("If seed_it_func is provided, random_seeds must also be provided.")
    elif random_seeds is None and seed_it_func is None:
        print("No random seeds provided, not setting at all")
        random_seeds = ["notset"] # type: ignore
        seed_it_func = lambda x: None
    

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
    stimulus_postprocessor_list, response_reducer,stimulus_regularizing_loss = generate_optimization_components(
        stimulus_range_constraints=opt_stim_generation_params["stimulus_range_constraints"],
        reducer_axis=0,
        reducer_start=opt_stim_generation_params["reducer_start"],
        reducer_length=opt_stim_generation_params["reducer_length"],
        temporal_gaussian_kwargs=opt_stim_generation_params["temporal_gaussian_kwargs"],
        spatial_gaussian_kwargs=opt_stim_generation_params["spatial_gaussian_kwargs"],
        range_regularization_kwargs=opt_stim_generation_params["range_regularization_kwargs"],
    )
    
    all_meis = {neuron_id: {} for neuron_id in neuron_ids_to_analyze}
    
    for i,seed in enumerate(random_seeds): # type: ignore
        seed_it_func(seed) # type: ignore

        for neuron_id in neuron_ids_to_analyze:
            
            # set the seed 
            single_neuron_seed_mei = generate_opt_stim(model=model,
                        new_session_id = new_session_id,
                        stimulus_postprocessor_list = stimulus_postprocessor_list,
                        response_reducer = response_reducer,
                        stimulus_regularizing_loss = stimulus_regularizing_loss,
                        stimulus_shape= STIMULUS_SHAPE,
                        neuron_id = neuron_id,
                        max_iterations=opt_stim_generation_params["max_iteration"],
                        lr=opt_stim_generation_params["lr"],
                        objective_name=objective_name,)
                        
            all_meis[neuron_id][seed] = single_neuron_seed_mei
    return all_meis



def generate_opt_stim_mulitple_objectives(
    model: BaseCoreReadout | EnsembleModel,
    new_session_id: str, 
    mei_generation_params: Dict[str, Any], 
    neuron_ids_to_analyze: List[int] = [0], # NOTE: this will optimize each id individually
    objective_name_list: List[str] = ["increase", "decrease"],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Generates a dictionary of MEIs for each neuron id and each random seed."""
    

    if isinstance(model, EnsembleModel):
        for member in model.members:
            member.eval()
            member.freeze()
    else:
        model.eval()
        model.freeze()


    # generate optimization components
    stimulus_postprocessor_list, response_reducer,stimulus_regularizing_loss = generate_optimization_components(
        stimulus_range_constraints=mei_generation_params["stimulus_range_constraints"],
        reducer_axis=0,
        reducer_start=mei_generation_params["reducer_start"],
        reducer_length=mei_generation_params["reducer_length"],
        temporal_gaussian_kwargs=mei_generation_params["temporal_gaussian_kwargs"],
        spatial_gaussian_kwargs=mei_generation_params["spatial_gaussian_kwargs"],
    )
    
    all_opt_stim = {neuron_id: {} for neuron_id in neuron_ids_to_analyze}
    

    for neuron_id in neuron_ids_to_analyze:
        
        for objective_name in objective_name_list:
            single_neuron_opt_stim = generate_opt_stim(model=model,
                        new_session_id = new_session_id,
                        stimulus_postprocessor_list = stimulus_postprocessor_list,
                        response_reducer = response_reducer,
                        stimulus_regularizing_loss = stimulus_regularizing_loss,
                        stimulus_shape= STIMULUS_SHAPE,
                        neuron_id = neuron_id,
                        max_iterations=mei_generation_params["max_iteration"],
                        lr=mei_generation_params["lr"],
                        objective_name="increase",)
        
            all_opt_stim[neuron_id][objective_name] = single_neuron_opt_stim
                    
    return all_opt_stim
