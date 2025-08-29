import torch
from typing import List, Tuple,Dict,Optional, Any
import numpy as np
import cv2
import os
import yaml
from openretina.models.core_readout import BaseCoreReadout
from simulations.loop_components.utils import log

# constants 
NORM_DICT = {'norm_mean': 36.979288270899204, 'norm_std': 36.98463253226166}


def draw_pixel_circle(tensor: torch.Tensor,
                     center: Tuple[int, int],
                     radius: int,
                     value: float) -> torch.Tensor:
    """
    Draws a circle on the tensor at the specified center with the given radius and value.
    The tensor is expected to be of shape (C, T, H, W).
    """
    c, t, h, w = tensor.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = ((x - center[1]) ** 2 + (y - center[0]) ** 2).sqrt()
    mask = dist_from_center <= radius
    tensor[:, :, mask] = value
    return tensor




def create_rf_test_tensor(heigth: int = 72, 
                          width: int = 64,
                          frames: int = 50, 
                          inter_stimulus_interval: int = 40,
                          radius: int | List[int] = [2,4], # half the size of the square in pixels
                          baseline_pixel_value: float = 122,
                          abs_pixel_increase_from_bsl: float = 100,
                         ) -> torch.Tensor:
    """
    Creates a tensor with the test stimulus for the receptive field (RF) test.
    Is a stimulus thats black and then has a circle with a given radius and pixel value on and then off. 
    There can be mulitple radii,
    assuming one pixel is 12,5 um then the default will draw circles of diameter 50um and 100 um."""
    
    if isinstance(radius, int):
        radius = [radius]


    trial_frame_nr = frames + inter_stimulus_interval

    # on, of for every radius
    total_nr_trials = len(radius) * 2  
    total_frame_nr = total_nr_trials * trial_frame_nr
   

    center_h = heigth // 2
    center_w = width // 2

    # create baseline tensor
    rf_test_tensor = baseline_pixel_value *  torch.ones((2, total_frame_nr , heigth, width), dtype=torch.float32)
    
    t_end = 0
    for rad in radius:
        for intensity_increase in [abs_pixel_increase_from_bsl, - abs_pixel_increase_from_bsl]:
            
            t_start = inter_stimulus_interval + t_end
            t_end = t_start + frames
            rf_test_tensor[:,t_start: t_end,] = draw_pixel_circle(
                rf_test_tensor[:,t_start: t_end,],
                center=(center_h, center_w),
                radius=rad,
                value=baseline_pixel_value + intensity_increase
            )

    # clip to [0, 255]
    torch.clamp(rf_test_tensor, min=0, max=255, out=rf_test_tensor)

    return rf_test_tensor

def extract_selected_meis(rois_seed: List[Tuple[int,int]],
                          neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
                          roi2readout_idx_wmeis: Dict[int,int]) -> Dict[str,torch.Tensor]:
    """
    Given a list of tuplse of roi_ids and random seeds, extracts the selected MEIS, and returns a dict with a unique identifier of roi and seed and the MEI.
      """
    
    selected_meis = {}
    for roi_id, seed in rois_seed:
        if roi_id not in roi2readout_idx_wmeis.keys():
            raise ValueError(f"roi_id {roi_id} not found in neuron_id_to_roi_id mapping.")
        
        # go from roi_id to neuron_idx in readout layer 
        neuron_id = roi2readout_idx_wmeis[roi_id]
        if neuron_id not in neuron_seed_mei_dict.keys() or seed not in neuron_seed_mei_dict[neuron_id].keys():
            raise ValueError(f"MEI for neuron_id {neuron_id} and seed {seed} not found in neuron_seed_mei_dict.")
        mei = neuron_seed_mei_dict[neuron_id][seed]
        unique_id = f"roi_{roi_id}_seed_{seed}"
        if unique_id in selected_meis.keys():
            print(f"Warning: Duplicate entry for {unique_id}. Overwriting previous MEI.")
            log(f"Warning: Duplicate entry for {unique_id}. Overwriting previous MEI.")
        selected_meis[unique_id] = mei
    
    return selected_meis

def create_all_mei_tensor(meis: List[torch.Tensor],
                          inter_stim_frames: int = 10,
                          baseline_pixel_value: float = 122,
                          ) -> torch.Tensor:
    """
    Given a list of MEI tensors, it concatenates them into a single tensor with the 
    provided inter stimulus interval."""
    c,t,h,w = meis[0].shape
    n_meis = len(meis)

    # create full tensor with coresponding shape
    full_tensor = baseline_pixel_value * torch.ones((c, n_meis * (t + inter_stim_frames), h, w), dtype=torch.float32)

    for i, mei in enumerate(meis):
        start_t = inter_stim_frames + i * (t + inter_stim_frames)
        end_t = start_t + t
        full_tensor[:, start_t:end_t, :, :] = mei
    return full_tensor


def generate_presentation_location_order(x_pos: List[float],
                                         y_pos: List[float],
                                         ) -> List[int]:
    """ WRITTEN BY AI
    Generates an ordering of the presentation locations such that the *total*
    (and therefore average) distance between consecutive locations is (approximately)
    maximized.

    Heuristic:
      1) Start from the farthest pair of points.
      2) Greedy max-insertion: insert each remaining point at the position that maximizes
         the *increase* in open-path length (including inserting at either end).
      3) Apply a 2-opt improvement pass for an open path to further increase length.

    Returns:
        ordering_idxs: a permutation of range(len(x_pos)).
    """
    import math
    assert len(x_pos) == len(y_pos), "x_pos and y_pos must have the same length"
    n = len(x_pos)
    if n <= 2:
        return list(range(n))

    # --- helpers ---
    def dist(i: int, j: int) -> float:
        dx = x_pos[i] - x_pos[j]
        dy = y_pos[i] - y_pos[j]
        return math.hypot(dx, dy)

    # total length of an open path
    def path_len(path: List[int]) -> float:
        return sum(dist(path[k], path[k+1]) for k in range(len(path)-1))

    # delta if we insert idx at position pos in path (between pos-1 and pos)
    # pos can be 0..len(path) inclusive; pos==0 or pos==len(path) means "at an end"
    def insertion_delta(path: List[int], idx: int, pos: int) -> float:
        if pos == 0:
            # insert at front: adds edge idx->path[0]
            return dist(idx, path[0])
        elif pos == len(path):
            # insert at end: adds edge path[-1]->idx
            return dist(path[-1], idx)
        else:
            a, b = path[pos-1], path[pos]
            # old edge a->b is replaced by a->idx and idx->b
            return dist(a, idx) + dist(idx, b) - dist(a, b)

    # open-path 2-opt: try reversing segments to increase total length
    def two_opt_open(path: List[int]) -> List[int]:
        improved = True
        while improved:
            improved = False
            L = len(path)
            # edges are (i-1,i) and (k, k+1) with 1 <= i < k < L-1
            for i in range(1, L-1):
                a, b = path[i-1], path[i]
                for k in range(i+1, L-0-1):  # k <= L-2
                    c, d = path[k], path[k+1]
                    gain = (dist(a, c) + dist(b, d)) - (dist(a, b) + dist(c, d))
                    if gain > 1e-12:
                        # Reverse the segment [i, k]
                        path[i:k+1] = reversed(path[i:k+1])
                        improved = True
                        break
                if improved:
                    break
        return path

    # --- 1) seed with farthest pair ---
    max_d = -1.0
    seed_i, seed_j = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            d = dist(i, j)
            if d > max_d:
                max_d = d
                seed_i, seed_j = i, j
    path = [seed_i, seed_j]

    remaining = set(range(n))
    remaining.discard(seed_i)
    remaining.discard(seed_j)

    # --- 2) greedy max-insertion ---
    while remaining:
        best_idx = None
        best_pos = None
        best_delta = -float('inf')
        for idx in remaining:
            # try all insertion positions, including ends
            for pos in range(len(path) + 1):
                delta = insertion_delta(path, idx, pos)
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_pos = pos
        # perform the best insertion
        path.insert(best_pos, best_idx)
        remaining.remove(best_idx)

    # --- 3) 2-opt improvement for open path ---
    path = two_opt_open(path)

    # return permutation of indices
    ordering_idxs = path
    return ordering_idxs


def save_stimulus_position_data_file(x_pos: List[float], y_pos: List[float], full_file_path: str) -> None:
    """
    Creates a text file where the desired positions where the MEIs should be presented are stored.
    It saves the x and y positions separated by comma. 
    Example x_pos = [1,2,3] and y_pos = [4,5,6] will result in: 1,4:2,5:3,6
    
    Args:
        x_pos: List of x positions
        y_pos: List of y positions
        filename: Path where to save the text file
    """
    # Check if the lists have the same length
    if len(x_pos) != len(y_pos):
        raise ValueError("x_pos and y_pos must have the same length")
    
    # Create the coordinate pairs
    coordinate_pairs = [f"{x},{y}" for x, y in zip(x_pos, y_pos)]
    
    # Join the pairs with colons
    content = ":".join(coordinate_pairs)
    
    # Write to file
    with open(full_file_path, 'w') as f:
        f.write(content)
    
    print(f"Position data saved to {full_file_path}")    

def put_mei_back_to_original_space(mei: torch.Tensor,
                                    norm_dict: Optional[Dict[str,float]] = None,
                                    mei_sd_scale_factor: float = 1.) -> torch.Tensor:
    """
    Puts the stimulus  back to the original space using the provided statistics.
    If no statistics are provided, it simply maps the tensor to [0, 255].
    """
    if norm_dict is not None:

        # revert normalization
        mei = mei * mei_sd_scale_factor * norm_dict["norm_std"] + norm_dict["norm_mean"]

        # clip to [0, 255]
        torch.clamp(mei,min = 0, max= 255, out=mei)

    else:
        # simply map to [0,255] 
        min_val = torch.min(mei)
        max_val = torch.max(mei)
        mei = 255 * (mei - min_val) / (max_val - min_val)
    
    return mei

    
def create_avi_from_tensor(stimulus: torch.Tensor, 
                           filename: str, 
                           fps: int = 60,
                           temporal_upsample_factor: int = 2,
                           rotate_90_cw: bool = True,
                           ) -> None:
    """Crates an AVI file from toch.Tensor stimulus and saves it at `filename`.
    Temporal upsmapling meand just repeating each frame.
    rotate_90_cw means that the stimulus is rotated 90 degrees clockwise because the way how QDSpy treats AVIs differently"""

    assert len(stimulus.shape) == 4, "Stimulus tensor must be of shape (C,T,H,W)"
    assert stimulus.shape[0] == 2, "Stimulus tensor must have 2 channels (G and UV)"
    assert isinstance(temporal_upsample_factor,int) and temporal_upsample_factor > 0, "Temporal upsample factor must be a positive integer"
    stimulus = stimulus.detach().cpu()

    # here has shape (C, T, H, W)
    stimulus_np = stimulus.numpy()

    if rotate_90_cw:
        # k = 1 is ccw, k = -1 is cw
        stimulus_np = np.rot90(stimulus_np, k=-1, axes=(2, 3))  # Rotate 90 degrees clockwise

    # open cv2 expects the stimulus in (T, H, W, C) format
    stimulus_np = np.transpose(stimulus_np, (1, 2, 3, 0)).astype(np.uint8)

    frames, height, width, channels = stimulus_np.shape

    rgb_frames = np.zeros((frames, height, width, 3), dtype=np.uint8)
    # Map channel 0 to green and channel 1 to blue (UV)
    rgb_frames[:, :, :, 1] = stimulus_np[:, :, :, 0]  # Green channel
    rgb_frames[:, :, :, 2] = stimulus_np[:, :, :, 1]  # Blue channel (for UV)
    stimulus_np = rgb_frames
            


    # Create video writer: FFV1 and  is lossless codec, XVID is lossy
    codec = 'FFV1' #'XVID' 
    fourcc = cv2.VideoWriter_fourcc(*codec)  # type : ignore # Use XVID codec 
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)
    
    # Write frames
    for i in range(frames * temporal_upsample_factor):
        frame = stimulus_np[i // temporal_upsample_factor]  # Select the frame based on the upsample factor
        # OpenCV uses BGR format
        if frame.shape[-1] == 3:  # RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore
        out.write(frame)  # type: ignore
     
    # Release resources
    out.release()

def retrieve_model_rf(trained_model: BaseCoreReadout):
    """
    Retrieves the receptive fields of the trained model.
    """
    pass


def get_roi_query_expression(roi_ids: List[int],) -> str:

    if len(roi_ids) == 1:
        _str = f"roi_id={roi_ids[0]}"
    elif len(roi_ids) > 1:
        _str = f"roi_id in {str(tuple(roi_ids))}"
    else:
        raise ValueError("roi_ids list is empty.")
    return _str

def transform_to_qdspy_coord(stimulus_table,all_x_pix: List[float], all_y_pix: List[float]) -> Tuple[List[float],List[float]]:
    """
    transforms indices of dense noise to QDSpy coordinates in microns.
    x_pix values are AXIS 0 in DENSE NOISE 
    y_pix values are AXIS 1 in DENSE NOISE
    """
    
    # Get stimulus parameters
    stim_query = stimulus_table & {'stim_family': 'noise'}
    stim_dict = stim_query.fetch1('stim_dict')

    # Get pixel size and offset
    pixel_size_x_um = stim_dict['pix_scale_x_um']
    pixel_size_y_um = stim_dict['pix_scale_y_um']

    # What about offset? in stim dict?
    offset_x_um = stim_dict['offset_x_um']
    offset_y_um = stim_dict['offset_y_um']

    # get number of pixels in x and y
    pix_n_x, pix_n_y = stim_dict["pix_n_x"],stim_dict["pix_n_y"]
    
    # RF center relative to stimcenter (assume centered noise presentation)
    # we need to further subtract one by the pixels to that it is in index 0 base
    x_center_pix = (pix_n_x - 1) / 2
    y_center_pix = (pix_n_y - 1) / 2

    if  isinstance(all_x_pix, (int, float)):
        # If single value is passed, convert to list
        all_x_pix_list = [all_x_pix]
    else:
        all_x_pix_list = all_x_pix

    if isinstance(all_y_pix, (int, float)):
        # If single value is passed, convert to list
        all_y_pix_list = [all_y_pix]
    else:
        all_y_pix_list = all_y_pix

    all_x_um = []
    all_y_um = []
    for x_pix,y_pix in zip(all_x_pix_list, all_y_pix_list):

        # add center in pixels (considering idx 0 base)
        x_pix = x_pix - x_center_pix
        y_pix = y_pix - y_center_pix

        # Calculate position in microns
        x_um = int(x_pix * pixel_size_x_um) + offset_x_um
        y_um = int(y_pix * pixel_size_y_um) + offset_y_um  
        all_x_um.append(x_um)
        all_y_um.append(y_um)

    return all_x_um, all_y_um


def extract_rf_means_from_selected_rois(roi_ids: List[int],
                                        stimulus_table: Any,
                                        gauss_rf_fit_table: Any,
                                        ) -> Tuple[List[float], List[float],List[int]]:
    """
    Extracts the RF  table for the given roi_ids.
    """
    expression = get_roi_query_expression(roi_ids)
    fit_query = gauss_rf_fit_table & expression
    if len(fit_query) == 0 or len(fit_query) != len(np.unique(roi_ids)):
        raise ValueError(f"Not all roi_ids {roi_ids} are present in the PeakSTAPosition table.")
    else:
        print(f"Found {len(fit_query)} rois in the FitGauss2D table.")

    srf_pamas,roi_order = fit_query.fetch('srf_params', 'roi_id',)

    # NOTE: the transform fom x to y is because in QDSpy axis 0 of dense noise is x
    all_x_pix = [srf_param["y_mean"] for srf_param in srf_pamas]
    all_y_pix = [srf_param["x_mean"] for srf_param in srf_pamas]

    # transform to QDSpy coordinates
    all_x_um, all_y_um = transform_to_qdspy_coord(stimulus_table, all_x_pix, all_y_pix)

    # turn roi_order to list of int
    roi_order = [int(r) for r in roi_order]


    return all_x_um, all_y_um,roi_order

def upsample_meis(meis: List [torch.Tensor] | torch.Tensor,
                  upsample_factor: int = 4,
                  ) -> List[torch.Tensor] | torch.Tensor:
    """
    Upsamples MEI tensors by repeating each pixel value `upsample_factor` times.
    Example if updampe_factor is 2 then pixel pixel value at (0,0) is also at (0,1) and (1,0) and (1,1).
    the default 4 will make that the 18x16 input is upsampled to 72x64."""

    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")

    if isinstance(meis, torch.Tensor):
        mei_list = [meis]
    else:
        meis_list = meis

    upsampled_meis = []
    for mei in mei_list:
        if not isinstance(mei, torch.Tensor):
            raise TypeError("All elements must be torch.Tensor.")
        if mei.ndim != 4:
            raise ValueError(f"Expected tensor of shape (C, T, H, W), got {mei.shape}.")

        # Repeat pixels into contiguous blocks along H and W
        upsampled = mei.repeat_interleave(upsample_factor, dim=2) \
                       .repeat_interleave(upsample_factor, dim=3)
        upsampled_meis.append(upsampled)

    return upsampled_meis if len(upsampled_meis) > 1 else upsampled_meis[0]
 
def create_rf_avi_from_roi_ids(roi_ids: List[int],
                                stimulus_table: Any,
                                gauss_rf_fit_table: Any,
                                abs_save_dir: str,
                              ) -> None:
    """
    Takes a list of rois, the PeakSTAPosition table, 
    and generated rf test stimuli as an avi file and stores the rf center coordingsates in a text file."""


    # extract the RF peaks from the PeakSTAPosition table
    rf_peak_x_um, rf_peak_y_um,roi_order = extract_rf_means_from_selected_rois(roi_ids, stimulus_table,gauss_rf_fit_table)
    log(f"Order of extracted RF peaks: {roi_order}.")

    # create the rf test stimulus
    print("Creating RF test stimulus tensor...")
    rf_test_tensor = create_rf_test_tensor(baseline_pixel_value=122,
                                           abs_pixel_increase_from_bsl= 122,)

    # create the stimulus position data file
    rf_metadata_filename = os.path.join(abs_save_dir,"rf_test_stimulus_metadata.txt")
    print("Saving RF test stimulus metadata to ", rf_metadata_filename)
    save_stimulus_position_data_file(rf_peak_x_um, rf_peak_y_um, rf_metadata_filename)
    


    # create the avi file
    rf_avi_filename = os.path.join(abs_save_dir,"rf_test_stimulus.avi")
    print("Creating RF test stimulus avi file at ", rf_avi_filename)
    create_avi_from_tensor(rf_test_tensor, rf_avi_filename)

    print("DONE!")


def retrieve_meis_and_save_as_avis(rois_seed: List[Tuple[int,int]],
                                   neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
                                   roi2readout_idx_wmeis: Dict[int,int],
                                   stimulus_directory: str,
                                   mei_sd_scale_factor: float = 1.0,
                                   n_bsl_fames_before_mei: int = 40,
                                   ) -> List[str]:
    """
    1. extracts the MEIs for the given roi_ids and seeds from the dict structure they were stored in the wrapper
    2. Upsamples and puts them back to original pixel space
    3. creates an avi file for each MEI and stores it in the given directory
    4. returns a list of the unique identifiers of the created MEI avi files."""

    # extract the MEIs for the given roi_ids and seeds from the dict structure they were stored in the wrapper
    selected_meis = extract_selected_meis(rois_seed, neuron_seed_mei_dict,roi2readout_idx_wmeis)


    ## Upsample and put back to space.
    for key, mei in selected_meis.items():

        # upsample
        mei = upsample_meis(mei, upsample_factor=4)

        # convert to orginal pixel space
        mei = put_mei_back_to_original_space(mei, # type: ignore
                                             norm_dict=NORM_DICT,
                                             mei_sd_scale_factor=mei_sd_scale_factor)
                
        # add the baseline frames before the mei
        assert mei.ndim == 4
        bsl_tensor = NORM_DICT["norm_mean"] * torch.ones((2, n_bsl_fames_before_mei, mei.shape[2], mei.shape[3]), dtype=torch.float32,device=mei.device)
        full_mei_stim = torch.cat((bsl_tensor, mei), dim=1)
    
        # create avi from stim tensor
        avi_filename = f"mei_{key}.avi"
        avi_full_path = os.path.join(stimulus_directory, avi_filename)
        print(f"Creating MEI avi file for {key} at {avi_full_path}.")
        create_avi_from_tensor(full_mei_stim, avi_full_path)

    return list(selected_meis.keys())

def generate_mei_ordering(mei_ids: List[str],n_pos,) -> List[List[str]]:
    """ 
    It generates len(mei_ids) number of lists containing shuffled of mei_ids. These sublists are of shape len(n_pos). 
    This is done such that: imagine the individual lists were columns of a matrix, 
    then it is not the case that in any row you have two identical mei_ids in neighboring columns or in the first or last column. 
    So basically for evey index in the list of sublists the mei_id is different from the one in the same index in the previous and next sublist.
    It returns a list of rows of this matrix, where basically each sublist containst the meis to play across the positions so that we only need a 
    list of positions and know what mei to play where for a given trial.
    """

    presentation_matrix = np.empty((len(mei_ids),n_pos), dtype=object)

    # get a base permutation
    np.random.seed(42)  # for reproducibility
    base_permutation = np.random.permutation(mei_ids).tolist()

    for col_idx in range(n_pos): 
        if col_idx == 0:
            permutation = base_permutation
        else:
            # shift permuation by one
            permutation = permutation[1:] + [permutation[0]]

        presentation_matrix[:,col_idx] = permutation

    mei_presentation_ordering = [presentation_matrix[trial].tolist() for trial in range(len(mei_ids))]

    return mei_presentation_ordering

    
def create_metadata_file(reordered_rf_peak_x_um: List[float],reordered_rf_peak_y_um: List[float],reordered_roi_ids: List[int],mei_presentation_ordering: List[List[str]],full_file_path: str) -> None:
    """
    Stored metadata in a dict and saves it as a yaml file.
    """
    
    # some checks
    assert len(mei_presentation_ordering[0]) == len(reordered_roi_ids) == len(reordered_rf_peak_x_um) == len(reordered_rf_peak_y_um), "Length of mei_presentation_ordering sublists, reordered_roi_ids, reordered_rf_peak_x_um and reordered_rf_peak_y_um must be the same"
    
    # save positions, roi_ids, mei_ids to be presented
    metadata = {"positions": [],
                "roi_ids": [],
                "mei_ids": mei_presentation_ordering,} 
    

    for x, y, roi_id in zip(reordered_rf_peak_x_um, reordered_rf_peak_y_um, reordered_roi_ids, strict=True):
        metadata["positions"].append([x, y])
        metadata["roi_ids"].append(roi_id)
    
    # save metadata as yaml
    with open(full_file_path, 'w') as file:
        yaml.dump(metadata, file)

def get_next_iteration_dir_from_remote(stimulus_output_dir: str,iteration_dir_name_base = "iter") -> Tuple[int,str]:
    """
    Given directory looks in dir to see other dirs containing iteration_dir_name_base 
    and returns the next iteration number, which is last in the folder name.
    
    Args:
        stimulus_output_dir: The directory to search in
        iteration_dir_name_base: The base name to look for in subdirectory names
        
    Returns:
        The next iteration number (max existing + 1, or 0 if none found)
    """    

    # Get all subdirectories in the stimulus_output_dir
    all_subdirs = [d for d in os.listdir(stimulus_output_dir) 
                  if os.path.isdir(os.path.join(stimulus_output_dir, d))]
    
    # Find subdirectories that match the pattern: base_name followed by a number
    iteration_numbers = []
    
    for subdir in all_subdirs:
        if iteration_dir_name_base in subdir:
            possible_nr = subdir[-1]
            try:
                iteration_number = int(possible_nr)
                iteration_numbers.append(iteration_number)
            except ValueError:
                continue       
    
    # If no matching directories found, return 0 as the first iteration
    if not iteration_numbers:
        next_iteation = 0
    else:
        # Return the next iteration number
        next_iteation = max(iteration_numbers) + 1
    next_dir = os.path.join(stimulus_output_dir, f"{iteration_dir_name_base}{next_iteation}")
    return next_iteation,next_dir







def create_single_mei_avis_and_metadata(
    rois_seed: List[Tuple[int,int]],
    neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
    roi2readout_idx_wmeis: Dict[int,int],
    stimulus_table: Any,
    fit_gauss_2d_rf_table: Any,
    abs_save_dir: str,
    mei_sd_scale_factor: float = 1.0,
):


    ## get the positions and their ordering
    rf_peak_x_um, rf_peak_y_um, initial_roi_id_order = extract_rf_means_from_selected_rois(
        [roi_id for roi_id, _ in rois_seed],
        stimulus_table=stimulus_table,
        gauss_rf_fit_table=fit_gauss_2d_rf_table,
    )
    log(f"Extracted the followin peak values for ROIs {initial_roi_id_order}:\nx {rf_peak_x_um}, \ny{rf_peak_y_um}.")
    print(f"Initial roi_id order: {initial_roi_id_order}.")

    # generate a ordering maximizing distance between MEIs
    presentation_ordering_idx = generate_presentation_location_order(rf_peak_x_um,
                                         rf_peak_y_um,
                                         )
    reordered_roi_ids = [initial_roi_id_order[i] for i in presentation_ordering_idx]
    reordered_rf_peak_x_um = [rf_peak_x_um[i] for i in presentation_ordering_idx]
    reordered_rf_peak_y_um = [rf_peak_y_um[i] for i in presentation_ordering_idx]

    log(f"Presentation ordering indices: {presentation_ordering_idx}\n corresponds to roi_id ordering of {reordered_roi_ids}.")
    print(f"Reordered roi_id order: {reordered_roi_ids}.")

    ## get the meis and their ordering
    # new file for iteration 
    iter_nr, abs_iteration_dir = get_next_iteration_dir_from_remote(
        stimulus_output_dir=abs_save_dir)
    os.makedirs(abs_iteration_dir, exist_ok=True)

    # retrieve meis and save as avis
    retrieved_mei_ids = retrieve_meis_and_save_as_avis(
        rois_seed=rois_seed,
        neuron_seed_mei_dict=neuron_seed_mei_dict,
        roi2readout_idx_wmeis=roi2readout_idx_wmeis,
        stimulus_directory=abs_iteration_dir,
        mei_sd_scale_factor=mei_sd_scale_factor,
        n_bsl_fames_before_mei=40,
    )
    log(f"Retrieved and saved MEIs with the following unique ids: {retrieved_mei_ids}.")
    print(f"Retrieved and saved MEIs with the following unique ids: {retrieved_mei_ids}.")
    # generate the presentation ordering of the retrieved meis.
    # List of sublists each idx per sublist is a mei id to be presented at a presentation location
    mei_presentation_ordering: List[List[str]] = generate_mei_ordering(retrieved_mei_ids,n_pos=len(reordered_roi_ids))
    log(f"Generated MEI presentation ordering: {mei_presentation_ordering}.")   
    print(f"Generated MEI presentation ordering: {mei_presentation_ordering}.") 

    ## save the metadata
    metadata_filename = os.path.join(abs_iteration_dir,"mei_metadata.yaml")
    create_metadata_file(reordered_rf_peak_x_um,
                         reordered_rf_peak_y_um,
                         reordered_roi_ids,
                         mei_presentation_ordering,
                         metadata_filename)
    log(f"Saved metadata file at {metadata_filename}.")
    print(f"Saved metadata file at {metadata_filename}.")
    print("DONE!")

