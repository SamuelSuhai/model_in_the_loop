import torch
from typing import List, Tuple,Dict,Optional, Any
import numpy as np
import cv2
import os

from openretina.models.core_readout import BaseCoreReadout


# constants 
NORM_DICT = {'norm_mean': 36.979288270899204, 'norm_std': 36.98463253226166}





def create_rf_test_tensor(heigth: int = 18, 
                          width: int = 16,
                          frames: int = 50, 
                          inter_stimulus_interval: int = 10,
                          half_square_pix_num = 1, # half the size of the square in pixels
                          baseline_pixel_value: float = 122,
                          abs_pixel_increase_from_bsl: float = 100,
                         ) -> torch.Tensor:
    """
    Creates a tensor """
    
   
    # create baseline tensor
    rf_test_tensor = baseline_pixel_value *  torch.ones((2, (frames  + inter_stimulus_interval) * 2 , heigth, width), dtype=torch.float32)

    # add a square in the middle in both channels (white)
    center_h = heigth // 2
    center_w = width // 2
    h_start = center_h - half_square_pix_num
    h_end = center_h + half_square_pix_num 
    w_start = center_w - half_square_pix_num
    w_end = center_w + half_square_pix_num 
    
    n_frames_trial = frames + inter_stimulus_interval
    for i,intensity_increase in enumerate([abs_pixel_increase_from_bsl, - abs_pixel_increase_from_bsl]):
        t_start = i * n_frames_trial + inter_stimulus_interval
        t_end = t_start + frames
        rf_test_tensor[:,t_start: t_end, h_start:h_end, w_start:w_end ] += intensity_increase  

    # clip to [0, 255]
    torch.clamp(rf_test_tensor, min=0, max=255, out=rf_test_tensor)

    return rf_test_tensor

def extract_selected_meis(rois_seed: List[Tuple[int,int]],
                          neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
                          roi_id_to_neuron_id: Dict[int,int]) -> Dict[str,torch.Tensor]:
    """
    Given a list of tuplse of roi_ids and random seeds, extracts the selected MEIS, and returns a dict with a unique identifier of roi and seed and the MEI.
      """
    
    selected_meis = {}
    for roi_id, seed in rois_seed:
        if roi_id not in roi_id_to_neuron_id.keys():
            raise ValueError(f"roi_id {roi_id} not found in neuron_id_to_roi_id mapping.")
        neuron_id = roi_id_to_neuron_id[roi_id]
        if neuron_id not in neuron_seed_mei_dict.keys() or seed not in neuron_seed_mei_dict[neuron_id].keys():
            raise ValueError(f"MEI for neuron_id {neuron_id} and seed {seed} not found in neuron_seed_mei_dict.")
        mei = neuron_seed_mei_dict[neuron_id][seed]
        unique_id = f"roi_{roi_id}_seed_{seed}"
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
                                    mei_sd_scale_factor: float = 2.) -> torch.Tensor:
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


def move_tensor_center_to_location(tensor: torch.Tensor, new_center_um: Tuple[float, float], pixel_size_um: float = 50) -> torch.Tensor:
    """
    It takes in a tensor where the center is in the middle of the tensor, and moves the center to the new location.
    For this the new_center_um is translated from um to pixels relative to the center.
    
    Args:
        tensor: Input tensor of shape (C, T, H, W)
        new_center_um: New center coordinates in micrometers (x, y)
        pixel_size_um: Size of each pixel in micrometers
        
    Returns:
        Shifted tensor with the center moved to the new location
    """
    # Get tensor dimensions
    _, _, height, width = tensor.shape
    
    # Calculate tensor center in pixels
    tensor_center_y = height // 2
    tensor_center_x = width // 2
    
    # Convert the new center from micrometers to pixels
    new_center_x_pix = new_center_um[0] / pixel_size_um
    new_center_y_pix = new_center_um[1] / pixel_size_um
    
    # Calculate the shift needed - we want to move the content so the tensor center
    # is at the new location. 
    shift_x = int(tensor_center_x - new_center_x_pix)
    shift_y = int(tensor_center_y - new_center_y_pix)
    
    # Use torch.roll to efficiently shift the tensor content
    # First shift in height dimension (dim=2)
    shifted_tensor = torch.roll(tensor, shifts=shift_y, dims=2)
    # Then shift in width dimension (dim=3)
    shifted_tensor = torch.roll(shifted_tensor, shifts=shift_x, dims=3)
    
    return shifted_tensor

def extract_rf_peaks_from_selected_rois(roi_ids: List[int],
                                        peak_sta_position_table: Any,
                                        ) -> Tuple[List[float], List[float]]:
    """
    Extracts the RF peaks from the PeakSTAPosition table for the given roi_ids.
    """
    dj_query = peak_sta_position_table & f"roi_id in {str(tuple(roi_ids))}"
    if len(dj_query) == 0 or len(dj_query) != len(np.unique(roi_ids)):
        raise ValueError(f"Not all roi_ids {roi_ids} are present in the PeakSTAPosition table.")
    else:
        print(f"Found {len(dj_query)} rois in the PeakSTAPosition table.")

    rf_peak_x_um = dj_query.fetch('rf_peak_x_um').tolist()
    rf_peak_y_um = dj_query.fetch('rf_peak_y_um').tolist()

    return rf_peak_x_um, rf_peak_y_um


 
def create_rf_avi_from_roi_ids(roi_ids: List[int],
                              peak_sta_position_table: Any,
                              abs_save_dir: str,
                              
                              ) -> None:
    """
    Takes a list of rois, the PeakSTAPosition table, 
    and generated rf test stimuli as an avi file and stores the rf center coordingsates in a text file."""


    # extract the RF peaks from the PeakSTAPosition table
    rf_peak_x_um, rf_peak_y_um = extract_rf_peaks_from_selected_rois(roi_ids, peak_sta_position_table)
    
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



def create_full_avi_from_roi_id_and_seed(rois_seed: List[Tuple[int,int]],
                                         neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
                                         roi_id_to_neuron_id: Dict[int,int],
                                         peak_sta_position_table: Any,
                                         trained_model: BaseCoreReadout, # for getting model RFs.
                                         abs_save_dir: str,
                                         mei_sd_scale_factor: float = 2,
                                         rf_scale_factor: float = 1.5,
                                         ) -> None:

    """
    Acts as a wrapper."""

    # some checks
    assert all([roi_id < 130 for roi_id, _ in rois_seed]), "roi_id must be less than 130"

    # extract the MEIs for the given roi_ids and seeds
    selected_meis = extract_selected_meis(rois_seed, neuron_seed_mei_dict,roi_id_to_neuron_id)

    # convert back to original space
    all_meis_list = []
    all_meis_ids = []
    for key, mei in selected_meis.items():
        mei = put_mei_back_to_original_space(mei,
                                             norm_dict=NORM_DICT,
                                             mei_sd_scale_factor=mei_sd_scale_factor)
        all_meis_list.append(mei)
        all_meis_ids.append(key)

    # swap two successive MEIs to avoid adptation.
    np.random.seed(42)  # for reproducibility
    suffled_order = np.random.permutation(list(range(len(all_meis_list))))
    all_meis_list = [all_meis_list[i] for i in suffled_order]
    all_meis_ids = [all_meis_ids[i] for i in suffled_order]
    print(f"MEIs shuffled. new index order:  {suffled_order}.")


    # create a tensor with all MEIs
    all_mei_tensor = create_all_mei_tensor(all_meis_list,
                                           baseline_pixel_value=NORM_DICT["norm_mean"],)

    # get the RF test stimulus
    rf_test_tensor = create_rf_test_tensor(baseline_pixel_value=NORM_DICT["norm_mean"],
                                           abs_pixel_increase_from_bsl= rf_scale_factor * NORM_DICT["norm_std"],)
    
    # concatenate the RF test stimulus with the MEIs
    full_stimulus = torch.cat((rf_test_tensor, all_mei_tensor), dim=1)

    # get the sta rf peak positions for selected rois
    rf_peak_x_um, rf_peak_y_um = extract_rf_peaks_from_selected_rois(
        [roi_id for roi_id, _ in rois_seed],
        peak_sta_position_table
    )

    # TODO: get model RF peaks
    rf_peak_x_um_model = []
    rf_peak_y_um_model = []

    full_rf_peak_x_um = rf_peak_x_um + rf_peak_x_um_model
    full_rf_peak_y_um = rf_peak_y_um + rf_peak_y_um_model

    # create metadata file
    mei_metadata_filename = os.path.join(abs_save_dir,"mei_test_stimulus_metadata.txt")
    print("Saving RF test stimulus metadata to ", mei_metadata_filename)
    save_stimulus_position_data_file(full_rf_peak_x_um, full_rf_peak_y_um, mei_metadata_filename)
  

    # create the avi file
    mei_avi_filename = os.path.join(abs_save_dir,"mei_test_stimulus.avi")
    print("Creating MEI test stimulus avi file at ", mei_avi_filename)
    create_avi_from_tensor(full_stimulus, mei_avi_filename)

    print("DONE!")

