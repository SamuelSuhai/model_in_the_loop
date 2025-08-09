import torch
from typing import List, Tuple,Dict,Optional, Any
import numpy as np
import cv2

from openretina.models.core_readout import BaseCoreReadout

def create_rf_test_tensor(heigth: int = 18, 
                          width: int = 16,
                          frames: int = 50, 
                          inter_stimulus_interval: int = 10,
                          baseline_pixel_value: float = 122
                         ) -> torch.Tensor:
    """
    Creates a tensor """
    
    # create baseline tensor
    rf_test_tensor = baseline_pixel_value *  torch.ones((2, (frames  + inter_stimulus_interval) * 2 , heigth, width), dtype=torch.float32)

    # add a square in the middle in both channels (white)
    center_h = heigth // 2
    center_w = width // 2
    square_pix_num = 2
    h_start = center_h - square_pix_num
    h_end = center_h + square_pix_num + 1
    w_start = center_w - square_pix_num
    w_end = center_w + square_pix_num + 1
    
    for i,intensity in zip([255,0],[1,2]):
        t_start = i * inter_stimulus_interval
        t_end = i * (inter_stimulus_interval + frames)
        rf_test_tensor[:,t_start: t_end, h_start:h_end, w_start:w_end ] = intensity  

    return rf_test_tensor

def extract_selected_meis(rois_seed: List[Tuple[int,int]],neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    """
    Given a list of tuplse of roi_ids and random seeds, extracts the selected MEIS, and returns a dict with a unique identifier of roi and seed and the MEI """
    pass 


def create_all_mei_tensor(meis: List[torch.Tensor],inter_stim_frames: int = 10) -> torch.Tensor:
    """
    Given a list of MEI tensors, it concatenates them into a single tensor with the 
    provided inter stimulus interval."""
    pass


def save_stimulus_position_data_file(x_pos:List[float], y_pos:List[float],filename: str) -> None:
    """
    creates a text file where the desired positions where the MEIs should be presented are stored."""
    pass 

def put_mei_back_to_original_space(mei: torch.Tensor,
                                    original_stimulus_stats: Optional[Dict[str,float]] = None) -> torch.Tensor:
    """
    Puts the stimulus  back to the original space using the provided statistics.
    If no statistics are provided, it simply maps the tensor to [0, 255].
    """
    if original_stimulus_stats is not None:
        mei = mei * original_stimulus_stats["std"] + original_stimulus_stats["mean"]
    else:
        # simply map to [0,255] 
        min_val = torch.min(mei)
        max_val = torch.max(mei)
        mei = 255 * (mei - min_val) / (max_val - min_val)
    
    return mei

    
def create_avi_from_tensor(stimulus: torch.Tensor, 
                           filename: str, 
                           fps: int = 30, 
                           ) -> None:
    """Crates an AVI file from toch.Tensor stimulus and saves it at `filename`"""

    assert len(stimulus.shape) == 4, "Stimulus tensor must be of shape (C,T,H,W)"
    assert stimulus.shape[0] == 2, "Stimulus tensor must have 2 channels (G and UV)"
    stimulus = stimulus.detach().cpu()

   
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

def retrieve_model_rf(trained_model: BaseCoreReadout):
    """
    Retrieves the receptive fields of the trained model.
    """
    pass

def create_rf_avi_from_roi_ids(roi_ids: List[int],
                              peak_sta_position_table: Any,
                              abs_save_dir: str,
                              ) -> None:
    """
    Takes a list of rois, the PeakSTAPosition table, 
    and generated rf test stimuli as an avi file and stores the rf center coordingsates in a text file."""

    dj_query = peak_sta_position_table & {'roi_id': roi_ids}
    if len(dj_query) == 0 or len(dj_query) != len(roi_ids):
        raise ValueError(f"Not all roi_ids {roi_ids} are present in the PeakSTAPosition table.")
    rf_peak_x_um = dj_query.fetch('rf_peak_x_um')
    rf_peak_y_um = dj_query.fetch('rf_peak_y_um')

    # create the rf test stimulus
    rf_test_tensor = create_rf_test_tensor()

    # create the stimulus position data file
    rf_metadata_filename = f"{abs_save_dir}/rf_test_stimulus_metadata.txt"
    save_stimulus_position_data_file(rf_peak_x_um, rf_peak_y_um, rf_metadata_filename)

    # create the avi file
    rf_avi_filename = f"{abs_save_dir}/rf_test_stimulus.avi"
    create_avi_from_tensor(rf_test_tensor, rf_avi_filename)




def create_rf_mei_avi_from_roi_id_and_seed(rois_seed: List[Tuple[int,int]],
                                         neuron_seed_mei_dict: Dict[int,Dict[int,torch.Tensor]],
                                         neuron_id_to_roi_id: Dict[int,int],
                                         trained_model: BaseCoreReadout, # for getting model RFs.
                                         abs_save_dir: str,

                                         ):
    """
    Acts as a wrapper."""

    # extract the MEIs for the given roi_ids and seeds
    selected_meis = extract_selected_meis(rois_seed, neuron_seed_mei_dict)

    # create a tensor with all MEIs
    all_mei_tensor = create_all_mei_tensor(list(selected_meis.values()))

    # get the RF test stimulus
    rf_test_tensor = create_rf_test_tensor()
    
    # concatenate the RF test stimulus with the MEIs
    full_stimulus = torch.cat((rf_test_tensor, all_mei_tensor), dim=1)

    # save the stimulus position data file

