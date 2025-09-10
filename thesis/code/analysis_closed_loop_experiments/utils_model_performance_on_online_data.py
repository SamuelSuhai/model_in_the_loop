import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional,Dict


def bring_pred_target_dict_to_array(pred_target_dict:Dict[int,Tuple[np.ndarray,np.ndarray]]) -> Tuple[np.ndarray,np.ndarray]:
    """
    Convert a dictionary of predictions and targets into two numpy arrays.
    
    Args:
        pred_target_dict (Dict[int, Tuple[np.ndarray, np.ndarray]]): 
            A dictionary where keys are neuron indices and values are tuples of (predictions, targets).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            Two numpy arrays: one for all predictions and one for all targets.
            Each array has shape (num_neurons, num_timepoints).
    """
    num_neurons = len(pred_target_dict)
    # Assuming all predictions and targets have the same length
    first_key = list(pred_target_dict.keys())[0]

    n_pred_ts = pred_target_dict[first_key][0].shape[0]
    n_target_ts = pred_target_dict[first_key][1].shape[0]
    if n_pred_ts != n_target_ts:
        print(f"Warning: Number of prediction timepoints ({n_pred_ts}) and target timepoints ({n_target_ts}) do not match.")
    
    all_predictions = np.zeros((num_neurons, n_pred_ts))
    all_targets = np.zeros((num_neurons, n_target_ts))
    
    
    for neuron_idx, (pred, target) in pred_target_dict.items():
        all_predictions[neuron_idx] = pred
        all_targets[neuron_idx] = target
    
    return all_predictions, all_targets



def plot_all_neuron_predicted_actual(responses,predictions,time_window=None,figsize=(15,10)) -> plt.Figure:
    """"""
    assert responses.shape == predictions.shape, f"Response shape {responses.shape} and model prediction shape {predictions.shape} do not match."
    assert responses.ndim == 2, f"Response should be 2D array for multiple neurons, got shape {responses.shape}."

    num_neurons = responses.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(num_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_neurons):
        ax = axes[i]
        response = responses[i]
        predicted = predictions[i]

        correl = np.corrcoef(response, predicted)[0,1]

        if time_window is None:
            time_window = (0, predicted.shape[0])

        ax.plot(response[time_window[0]:time_window[1]], label='Actual', color='blue')
        ax.plot(predicted[time_window[0]:time_window[1]], label='Predicted', color='orange')
        ax.set_title(f'Neuron {i+1} Correlation: {correl:.4f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return fig

def plot_single_neuron_predicted_actual(target_response,predicted_response,time_window=None,ax =None) -> plt.Axes | Tuple[plt.Axes,float]:
    """"""
    
    assert target_response.shape == predicted_response.shape, f"Response shape {target_response.shape} and model prediction shape {predicted_response.shape} do not match."
    assert target_response.ndim == 1, f"Response should be 1D array for a single neuron, got shape {target_response.shape}."

    # print correl 
    correl = np.corrcoef(target_response, predicted_response)[0,1]
    print(f"Correlation between actual and predicted: {correl:.4f}")
    
    if time_window is None:
        time_window = (0, predicted_response.shape[0])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.plot(target_response[time_window[0]:time_window[1]], label='Actual', color='blue')
    ax.plot(predicted_response[time_window[0]:time_window[1]], label='Predicted', color='orange')
    ax.set_title(f'Response vs Prediction. Correlation: {correl:.4f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Response')
    ax.legend()

    return ax,correl



def plot_predicted_actural(response,model_pred,neuron_idx, time_window=None,ax =None):

    
    cut_n_frames = response.shape[1] - model_pred.shape[1]
    responses_reshaped = response[:, cut_n_frames:]

    assert responses_reshaped.shape == model_pred.shape, f"Response shape {responses_reshaped.shape} and model prediction shape {model_pred.shape} do not match after cutting {cut_n_frames} frames."
    

    ax,correl = plot_single_neuron_predicted_actual(responses_reshaped[neuron_idx],model_pred[neuron_idx])



def extract_data_from_wrapper(wrapper,pipeline_data_path,cfg):
    # look at rois and traces etc
    from openretina.data_io.hoefling_2024.responses import make_final_responses
    from model_in_the_loop.utils.model_training import load_stimuli
    from openretina.data_io.hoefling_2024.dataloaders import natmov_dataloaders_v2

    wrapper.load_all_data_from_dir(pipeline_data_path)
    movies_dict = load_stimuli(cfg.model_configs)
    neuron_data_dict = make_final_responses(wrapper.session_dict_raw)
    seesion_id = list(wrapper.session_dict_raw.keys())[0]
    model = wrapper.model
    data_loader_kwargs = {
    "batch_size": 64,
    "train_chunk_size": 50,
    "allow_over_boundaries": True,
    "validation_clip_indices": [0, 4, 10, 11, 18, 30, 45, 62, 67, 77, 79, 80, 81, 83, 95],

}
    dataloader = natmov_dataloaders_v2(
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
        **data_loader_kwargs
    )
    return movies_dict, neuron_data_dict, seesion_id, model,dataloader
