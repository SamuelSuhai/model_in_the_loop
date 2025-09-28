
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from thesis.code.plot.roi_mask import plot_roi_mask_on_stack

import os
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.data_io.hoefling_2024.stimuli import movies_from_pickle

# import plotter
import thesis.code.plot.plot as plotter
import thesis.code.plot.style as styler

RESPONSES_PATH = "/gpfs01/euler/User/ssuhai/openretina_cache/euler_lab/hoefling_2024/responses/rgc_natstim_2024-08-14.h5"
MOVIES_PATH = "/gpfs01/euler/User/ssuhai/openretina_cache/euler_lab/hoefling_2024/stimuli/rgc_natstim_18x16_joint_normalized_2024-01-11.pkl"
OFFLINE_SESSION_ID = "session_1_ventral1_20200226"
ONLINE_SESSION_ID = "online_session_1_ventral1_20250717"


def load_openretina_data(cfg):
    # load OR data

    offline_raw_responses_dict = load_h5_into_dict(file_path=RESPONSES_PATH)

    offline_filtered_responses_dict = filter_responses(offline_raw_responses_dict, **cfg.model_configs.quality_checks)

    offline_neuron_data_dict = make_final_responses(offline_filtered_responses_dict, response_type="natural")

    movies_dict = movies_from_pickle(MOVIES_PATH)


def load_online_seesion_dict(path):
    with open(path, "rb") as f:
        online_session_dict = pickle.load(f)
    return online_session_dict


def find_offline_roi_id_from_session_dicts(online_roi_id,online_session_dict, offline_session_dict):

    online_idx = np.where(online_raw_session_dict["roi_ids"]== online_roi_id)[0].item()
    online_spikes = online_session_dict["natural_spikes"][online_idx]

    max_corr = -1
    best_offline_roi_id = None

    for or_roi_id in offline_session_dict["roi_ids"]:
        offline_idx = np.where(offline_session_dict["roi_ids"]== or_roi_id)[0].item()
        or_spikes = offline_session_dict["natural_spikes"][offline_idx]
        corr = np.corrcoef(online_spikes[~ np.isnan(online_spikes)], or_spikes[~ np.isnan(or_spikes)])[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_offline_roi_id = or_roi_id

    return best_offline_roi_id, offline_idx, online_idx, max_corr

def plot_online_offline_dict_val(online_idx, offline_idx, online_session_dict, offline_session_dict, key = "natural_spikes",win=(100,200), ax= None):
    
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()


    online_val = online_session_dict[key][online_idx]
    offline_val = offline_session_dict[key][offline_idx]


    palette = styler.get_palette('online_offline')
    ax.plot(online_val, color= palette['online'], label = 'online')
    ax.plot(offline_val, color= palette['offline'], label = 'offline')
    ax.legend()

    ax.set_xlim(win)
    ax.set_xlabel('Time [frames]')
    ax.set_ylabel('Spike Probability [a.u.]')

    return fig, ax


def load_mask_comparison_data(save_dir):

    # load stacks
    stack_path = os.path.join(save_dir, "ch0_stacks_20200226.pkl")
    with open(stack_path, "rb") as f:
        ch0_stacks = pickle.load(f)

    # load roi mask
    roi_mask_path = os.path.join(save_dir, "autorois_roi_mask_python_format_20200226.pkl")
    with open(roi_mask_path, "rb") as f:
        roi_mask = pickle.load(f)

    # load online traces
    online_traces_path = os.path.join(save_dir, "online_traces_20200226.pkl")
    with open(online_traces_path, "rb") as f:
        online_traces = pickle.load(f)

    # load offline traces
    offline_traces_path = os.path.join(save_dir, "offline_traces_20200226.pkl")
    with open(offline_traces_path, "rb") as f:
        offline_traces = pickle.load(f)

    # load online to offline roi mapping
    mapping_path = os.path.join(save_dir, "online2offline_roi_mapping_20200226.pkl")
    with open(mapping_path, "rb") as f:
        online2offline_roi_mapping = pickle.load(f)


    # openretina roi mask
    openretina_roi_mask_path = os.path.join(save_dir, "openretina_roi_mask_20200226.pkl")
    with open(openretina_roi_mask_path, "rb") as f:
        openretina_roi_mask = pickle.load(f)

    return {
        "ch0_stacks": ch0_stacks,
        "roi_mask": roi_mask,
        "online_traces": online_traces,
        "offline_traces": offline_traces,
        "online2offline_roi_mapping": online2offline_roi_mapping,
        "openretina_roi_mask": openretina_roi_mask,
    }




def mask_comparison(stack: np.ndarray,
                    online_mask: np.ndarray, 
                    offline_mask:np.ndarray,
                    online_rois: list,
                    offline_rois: list,
                    **plotting_kwargs) -> tuple[plt.figure,plt.Axes]:
    
    fig,ax = plt.subplots(1,2)


    plot_roi_mask_on_stack(ax = ax[0], 
                        ch_average=stack,
                        roi_mask = online_mask,
                        roi_ids=online_rois,
                        **plotting_kwargs
                        )

    # offline 

    plot_roi_mask_on_stack(ax = ax[1],
                            ch_average=stack,
                            roi_mask = offline_mask,
                            roi_ids=offline_rois,
                            **plotting_kwargs
                            )

    return fig,ax 


def load_online_session_dict(path):
    with open(path, "rb") as f:
        online_session_dict = pickle.load(f)
    return online_session_dict



def plot_online_offline(offline_signals, 
                        online_signals, time_window=(0, None)):
    """
    Plot multiple traces of offline (black) vs online (blue) signals with correlation values (r).

    Parameters
    ----------
    offline_signals : np.ndarray
        Shape (n_traces, n_timepoints). Offline/ground-truth signals.
    online_signals : np.ndarray
        Same shape as offline_signals. Online/model signals.
    time_window : tuple(int,int or None)
        (start, stop) frame indices to plot. If stop is None, goes to the end.
    """
    assert offline_signals.shape == online_signals.shape, "Offline and online must have the same shape"
    n_traces, n_timepoints = offline_signals.shape

    start, stop = time_window
    if stop is None:
        stop = n_timepoints
    if not (0 <= start < stop <= n_timepoints):
        raise ValueError("time_window must satisfy 0 <= start < stop <= n_timepoints")

    # Use frame indices for x-axis
    time = np.arange(start, stop)

    # Slice windows
    off_win = offline_signals[:, start:stop]
    on_win  = online_signals[:,  start:stop]

    # r per row, ignoring NaNs at shared positions
    r_values = []
    for off_row, on_row in zip(off_win, on_win):
        mask = ~np.isnan(off_row) & ~np.isnan(on_row)
        if np.count_nonzero(mask) >= 2:
            r = np.corrcoef(off_row[mask], on_row[mask])[0, 1]
        else:
            r = np.nan
        r_values.append(r)

    # Create subplots
    fig, axes = plt.subplots(n_traces, 1, figsize=(5, 0.5 * n_traces), sharex=True)
    if n_traces == 1:
        axes = [axes]

    # Add extra room on the right for the r text
    x_range = (stop - start)
    right_margin = 0.08 * x_range
    x_text = stop + 0.06 * x_range

    for i, ax in enumerate(axes):
        ax.plot(time, off_win[i], color='black', lw=1, label='Offline' if i == 0 else "")
        ax.plot(time, on_win[i],  color='dodgerblue', lw=1, label='Online'  if i == 0 else "")

        # y ticks off
        ax.set_yticks([])

        # Only bottom axis shows the x-axis (frames and label)
        if i != n_traces - 1:
            ax.xaxis.set_visible(False)

        # Remove all spines for every subplot…
        for spine in ax.spines.values():
            spine.set_visible(False)
        # …but keep a minimal bottom spine only for the last subplot
        if i == n_traces - 1:
            ax.spines['bottom'].set_visible(True)

        # Set x-limits with margin to the right (for r text)
        ax.set_xlim(start, stop + right_margin)

        # r text a bit more to the right
        ax.text(x_text, 0, f"r = {r_values[i]:.2f}" if np.isfinite(r_values[i]) else "r = n/a",
                va='center')

        # y-label in the middle subplot (like before)
        if i == n_traces // 2:
            ax.set_ylabel("Spike Probability [a.u.]")

    # Bottom x-label
    axes[-1].set_xlabel("Time [frames]")

    # Figure-level legend above all traces
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.05))

    return fig,ax



def load_spikes(save_dir):
    # load online spikes
    online_spikes_path = os.path.join(save_dir, "online_spikes_20200226.pkl")
    with open(online_spikes_path, "rb") as f:
        online_spikes_dict = pickle.load(f)

    # load offline spikes
    offline_spikes_path = os.path.join(save_dir, "offline_spikes_20200226.pkl")
    with open(offline_spikes_path, "rb") as f:
        offline_spikes_dict = pickle.load(f)

    return online_spikes_dict, offline_spikes_dict

def select_spikes_by_roi_ids(spikes_dict, roi_ids):
    roi_id_to_index = {roi_id: idx for idx, roi_id in enumerate(spikes_dict["roi_ids"])}
    selected_indices = []
    for roi_id in roi_ids: 
        if roi_id in roi_id_to_index:
            selected_indices.append(roi_id_to_index[roi_id])
        else:
            print(f"Warning: ROI ID {roi_id} not found in spikes_dict")
    
    selected_spikes = spikes_dict["spikes"][selected_indices]
    return selected_spikes
