import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import thesis.code.plot.style as styler

from typing import List, Dict, Any

from model_in_the_loop.utils.model_training import (get_predictions_targets_one_dataloader,
                                                    load_pretrained_ensemble_model,
                                                    train_or_refine_member_or_ensemble,
                                                    get_dataloaders_and_data_info,
                                                    load_stimuli,
                                                    )

from model_in_the_loop.utils.stimulus_optimization import (reconstruct_mei_from_decomposed,center_member_or_ensemble_readouts,
                                                           generate_opt_stim_for_neuron_list,
                                                            decompose_mei, get_model_mei_response
                                                           )


from openretina.data_io.hoefling_2024.responses import make_final_responses,filter_responses


# initialize mei data containter

def get_mei_container(model,
                      session_id,
                      cfg,
                      ):

    n_neurons = model.members[0].data_info['n_neurons_dict'][session_id]
    print(n_neurons)

    # set variables needed for container code
    seeds = [111,222]
    idx2stability = {i:"stable" for i in range(n_neurons)}
    roi_ids = model.members[0].data_info["sessions_kwargs"][session_id]["roi_ids"]
    readout_idx_wmei2rois = {i:r for i,r in zip(range(n_neurons),roi_ids)}


    mei_data_container_entries = []
    _ = center_member_or_ensemble_readouts(model, session_id)

    ## generate meis
    phase = "stable"
    print(f"Generating {phase} MEIs for neurons (readout idx): {[idx for idx,stab in idx2stability.items() if stab ==phase ]}.")
    set_model_to_eval_mode = True if phase == 'stable' else False
    neuron_ids_to_analyze = [neuron_id for neuron_id, stability in idx2stability.items() if stability == phase]
    seeds = seeds if phase == 'unstable' else [seeds[0]] # only one seed for stable meis
    neuron_seed_mei_dict =  generate_opt_stim_for_neuron_list(
                                    model = model,
                                    new_session_id = session_id,
                                    opt_stim_generation_params= cfg.stimulus_optimization,
                                    random_seeds = seeds,
                                    seed_it_func= torch.manual_seed,
                                    neuron_ids_to_analyze = neuron_ids_to_analyze, # NOTE: this will optimize each id individually 
                                    set_model_to_eval_mode = set_model_to_eval_mode, # model in training mode for noisy MEIs
                                    )

    print(f"Start decomposing ...")    
    ## decompose meis
    device =  model.members[0].device
    for neuron_id,seed_dict in neuron_seed_mei_dict.items():
        print(f"Decomposing MEIs for neuron (readout idx) {neuron_id} ...")
        for seed,mei in seed_dict.items():

            # decompose the MEIs
            temporal_kernels, spatial_kernels, _ = decompose_mei(stimulus = mei.detach().cpu().numpy())
        

            if cfg.stimulus_optimization["reconstruct_mei"]:
                reconstruction = reconstruct_mei_from_decomposed(
                            temporal_kernels=temporal_kernels,
                            spatial_kernels=spatial_kernels,
                            turn_to_tensor=True)

                assert reconstruction.shape == mei.shape, "Reconstructed MEI shape does not match original MEI shape."
                
                # make reonstruction same norm as mei
                print(f"changing norm of reconstruction {torch.norm(reconstruction)} to match original mei norm {torch.norm(mei)}")
                reconstruction = reconstruction / torch.norm(reconstruction) * torch.norm(mei)
                print(f"new reconstruction norm {torch.norm(reconstruction)}")
                mei = reconstruction # use the reconstructed MEI for further analysis
                print(f"Done reconstructing MEI for neuron (readout idx) {neuron_id}, seed {seed}.")
            
            # add entry to data container 
            mei_data_container_entries.append({
                "readout_idx": neuron_id,
                "roi_id": readout_idx_wmei2rois[neuron_id],
                "mei_id": f"roi_{readout_idx_wmei2rois[neuron_id]}_seed_{seed}",
                "seed": seed,
                "mei": mei.detach(),
                "temporal_kernels": temporal_kernels,
                "spatial_kernels": spatial_kernels,
                "stability": phase,
            })


    # make df container from all meis
    mei_data_container = pd.DataFrame(mei_data_container_entries)
    return mei_data_container


def load_file_from_pickle(file_path):
    with open(file_path,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_torch_file(file_path):
    try:
        # First try loading with pickle
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except:
        try:
            # If pickle fails, try torch.load with CPU mapping
            obj = torch.load(file_path, map_location='cpu', weights_only=False)
            return obj
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Trying alternative loading method...")
            # Try loading with torch.load and strict=False
            obj = torch.load(
                file_path,
                map_location='cpu',
                weights_only=False,
                pickle_module=pickle
            )
            return obj


def plot_single_trf_temp_kernel_comparison(trf: np.ndarray,
                                           temp_kernel: np.ndarray,
                                           t_trf: np.ndarray,
                                            t_temp_kernel: np.ndarray,
                                           ax: plt.Axes=None,
                                           trf_label:str='trf',
                                           temp_kernel_label:str='temp kernel',
                                           normalize=True,
                                           set_legend=True,
                                           **plotting_kwargs) -> plt.Axes:

    if ax is None:
        fig,ax = plt.subplots(1,1)

    if normalize:
        # scale from 0,1
        trf = (trf - np.min(trf))/ (np.max(trf) - np.min(trf))
        temp_kernel = (temp_kernel - np.min(temp_kernel))/ (np.max(temp_kernel) - np.min(temp_kernel))
    
    ax.plot(t_trf,trf, label=trf_label, **plotting_kwargs)
    ax.plot(t_temp_kernel,temp_kernel, label=temp_kernel_label, **plotting_kwargs)
    if set_legend:
        ax.legend()

    # remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def add_summed_kernels_column(df: pd.DataFrame,
                              kernel_column_name: str):
    """add col containingn sum """
    df["summed_"+kernel_column_name] = df[kernel_column_name].apply(lambda x: x[0] + x[1])
    return df



def adjust_trf_sign(trf,srf) -> np.ndarray:
    """Flip trf sign if srf is negative"""
    if np.abs(srf.min()) > np.abs(srf.max()):
        trf = -trf
    return trf


def get_time_vector(n_time_bins: int, dt: float,t_start:float = None,t_stop: float=None) -> np.ndarray:
    """Get time vector given number of time bins and time step"""

    assert not (t_start is None and t_stop is None), "Either both t_start and t_stop are None or both are not None"
    ts = np.arange(0,n_time_bins*dt,dt)
    if t_start is not None:
        ts = ts + t_start
    if t_stop is not None:
        ts = ts - ts[-1] + t_stop
    return ts


def plot_multiple_trf_temp_kernel_comparisons(
    trfs: list[np.ndarray],
    temp_kernels: list[np.ndarray],
    t_trfs: list[np.ndarray],
    t_temp_kernels: list[np.ndarray],
    celltypes: list[str] = None,
    figsize=(10, 6),
    trf_labels: list[str] = None,
    temp_kernel_labels: list[str] = None,
    xlabel: str = "Time [s]",
    ylabel: str = "Normalized Stimulus Intensity",
    remove_individual_labels: bool = True,
    time_window=None,
    shift_kernel_by = 0,
    **plotting_kwargs
) -> plt.Figure:
    """
    Plot multiple TRF and temperature kernel comparisons in subplots sharing an x-axis.
    """
    n_plots = len(trfs)
    assert n_plots == len(temp_kernels) == len(t_trfs) == len(t_temp_kernels), "Input lists must be of the same length."
    if celltypes is not None:
        assert len(celltypes) == n_plots, "Length of celltypes must match the number of plots."

    if trf_labels is None:
        trf_labels = ["TRF"] * n_plots
    else:
        assert len(trf_labels) == n_plots, "Length of trf_labels must match the number of plots."
    if temp_kernel_labels is None:
        temp_kernel_labels = ["Temp Kernel"] * n_plots
    else:
        assert len(temp_kernel_labels) == n_plots, "Length of temp_kernel_labels must match the number of plots."

    # constrained_layout helps with super labels
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # Handle the case of a single plot
    if n_plots == 1:
        axes = [axes]


    for i, (trf, temp_kernel, t_trf, t_temp_kernel, ax) in enumerate(zip(trfs, temp_kernels, t_trfs, t_temp_kernels, axes)):
        plot_single_trf_temp_kernel_comparison(
            trf=trf,
            temp_kernel=temp_kernel ,
            t_trf=t_trf,
            t_temp_kernel=t_temp_kernel + shift_kernel_by,
            ax=ax,
            trf_label=trf_labels[i],
            temp_kernel_label=temp_kernel_labels[i],
            set_legend=False,
            **plotting_kwargs
        )

        if celltypes is not None:
            ax.text(0.95, 0.95, celltypes[i], transform=ax.transAxes, va='top', ha='right', fontsize=10)

        if remove_individual_labels:
            # Remove y ticks entirely (not just labels) on all subplots
            ax.set_yticks([])

            # Hide x tick labels on all but the bottom subplot
            if i < n_plots - 1:
                pass
                # ax.set_xticklabels([])
                # ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
            else:
                # Ensure bottom subplot shows x ticks and labels
                ax.tick_params(axis='x', which='both', length=4, labelbottom=True)

    # Shared labels centered on the figure
    if remove_individual_labels:
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
    else:
        # Per-axis labeling: put ylabel on the middle subplot, x on the bottom
        mid = n_plots // 2
        axes[mid].set_ylabel(ylabel)
        axes[-1].set_xlabel(xlabel)

    # Remove any leftover title on the first axis (if your single-plot function sets it)
    if hasattr(axes[0], 'set_title'):
        axes[0].set_title("")

    # set last time window -> all will get bx shapre x
    if time_window is not None:
        axes[-1].set_xlim(*time_window)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False
    )

    return fig

def prepare_trf_kernel_data_for_plotting(
    roi_ids: list,
    roi_data_df: pd.DataFrame,
    kernel_data_df: pd.DataFrame,
    trf_column: str = "trf_signed",
    kernel_column: str = "summed_temporal_kernels",
    trf_dt: float = 0.05,
    kernel_dt: float = 1/30,
    return_celltype: bool = True
):
    """
    Prepare data for plotting multiple TRF and kernel comparisons.
    
    Parameters:
    ----------
    roi_ids : list
        List of ROI IDs to extract data for.
    roi_data_df : pd.DataFrame
        DataFrame containing ROI data with columns for roi_id and TRFs.
    kernel_data_df : pd.DataFrame
        DataFrame containing kernel data with a column for temporal kernels.
    trf_column : str, optional
        Column name in roi_data_df for TRFs. Default is "trf_signed".
    kernel_column : str, optional
        Column name in kernel_data_df for kernels. Default is "summed_temporal_kernels".
    trf_dt : float, optional
        Time step for TRF time vector. Default is 0.05.
    kernel_dt : float, optional
        Time step for kernel time vector. Default is 1/30.
    return_celltype : bool, optional
        If True, return cell types as ROI IDs. Default is True.
        
    Returns:
    -------
    dict
        Dictionary containing the data needed for plot_multiple_trf_temp_kernel_comparisons:
        - trfs: list of TRFs
        - temp_kernels: list of temporal kernels
        - t_trfs: list of time vectors for TRFs
        - t_temp_kernels: list of time vectors for temporal kernels
        - celltypes: list of ROI IDs (if return_celltype is True)
    """
    # Filter dataframes to only include the specified ROIs
    roi_data_filtered = roi_data_df[roi_data_df['roi_id'].isin(roi_ids)]

    # if not all([roi in kernel_data_df["roi_id"] for roi in roi_ids]):
    #     raise ValueError
    
    if (kernel_data_df.query("roi_id in @roi_ids")["stability"] =="unstable").any():
        raise ValueError
    
    # Sort by the order of roi_ids
    roi_data_filtered = roi_data_filtered.set_index('roi_id').loc[roi_ids].reset_index()
    
    # Extract TRFs
    trfs = roi_data_filtered[trf_column].tolist()
    
    # Extract temporal kernels
    # Assuming kernel_data_df has a 'roi_id' column or similar for matching
    if 'roi_id' in kernel_data_df.columns:
        kernel_data_filtered = kernel_data_df[kernel_data_df['roi_id'].isin(roi_ids)]
        kernel_data_filtered = kernel_data_filtered.set_index('roi_id').loc[roi_ids].reset_index()
        temp_kernels = kernel_data_filtered[kernel_column].tolist()
    else:
        # If no roi_id column, assume the dataframe is already ordered correctly
        # and take the first len(roi_ids) entries
        temp_kernels = kernel_data_df[kernel_column].iloc[:len(roi_ids)].tolist()
    
    # Create time vectors
    t_trfs = [get_time_vector(len(trf), dt=trf_dt, t_stop=0.0) for trf in trfs]
    t_temp_kernels = [get_time_vector(len(kernel), dt=kernel_dt, t_stop=0.0) for kernel in temp_kernels]
    
    result = {
        'trfs': trfs,
        'temp_kernels': temp_kernels,
        't_trfs': t_trfs,
        't_temp_kernels': t_temp_kernels,
    }
    
    # Add celltypes if requested
    if return_celltype:
        result['celltypes'] = [f'ROI {roi_id}' for roi_id in roi_ids]
    
    return result



def add_online_offline_comparison_legend(ax: np.ndarray[plt.Axes] | plt.Axes,also_comparison: bool=True):
    """
    Adds 2 (or 3) proxy artists above axis in center for online offline comparison legend.
    """
    palette = styler.get_palette('online_offline')
    online,offline = palette['online'],palette['offline']
    comparison = "purple"

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=offline, label='Offline'),
        Patch(facecolor=online, label='Online'),
        ]
    if also_comparison:
        legend_elements.append(
            Patch(facecolor=comparison, label='Difference')
        )
    if isinstance(ax, np.ndarray):
        _ax = ax.flatten()[0]
    else:
        _ax = ax

    _ax.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(1.1, 1.55),
              ncol=len(legend_elements),
              frameon=False
              )
    return ax
    



##################################################################### ROI subset selection plotting /data utils #######

def load_wrapper_data_for_subset_selection(wrapper_save_dir: str):
    """
    Given dir where RandomSeedMEIWrapper data is saved, load the relevant data.
    mei_data_container: pd.DataFrame
    neuron_data_dict: Dict[int, NeuronData]

    metadata: Dict (from here it gets new_session_id and readout_idx_wmei2rois)
    """
    mei_data_container = load_file_from_pickle(f"{wrapper_save_dir}/mei_data_container.pkl")
    neuron_data_dict = load_file_from_pickle(f"{wrapper_save_dir}/neuron_data_dict.pkl")    
    metadata = load_file_from_pickle(f"{wrapper_save_dir}/metadata.pkl")
    new_session_id = metadata['new_session_id']
    roi2readout_idx_wmeis = metadata['roi2readout_idx_wmeis']
    return mei_data_container, neuron_data_dict, new_session_id, roi2readout_idx_wmeis


def plot_responses_and_mei_info_one_roi(
        roi_id: int,
        roi_id2mei_ids: Dict[int, List[str]],
        roi_id2info: Dict[int, Dict[str, Any]],
        verbose = True,
        figsize: tuple[int,...] =(10, 6),
        ax = None,

    ):

    """
    Plot for a given roi_id:
    On the x axis the MEI ids as tick labels and in parenthesis in a row below either:
        1. own (if its the mei of the roi ie roi_id is in mei_id)
        2. same celltype (if the mei is from a different roi but same celltype)
        otherwise nothing.
    roi_id2mei_ids: dict from roi id to mei ids list
    roi_id2info: dict from roi id to dict with keys:
        "all_stabilities": stabilities of the roi in key to the meis we have in roi_id2mei_ids[roi_id],
        "celltype": celltypes of the roi in key to the meis we have in roi_id2mei_ids[roi_id],
        "responses": mei_responses of the roi in key to the meis we have in roi_id2mei_ids[roi_id]
    }


    """
    # Get data for this ROI
    mei_ids = roi_id2mei_ids[roi_id]
    responses = roi_id2info[roi_id]["responses"]
    celltypes = roi_id2info[roi_id]["celltype"]
    if verbose:
        print(f"All responses {responses}\nall celltypes {celltypes}\nall meis {mei_ids}")
    
    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # Plot bars
    bars = ax.bar(range(len(mei_ids)), responses)
    
    # Add labels and annotations
    ax.set_xticks(range(len(mei_ids)))
    
    # Create second row of labels
    labels = []
    for i, mei_id in enumerate(mei_ids):
        # Check if this is the ROI's own MEI
        if f"roi_{roi_id}_" in mei_id:
            labels.append(f"{mei_id}\n(own, G{str(celltypes[i])})")
            # Show type
        else:
            labels.append(mei_id + "\n(G" + str(celltypes[i]) + ")")
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Set titles and labels
    ax.set_title(f'Responses for ROI {roi_id}')
    ax.set_ylabel('Response Strength')
    
    # Add stability info as color
    for i, (bar, stability) in enumerate(zip(bars, roi_id2info[roi_id]["all_stabilities"])):
        color = 'skyblue' if stability == 'stable' else 'salmon'
        bar.set_color(color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Stable MEI'),
        Patch(facecolor='salmon', label='Unstable MEI')
    ]
    ax.legend(handles=legend_elements)
    
    return fig,ax

