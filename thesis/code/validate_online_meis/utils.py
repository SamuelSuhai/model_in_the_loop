import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List, Dict, Any


def load_file_from_pickle(file_path):
    with open(file_path,'rb') as f:
        obj = pickle.load(f)
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
        if f"roi_{roi_id}" in mei_id:
            labels.append(f"{mei_id}\n(own)")
        # Check if it's the same cell type
        elif celltypes[i] == celltypes[0]:  # Compare to this ROI's cell type
            labels.append(f"{mei_id}\n(same type)")
        else:
            labels.append(mei_id)
    
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

