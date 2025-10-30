from djimaging.utils.snippet_utils import split_trace_by_reps    
import pandas as pd
import numpy as np
from typing import Tuple, List,Dict,Any,Iterable
import os
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import thesis.code.analysis_closed_loop_experiments.rf_mei_test.utils.plot_utils as pu
import thesis.code.analysis_closed_loop_experiments.rf_mei_test.utils.dj_utils as dj_ut

# TODO: Move to data joint lookup table
MEI_CONTAINER_BASENAME = "mei_data_container.pkl"
MODEL_FRAME_RATE_HZ = 30.0
STIM_BSL_FRAME_NR = 40
STIM_FRAME_NR = 50
CONV_EATS_FRAME_NR = 30




def build_restriction(
        roi_id: int | List[int],
        field_key: Dict[str, str],
        cond2: str | int | None = None,
        exp_num : int | None = None,
        stim_name: str | None= "optstim",

) -> str:
    
    # build field restriction string
    restriction = ""
    for i,(k,v) in enumerate(field_key.items()):
        if i > 0:
            restriction += " AND "
        restriction += f"{k}='{v}'"
    
    if stim_name is not None:
        restriction += f"AND stim_name='{stim_name}'"
    
    if isinstance(roi_id, Iterable):
        roi_id_str = ",".join([str(int(r)) for r in roi_id])
        restriction += f"AND roi_id IN ({roi_id_str})"
    elif isinstance(roi_id, int):
        restriction += f"AND roi_id={roi_id}"
    else:
        raise ValueError("roi_id must be int or list of int.")

    if cond2 is not None:
        restriction += f"AND cond2='{cond2}'" 
    if exp_num is not None:
        restriction += f"AND exp_num={exp_num}"

    return restriction

def fetch_df(
        CascadeTraces,
        CascadeSpikes,
        Offline2OnlineRoiId,
        StimulusPresentationInfo,
        OnlineInferredRFPosition,
        Presentation,
        restriction: str | dict = {},
) -> pd.DataFrame:

    cols = ["roi_id",
        "true_online_roi_id",
        "triggeridx2positions",
        "triggeridx2online_roi_id",
        "spike_prob",
        "pp_trace_t0",
        "pp_trace_dt",
        "triggeridx2is_first_pres_of_stimulus",
        "x_rf","y_rf","triggeridx2stim_type","cond2","triggertimes"]
    compressed_df = pd.DataFrame(((CascadeTraces() *  CascadeSpikes() * Offline2OnlineRoiId() * StimulusPresentationInfo()  *OnlineInferredRFPosition() * Presentation()) & restriction).fetch(*cols,as_dict=True))
    return compressed_df

def add_nonnan_column(df: pd.DataFrame, col_being_masked: str, col_mask_source: str ) -> pd.DataFrame:
    df = df.copy()
    if col_being_masked != col_mask_source:
        # make sure equal length of entries for all
        assert all(df.apply(lambda row: len(row[col_being_masked]) == len(row[col_mask_source]), axis=1)), "Length mismatch between columns."

    df["nonnan_"+col_being_masked] = df.apply(lambda row:row[col_being_masked][~np.isnan(row[col_mask_source])], axis=1)
    return df

def add_spike_times_column(df: pd.DataFrame, trace_times_col: str = "pp_trace_t0",
                           trace_dt_col: str = "pp_trace_dt",
                           spike_prob_col: str = "spike_prob") -> pd.DataFrame:
    df = df.copy()
    df["spike_times"] = df.apply(lambda row: row[trace_times_col] + np.arange(len(row[spike_prob_col])) * row[trace_dt_col], axis=1)
    return df

def explode_snippets(compressed_df: pd.DataFrame,
                     trace_col: str = "nonnan_spike_prob",
                     trace_times_col: str  = "nonnan_spike_times",
                     triggertimes_col: str = "triggertimes") -> pd.DataFrame:
    
    all_exploded_dfs = []
    for i,row in compressed_df.iterrows():

        snippets, snippets_times, snippets_triggertimes, droppedlastrep_flag = split_trace_by_reps(
            trace = row[trace_col],
            times = row[trace_times_col],
            triggertimes = row[triggertimes_col],
            ntrigger_rep = 1
            )
        
        # some checks
        if not snippets.shape[1] == len(row["triggeridx2positions"]) == len(row["triggeridx2stim_type"]) == len(row["triggeridx2online_roi_id"]):
            print(f"Warning: Mismatch in lengths for row {i}: snippets.shape={snippets.shape}, triggeridx2positions={len(row['triggeridx2positions'])}, triggeridx2stim_type={len(row['triggeridx2stim_type'])}, triggeridx2online_roi_id={len(row['triggeridx2online_roi_id'])}")
            raise ValueError("Length mismatch detected.")
        

        # create long format df one snippet per row
        n_snippets = snippets.shape[1]

        snippets_dt = row[trace_times_col][1] - row[trace_times_col][0] if len(row[trace_times_col]) > 1 else np.nan

        exploded_df = pd.DataFrame({
            "snippet": list(snippets.T),
            "snippet_times": list(snippets_times.T),
            "snippet_triggertimes": list(snippets_triggertimes.flatten()),
            "cond2": [row["cond2"]] * n_snippets,
            "roi_id": [row["roi_id"]] * n_snippets,
            "true_online_roi_id": [row["true_online_roi_id"]] * n_snippets,
            "x_rf": [row["x_rf"]] * n_snippets,
            "y_rf": [row["y_rf"]] * n_snippets,
            "positions": [row["triggeridx2positions"][j] for j in range(n_snippets)],
            "stim_type": [row["triggeridx2stim_type"][j] for j in range(n_snippets)],
            "online_roi_id": [row["triggeridx2online_roi_id"][j] for j in range(n_snippets)],
            "snippet_dt": [snippets_dt] * n_snippets,
            "snippet_t0": [snip[0] for snip in snippets_times.T],
            })
        all_exploded_dfs.append(exploded_df)


    # create one large dataframe
    exploded_snippets_df = pd.concat(all_exploded_dfs, ignore_index=True)

    return exploded_snippets_df



def sanity_check1(avg_spike_snippets:pd.DataFrame):

    # check: if true_online_roi_id is the same as online_roi_is, positions[0] = x_rf and positions[1] = y_rf
    for _, row in avg_spike_snippets.iterrows():
        if row["true_online_roi_id"] == row["online_roi_id"]:
                
            assert row["positions"][0] == row["x_rf"]
            assert row["positions"][1] == row["y_rf"]

    # for each true_online_roi_id, there is no double stim_type at the same position
    grouped = avg_spike_snippets.groupby("true_online_roi_id")
    for true_online_roi_id, group in grouped:
        pos_stimtype_set = set()
        for _, row in group.iterrows():
            pos = tuple(row["positions"])
            stim_type = row["stim_type"]
            if (pos, stim_type) in pos_stimtype_set:
                raise ValueError(f"Duplicate stim_type {stim_type} at position {pos} for true_online_roi_id {true_online_roi_id}")
            pos_stimtype_set.add((pos, stim_type))
   


def load_mei_dict(
    StimulusPresentationInfo,
    restriction: str | dict = {},
):  
    
    metadata_files = (StimulusPresentationInfo() & restriction).fetch("metadata_file", as_dict=True)
    
    # assert all metadata files are the same
    metadata_file_set = set([mdf["metadata_file"] for mdf in metadata_files])
    assert len(metadata_file_set) == 1, "All metadata files must be the same."
    metadata_file = metadata_files[0]["metadata_file"]

    # mei_dict file path is in same directory
    parent_dir = os.path.dirname(metadata_file)
    print(parent_dir)
    mei_dirs = [f for f in os.listdir(parent_dir) if "GCL" in f]
    assert len(mei_dirs) == 1, f"There should be exactly one MEI directory but found {len(mei_dirs)}. {mei_dirs=}"
    mei_dir = mei_dirs[0]
    full_mei_dir = os.path.join(parent_dir, mei_dir)
    mei_dict_file = os.path.join(full_mei_dir, MEI_CONTAINER_BASENAME)
    
    with open(mei_dict_file, "rb") as f:
        mei_dict = pickle.load(f)

    return mei_dict


def get_model_true_df(
        avg_spike_snippets: pd.DataFrame,
        mei_container: pd.DataFrame,
) -> pd.DataFrame:
    
    n_rows_before = avg_spike_snippets.shape[0]
    # merge on stim_type_new
    merged_df = avg_spike_snippets.merge(
        mei_container,
        left_on=["stim_type_new"],
        right_on=["stim_type_new"],
        suffixes=("_data", "_model"),
    )
    assert merged_df.shape[0] == n_rows_before, "Number of rows changed after merge. Check stim_type_new matching."

    return merged_df

def add_predicted_response_column(df: pd.DataFrame,
                                 ) -> pd.DataFrame:
    # exaclty one true_online_roi_id in df

    df = df.copy()

    def get_predicted_response(row: pd.Series) -> np.ndarray:
        true_online_roi_id = row["true_online_roi_id"]
        responses_all_readout_idx = row["responses_all_readout_idx"]

        # readout index of true_online_roi_id in model 
        readout_idx = df[df["roi_id_model"] == true_online_roi_id]["readout_idx"]
        unique_readout_idxs = readout_idx.unique()
        assert len(unique_readout_idxs) == 1, f"Expected exactly one unique readout_idx for true_online_roi_id {true_online_roi_id}, but found {unique_readout_idxs}"
        readout_idx = unique_readout_idxs[0]
        print(f"true_online_roi_id: {true_online_roi_id}, readout_idx: {readout_idx}")

        # the predicted snippet is the readout_idx-th column of the responses_all_readout_idx array at a given row
        predicted_response = responses_all_readout_idx[:, readout_idx]
        return predicted_response

    df["predicted_response"] = df.apply(get_predicted_response, axis=1)
    
    return df

def plot_predicted_vs_true_scalar_value(
    df: pd.DataFrame,
   # celltype_df: pd.DataFrame,
    predicted_col: str = "predicted_response_last_10_frames_mean",
    true_col: str = "snippet_last_10_frames_mean",
    xlabel: str = "Predicted mean spike probability [a.u.]",
    ylabel: str = "True mean spike probability [a.u.]",
    ax: plt.Axes | None = None,
    ):
    
    if ax is None:
        fig, ax = plt.subplots()

    import seaborn as sns

    # make true_online_roi_id int if not so
    df = df.copy()
    df["true_online_roi_id"] = df["true_online_roi_id"].astype(int)


    color_palette = sns.color_palette("tab10", n_colors=df["true_online_roi_id"].nunique())
    color_map = {roi_id: color_palette[i] for i, roi_id in enumerate(sorted(df["true_online_roi_id"].unique()))}

    ax = pu.plot_mulit_group_scatter_fits(
        full_df=df,
        x=predicted_col,
        y=true_col,
        hue="true_online_roi_id",
        xlabel=xlabel,
        ylabel=ylabel,
        single_group_fit_kwargs={
            "order":1
        },
        overall_fit_kwargs={
            "order":1,
        },
        color_map=color_map,
        show_legend=False,
        ax=ax
    )

    # create proxy legend
    ax = pu.add_mulitgroup_proxy_legend(ax=ax,
                                        dot_label="RGC repsponse\n(one color = one RGC)",
                                        single_reg_label="RGC linear fit",
                                        full_reg_label="Overall linear fit")
    
    return ax


def add_response_t0_column(df: pd.DataFrame,
                                 ) -> pd.DataFrame:
    df = df.copy()
    def _get_response_t0(row: pd.Series) -> float:
        snippet_t0 = row["snippet_t0"]
        response_t0 = snippet_t0 + (STIM_BSL_FRAME_NR + STIM_FRAME_NR - CONV_EATS_FRAME_NR) / MODEL_FRAME_RATE_HZ
        return response_t0

    df["predicted_response_t0"] = df.apply(_get_response_t0, axis=1)
    return df


def add_predicted_response_timevector_column(df: pd.DataFrame,
                                             t0_column: str = "predicted_response_t0",
                                             new_column: str = "predicted_response_time",
                                             
                                                ) -> pd.DataFrame:
    
    # get general time vecotor from length of predicted response, FRAME RATE MODEL
    df = df.copy()

    def _get_time_vector(row: pd.Series) -> np.ndarray:
        n_timepoints = len(row["predicted_response"])
        dt = 1.0 / MODEL_FRAME_RATE_HZ
        t0 = row[t0_column]
        time_vector = t0 + np.arange(n_timepoints) * dt
        return time_vector 
    df[new_column] = df.apply(_get_time_vector, axis=1)
    
    return df

def add_snippet_timevector_column(df: pd.DataFrame,
                                 t0_column: str = "snippet_t0",
                                 dt_column: str = "snippet_dt",
                                 new_column: str = "snippet_time",
                                 ) -> pd.DataFrame:
    df = df.copy()

    def _get_time_vector(row: pd.Series) -> np.ndarray:
        n_timepoints = len(row["snippet"])
        dt = row[dt_column]
        t0 = row[t0_column]
        time_vector = t0 + np.arange(n_timepoints) * dt
        return time_vector
    df[new_column] = df.apply(_get_time_vector, axis=1)
    return df

def add_stim_onset_time_column(
        df: pd.DataFrame,) -> pd.DataFrame:
    df = df.copy()
    df.apply(lambda row: row["snippet_t0"] + STIM_BSL_FRAME_NR / MODEL_FRAME_RATE_HZ, axis=1)
    return df

def format_data_for_ordered_snippets(df:pd.DataFrame, 
                                     sort_by: str) -> Dict[str, Any]:

    data = {}
    # sort df by
    df = df.copy()
    df = df.sort_values(by=sort_by).reset_index(drop=True)

    # get snippet_trace_list 
    data["snippet_trace_list"] = df["snippet"].tolist()
    data["single_snippet_dt"] = df["snippet_dt"].iloc[0]

    # highlight_bg_times (start, end) and highlight_stim_times in seconds
    data["highlight_bg_times"] = (0, STIM_BSL_FRAME_NR / MODEL_FRAME_RATE_HZ)
    data["highlight_stim_times"] = (STIM_BSL_FRAME_NR / MODEL_FRAME_RATE_HZ,
                            (STIM_BSL_FRAME_NR + STIM_FRAME_NR) / MODEL_FRAME_RATE_HZ)
    
    # stim_type as labels
    data["x_tick_lables"] = df["stim_type_new"].tolist()

    return  data

def add_upsampled_snippet_column(
    df: pd.DataFrame,
    target_fs: float,
    snippet_col: str = "snippet",
    dt_col: str = "snippet_dt",
    interpolation_method: str = "linear",
    new_col: str = "upsampled_snippet"
) -> pd.DataFrame:
    """
    Upsample snippets to specified sampling frequency using interpolation.
    
    Args:
        df: DataFrame containing snippets
        target_fs: Target sampling frequency in Hz
        snippet_col: Name of column containing snippets
        dt_col: Name of column containing time step (dt)
        interpolation_method: Interpolation method ('linear', 'cubic', etc.)
        new_col: Name of new column for upsampled snippets
    
    Returns:
        DataFrame with new column containing upsampled snippets
    """
    df = df.copy()
    
    def upsample_snippet(row: pd.Series) -> np.ndarray:
        original_snippet = row[snippet_col]
        original_dt = row[dt_col]
        original_fs = 1.0 / original_dt
        
        # Original time points
        t_original = np.arange(len(original_snippet)) * original_dt
        
        # New time points
        new_dt = 1.0 / target_fs
        t_new = np.arange(0, t_original[-1] + new_dt, new_dt)
        
        # Interpolate
        return np.interp(t_new, t_original, original_snippet)
    
    df[new_col] = df.apply(upsample_snippet, axis=1)
    # Add the new dt as well
    df[f"{new_col}_dt"] = 1.0 / target_fs
    
    return df

def add_last_n_frames_column(
    df: pd.DataFrame,
    n_frames: int,
    snippet_col: str = "snippet",
    new_col: str = "snippet_last_n_frames"
) -> pd.DataFrame:
    """
    Extract last n frames from snippets into a new column.
    
    Args:
        df: DataFrame containing snippets
        n_frames: Number of frames to extract from end
        snippet_col: Name of column containing snippets
        new_col: Name of new column for extracted frames
    
    Returns:
        DataFrame with new column containing last n frames
    """
    df = df.copy()
    
    def extract_last_n(snippet: np.ndarray) -> np.ndarray:
        if len(snippet) < n_frames:
            raise ValueError(f"Snippet length {len(snippet)} is shorter than requested {n_frames} frames")
        return snippet[-n_frames:]
    
    df[new_col] = df[snippet_col].apply(extract_last_n)
    return df

def fetch_and_plot_snippets_subplots(
        df: pd.DataFrame,
        celltypes_df: pd.DataFrame,
        col1: str = "snippet_last_n_frames", 
        col2: str = "predicted_response",
        optimization_window: Tuple[float,float] | None = (10/30,19/30),
        axes: np.ndarray[plt.Axes] | None = None,
        proxy_legend: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Fetch snippets from DataFrame and plot them in subplots.
    """
    snippets_list1 = df[col1].tolist()
    snippets_list2 = df[col2].tolist()
    stim_types = df["stim_type_new"].tolist()
    roi_id_of_optimized_stimulus = list(map(lambda s: int(s.split("_")[1]), stim_types))
    celltype_of_optimized_stimulus = [
        celltypes_df[celltypes_df["roi_id_data"] == roi_id]["celltype"].iloc[0] 
        for roi_id in roi_id_of_optimized_stimulus
    ]
    print(stim_types)
    print(roi_id_of_optimized_stimulus)
    print(celltype_of_optimized_stimulus)    
    own_celltype = celltypes_df[celltypes_df["roi_id_data"] == df["true_online_roi_id"].iloc[0]]["celltype"].item() 
    true_online_roi_id = df["true_online_roi_id"].iloc[0]

    # set the linestyle and linewidth based on celltype and roi_id
    linestyles = []
    linewidths = []
    for i, snippet in enumerate(snippets_list1):
        
        # if same type - else -- linsetyle
        if celltype_of_optimized_stimulus[i] == own_celltype:
            linestyle = "-"
        else:
            linestyle = "--"
        
        if roi_id_of_optimized_stimulus[i] == true_online_roi_id:
            linewidth = 2.0
        else:
            linewidth = 1.0

        linestyles.append(linestyle)
        linewidths.append(linewidth)
        

    # labels = []
    # for stim in stim_types:
    #     if str(int(true_online_roi_id)) in stim:
    #         lab = f"Target RGC"
    #     elif celltypes_df is not None:
    #         _type = celltypes_df[celltypes_df["roi_id_data"] == roi_id_data]["celltype"].item()
            
    #         lab = f"Same type ({_type})"
    #     else:
    #         lab = f"RGC {int(true_online_roi_id)} (other type)"
    #     labels.append(lab)
    
    
    # Create time vector based on first snippet length
    times = np.arange(len(snippets_list1[0])) / MODEL_FRAME_RATE_HZ
    print(times)
    # Create figure and axes if not provided, make sure axes is array-like
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize = (3.75,3.75),sharex=True)
    else:
        fig = plt.gcf()
        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)
    
    # Plot first set of snippets
    for i, snippet in enumerate(snippets_list1):
        axes[0].plot(times, snippet, color=plt.cm.tab10(i), linestyle=linestyles[i], linewidth=linewidths[i],label=stim_types[i])
    axes[0].text(0.02, 0.98, "Data", 
                 transform=axes[0].transAxes,
                 verticalalignment='top',
                 horizontalalignment='left')
    
    # Plot second set of snippets
    for i, snippet in enumerate(snippets_list2):
        axes[1].plot(times, snippet, color=plt.cm.tab10(i),linestyle=linestyles[i], linewidth=linewidths[i],label=stim_types[i])
    axes[1].text(0.02, 0.98, "Model",
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 horizontalalignment='left')
    # add blue optimization window
    if optimization_window is not None:
        for ax in axes:
            ax.axvspan(optimization_window[0], optimization_window[1], color='blue', alpha=0.1)
    # Set labels and remove spines
    axes[0].set_ylabel("Spike probability [a.u]")
    axes[1].set_ylabel("Spike probability [a.u]")
    axes[1].set_xlabel("Time [s]")
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

    legend_params = {
        "bbox_to_anchor": (0.5, 1),
        "loc": "lower center",
        "ncol": 2,
        "frameon": False,
    }


    if proxy_legend:
        # prxy patch for optimization window
        optimization_proxy = plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.1)

        # add proxy legend thick same cell - same type - thin other type
        same_cell_proxy = plt.Line2D([0], [0], color='grey', linewidth=2.0, linestyle='-')
        same_type_proxy = plt.Line2D([0], [0], color='grey', linewidth=1.0, linestyle='-')
        other_type_proxy = plt.Line2D([0], [0], color='grey', linewidth=1.0, linestyle='--')
        
        axes[0].legend([optimization_proxy,same_cell_proxy, same_type_proxy, other_type_proxy],
                    ["Optimization window","Target RGC", f"Same type (G {own_celltype})", "Other type"],
                    **legend_params)
    else:
        axes[0].legend(**legend_params)
        
    
    return fig, axes


def wrapper_plot_ordered_spike_snippets(
        df: pd.DataFrame,
        sort_by: str,
        axes: np.ndarray[plt.Axes] | None = None,
    ) -> plt.Axes:


    # unique true_online_roi_id
    unique_true_online_roi_ids = df["true_online_roi_id"].unique()
    nrow = 3
    ncol = len(unique_true_online_roi_ids) // nrow + (1 if len(unique_true_online_roi_ids) % nrow > 0 else 0)   
    if axes is None:
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,)
        axes = axes.flatten()
    else:
        fig = axes[0].figure


    for idx,roi in enumerate(unique_true_online_roi_ids):
        df_roi = df[df["true_online_roi_id"] == roi]
        ax = plot_ordered_spike_snippets(
            df=df_roi,
            sort_by=sort_by,
            ax=axes[idx]
        )
    return ax
    


def plot_ordered_spike_snippets(df: pd.DataFrame,
                               sort_by: str,
                               ax: plt.Axes |None= None,
) -> plt.Axes:
    
    if ax is None:
        fig, ax = plt.subplots()

    plot_data_snippets = format_data_for_ordered_snippets(
        df=df,
        sort_by=sort_by,
    )
    df = df.copy()
    df.sort_values(by=sort_by, inplace=True)


    plot_kwargs = {"color":"#1f77b4"}


    ax = pu.plot_ordered_snippets(
        snippet_trace_list=df["predicted_response"].tolist(),
        single_snippet_dt=1.0 / MODEL_FRAME_RATE_HZ,
        ax = ax,
        single_snippet_time_shift= (STIM_BSL_FRAME_NR + STIM_FRAME_NR - CONV_EATS_FRAME_NR) / MODEL_FRAME_RATE_HZ,
    )
    ax = pu.plot_ordered_snippets(
        **plot_data_snippets,
        x_ticks_kwargs={"rotation":45},
        plot_kwargs=plot_kwargs,
        ax=ax,
    )

    ax.set_xlabel("Stimulus type")
    ax.set_ylabel("Spike probability [a.u]")
    return ax

def reduce_to_scalar_value(
        df: pd.DataFrame,
        last_n_frames: int,
        type_of_reduction: str = "mean",
        col_name: str = "last_n_frames",
        new_col_name: str = "mean_last_n_frames",
    ) -> pd.DataFrame:
    df = df.copy()
    if type_of_reduction == "mean":
        func = np.mean
    elif type_of_reduction == "max":
        func = np.max
    elif type_of_reduction == "max-min":
        func = lambda x: np.max(x) - np.min(x)
    else:
        raise ValueError(f"Unknown type_of_reduction: {type_of_reduction}")
    
    
    df[new_col_name] = df[col_name].apply(lambda x: func(x[-last_n_frames:]))
    return df



def restrict_df_to_same_presentation(
        df: pd.DataFrame,) -> pd.DataFrame:
    """
    Subsets df where the optstim was presented at the same location the rf is
    """
    df = df.copy()
    df = df[df.apply(lambda row: row["positions"][0] == row["x_rf"] and row["positions"][1] == row["y_rf"], axis=1)]
    return df


def modify_data_stim_type(
        df: pd.DataFrame,
        old_col: str = "stim_type",
        new_col: str = "stim_type_new",
    ):
    df = df.copy()

    def _stim_func(x: str):
        if "MEI" in x:
            return x
        elif "DEI" in x:
            # remove last 3
            last_3 = x[-3:]
            new = x[:-3] + "_" + last_3[0]
            return new
        else:
            raise ValueError(f"Unknown stim_type_data: {x}")

    df[new_col] = df[old_col].apply(_stim_func)
    return df
def fetch_celltype_df(CelltypeAssignment,
                      Offline2OnlineRoiId,
                      Presentation,
                      field_key: Dict[str, str],
                      ) -> pd.DataFrame:
    """
    Fetch celltype dataframe for given field_key and stim_name
    """
    restriction = build_restriction(
        roi_id = Offline2OnlineRoiId().fetch("roi_id"),
        field_key=field_key,
        stim_name=None,
    )
    print(restriction)
    celltype_df = pd.DataFrame((CelltypeAssignment() * Offline2OnlineRoiId() * Presentation() & restriction).fetch(as_dict=True))
    
    celltype_df =celltype_df[["celltype", "roi_id"]]
    celltype_df.rename(columns={"roi_id":"roi_id_data"}, inplace=True)

    return celltype_df

def add_celltype_column(
        df: pd.DataFrame,
        celltype_df: pd.DataFrame,
        offline_roi_id_col: str = "roi_id_data",
        celltype_col: str = "celltype",) -> pd.DataFrame:
    df = df.copy()
    merged_df = df.merge(
        celltype_df[[offline_roi_id_col, celltype_col]],
        left_on=offline_roi_id_col,
        right_on=offline_roi_id_col,
        how='left',
        suffixes=('', '_celltype')
    )
    return merged_df




def add_new_stim_type_naming_convention(
        df: pd.DataFrame,
        old_col: str = "mei_id",
        new_col: str = "stim_type_new",
        inplace = True,
    ):
    """
    Add stim_type_new column based on mei_id column.
    for each roi_id,
    if there is only one seed -> roi_<roi_id>_type_MEI
    if multiple seeds -> roi_<roi_id>_type_DEI_<first_letter_of_seed>
    """
    if not inplace:
        df = df.copy()
    
    # Group by roi_id to count unique seeds per ROI
    roi_groups = df.groupby('roi_id')
    
    # For each ROI, determine if it's MEI or DEI based on number of seeds
    for roi_id, group in roi_groups:
        # Extract unique seeds from mei_id column
        seeds = group[old_col].apply(lambda x: x.split('_')[-1]).unique()
        
        if len(seeds) == 1:
            # Single seed -> MEI
            df.loc[group.index, new_col] = f"roi_{roi_id}_type_MEI"
        else:
            # Multiple seeds -> DEI with seed initial
            for idx in group.index:
                seed = df.loc[idx, old_col].split('_')[-1]
                df.loc[idx, new_col] = f"roi_{roi_id}_type_DEI_{seed[0]}"
    
    if inplace:
        return None
    return df  # Return the modified DataFrame if not inplace




############################################ Wrappers ##############


def wrapper_fetch_complete_field_df(
        field_key: Dict[str, Any],
        Offline2OnlineRoiId,
        CascadeTraces,
        CascadeSpikes,
        StimulusPresentationInfo,
        OnlineInferredRFPosition,
        Presentation,
) -> Tuple[pd.DataFrame,List[int]]:
    


    roi_ids = (Offline2OnlineRoiId() & (Presentation() & field_key & "stim_name='optstim'")).fetch("roi_id")
    print("rois:",roi_ids)
    restriction = build_restriction(
    roi_id=roi_ids,
    field_key=field_key,   
    )

    compressed_df = fetch_df(
        CascadeTraces,
        CascadeSpikes,
        Offline2OnlineRoiId,
        StimulusPresentationInfo,
        OnlineInferredRFPosition,
        Presentation,
    restriction=restriction
    )
    compressed_df = add_nonnan_column(
        df = compressed_df,
        col_being_masked= "spike_prob",
        col_mask_source = "spike_prob",
    )

    compressed_df = add_spike_times_column(
        df = compressed_df,
    )

    compressed_df = add_nonnan_column(
        df = compressed_df,
        col_being_masked= "spike_prob",
        col_mask_source = "spike_prob",
    )

    compressed_df = add_nonnan_column(
        df = compressed_df,
        col_being_masked= "spike_times",
        col_mask_source = "spike_prob",
    )
    exploded_snippets_df = explode_snippets(
    compressed_df = compressed_df,)

    avg_spike_snippets = dj_ut.average_df_over_colvalues(
    df=exploded_snippets_df,
        cols_to_gb = ["stim_type","roi_id", "online_roi_id"],
        cols_to_average = ["snippet","snippet_triggertimes",
                           "positions","x_rf","y_rf",
                           "true_online_roi_id","snippet_dt", "snippet_t0"],


    )

    # add new stim type col
    avg_spike_snippets = modify_data_stim_type(avg_spike_snippets)

    sanity_check1(avg_spike_snippets)

    avg_spike_snippets_same_location = restrict_df_to_same_presentation(
    df=avg_spike_snippets,
    )

    
    mei_container = load_mei_dict(StimulusPresentationInfo, f"stim_name='optstim' AND field='{field_key['field']}'")
    mei_container = add_new_stim_type_naming_convention(mei_container, inplace=False)



    model_true_df = get_model_true_df(
    avg_spike_snippets=avg_spike_snippets_same_location,
    mei_container=mei_container,
    )




    # print columns
    print(model_true_df.columns)

    model_true_df = add_predicted_response_column(model_true_df)

    model_true_df = add_response_t0_column(model_true_df)
    
    model_true_df = add_predicted_response_timevector_column(model_true_df)
    
    model_true_df = add_snippet_timevector_column(model_true_df)
    model_true_df = add_stim_onset_time_column(model_true_df)

        # Upsample snippets to 100 Hz
    model_true_df = add_upsampled_snippet_column(
        df=model_true_df,
        target_fs=30.0,
        interpolation_method='linear'
    )

    # Extract last 20 frames
    model_true_df = add_last_n_frames_column(
        df=model_true_df,
        n_frames=20
    )


    model_true_df = reduce_to_scalar_value(
    df = model_true_df,
    col_name = "snippet_last_n_frames",
    last_n_frames = 10,
    type_of_reduction="mean",
    new_col_name="snippet_last_10_frames_mean"
    )

    # same for response
    model_true_df = reduce_to_scalar_value(
        df = model_true_df,
        col_name = "predicted_response",
        last_n_frames = 10,
        type_of_reduction="mean",
        new_col_name="predicted_response_last_10_frames_mean"
    )


    ## add max in entire window
    model_true_df = reduce_to_scalar_value(
        df = model_true_df,
        col_name = "snippet_last_n_frames",
        last_n_frames = 20,
        type_of_reduction="max",
        new_col_name="snippet_last_20_frames_max"
    )

    # also for predicted response
    model_true_df = reduce_to_scalar_value(
        df = model_true_df,
        col_name = "predicted_response",
        last_n_frames = 20,
        type_of_reduction="max",
        new_col_name="predicted_response_last_20_frames_max"
    )

    ## add min-max in entire window
    model_true_df = reduce_to_scalar_value(
        df = model_true_df,
        col_name = "snippet_last_n_frames",
        last_n_frames = 20,
        type_of_reduction="max-min",
        new_col_name="snippet_last_20_frames_max_min"
    )
    # also for predicted response
    model_true_df = reduce_to_scalar_value(
        df = model_true_df,
        col_name = "predicted_response",
        last_n_frames = 20,
        type_of_reduction="max-min",
        new_col_name="predicted_response_last_20_frames_max_min"
    )



    # take out some columns to save memory
    cols_to_drop = ["mei","temporal_kernels","spatial_kernels"]
    model_true_df = model_true_df.drop(columns=cols_to_drop)
    return model_true_df,roi_ids