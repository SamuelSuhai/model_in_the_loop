


from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import numpy as np
import pandas as pd

import numpy as np



####################################################################################################### DATA MANIPULATION AND COMPUTATION ######################################################
def add_field_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds inplace a field_id column to the dataframe that uniquely identifies a field (WITHOUT COND1)
    """

    # List the columns that uniquely define a field EXCLUDING COND1
    field_id_cols = ['experimenter', 'date', 'exp_num', 'raw_id', 'field', 'region']

    assert all(col in df.columns for col in field_id_cols), "One or more required columns are missing in the DataFrame"

    # Convert date to string if needed
    df['date'] = df['date'].astype(str)

    # Create the field_id column
    df['field_id'] = df[field_id_cols].astype(str).agg('_'.join, axis=1)

    # Display the result
    print("Nr unique field ids:", len(df['field_id'].unique()))

# define some functions to call 
def mean_mb_qidx(group):
    return group['mb_qidx'].mean()

def mean_chrip_qidx(group):
    return group['chirp_qidx'].mean()


def nr_passing_or(group,mb_thresh = 0.6,chirp_thresh= 0.35):
    return ((group['mb_qidx'] >= mb_thresh) | (group['chirp_qidx'] >= chirp_thresh)).sum()


def frac_passing_or(group,mb_thresh = 0.6,chirp_thresh = 0.35):
    return ((group['mb_qidx'] >= mb_thresh) | (group['chirp_qidx'] >= chirp_thresh)).mean()


def frac_passing_and(group,mb_thresh = 0.7,chirp_thresh = 0.7):
    return ((group['mb_qidx'] >= mb_thresh) & (group['chirp_qidx'] >= chirp_thresh)).mean()



def apply_func_and_pivot(quality_df, func):

    gb = quality_df.groupby(
        ["field_id","cond1"]
    ).apply(func,).reset_index()


    gb.columns = ["field_id", "cond1", "score"]

    # 2. Pivot the data to have field_id as index and cond1 as columns
    quality_pivot = gb.pivot(index="field_id", columns="cond1", values="score")

    return quality_pivot


def find_row_with_highest_correl(row: np.ndarray,arr:np.ndarray) -> tuple[int,float]:
    """
    returns the row in arr with the highest correlation to row
    """
    assert len(row.shape) == 1, "row must be 1D"
    assert len(arr.shape) == 2, "arr must be 2D"
    correlations = np.corrcoef(row, arr)[0, 1:]
    max_corr_index = np.argmax(correlations).astype(int)
    max_corr_value = correlations[max_corr_index].astype(float)
    return max_corr_index, max_corr_value

def find_row_closest(row: np.ndarray,arr:np.ndarray):
    """
    fiven row as positions x,y and arr as array of positions [[x1,y1],[x2,y2],...], returns row in arr closest to row
    """
    assert len(row.shape) == 1, "row must be 1D"
    assert len(arr.shape) == 2, "arr must be 2D"
    assert arr.shape[1] == 2, "arr must have 2 columns"
    distances = np.linalg.norm(arr - row, axis=1)
    closest_index = np.argmin(distances).astype(int)
    closest_value = distances[closest_index].astype(float)
    return closest_index, closest_value



def find_roi_partner(template_field_traces: pd.DataFrame,match_to_these_field_traces: pd.DataFrame,corr_thresh=0.7,distance_thresh=10):
    """
    Maps the rois in template_field_traces to the rois in match_to_these_field_traces based on highest correlation of their traces and the distance or their roi positions.
    Returns a map from each roi_id in template_field_traces to the roi_id in match_to_these_field_traces

    """
    # requires cols: x_pos, y_pos, trace
    assert all(col in template_field_traces.columns for col in ['x_pos', 'y_pos', 'trace','roi_id']), "template_field_traces must contain x_pos, y_pos, trace columns"
    assert template_field_traces['field_id'].unique().size == 1, "template_field_traces must contain only one field_id"

    mapping = {}
    corrs = {}
    all_dists = {}


    match_to_these_trace_array = np.stack(match_to_these_field_traces['trace'].to_list(),axis = 0)
    match_to_these_positions_array = match_to_these_field_traces[['x_pos','y_pos']].to_numpy()

    for i,roi_data in template_field_traces.iterrows():
        roi_id = roi_data['roi_id']
        roi_trace = roi_data['trace']
        roi_x = roi_data['x_pos']
        roi_y = roi_data['y_pos']

        # compare and see what roi_ids in match_to_these_field_traces could match
        corr_idx, corr = find_row_with_highest_correl(roi_trace,match_to_these_trace_array)

        # get distance to roi_id
        distance = np.linalg.norm(match_to_these_positions_array[corr_idx] - np.array([roi_x,roi_y]))
        
        corrs[roi_id] = corr
        all_dists[roi_id] = distance

        # store mapping if pass criteria
        if corr > corr_thresh and distance < distance_thresh:
            taget_roi = int(match_to_these_field_traces.iloc[corr_idx]['roi_id'])

            # check if already assingled
            if taget_roi in mapping.values():
                print(f"Warning: roi_id {roi_id} target roi {taget_roi} already assigned to another roi, skipping...")
                continue
            mapping[roi_id] = int(match_to_these_field_traces.iloc[corr_idx]['roi_id'])

        else:
            mapping[roi_id] = None
            print(f"Warning: roi_id {roi_id} could not be assigned, max corr {corr:.2f}, distance {distance:.2f}")
    return mapping, corrs, all_dists


def find_field_roi_id_partner(field_traces: pd.DataFrame,base_col_val = 'cl',find_corresponding_in_this_col_val = 'n1',field_id_col = 'field_id',cond1_col = 'cond1'):
    """
    Find the roi partner for traces from one field.
    base_col_val is the cond1 value for which we want to find partners in rows that have the cond1 value find_corresponding_in_this_col_val
    """
    assert field_traces[field_id_col].unique().size == 1, "field_traces must contain only one field_id"
    base_field_traces = field_traces[field_traces[cond1_col] == base_col_val]
    match_to_field_traces = field_traces[field_traces[cond1_col] == find_corresponding_in_this_col_val]

    mapping, corrs, all_dists = find_roi_partner(base_field_traces,match_to_field_traces)
    return mapping, corrs,all_dists

    

def cond1_to_cond1_celltype(field_celltypes_df, traces_df, online_col_val = 'cl',offline_col_val = 'n1',field_id_col = 'field_id',cond1_col = 'cond1',stim_name = 'gChirp', quality_df = None,reference_col_val = 'n1'):
    """
    Given a dataframe with celltypes for online rois, find the corresponding offline roi celltypes if reference_col_val is equal to online_col_val, otherwise 
    maps the offline rois to the online ones
    """
    if reference_col_val not in (online_col_val,offline_col_val):
        raise ValueError("reference_col_val must be either online_col_val or offline_col_val")

    all_field_ids = field_celltypes_df[field_id_col].unique()
    assert len(all_field_ids) == 1, "field_celltypes_df must contain only one field_id"
    field_id = all_field_ids[0]

    stim_mask = traces_df['stim_name'] == stim_name
    field_traces = traces_df[(traces_df[field_id_col] == field_id) & stim_mask]
    
    mapping, online_offline_corrs,all_dists = find_field_roi_id_partner(field_traces,base_col_val= online_col_val,
                                                                            find_corresponding_in_this_col_val = offline_col_val,
                                                                            field_id_col = field_id_col,cond1_col= cond1_col)
    
    # first get onlz online celltypes 
    online_celltypes = field_celltypes_df[field_celltypes_df['cond1'] == online_col_val][['roi_id','celltype','max_confidence',field_id_col]].rename(columns={'roi_id':'online_roi_id','celltype':'online_cell_type','max_confidence':'online_max_confidence'})
    offline_celltypes = field_celltypes_df[field_celltypes_df['cond1'] == offline_col_val][['roi_id','celltype','max_confidence',field_id_col]].rename(columns={'roi_id':'offline_roi_id','celltype':'offline_cell_type','max_confidence':'offline_max_confidence'})
    
    # reset indices
    online_celltypes = online_celltypes.reset_index(drop=True)
    offline_celltypes = offline_celltypes.reset_index(drop=True)


    # map the roi_ids to each tother
    if reference_col_val == online_col_val:
        out_df = online_celltypes
        
        # add offline data 
        out_df["offline_roi_id"] = out_df['online_roi_id'].map(mapping)
        out_df["correlation"] = out_df['online_roi_id'].map(online_offline_corrs)
        out_df["distance"] = out_df['online_roi_id'].map(all_dists)
        out_df = out_df.merge(offline_celltypes[['offline_roi_id','offline_cell_type','offline_max_confidence']],on='offline_roi_id',how='left')

    else:
        out_df = offline_celltypes

        # add online data
        out_df["online_roi_id"] = out_df['offline_roi_id'].map(mapping)
        out_df["correlation"] = out_df['offline_roi_id'].map(online_offline_corrs)
        out_df["distance"] = out_df['offline_roi_id'].map(all_dists)
        out_df = out_df.merge(online_celltypes[['online_roi_id','online_cell_type','online_max_confidence']],on='online_roi_id',how='left')


    # find quality indices for online rois
    if quality_df is not None:
        field_quality_online = quality_df[(quality_df[field_id_col] == field_id) & (quality_df['cond1'] == online_col_val)]
        field_quality_offline = quality_df[(quality_df[field_id_col] == field_id) & (quality_df['cond1'] == offline_col_val)]

        # add qidxs
        if reference_col_val == online_col_val:
            out_df = out_df.merge(field_quality_online[['roi_id','mb_qidx','chirp_qidx']].rename(columns={'roi_id':'online_roi_id','mb_qidx':'online_mb_qidx','chirp_qidx':'online_chirp_qidx'}),on='online_roi_id',how='left')
        else:
            out_df = out_df.merge(field_quality_offline[['roi_id','mb_qidx','chirp_qidx']].rename(columns={'roi_id':'offline_roi_id','mb_qidx':'offline_mb_qidx','chirp_qidx':'offline_chirp_qidx'}),on='offline_roi_id',how='left')


    return out_df

def get_all_cond1_to_cond1_celltype(all_celltypes_df,
                                     traces_df, 
                                     quality_df = None, online_col_val = 'cl',offline_col_val = 'n1',field_id_col = 'field_id',cond1_col = 'cond1', reference_col_val = 'n1'):
    """
    Loops over field_ids in the entire celltype df and then calls online2offline_celltype and concatenates the results to one df with some sanity checks at end.
    """
    out_dfs = []
    for i,field_id in enumerate(all_celltypes_df[field_id_col].unique()):
        field_celltypes_df = all_celltypes_df[all_celltypes_df[field_id_col] == field_id]
        field_out_df = cond1_to_cond1_celltype(field_celltypes_df,
                                               traces_df = traces_df,
                                               quality_df = quality_df,
                                               online_col_val = online_col_val,
                                               offline_col_val = offline_col_val,
                                               field_id_col = field_id_col,
                                               cond1_col = cond1_col,
                                               reference_col_val= reference_col_val)
        out_dfs.append(field_out_df)
    out_df = pd.concat(out_dfs,ignore_index=True)
    
    # sanity checks: no missing rois in total
    assert sum(all_celltypes_df['cond1'] == reference_col_val) == len(out_df)

    # same nr of rois as for con1 value for each field 
    assert all(out_df.groupby(field_id_col).size() == all_celltypes_df[all_celltypes_df['cond1'] == reference_col_val].groupby(field_id_col).size())


    return out_df


   




def prepare_celltype_data(df, offline_col='offline_cell_type', online_col='online_cell_type',
                          max_type=32, chirp_qidx_threshold=None, mb_qidx_threshold=None,
                          chirp_percentile=None, mb_percentile=None):
    """
    Prepare cell type data for confusion matrix analysis, with optional quality filtering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cell type and quality data
    offline_col : str
        Column name for offline cell types
    online_col : str
        Column name for online cell types
    max_type : int
        Maximum cell type to show individually (higher values grouped)
    chirp_qidx_threshold : float, optional
        Minimum chirp quality index threshold (cells below will be filtered out)
    mb_qidx_threshold : float, optional
        Minimum moving bar quality index threshold (cells below will be filtered out)
    chirp_percentile : float, optional
        Percentile for automatic chirp threshold selection (between 0 and 100)
    mb_percentile : float, optional
        Percentile for automatic moving bar threshold selection (between 0 and 100)
        
    Returns:
    --------
    df_prepared : pandas.DataFrame
        Filtered and prepared DataFrame with grouped cell types
    applied_thresholds : dict
        Dictionary with the applied threshold values
    """
    import pandas as pd
    import numpy as np
    
    # Assert that user doesn't specify both percentile and threshold for the same metric
    if chirp_percentile is not None and chirp_qidx_threshold is not None:
        raise ValueError("Cannot specify both chirp_percentile and chirp_qidx_threshold")
        
    if mb_percentile is not None and mb_qidx_threshold is not None:
        raise ValueError("Cannot specify both mb_percentile and mb_qidx_threshold")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Calculate and apply percentile-based thresholds if specified
    applied_thresholds = {}
    
    if chirp_percentile is not None:
        if 'chirp_qidx' not in df_copy.columns:
            raise ValueError("Column 'chirp_qidx' not found but chirp_percentile was specified")
        chirp_qidx_threshold = np.percentile(df_copy['chirp_qidx'], chirp_percentile)
        applied_thresholds['chirp_qidx'] = chirp_qidx_threshold
    
    if mb_percentile is not None:
        if 'mb_qidx' not in df_copy.columns:
            raise ValueError("Column 'mb_qidx' not found but mb_percentile was specified")
        mb_qidx_threshold = np.percentile(df_copy['mb_qidx'], mb_percentile)
        applied_thresholds['mb_qidx'] = mb_qidx_threshold
    
    # Apply quality thresholds if specified
    if chirp_qidx_threshold is not None:
        if 'chirp_qidx' not in df_copy.columns:
            raise ValueError("Column 'chirp_qidx' not found but chirp_qidx_threshold was specified")
        df_copy = df_copy[df_copy['chirp_qidx'] >= chirp_qidx_threshold]
        if 'chirp_qidx' not in applied_thresholds:
            applied_thresholds['chirp_qidx'] = chirp_qidx_threshold
    
    if mb_qidx_threshold is not None:
        if 'mb_qidx' not in df_copy.columns:
            raise ValueError("Column 'mb_qidx' not found but mb_qidx_threshold was specified")
        df_copy = df_copy[df_copy['mb_qidx'] >= mb_qidx_threshold]
        if 'mb_qidx' not in applied_thresholds:
            applied_thresholds['mb_qidx'] = mb_qidx_threshold
    
    # Group cell types > max_type into a single category
    df_copy[offline_col + '_grouped'] = df_copy[offline_col].apply(
        lambda x: x if x <= max_type else max_type + 1)
    df_copy[online_col + '_grouped'] = df_copy[online_col].apply(
        lambda x: x if x <= max_type else max_type + 1)
    
    return df_copy, applied_thresholds


def create_confusion_matrices(df, offline_col_grouped, online_col_grouped, max_type=32):
    """
    Create confusion matrices for cell type classification analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Prepared DataFrame containing grouped cell type columns
    offline_col_grouped : str
        Column name for grouped offline cell types
    online_col_grouped : str
        Column name for grouped online cell types
    max_type : int
        Maximum cell type shown individually
        
    Returns:
    --------
    confusion_counts : pandas.DataFrame
        Matrix of raw counts
    confusion_probs : pandas.DataFrame
        Matrix of row-normalized probabilities
    """
    import pandas as pd
    
    # Create the contingency table
    confusion_counts = pd.crosstab(
        df[offline_col_grouped], 
        df[online_col_grouped],
        rownames=['Offline Type'], 
        colnames=['Online Type']
    )
    
    # Ensure all values from 1 to max_type+1 are present in both axes
    all_values = list(range(1, max_type + 2))
    for val in all_values:
        if val not in confusion_counts.index:
            confusion_counts.loc[val] = 0
        if val not in confusion_counts.columns:
            confusion_counts[val] = 0
    
    # Sort both axes
    confusion_counts = confusion_counts.sort_index().sort_index(axis=1)
    
    # Create row-normalized matrix (probabilities)
    confusion_probs = confusion_counts.div(confusion_counts.sum(axis=1), axis=0).fillna(0)
    
    return confusion_counts, confusion_probs






















############################################################################# Plotting ##########################################################


def plot_confusion_matrix(matrix, is_counts=True, max_type=32, figsize=(20, 16), 
                          cmap='Blues', annot_fmt='.2f', applied_thresholds=None):
    """
    Plot a single confusion matrix (either counts or probabilities).
    
    Parameters:
    -----------
    matrix : pandas.DataFrame
        Matrix to plot (either counts or probabilities)
    is_counts : bool
        Whether the matrix contains counts (True) or probabilities (False)
    max_type : int
        Maximum cell type shown individually
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for the heatmaps
    annot_fmt : str
        Format string for annotations
    applied_thresholds : dict, optional
        Dictionary with applied quality thresholds to show in title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Format depends on whether this is counts or probabilities
    fmt = 'd' if is_counts else annot_fmt
    label = 'Count' if is_counts else 'Probability'
    
    # Plot the matrix
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, 
                ax=ax, cbar_kws={'label': label})
    
    # Update labels
    ax.set_xlabel('Online Cell Type')
    ax.set_ylabel('Offline Cell Type')
    
    # Set title
    if is_counts:
        title = 'Cell Type Classification Confusion Matrix (Counts)'
        total_cells = matrix.sum().sum()
        title += f'\nTotal: {int(total_cells)} cells'
    else:
        title = 'Cell Type Classification Conditional Probability\n(P(online type | offline type))'
    
    if applied_thresholds:
        threshold_str = ', '.join([f"{k}: {v:.3f}" for k, v in applied_thresholds.items()])
        title += f'\nQuality thresholds: {threshold_str}'
    
    ax.set_title(title)
    
    # Rename the last tick label to ">32"
    xticks = list(range(1, max_type + 1)) + [f">{max_type}"]
    yticks = list(range(1, max_type + 1)) + [f">{max_type}"]
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    
    plt.tight_layout()
    
    return fig, ax


def plot_celltype_confusion_matrix(df, offline_col='offline_cell_type', online_col='online_cell_type', 
                                  max_type=32, figsize=(20, 16), cmap='Blues', annot_fmt='.2f',
                                  plot_counts=True, save_path=None,
                                  chirp_qidx_threshold=None, mb_qidx_threshold=None,
                                  chirp_percentile=None, mb_percentile=None):
    """
    Plot a confusion matrix for cell type classifications with optional quality filtering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the classification data
    offline_col : str
        Column name for offline cell types (rows)
    online_col : str
        Column name for online cell types (columns)
    max_type : int
        Maximum cell type to show individually (higher values grouped)
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for the heatmaps
    annot_fmt : str
        Format string for annotations
    plot_counts : bool
        Whether to plot counts (True) or probabilities (False)
    save_path : str, optional
        Path to save the figure
    chirp_qidx_threshold : float, optional
        Minimum chirp quality index threshold
    mb_qidx_threshold : float, optional
        Minimum moving bar quality index threshold
    chirp_percentile : float, optional
        Percentile for automatic chirp threshold selection (between 0 and 100)
    mb_percentile : float, optional
        Percentile for automatic moving bar threshold selection (between 0 and 100)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Step 1: Prepare the data
    df_prepared, applied_thresholds = prepare_celltype_data(
        df, offline_col, online_col, max_type,
        chirp_qidx_threshold, mb_qidx_threshold,
        chirp_percentile, mb_percentile
    )
    
    # Step 2: Create the confusion matrices
    confusion_counts, confusion_probs = create_confusion_matrices(
        df_prepared, 
        offline_col + '_grouped', 
        online_col + '_grouped',
        max_type
    )
    
    # Step 3: Plot the selected matrix type
    matrix = confusion_counts if plot_counts else confusion_probs
    fig, ax = plot_confusion_matrix(
        matrix,
        is_counts=plot_counts,
        max_type=max_type, 
        figsize=figsize, 
        cmap=cmap, 
        annot_fmt=annot_fmt,
        applied_thresholds=applied_thresholds
    )
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

# Example usage:
# fig, ax = plot_celltype_confusion_matrix(
#     online2offline_celltype_df,
#     plot_counts=True,              # Plot counts (False for probabilities)
#     chirp_qidx_threshold=0.3,      # Only include cells with chirp quality index >= 0.3
#     mb_percentile=25,              # Automatically set mb threshold to exclude bottom 25%
#     save_path='celltype_confusion_matrix.png'
# )


def plot_online_offline_field_scalar(online_values,offline_values,title = "",ax_tick_labels=None):
    pass