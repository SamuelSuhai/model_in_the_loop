


from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



####################################################################################################### DATA MANIPULATION AND COMPUTATION ######################################################
def add_field_id_col(df: pd.DataFrame) -> None:
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


def fetch_traces_df(dj_table_holder):
    from model_in_the_loop.core.dj_schemas.full_mitl_schema import RelativeRoiLocationWrtField, RelativeRoiLocation
    RelativeRoiLocationWrtField().populate()
    traces_df = (dj_table_holder("Traces")() * RelativeRoiLocationWrtField().proj(**{"x_pos": "relx_wrt_field","y_pos":"rely_wrt_field",})).fetch(as_dict=True)
    traces_df = pd.DataFrame(traces_df)
    return traces_df


def find_roi_partner_highest_correl(online_traces: np.ndarray, offline_traces: np.ndarray):
    """
    Given two arrays of traces, finds for each trace in online_traces the index of the trace in offline_traces with highest correlation
    Returns a list of indices and a list of correlations
    """
    assert len(online_traces.shape) == 2, "online_traces must be 2D"
    assert len(offline_traces.shape) == 2, "offline_traces must be 2D"
    assert online_traces.shape[1] == offline_traces.shape[1], "online_traces and offline_traces must have the same number of columns"

    indices = []
    corrs = []
    for i in range(online_traces.shape[0]):
        idx, corr = find_row_with_highest_correl(online_traces[i], offline_traces)
        indices.append(idx)
        corrs.append(corr)
    return indices, corrs


def find_roi_partner(online_traces: pd.DataFrame,offline_traces: pd.DataFrame,corr_thresh=0.7,distance_thresh=10,offline_to_online = True):
    """
    Maps the rois based on highest correlation of their traces and the distance or their roi positions.
    Returns a map from each roi_id in template_field_traces to the roi_id in match_to_these_field_traces

    """
    # requires cols: x_pos, y_pos, trace
    assert all(col in online_traces.columns for col in ['x_pos', 'y_pos', 'trace','roi_id']), "template_field_traces must contain x_pos, y_pos, trace columns"
    assert online_traces['field_id'].unique().size == 1, "template_field_traces must contain only one field_id"
    assert all(col in offline_traces.columns for col in ['x_pos', 'y_pos', 'trace','roi_id']), "template_field_traces must contain x_pos, y_pos, trace columns"
    assert offline_traces['field_id'].unique().size == 1, "template_field_traces must contain only one field_id"


    mapping = {}
    corrs = {}
    all_dists = {}


    offline_trace_array = np.stack(offline_traces['trace'].to_list(),axis = 0)
    match_to_these_positions_array = offline_traces[['x_pos','y_pos']].to_numpy()

    for i,roi_data in online_traces.iterrows():
        online_roi_id = roi_data['roi_id']
        online_roi_trace = roi_data['trace']
        online_roi_x = roi_data['x_pos']
        online_roi_y = roi_data['y_pos']

        # compare and see what roi_ids in match_to_these_field_traces could match
        target_idx, corr = find_row_with_highest_correl(online_roi_trace,offline_trace_array)

        # # distace
        # target_idx, distance = find_row_closest(np.array([roi_x,roi_y]),match_to_these_positions_array)
        # corr = np.corrcoef(roi_trace, match_to_these_trace_array[target_idx])[0, 1]
        
        # get distance to roi_id
        distance = np.linalg.norm(match_to_these_positions_array[target_idx] - np.array([online_roi_x,online_roi_y]))


        # store mapping if pass criteria
        if corr > corr_thresh and distance < distance_thresh:
            offline_roi = int(offline_traces.iloc[target_idx]['roi_id'])

            # check if already assingled
            already_have_offline_roi = offline_roi in mapping.keys() if offline_to_online else offline_roi in mapping.values() 

            if already_have_offline_roi:
                print(f"warning, already have mapping for offline roi {offline_roi}, skipping assignment for online roi {online_roi_id} with corr {corr:.2f} and distance {distance:.2f}")
                continue
            
            key = int(offline_roi) if offline_to_online else int(online_roi_id)
            roi_id_value = int(online_roi_id) if offline_to_online else int(offline_roi)
            mapping[key] = roi_id_value
            
            corrs[key] = corr
            all_dists[key] = distance

        else:
            print(f"Warning: onlnie roi_id {online_roi_id} could not be assigned, max corr {corr:.2f}, distance {distance:.2f}")

    # if we mapp offline to online fill up any missing dict entries with None
    if offline_to_online:
        for offline_roi in offline_traces['roi_id']:
            if int(offline_roi) not in mapping.keys():
                mapping[int(offline_roi)] = None
                corrs[int(offline_roi)] = None
                all_dists[int(offline_roi)] = None
        
    return mapping, corrs, all_dists


def find_field_roi_id_partner(field_traces: pd.DataFrame,
                              online_col_val = 'cl', 
                              offline_col_val = 'n1',
                              field_id_col = 'field_id',
                              cond1_col = 'cond1',
                              offline_to_online = True,):
    """
    Find the roi partner for traces from one field.
    base_col_val is the cond1 value for which we want to find partners in rows that have the cond1 value find_corresponding_in_this_col_val
    """
    assert field_traces[field_id_col].unique().size == 1, "field_traces must contain only one field_id"
    online_trace_df = field_traces[field_traces[cond1_col] == online_col_val]
    offline_trace_df = field_traces[field_traces[cond1_col] == offline_col_val]

    mapping, corrs, all_dists = find_roi_partner(
        online_traces=online_trace_df,
        offline_traces=offline_trace_df,
        offline_to_online=offline_to_online
    )
    return mapping, corrs,all_dists

    

def cond1_to_cond1_celltype(field_celltypes_df, 
                            traces_df, 
                            online_col_val = 'cl',
                            offline_col_val = 'n1',
                            field_id_col = 'field_id',
                            cond1_col = 'cond1',
                            stim_name = 'gChirp',
                            quality_df = None,
                            offline_to_online = True,):
    """
    Given a dataframe with celltypes for online rois, find the corresponding offline roi celltypes if reference_col_val is equal to online_col_val, otherwise 
    maps the offline rois to the online ones
    """

    all_field_ids = field_celltypes_df[field_id_col].unique()
    assert len(all_field_ids) == 1, "field_celltypes_df must contain only one field_id"
    field_id = all_field_ids[0]

    stim_mask = traces_df['stim_name'] == stim_name
    field_traces = traces_df[(traces_df[field_id_col] == field_id) & stim_mask]
    
    mapping, corrs,all_dists = find_field_roi_id_partner(
        field_traces=field_traces,
        online_col_val=online_col_val,
        offline_col_val=offline_col_val,
        field_id_col=field_id_col,
        cond1_col=cond1_col,
        offline_to_online=offline_to_online
    )
    
    # first get onlz online celltypes 
    online_celltypes = field_celltypes_df[field_celltypes_df['cond1'] == online_col_val][['roi_id','celltype','max_confidence',field_id_col]].rename(columns={'roi_id':'online_roi_id','celltype':'online_cell_type','max_confidence':'online_max_confidence'})
    offline_celltypes = field_celltypes_df[field_celltypes_df['cond1'] == offline_col_val][['roi_id','celltype','max_confidence',field_id_col]].rename(columns={'roi_id':'offline_roi_id','celltype':'offline_cell_type','max_confidence':'offline_max_confidence'})
    
    # reset indices
    online_celltypes = online_celltypes.reset_index(drop=True)
    offline_celltypes = offline_celltypes.reset_index(drop=True)


    # map the roi_ids to each tother
    if not offline_to_online:
        # mapping online to offline in data frame
        out_df = online_celltypes
        
        # add offline data 
        out_df["offline_roi_id"] = out_df['online_roi_id'].map(mapping)
        out_df["correlation"] = out_df['online_roi_id'].map(corrs)
        out_df["distance"] = out_df['online_roi_id'].map(all_dists)
        out_df = out_df.merge(offline_celltypes[['offline_roi_id','offline_cell_type','offline_max_confidence']],on='offline_roi_id',how='left')

    else:
        # mapping offline to online in data frame
        out_df = offline_celltypes

        # add online data
        out_df["online_roi_id"] = out_df['offline_roi_id'].map(mapping)
        out_df["correlation"] = out_df['offline_roi_id'].map(corrs)
        out_df["distance"] = out_df['offline_roi_id'].map(all_dists)
        out_df = out_df.merge(online_celltypes[['online_roi_id','online_cell_type','online_max_confidence']],on='online_roi_id',how='left')

    
    # find quality indices for online rois
    if quality_df is not None:
        field_quality_online = quality_df[(quality_df[field_id_col] == field_id) & (quality_df['cond1'] == online_col_val)]
        field_quality_offline = quality_df[(quality_df[field_id_col] == field_id) & (quality_df['cond1'] == offline_col_val)]

        # add qidxs
        if  not offline_to_online:
            # join on online roi id
            out_df = out_df.merge(field_quality_online[['roi_id','mb_qidx','chirp_qidx']].rename(columns={'roi_id':'online_roi_id','mb_qidx':'online_mb_qidx','chirp_qidx':'online_chirp_qidx'}),on='online_roi_id',how='left')
        else:
            out_df = out_df.merge(field_quality_offline[['roi_id','mb_qidx','chirp_qidx']].rename(columns={'roi_id':'offline_roi_id','mb_qidx':'offline_mb_qidx','chirp_qidx':'offline_chirp_qidx'}),on='offline_roi_id',how='left')


    return out_df

def get_all_cond1_to_cond1_celltype(all_celltypes_df,
                                     traces_df, 
                                     quality_df = None, 
                                    online_col_val = 'cl',
                                    offline_col_val = 'n1',
                                    field_id_col = 'field_id',
                                    cond1_col = 'cond1', 
                                    offline_to_online = True,):
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
                                               offline_to_online = offline_to_online)
        out_dfs.append(field_out_df)
    out_df = pd.concat(out_dfs,ignore_index=True)
    
    # sanity checks: no missing rois in total
    reference_col_val = offline_col_val if offline_to_online else online_col_val
    assert sum(all_celltypes_df['cond1'] == reference_col_val) == len(out_df)

    # same nr of rois as for con1 value for each field 
    assert all(out_df.groupby(field_id_col).size() == all_celltypes_df[all_celltypes_df['cond1'] == reference_col_val].groupby(field_id_col).size())


    return out_df


   




def prepare_celltype_data(df, offline_col='offline_cell_type', online_col='online_cell_type',
                          max_type=32, chirp_qidx_threshold=None, mb_qidx_threshold=None,
                          chirp_percentile=None, 
                          mb_percentile=None,
                          nan_strategy='drop',
                        ):
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
    nan_strategy : str
        Strategy for handling NaN values in cell type columns ('drop' to remove, 'group' to group into a separate category)
        
    Returns:
    --------
    df_prepared : pandas.DataFrame
        Filtered and prepared DataFrame with grouped cell types
    applied_thresholds : dict
        Dictionary with the applied threshold values
    """
    import pandas as pd
    import numpy as np


    assert nan_strategy in ['drop','group']
    
    # Assert that user doesn't specify both percentile and threshold for the same metric
    if chirp_percentile is not None and chirp_qidx_threshold is not None:
        raise ValueError("Cannot specify both chirp_percentile and chirp_qidx_threshold")
        
    if mb_percentile is not None and mb_qidx_threshold is not None:
        raise ValueError("Cannot specify both mb_percentile and mb_qidx_threshold")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()


    # drop nans if requested
    if nan_strategy == 'drop':
        df_copy = df_copy.dropna(subset=[offline_col, online_col])

    
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

    # group nans 
    
    # Group cell types > max_type into a single category
    df_copy[offline_col + '_grouped'] = df_copy[offline_col].apply(
        lambda x: x if (not pd.isna(x) and x <= max_type) else (max_type + 1 if not pd.isna(x) else np.nan))
    df_copy[online_col + '_grouped'] = df_copy[online_col].apply(
        lambda x: x if  (not pd.isna(x) and x <= max_type) else (max_type + 1 if not pd.isna(x) else np.nan))
    if nan_strategy == 'group':
        df_copy[offline_col + '_grouped'] = df_copy[offline_col + '_grouped'].fillna(max_type + 2)
        df_copy[online_col + '_grouped'] = df_copy[online_col + '_grouped'].fillna(max_type + 2)
    
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

    has_grouped_nan = ((df[offline_col_grouped] == max_type + 2) | (df[online_col_grouped] == max_type + 2)) .any()
    if has_grouped_nan:
        # add extra entry for online (columns)
        if (max_type + 2) not in confusion_counts.columns:
            confusion_counts.loc[max_type + 2] = 0
    
    # Sort both axes
    confusion_counts = confusion_counts.sort_index().sort_index(axis=1)
    
    # Create row-normalized matrix (probabilities)
    confusion_probs = confusion_counts.div(confusion_counts.sum(axis=1), axis=0).fillna(0)
    
    return confusion_counts, confusion_probs






















############################################################################# Plotting ##########################################################


def plot_confusion_matrix(matrix, 
                          is_counts=True, 
                          max_type=32, 
                          figsize=(20, 16), 
                          cmap='Blues', 
                          annot_fmt='.2f', 
                          hide_zero=True, 
                          applied_thresholds=None,
                          cbar_kws = {},
                          round_to=False,
                          **heatmap_kws):
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

    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Format depends on whether this is counts or probabilities
    fmt = 'd' if is_counts else annot_fmt
    label = 'Count' if is_counts else 'Probability'
    
    if round_to:
        matrix = matrix.round(round_to)
    

    if hide_zero:
        if isinstance(hide_zero,float):
            thresh = hide_zero
        else:
            thresh = 0.01
        mask = matrix < thresh if not is_counts else matrix == 0
        heatmap_kws['mask'] = mask

    
    # Plot the matrix
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, 
                ax=ax, 
                cbar_kws={'label': label, **cbar_kws},**heatmap_kws)
    
    # Update labels
    ax.set_xlabel('Online Cell Type')
    ax.set_ylabel('Offline Cell Type')
    
    # Set title
    if is_counts:
        total_cells = matrix.sum().sum()
        title = f'\nTotal: {int(total_cells)} cells'
    else:
        title = 'P(online type | offline type)'
    
    if applied_thresholds:
        threshold_str = ', '.join([f"{k}: {v:.3f}" for k, v in applied_thresholds.items()])
        title += f'\nQuality thresholds: {threshold_str}'
    
    ax.set_title(title)
    
    # Rename the last tick label to "AC" or "Missed" if there are grouped types
    xticks = list(range(1, max_type + 1)) + ["AC"] + (["Missed"] if matrix.columns.max() == max_type + 2 else [])
    yticks = list(range(1, max_type + 1)) + ["AC"] + (["Missed"] if matrix.index.max() == max_type + 2 else [])
    ax.set_xticklabels(xticks, rotation=45, ha='center')
    ax.set_yticklabels(yticks,rotation=45, va='center') 
    
    
    return fig, ax


def plot_celltype_confusion_matrix(df, offline_col='offline_cell_type', online_col='online_cell_type', 
                                  max_type=32, figsize=(20, 16), cmap='Blues', annot_fmt='.2f',
                                  plot_counts=True, save_path=None, hide_zero=True,cbar_kws = {},
                                  chirp_qidx_threshold=None, mb_qidx_threshold=None,round_to=False,
                                  chirp_percentile=None, mb_percentile=None,nan_strategy='drop',
                                  heatmap_kws: Dict = {}):
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
        chirp_percentile, mb_percentile,
        nan_strategy=nan_strategy,
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
        hide_zero=True,
        cbar_kws = cbar_kws,
        round_to= round_to,
        applied_thresholds=applied_thresholds,
        **heatmap_kws
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

def plot_percentage_gain(results_df: pd.DataFrame,
                         celltype_col: str = 'target_type_idx',
                         percentage_gain_col: str = 'percentage_gain',
                        subplots_kws = {}, 
                        plt_kws = {},
                        open_bar_celltypes = [])-> Tuple[plt.Figure, plt.Axes]:
    """
    """




    df = results_df.copy()
    df["celltype"] = df[celltype_col] + 1
    
    # Sort by celltype to ensure correct order
    df = df.sort_values('celltype')
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, **subplots_kws)
    
    # Create bar colors based on percentage gain
    gains = df[percentage_gain_col].to_list()
    celltypes = df['celltype'].to_list()
    bar_colors = ['green' if gain >= 0 else 'red' for gain in gains]

    y_max = max(gains) * 1.1
    y_min = min(gains) * 1.1

    for ct,gain,color in zip(celltypes,gains,bar_colors,strict=True):
        if ct in open_bar_celltypes:
            bar_kwas = {"x": ct,
                "bottom": y_min,
                "height": y_max - y_min,
                "color": 'white',
                "edgecolor": "grey",
                "linewidth": 2,
                "hatch": "//",
                "width": 0.7,
                "alpha": 0.3,
                "zorder": 0,
            }
        else:
            bar_kwas = {"x": ct,
                "height": gain,
                "color": color,
                "width": 0.7,
                "alpha": 0.7,
            }
       
        ax.bar(
            **bar_kwas
        )
        

    # Add horizontal dotted line at y=0
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Ensure there's an x-tick for each cell type
    celltypes = df['celltype'].unique()
    ax.set_xticks(celltypes,celltypes,rotation=45)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set labels and title
    ax.set_xlabel('Cell type')
    ax.set_ylabel('Yield increase [%]')
    

    
    # Prettify the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax










def add_terciles(quality_pivot):
    quality_pivot['n1_tercile'] = pd.qcut(quality_pivot['n1'], q=[0,1/3,2/3,1], labels=["lower","middle","upper"])
    quality_pivot['cl_tercile'] = pd.qcut(quality_pivot['cl'], q=[0,1/3,2/3,1], labels=["lower","middle","upper"])
    return quality_pivot


def plot_ballpark_quality_contingency(quality_pivot,
                                      subplot_kws = {}, heatmap_kws = {}):
    
    df = quality_pivot.copy()
    # add terciles
    df = add_terciles(df)
    
    counts_crosstab = pd.crosstab(df['n1_tercile'],df['cl_tercile'],rownames=['Offline Quality'],colnames=['Online Quality'])
    prob_crosstab = counts_crosstab.div(counts_crosstab.sum(axis=1), axis=0)
    
    # Get the range values for each tercile
    cl_ranges = pd.qcut(quality_pivot['cl'], q=[0,1/3,2/3,1]).value_counts().sort_index()
    n1_ranges = pd.qcut(quality_pivot['n1'], q=[0,1/3,2/3,1]).value_counts().sort_index()
    
    # Create new labels with ranges
    cl_labels = [f"{label}\n({interval.left:.2f} - {interval.right:.2f})" 
                for label, interval in zip(["lower", "middle", "upper"], cl_ranges.index)]
    n1_labels = [f"{label}\n({interval.left:.2f} - {interval.right:.2f})" 
                for label, interval in zip(["lower", "middle", "upper"], n1_ranges.index)]

    fig,ax = plt.subplots(1,1, **subplot_kws)
    heatmap = sns.heatmap(counts_crosstab, ax =ax,
                annot=True, 
                fmt=".0f", 
                cmap="Blues",
                cbar_kws={'label': "Count"},
                **heatmap_kws)
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([])

    # Set the new labels
    ax.set_xticklabels(cl_labels)
    ax.set_yticklabels(n1_labels, rotation=0)
    
    ax.set_xlabel('Online quality tercile')
    ax.set_ylabel('Offline quality tercile')
    ax.set_title(f'Total: {len(df)} recording fields')

    
    return fig,ax

def plot_quality_scatter(quality_pivot,
                         subplot_kws = {}, 
                         scatter_kws = {},
                         summary_stats = True,)-> Tuple[plt.Figure, plt.Axes]:
 
    fig,ax = plt.subplots(**subplot_kws)
    xs = quality_pivot['cl']
    ys = quality_pivot['n1']

    if summary_stats:
        print(f"Correlation between online and offline quality: {xs.corr(ys):.3f}")
        
    
    scatter = ax.scatter(xs, ys, alpha=0.7,  **scatter_kws)  

    max_val = max(ys.max(), xs.max())
    min_val = min(ys.min(), xs.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    ax.set_xlabel('Offline Quality')
    ax.set_ylabel('Online Quality')

    ax.legend([scatter], ['Recording field'], loc='upper left')


    plt.axis('square')
    plt.xlim(min_val-0.05, max_val+0.05)
    plt.ylim(min_val-0.05, max_val+0.05)

    # remove 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig,ax