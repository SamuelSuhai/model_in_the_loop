from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from openretina.data_io.base import ResponsesTrainTestSplit


"""
NOTE:
roi_id: is an int!!! mei_id is a str!!!

"""


def validate_container(sub_container: pd.DataFrame,
                       only_consider_these_rois: List[int]) -> None:

    # 1. check that all roi_ids in only_consider_these_rois are in sub_container
    if not all(roi in sub_container['roi_id'].values for roi in only_consider_these_rois):
        raise ValueError("Some roi_ids in only_consider_these_rois are not in sub_container.")


    # only one celltype,stabiliy for each roi id
    for roi_id in only_consider_these_rois:
        roi_data = sub_container[sub_container['roi_id'] == roi_id]
        if roi_data['celltype'].nunique() != 1:
            raise ValueError(f"Multiple celltypes for roi_id {roi_id}.")
        if roi_data['stability'].nunique() != 1:
            raise ValueError(f"Multiple stabilities for roi_id {roi_id}.")
        
def validate_input(only_consider_these_rois: List[int],
                   readout_idx_wmei2rois: Dict[int, int],
                   ) -> None:
    """some checks"""

    # Validate input
    if len(only_consider_these_rois) < 6:
        raise ValueError("Need at least 6 ROIs to select MEIs for.")
    
    # Check that all requested ROIs have readout indices
    missing_rois = [roi for roi in only_consider_these_rois if roi not in readout_idx_wmei2rois.values()]
    if missing_rois:
        raise ValueError(f"These ROIs don't have associated readout indices: {missing_rois}\
                         only have readout indices for these ROIs: {list(readout_idx_wmei2rois.values())}")
    

    


def build_common_df(only_consider_these_rois: List[int],
                    mei_data_container: pd.DataFrame,
                    neuron_data_dict: Dict[int,ResponsesTrainTestSplit],
                    new_session_id: int,
                    readout_idx_wmei2rois: Dict[int, int],):

    """
    Constructs a df that extends mei_data_container subset to include
    1. celltype 
    2. the response of all the readout_idxs of interest (that we selected, one per column) to the MEIs in the rows 
    """
    # 1. subset mei_data_container to only consider the ROIs of interest
    sub_container = mei_data_container.query("roi_id in @only_consider_these_rois")[["readout_idx","roi_id","mei_id","stability","mean_responses_all_readout_idx"]].copy()

    # 2. add celltype column 
    all_readout_idx_groups = neuron_data_dict[new_session_id].session_kwargs["group_assignment"]
    sub_container['celltype'] = sub_container['readout_idx'].apply(lambda x: all_readout_idx_groups[x])

    # 3. add the responses of all readout idx of interest as separate columns
    readout_idxs_of_interest = [readout_idx for readout_idx,roi in readout_idx_wmei2rois.items() if roi in only_consider_these_rois]
    for readout_idx in readout_idxs_of_interest:
        sub_container[f'response_readout_idx_{readout_idx}'] = sub_container['mean_responses_all_readout_idx'].apply(lambda x: x[readout_idx])

    # 4. drop the mean_responses_all_readout_idx column
    sub_container = sub_container.drop(columns=['mean_responses_all_readout_idx'])
    return sub_container

def find_mei_id_oder_for_one_readout_idx(roi_id: int,
                                        sub_container: pd.DataFrame,
                                        n_stimuli_total: int = 6,
                                        random_seed: int = 42) -> Tuple[List[str], Dict[str, List[Any]]]:
    """
    Select MEIs for a specific ROI using the standard heuristic.
    
    Args:
        roi_id: The ROI ID to select MEIs for
        sub_container: DataFrame with MEI data prepared by build_common_df
        n_stimuli_total: Total number of stimuli to select
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple containing:
            - List of selected MEI IDs
            - Dictionary with metadata about the selected MEIs
    """
    # Get the DataFrame filtered to this ROI
    roi_meis = sub_container[sub_container['roi_id'] == roi_id]
    if roi_meis.nunique()['mei_id'] not in  [1,2]:
        raise ValueError(f"Unexpected nr of MEIs for ROI {roi_id}")
    
    # Get the readout_idx for this ROI 
    readout_idx = roi_meis['readout_idx'].iloc[0]
    
    # Initialize lists for tracking selections
    selected_mei_ids = []
    mei_responses = []
    celltypes = []
    stabilities = []
    
    # Get cell type for this ROI
    celltype = roi_meis['celltype'].iloc[0]
    
    # Step 1: Add this ROI's own MEIs
    stability = roi_meis['stability'].iloc[0]
    own_meis = roi_meis['mei_id'].tolist()
        
    expected_count = 1 if stability == 'stable' else 2
    if len(own_meis) != expected_count:
        raise ValueError(f"Expected {expected_count} MEIs for {stability} ROI {roi_id}, found {len(own_meis)}")
    
    selected_mei_ids.extend(own_meis)
    
    # Step 2: Add a MEI from another cell of same type if available
    same_type_meis = sub_container[(sub_container['celltype'] == celltype) & 
                                  (sub_container['roi_id'] != roi_id)]
    
    if len(same_type_meis) > 0:
        random_mei_id = same_type_meis.sample(n=1, random_state=random_seed)['mei_id'].iloc[0]
        selected_mei_ids.append(random_mei_id)
    
    # Step 3: Fill remaining slots based on response strength

    # Get all MEIs not yet selected
    remaining_meis = sub_container[~sub_container['mei_id'].isin(selected_mei_ids)]
    n_remaining = n_stimuli_total - len(selected_mei_ids)
    if n_remaining < 3:
        raise ValueError("Not enough remaining MEIs to select from.")

    response_col = f'response_readout_idx_{readout_idx}'
    sorted_meis = remaining_meis.sort_values(by=response_col, ascending=False)
    sorted_mei_ids = sorted_meis['mei_id'].tolist()

    if len(sorted_mei_ids) < n_remaining:
        raise ValueError(f"Only {len(sorted_mei_ids)} MEIs available for ROI {roi_id}, but need {n_remaining} more")


    # Pick evenly spaced MEIs from remaining, includes strongest and weakest
    indices = np.linspace(0, len(sorted_mei_ids)-1, n_remaining, dtype=int)
    for idx in indices:
        selected_mei_ids.append(sorted_mei_ids[idx])
  

    # Gather metadata for selected MEIs
    for mei_id in selected_mei_ids:
        mei_row = sub_container[sub_container['mei_id'] == mei_id]
        if len(mei_row) != 1:  # Safety check
            raise ValueError(f"MEI ID {mei_id} should be unique in sub_container.")
        mei_responses.append(float(mei_row[response_col].iloc[0]))
        celltypes.append(mei_row['celltype'].iloc[0])
        stabilities.append(mei_row['stability'].iloc[0])
    
    # Prepare info dictionary
    info = {
        "all_stabilities": stabilities,
        "celltype": celltypes,
        "responses": mei_responses,
    }
    
    return selected_mei_ids, info

def select_subset_of_meis_for_each_roi( only_consider_these_rois: List[int],
                                        neuron_data_dict: Dict[int,ResponsesTrainTestSplit],
                                        new_session_id: int,
                                        mei_data_container: pd.DataFrame,
                                        readout_idx_wmei2rois: Dict[int, int],
                                        n_stimuli_total = 6,
                                        ) -> Tuple[Dict[int,List[str]], Dict[int, Dict[int, List[Any]]]]:
    """
    Selects which MEIs to show what rois.
    only_consider_these_rois contains all the ROIs that we want to stimulate and that require a list of mei_ids. Only MEIs from these ROIs are considered (also for validataion).
    reurns dict with roi_id as key and mei_id list as value.
    requieres (takes from self):
    mei_data_container,
    neuron_data_dict
    
    Uses the following heuristic: n_stimuli_total meis total:
    1. for a given roi_id,
        i) add its own meis (one if stable, two if unstable)
        ii) if there is another cell with same type add its mei (if there are mutliple seeds or cells take random mei)
        iii) Take a list of mei_ids sorted by response strength. fill up the list with mei_ids arrcording to the respnose strength until we reach n_stimuli_total.

    reuturn two dicts:
    roi_id2mei_ids: Dict[int, List[str]]: mapping from roi_id to list of mei_ids
    roi_id2info: Dict[int, Dict[str, Any]]: mapping from roi_id to dict with info about the selected meis (stability, celltype, responses

    """
    validate_input(only_consider_these_rois,
                   readout_idx_wmei2rois,
                   )

    # Build the common DataFrame with all necessary data
    sub_container = build_common_df(
        only_consider_these_rois,
        mei_data_container,
        neuron_data_dict,
        new_session_id,
        readout_idx_wmei2rois
    )

    validate_container(sub_container, only_consider_these_rois)
    
    # Process each ROI
    roi_id2mei_ids = {}
    roi_id2info = {}
    
    for roi_id in only_consider_these_rois:
        selected_mei_ids, info = find_mei_id_oder_for_one_readout_idx(
            roi_id, 
            sub_container,
            n_stimuli_total
        )
        roi_id2mei_ids[roi_id] = selected_mei_ids
        roi_id2info[roi_id] = info

    
    return roi_id2mei_ids, roi_id2info


