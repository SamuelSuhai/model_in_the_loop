
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from openretina.data_io.base import ResponsesTrainTestSplit


def validate_input():
    pass


def build_common_df(only_consider_these_rois: List[int],
                    mei_data_container: pd.DataFrame,
                    neuron_data_dict: Dict[str,ResponsesTrainTestSplit],
                    new_session_id: str,
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




def select_subset_of_meis_for_each_roi( only_consider_these_rois: List[int],
                                        neuron_data_dict: Dict[str,ResponsesTrainTestSplit],
                                        new_session_id: str,
                                        mei_data_container: pd.DataFrame,
                                        readout_idx_wmei2rois: Dict[int, int],
                                        n_stimuli_total = 6,
                                        ) -> Tuple[Dict[int,List[str]], Dict[int, Dict[str, List[Any]]]]:
    """
    Selects which MEIs to show what rois.
    only_consider_these_rois contains all the ROIs that we want to stimulate and that require a list of mei_ids. Only MEIs from these ROIs are considered (also for validataion).
    reurns dict with roi_id as key and mei_id list as value.
    requieres (takes from self):
    mei_data_container,
    neuron_data_dict
    
    Uses the following heuristic: n_stimuli_total meis total: Three to four exciting and rest depressing meis 
    1. for a given roi_id,
        i) add its own meis (one if stable, two if unstable)
        ii) if there is another cell with same type add its mei (if there are mutliple seeds or cells take random mei)
        iii) Take a list of mei_ids sorted by response strength. fill up the list with mei_ids arrcording to the respnose strength until we reach n_stimuli_total.

    reuturn two dicts:
    roi_id2mei_ids: Dict[int, List[int]]: mapping from roi_id to list of mei_ids
    roi_id2info: Dict[int, Dict[str, Any]]: mapping from roi_id to dict with info about the selected meis (stability, celltype, responses

    """

    # 0. First: subset the data so we only perform this analysis on selected ROIs:
    readout_idxs_of_interest = [readout_idx for readout_idx,roi in readout_idx_wmei2rois.items() if roi in only_consider_these_rois]
    if len(readout_idxs_of_interest) == len(only_consider_these_rois): 
        raise ValueError("Mismatch between readout idx of interest and roi ids of interest.perhaps you selected a roi id that does not have an mei?")

    mei_is_of_desired_roi = mei_data_container['readout_idx'].isin(readout_idxs_of_interest)
    sub_mei_data_container = mei_data_container[mei_is_of_desired_roi]
    assert len(sub_mei_data_container) > 0, "No MEIs found for the given ROIs of interest."


    # 1. map readout_idxwmei to cell type
    all_readout_idx_groups = neuron_data_dict[new_session_id].session_kwargs["group_assignment"]
    readout_idx_wmei2group = {idx:all_readout_idx_groups[idx] for idx in readout_idxs_of_interest}
    

    # some checks
    assert len(readout_idx_wmei2group) == len(sub_mei_data_container['readout_idx'].unique()), "Mismatch between readout idx with meis and neuron data dict."
    assert all([idx in readout_idx_wmei2group.keys() for idx in sub_mei_data_container['readout_idx'].unique()]), "Some readout idx in mei data container not in neuron data dict."

    
    ## fetch all mean repsonses array size (nr meis, nr readouts in model)
    all_readout_idx_mean_responses = np.stack(sub_mei_data_container['mean_responses_all_readout_idx'].tolist(), axis=0)
    assert all_readout_idx_mean_responses.shape[0] == len(sub_mei_data_container), "Mismatch between number of MEIs and mean responses."
    assert all_readout_idx_mean_responses.shape[1] == len(neuron_data_dict[new_session_id].session_kwargs["group_assignment"]), "Mismatch between number of readouts in model and mean responses."        
    
    ## 3. select mei ids for each roi based on the mean response in the time window and possibly cell type 
    roi_id2mei_ids = {}
    roi_id2info = {}
    # loop over readout_idx
    for readout_idx,celltype in readout_idx_wmei2group.items():
        
        # 3 a) get the necessary data for this readout idx
        # store the mei_ids for this roi/readout idx
        selected_mei_ids = []
        mei_responses = []
        celltypes_or_neurons_from_meis = []
        all_stabilites = []
        
        # get the roi_id
        roi_id = readout_idx_wmei2rois[readout_idx]

        # get all mei data for this readout idx
        all_meis_for_readout = sub_mei_data_container[sub_mei_data_container['readout_idx'] == readout_idx]
        assert len(all_meis_for_readout) > 0, f"No MEIs found for readout idx {readout_idx}."
        
        # whether stable or not
        stability: str = all_meis_for_readout.iloc[0]['stability']
        assert all_meis_for_readout['stability'].nunique() == 1, f"Mixed stability for readout idx {readout_idx}."

        # the mean repsonses in the optimization window
        mean_responses_of_idx = all_readout_idx_mean_responses[:, readout_idx] # shape (nr_meis,) 
        assert mean_responses_of_idx.shape[0] == mei_is_of_desired_roi.shape[0], "Mismatch between number of MEIs and mean responses array length,"
        
        # since we took  a subset of mei_data_container we need to only take certain indices of mean_responses_of_idx again 
        mean_responses_of_idx = mean_responses_of_idx[mei_is_of_desired_roi]
        assert mean_responses_of_idx.shape[0] == len(sub_mei_data_container), "Mismatch between number of MEIs in subset and mean responses array length after subsetting."
        
        # 3 b) decide on mei_ids accroding to heuristic
        # step i) add its own meis (one if stable, two if unstable)
        own_meis = all_meis_for_readout['mei_id'].tolist()
        assert len(own_meis) == (1 if stability == 'stable' else 2), f"Unexpected number of own MEIs for readout idx {readout_idx} with stability {stability}."
        selected_mei_ids.extend(own_meis)



        # step ii) if there is another cell with same type add its mei (if there are mutliple seeds take one random)
        same_type_mei_entries = sub_mei_data_container[sub_mei_data_container['readout_idx'].isin(
            [idx for idx,grp in readout_idx_wmei2group.items() if grp == celltype and idx != readout_idx])]

        if len(same_type_mei_entries) > 0:
            # take one random mei from the same type
            random_same_type_mei_id = same_type_mei_entries.sample(n=1, random_state=42)['mei_id'].item()
            selected_mei_ids.append(random_same_type_mei_id)

        # step iii) Take a list of mei_ids sorted by response strength. 
        # fill up the list with mei_ids arrcording to the respnose strength until we reach 6. 
        # definately include the strongest and weakest one
        assert 1 <= len(set(selected_mei_ids)) <= 3, f"Unexpected number of MEIs selected so far for readout idx {readout_idx}: {len(selected_mei_ids)}."
        nr_missing = n_stimuli_total - len(selected_mei_ids)
        sorted_mei_indices = np.argsort(mean_responses_of_idx)[::-1] # descending order
        remaining_mei_ids = [sub_mei_data_container.iloc[idx]['mei_id'] for idx in sorted_mei_indices if sub_mei_data_container.iloc[idx]['mei_id'] not in selected_mei_ids]
        
        # select evely but definately include strongerst
        step_size = len(remaining_mei_ids) / nr_missing 
        for i in range(nr_missing - 1): # -1 because we add the weakest one at the end
            selected_mei_ids.append(remaining_mei_ids[int(i * step_size)])

        
        # add the weakest one 
        selected_mei_ids.append(remaining_mei_ids[-1])

        # to have bettwe ovreview if its all corect we add the responses and celltypes of the selected meis
        for mei_id in selected_mei_ids:
            bool_mask_mei_ids = sub_mei_data_container['mei_id'] == mei_id
            mei_responses.extend(mean_responses_of_idx[bool_mask_mei_ids].tolist())
            celltypes_or_neurons_from_meis.extend([readout_idx_wmei2group[idx] for idx in sub_mei_data_container[bool_mask_mei_ids]['readout_idx'].tolist()])
            all_stabilites.extend(sub_mei_data_container[bool_mask_mei_ids]['stability'].tolist())

        # store the metadata
        roi_id2info[roi_id] = {
            "all_stabilities": all_stabilites,
            "celltype": celltypes_or_neurons_from_meis,
            "responses": mei_responses,
        }

        # store the mei_ids 
        roi_id2mei_ids[roi_id] = selected_mei_ids

    return roi_id2mei_ids, roi_id2info

