from abc import abstractmethod
from typing import Any, Dict,List
import datajoint as dj
import os
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CIRCLE_TYPES = ["on_small", "off_small", "on_big", "off_big"]



class StimulusPresentationInfoTemplate(dj.Imported):
    database = ""
    
    @property
    def definition(self):

        definition = """
        # Parameters for RF MEI test stimulus presentations
        -> self.presentation_table
        ---
        online_roi_id_order:      longblob   # roi id got in online experiment 
        positions:               longblob   # positions of stimulus
        metadata_file:     varchar(255)  # path to metadata file
        triggeridx2positions: longblob  # mapping from trigger idx to position idx
        triggeridx2online_roi_id: longblob  # mapping from trigger idx to online roi id
        triggeridx2stim_type: longblob  # mapping from trigger idx to stimulus type
        triggeridx2is_first_pres_of_stimulus: longblob  # mapping from trigger idx to whether it is the first presentation of the stimulus
        """
        return definition


    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            keys =  ((self.presentation_table() & "stim_name IN ('circle','optstim')") - self).proj()
            return keys
        except (AttributeError, TypeError):
            pass

    def insert_from_metadata_file(self,metadata_file:str,presentation_key: Dict[str,Any]):
        """
        Insert entries from a metadata file. 

        Args:
            metadata_file (str): path to metadata file (yaml)
        """

        import yaml
        import warnings


        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        roi_ids = metadata["roi_ids"]
        positions = metadata["positions"]

        assert len(positions) == len(roi_ids)

        key = presentation_key.copy()
        key["online_roi_id_order"] = roi_ids
        key["positions"] = positions
        key["metadata_file"] = metadata_file
        

        # get stim types
        stim_types = metadata.get("mei_ids",CIRCLE_TYPES)
        triggeridx2positions,triggeridx2online_roi_id,triggeridx2stim_type,triggeridx2is_first_pres_of_stimulus = get_trigidx2info(roi_ids,positions,stim_types)
        key["triggeridx2positions"] = triggeridx2positions
        key["triggeridx2online_roi_id"] = triggeridx2online_roi_id
        key["triggeridx2stim_type"] = triggeridx2stim_type
        key["triggeridx2is_first_pres_of_stimulus"] = triggeridx2is_first_pres_of_stimulus
        

        self.insert1(key)

    def make(self, key):
        pres_data_file = (self.presentation_table & key).fetch1("pres_data_file")
        metadata_file = get_metadata_file_from_pres_file(pres_data_file)
        if os.path.exists(metadata_file):
            self.insert_from_metadata_file(metadata_file, key)


            

class SingleSnippetTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Single circle snippet
        -> self.snippets_table
        -> self.stimulus_presentation_info_table
        x_pos:          float  # x position in qdspy
        y_pos:          float  # y position in qdspy
        online_roi_id:  smallint  # roi id got in online experiment
        stimulus_type:    varchar(64)  # elypse type: on_small, off_small, on_big, off_big OR mei id 
        ---
        single_snippet:                 longblob  # actual snippet data
        snippets_dt:                   float     # time between frames in seconds
        triggertime_single_snippet:     longblob  # trigger time for 
        single_snippet_t0:              float  # start time of snippet
        is_first_pres_of_stimulus:   tinyint # bool whether presentation is first of a type of elypse
        """
        return definition
   
    @property
    def key_source(self):
        try:
            keys = (self.snippets_table() & self.stimulus_presentation_info_table()).proj()
            return keys
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def stimulus_presentation_info_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass


     
    def make(self, key,):
        restricted_snippets = (self.snippets_table() & key )
        snippets = restricted_snippets.fetch1("snippets")
        triggertimes_snippets = restricted_snippets.fetch1("triggertimes_snippets")
        snippets_dt = restricted_snippets.fetch1("snippets_dt")
        snippets_t0 = restricted_snippets.fetch1("snippets_t0")

        restricted_stim_pres_loc = (self.stimulus_presentation_info_table() & key) 
        all_fetched = restricted_stim_pres_loc.fetch1("triggeridx2positions","triggeridx2online_roi_id",\
                                                                                                        "triggeridx2stim_type","triggeridx2is_first_pres_of_stimulus")
        print(len(all_fetched))
        triggeridx2positions,triggeridx2online_roi_id, triggeridx2stim_type,triggeridx2is_first_pres_of_stimulus = all_fetched

        n_snippets = snippets.shape[1] # (time,n_snippets)
        n_triggerswinfo = len(triggeridx2positions)

        if  n_snippets != n_triggerswinfo:
            warn(f"n_snippets {n_snippets} != n_triggerswinfo {n_triggerswinfo}\
                  , ONLY TAKING UP TO {n_triggerswinfo} SNIPPETS ({key=})")
            n_snippets = n_triggerswinfo
            snippets = snippets[:,:n_snippets]
        
        # transpose for nice zipping later
        snippets = snippets.T # (n_snippets,time)

        for snippet_idx,(single_snippet,
                         position,
                         online_roi_id,
                         stimulus_type,
                         is_first_pres_of_stimulus) in enumerate(zip(snippets,
                                                                    triggeridx2positions,
                                                                    triggeridx2online_roi_id,
                                                                    triggeridx2stim_type,
                                                                    triggeridx2is_first_pres_of_stimulus,
                                                                    strict=True)):

            new_key = {**key,
                "x_pos": position[0],
                "y_pos": position[1],
                "online_roi_id": online_roi_id,
                "stimulus_type": stimulus_type,
                "single_snippet": single_snippet,
                "snippets_dt": snippets_dt,
                "triggertime_single_snippet": triggertimes_snippets[:,snippet_idx],
                "single_snippet_t0": snippets_t0[snippet_idx],
                "is_first_pres_of_stimulus": is_first_pres_of_stimulus,
                }
            self.insert1(
                new_key
                )

class OnlineInferredRFPositionTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Stores the RF position of the online roi (just taken out of the list).
        -> self.offline2online_roi_id_table
        --- 
        x_rf: float  # x position of RF in qdspy coordinates
        y_rf: float  # y position of RF in qdspy coordinates
        """
        return definition
     
        
    @property
    @abstractmethod
    def offline2online_roi_id_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_presentation_info_table(self):
        pass
    
    @property
    def key_source(self):
        keys =  (self.offline2online_roi_id_table() & self.stimulus_presentation_info_table() ).proj()
        return keys
    
    def make(self, key):

        stimulus_presentation_info_table = (self.stimulus_presentation_info_table() * self.offline2online_roi_id_table() & key) & "cond2='control'"
        positions,online_roi_id_order = stimulus_presentation_info_table.fetch1("positions","online_roi_id_order")

        true_online_roi_id = stimulus_presentation_info_table.fetch1("true_online_roi_id")

        rf_position = positions[online_roi_id_order.index(true_online_roi_id)]
        self.insert1(dict(**key,**{"x_rf":rf_position[0],"y_rf":rf_position[1]}))

class Offline2OnlineRoiIdTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Holds a mapping from Offline roi id to Online roi id
        -> self.roi_table
        stim_name: varchar(32)  # name of stimulus
        ---
        true_online_roi_id: smallint  # roi id got in online experiment
        """
        return definition
    
    @property
    @abstractmethod
    def roi_table(self):
        pass

    def apply_mapping(self,
                      field_key,
                      mapping:Dict[int,int],
                      stim_name,
                      is_from_online2offline  = True) -> None:
        """
        Apply a mapping"""
        roi_keys = (self.roi_table & field_key).proj().fetch(as_dict=True)

        if is_from_online2offline:
            offline2online_map = {offline:online for online,offline in mapping.items()}
        else:
            offline2online_map = mapping

        for key in roi_keys:
            new_key = key.copy()
            offline_roi = key["roi_id"]
            if offline_roi not in offline2online_map:
                print(f"Warning: offline roi id {offline_roi} not in mapping, skipping")
                continue
            online_roi = offline2online_map[offline_roi]
            new_key["stim_name"] = stim_name
            new_key["true_online_roi_id"] = online_roi
            self.insert1(new_key,skip_duplicates=True)


def get_metadata_file_from_pres_file(pres_data_file):
    expected_stims = ["RF","MEI"]
    
    
    folder_list = pres_data_file.split("/")
    
    # replace dj with other_data
    folder_list[7] = "other_data"

    # remove Raw
    folder_list.remove("Raw")
    base_name = folder_list.pop(-1)

    field = list(filter(lambda x: 'GCL' in x, base_name.split("_")))[0]
    stim_name = [s for s in expected_stims if any(s in part for part in base_name.split("_"))][0]
    stim_name = stim_name.lower()
    folder_list.append(f"{stim_name}_{field}")

    folder_list.append("metadata.yaml")

    metadata_file = "/" +  os.path.join(*folder_list)
    return metadata_file



def get_trigidx2info(online_roi_ids_order,positions,stim_types):
    """
    Get mapping from trigger idx to position idx, online roi id and circle type

    Args:
        online_roi_ids_order (List[int]): list of online roi ids
        positions (List[Tuple[float,float]]): list of positions
        stim_types (List[str] OR List[List[str]]): list of stimulus types (circle types) or list of list of stimulus types (eg mei types)
    Returns:
        triggeridx2positions (List[int]): mapping from trigger idx to position idx
        triggeridx2online_roi_id (List[int]): mapping from trigger idx to online roi id
        triggeridx2stim_type (List[str]): mapping from trigger idx to stimulus type
        triggeridx2is_first_pres_of_stimulus (List[bool]): mapping from trigger idx to whether it is the first presentation of the stimulus
    """
    triggeridx2positions = []
    triggeridx2online_roi_id = []
    triggeridx2stim_type = []
    triggeridx2is_first_pres_of_stimulus = []

    if isinstance(stim_types[0],list):
        assert len(stim_types) == len(online_roi_ids_order), "If stim_types is a list of list, it should have the same length as online_roi_ids_order"
        assert all([len(st) == len(stim_types[0]) for st in stim_types]), "If stim_types is a list of list, each sublist should have the same length"
        n_stim_types = len(stim_types[0])
        mode= "list_of_list"
    else:
        assert isinstance(stim_types,list), "stim_types should be a list or a list of list"
        n_stim_types = len(stim_types)
        mode = "list"

    for stim_trial_idx in range(n_stim_types):
        for roi_idx,(roi_id,pos) in enumerate(zip(online_roi_ids_order,positions)):
            triggeridx2positions.append(pos)
            triggeridx2online_roi_id.append(roi_id)
            if mode == "list_of_list":
                type_stim = stim_types[roi_idx][stim_trial_idx]
                is_first_pres_of_stimulus = stim_trial_idx == 0 and roi_idx == 0
            else:
                type_stim = stim_types[stim_trial_idx]
                is_first_pres_of_stimulus = roi_idx == 0
            triggeridx2stim_type.append(type_stim)
            triggeridx2is_first_pres_of_stimulus.append(is_first_pres_of_stimulus)


    # transform labels in list_of_list case to dei
    if mode == "list_of_list":
        triggeridx2stim_type = transform_seed_mei_ids_to_new(
            triggeridx2stim_type
        )

    return triggeridx2positions,triggeridx2online_roi_id,triggeridx2stim_type,triggeridx2is_first_pres_of_stimulus


def transform_seed_mei_ids_to_new(triggeridx2stim_type: List[str]) -> List[str]:
    """
    Transform old mei ids to new mei ids.
    Old have form roi_<roi_id>_seed_<seed>
    the new ones have form roi_<roi_id>_type_<MEI/DEI> where 
    MEI is the case if there is only one seed per roi_id (eg say roi_22_seed_111) but if there are mulitple seeds ()eg roi_22_seed_111,roi_22_seed_222 then we use DEI 
    """
    from collections import Counter
    import re

    pattern = r'roi_(\d+)_seed_(\d+)'
    parsed_ids = [re.match(pattern, mei_id) for mei_id in triggeridx2stim_type]
    if any([p is None for p in parsed_ids]):
        raise ValueError("Some mei ids do not match the expected pattern 'roi_<roi_id>_seed_<seed>'")
    
    roi_ids = [int(p.group(1)) for p in parsed_ids]
    seed_ids = [int(p.group(2)) for p in parsed_ids]

    unique_roi_seeds = set(zip(roi_ids,seed_ids))
    roi_id_counts =Counter([roi_seed[0] for roi_seed in unique_roi_seeds ])
    
    # only one seed -> mei, two seeds -> dei with first number of seed as identifyer
    # NOTE this is brittle 
    roi_seed_name = {roi_seed:f"roi_{roi_seed[0]}_type_MEI" if roi_id_counts[roi_seed[0]] == 1 else f"roi_{roi_seed[0]}_type_DEI{roi_seed[1]}" for roi_seed in unique_roi_seeds}

    new_mei_ids = []
    for roi_id, seed_id in zip(roi_ids, seed_ids):
        new_mei_id = roi_seed_name[(roi_id,seed_id)]
        new_mei_ids.append(new_mei_id)
    
    return new_mei_ids