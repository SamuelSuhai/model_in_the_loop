from abc import abstractmethod
from typing import Any, Dict,List
import datajoint as dj
import os
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CIRCLE_TYPES = ["on_small", "off_small", "on_big", "off_big"]



class CirclePresentationLocationTemplate(dj.Imported):
    database = ""
    
    @property
    def definition(self):

        definition = """
        # Parameters for RF MEI test stimulus presentations
        -> self.presentation_table
        ---
        online_roi_id_order:      longblob   # roi id got in online experiment 
        positions:               longblob   # positions of circles
        metadata_file:     varchar(255)  # path to metadata file
        triggeridx2positions: longblob  # mapping from trigger idx to position idx
        triggeridx2online_roi_id: longblob  # mapping from trigger idx to online roi id
        triggeridx2circle_type: longblob  # mapping from trigger idx to circle type
        """
        return definition


    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    def key_source(self):
        try:
            keys =  ((self.presentation_table() & "stim_name = 'circle'") - self).proj()
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
        
        triggeridx2positions,triggeridx2online_roi_id,triggeridx2circle_type = get_trigidx2info(roi_ids,positions,CIRCLE_TYPES)
        key["triggeridx2positions"] = triggeridx2positions
        key["triggeridx2online_roi_id"] = triggeridx2online_roi_id
        key["triggeridx2circle_type"] = triggeridx2circle_type
        self.insert1(key)

    def make(self, key):
        pres_data_file = (self.presentation_table & key).fetch1("pres_data_file")
        metadata_file = get_metadata_file_from_pres_file(pres_data_file)
        if os.path.exists(metadata_file):
            self.insert_from_metadata_file(metadata_file, key)


            

class SingleCircleSnippetTemplate(dj.Computed):
    database = ""

    circle_types = CIRCLE_TYPES
    @property
    def definition(self):
        definition = """
        # Single circle snippet
        -> self.snippets_table
        -> self.circle_presentation_location_table()
        x_pos:          float  # x position in qdspy
        y_pos:          float  # y position in qdspy
        online_roi_id:  smallint  # roi id got in online experiment
        circle_type:    varchar(16)  # elypse type: on_small, off_small, on_big, off_big
        ---
        single_snippet:                 longblob  # actual snippet data
        triggertime_single_snippet:     longblob  # trigger time for 
        single_snippet_t0:              float  # start time of snippet
        is_first_pres_of_circle_type:   tinyint # bool whether presentation is first of a type of elypse
        """
        return definition

    @property
    @abstractmethod
    def circle_presentation_location_table(self):
        pass

    @property
    @abstractmethod
    def snippets_table(self):
        pass
    
    @property
    def key_source(self):
        keys =  (self.snippets_table() & self.circle_presentation_location_table()).proj()
        return keys

     
    def make(self, key,restrictions = {}):
        restricted_snippets = (self.snippets_table() & key & restrictions)
        snippets = restricted_snippets.fetch1("snippets")
        triggertimes_snippets = restricted_snippets.fetch1("triggertimes_snippets")
        snippets_dt = restricted_snippets.fetch1("snippets_dt")
        snippets_t0 = restricted_snippets.fetch1("snippets_t0")

        restricted_circle_pres = (self.circle_presentation_location_table() & key) & restrictions
        positions,online_roi_id_order = restricted_circle_pres.fetch1("positions","online_roi_id_order")

        n_snippets = snippets.shape[1] # (time,n_snippets)
        n_online_rois = len(online_roi_id_order)
        n_circle_types = len(self.circle_types)

        if  n_snippets != n_online_rois * n_circle_types:
            warn(f"n_snippets {n_snippets} != n_online_rois {n_online_rois}\
                  * n_circle_types {n_circle_types}, ONLY TAKING UP TO {n_online_rois * n_circle_types} SNIPPETS")
            n_snippets = n_online_rois * n_circle_types
            snippets = snippets[:,:n_snippets]
        assert n_online_rois == len(positions)

        for snippet_idx in range(n_snippets):
            single_snippet = snippets[:, snippet_idx]
            circle_type_idx = snippet_idx // n_online_rois
            online_roi_idx = snippet_idx % n_online_rois
            circle_type = self.circle_types[circle_type_idx]
            online_roi_id = online_roi_id_order[online_roi_idx]
            is_first_pres_of_circle_type = online_roi_idx == 0

            self.insert1(
                {**key,
    
                "x_pos": positions[online_roi_idx][0],
                "y_pos": positions[online_roi_idx][1],
                "online_roi_id": online_roi_id,
                "circle_type": circle_type,
                "single_snippet": single_snippet,
                "triggertime_single_snippet": triggertimes_snippets[:,snippet_idx],
                "single_snippet_t0": snippets_t0[snippet_idx],
                "is_first_pres_of_circle_type": is_first_pres_of_circle_type,
                }
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
    def circle_presentation_location_table(self):
        pass
    
    @property
    def key_source(self):
        keys =  (self.offline2online_roi_id_table() & self.circle_presentation_location_table()).proj()
        return keys
    
    def make(self, key):

        circle_presentation_locations_rois_control = (self.circle_presentation_location_table() * self.offline2online_roi_id_table() & key & "cond2='control'")
        positions,online_roi_id_order = circle_presentation_locations_rois_control.fetch1("positions","online_roi_id_order")

        true_online_roi_id = circle_presentation_locations_rois_control.fetch1("true_online_roi_id")

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
    folder_list = pres_data_file.split("/")
    # replace dj with other_data
    folder_list[7] = "other_data"

    # remove Raw
    folder_list.remove("Raw")
    base_name = folder_list.pop(-1)

    field = list(filter(lambda x: 'GCL' in x, base_name.split("_")))[0]
    folder_list.append(f"rf_{field}")

    folder_list.append("metadata.yaml")

    metadata_file = "/" +  os.path.join(*folder_list)
    return metadata_file



def get_trigidx2info(online_roi_ids_order,positions,circle_types):
    """
    Get mapping from trigger idx to position idx, online roi id and circle type

    Args:
        online_roi_ids_order (List[int]): list of online roi ids
        positions (List[Tuple[float,float]]): list of positions
        circle_types (List[str]): list of circle types
    Returns:
        triggeridx2positions (List[int]): mapping from trigger idx to position idx
        triggeridx2online_roi_id (List[int]): mapping from trigger idx to online roi id
        triggeridx2circle_type (List[str]): mapping from trigger idx to circle type
    """
    triggeridx2positions = []
    triggeridx2online_roi_id = []
    triggeridx2circle_type = []

    for circle in circle_types:
        for roi_id,pos in zip(online_roi_ids_order,positions):
            triggeridx2positions.append(pos)
            triggeridx2online_roi_id.append(roi_id)
            triggeridx2circle_type.append(circle)

    return triggeridx2positions,triggeridx2online_roi_id,triggeridx2circle_type


