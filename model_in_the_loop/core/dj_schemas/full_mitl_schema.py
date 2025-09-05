import re
import numpy as np
from typing import Dict, Any, Tuple

########################## Basics for RGC type classification ##########################
from abc import abstractmethod
import datetime
from djimaging.schemas.full_rgc_schema import *

######################### spike estimation #########################
from djimaging.tables import spike_estimation

@schema
class CascadeTraceParams(spike_estimation.CascadeTracesParamsTemplate):
    stimulus_table = Stimulus

@schema
class CascadeTraces(spike_estimation.CascadeTracesTemplate):
    cascadetraces_params_table = CascadeTraceParams
    presentation_table = Presentation
    traces_table = Traces

@schema
class CascadeParams(spike_estimation.CascadeParamsTemplate):
    pass

@schema
class CascadeSpikes(spike_estimation.CascadeSpikesTemplate):
    presentation_table = Presentation
    cascadetraces_params_table = CascadeTraceParams
    cascadetraces_table = CascadeTraces
    cascade_params_table = CascadeParams


######################### location #########################
from djimaging.tables.location.roi_location import (
    RelativeRoiLocationTemplate,
    RelativeRoiLocationWrtFieldTemplate,
    RetinalRoiLocationTemplate,
)

@schema
class RelativeRoiLocationWrtField(RelativeRoiLocationWrtFieldTemplate):
    roi_table = Roi
    roi_mask_table = RoiMask # I changed this from Field.RoiMask to RoiMask this could cause problems if we have multiple fields
    field_table = Field
    presentation_table = Presentation


@schema
class RelativeRoiLocation(RelativeRoiLocationTemplate):
    relative_field_location_wrt_field_table = RelativeRoiLocationWrtField
    relative_field_location_table = RelativeFieldLocation
    roi_table = Roi
    roi_mask_table = RoiMask
    field_table = Field
    presentation_table = Presentation

@schema
class RetinalRoiLocation(RetinalRoiLocationTemplate):
    relative_roi_location_table = RelativeRoiLocation
    expinfo_table = Experiment.ExpInfo






################### open retina #########################

class OpenRetinaHoeflingFormatTemplate(dj.Manual):
    database = ""
    ignore_iteration_as_primary_key = False # if True then the class assumes the data is extracted over multiple iterations and cond1 is not used as primary key.

    @property
    def definition(self):
        definition = """
        # Brings data in the the Open Retina format. For analysis like tha data in Hoefling et al. 2024
        -> self.field_table
        session_name: varchar(100)  # session name,
        ---
        session_data_dict:      longblob    # preprocessed trace
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        """The table that contains the fields.This is used as an iterations table since.
        cond1 is also a primary key and it denotes the iteration. """
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def preprocess_traces_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    @abstractmethod
    def cascade_spikes_table(self):
        pass

    @property
    @abstractmethod
    def experiment_info_table(self):
        pass

    @property
    @abstractmethod
    def animal_table(self):
        pass

    @property
    @abstractmethod
    def chirp_qi_table(self):
        pass

    @property
    @abstractmethod
    def os_ds_indexes_table(self):
        pass

    @property
    @abstractmethod
    def celltype_assignment_table(self):
        pass

    @property
    @abstractmethod
    def roi_mask_table(self):
        pass

    @property
    @abstractmethod
    def roi_table(self):
        pass

    @property
    @abstractmethod
    def retinal_field_location_table(self):
        pass

    @property
    @abstractmethod
    def retinal_roi_location_table(self):
        pass

    
    def get_query(self, key):

        # Define stimulus keys TODO: move hardcoded key to dynamically get from stimulus table
        natural_movie_key = {"stim_name": "mouse_cam"}

        # join all traces
        all_traces = (
            self.preprocess_traces_table() *
            self.traces_table() *
            self.presentation_table().proj("pres_data_file", "triggertimes")
        ) & key

        # get stimulus specific queries
        natural_query = (all_traces & natural_movie_key)

        # Check if spike data exists for each stimulus type
        natural_has_spikes = len(self.cascade_spikes_table() & key & natural_movie_key) > 0  
        

        # mappings for changing the field names, since for different stimuli the field names should be different
        natural_mapping = {
            "natural_raw_traces": "trace",
            "natural_smoothed_traces": "smoothed_trace",
            "natural_preprocessed_traces": "pp_trace",
            "natural_stim_name": "stim_name",
            "natural_traces_times_t0": "trace_t0",
            "natural_traces_times_dt": "trace_dt",
            "natural_preprocessed_traces_times_t0": "pp_trace_t0",
            "natural_preprocessed_traces_times_dt": "pp_trace_dt",
            "natural_trigger_times": "triggertimes",
            "natural_pres_data_file": "pres_data_file",
            "natural_pp_id": "preprocess_id",
        }

        # Conditionally add spike fields to mappings and join the queries
        # if the spikes exist for the stimulus type
        if natural_has_spikes:
            natural_query = natural_query * (self.cascade_spikes_table() & key & natural_movie_key)
            natural_mapping.update({
                "natural_spikes": "spike_prob",
                "natural_spike_times": "spike_prob_times"
            })

        # create a table that has one row for each ROI and contains all traces (and possibly spikes)
        if self.ignore_iteration_as_primary_key:
            natural_mapping = {**natural_mapping, "natural_cond1": "cond1"}
        all_roi_traces = natural_query.proj(**natural_mapping)

        # add the celltype assignment
        celltype_table = self.celltype_assignment_table().proj(group_assignment="celltype", group_confidences="confidence",celltype_pp_id="preprocess_id") 


        # for quality filtering
        qi_table = self.chirp_qi_table().proj(
            chirp_stim_name="stim_name", chirp_qi="qidx", quality_pp_id="preprocess_id"
            ) * self.os_ds_indexes_table().proj(
                mb_stim_name="stim_name", d_qi="d_qi", ds_index="ds_index",
                os_index="os_index", pref_dir="pref_dir")
        
        for table in [qi_table, celltype_table]:
            if len(table) == 0:
                raise ValueError(f"Table {table.__class__.__name__} is empty. Please check the database for the required data.")
   
        # combine data
        result = (
            all_roi_traces * \
            self.experiment_info_table().proj("eye") * \
            self.animal_table().proj(animal_gender="animgender") * \
            self.roi_mask_table().proj("roi_mask") * \
            celltype_table * \
            qi_table * \
            self.roi_table().proj("roi_size_um2"))
        #     * self.retinal_field_location_table().proj("ventral_dorsal_pos_um", "temporal_nasal_pos_um") *
        #     self.retinal_roi_location_table().proj( # location stuff is commented bc why tho
        #     roi_ventral_dorsal_pos_um="ventral_dorsal_pos_um",
        #     roi_temporal_nasal_pos_um="temporal_nasal_pos_um"
        # )


        return result

    def from_query_to_data_dict(self, dj_query):
        """ Extracts the data from the query and returns it as a dictionary.
        Assumes that three sitmuli are in database: natural, chirp and mb."""

        all_data = dj_query.fetch(as_dict=True)

        if len(all_data) == 0:
            raise ValueError("No data found for the given key. Please check the key and try again.")
       
        # which fields to map to which name when extracting data (to be able to do 
        # all analyses like on Hoefling et al. 2024 in Open Retina)
        field_mapping = {
        "roi_ids": "roi_id",
        "group_assignment": "group_assignment",
        "group_confidences": "group_confidences",
        "chirp_qi": "chirp_qi",
        "ds_index": "ds_index",
        "pref_dir": "pref_dir",
        "os_index": "os_index",
        "d_qi": "d_qi",
        "roi_size_um2": "roi_size_um2",
        }
        stimuli = ["natural"]#, "chirp", "mb"]
        
        partial_name_trace_fields = ["trace","spike","trigger_times"]
        full_name_trace_fields = [ key for key in all_data[0].keys() if any(partial_name in key for partial_name in partial_name_trace_fields) ]
       
        session_data_dict = {}
        for row_idx,data in enumerate(all_data):
            if row_idx == 0:
                # Extact data thats the same in the iteration
                pres_data_file_path = data["natural_pres_data_file"]
                filename = pres_data_file_path.split("/")[-1]
                scan_sequence_idx = int(re.search(r"MC(.+?)_.*\.smp", filename).group(1))
                session_data_dict["animal_gender"] = data["animal_gender"]
                session_data_dict["date"] = str(data["date"])
                session_data_dict["scan_sequence_idx"] = scan_sequence_idx    
                assert scan_sequence_idx in [i for i in range(21)]

                static_saved_fields = ["experimenter","eye", "exp_num", "field","cond1","roi_mask","animal_gender",]
                session_data_dict.update( {field: data[field] for field in static_saved_fields} )

                # initialize lists to store data 
                session_data_dict.update({dest_key: [] for dest_key in field_mapping.keys()})
                session_data_dict.update({key: [] for key in full_name_trace_fields})

                # trace times
                for stim in stimuli:
                    session_data_dict.update({f"{stim}_traces_times": [],
                                          f"{stim}_preprocessed_traces_times": [],
                                           })
                
         

            # Extract data that differes within iteration.
            for dest_key, source_key in field_mapping.items():
                session_data_dict[dest_key].append(data[source_key])

            for field_name in full_name_trace_fields:
                session_data_dict[field_name].append(data[field_name])
            
            for stim in stimuli:
                session_data_dict[f"{stim}_traces_times"].append(np.arange(data[f"{stim}_raw_traces"].shape[0]) * data[f"{stim}_traces_times_dt"] + data[f"{stim}_traces_times_t0"])
                session_data_dict[f"{stim}_preprocessed_traces_times"].append(np.arange(data[f"{stim}_preprocessed_traces"].shape[0]) * data[f"{stim}_preprocessed_traces_times_dt"] + data[f"{stim}_preprocessed_traces_times_t0"])

        
            # in final row convert lists to numpy arrays       
            if row_idx == len(all_data) - 1:
                for field in session_data_dict.keys():
                    if isinstance( session_data_dict[field], list):
                        # turn to numpy 
                        session_data_dict[field] = np.array(session_data_dict[field])
        
        return session_data_dict

    @staticmethod            
    def get_field_key(field: str, eye: str, date: datetime.date,) -> str:
        # build key, e.g. '1_ventral1_20210929_iter0'
        eye_id = 1 if eye == "left" else 2
        field_id = int(field[3:]) + 1
        key_parts = [str(field_id), f"ventral{eye_id}", str(date).replace("-", "")]
        key_str = "_".join(key_parts)
        return key_str
    

    def extract_data(self) -> Dict[str, Dict] | None:
        """ Extracts all the data for the model from the database. 
        """
        
        # get the key for the iteration
        key = (self.field_table - self).proj().fetch(as_dict=True)[0]
        if self.ignore_iteration_as_primary_key:
            key.pop("cond1") # remove iteration dependence because we want to get all stimuli

        # get all iteration data in long format:
        all_iter_data = self.get_query(key)

        if len(all_iter_data) == 0:
            return None


        # Get the data and build the dictionary
        session_data_dict = self.from_query_to_data_dict(all_iter_data)


        # get the session name from the key
        session_name = "online_session_" + self.get_field_key(
                session_data_dict["field"],
                session_data_dict["eye"],
                session_data_dict["date"],
            )
        
        return {session_name:session_data_dict}


    
@schema
class OpenRetinaHoeflingFormat(OpenRetinaHoeflingFormatTemplate):
    field_table = Field
    presentation_table = Presentation
    preprocess_traces_table = PreprocessTraces
    traces_table = Traces
    cascade_spikes_table = CascadeSpikes
    experiment_info_table = Experiment.ExpInfo
    animal_table = Experiment.Animal
    chirp_qi_table = ChirpQI
    os_ds_indexes_table = OsDsIndexes
    celltype_assignment_table = CelltypeAssignment
    roi_mask_table = RoiMask
    roi_table = Roi
    retinal_field_location_table = RetinalFieldLocation
    retinal_roi_location_table = RetinalRoiLocation
    


class OnlineMEIsTemplate(dj.Manual):
    database = ""
    

    @property
    def definition(self):
        definition = """
        # Saves the meis generated online.
        -> self.openretina_hoefling_format_table  # the table that contains the Open Retina Hoefling format (dummy) data
        roi_id:         int
        readout_idx:    int  # the index of the readout neuron in the model
        seed:           int  # the seed used to generate the MEI
        ---
        mei:            longblob    # np.ndarray of shape (batch,channel,time,height,width)
        model_response:  longblob    # np.ndarray of shape (time,)
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def openretina_hoefling_format_table(self):
        """The table that contains the Open Retina Hoefling format (dummy) data."""
        pass

@schema
class OnlineMEIs(OnlineMEIsTemplate):
    field_table = Field
    openretina_hoefling_format_table = OpenRetinaHoeflingFormat


class OnlineTrainedModelTemplate(dj.Manual):
    database = ""
    
    @property
    def definition(self):
        definition = """
        # Saves the trained model online.
        -> self.openretina_hoefling_format_table  # the table that contains the Open Retina Hoefling format (dummy) data
        ---
        model_chkpt_path: varchar(1024)  # the path to model checkpoint
        """
        return definition
    
    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def openretina_hoefling_format_table(self):
        """The table that contains the Open Retina Hoefling format (dummy) data."""
        pass


@schema
class OnlineTrainedModel(OnlineTrainedModelTemplate):
    field_table = Field
    openretina_hoefling_format_table = OpenRetinaHoeflingFormat