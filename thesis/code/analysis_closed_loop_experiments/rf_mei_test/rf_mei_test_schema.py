import re
import numpy as np
from typing import Dict, Any, Tuple,List

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


from .rf_mei_test_tables import (StimulusPresentationInfoTemplate,
                                 SingleSnippetTemplate,
                                 Offline2OnlineRoiIdTemplate,
                                 OnlineInferredRFPositionTemplate,
                                 )




@schema
class StimulusPresentationInfo(StimulusPresentationInfoTemplate):
    presentation_table = Presentation


@schema
class SingleSnippet(SingleSnippetTemplate):
    snippets_table = Snippets
    stimulus_presentation_info_table = StimulusPresentationInfo


@schema 
class Offline2OnlineRoiId(Offline2OnlineRoiIdTemplate):
    roi_table = Roi 


@schema
class OnlineInferredRFPosition(OnlineInferredRFPositionTemplate):
    stimulus_presentation_info_table = StimulusPresentationInfo
    offline2online_roi_id_table = Offline2OnlineRoiId
