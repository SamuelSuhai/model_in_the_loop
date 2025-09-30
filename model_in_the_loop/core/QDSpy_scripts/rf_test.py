#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------
import QDS
import os
from model_in_the_loop.utils.QDSpy_helpers import get_latest_remote_stimulus_subdir, read_metadata,copy_stim_dir_to_local


STIMULUS_OUTPUT_DIR = r"\\172.25.250.112\euler_data\Data\Suhai\stimuli_closed_loop"
LOCAL_STIMULUS_DIR = r"C:\Users\eulerlab\QDSpy\Stimuli\Samuel\rf_and_mei"
ELYPSE_ON_COLOR = (255,255,255)
ELYPSE_OFF_COLOR = (0,0,0)
ELYPSE_BACKGROUND = (122,122,122)
ELYPSE_SIZE_SMALL = (50,50)
ELYPSE_SIZE_BIG = (100,100)
ELYPSE_TYPE_ORDER = ["on_small","off_small","on_big","off_big"] 
"""
Thes is a fixed spatial presentation order. 
The scrip shows one elypse type sequentially at each location then the next elypse type one after the oter at each location
etc ...
repeated for each trial 

Example:
trial 0:
elypse tpe on_small:
roi 1, roi 2 ....

"""




def execute (metadata: dict,) -> None:
    QDS.Initialize("RF TEST", "Closed loop RF test")

    # Define global stimulus parameters
    #
    FrRefr_Hz = QDS.GetDefaultRefreshRate()

    ## unpack metadata
    roi_ids = metadata["roi_ids"]
    positions = metadata["positions"]

    # turn positions to list of tupples (x,y)
    positions = [(float(p[0]), float(p[1])) for p in positions]


    ## some checks 
    assert len(positions) == len(roi_ids) 



    p = {"nTrials"         : 3,           # number of stimulus presentations  
        "MarkPer_s"       : 1/3,         # number of markers per second
        "durFr_s"         : 1/FrRefr_Hz, # frame duration
        "nFrPerMarker"      : 3,
        "elypse_on_color": ELYPSE_ON_COLOR,
        "elypse_off_color": ELYPSE_OFF_COLOR,
        "elypse_background_color": ELYPSE_BACKGROUND,
        "elypse_size_small": ELYPSE_SIZE_SMALL,
        "elypse_size_big": ELYPSE_SIZE_BIG,
        "elypse_type_order": ELYPSE_TYPE_ORDER,
        **metadata # put metadata in params
        }
        


    QDS.LogUserParameters(p)

    ## Define objects
    # RF test elypse: ON small, off small On BIG big, off big

    all_object_id_param_map = []

    id = 1
    for i_outer,type in enumerate(p["elypse_type_order"]):
        for i_inner,(roi,pos) in enumerate(zip(roi_ids,positions)):
            
            dx = p["elypse_size_small"][0] if "small" in type.split("_") else p["elypse_size_big"][0]
            dy = p["elypse_size_small"][1] if "small" in type.split("_") else p["elypse_size_big"][1]
            QDS.DefObj_Ellipse(id,dx, dy, 0)

            rgb = p["elypse_on_color"] if "on" in type.split("_") else p["elypse_off_color"]

            all_object_id_param_map.append({"id":id, 
                                            "type": type, 
                                            "roi_id": roi, 
                                            "dxdy": (dx,dy),
                                            "rgb":rgb,
                                            "pos": pos,
                                            })

            id += 1


    # marker and framerate params

    t_break_and_stim  = 3 # in seconds 
    dFr               = 1 /FrRefr_Hz
    nMark             = int(t_break_and_stim * p["MarkPer_s"])
    dMark_s           = p["nFrPerMarker"] *dFr
    dInterval_s       = 1.0/p["MarkPer_s"]

    ### Start of stimulus run
    QDS.StartScript()
    QDS.Scene_Clear(1.00, 0)

    ## RF test
    QDS.SetBkgColor(p["elypse_background_color"])


    # draw objects
    for trial_idx in range(p["nTrials"]):
        for obj_idx,obj_params in enumerate(all_object_id_param_map):
            
            ## backgourund no elypse
            # draw marker
            QDS.Scene_Clear(dMark_s, 1)
            QDS.SetBkgColor(p["elypse_background_color"])

            # calculate remaining time
            remaining_trial_time = dInterval_s - dMark_s
            remaining_bg_time = remaining_trial_time / 2
            remaining_time_elypse = remaining_trial_time - remaining_bg_time
            
            # draw bg no marker
            QDS.Scene_Clear(remaining_bg_time, 0)
            QDS.SetBkgColor(p["elypse_background_color"])


            ## draw elypse 
            # get params 
            obj_id = obj_params["id"]
            pos = obj_params["pos"]
            rgb = obj_params["rgb"]
            dxdy = obj_params["dxdy"]

            # set color
            QDS.SetObjColor(1,[obj_id], [rgb])

            # show for half duration of remaining time
            QDS.Scene_Render(remaining_time_elypse, 1, [obj_id], [pos], 0) #wo marker

        
    QDS.Scene_Clear(1.00, 0)

    # final log
    QDS.LogUserParameters(all_object_id_param_map * p["nTrials"])

    # Finalize stimulus
    QDS.EndScript()

  # -----------------------------------------------------------------------------


if __name__ == "__main__":


  # find the latest directory with stimuli
  latest_remote_subdir = get_latest_remote_stimulus_subdir(STIMULUS_OUTPUT_DIR)

  # copy dir to local
  copy_stim_dir_to_local(STIMULUS_OUTPUT_DIR, latest_remote_subdir, LOCAL_STIMULUS_DIR)

  # read metadata
  abs_local_subdir = os.path.join(LOCAL_STIMULUS_DIR, latest_remote_subdir)
  metadata = read_metadata(abs_local_subdir)


  execute(metadata)
