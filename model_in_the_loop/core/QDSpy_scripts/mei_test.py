#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------
import QDS
import os


from model_in_the_loop.utils.QDSpy_helpers import get_latest_remote_stimulus_subdir, read_metadata,copy_stim_dir_to_local,check_remote_files

STIMULUS_OUTPUT_DIR = r"\\172.25.250.112\euler_data\Data\Suhai\stimuli_closed_loop"
LOCAL_STIMULUS_DIR = r"C:\Users\eulerlab\QDSpy\Stimuli\Samuel\rf_and_mei"


"""
MEI PRESENTATION LOGIC:
The metadata file contains mei_ids as a list of lists. Each sublist contains the mei_ids to be presented for a specific roi.
so the number of sublists is equal to the number of rois and the length of each sublist is the number of meis to be presented per roi.
In the script we do the following:
loop over rois, if frst time always take idx 0 if second time idx 1 etc.
Example assume 3 rois and 2 mei ids 
all_mei_presentation_orders = [
    ["roi_10_seed_111", "roi_10_seed_222"],  # MEIs for ROI 10
    ["roi_20_seed_111", "roi_20_seed_222"],  # MEIs for ROI 20
    ["roi_30_seed_111", "roi_30_seed_222"]   # MEIs for ROI 30
]


NAMING CONVENTION:
roi_id is an int 
seed is an int
mei_id is a string like "roi_<roi_id>_seed_<seed>"

"""



def execute (metadata: dict,
             abs_local_subdir: str) -> None:
    QDS.Initialize("RF and MEI", "Closed loop RF and MEI test")

    # Define global stimulus parameters
    #
    FrRefr_Hz = QDS.GetDefaultRefreshRate()

    ## unpack metadata
    roi_ids = metadata["roi_ids"]
    positions = metadata["positions"]


    # turn positions to list of tupples (x,y)
    positions = [(float(p[0]), float(p[1])) for p in positions]

    all_mei_presentation_orders = metadata["mei_ids"]


    ## some checks 
    assert len(positions) == len(roi_ids) 


    # each mei_id list should have the same legth
    n_mei_per_roi = len(all_mei_presentation_orders[0])
    for mei_id_list in all_mei_presentation_orders:
        print(f"PRESENTING {n_mei_per_roi} MEIs per ROI")
        assert len(mei_id_list) == n_mei_per_roi, "Each MEI presentation order should have the same number of MEIs"

    # nr or mei_trials i.e. nr of sublists in all_mei_presentation_orders is equal to the nr or rois
    n_mei_orderings = len(all_mei_presentation_orders)
    assert n_mei_orderings == len(roi_ids), "Number of MEI orderings should be equal to number of ROIs"

    # make sure we have all avi videos we want to shoe
    for mei_id_list in all_mei_presentation_orders:
        for mei_id in mei_id_list:
            abs_file_name = os.path.join(abs_local_subdir, f"{mei_id}.avi")
            if not os.path.exists(abs_file_name):
                raise FileNotFoundError(f"MEI video file '{abs_file_name}' does not exist.")

    p = {"nTrials"         : 1,           # number of stimulus presentations  
        "vidScale"        : (12.5, 12.5),  # movie scaling (x, y)
        "vidOrient"       : 0,           # movie orientation
        "vidAlpha"        : 255,         # transparency of movie
        "MarkPer_s"       : 1/3,         # number of markers per second
        "durFr_s"         : 1/FrRefr_Hz, # frame duration
        "nFrPerMarker"      : 3,
        **metadata # put metadata in params

        }
      


    QDS.LogUserParameters(p)

    ## Define objects
  
    all_object_id_param_map =[]
    # define mei objects
    id = 1
    for mei_trial_idx in range(n_mei_per_roi):
        for idx_roi,(roi,pos) in enumerate(zip(roi_ids,positions)):

            # mei_trial_idx denotes the index we need in the sublists
            mei_order_for_roi = all_mei_presentation_orders[idx_roi]
            mei_id = mei_order_for_roi[mei_trial_idx]
            abs_file_name = os.path.join(abs_local_subdir, f"{mei_id}.avi")
            QDS.DefObj_Video(id, abs_file_name)

            # get more metadata

            # save the parameters for later chekcing
            all_object_id_param_map.append({"id":id, 
                                            "type": "mei", 
                                            "roi_id": roi,
                                            "roi_idx": idx_roi,
                                            "mei_trial_idx": mei_trial_idx,
                                            "pos":pos,
                                            "mei_id": mei_id, 
                                            "file": abs_file_name,
                                            })   
            
            # for the first mei video set additional parameters, since they dont 
            # change across avis
            if id == 1:
                vidparams         = QDS.GetVideoParameters(id)
                p["vidparams"]    = vidparams
                dFr               = 1 /FrRefr_Hz
                nMark             = int(vidparams["nFr"] / FrRefr_Hz * p["MarkPer_s"])
                dMark_s           = p["nFrPerMarker"] *dFr
                dInterval_s       = 1.0/p["MarkPer_s"]

            
            # increment id 
            id += 1
            



    ### Start of stimulus run
    QDS.StartScript()
    QDS.Scene_Clear(1.00, 0)

    for obj_idx,obj_params in enumerate(all_object_id_param_map):
        obj_id = obj_params["id"]
        pos = obj_params["pos"]
        QDS.Start_Video(obj_id, pos, p["vidScale"], p["vidAlpha"], p["vidOrient"])

        # draw marker
        for iM in range(nMark):
            QDS.Scene_Clear(dMark_s, 1)
            QDS.Scene_Clear(dInterval_s -dMark_s, 0)
    
    QDS.Scene_Clear(1.00, 0)

    # final log
    QDS.LogUserParameters(all_object_id_param_map)

    # Finalize stimulus
    #
    QDS.EndScript()

    # -----------------------------------------------------------------------------


if __name__ == "__main__":


  # find the latest directory with stimuli
  latest_remote_subdir = get_latest_remote_stimulus_subdir(STIMULUS_OUTPUT_DIR)

  # make sure one metadata and at least one avi
  abs_remote_subdir = os.path.join(STIMULUS_OUTPUT_DIR, latest_remote_subdir)
  check_remote_files(abs_remote_subdir)

  # copy dir to local
  copy_stim_dir_to_local(STIMULUS_OUTPUT_DIR, latest_remote_subdir, LOCAL_STIMULUS_DIR)

  # read metadata
  abs_local_subdir = os.path.join(LOCAL_STIMULUS_DIR, latest_remote_subdir)
  metadata = read_metadata(abs_local_subdir)


  execute(metadata,
          abs_local_subdir)
