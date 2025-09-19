
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle


from openretina.utils.h5_handling import load_h5_into_dict
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.data_io.hoefling_2024.stimuli import movies_from_pickle

# import plotter
import thesis.code.plot.plot as plotter
import thesis.code.plot.style as styler

RESPONSES_PATH = "/gpfs01/euler/User/ssuhai/openretina_cache/euler_lab/hoefling_2024/responses/rgc_natstim_2024-08-14.h5"
MOVIES_PATH = "/gpfs01/euler/User/ssuhai/openretina_cache/euler_lab/hoefling_2024/stimuli/rgc_natstim_18x16_joint_normalized_2024-01-11.pkl"
OFFLINE_SESSION_ID = "session_1_ventral1_20200226"
ONLINE_SESSION_ID = "online_session_1_ventral1_20250717"


def load_openretina_data(cfg):
    # load OR data

    offline_raw_responses_dict = load_h5_into_dict(file_path=RESPONSES_PATH)

    offline_filtered_responses_dict = filter_responses(offline_raw_responses_dict, **cfg.model_configs.quality_checks)

    offline_neuron_data_dict = make_final_responses(offline_filtered_responses_dict, response_type="natural")

    movies_dict = movies_from_pickle(MOVIES_PATH)


def load_online_seesion_dict(path):
    with open(path, "rb") as f:
        online_session_dict = pickle.load(f)
    return online_session_dict


def find_offline_roi_id_from_session_dicts(online_roi_id,online_session_dict, offline_session_dict):

    online_idx = np.where(online_raw_session_dict["roi_ids"]== online_roi_id)[0].item()
    online_spikes = online_session_dict["natural_spikes"][online_idx]

    max_corr = -1
    best_offline_roi_id = None

    for or_roi_id in offline_session_dict["roi_ids"]:
        offline_idx = np.where(offline_session_dict["roi_ids"]== or_roi_id)[0].item()
        or_spikes = offline_session_dict["natural_spikes"][offline_idx]
        corr = np.corrcoef(online_spikes[~ np.isnan(online_spikes)], or_spikes[~ np.isnan(or_spikes)])[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_offline_roi_id = or_roi_id

    return best_offline_roi_id, offline_idx, online_idx, max_corr

def plot_online_offline_dict_val(online_idx, offline_idx, online_session_dict, offline_session_dict, key = "natural_spikes",win=(100,200) ax= None):
    
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()


    online_val = online_session_dict[key][online_idx]
    offline_val = offline_session_dict[key][offline_idx]


    palette = styler.get_palette('online_offline')
    ax.plot(online_val, color= palette['online'], label = 'online')
    ax.plot(offline_val, color= palette['offline'], label = 'offline')
    ax.legend()

    ax.set_xlim(win)
    ax.set_xlabel('Time [frames]')
    ax.set_ylabel('Spike Probability [a.u.]')

    return fig, ax
