from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base import DJComputeWrapper, DJTableHolder
from omegaconf import DictConfig
from model_in_the_loop.utils.stimulus_optimization import (reconstruct_mei_from_decomposed,generate_opt_stim_mulitple_objectives,
                                decompose_mei, get_model_mei_response,Center,get_model_gaussian_scaled_means,
                                STIMULUS_SHAPE,FRAME_RATE_MODEL
                            )

from model_in_the_loop.utils.datajoiont_utils import get_rois_in_field_restriction_str
import pickle
import torch
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.layers.ensemble import EnsembleModel
from openretina.data_io.hoefling_2024.responses import make_final_responses
from openretina.data_io.hoefling_2024.constants import pre_normalisation_values_18x16


from model_in_the_loop.utils.model_training import load_stimuli, train_model_online
from model_in_the_loop.utils.simple_logging import log





def fetch_roi_data_from_container( opt_stim_data_container: pd.DataFrame,
                                  field_key: Dict[str, Any],
                                  roi2readout_idx_wmeis: Dict[int, int],
                                  roi_id: int,
                                  ) ->  Dict[str, Any]:
    
    # find neuron_id for roi_id
    neuron_idx = roi2readout_idx_wmeis[roi_id]


    # subset the data
    bool_mask_neuron_idx = opt_stim_data_container["readout_idx"] == neuron_idx
    data_subset = opt_stim_data_container[bool_mask_neuron_idx][["objective_name", "temporal_kernels","spatial_kernels","roi_id"]]
    
    objective_names_list = data_subset["objective_name"].tolist()

    
    assert len(data_subset) == len(np.unique(objective_names_list)), f"Expected at most {len(np.unique(objective_names_list))} responses for neuron idx {neuron_idx}, found {len(data_subset)}"
    assert len(data_subset["roi_id"].unique()) == 1, f"Expected exactly one roi_id for neuron idx {neuron_idx}, found {data_subset["roi_id"].unique()}"
    roi_id = data_subset["roi_id"].iloc[0]
    
    # Get all spatial kernels for this neuron (all obj_names)
    spatial_kernels_list = data_subset["spatial_kernels"].tolist()
    temporal_kernels_list = data_subset["temporal_kernels"].to_list()

    out = {
        "objective_names": objective_names_list,
        "temporal_kernels_list": temporal_kernels_list,
        "spatial_kernels_list": spatial_kernels_list,
        "neuron_idx": neuron_idx,
    }

    return out




    
def plot_spatial_kernels(readout_idx: int,
                            roi_id: int,
                            objective_names: List[int],
                            spatial_kernels_list: List[List[np.ndarray]], 
                        ax: plt.Axes) -> None:
    """
    Written by AI: 
    Plots the spatial kernels in the following way:
    - For each obj_name, green and UV channels are concatenated side by side
    - Different objective_names are stacked vertically
    - Small labels above each obj_name indicate the obj_name number
    """
    
    
    # Calculate total height needed for all objective_names
    total_height = 0
    all_combined_kernels = []
    
    # Process each objective's spatial kernels
    for i, (obj, kernel_pair) in enumerate(zip(objective_names, spatial_kernels_list)):
        # Concatenate green and UV kernels horizontally
        combined_kernel = np.concatenate(kernel_pair, axis=1)
        all_combined_kernels.append(combined_kernel)
        total_height += combined_kernel.shape[0]
    
    # Add small gaps between obj (10% of kernel height)
    gap_height = max(1, int(all_combined_kernels[0].shape[0] * 0.1))
    total_height += gap_height * (len(objective_names) - 1)
    
    # Create a large image to hold all kernels with gaps
    kernel_width = all_combined_kernels[0].shape[1]
    combined_image = np.zeros((total_height, kernel_width))
    
    # Find global min/max for consistent color scaling
    all_values = np.concatenate([k.flatten() for k in all_combined_kernels])
    abs_max = np.max(np.abs(all_values))
    
    # Place each kernel in the combined image with gaps
    y_offset = 0
    for i, kernel in enumerate(all_combined_kernels):
        h, w = kernel.shape
        combined_image[y_offset:y_offset+h, :] = kernel
        
        # Add obj label above each kernel
        ax.text(w//2, y_offset - 2, f"objective_name {objective_names[i]}", 
                ha='center', va='bottom', fontsize=8, color='black')
        
        # Update y_offset for next kernel
        y_offset += h + gap_height
    
    # Display the combined image
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    im = ax.imshow(combined_image, cmap="RdBu_r", norm=norm)
    
    # Add color channel labels at the top
    kernel_width_single = all_combined_kernels[0].shape[1] // 2
    ax.text(kernel_width_single // 2, -10, "Green", ha='center', fontsize=6)
    ax.text(kernel_width_single + kernel_width_single // 2, -10, "UV", ha='center', fontsize=6)
    
    # Add a scale bar (adjust position as needed)
    scale_bar_width = 4 if kernel_width > 30 else 1
    scale_bar = plt.Rectangle(xy=(6, total_height-5), width=scale_bar_width, 
                            height=1, color="k", transform=ax.transData)
    ax.add_patch(scale_bar)
    ax.text(6, total_height-8, "50 µm", fontsize=8)
    
    # Set title and turn off axis
    ax.set_title(f"Spatial Kernels for ROI {roi_id} (neuron idx {readout_idx})", fontsize=6)
    ax.axis("off")
    
    # Add a colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)




def plot_temporal_kernels(colors,
                              readout_idx: int,
                              roi_id: int,
                              objective_names: List[int],
                              temporal_kernels_list: List[np.ndarray],
                              axs: List[plt.Axes],
                              lableit = True) -> None:
        """
        Plots the temporal kernels for all objective_names for a single neuron.
        Green is in axs[0], UV in axs[1]
        all objective_names are plotted for each channel
        """


        # stack temporal_kernels into one array of shape (n_obj_names, 2, timepoints)
        temporal_kernels = np.stack(temporal_kernels_list,axis=0)

        # split channels
        green_temporal_kernels = temporal_kernels[:, 0, :]
        uv_temporal_kernels = temporal_kernels[:, 1, :]

        x = np.arange(green_temporal_kernels.shape[1]) / FRAME_RATE_MODEL

        # plotting
        for i,(obj_name, kernel) in enumerate(zip(objective_names, green_temporal_kernels)):
            axs[0].plot(x, kernel, label=f"objective_name {obj_name}", color=colors[i], linestyle='-' if objective_names.index(obj_name) % 2 == 0 else '--')

        for i,(obj_name, kernel) in enumerate(zip(objective_names, uv_temporal_kernels)):
            axs[1].plot(x, kernel, label=f"objective_name {obj_name}", color=colors[i], linestyle='-' if objective_names.index(obj_name) % 2 == 0 else '--')

        # labeling
        if lableit:
            axs[0].set_title(f"Green Temporal Kernels for ROI {roi_id} (neuron idx {readout_idx})", fontsize=6)
            axs[0].set_ylabel("Amplitude")
            axs[1].set_title(f"UV ROI {roi_id} (neuron idx {readout_idx})", fontsize=6)
            axs[1].set_ylabel("Amplitude")
            axs[1].set_xlabel("Time (s)")
            axs[0].legend(fontsize=6)
            axs[1].legend(fontsize=6)






class LandMEIWrapper(DJComputeWrapper):

    def __init__(self,dj_table_holder: DJTableHolder,
                cfg: DictConfig,

                ) -> None:
        
        self.dj_table_holder = dj_table_holder

      
        self.model_configs = cfg.model_configs
        self.mei_generation_params = cfg.openretina.mei_generation
        self.quality_filtering = cfg.quality_filtering
        self.save_dir_parent = os.path.join(cfg.paths.repo_directory,
                                            "model_in_the_loop/data/online_computed_data" )
        self.log_dir = os.path.join(cfg.paths.repo_directory,"model_in_the_loop/outputs/logs")

        self.colors = plt.cm.nipy_spectral(np.linspace(0, 1,2))

    def clear_tables(self, field_key, safemode=True) -> None:
        """
        Clears the tables the wrapper populates"""
        (self.dj_table_holder("CascadeTraces")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("CascadeSpikes")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OpenRetinaHoeflingFormat")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OnlineMEIs")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OnlineTrainedModel")() & field_key).delete(safemode=safemode)



    def plot_objective_respones(self,
                           readout_idx: int,
                           ax: plt.Axes, 
                           stimulus_shape: Tuple[int,...],
                           reducer_start: int,
                           reducer_length: int,
                           y_axis_lim: Tuple[float,float] | None= None,
                           ) -> None:
        """
        plots the reponses of one readout neuron to all meis of different objectivess.
        The optimization window is highlighted in yellow.
        """
        # fetch data
        bool_mask_neuron_idx = self.opt_stim_data_container["readout_idx"] == readout_idx
        data_subset = self.opt_stim_data_container[bool_mask_neuron_idx][["objective_name", "responses_all_readout_idx","roi_id"]]

        # objective names list
        objective_name_list = data_subset["objective_name"].tolist()
        nr_objectives = len(np.unique(objective_name_list))
        
        assert len(data_subset) == nr_objectives, f"Expected exactly {nr_objectives=} responses for neuron idx {readout_idx}, found {len(data_subset)}"
        assert len(data_subset["roi_id"].unique()) == 1, f"Expected exactly one roi_id for neuron idx {readout_idx}, found {data_subset['roi_id'].unique()}"                                                                                                   



        # get responses of meis. This is list of len nr objective_names, and has arrays of shape (time, nr readouts)
        responses_of_all_readout_idxs = data_subset["responses_all_readout_idx"].to_list()
        assert len(responses_of_all_readout_idxs) == nr_objectives 

        len_response = responses_of_all_readout_idxs[0].shape[0]
        response_start = stimulus_shape[2] - len_response 
        respones_end = stimulus_shape[2]
        stim_start = 1
        stim_end = stimulus_shape[2]
        opt_window_start = response_start + reducer_start
        opt_window_end = opt_window_start + reducer_length 



        x = np.arange(response_start + 1, respones_end + 1)
        min_response,max_response = np.inf, -np.inf
        for i,(obj_name, objective_responses_all_idx) in enumerate(zip(objective_name_list, responses_of_all_readout_idxs, strict=True)):
            target_idx_response = objective_responses_all_idx[:, readout_idx]
            min_response = min(min_response, np.min(target_idx_response))
            max_response = max(max_response, np.max(target_idx_response))
            ax.plot(x,target_idx_response, label=f"objective_name {obj_name}", color=self.colors[i], linestyle='-' if objective_name_list.index(obj_name) % 2 == 0 else '--')

        ax.set_xlabel("Time (frames)")
        ax.set_xlim(stim_start, stim_end)
        ax.set_ylabel("Response", fontsize=6)
        if y_axis_lim is None:
            y_axis_lim = (min_response - 0.1 * abs(min_response), max_response + 0.1 * abs(max_response))
        ax.set_ylim(y_axis_lim)
        
        # Highlight the optimization window
        ax.axvspan(opt_window_start , opt_window_end, color='yellow', alpha=0.3, label='Optimization Window')
        ax.legend(ncol=2, fontsize=6)
        ax.set_title(f"Model neuron responses to MEIs")





    def plot1(self,
              field_key: Dict[str, Any],
              roi_id: int,
              axs = None, 
              show = True) -> None:
        
        if not hasattr(self,"roi2readout_idx_wmeis"):
            raise ValueError("No readouts found with meis for this wrapper instance. \
                             Please run opt_stim_subanalysis first before plotting.")

        if roi_id not in self.roi2readout_idx_wmeis.keys():
            print(f"ROI {roi_id} does not have an MEI. Select among the following: \n{list(self.roi2readout_idx_wmeis.keys())}")
            return
        
        # sanity check field_key should be in CascadeSpikes 
        restricted_spikes = (self.dj_table_holder("CascadeSpikes")() & field_key & {'roi_id': roi_id})
        if len(restricted_spikes) == 0:
            raise ValueError (f"No spikes found in CascadeSpikes for roi_id {roi_id} and given field_key {field_key}. \
                              Please run check_requirements first.")

        if axs is None:
            fig,axs = plt.subplots(2,2,figsize=(8, 8))
        
        
        out = fetch_roi_data_from_container(self.opt_stim_data_container,
                                            field_key,
                                            self.roi2readout_idx_wmeis,
                                            roi_id)
        neuron_idx = out["neuron_idx"]
        objective_names_list = out["objective_names"]
        temporal_kernels_list = out["temporal_kernels_list"]
        spatial_kernels_list = out["spatial_kernels_list"]
        

        ## temporal kernels for obj_names. 
        # ax[0,0] has green temp kernels for obj_names ax[1,0] has uv temp kernels for obj_names
        plot_temporal_kernels(      colors=self.colors,
                                    readout_idx=neuron_idx,
                                   roi_id=roi_id,
                                   objective_names=objective_names_list,
                                   temporal_kernels_list=temporal_kernels_list,
                                   axs=[axs[0,0],axs[1,0]])

        ## spatial kernels in ax[0,1]
        plot_spatial_kernels(
                            readout_idx = neuron_idx,
                            roi_id=roi_id,
                            objective_names=objective_names_list,
                            spatial_kernels_list=spatial_kernels_list,
                            ax=axs[0, 1])

        ## ax[1,1] has responses
        self.plot_objective_respones(neuron_idx, 
                                ax=axs[1,1],
                                stimulus_shape=STIMULUS_SHAPE,
                                reducer_start=self.mei_generation_params["reducer_start"],
                                reducer_length=self.mei_generation_params["reducer_length"],) 
        if show: 
            plt.show()


    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        pass

    
    @property
    def name(self) -> str:
        return "Random Seed MEI"

    def check_requirements(self, 
                           field_key: Dict[str, Any],
                           roi_id_subset: Optional[List[int]] = None,
                           progress_callback: Optional[Callable] = None) -> None:
        """
        Check if the required tables are populated in the database.
        """

        # construct the complete restriction string
        complete_restriction = get_rois_in_field_restriction_str(field_key, roi_id_subset)

        progress: int = 0 
        
        if progress_callback is not None:
            progress_callback(0)

        # populate the traces table
        self.dj_table_holder("CascadeTraces")().populate(complete_restriction, processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        progress += 15
        if progress_callback is not None:
            progress_callback(progress)
        
        
        # spikes: no restriction, since the trstriction is in traces already and somehow
        # it has different primary keys
        self.dj_table_holder("CascadeSpikes")().populate( processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        progress += 15
        if progress_callback is not None:
            progress_callback(progress)
    


    def apply_quality_filter(self) -> None:
        
        n_neurons_before = len(self.neuron_testset_correls)
        neuron_idxs_passing_filter = []
        for neuron_idx, corr in self.neuron_testset_correls.items():
            if corr >= self.quality_filtering["min_testset_correl"]:
                neuron_idxs_passing_filter.append(neuron_idx)
        
        self.neuron_idxs_passing_filter = neuron_idxs_passing_filter
        nr_neurons_after = len(self.neuron_idxs_passing_filter)
        if nr_neurons_after < 6:
            lowest_allowed = sorted(self.neuron_testset_correls.values(), reverse=True)[5]
            raise ValueError (f"Pipeline requires at least 6 neurons in readout got {nr_neurons_after} with min_testset_correl of {self.quality_filtering["min_testset_correl"]}.\
              adjust quality_filtering[`min_testset_correl`] to {lowest_allowed} to get this, then call \
                wrapper.apply_quality_filter() and random_seed_mei_wrapper.opt_stim_subanalysis() again.\
                    testset correlations are: {self.neuron_testset_correls}")
        print(f"Filtered neurons based on testset correlation: {n_neurons_before} -> {nr_neurons_after}")

    def get_roi2rgb_and_alpha_255_map(self,
                                      field_key: Dict[str, Any],
                                      all_roi_ids:List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Get two mappings: one for roi to rgb based on whether there is an mei.
        all_roi_ids: a list of all roi ids that should be included in the mapping.
        
        """
        rgb_of_included = np.array([255,0,0]) # red for included rois
        rgb_nonincluded = np.array([0,0,255]) # blue for non-included rois
        alpha_of_included = 122.0 # full alpha for included rois
        alpha_nonincluded = 20.0
        

        roi2rgb255 = {roi: rgb_of_included if roi in self.roi2readout_idx_wmeis.keys() else rgb_nonincluded
                      for roi in all_roi_ids}
        roi2alpha = {roi: alpha_of_included if roi in self.roi2readout_idx_wmeis.keys() else alpha_nonincluded
                     for roi in all_roi_ids}

        return roi2rgb255, roi2alpha

    def upload_to_db(self,
                     field_key) -> None:
        """
        Uploads the generated stims and their responses to the database.
        """
        if len(self.opt_stim_data_container) == 0:
            raise ValueError("No stims generated. Call opt_stim_subanalysis first.")

        ## 1. upload raw data to db
        orhf_key = {**field_key,
                    "session_name": self.new_session_id,
                    "session_data_dict": self.session_dict_raw,}
        
        self.dj_table_holder("OpenRetinaHoeflingFormat")().insert1(
            orhf_key
        )

        ## 2. the stimuli, decomposition and responses 
        for i,row in self.opt_stim_data_container.iterrows():
            readout_idx = row["readout_idx"]
            objective_name = row["objective_name"]
            stimulus = row["optimized_stimulus"]
            ch0_temporal_kernel,ch1_temporal_kernel = row["temporal_kernels"]
            ch0_spatial_kernel,ch1_spatial_kernel = row["spatial_kernels"]
    
            response = row["responses_all_readout_idx"]
            roi_id = row["roi_id"]

            opt_stim_table_key = {**field_key,
                   "objective_name": objective_name, 
                   "readout_idx": readout_idx, 
                   "roi_id": roi_id,
                   **orhf_key,
                   "stimulus": stimulus.detach().cpu().numpy(), # store the array
                   }
            
            # insert to table 
            self.dj_table_holder("OnlineOptimizedStimulus")().insert1(
                {
                    **opt_stim_table_key,
                    
                },
            )

            # decomposition
            decomp_key = {**opt_stim_table_key,
                         "ch0_temporal_kernel": ch0_temporal_kernel,
                         "ch1_temporal_kernel": ch1_temporal_kernel,
                         "ch0_spatial_kernel": ch0_spatial_kernel,
                        "ch1_spatial_kernel": ch1_spatial_kernel,}
            self.dj_table_holder("StimulusDecomposition")().insert1(
                **decomp_key
            )

            # responses 
            ModelStimulusResponse_key = {
                **opt_stim_table_key,
                "response": response,

            }
            self.dj_table_holder("ModelStimulusResponse")().insert1(
                **ModelStimulusResponse_key
            )

                                    
            




        ## 3. the model

        # ## the model checkpoint
        # self.dj_table_holder("OnlineTrainedModel")().insert1(
        #     {
        #         **field_key,
        #         "session_name": self.new_session_id,
        #         "model_chkpt_path": self.best_model_ckp

        #     }
        # )
    
    def save_local_and_upload(self) -> None:
        """
        A wrapper to save all data to a local directory and then upload to db.
        """
        self.save_all_data_to_dir(self.save_dir_parent)
        self.upload_to_db(self.field_key)

        

    def save_all_data_to_dir(self, save_dir_parent: str) -> None:
        assert os.path.isdir(save_dir_parent), f"save_dir_parent {save_dir_parent} is not a directory."
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir_child = self.field_key["field"] + "_" + timestamp 
        save_dir = os.path.join(save_dir_parent,timestamp)
                
        os.makedirs(save_dir, exist_ok=True)

        # save raw session data
        session_dict_raw_path = os.path.join(save_dir, "session_dict_raw.pkl")
        with open(session_dict_raw_path, 'wb') as f:
            pickle.dump(self.session_dict_raw, f)
        print(f"Saved raw session dict to {session_dict_raw_path}")

        # save neuron data dict
        neuron_data_dict = self.neuron_data_dict
        neuron_data_dict_path = os.path.join(save_dir, "neuron_data_dict.pkl")
        with open(neuron_data_dict_path, 'wb') as f:
            pickle.dump(neuron_data_dict, f)
        print(f"Saved neuron data dict to {neuron_data_dict_path}")

        # save mei data container
        opt_stim_data_container_path = os.path.join(save_dir, "opt_stim_data_container.pkl")
        with open(opt_stim_data_container_path, 'wb') as f:
            pickle.dump(self.opt_stim_data_container, f)
        print(f"Saved MEI data container to {opt_stim_data_container_path}")

        # save models
        model_path_state_dict = os.path.join(save_dir, "model_state_dict.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # Optionally save other items like optimizer state
        }, model_path_state_dict)
        full_model_path = os.path.join(save_dir, "model_full.pt")
        torch.save(self.model,full_model_path)
        print(f"Saved model state dict to {model_path_state_dict}")
        print(f"Saved full model to {full_model_path}")


        # save metadata
        metadata = {
            "roi2readout_idx_wmeis": self.roi2readout_idx_wmeis,
            "roi_ids2readout_idx": self.roi_ids2readout_idx,
            "neuron_idxs_passing_filter": self.neuron_idxs_passing_filter,
            "neuron_testset_correls": self.neuron_testset_correls,
            "new_session_id": self.new_session_id,
            "scaled_means_before_centering": self.scaled_means_before_centering,
            "field_key": self.field_key,
        }

        # selected mei order based on activity type etc as in self.select_subset_of_meis_for_each_roi
        if hasattr(self, 'roi_id2mei_ids'):
            metadata["roi_id2mei_ids"] = self.roi_id2mei_ids
        if hasattr(self, 'roi_id2selected_mei_ids'):
            metadata["roi_id2info"] = self.roi_id2info

        metadata_path = os.path.join(save_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata to {metadata_path}")
    
    def load_all_data_from_dir(self, load_dir: str) -> None:
        """
        does the opposite of save_all_data_to_dir
        """
        # load raw session data
        session_dict_raw_path = os.path.join(load_dir, "session_dict_raw.pkl")
        with open(session_dict_raw_path, 'rb') as f:
            self.session_dict_raw = pickle.load(f)
        print(f"Loaded raw session dict from {session_dict_raw_path}")

        # load neuron data dict
        neuron_data_dict_path = os.path.join(load_dir, "neuron_data_dict.pkl")
        with open(neuron_data_dict_path, 'rb') as f:
            self.neuron_data_dict = pickle.load(f)
        print(f"Loaded neuron data dict from {neuron_data_dict_path}")

        # load mei data container
        opt_stim_data_container_path = os.path.join(load_dir, "opt_stim_data_container.pkl")
        with open(opt_stim_data_container_path, 'rb') as f:
            self.opt_stim_data_container = pickle.load(f)
        print(f"Loaded MEI data container from {opt_stim_data_container_path}")

        # # load model
        # model_path = os.path.join(load_dir, "model.pt")
        # checkpoint = torch.load(model_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.eval()
        # print(f"Loaded model from {model_path}")

        # load full model
        full_model_path = os.path.join(load_dir, "model_full.pt")
        self.model = torch.load(full_model_path)
        self.model.eval()
        print(f"Loaded full model from {full_model_path}")

        # load metadata
        metadata_path = os.path.join(load_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.roi2readout_idx_wmeis = metadata["roi2readout_idx_wmeis"]
        self.roi_ids2readout_idx = metadata["roi_ids2readout_idx"]
        self.neuron_idxs_passing_filter = metadata["neuron_idxs_passing_filter"]
        self.neuron_testset_correls = metadata["neuron_testset_correls"]
        self.new_session_id = metadata["new_session_id"]
        self.scaled_means_before_centering = metadata["scaled_means_before_centering"]
        self.field_key = metadata["field_key"]

        if "roi_id2mei_ids" in metadata:
            self.roi_id2mei_ids = metadata["roi_id2mei_ids"]
        if "roi_id2info" in metadata:
            self.roi_id2info = metadata["roi_id2info"]
        print(f"Loaded metadata from {metadata_path}, with keys: {list(metadata.keys())}")



    def fetch_from_db(self, field_key: Dict[str, Any] = {}) -> None:
        """
        =
        """
        pass
        ## TODO
        
    def select_subset_of_stims_for_each_roi(self) -> Tuple[Dict[int,List[str]], Dict[int, Dict[str, List[Any]]]]:
        """
        Selects a subset of stims for rach roi to show.
        reurns dict with roi_id as key and mei_id list as value.
        requieres (takes from self):
        opt_stim_data_container,
        neuron_data_dict
        
        Uses the following heuristic: 6 stims total: Three to four exciting and rest depressing stims 
        1. for a given roi_id,
            i) add its own stims (one for each objective)
            ii) if there is another cell with same type add its stims
            iii) Take a list of stims_ids sorted by response strength. fill up the list with stims_ids arrcording to the respnose strength until we reach 6.
  
        reuturn two dicts:
        roi_id2stim_ids: Dict[int, List[int]]: mapping from roi_id to list of stim_ids
        roi_id2info: Dict[int, Dict[str, Any]]: mapping from roi_id to dict with info about the selected stims (objective, celltype, responses

        """
        pass
        
        # # 1. map readout_idxwmei to cell type
        # readout_idx_groups = self.neuron_data_dict[self.new_session_id].session_kwargs["group_assignment"]
        # readout_idx_wmei2group = {idx:readout_idx_groups[idx] for idx in self.neuron_idxs_passing_filter}
        
        # # some checks
        # assert len(readout_idx_wmei2group) == len(self.opt_stim_data_container['readout_idx'].unique()), "Mismatch between readout idx with meis and neuron data dict."
        # assert all([idx in readout_idx_wmei2group.keys() for idx in self.opt_stim_data_container['readout_idx'].unique()]), "Some readout idx in mei data container not in neuron data dict."

        
        # ## fetch all mean repsonses array size (nr stims, nr readouts in model)
        # all_mean_responses = np.stack(self.opt_stim_data_container['mean_responses_all_readout_idx'].tolist(), axis=0)
        # assert all_mean_responses.shape[0] == len(self.opt_stim_data_container), "Mismatch between number of stims and mean responses."
        # assert all_mean_responses.shape[1] == len(self.neuron_data_dict[self.new_session_id].session_kwargs["group_assignment"]), "Mismatch between number of readouts in model and mean responses."        
        
        # ## 3. select stims ids for each roi based on the mean response in the time window and possibly cell type 
        # roi_id2mei_ids = {}
        # roi_id2info = {}
        # # loop over readout_idx
        # for readout_idx,celltype in readout_idx_wmei2group.items():
            
        #     # 3 a) get the necessary data for this readout idx
        #     # store the mei_ids for this roi/readout idx
        #     selected_stim_ids = []
        #     stim_responses = []
        #     celltypes_or_neurons_from_meis = []
        #     all_stabilites = []
            
        #     # get the roi_id
        #     roi_id = self.readout_idx_wmei2rois[readout_idx]

        #     # get all mei data for this readout idx
        #     all_stims_for_readout = self.opt_stim_data_container[self.opt_stim_data_container['readout_idx'] == readout_idx]
        #     assert len(all_stims_for_readout) > 0, f"No MEIs found for readout idx {readout_idx}."
            
        #     # the mean repsonses in the optimization window
        #     mean_responses_of_idx = all_mean_responses[:, readout_idx] # shape (nr_stimss,) 
            
        #     # 3 b) decide on mei_ids accroding to heuristic
        #     # step i) add its own meis (one if stable, two if unstable)
        #     own_meis = all_meis_for_readout['mei_id'].tolist()
        #     assert len(own_meis) == (1 if stability == 'stable' else 2), f"Unexpected number of own MEIs for readout idx {readout_idx} with stability {stability}."
        #     selected_mei_ids.extend(own_meis)



        #     # step ii) if there is another cell with same type add its mei (if there are mutliple seeds take one random)
        #     same_type_mei_entries = self.opt_stim_data_container[self.opt_stim_data_container['readout_idx'].isin(
        #         [idx for idx,grp in readout_idx_wmei2group.items() if grp == celltype and idx != readout_idx])]

        #     if len(same_type_mei_entries) > 0:
        #         # take one random mei from the same type
        #         random_same_type_mei_id = same_type_mei_entries.sample(n=1, random_state=42)['mei_id'].item()
        #         selected_mei_ids.append(random_same_type_mei_id)

        #     # step iii) Take a list of mei_ids sorted by response strength. 
        #     # fill up the list with mei_ids arrcording to the respnose strength until we reach 6. 
        #     # definately include the strongest and weakest one
        #     assert 1 <= len(set(selected_mei_ids)) <= 3, f"Unexpected number of MEIs selected so far for readout idx {readout_idx}: {len(selected_mei_ids)}."
        #     nr_missing_to_six = 6 - len(selected_mei_ids)
        #     sorted_mei_indices = np.argsort(mean_responses_of_idx)[::-1] # descending order
        #     remaining_mei_ids = [self.opt_stim_data_container.iloc[idx]['mei_id'] for idx in sorted_mei_indices if self.opt_stim_data_container.iloc[idx]['mei_id'] not in selected_mei_ids]
            
        #     # select evely but definately include strongerst
        #     step_size = len(remaining_mei_ids) / nr_missing_to_six 
        #     for i in range(nr_missing_to_six - 1): # -1 because we add the weakest one at the end
        #         selected_mei_ids.append(remaining_mei_ids[int(i * step_size)])

            
        #     # add the weakest one 
        #     selected_mei_ids.append(remaining_mei_ids[-1])

        #     # to have bettwe ovreview if its all corect we add the responses and celltypes of the selected meis
        #     for mei_id in selected_mei_ids:
        #         bool_mask_mei_ids = self.opt_stim_data_container['mei_id'] == mei_id
        #         mei_responses.extend(mean_responses_of_idx[bool_mask_mei_ids].tolist())
        #         celltypes_or_neurons_from_meis.extend([readout_idx_wmei2group[idx] for idx in self.opt_stim_data_container[bool_mask_mei_ids]['readout_idx'].tolist()])
        #         all_stabilites.extend(self.opt_stim_data_container[bool_mask_mei_ids]['stability'].tolist())

        #     # store the metadata
        #     roi_id2info[roi_id] = {
        #         "all_stabilities": all_stabilites,
        #         "celltype": celltypes_or_neurons_from_meis,
        #         "responses": mei_responses,
        #     }

        #     # store the mei_ids 
        #     roi_id2mei_ids[roi_id] = selected_mei_ids

        # # store for analysis
        # self.roi_id2mei_ids = roi_id2mei_ids
        # self.roi_id2info = roi_id2info

        # return roi_id2mei_ids, roi_id2info



                            
            

    def opt_stim_subanalysis(self,
                        ) -> None:
        """
        Does the following things:
        1. stores important mapping from reaout index passing quality filtering to roi_id and vice versa.
        2. Centers the readouts of the model.
        4. Generates the stims: for each objective, then It decomposes them, redoncstructs them if desired and stores them in a 
              data container.
        5. Gets the responses of all neurons in the readout to all meis and stores them in the data container. This is so we can use the
        mapping from roi_id to readout index with meis (passsing quality) and it refers to the column index of the responses.
        6. The data container has the following columns:
            - readout_idx: index of the neuron in the readout
            - roi_id: corresponding roi_id
            - opt_stim_id: roi_<roi id>_objective_<objective name>
            - opt_stim: optimized_stimulus
            - objective: increase or decreae
            - temporal_kernels: the temporal kernels of the mei decomposition (list of arrays, one per channel 0 - green 1 - UV)
            - spatial_kernels: the spatial kernels of the mei decomposition (list of 2d arrays, one per channel 0 - green 1 - UV)


        """

        if len(self.neuron_idxs_passing_filter) == 0:
                raise ValueError(f"No neurons to perform MEI analysis on.\
                                 \nSelect less strtict filtering criterium and call opt_stim_subanalysis again.\
                                 \nCurrent criterium: min_testset_correl = {self.mei_generation_params['min_testset_correl']} \
                                 \nTestset correlations: {self.neuron_testset_correls}")
        

        # map roi_id to model neuron idx
        self.roi2readout_idx_wmeis = {roi:idx for roi,idx in self.roi_ids2readout_idx.items() if idx in self.neuron_idxs_passing_filter}
        log(f"{self.roi2readout_idx_wmeis=}",self.log_dir)
        self.readout_idx_wmei2rois = {idx:roi for roi,idx in self.roi2readout_idx_wmeis.items()}
        log(f"{self.readout_idx_wmei2rois=}",self.log_dir)

        
        ## center the readouts            
        if self.model_configs["is_ensemble_model"]:
            self.scaled_means_before_centering = []
            for member in self.model.members:
                self.scaled_means_before_centering.append(get_model_gaussian_scaled_means(member,session= self.new_session_id)) # type: ignore
        elif isinstance(self.model, BaseCoreReadout):
            self.scaled_means_before_centering = get_model_gaussian_scaled_means(self.model,session= self.new_session_id)
        else:
            raise ValueError("Model is neither ensemble nor BaseCoreReadout. Cannot center readouts.")
        # apply centering
        center = Center(target_mean = 0.0)
        center(self.model)


        
        # initialize mei data containter
        opt_stim_data_container_entries = []

        ## generate l and mei
  
        neuron_ids_to_analyze = self.neuron_idxs_passing_filter
        neuron_objective_stim_dict = generate_opt_stim_mulitple_objectives(
                                        model = self.model,
                                        new_session_id = self.new_session_id,
                                        mei_generation_params= self.mei_generation_params,
                                        neuron_ids_to_analyze = neuron_ids_to_analyze, # NOTE: this will optimize each id individually 
                                        )
        print(f"Done with meis")
        print(f"Start decomposing ...")    
        
        
        
        ## decompose meis
        device = self.model.device if isinstance(self.model,BaseCoreReadout) else self.model.members[0].device
          
        for neuron_id,objective_stim_dict in neuron_objective_stim_dict.items():
            print(f"Decomposing MEIs for neuron (readout idx) {neuron_id} ...")
            for objective_name,opt_stim in objective_stim_dict.items():

                # decompose the MEIs
                temporal_kernels, spatial_kernels, _ = decompose_mei(stimulus = opt_stim.detach().cpu().numpy())
            

                if self.mei_generation_params["reconstruct_mei"]:
                    reconstruction = reconstruct_mei_from_decomposed(
                                temporal_kernels=temporal_kernels,
                                spatial_kernels=spatial_kernels,)

                    reconstruction = torch.tensor(reconstruction,dtype=torch.float32).to(device)
                    assert reconstruction.shape == opt_stim.shape, "Reconstructed MEI shape does not match original MEI shape."
                    
                    # make reonstruction same norm as mei
                    print(f"changing norm of reconstruction {torch.norm(reconstruction)} to match original mei norm {torch.norm(opt_stim)}")
                    reconstruction = reconstruction / torch.norm(reconstruction) * torch.norm(opt_stim)
                    print(f"new reconstruction norm {torch.norm(reconstruction)}")
                    opt_stim = reconstruction # use the reconstructed MEI for further analysis
                    print(f"Done reconstructing MEI for neuron (readout idx) {neuron_id}, objective {objective_name}.")
                    # add entry to data container 
                    opt_stim_data_container_entries.append({
                        "opt_stim_id": f"roi_{self.readout_idx_wmei2rois[neuron_id]}_objective_{objective_name}",
                        "readout_idx": neuron_id,
                        "roi_id": self.readout_idx_wmei2rois[neuron_id],
                        "objective_name": objective_name,
                        "opt_stim": opt_stim.detach(),
                        "temporal_kernels": temporal_kernels,
                        "spatial_kernels": spatial_kernels,
                    })
        

                    

        # make df container from all meis
        self.opt_stim_data_container = pd.DataFrame(opt_stim_data_container_entries)
        
        print(f"Generating responses for neurons in readout {len(self.neuron_idxs_passing_filter)} to all opt stims {len(self.opt_stim_data_container)} ...")
        self.get_responses_and_add_to_container()


    def get_responses_and_add_to_container(self) -> None:
        """
        Adds two columns to the mei data container: 
        - one with the resonses of each neuron EACH NEURON IN READOUT to all meis THIS IS NOT FILTERED FOR QUALITY. 
        This so that later the index of the column corresponds to the index of the neuorns in the readouts. 
        - one with the mean of that response in the optiization time window.
        """

        ## responses: use batch of meis
        opt_stim_batch = self.opt_stim_data_container['opt_stim'].to_list()
        device = self.model.device if isinstance(self.model,BaseCoreReadout) else self.model.members[0].device
        opt_stim_batch = torch.stack(opt_stim_batch, dim=0).to(device)
        all_neuron_ids_in_readout = list(range(self.neuron_data_dict[self.new_session_id].session_kwargs["roi_ids"].shape[0]))
        all_responses = get_model_mei_response(model = self.model,
                                                mei=opt_stim_batch,
                                                session_id = self.new_session_id,
                                                neuron_id = all_neuron_ids_in_readout,)
        
        # resonses are shape nr meis, nr time points response, nr neurons in readout 
        assert all_responses.shape[0] == opt_stim_batch.shape[0], "Number of responses does not match number of MEIs."
        assert all_responses.shape[2] == len(all_neuron_ids_in_readout)
        
        # store the responses in the data container
        self.opt_stim_data_container['responses_all_readout_idx'] = list(all_responses)

        # reduce responses to the mean in the response window/ optimization time window.
        t0 =self.mei_generation_params["reducer_start"]
        t1 = t0 + self.mei_generation_params["reducer_length"]
        all_mean_responses = np.mean(all_responses[:,t0:t1,:], axis=1) # shape (nr_meis, nr_neurons in readout )
        assert all_mean_responses.shape[0] == opt_stim_batch.shape[0], "Number of mean responses does not match number of MEIs."
        assert all_mean_responses.shape[1] == len(all_neuron_ids_in_readout), "Number of mean responses does not match number of neurons."
        self.opt_stim_data_container['mean_responses_all_readout_idx'] = list(all_mean_responses)





    def extract_and_preprocess(self) -> None:
        
        ## model training 
        self.session_dict_raw = self.dj_table_holder('OpenRetinaHoeflingFormat')().extract_data()
        
        # preprocess and bring to open retina classes 
        self.movies_dict = load_stimuli(self.model_configs)

        self.neuron_data_dict =  make_final_responses(self.session_dict_raw,response_type="natural") 
    
    def train_model(self) -> None:

        # load and refine model
        self.model,self.neuron_testset_correls,self.best_model_ckp = train_model_online(self.model_configs,
                                                                    self.neuron_data_dict,
                                                                    self.movies_dict)


        # store eome data
        self.new_session_id = list(self.session_dict_raw.keys())[0]
        
        # mappings from roi_id to to model_neuron idx
        self.roi_ids2readout_idx = {roi:idx for idx,roi in enumerate(self.neuron_data_dict[self.new_session_id].session_kwargs["roi_ids"].tolist())}



    def compute_analysis(self,
                        field_key,
                         roi_id_subset: Optional[List[int]] = None,
                         progress_callback: Optional[Callable] = None,
                         save_data = True) -> None:

        # extract data in hoefling format from DB 
        if len(self.dj_table_holder('OpenRetinaHoeflingFormat')() & field_key) == 0:
            if progress_callback is not None:
                progress_callback(0)
            
            if hasattr(self, 'field_key'):
                if self.field_key == field_key:
                    print("WARNING: field_key in wrapper is the same as the one being passed. Overwriting anyway.")
            self.field_key = field_key  

            self.check_requirements(field_key,
                                    roi_id_subset=roi_id_subset,
                                    progress_callback= progress_callback)

            if progress_callback is not None:
                progress_callback(30)
            
            # fetch data and train model
            self.extract_and_preprocess()
            self.train_model()
            
            # quality filter neurons
            self.apply_quality_filter()   

            if progress_callback is not None:
                progress_callback(60)

            ## MEI generation
            self.opt_stim_subanalysis()

            if progress_callback is not None:
                progress_callback(100)
            
            if save_data:
                self.save_local_and_upload()
            

        else:
            print("OpenRetinaHoeflingFormat table is already populated for the given field_key. Skipping analysis.")
            
