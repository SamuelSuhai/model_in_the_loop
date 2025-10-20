from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base import DJComputeWrapper, DJTableHolder
from omegaconf import DictConfig
from model_in_the_loop.utils.stimulus_optimization import (reconstruct_mei_from_decomposed,center_member_or_ensemble_readouts,
                                                           generate_opt_stim_for_neuron_list,
                                                            decompose_mei, get_model_mei_response,Center,get_model_gaussian_scaled_means,
                                                            STIMULUS_SHAPE,FRAME_RATE_MODEL
                                                           )

from model_in_the_loop.utils.datajoiont_utils import get_rois_in_field_restriction_str
import pickle
import torch
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.layers.ensemble import EnsembleModel
from openretina.data_io.hoefling_2024.responses import make_final_responses,filter_responses
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.utils.h5_handling import load_h5_into_dict


from model_in_the_loop.utils.model_training import (load_stimuli, 
                                                    train_or_refine_member_or_ensemble,
                                                    get_single_neuron_split_predictions,
                                                    get_single_neuron_session_correlations,
                                                    get_dataloaders_and_data_info,
                                            )
from model_in_the_loop.utils.mei_subset_selection import select_subset_of_meis_for_each_roi
from model_in_the_loop.utils.stimulus_optimization import generate_deis

from model_in_the_loop.utils.simple_logging import log

class RandomSeedMEIWrapper(DJComputeWrapper):

    def __init__(self,dj_table_holder: DJTableHolder,
                cfg: DictConfig,
                 seeds: List[int],

                ) -> None:
        
        self.dj_table_holder = dj_table_holder

      
        self.model_configs = cfg.model_configs
        self.mei_generation_params = cfg.stimulus_optimization
        self.quality_filtering = cfg.quality_filtering
        self.save_dir_parent = os.path.join(cfg.paths.repo_directory,
                                            "model_in_the_loop/data/online_computed_data" )


        self.seeds = seeds
        self.colors = plt.cm.nipy_spectral(np.linspace(0, 1,len(self.seeds)))

    def clear_tables(self, field_key, safemode=True) -> None:
        """
        Clears the tables the wrapper populates"""
        (self.dj_table_holder("CascadeTraces")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("CascadeSpikes")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OpenRetinaHoeflingFormat")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OnlineMEIs")() & field_key).delete(safemode=safemode)
        (self.dj_table_holder("OnlineTrainedModel")() & field_key).delete(safemode=safemode)



    def plot_seed_respones(self,
                           readout_idx: int,
                           ax: plt.Axes, 
                           stimulus_shape: Tuple[int,...],
                           reducer_start: int,
                           reducer_length: int,
                           y_axis_lim: Tuple[float,float] | None= None,
                           ) -> None:
        """
        plots the reponses of one readout neuron to all meis of different seeds.
        The optimization window is highlighted in yellow.
        """
        # fetch data
        bool_mask_neuron_idx = self.mei_data_container["readout_idx"] == readout_idx
        data_subset = self.mei_data_container[bool_mask_neuron_idx][["seed", "responses_all_readout_idx","roi_id"]]
        assert len(data_subset) <= len(self.seeds), f"Expected at most {len(self.seeds)} seeds for neuron idx {readout_idx}, found {len(data_subset)}"
        assert len(data_subset["roi_id"].unique()) == 1, f"Expected exactly one roi_id for neuron idx {readout_idx}, found {data_subset['roi_id'].unique()}"                                                                                                   

        # seeds as a list
        seeds = data_subset["seed"].tolist()

        # get responses of meis. This is list of len nr seeds, and has arrays of shape (time, nr readouts)
        responses_of_all_readout_idxs = data_subset["responses_all_readout_idx"].to_list()
        assert len(responses_of_all_readout_idxs) <= len(self.seeds)

        len_response = responses_of_all_readout_idxs[0].shape[0]
        response_start = stimulus_shape[2] - len_response 
        respones_end = stimulus_shape[2]
        stim_start = 1
        stim_end = stimulus_shape[2]
        opt_window_start = response_start + reducer_start
        opt_window_end = opt_window_start + reducer_length 



        x = np.arange(response_start + 1, respones_end + 1)
        min_response,max_response = np.inf, -np.inf
        for i,(seed, seed_responses_all_idx) in enumerate(zip(seeds, responses_of_all_readout_idxs, strict=True)):
            target_idx_response = seed_responses_all_idx[:, readout_idx]
            min_response = min(min_response, np.min(target_idx_response))
            max_response = max(max_response, np.max(target_idx_response))
            ax.plot(x,target_idx_response, label=f"Seed {seed}", color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')

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


    def plot_temporal_kernels(self,
                              readout_idx: int,
                              roi_id: int,
                              seeds: List[int],
                              temporal_kernels_list: List[np.ndarray],
                              axs: List[plt.Axes]) -> None:
        """
        Plots the temporal kernels for all seeds for a single neuron.
        Green is in axs[0], UV in axs[1]
        all seeds are plotted for each channel
        """


        # stack temporal_kernels into one array of shape (n_seeds, 2, timepoints)
        temporal_kernels = np.stack(temporal_kernels_list,axis=0)

        # split channels
        green_temporal_kernels = temporal_kernels[:, 0, :]
        uv_temporal_kernels = temporal_kernels[:, 1, :]

        x = np.arange(green_temporal_kernels.shape[1]) / FRAME_RATE_MODEL

        # y axis limits
        glob_max = max(np.max(green_temporal_kernels), np.max(uv_temporal_kernels))
        glob_min = min(np.min(green_temporal_kernels), np.min(uv_temporal_kernels))


        # plotting
        for i,(seed, kernel) in enumerate(zip(seeds, green_temporal_kernels)):
            axs[0].plot(x, kernel, label=f"seed {seed}", color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')

        for i,(seed, kernel) in enumerate(zip(seeds, uv_temporal_kernels)):
            axs[1].plot(x, kernel, label=f"seed {seed}", color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')

        # labeling
        axs[0].set_title(f"Green Temporal Kernels for ROI {roi_id} (neuron idx {readout_idx})", fontsize=6)
        axs[0].set_ylabel("Amplitude")
        axs[1].set_title(f"UV ROI {roi_id} (neuron idx {readout_idx})", fontsize=6)
        axs[1].set_ylabel("Amplitude")
        axs[1].set_xlabel("Time (s)")
        axs[0].set_ylim(glob_min - 0.1 * abs(glob_min), glob_max + 0.1 * abs(glob_max))
        axs[1].set_ylim(glob_min - 0.1 * abs(glob_min), glob_max + 0.1 * abs(glob_max))
        axs[0].legend(fontsize=6)
        axs[1].legend(fontsize=6)

    def plot_spatial_kernels(self, 
                             readout_idx: int,
                             roi_id: int,
                             seeds: List[int],
                             spatial_kernels_list: List[List[np.ndarray]], 
                            ax: plt.Axes) -> None:
        """
        Written by AI: 
        Plots the spatial kernels in the following way:
        - For each seed, green and UV channels are concatenated side by side
        - Different seeds are stacked vertically
        - Small labels above each seed indicate the seed number
        """
        
        
        # Calculate total height needed for all seeds
        total_height = 0
        all_combined_kernels = []
        
        # Process each seed's spatial kernels
        for i, (seed, kernel_pair) in enumerate(zip(seeds, spatial_kernels_list)):
            # Concatenate green and UV kernels horizontally
            combined_kernel = np.concatenate(kernel_pair, axis=1)
            all_combined_kernels.append(combined_kernel)
            total_height += combined_kernel.shape[0]
        
        # Add small gaps between seeds (10% of kernel height)
        gap_height = max(1, int(all_combined_kernels[0].shape[0] * 0.1))
        total_height += gap_height * (len(seeds) - 1)
        
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
            
            # Add seed label above each kernel
            ax.text(w//2, y_offset - 2, f"Seed {seeds[i]}", 
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


    def plot1(self,
              field_key: Dict[str, Any],
              roi_id: int,
              axs = None, 
              show = True) -> None:
        
        if not hasattr(self,"roi2readout_idx_wmeis"):
            print("No readouts found with meis for this wrapper instance. Please run mei_subanalysis first before plotting.")
            return

        if roi_id not in self.roi2readout_idx_wmeis.keys():
            print(f"ROI {roi_id} does not have an MEI. Select among the following: \n{list(self.roi2readout_idx_wmeis.keys())}")
            return
        
        # sanity check field_key should be in CascadeSpikes 
        restricted_spikes = (self.dj_table_holder("CascadeSpikes")() & field_key & {'roi_id': roi_id})
        if len(restricted_spikes) == 0:
            raise ValueError (f"No spikes found in CascadeSpikes for roi_id {roi_id} and given field_key {field_key}. \
                              Please run check_requirements first.")

        
        # find neuron_id for roi_id
        neuron_idx = self.roi2readout_idx_wmeis[roi_id]
        if axs is None:
            fig,axs = plt.subplots(2,2,figsize=(8, 8))

        # subset the data
        bool_mask_neuron_idx = self.mei_data_container["readout_idx"] == neuron_idx
        data_subset = self.mei_data_container[bool_mask_neuron_idx][["seed", "temporal_kernels","spatial_kernels","roi_id"]]
        assert len(data_subset) <= len(self.seeds), f"Expected at most {len(self.seeds)} seeds for neuron idx {neuron_idx}, found {len(data_subset)}"
        assert len(data_subset["roi_id"].unique()) == 1, f"Expected exactly one roi_id for neuron idx {neuron_idx}, found {data_subset["roi_id"].unique()}"
        roi_id = data_subset["roi_id"].iloc[0]
        seeds = data_subset["seed"].tolist()
        
        # Get all spatial kernels for this neuron (all seeds)
        spatial_kernels_list = data_subset["spatial_kernels"].tolist()
        temporal_kernels_list = data_subset["temporal_kernels"].to_list()

        ## temporal kernels for seeds. 
        # ax[0,0] has green temp kernels for seeds ax[1,0] has uv temp kernels for seeds
        self.plot_temporal_kernels(neuron_idx,
                                   roi_id=roi_id,
                                   seeds=seeds,
                                   temporal_kernels_list=temporal_kernels_list,
                                   axs=[axs[0,0],axs[1,0]])

        ## spatial kernels in ax[0,1]
        self.plot_spatial_kernels(neuron_idx,
                                  roi_id=roi_id,
                                   seeds=seeds,
                                    spatial_kernels_list=spatial_kernels_list,
                                      ax=axs[0, 1])

        ## ax[1,1] has responses
        self.plot_seed_respones(neuron_idx, 
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
        self.dj_table_holder("CascadeSpikes")().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        
        # assert equal length
        if len(self.dj_table_holder("CascadeTraces")()) != len(self.dj_table_holder("CascadeSpikes")()):
            raise ValueError("CascadeTraces and CascadeSpikes tables have different number of entries after population. Delete cascade entries first then repeat")

        progress += 15
        if progress_callback is not None:
            progress_callback(progress)
    


    def apply_quality_filter(self, min_nr_neurons_post = 6) -> None:

        session_neuron_correls = self.neuron_testset_correls[self.new_session_id]
        n_neurons_before = len(session_neuron_correls)
        neuron_idxs_passing_filter = []
        for neuron_idx, corr in session_neuron_correls.items():
            if corr >= self.quality_filtering["min_testset_correl"]:
                neuron_idxs_passing_filter.append(neuron_idx)
        
        self.neuron_idxs_passing_filter = neuron_idxs_passing_filter
        nr_neurons_after = len(self.neuron_idxs_passing_filter)
        if nr_neurons_after < min_nr_neurons_post:
            lowest_allowed = sorted(session_neuron_correls.values(), reverse=True)[5]
            raise ValueError (f"Pipeline requires at least {min_nr_neurons_post} neurons in readout got {nr_neurons_after} with min_testset_correl of {self.quality_filtering["min_testset_correl"]}.\
              adjust quality_filtering[`min_testset_correl`] to {lowest_allowed} to get this, then call \
                random_seed_mei_wrapper.apply_quality_filter() and random_seed_mei_wrapper.mei_subanalysis() again.\
                    testset correlations are: {session_neuron_correls}")
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
        Uploads the generated MEIs and their responses to the database.
        """
        if len(self.mei_data_container) == 0:
            raise ValueError("No MEIs generated. Call mei_subanalysis first.")

        ## 1. upload raw data to db
        orhf_key = {**field_key,
                    "session_name": self.new_session_id,
                    "session_data_dict": self.session_dict_raw,}
        
        self.dj_table_holder("OpenRetinaHoeflingFormat")().insert1(
            orhf_key
        )

        ## 2. the meis 
        for i,row in self.mei_data_container.iterrows():
            readout_idx = row["readout_idx"]
            seed = row["seed"]
            mei = row["mei"]
            response = row["responses_all_readout_idx"]
            roi_id = row["roi_id"]

            key = {**field_key,
                   "seed": seed, 
                   "readout_idx": readout_idx, 
                   "roi_id": roi_id,
                   "session_name": self.new_session_id,
                   }
            
            # insert to table 
            self.dj_table_holder("OnlineMEIs")().insert1(
                {
                    **key,
                    "mei": mei.detach().cpu().numpy(), # store the array
                },
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
    

        

    def save_all_data_to_dir(self, save_dir_parent: str) -> None:
        assert os.path.isdir(save_dir_parent), f"save_dir_parent {save_dir_parent} is not a directory."
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir_child = self.field_key["field"] + "_" + timestamp 
        save_dir = os.path.join(save_dir_parent,save_dir_child)
                
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
        mei_data_container_path = os.path.join(save_dir, "mei_data_container.pkl")
        with open(mei_data_container_path, 'wb') as f:
            pickle.dump(self.mei_data_container, f)
        print(f"Saved MEI data container to {mei_data_container_path}")

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
        mei_data_container_path = os.path.join(load_dir, "mei_data_container.pkl")
        with open(mei_data_container_path, 'rb') as f:
            self.mei_data_container = pickle.load(f)
        print(f"Loaded MEI data container from {mei_data_container_path}")

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
    
    @staticmethod
    def select_subset_of_meis_for_each_roi( only_consider_these_rois: List[int],
                                           neuron_data_dict: Dict[str,ResponsesTrainTestSplit],
                                           new_session_id: str,
                                           mei_data_container: pd.DataFrame,
                                           readout_idx_wmei2rois: Dict[int, int],
                                           n_stimuli_total = 6,
                                           ) -> Tuple[Dict[int,List[str]], Dict[int, Dict[str, List[Any]]]]:
        """
        
        """
        roi_id2mei_ids, roi_id2info = select_subset_of_meis_for_each_roi(only_consider_these_rois,
                                           neuron_data_dict,
                                           new_session_id,
                                           mei_data_container,
                                           readout_idx_wmei2rois,
                                           n_stimuli_total = n_stimuli_total,
                                           )
        return roi_id2mei_ids, roi_id2info



                            
            

    def mei_subanalysis(self,
                        ) -> None:
        """
        Does the following things:
        1. stores important mapping from reaout index passing quality filtering to roi_id and vice versa.
        2. Centers the readouts of the model.
        3. Decides which neurons get stable and which unstable meis.
        4. Generates the meis: first for unstable then stable. It decomposes them, redoncstructs them if desired and stores them in a 
              data container.
        5. Gets the responses of all neurons in the readout to all meis and stores them in the data container. This is so we can use the
        mapping from roi_id to readout index with meis (passsing quality) and it refers to the column index of the responses.
        6. The data container has the following columns:
            - readout_idx: index of the neuron in the readout
            - roi_id: corresponding roi_id
            - mei_id: unique id for the mei (roi_id and seed) of form "roi_{roi_id}_seed_{seed}"
            - seed: random seed used to generate the mei
            - mei: the mei itself as torch tensor
            - temporal_kernels: the temporal kernels of the mei decomposition (list of arrays, one per channel 0 - green 1 - UV)
            - spatial_kernels: the spatial kernels of the mei decomposition (list of 2d arrays, one per channel 0 - green 1 - UV)


        """

        if len(self.neuron_idxs_passing_filter) == 0:
                raise ValueError(f"No neurons to perform MEI analysis on.\
                                 \nSelect less strtict filtering criterium and call mei_subanalysis again.\
                                 \nCurrent criterium: min_testset_correl = {self.mei_generation_params['min_testset_correl']} \
                                 \nTestset correlations: {self.neuron_testset_correls}")
        

        # map roi_id to model neuron idx
        self.roi2readout_idx_wmeis = {roi:idx for roi,idx in self.roi_ids2readout_idx.items() if idx in self.neuron_idxs_passing_filter}
        log(f"{self.roi2readout_idx_wmeis=}")
        self.readout_idx_wmei2rois = {idx:roi for roi,idx in self.roi2readout_idx_wmeis.items()}
        log(f"{self.readout_idx_wmei2rois=}")

        
        ## center the readouts            
        self.scaled_means_before_centering = center_member_or_ensemble_readouts(model = self.model,
                                                                                new_session_id=self.new_session_id
                                                                                )



        ## decide which neurons get stable and which unstable meis
        idx2stability = self.get_stable_unstable_split(self.neuron_idxs_passing_filter)
        log(f"{idx2stability=}")
        
        # initialize mei data containter
        mei_data_container_entries = []

        # ## generate meis
        # for phase in ['unstable', 'stable']:
        #     print(f"Generating {phase} MEIs for neurons (readout idx): {[idx for idx,stab in idx2stability.items() if stab ==phase ]}.")
        #     log(f"Generating {phase} MEIs for neurons (readout idx): {[idx for idx,stab in idx2stability.items() if stab == phase ]}.")
        #     set_model_to_eval_mode = True if phase == 'stable' else False
        #     neuron_ids_to_analyze = [neuron_id for neuron_id, stability in idx2stability.items() if stability == phase]
        #     seeds = self.seeds if phase == 'unstable' else [self.seeds[0]] # only one seed for stable meis
        #     neuron_seed_mei_dict =  generate_opt_stim_for_neuron_list(
        #                                     model = self.model,
        #                                     new_session_id = self.new_session_id,
        #                                     opt_stim_generation_params= self.mei_generation_params,
        #                                     random_seeds = seeds,
        #                                     seed_it_func= torch.manual_seed,
        #                                     neuron_ids_to_analyze = neuron_ids_to_analyze, # NOTE: this will optimize each id individually 
        #                                     set_model_to_eval_mode = set_model_to_eval_mode, # model in training mode for noisy MEIs
        #                                     )
        #     print(f"Done with meis in phase {phase}.")
        
        seeds = [self.seeds[0]] 
        for phase in ['unstable', 'stable']:
            print(f"Generating {phase} MEIs for neurons (readout idx): {[idx for idx,stab in idx2stability.items() if stab ==phase ]}.")
            log(f"Generating {phase} MEIs for neurons (readout idx): {[idx for idx,stab in idx2stability.items() if stab == phase ]}.")
            neuron_ids_to_analyze = [neuron_id for neuron_id, stability in idx2stability.items() if stability == phase]
            neuron_seed_mei_dict =  generate_opt_stim_for_neuron_list(
                                            model = self.model,
                                            new_session_id = self.new_session_id,
                                            opt_stim_generation_params= self.mei_generation_params,
                                            random_seeds = seeds,
                                            seed_it_func= torch.manual_seed,
                                            neuron_ids_to_analyze = neuron_ids_to_analyze, # NOTE: this will optimize each id individually 
                                            set_model_to_eval_mode = True, 
                                            )
            print(f"Done with meis in phase {phase}.")
            
            ## DEIS
            if phase == "unstable":
                # replace the meis with DEIS and add a second seed as key
                print(f"DEI GENERATION... ADD SECOND SEED DEI")
                # diversify 
                for neuron,seed_mei_dict in neuron_seed_mei_dict.items():
                    assert len(seed_mei_dict) == 1, "Expected only one seed for DEIS generation."
                    mei = seed_mei_dict[self.seeds[0]]
                    print(f"DIVERSIFYING MEI neuron id {neuron} with seed {self.seeds[0]} ...")
                    deis = generate_deis(
                        model=self.model,
                        mei = mei,
                        neuron_id = neuron,
                        session_id = self.new_session_id,
                        n_deis= len(self.seeds),
                        opt_stim_generation_params= self.mei_generation_params,
                    )
                    neuron_seed_mei_dict[neuron] = {seed:deis[i] for i,seed in enumerate(self.seeds)}
            



            
            print(f"Start decomposing ...")    
            ## decompose meis
            device = self.model.device if isinstance(self.model,BaseCoreReadout) else self.model.members[0].device
            for neuron_id,seed_dict in neuron_seed_mei_dict.items():
                print(f"Decomposing MEIs for neuron (readout idx) {neuron_id} ...")
                for seed,mei in seed_dict.items():

                    # decompose the MEIs
                    temporal_kernels, spatial_kernels, _ = decompose_mei(stimulus = mei.detach().cpu().numpy())
               

                    if self.mei_generation_params["reconstruct_mei"]:
                        reconstruction = reconstruct_mei_from_decomposed(
                                    temporal_kernels=temporal_kernels,
                                    spatial_kernels=spatial_kernels,
                                    turn_to_tensor=True)

                        assert reconstruction.shape == mei.shape, "Reconstructed MEI shape does not match original MEI shape."
                        
                        # make reonstruction same norm as mei
                        print(f"changing norm of reconstruction {torch.norm(reconstruction)} to match original mei norm {torch.norm(mei)}")
                        reconstruction = reconstruction / torch.norm(reconstruction) * torch.norm(mei)
                        print(f"new reconstruction norm {torch.norm(reconstruction)}")
                        mei = reconstruction # use the reconstructed MEI for further analysis
                        print(f"Done reconstructing MEI for neuron (readout idx) {neuron_id}, seed {seed}.")
                    
                    # add entry to data container 
                    mei_data_container_entries.append({
                        "readout_idx": neuron_id,
                        "roi_id": self.readout_idx_wmei2rois[neuron_id],
                        "mei_id": f"roi_{self.readout_idx_wmei2rois[neuron_id]}_seed_{seed}",
                        "seed": seed,
                        "mei": mei.detach(),
                        "temporal_kernels": temporal_kernels,
                        "spatial_kernels": spatial_kernels,
                        "stability": phase,
                    })
            

                        

        # make df container from all meis
        self.mei_data_container = pd.DataFrame(mei_data_container_entries)
        
        print(f"Generating responses for neurons in readout {len(self.neuron_idxs_passing_filter)} to all meis {len(self.mei_data_container)} ...")
        self.get_responses_and_add_to_container(mei_data_container=self.mei_data_container,
                                               model=self.model,
                                               new_session_id=self.new_session_id,
                                               neuron_data_dict=self.neuron_data_dict,
                                               mei_generation_params=self.mei_generation_params,)

    @staticmethod
    def get_responses_and_add_to_container(mei_data_container: pd.DataFrame,
                                           model:BaseCoreReadout |EnsembleModel,
                                           new_session_id: str,
                                           neuron_data_dict: Dict[str,ResponsesTrainTestSplit],
                                           mei_generation_params: Dict[str, Any]) -> None:
        get_responses_and_add_to_container(mei_data_container,
                                           model,
                                           new_session_id,
                                           neuron_data_dict,
                                           mei_generation_params)



    @staticmethod
    def get_stable_unstable_split(readout_idxs: List[int],seed = 42) -> Dict[int, str]:
        """
        Randomly selects 1/5 of neurons (or two which ever is larger) to generate unstable MEIs for.
        Outputs a dict with key neuron_redout_idx, and value `stable` or `unstable`."""
        np.random.seed(seed) 
        n_unstable = max(2, len(readout_idxs) // 5)
        unstable_neurons = np.random.choice(readout_idxs, size=n_unstable, replace=False)
        stability_dict = {idx: 'unstable' if idx in unstable_neurons else 'stable' for idx in readout_idxs}
        return stability_dict



    def extract_data(self,
                    field_key = None,) -> None:
        
        # check for what training mode
        only_train_readout = self.model_configs.get("only_train_readout", True)

        # extract data in hoefling format from DB
        self.session_dict_raw = self.dj_table_holder('OpenRetinaHoeflingFormat')().extract_data(field_key=field_key)
        all_session_keys = list(self.session_dict_raw.keys())
        assert len(all_session_keys) == 1
        self.new_session_id = all_session_keys[0]


        # add the openretina training data if desired
        if not only_train_readout:
            responses_path_local = self.model_configs.get("responses_path_local", "")
            assert os.path.isfile(responses_path_local), f"{responses_path_local} is not a file."
            raw_session_loaded = load_h5_into_dict(responses_path_local)
            
            ### DEBUG remove keys that have 20200226 in them
            keys_to_remove = [key for key in raw_session_loaded.keys() if '20200226' in key]
            for key in keys_to_remove:
                del raw_session_loaded[key]

            # filter the raw seesions 
            raw_session_loaded = filter_responses(raw_session_loaded, self.model_configs.quality_checks)
            
            self.session_dict_raw.update(raw_session_loaded)
            print(f"Loaded additional openretina data from {responses_path_local} and added to session_dict_raw.")

        

    def format_data(self) -> None:
        # preprocess and bring to open retina classes 
        self.movies_dict = load_stimuli(self.model_configs)

        self.neuron_data_dict =  make_final_responses(self.session_dict_raw,response_type="natural") 
        



    def compute_analysis(self, field_key,
                         roi_id_subset: Optional[List[int]] = None,
                         progress_callback: Optional[Callable] = None,
                         save_local: bool = False,
                         upload_to_db: bool = False,) -> None:
        
        if progress_callback is None:
            progress_callback = lambda x: None


        # extract data in hoefling format from DB 
        if len(self.dj_table_holder('OpenRetinaHoeflingFormat')() & field_key) == 0:
            progress_callback(0)
            
            if hasattr(self, 'field_key'):
                if self.field_key == field_key:
                    print("WARNING: field_key in wrapper is the same as the one being passed. Overwriting anyway.")
            self.field_key = field_key  

            self.check_requirements(field_key,
                                    roi_id_subset=roi_id_subset,
                                    progress_callback= progress_callback)

            progress_callback(30)
            
            # fetch data and train model
            self.extract_data(field_key=field_key)
            self.format_data()
            
            # mappings from roi_id to to model_neuron idx
            self.roi_ids2readout_idx = {roi:idx for idx,roi in enumerate(self.neuron_data_dict[self.new_session_id].session_kwargs["roi_ids"].tolist())}

            
            # get dataloader
            dataloaders, data_info = get_dataloaders_and_data_info(
                cfg = self.model_configs,
                neuron_data_dict = self.neuron_data_dict,
                movies_dict = self.movies_dict,
            )

            # train or refine model
            self.model,self.best_model_ckp = train_or_refine_member_or_ensemble(
                model_configs = self.model_configs,
                dataloaders = dataloaders,
                data_info = data_info,
            )

            # evaluate model
            all_preds, all_targets = get_single_neuron_split_predictions(
                dataloaders = dataloaders,
                model = self.model,
                split = 'test',
                only_this_session_id = self.new_session_id, # only for the new session 
                )
            
            self.neuron_testset_correls = get_single_neuron_session_correlations(
            all_preds = all_preds,
            all_targets = all_targets,)            
            progress_callback(70)
            

            # quality filter neurons
            self.apply_quality_filter()   

            progress_callback(60)

            ## MEI generation
            self.mei_subanalysis()

            progress_callback(100)
            
            # save locally
            if save_local:
                self.save_all_data_to_dir(save_dir_parent=self.save_dir_parent)
            
            # upload to db
            if upload_to_db:
                self.upload_to_db(field_key)
            


        else:
            print("OpenRetinaHoeflingFormat table is already populated for the given field_key. Skipping analysis.")

            


def get_responses_and_add_to_container(mei_data_container: pd.DataFrame,
                                        model:BaseCoreReadout |EnsembleModel,
                                        new_session_id: str,
                                        neuron_data_dict: Dict[str,ResponsesTrainTestSplit],
                                        mei_generation_params: Dict[str, Any]) -> None:
    """
    Adds two columns to the mei data container: 
    - one with the resonses of each neuron EACH NEURON IN READOUT to all meis THIS IS NOT FILTERED FOR QUALITY. 
    This so that later the index of the column corresponds to the index of the neuorns in the readouts. 
    - one with the mean of that response in the optiization time window."""

    # make sure required cols are there
    required_cols = ['readout_idx', 'roi_id', 'mei']
    if not all(col in mei_data_container.columns for col in required_cols):
        raise ValueError(f"mei_data_container must contain the following columns: {required_cols}")


    ## responses: use batch of meis
    mei_batch = mei_data_container['mei'].to_list()
    device = model.device if isinstance(model,BaseCoreReadout) else model.members[0].device
    mei_batch = torch.stack(mei_batch, dim=0).to(device)
    all_neuron_ids_in_readout = list(range(neuron_data_dict[new_session_id].session_kwargs["roi_ids"].shape[0]))
    all_responses = get_model_mei_response(model = model,
                                            mei=mei_batch,
                                            session_id = new_session_id,
                                            neuron_id = all_neuron_ids_in_readout,)
    
    # resonses are shape nr meis, nr time points response, nr neurons in readout 
    assert all_responses.shape[0] == mei_batch.shape[0], "Number of responses does not match number of MEIs."
    assert all_responses.shape[2] == len(all_neuron_ids_in_readout)
    
    # store the responses in the data container
    mei_data_container['responses_all_readout_idx'] = list(all_responses)

    # reduce responses to the mean in the response window/ optimization time window.
    t0 = mei_generation_params["reducer_start"]
    t1 = t0 + mei_generation_params["reducer_length"]
    all_mean_responses = np.mean(all_responses[:,t0:t1,:], axis=1) # shape (nr_meis, nr_neurons in readout )
    assert all_mean_responses.shape[0] == mei_batch.shape[0], "Number of mean responses does not match number of MEIs."
    assert all_mean_responses.shape[1] == len(all_neuron_ids_in_readout), "Number of mean responses does not match number of neurons."
    mei_data_container['mean_responses_all_readout_idx'] = list(all_mean_responses)
