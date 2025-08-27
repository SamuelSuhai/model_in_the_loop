import os
import datajoint as dj
import subprocess
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any, Tuple, Callable,Optional,Union
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from djimaging.utils import plot_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.mask_utils import to_roi_mask_file

from .utils import time_it
from .model_to_stimulus import (load_stimuli, preprocess_for_openretina,
                                train_model_online,reconstruct_mei_from_decomposed,generate_meis_with_n_random_seeds,
                                decompose_mei, get_model_mei_response,Center,get_model_gaussian_scaled_means
            )

from omegaconf import DictConfig, ListConfig





class DJTableHolder:
    """
    A class that 
    1) Stores the tables so other compute classes can call operations on them 
    2) activates the schema and established a connection to the database.
    3) Calls a setup functio that populates the parameters tabeles
    """

    def __init__(self,
                 username: str,
                 home_directory: str,
                 repo_directory: str,
                 dj_config_directory: str,
                 rgc_output_directory: str,
                 data_subfolders: Dict[str, str],
                 
                 userinfo: dict,

                 table_parameters: dict,

                 sleep_time_between_table_ops: int  = 1,
                 debug: bool = False,
                 plot_results: bool = False,

                 ):
        """
        Store information needed to run table ops
        """
        self.iteration: int = 0
        self.debug: bool = debug
        self.multiprocessing_threads: int = 20 if not debug else 1 # this is so I can go in with debugger wo problems othrewise get problems
        self.plot_results: bool  = plot_results
        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops
        self.tables = {}

        # check who is running command
        self.username: str = username

        self.table_parameters: dict = table_parameters

        # get home directory
        self.home_directory: str = home_directory
        self.config_file: str = os.path.join(self.home_directory,dj_config_directory, f'dj_{self.username}_conf.json')
        self.rgc_output_folder = os.path.join(home_directory, rgc_output_directory)
        self.data_subfolders = data_subfolders

        # get repo directory
        self.repo_directory: str = repo_directory

        # information for UserInfo table
        self.userinfo: dict = userinfo
        self.schema_name = f"ageuler_{self.username}_closed_loop"

        # some tests idk if necessary
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()
        assert self.home_directory == os.path.expanduser("~")
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'
        assert os.path.exists(userinfo["data_dir"]), f"Path to data dir does not exist: {username["data_dir"]}"
        assert os.path.isdir(self.rgc_output_folder), f'Set path to output directory: {self.rgc_output_folder}'
    
    
    def __call__(self, table_name):
        
        """
        Access tables directly using the instance as a callable
        """
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} not found in loaded tables")
        return self.tables[table_name]

    def load_tables(self):
        """
        Import and load all database tables
        """
        

        from djimaging.user.ssuhai.schemas.ssuhai_schema_closed_loop import (
                UserInfo, Experiment, OpticDisk,RelativeFieldLocation, RelativeRoiLocation,
                Field, Stimulus, RoiMask, Roi, Traces,
                Presentation, RawDataParams, PreprocessParams, PreprocessTraces,
                Snippets, Averages, ChirpQI, OsDsIndexes,
                ClassifierMethod, ClassifierTrainingData, Classifier,
                Baden16Traces, CelltypeAssignment,
                CascadeTraceParams, CascadeParams, CascadeTraces, CascadeSpikes,
                # RF
                DNoiseTraceParams,DNoiseTrace,STAParams,STA, SplitRFParams,SplitRF,FitGauss2DRF,
                
                # OpenRetina
                OpenRetinaHoeflingFormat,OnlineMEIs,OnlineTrainedModel,
                
                schema,
            )
        
        self.schema = schema

        # Store tables in dictionary
        self.tables = {
            'UserInfo': UserInfo,
            'Experiment': Experiment,
            'Field': Field,
            'OpticDisk': OpticDisk,
            'RelativeFieldLocation': RelativeFieldLocation,
            'RelativeRoiLocation': RelativeRoiLocation,
            'Stimulus': Stimulus,
            'RoiMask': RoiMask,
            'Roi': Roi,
            'Traces': Traces,
            'Presentation': Presentation,
            'RawDataParams': RawDataParams,
            'PreprocessParams': PreprocessParams,
            'PreprocessTraces': PreprocessTraces,
            'Snippets': Snippets,
            'Averages': Averages,
            'ChirpQI': ChirpQI,
            'OsDsIndexes': OsDsIndexes,
            'ClassifierMethod': ClassifierMethod,
            'ClassifierTrainingData': ClassifierTrainingData,
            'Classifier': Classifier,
            'Baden16Traces': Baden16Traces,
            'CelltypeAssignment': CelltypeAssignment,
            'CascadeTraceParams': CascadeTraceParams,
            'CascadeParams': CascadeParams,
            'CascadeTraces': CascadeTraces,
            'CascadeSpikes': CascadeSpikes,
            
            # RF 
            'DNoiseTraceParams': DNoiseTraceParams,
            'DNoiseTrace': DNoiseTrace,
            'STAParams': STAParams,
            'STA': STA,
            'SplitRFParams': SplitRFParams,
            'SplitRF': SplitRF,
            'FitGauss2DRF': FitGauss2DRF,

            

            # OpenRetina
            'OpenRetinaHoeflingFormat': OpenRetinaHoeflingFormat,
            'OnlineMEIs': OnlineMEIs,
            'OnlineTrainedModel': OnlineTrainedModel,
        }

        sleep(self.sleep_time_between_table_ops)
        from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
        prepare_dj_config_rgc_classifier(self.rgc_output_folder)
        sleep(self.sleep_time_between_table_ops)

        from djimaging.utils.dj_utils import activate_schema
        activate_schema(schema=self.schema, create_schema=True, create_tables=True)
        


            
            

    def load_config(self) -> None:
        """
        load config file 
        """
        # Load configuration for user
        dj.config.load(self.config_file)

        dj.config['schema_name'] = self.schema_name
        dj.config['enable_python_native_blobs'] = True
        dj.config["display.limit"] = 20
        

        print("schema_name:", dj.config['schema_name'])
        dj.conn()
        
        
        

    def set_params_and_userinfo(self) -> None:
        
        # make sure tables are empty 
        if len(self('UserInfo')()) == 0:
            self('UserInfo')().upload_user(self.userinfo)
            

        self('RawDataParams')().add_default()
        

        # TODO extract hard coded values to config
        self('RawDataParams')().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        

        preprocess_params =self.table_parameters.get("PreprocessParams", {})
        if isinstance(preprocess_params, ListConfig):
            for params in preprocess_params:
                self('PreprocessParams')().add_default(**params)
        elif isinstance(preprocess_params, DictConfig):
            self('PreprocessParams')().add_default(**preprocess_params)
        else:
            raise ValueError(f"Expected preprocess_params to be DictConfig or ListConfig, got {type(preprocess_params)}")
        print("preprocessing params:\n",preprocess_params)

        

        # Celltype assignment
        self('ClassifierTrainingData')().add_default(skip_duplicates=True)
        self('ClassifierMethod')().add_default(skip_duplicates=True)
        self('Classifier')().populate()

        # rf estimation 
        dense_noise_parmas = self.table_parameters.get("DNoiseTraceParams", {})
        self('DNoiseTraceParams')().add_default(**dense_noise_parmas)


        self('STAParams')().add_default()
        self('SplitRFParams')().add_default()



        # spike estimation
        self('CascadeTraceParams')().add_default(stim_names=['mouse_cam'])
        self('CascadeParams')().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel') # for spike estimation itself

    def add_all_stimuli(self) -> None:


        
        import h5py

        with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)
        noise_stimulus = noise_stimulus*2-1
        
        
        self('Stimulus')().add_nostim(skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)

        
        self('Stimulus')().add_noise(**self.table_parameters.Stimulus.noise, stim_trace=noise_stimulus,)
            
        self('Stimulus')().add_movingbar(skip_duplicates=True)
        
        self('Stimulus')().add_stimulus(
            stim_name="mouse_cam", 
            alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
            stim_family="natural", 
            framerate=30.0,  
            stim_path="",
            ntrigger_rep=123,
            unique_alias=True,
            skip_duplicates=True,
        )

        

    def setup(self) -> None:
        """
        Wrapper for the things we can do before the first iteration. 
        These are: 
        1) Connecting to the database
        2) Loading the tables
        3) Activating the schema
        4) setting the information of the stimuli used
        5) setting geenral pipeline parameters and user infor
        """
        
        self.load_config()        
        self.load_tables()

        if len(self("UserInfo")()) > 0:
            warnings.warn("\nSome DJ tables (like UserInfo) are not empty, skipping adding new entries from config.\nMake sure this is wanted. Call clear_tables(`all`) if you want different data in there")
            print("Done reconnecting. Skipping adding new entries from config.")
            return
        
        self.add_all_stimuli()
        self.set_params_and_userinfo()
        

        print("Done setting up!")



    def clear_tables(self,target: str = "all", 
                     field_key : Dict[str,str] = {},
                     safemode=True) -> None:
        """
        Clear tables.
        This is useful to start a new iteration with a clean slate.
        """
        if target == "all":
            # delete entries of userinfo will result in all being deleted
            self('UserInfo')().delete(safemode=safemode)

            # delete Stimulus, all params tables and  classifier tables as these do not depend on userinfo
            self('Stimulus')().delete(safemode=safemode)
            self('RawDataParams')().delete(safemode=safemode)
            self('PreprocessParams')().delete(safemode=safemode)
            self('ClassifierMethod')().delete(safemode=safemode)
            self('Classifier')().delete(safemode=safemode)
            self('ClassifierTrainingData')().delete(safemode=safemode)
            self('CascadeTraceParams')().delete(safemode=safemode)
            self('CascadeParams')().delete(safemode=safemode)
            self('DNoiseTraceParams')().delete(safemode=safemode)

            self('STAParams')().delete(safemode=safemode)
            self('SplitRFParams')().delete(safemode=safemode)

        
        elif target == "experiment":
            # delete all entries of the current experiment
            self('Experiment')().delete(safemode=safemode)

        elif target == "field":
            # delete all entries of the current field
            if field_key == {}:
                raise ValueError("field_key must be provided when target is 'field'")
            (self('Field')() & field_key).delete(safemode=safemode)
        
        elif target == "rois":
            (self('RoiMask')() & field_key).delete(safemode=safemode)

        elif target == "model":
            (self("CascadeTraces")() & field_key).delete(safemode=safemode)
            (self("OpenRetinaHoeflingFormat")() & field_key).delete(safemode=safemode)
            return # no roi masks to delete here
         
        # remove any roi masks saved in the directory
        all_field_presentation_files = (self("Presentation")() & field_key).fetch("pres_data_file")
        all_roi_mask_files = [to_roi_mask_file(file, roi_mask_dir="ROIs") for file in all_field_presentation_files]
        if len(all_roi_mask_files) == 0:
            print("No ROI mask files found to delete.")
            return
        # prompt user to confirm deletion
        if safemode:
            confirm = input(f"Are you sure you want to remove ROI maskfiles \n{"\n".join(all_roi_mask_files)} ROI mask files? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Deletion cancelled.")
                return

        # remove all roi mask files
        for file in all_roi_mask_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed file: {file}")
        

        


######################################################## For Table operations up to Traces ##########################################################
class Preprocessor:
    """ Class that hold table operations for 
    1) loading data into database
    2) addin ROIs
    3) adding traces"""

    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the Preprocessor with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder


    def upload_iteration_metadata(self) -> None:
        """
        Upload metadata for the current iteration, including experiments, fields, and presentations.
        """
        # TODO: rescan vermeiden, scant alle 
        self.dj_table_holder('Experiment')().rescan_filesystem(verboselvl=3)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)
        
        self.dj_table_holder('OpticDisk')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)



        self.dj_table_holder('Field')().rescan_filesystem(verboselvl=3)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('RelativeFieldLocation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('Presentation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)
        
    def add_iteration_roi_mask(self, field_key = {},save_to_file = False) -> None:
        """
        Add ROIs for the current iteration by drawing and inserting them into the database.
        NOTE that this will not be called from the GUI, but its for when we use this from the command line
        """
        self.dj_table_holder('RoiMask')().rescan_filesystem(verboselvl=3)
        
        if field_key == {}:
            missing_fields = self.dj_table_holder('RoiMask')().list_missing_field()
            assert len(missing_fields) == 1 , f"Expecting exatly no missing fields but found {len(missing_fields)}"
            field_key = missing_fields[0]
            
        assert len(self.dj_table_holder('RoiMask')() & field_key) == 0, f"RoiMask table already has entries for field_key {field_key}. Use clear_tables('rois') to remove them first."

        roi_canvas = self.dj_table_holder('RoiMask')().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        # execute autorois for the main stack
        roi_canvas.exec_autorois_all()
        
        # apply autoshoft to all stimuli except the main one
        for i,pres_key in enumerate(roi_canvas.pres_names):
            if i == roi_canvas.main_stim_idx:
                continue
            roi_canvas.set_selected_stim(pres_key)
            roi_canvas.exec_auto_shift()


        # add to database
        roi_canvas.insert_database(roi_mask_tab=self.dj_table_holder('RoiMask'), field_key=field_key)

        # save masks
        if save_to_file:
            roi_canvas.exec_save_all_to_file()

    
    def add_iteration_rois(self) -> None:
        self.dj_table_holder('Roi')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

    def add_iteration_traces(self) -> None:
        """
        Populate the Traces and PreprocessTraces tables for the current iteration.
        """
        self.dj_table_holder('Traces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('PreprocessTraces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)



#################################################### DJ Compute Wrappers ###############################################


class DJComputeWrapper(ABC):
    """
    ABC for DJ wrappers: They group table operations together and have 
    functions that allow them to be inetrated into the GUI.
    The idea is that this makes the whole pipeline more modualr and expandable: users can create their own wrappers that can be used in the GUI."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the wrapper, used to identify it in the GUI.
        """
        pass


    @abstractmethod
    def plot1(self, roi_key: Dict[str, Any]) -> None:
        """
        Plot the first roi of the wrapper. This is called in the gui.
        """
        pass

    @abstractmethod
    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        """
        Plot an overview of the rois of the wrapper. This is called in the gui.
        """
        pass

    @abstractmethod
    def check_requirements(self) -> bool:
        """
        Check if the required tables are populated in the database.
        """
        pass

    @abstractmethod
    def compute_analysis(self) -> None:
        """
        Compute the main analysis for the wrapper. This is called in the gui. """
        pass

class QualityAndTypeWrapper(DJComputeWrapper):
    """Groups table operations for quality metrics and cell type assignment."""
    
    # a color map used to get an roi 2 color mapping based on the g name
    g_name_to_rgb255 = {
        "OFF": np.array([255, 0, 0]),          # Red
        "ON-OFF": np.array([0, 255, 0]),      # Green
        "Fast ON": np.array([64, 224, 208]),  # Turquoise
        "Slow ON": np.array([0, 0, 255]),     # Blue
        "Uncertain RGCs": np.array([128, 0, 128]),  # Purple
        "ACs": np.array([255, 255, 255])        # white
    }



    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the QualityAndType wrapper with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder
        self.requires_tables = [
            'Snippets',
            'Averages',
            ]

    @property
    def name(self) -> str:
        return "Quality and Type"
    
    def add_quality_metrics(self,field_key) -> None:
        """
        Populate the ChirpQI and OsDsIndexes tables for quality metrics.
        """
        if len(self.dj_table_holder('ChirpQI')() & field_key) == 0:
            
            self.dj_table_holder('ChirpQI')().populate(display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)
            
        if len(self.dj_table_holder('OsDsIndexes')() & field_key) == 0:
            self.dj_table_holder('OsDsIndexes')().populate(display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)

    def add_celltype_assignments(self, field_key) -> None:
        """
        Populate the Baden16Traces and CelltypeAssignment tables for cell type assignments.
        """

        if len(self.dj_table_holder('Baden16Traces')() & field_key) == 0:
            self.dj_table_holder('Baden16Traces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        
        if len(self.dj_table_holder('CelltypeAssignment')() & field_key) == 0:
            self.dj_table_holder('CelltypeAssignment')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

    def compute_analysis(self, field_key = {},progress_callback: Optional[Callable]  = None) -> None:
        """
        Compute quality metrics and cell type assignment.
        """
        if progress_callback is not None:
            progress_callback(0)

        self.check_requirements(field_key)
        if progress_callback is not None:
            progress_callback(30)

        self.add_quality_metrics(field_key)
        if progress_callback is not None:
            progress_callback(60)

        self.add_celltype_assignments(field_key)
        if progress_callback is not None:
            progress_callback(100)



    
    @staticmethod
    def g_to_type_name(g: int) -> str:

        if g in range(1,10):
            return "OFF"
        elif g in range(10,15):
            return "ON-OFF"
        elif g in range(15,21):
            return "Fast ON"
        elif g in range(21,29):
            return "Slow ON"
        elif g in range(29,33):
            return "Uncertain RGCs"
        elif g >= 33:
            return "ACs"
        else:
            raise ValueError(f"Unknown group {g} for g_to_type_name")

    
    def g_to_rgb255(self,g: int) -> np.ndarray:
        """Converts the G (celltype) to a superclass type name and from there the designated RGB255 color."""

        g_name = self.g_to_type_name(g)
        return self.g_name_to_rgb255[g_name]


    # range mapping from quality index to alpha value 
    
    def qi2alpha255(self,qi: float,alpha_min = 0.1,alpha_max = 0.6) -> float:
        assert 0 <= qi <= 1, "Quality index must be between 0 and 1"
        return (alpha_min + (qi * (alpha_max - alpha_min))) * 255

    def get_roi_ids_passing_criterion(self, 
                                      field_key: Dict[str, Any],
                                      d_qi_min: float, 
                                      qidx_min: float,
                                      celltypes: List[int] = list(range(99)),
                                      classifier_confidence: float = 0.0) -> List[int]:
        """
        For a field key it looks in the tables ChirpQI and OsDsIndexes for the d_qi and qidx values. Takes rois that pass either chrip or ori_dir quality index as passing.
        """

        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        ori_dir_qi_table = self.dj_table_holder('OsDsIndexes')() & field_key

        if len(chirp_qi_table) == 0 or len(ori_dir_qi_table) == 0:
            raise ValueError("ChirpQI or OsDsIndexes table is empty for the given field_key")
        
        # Fetch all roi_ids and their corresponding d_qi and qidx values
        roi_ids_chirp, qidx_values = chirp_qi_table.fetch('roi_id', 'qidx')
        chirp_data_dict = {roi_id: qidx for roi_id, qidx in zip(roi_ids_chirp, qidx_values)}
        
        roi_ids_mb,d_qi_values = ori_dir_qi_table.fetch('roi_id', 'd_qi')
        ori_dir_data_dict = {roi_id: d_qi for roi_id, d_qi in zip(roi_ids_mb, d_qi_values)}

        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key
        if len(celltype_table) == 0:
            raise ValueError("CelltypeAssignment table is empty for the given field_key")
        
        # create a criterion
        roi_ids_celltype,celltype_data,confidence_data = celltype_table.fetch('roi_id','celltype','confidence')
        celltype_data_dict = {roi_id: celltype for roi_id, celltype in zip(roi_ids_celltype, celltype_data)}
        
        #  get the confidence of assigned group which is the max 
        confidence_scors = [all_confidences.max() for all_confidences in confidence_data]
        confidence_data_dict = {roi_id: confidence for roi_id, confidence in zip(roi_ids_celltype, confidence_scors)}

        # Filter roi_ids based on the criteria
        passing_roi_ids = []
        for roi_id in set(roi_ids_mb + roi_ids_chirp + roi_ids_celltype):
            d_qi = ori_dir_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            qidx = chirp_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            confidence = confidence_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            celltype = celltype_data_dict.get(roi_id, -1) 
            
            if (d_qi >= d_qi_min or qidx >= qidx_min) and (confidence >= classifier_confidence) and celltype in celltypes:
                passing_roi_ids.append(roi_id)


        return passing_roi_ids
    
    def text1(self,roi_id: int, field_key = {}, ) -> str:
        """
        Get the text for the requested roi.
        """
        roi_restriction = {'roi_id': roi_id}
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        ori_dir_qi_table = self.dj_table_holder('OsDsIndexes')() & field_key
        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key

        d_qi = (ori_dir_qi_table & roi_restriction).fetch("d_qi").item()
        qidx_chirp = (chirp_qi_table & roi_restriction).fetch("qidx").item()

        celltype_data = (celltype_table & roi_restriction).fetch1()
        confidence_scores = celltype_data['confidence']

        # Get top 3 with scores
        top_3_indices = confidence_scores.argsort()[-3:][::-1]
        top_3_groups = top_3_indices + 1 # assume index based group assignment
        top_3_scores = confidence_scores[top_3_indices]
        top_3_group_names = [self.g_to_type_name(g) for g in top_3_groups]

        assert celltype_data["celltype"] == top_3_groups[0], "Top group does not match celltype assignment"

        text = "Quality Metrics:\n"
        text += f"Chirp QI: {qidx_chirp:.2f}\n"
        text += f"d_qi: {d_qi:.2f}\n\n"

        text += "Top 3 types:\n"
        for g,name,score in zip(top_3_groups, top_3_group_names,top_3_scores):
            text += f"{g} ({name}): {score:.2f}\n"
        return text


    def plot_g_name_legend(self, ax, y_start=0.05, fontsize=10):
        """
        Plots a legend mapping g_name to RGB color below the text area,
        with rectangles drawn just before the labels.
        """
        y_step = 0.07
        rect_x = 0.05  # x-position for rectangles
        text_x = rect_x + 0.07  # x-position for text, just after rectangle

        for i, (g_name, rgb) in enumerate(self.g_name_to_rgb255.items()):
            color = tuple(rgb / 255)
            y = y_start + i * y_step
            # Draw colored rectangle
            ax.add_patch(plt.Rectangle((rect_x, y), 0.06, 0.05, color=color, transform=ax.transAxes, clip_on=False))
            # Draw label just after rectangle
            ax.text(text_x, y + 0.025, f"{g_name}", fontsize=fontsize, color='black',
                    verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)
        
    def plot1(self, roi_id, stim_name= "gChirp", field_key = {}, xlim=None,show_fig = True) -> None:
        """
        Plot the Averages of the requested roi.
        """
        import matplotlib.gridspec as gridspec


        single_averages_table = (self.dj_table_holder('Averages')() & field_key & {'roi_id': roi_id, 'stim_name': stim_name})
        snippets_table = self.dj_table_holder('Snippets')() & field_key


        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (snippets_table & single_averages_table).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets', 'triggertimes_snippets')

        average, average_norm, average_t0, average_dt, triggertimes_rel = \
            single_averages_table.fetch1('average', 'average_norm', 'average_t0', 'average_dt', 'triggertimes_rel')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)
        average_times = np.arange(len(average)) * average_dt + average_t0

        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 4))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])
        
        # Text area (spans both rows, left column)
        text_ax = fig.add_subplot(gs[:, 0])
        text_ax.axis('off')
        
        # Plot areas (right column)
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        
        # plotting
        ax1.plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        ax1.set(ylabel='trace', xlim=xlim)
        
        plot_utils.plot_trace_and_trigger(
            ax=ax2, time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)
        ax2.set(xlabel='rel. to trigger')
        
        # Add text to dedicated text area
        text = self.text1(roi_id,field_key=field_key)
        text_ax.text(0.05, 0.95, text, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    transform=text_ax.transAxes)
        
        # add colors for g_name
        self.plot_g_name_legend(text_ax, y_start=0.05, fontsize=10)

        # set tite
        fig.suptitle(f"ROI {roi_id} - {stim_name}", fontsize=16)
        
        if show_fig:
            plt.show()

    
    
    def check_requirements(self, field_key) -> None:
        """
        Check if the required tables are populated in the database.
        """
        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & field_key) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
                sleep(self.dj_table_holder.sleep_time_between_table_ops)

    def get_roi2rgb_and_alpha_255_map(self,
                                      field_key: Dict[str, Any],
                                      all_roi_ids: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Get two mappings: one for roi to rgb based on celltype and one roi to alpha based on chirpQI.
        all_roi_ids is a list of roi_ids that are in the canvas so we cover all rois in the cavas with the color code.

        """
        
        # get celltype for each roi 
        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key
        celltype_data = celltype_table.fetch('roi_id', 'celltype',as_dict=True)
        roi2rgb255 = {data['roi_id']: self.g_to_rgb255(data['celltype']) for data in celltype_data}

        # get chirpQI for each roi
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        chirp_qi_data = chirp_qi_table.fetch('roi_id', 'qidx', as_dict=True)
        roi2alpha = {data['roi_id']: self.qi2alpha255(data['qidx']) for data in chirp_qi_data}

        # ensure all roi_ids are covered
        for roi_id in all_roi_ids:
            if roi_id not in roi2rgb255:
                # default color for missing roi_ids
                roi2rgb255[roi_id] = np.array([128, 128, 128])
            if roi_id not in roi2alpha:
                # default alpha for missing roi_ids
                roi2alpha[roi_id] = self.qi2alpha255(0.0)
        

        return roi2rgb255, roi2alpha
         
    
    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        """
        Plot an overview of the rois of the wrapper.
        """
        pass


class STAWrapper(DJComputeWrapper):
    """Groups table operations for STA estimation."""

    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the STA wrapper with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder
        self.requires_tables = [
            
        ]

    @property
    def name(self) -> str:
        return "STA"
    
    def compute_analysis(self, 
                         field_key = {},
                         roi_id_subset: Optional[List[int]] = None ,
                         progress_callback: Optional[Callable] = None) -> None:
        """
        Compute the STA analysis.
        """
        complete_restriction = get_rois_in_field_restriction_str(field_key, roi_id_subset)
        
        if progress_callback is not None:
            progress_callback(0)
        
        self.dj_table_holder('DNoiseTrace')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        

        if progress_callback is not None:
            progress_callback(30)
        self.dj_table_holder('STA')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

        if progress_callback is not None:
            progress_callback(80)
        self.dj_table_holder('SplitRF')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

        self.dj_table_holder("FitGauss2DRF")().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        

        if progress_callback is not None:
            progress_callback(100)

    def check_requirements(self, 
                           field_key,
                           roi_id_subset: Optional[List[int]] = None ,) -> None:
        """
        Check if the required tables are populated in the database.
        """
        complete_restriction = get_rois_in_field_restriction_str(field_key, roi_id_subset)

        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & complete_restriction) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
    
    def get_roi_ids_passing_criterion(self, 
                                      field_key: Dict[str, Any], 
                                      rf_qidx_min: float = 0.5,) -> List[int]:
        """
        Looks at the FitGauss2DRF table and returns the roi_ids that have a qidx value above the given threshold.
        """

        fit_gauss_table = self.dj_table_holder('FitGauss2DRF')() & field_key

        if len(fit_gauss_table) == 0:
            raise ValueError("FitGauss2DRF table is empty for the given field_key")
        
        # Fetch all roi_ids and their corresponding qidx values
        roi_ids, qidx_values = fit_gauss_table.fetch('roi_id', 'rf_qidx')

        # Filter roi_ids based on the criteria
        passing_roi_ids = [
            roi_id for roi_id, qidx in zip(roi_ids, qidx_values, strict=True) if qidx >= rf_qidx_min
        ]

        return passing_roi_ids


    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        pass
        
    def plot1(self,roi_id: int,field_key={},show = True) -> None:

        restricted_split_rf = (self.dj_table_holder('SplitRF')() & field_key & {'roi_id': roi_id})
        if len(restricted_split_rf) == 0:
            print(f"No RF computed for roi_id {roi_id}.")
            return
        elif len(restricted_split_rf) > 1:
            raise ValueError(f"Expected exactly one SplitRF for roi_id {roi_id}, found {len(restricted_split_rf)}")

        # plot it and 
        key = get_primary_key(table=restricted_split_rf, key=None)

        rf_time = restricted_split_rf.fetch1_rf_time(key=key)
        srf, trf, peak_idxs = (restricted_split_rf & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex='col')

        ax = axs[0]
        plot_utils.plot_srf(srf, ax=ax)
        ax.set_title(f'sRF for ROI {roi_id}')

        # add x mark at the peak position in index coordinates

        fit_restricted = (self.dj_table_holder("FitGauss2DRF")() & field_key & {'roi_id': roi_id})
        assert len(fit_restricted) == 1, f"Expected exactly one FitGauss2DRF for roi_id {roi_id}, found {len(fit_restricted)}"
        gauss_fit = fit_restricted.fetch1("srf_params")
        
        # the flip is because the axis 0 is in QDSpy dense noise x but in the fit tables its y
        x = gauss_fit['y_mean']
        y = gauss_fit['x_mean']
        
        # flip x and y becaus in QDSpy dense noise the axis 0 is x so I called it x but its axis 0. 
        ax.plot(y,x, 'x', color='black', markersize=10, label='Mean Position')
        ax.legend()

        ax = axs[1]
        plot_utils.plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)

        if show:
            plt.show()


 



class RandomSeedMEIWrapper(DJComputeWrapper):

    def __init__(self,dj_table_holder,
                 model_configs,
                 mei_optimization_params,
                 seeds: List[int],
                 reconstruct_mei: bool = True,
                ) -> None:
        
        self.dj_table_holder = dj_table_holder

      
        self.model_configs = model_configs
        self.mei_optimization_params = mei_optimization_params

        self.reconstruct_mei = reconstruct_mei

        # to store the data: the key is the index in the readout and not the roi_id
        self.neuron_seed_mei_dict = {}
        self.neuron_seed_mei_responses = {}
        self.neuron_seed_decomposed_meis = {}
        self.model = None

        self.display_channel = 1 # UV channel 

        self.seeds = seeds
        self.colors = plt.cm.nipy_spectral(np.linspace(0, 1,len(self.seeds)))

        self.testset_correl_min = 0.4


    def plot_seed_respones(self,neuron_id: int,ax: plt.Axes, optimization_window= (10,20),response_window = (21,50)):
        """
        Plots the responses of all seeds for a single neuron.
        all_meis_responses: has the structure {seed: response}
        optimization_window: what part of the  repsonse of the neuron was optimized. Then the frame of the start of the repsonse is added during plotting
        """
        response_start, respones_end = response_window
    
        all_meis_responses: Dict[int, np.ndarray] = self.neuron_seed_mei_responses[neuron_id]

        x = np.arange(response_start, respones_end + 1)
        for i,(seed, response) in enumerate(all_meis_responses.items()):
            ax.plot(x,response, label=f"Seed {seed}", color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')

        ax.set_xlabel("Time (frames)")
        ax.set_xlim(0, respones_end)
        ax.set_ylabel("Response")
        
        # Highlight the optimization window
        if optimization_window is not None:
            start , end  = optimization_window
            start += response_start
            end += response_start
            ax.axvspan(start , end, color='yellow', alpha=0.3, label='Optimization Window')
        ax.legend(ncol=2, fontsize=6)


    def plot_temporal_kernels(self,neuron_id: int, ax: plt.Axes) -> None:
        """
        Plots the temporal kernels of all seeds for a single neuron.
        """
        seed_neuron_decomposed_meis = self.neuron_seed_decomposed_meis[neuron_id]
        for i,(seed, decomposition) in enumerate(seed_neuron_decomposed_meis.items()):
            temporal_kernel = decomposition['temporal_kernels'][self.display_channel]
            ax.plot(temporal_kernel, label=f'Seed {seed}', color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')
        
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Temporal Kernel')
        ax.legend()

    def plot1(self,roi_id: int, field_key: Dict[str,Any] = {}) -> None:

        if roi_id not in self.roi2readout_idx_wmeis.keys():
            print(f"ROI {roi_id} does not have an MEI. Select among the following: \n{list(self.roi2readout_idx_wmeis.keys())}")
            return

        
        # find neuron_id for roi_id
        neuron_idx = self.roi2readout_idx_wmeis[roi_id]

        # plot temporal kernels in a line plot
        fig,axs = plt.subplots(1,2,figsize=(10, 5))
        self.plot_temporal_kernels(neuron_idx, ax=axs[0])
        axs[0].set_title(f"Temporal Kernels for ROI {roi_id} (neuron idx {neuron_idx})")

    

        # plot responses 
        self.plot_seed_respones(neuron_idx, ax=axs[1], optimization_window=(10,20), response_window=(21,50))
        axs[1].set_title(f"Responses to MEIs")

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

        # Traces
        restricted_traces = self.dj_table_holder("CascadeTraces")() & complete_restriction
        if len(restricted_traces) == 0:
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
    


    def get_neuron_idxs_passing_criterion(self) -> List[int]:

        passing_neuron_idxs = []
        for neuron_idx, corr in self.neuron_testset_correls.items():
            if corr >= self.testset_correl_min:
                passing_neuron_idxs.append(neuron_idx)
        
        return passing_neuron_idxs
    

    def get_roi2rgb_and_alpha_255_map(self,
                                      field_key: Dict[str, Any],
                                      all_roi_ids:List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Get two mappings: one for roi to rgb based on whether there is an mei.
        all_roi_ids: a list of all roi ids that should be included in the mapping.
        
        """
        rgb_of_included = np.array([255,0,0]) # red for included rois
        rgb_nonincluded = np.array([122,122,122]) # gray for non-included rois
        alpha_of_included = 122.0 # full alpha for included rois
        alpha_nonincluded = 20.0
        

        roi2rgb255 = {roi: rgb_of_included if roi in self.roi2readout_idx_wmeis.keys() else rgb_nonincluded
                      for roi in all_roi_ids}
        roi2alpha = {roi: alpha_of_included if roi in self.roi2readout_idx_wmeis.keys() else alpha_nonincluded
                     for roi in all_roi_ids}

        return roi2rgb255, roi2alpha

    def upload_to_db(self,field_key = {}) -> None:
        """
        Uploads the generated MEIs and their responses to the database.
        """
        if len(self.neuron_seed_mei_dict) == 0:
            raise ValueError("No MEIs generated. Call mei_subanalysis first.")
        
        if field_key == {}:
            # fetch from db 
            field_table = self.dj_table_holder('Field')()
            if len(field_table) != 1:
                raise ValueError("Expecte dexactly one field key.")
            field_key = self.dj_table_holder('Field')().proj().fetch(as_dict=True)[0]
        
        # get openretina hoefling format session_name and data
        session_name = (self.dj_table_holder("OpenRetinaHoeflingFormat")() & field_key).fetch1("session_name")
        session_data_dict = (self.dj_table_holder("OpenRetinaHoeflingFormat")() & field_key).fetch1("session_data_dict")

        ## the meis 
        # mapping readout idx with mei to rois
        readout_idx_wmei2rois = {readout_idx:roi_id for roi_id,readout_idx in self.roi2readout_idx_wmeis.items()}
        
        for readout_idx, seed_mei_dict in self.neuron_seed_mei_dict.items():
            for seed, mei in seed_mei_dict.items():
                
                # get the corresponding roi_id
                roi_id = readout_idx_wmei2rois[readout_idx]

                # get the response of the model 
                response = self.neuron_seed_mei_responses[readout_idx][seed]
            
                key = {**field_key,
                       "seed": seed, 
                       "readout_idx": readout_idx, 
                       "roi_id": roi_id,
                       "session_name": session_name,
                       }
                
                #insert to table 
                self.dj_table_holder("OnlineMEIs")().insert1(
                    {
                        **key,
                        "mei": mei.detach().cpu().numpy(), # store the array
                        "model_response": response.detach().cpu().numpy(),
                    },
                )

                ## the model checkpoint
                self.dj_table_holder("OnlineTrainedModel")().insert1(
                    {
                        **field_key,
                        "session_name": session_name,
                        "model_chkpt_path": self.best_model_ckp

                    }
                )        

    def fetch_from_db(self, field_key: Dict[str, Any] = {}) -> None:
        """
        Fetches the MEIs from the database and stores them in self.neuron_seed_mei_dict. 
        Assumes that this dict is empty before or not set. also stres the roi readout idx mapping in self.roi_ids2readout_idx_wmei.
        """

        assert len(self.neuron_seed_mei_dict) == 0, "The neuron_seed_mei_dict should be empty before fetching from the database."

        if not hasattr(self, 'roi2readout_idx_wmeis'):
            self.roi2readout_idx_wmeis = {}

        # fetch all MEIs for the given field_key
        mei_table = self.dj_table_holder("OnlineMEIs")() & field_key
        
        if len(mei_table) == 0:
            raise ValueError("No MEIs found for the given field_key.")
        
        # iterate over the MEIs and store them in the dictionary
        for row in mei_table.fetch(as_dict=True):
            readout_idx = row['readout_idx']
            seed = row['seed']
            mei = row['mei']
            roi_id = row['roi_id']

            # add to neuron_seed_mei_dict
            self.neuron_seed_mei_dict[readout_idx][seed] = mei

            # also store the mapping from roi_id to readout_idx
            if readout_idx not in self.roi2readout_idx_wmeis:
                self.roi2readout_idx_wmeis[roi_id] = readout_idx

        ## Model TODO
        



    def mei_subanalysis(self,
                        new_session_id: str,
                        neurons_idxs_to_analyze: List[int],
                        progress_callback: Optional[Callable] = None,
                        ) -> None:
        """lil wrapper for MEI analysys"""

        if len(neurons_idxs_to_analyze) == 0:
                raise ValueError("No neurons to perform MEI analysis on.\
                                 \nSelect less strtict filtering criterium and call mei_subanalysis again.\
                                 \nTestset correlations: {}".format(self.neuron_testset_correls))
            
        # map roi_id to model neuron idx
        self.roi2readout_idx_wmeis = {roi:idx for roi,idx in self.roi_ids2readout_idx.items() if idx in neurons_idxs_to_analyze}


        if progress_callback is not None:
            progress_callback(70)
                    # center readouts in mei generation 
        
        # center the readouts
        self.scaled_means_before_centering = get_model_gaussian_scaled_means(self.model,session= new_session_id)
        center = Center(target_mean = 0.0)
        center(self.model)
        
        
        self.neuron_seed_mei_dict = generate_meis_with_n_random_seeds(
                                        model = self.model,
                                        new_session_id = new_session_id,
                                        random_seeds =self.seeds,
                                        mei_optimization_params= self.mei_optimization_params,
                                        neuron_ids_to_analyze = neurons_idxs_to_analyze, # NOTE: this will optimize each id individually 
                                        set_model_to_eval_mode = False, # model in training mode for noisy MEIs
                                    )

            

        ## generate responses for the MEIs and decompose them
        self.neuron_seed_mei_responses = {neuron_id: {} for neuron_id in self.neuron_seed_mei_dict.keys()}
        self.neuron_seed_decomposed_meis = {neuron_id: {} for neuron_id in self.neuron_seed_mei_dict.keys()}
        for neuron_id,seed_dict in self.neuron_seed_mei_dict.items():
            for seed,mei in seed_dict.items():

                # decompose the MEIs
                temporal_kernels, spatial_kernels, stimulus_time = decompose_mei(stimulus = mei.detach().cpu().numpy())
                self.neuron_seed_decomposed_meis[neuron_id][seed] = {
                    "temporal_kernels": temporal_kernels,
                    "spatial_kernels": spatial_kernels,
                    "stimulus_time": stimulus_time,
                }
                if self.reconstruct_mei:
                    reconstruction = reconstruct_mei_from_decomposed(
                                temporal_kernels=temporal_kernels,
                                spatial_kernels=spatial_kernels,)
                    reconstruction = torch.tensor(reconstruction,dtype=torch.float32).to(self.model.device)
                    assert reconstruction.shape == mei.shape, "Reconstructed MEI shape does not match original MEI shape."
                    
                    # overwrite the mei with the reconstruction
                    mei = reconstruction
                    self.neuron_seed_mei_dict[neuron_id][seed] = mei


                # responses 
                response = get_model_mei_response(model = self.model,
                                                    mei=mei,
                                                    session_id = new_session_id,
                                                    neuron_id = neuron_id,)
                self.neuron_seed_mei_responses[neuron_id][seed] = response

        if progress_callback is not None:
            progress_callback(100)


    def extract_and_train(self) -> None:
        ## model training 
        self.session_dict_raw = self.dj_table_holder('OpenRetinaHoeflingFormat')().extract_data()
        
        # preprocess and filter further 
        self.movies_dict = load_stimuli(self.model_configs)

        self.neuron_data_dict = preprocess_for_openretina(self.session_dict_raw,self.model_configs)

        # load and refine model
        self.model,self.neuron_testset_correls,self.best_model_ckp = train_model_online(self.model_configs,
                                                                    self.neuron_data_dict,
                                                                    self.movies_dict)


        # store eome data
        self.new_session_id = list(self.session_dict_raw.keys())[0]
        
        # mappings from roi_id to to model_neuron idx
        self.roi_ids2readout_idx = {roi:idx for idx,roi in enumerate(self.neuron_data_dict[self.new_session_id].session_kwargs["roi_ids"].tolist())}



    def compute_analysis(self, field_key = {},
                         roi_id_subset: Optional[List[int]] = None,
                         progress_callback: Optional[Callable] = None) -> None:

        # extract data in hoefling format from DB 
        if len(self.dj_table_holder('OpenRetinaHoeflingFormat')() & field_key) == 0:
            if progress_callback is not None:
                progress_callback(0)

            self.check_requirements(field_key,
                                    roi_id_subset=roi_id_subset,
                                    progress_callback= progress_callback)

            if progress_callback is not None:
                progress_callback(30)
            
            ## fetch data and train model
            self.extract_and_train()
            
            # quality filter neurons
            self.neuron_idx_passed_filtering = self.get_neuron_idxs_passing_criterion()
            
            ## MEI generation

            self.mei_subanalysis(
                        new_session_id= self.new_session_id,
                        neurons_idxs_to_analyze = self.neuron_idx_passed_filtering,
                        progress_callback =progress_callback
                        )
        else:
            print("OpenRetinaHoeflingFormat table is already populated for the given field_key. Skipping analysis.")
            


def get_rois_in_field_restriction_str(field_key: Dict[str, Any],roi_id_subset:Optional[List[int]] = None) -> Union[str, Dict]:
    """
    Constructs a restriction string for the given field_key and optional roi_id_subset.
    """
    if field_key == {}:
        return {} # no restriction

    complete_restriction = " AND ".join([f"{k}='{v}'" for k,v in field_key.items()])
    if roi_id_subset is not None:
        roi_restriction_string = f"roi_id in {str(tuple(roi_id_subset))}" if len(roi_id_subset) >= 2 else f"roi_id={str(roi_id_subset[0])}"
        complete_restriction =  complete_restriction + " AND " + roi_restriction_string

    return complete_restriction