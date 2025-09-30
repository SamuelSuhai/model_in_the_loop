import os
import datajoint as dj
import subprocess
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any, Tuple, Callable,Optional,Union
from omegaconf import OmegaConf, DictConfig, ListConfig
from abc import ABC, abstractmethod

from model_in_the_loop.utils.file_management import clear_roi_field_field







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

        # get repo directory
        self.repo_directory: str = repo_directory
        # save as environment variable 
        os.environ["MITL_REPO_DIRECTORY"] = repo_directory

        # information for UserInfo table
        self.userinfo: dict = userinfo
        self.schema_name = f"ageuler_{self.username}_closed_loop"

        # some tests idk if necessary
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()
        assert self.home_directory == os.path.expanduser("~")
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'
        assert os.path.exists(userinfo["data_dir"]), f"Path to data dir does not exist: {userinfo["data_dir"]}"
        assert os.path.isdir(self.rgc_output_folder), f'Set path to output directory: {self.rgc_output_folder}'
    
    
    def __call__(self, table_name) -> Callable:
        
        """
        Access tables directly using the instance as a callable
        """
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} not found in loaded tables")
        return self.tables[table_name]

    def load_tables(self) -> None:
        """
        Import and load all database tables
        """
        
        
        from model_in_the_loop.core.dj_schemas.full_mitl_schema import (
                UserInfo, Experiment, OpticDisk,RelativeFieldLocation, RelativeRoiLocationWrtField,RelativeRoiLocation,
                Field, Stimulus, RoiMask, Roi, Traces,
                Presentation, RawDataParams, PreprocessParams, PreprocessTraces,
                Snippets, Averages, ChirpQI, OsDsIndexes,
                ClassifierMethod, ClassifierTrainingData, Classifier,
                Baden16Traces, CelltypeAssignment,
                CascadeTraceParams, CascadeParams, CascadeTraces, CascadeSpikes,
                # RF
                DNoiseTraceParams,DNoiseTrace,STAParams,STA, SplitRFParams,SplitRF,FitGauss2DRF,
                
                # OpenRetina
                OpenRetinaHoeflingFormat,OnlineMEIs,OnlineTrainedModel,OnlineOptimizedStimulus,StimulusDecomposition,ModelStimulusResponse
                
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
            'RelativeRoiLocationWrtField':RelativeRoiLocationWrtField,
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
            "OnlineOptimizedStimulus": OnlineOptimizedStimulus,
            "StimulusDecomposition": StimulusDecomposition,
            "ModelStimulusResponse": ModelStimulusResponse,
            
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
        
        clear_roi_field_field(Presentation = self('Presentation')(),field_key=field_key,safemode=safemode)

        

        


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
        
        self.dj_table_holder('OpticDisk')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)



        self.dj_table_holder('Field')().rescan_filesystem(verboselvl=3)
        self.dj_table_holder('RelativeFieldLocation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        self.dj_table_holder('Presentation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        
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
        self.dj_table_holder('PreprocessTraces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)


    def clear_tables(self, field_key, safemode=True) -> None:
        """
        Clear tables related to preprocessing: RoiMask, Roi, Traces, PreprocessTraces.
        """

        (self.dj_table_holder('RoiMask')() & field_key).delete(safemode=safemode)
        (self.dj_table_holder('Roi')() & field_key).delete(safemode=safemode)
        (self.dj_table_holder('Traces')() & field_key).delete(safemode=safemode)
        (self.dj_table_holder('PreprocessTraces')() & field_key).delete(safemode=safemode)


        clear_roi_field_field(Presentation = self.dj_table_holder('Presentation')(),field_key=field_key,safemode=safemode)



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
