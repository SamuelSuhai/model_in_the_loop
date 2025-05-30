import os
import datajoint as dj
import subprocess
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from djimaging.utils.dj_utils import activate_schema
from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier


from djimaging.user.ssuhai.schemas.ssuhai_schema_closed_loop import UserInfo, \
    Experiment, Field, Stimulus, RoiMask, Roi, Traces, \
    Presentation, RawDataParams, PreprocessParams, PreprocessTraces,\
    Snippets, Averages, \
    ChirpQI, OsDsIndexes, \
    ClassifierMethod, ClassifierTrainingData, Classifier, \
    Baden16Traces, CelltypeAssignment, \
    CascadeTraceParams, CascadeParams,\
    CascadeTraces, CascadeSpikes, \
    OpenRetinaHoeflingFormat,\
    schema



class OpenRetinaWrapper:
    """
    After init call setup once and the process_iteration_data in each iteration."""


    def __init__(self,
                 username: str,
                 home_directory: str,
                 repo_directory: str,
                 dj_config_directory: str,
                 rgc_output_directory: str,
                 
                 userinfo: dict,
                 
                 sleep_time_between_table_ops: int  = 1,
                 debug: bool = False,
                 plotting: dict = {},

                 ):
        """
        Store information needed to run table ops
        """


        self.iteration: int = 0
        self.debug: bool = debug
        self.plotting: dict = plotting
        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops

        # check who is running command
        self.username: str = username

        # get home directory
        self.home_directory: str = home_directory
        self.config_file: str = os.path.join(self.home_directory,dj_config_directory, f'dj_{self.username}_conf.json')
        self.rgc_output_folder = os.path.join(home_directory, rgc_output_directory)

        # information for UserInfo table
        _data_dir = os.path.join(repo_directory, 'data','recordings','updated_loop_data')
        self.userinfo: dict = {**userinfo,'data_dir': _data_dir}      
        self.schema_name = f"ageuler_{self.username}_closed_loop"


        # some tests idk if necessary
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()
        assert self.home_directory == os.path.expanduser("~")
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'
        assert os.path.exists(_data_dir), f"Path to data dir does not exist: {_data_dir}"
        assert os.path.isdir(self.rgc_output_folder), f'Set path to output directory: {self.rgc_output_folder}'

        
  


    def load_config(self) -> None:
        """
        load config file 
        """
        # Load configuration for user
        dj.config.load(self.config_file)

        dj.config['schema_name'] = self.schema_name
        dj.config['enable_python_native_blobs'] = True
        dj.config["display.limit"] = 20
        sleep(self.sleep_time_between_table_ops)

        print("schema_name:", dj.config['schema_name'])
        dj.conn()
        sleep(self.sleep_time_between_table_ops)


    def set_params_and_userinfo(self) -> None:

        # make sure tables are empty 
        if len(UserInfo()) == 0:
            UserInfo().upload_user(self.userinfo)
            sleep(self.sleep_time_between_table_ops)

        RawDataParams().add_default()
        sleep(self.sleep_time_between_table_ops)

        # TODO extract hard coded values to config
        RawDataParams().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        sleep(self.sleep_time_between_table_ops)

        PreprocessParams().add_default(
            window_length=60,  # default
            poly_order=3,  # default
            non_negative=1,  # non default
            subtract_baseline=0,  # non default
            standardize=1,  # default
        )

        # Celltype assignment
        prepare_dj_config_rgc_classifier(self.rgc_output_folder)
        ClassifierMethod().add_default(skip_duplicates=True)
        ClassifierTrainingData().add_default(skip_duplicates=True)
        Classifier().populate()


        # spike estimation
        CascadeTraceParams().add_default()
        CascadeParams().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel') # for spike estimation itself
        

    def setup(self) -> None:
        """
        Wrapper for the things we can do before the first iteration. 
        These are: 
        1) Connecting to the database
        2) Activating the schema
        3) setting the information of the stimuli used
        4) setting geenral pipeline parameters and user infor
        """
        
        self.load_config()
        sleep(self.sleep_time_between_table_ops)
        
        activate_schema(schema=schema, create_schema=True, create_tables=True)
        sleep(self.sleep_time_between_table_ops)

        self.add_all_stimuli()
        sleep(self.sleep_time_between_table_ops)
        
        self.set_params_and_userinfo()
        sleep(self.sleep_time_between_table_ops)

##################################################################### During iteration functions ##############################################################################

    def upload_iteration_data(self) -> None:
        

        # TODO: rescan vermeiden, scant alle 
        if self.iteration == 0:
            Experiment().rescan_filesystem(verboselvl=3)
            sleep(self.sleep_time_between_table_ops)


        Field().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        Presentation().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)
 
        


    def add_iteration_rois(self) -> None:
        """
  
        """
        RoiMask().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        missing_fields = RoiMask().list_missing_field()
        assert len(missing_fields) > 0 , "no missing fields found"
        field_key = missing_fields[0]
        sleep(self.sleep_time_between_table_ops)


        roi_canvas = RoiMask().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        #TODO : REALLY UNSURE IF THIS DOES WHAT I WANT 
        roi_canvas.exec_autorois_all()
        roi_canvas.exec_save_to_file()

        # add to database
        roi_canvas.insert_database(roi_mask_tab=RoiMask, field_key=field_key)
        sleep(self.sleep_time_between_table_ops)
        Roi().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)


    def add_iteration_traces(self) -> None:
        

        Traces().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        PreprocessTraces().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)



    def add_trace_reformatting(self,snippets:bool = True, averages:bool = True,) -> None:
        
        if snippets:
            Snippets().populate(processes=20, display_progress=True)
        
        if averages:
            Averages().populate(processes=20, display_progress=True)


    def add_quality_metrics(self) -> None:

        ChirpQI().populate(display_progress=True, processes=20)
        OsDsIndexes().populate(display_progress=True, processes=20)



    
    def add_spikes(self) -> None:

        CascadeTraces().populate()
        sleep(self.sleep_time_between_table_ops)
        CascadeSpikes().populate()


    def add_all_stimuli(self) -> None:
        import h5py

        with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)
        
        Stimulus().add_nostim(skip_duplicates=True)
        Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
        Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)
        Stimulus().add_noise(stim_name='noise', pix_n_x=20, pix_n_y=15, pix_scale_x_um=30, pix_scale_y_um=30, stim_trace=noise_stimulus, skip_duplicates=True)
        Stimulus().add_movingbar(skip_duplicates=True)
        
        Stimulus().add_stimulus(
            stim_name="mouse_cam", 
            alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
            stim_family="natural", 
            framerate=30.0,  
            stim_path="",
            ntrigger_rep=123,
            unique_alias=True
        )

    

    def add_celltype_assignments(self) -> None:

        Baden16Traces().populate(display_progress=True, processes=20)
        CelltypeAssignment().populate(display_progress=True)

        if self.debug:
            CelltypeAssignment().plot(threshold_confidence=0.0)
            CelltypeAssignment().plot(threshold_confidence=0.25)
            CelltypeAssignment().plot(threshold_confidence=0.5)

    

    def extract_data(self) -> Dict[str,Dict[str, Any]]:
        
        session_dict = OpenRetinaHoeflingFormat().populate()
        return session_dict



    def process_iteration_data(self,) -> Dict[str,Dict[str, Any]]:
        """
        main function =
        Call this function to run the pipeline on each iteration of the loop
        """
        

        if self.iteration == 0:
            self.add_trace_reformatting()
            self.add_quality_metrics()
            self.add_celltype_assignments()


 
        
        self.upload_iteration_data()
        self.add_iteration_rois()
        self.add_iteration_traces()
        self.add_spikes()
        iteration_data = self.extract_data()

        self.iteration += 1

        return iteration_data


        
##################################################################  Other utils  #############################################################################################
    def clean_up(self, all: bool = False) -> None:
            """
            Clear schema tables.
            
            Args:
                all: If True, delete all tables. If False, only delete tables modified in each iteration.
            """
            # Tables that are modified in each iteration
            iteration_modified_tables = [
                RoiMask(),
                Roi(), 
                Traces(),
                PreprocessTraces(),
                Presentation(),
                CascadeTraces(),
                CascadeSpikes(),
                OpenRetinaHoeflingFormat()
            ]
            
            # Tables that are static across iterations
            static_tables = [
                UserInfo(),
                Experiment(),
                Field(),
                Stimulus(), 
                RawDataParams(),
                PreprocessParams(),
                Snippets(),
                Averages(),
                ChirpQI(),
                OsDsIndexes(),
                ClassifierMethod(),
                ClassifierTrainingData(),
                Classifier(),
                CascadeTraceParams(),
                CascadeParams(),
                Baden16Traces(),
                CelltypeAssignment()
            ]
            
            # Tables to delete based on 'all' parameter
            tables_to_delete = iteration_modified_tables + (static_tables if all else [])
            
            # Delete tables with pause between each operation
            for table in tables_to_delete:
                table.delete()
                sleep(self.sleep_time_between_table_ops)

            self.iteration = 0