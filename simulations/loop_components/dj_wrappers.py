import os
import datajoint as dj
import subprocess
import shutil
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .utils import time_it

class OpenRetinaWrapper:
    """
    After init call setup once and the process_iteration_data in each iteration."""

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
                DNoiseTraceParams,DNoiseTrace,STAParams,STA,
                PeakSTAPosition,
                OpenRetinaHoeflingFormat, schema,
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
            'DNoiseTraceParams': DNoiseTraceParams,
            'DNoiseTrace': DNoiseTrace,
            'STAParams': STAParams,
            'STA': STA,
            'PeakSTAPosition': PeakSTAPosition,
            'OpenRetinaHoeflingFormat': OpenRetinaHoeflingFormat
        }

        sleep(self.sleep_time_between_table_ops)
        from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
        prepare_dj_config_rgc_classifier(self.rgc_output_folder)

        from djimaging.utils.dj_utils import activate_schema

        activate_schema(schema=self.schema, create_schema=True, create_tables=True)
        sleep(self.sleep_time_between_table_ops)


            
            

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
        if len(self('UserInfo')()) == 0:
            self('UserInfo')().upload_user(self.userinfo)
            sleep(self.sleep_time_between_table_ops)

        self('RawDataParams')().add_default()
        sleep(self.sleep_time_between_table_ops)

        # TODO extract hard coded values to config
        self('RawDataParams')().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        sleep(self.sleep_time_between_table_ops)

        preprocess_params =self.table_parameters.get("PreprocessParams", {})
        self('PreprocessParams')().add_default(**preprocess_params)
   
        

        # Celltype assignment
        self('ClassifierTrainingData')().add_default(skip_duplicates=True)
        self('ClassifierMethod')().add_default(skip_duplicates=True)
        self('Classifier')().populate()

        # rf estimation 
        dense_noise_parmas = self.table_parameters.get("DNoiseTraceParams", {})
        self('DNoiseTraceParams')().add_default(**dense_noise_parmas)


        self('STAParams')().add_default()


        # spike estimation
        self('CascadeTraceParams')().add_default(stim_names=['mouse_cam'])
        self('CascadeParams')().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel') # for spike estimation itself


        

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
        sleep(self.sleep_time_between_table_ops)
        
        
        self.load_tables()
        sleep(self.sleep_time_between_table_ops)


        self.add_all_stimuli()
        sleep(self.sleep_time_between_table_ops)
        
        self.set_params_and_userinfo()
        sleep(self.sleep_time_between_table_ops)

##################################################################### During iteration functions ##############################################################################
    @time_it
    def upload_iteration_metadata(self) -> None:
        

        # TODO: rescan vermeiden, scant alle 
        if self.iteration == 0:
            self('Experiment')().rescan_filesystem(verboselvl=3)
            sleep(self.sleep_time_between_table_ops)
            
            self('OpticDisk')().populate(processes=self.multiprocessing_threads, display_progress=True)



        self('Field')().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        self('RelativeFieldLocation')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('Presentation')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)
 
        

    @time_it
    def add_iteration_rois(self) -> None:
        """
  
        """
        self('RoiMask')().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        missing_fields = self('RoiMask')().list_missing_field()
        assert len(missing_fields) == 1 , f"Expecting exatly no missing fields but found {len(missing_fields)}"
        field_key = missing_fields[0]
        sleep(self.sleep_time_between_table_ops)


        roi_canvas = self('RoiMask')().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        #TODO : REALLY UNSURE IF THIS DOES WHAT I WANT 
        # roi_canvas.exec_autorois_all_stacks()
        roi_canvas.exec_autorois_all()
        roi_canvas.exec_save_to_file()

        # add to database
        roi_canvas.insert_database(roi_mask_tab=self('RoiMask'), field_key=field_key)
        sleep(self.sleep_time_between_table_ops)
        self('Roi')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        if self.plot_results:
            self('RoiMask')().plot1()
            plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"roi_mask_{field_key}.png"), dpi=300, bbox_inches='tight')

    @time_it
    def add_iteration_traces(self) -> None:
        

        self('Traces')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('PreprocessTraces')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)


    @time_it
    def add_trace_reformatting(self) -> None:
        
        
        self('Snippets')().populate(processes=self.multiprocessing_threads, display_progress=True)
        
        sleep(self.sleep_time_between_table_ops)
        self('Averages')().populate(processes=self.multiprocessing_threads, display_progress=True)

    @time_it
    def add_quality_metrics(self) -> None:

        self('ChirpQI')().populate(display_progress=True, processes=self.multiprocessing_threads)
        self('OsDsIndexes')().populate(display_progress=True, processes=self.multiprocessing_threads)

    @time_it
    def add_sta(self) -> None:

        self('DNoiseTrace')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('STA')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

    @time_it
    def add_spikes(self) -> None:

        self('CascadeTraces')().populate()
        sleep(self.sleep_time_between_table_ops)

        # TODO: see if we can only do this for mouse cam??? bc takes a lot of time
        self('CascadeSpikes')().populate()


    def add_all_stimuli(self) -> None:
        import h5py

        with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)
        noise_stimulus = noise_stimulus*2-1
        
        
        self('Stimulus')().add_nostim(skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)

        
        self('Stimulus')().add_noise(**self.table_parameters.Stimulus.noise, stim_trace=noise_stimulus,)
        
        # self('Stimulus')().update1(dict(stim_name='densnoise',stim_family='noise', 
        #               stim_dict=dict( pix_scale_x_um=40, pix_scale_y_um=40, pix_n_x=20, pix_n_y=15, framerate = 5)))
        
        self('Stimulus')().add_movingbar(skip_duplicates=True)
        
        self('Stimulus')().add_stimulus(
            stim_name="mouse_cam", 
            alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
            stim_family="natural", 
            framerate=30.0,  
            stim_path="",
            ntrigger_rep=123,
            unique_alias=True
        )

    
    @time_it
    def add_celltype_assignments(self) -> None:

        self('Baden16Traces')().populate(display_progress=True, processes=self.multiprocessing_threads)
        self('CelltypeAssignment')().populate(display_progress=True)
        
        if self.plot_results:
            for threshold_confidence in [0, 0.25, 0.5]:
                self('CelltypeAssignment')().plot(threshold_confidence=threshold_confidence)
                plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')
                
                # self('CelltypeAssignment')().plot_features(threshold_confidence= threshold_confidence)
                # plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_features_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')
                
                self('CelltypeAssignment')().plot_group_traces(threshold_confidence= threshold_confidence)
                plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_traces_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')

            
        # if self.debug:
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.0)
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.25)
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.5)

    @time_it
    def add_peak_sta_positions(self) -> None:

        self('PeakSTAPosition')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

       
    
    @time_it
    def extract_data(self) -> Dict[str,Dict[str, Any]] | None:
        
        session_dict = self('OpenRetinaHoeflingFormat')().extract_data()
        return session_dict



    def process_iteration_data(self,) -> Dict[str,Dict[str, Any]] | None:
        """
        main function =
        Call this function to run the pipeline on each iteration of the loop
        """
        
        
        
        self.upload_iteration_metadata()
        self.add_iteration_rois()
        self.add_iteration_traces()


        self.add_trace_reformatting()
        self.add_quality_metrics()
        self.add_celltype_assignments()

        self.add_sta()
        self.add_peak_sta_positions()


        self.add_spikes()
        iteration_data = self.extract_data()

        self.iteration += 1

        return iteration_data


        
##################################################################  Other utils  #############################################################################################
    
    def clean_up(self, at_processing_stage = 'setup') -> None:
            """
            Clear schema tables.
            
            Args:
                all: If True, delete all tables. If False, only delete tables modified in each iteration.
            """


            self.iteration = 0
            


            # Tables that are modified in each iteration
            iteration_modified_table_names = [
                'RoiMask',
                'Roi', 
                'Traces',
                'PreprocessTraces',
                'Presentation',
                'PeakSTAPosition',
               
                'DNoiseTrace',
                'STA',

                'CascadeTraces',
                'CascadeSpikes',
            ]
            
            data_extraction_table_names = [
                'OpenRetinaHoeflingFormat',
            ]
            # Tables that are static across iterations
            static_table_names = [
                'UserInfo',
                'Experiment',
                'Field',
                'Stimulus', 
                'RawDataParams',
                'PreprocessParams',
                'Snippets',
                'Averages',
                'ChirpQI',
                'OsDsIndexes',
                'ClassifierMethod',
                'ClassifierTrainingData',
                'Classifier',
                'CascadeTraceParams',
                'CascadeParams',
                'Baden16Traces',
                'CelltypeAssignment',
                'DNoiseTraceParams',
                'STAParams',
            ]
            
            # Tables to delete based on 'all' parameter
            if at_processing_stage == 'setup':
                table_names_to_delete = static_table_names + iteration_modified_table_names + data_extraction_table_names
            elif at_processing_stage == 'iteration':
                table_names_to_delete =  iteration_modified_table_names + data_extraction_table_names
            elif at_processing_stage == 'data_extraction':
                table_names_to_delete = data_extraction_table_names
            else:
                raise ValueError("Invalid processing stage. Choose from 'setup', 'iteration', or 'data_extraction'.")
             
            # Delete tables with pause between each operation
            for table_name in table_names_to_delete:
                if self(table_name) is not None:
                    print(f"Deleting table: {table_name}")
                    
                    self(table_name)().delete()
                sleep(self.sleep_time_between_table_ops / 5)
            
            # check if there are ROIs dir in recording dir with Pre and Raw and delete it if so
            
            roi_dir = os.path.join(self.userinfo['data_dir'], str(self.data_subfolders["day"]),str(self.data_subfolders["experiment"]),'ROIs')
            if os.path.exists(roi_dir) and at_processing_stage != 'data_extraction':
                print(f"Deleting ROIs directory: {roi_dir}")
                user_ok = input(f"Are you sure you want to delete this directory? (y/n): ")
                if user_ok.lower() == 'y':
                    shutil.rmtree(roi_dir, ignore_errors=False)

            
