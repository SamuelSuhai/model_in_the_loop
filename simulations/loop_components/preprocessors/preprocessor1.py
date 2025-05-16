
import os
import datajoint as dj
import subprocess
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

from djimaging.utils.dj_utils import activate_schema

# This imports `schema`. TODO: change this to general closde loop schema
from djimaging.user.ssuhai.schemas.ssuhai_schema import UserInfo, \
    Experiment, Field, Stimulus, RoiMask, Roi, Traces, \
    Presentation, RawDataParams, PreprocessParams, PreprocessTraces,\
    schema



class Preprocessor1:
    """
    This class 
    1. takes recording files (in the correct folder structure)
    2. sets up a schema in djimaging to store them if there does not exist one 
    3. 

    """

    permissible_stimulus_types = ["closedloopdensenoise", "closedloopmousecamera","closedloopchirp"]

    def __init__(self,
                 username: str,
                 home_directory: str,
                 repo_directory: str,
                 path_to_djimaging_rel_to_home: str,
                 path_to_djconfig_rel_to_home: str,
                 userinfo: dict,
                 
                 openretina_processed_data_path: str,
                 stimulus_shape: List[int],

                 sleep_time_between_table_ops: int  = 1,
                 stimulus_type: str = 'closedloopdensenoise',
                 test_fraction: float = 0.2,
                 debug: bool = False,
                 plotting: dict = {},

                 ):
        """
        
        Should I put these in configs?
        """


        self.iteration: int = 0

        # open retina stuff
        self.test_fraction: float = test_fraction

        self.debug: bool = debug
        self.plotting: dict = plotting

        # repo directory
        self.repo_directory: str = repo_directory
        assert os.path.exists(self.repo_directory), f"Path to repo does not exist: {self.repo_directory}"

        # check who is running command
        self.username: str = username
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()

        # get home directory
        self.home_directory: str = home_directory
        assert self.home_directory == os.path.expanduser("~")

        # check if the path to djimaging is correct
        self.path_to_djimaging: str = os.path.join(self.home_directory, path_to_djimaging_rel_to_home)
        if not os.path.exists(self.path_to_djimaging):
            raise ValueError(f"Path to djimaging does not exist: {self.path_to_djimaging}")
        
        # Set djimaging condfig file
        self.config_file:str = os.path.join(self.home_directory,path_to_djconfig_rel_to_home, f'dj_{self.username}_conf.json')
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'

        # information for UserInfo table
        _data_dir = os.path.join(self.repo_directory, 'data','recordings','updated_loop_data')
        assert os.path.exists(_data_dir), f"Path to data dir does not exist: {_data_dir}"
        self.userinfo: dict = {**userinfo,'data_dir': _data_dir}      
        self.schema_name = f"ageuler_{self.username}_test"
        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops




        self.store_initial_stimulus_info(
            stimulus_type=stimulus_type,
            stimulus_shape=stimulus_shape,
        )
        

        # set openretina processed data path
        self.openretina_processed_data_path = os.path.join(self.repo_directory,openretina_processed_data_path)
        os.makedirs(self.openretina_processed_data_path, exist_ok=True)
        if os.listdir(self.openretina_processed_data_path) != []:
            warnings.warn(f"There are files already in {self.openretina_processed_data_path}. They will be overwritten", category=UserWarning)


 
        
    def store_initial_stimulus_info(self,
                                    stimulus_type: str,
                                    stimulus_shape: List[int],
                                    ) -> None:
        # TODO: these are configurations I think. Maybe they should be in a config file. Especially stimulus_info_dict

        # set static stimulus information that does not change every loop iteration
        self.stimulus_type: str = stimulus_type
        
        self.dir_where_new_stim_appear: str = os.path.join(self.repo_directory, 'data', 'stimuli','updated_loop_data')
        self.stimulus_shape: List[int] = stimulus_shape 

        if not self.stimulus_type in self.permissible_stimulus_types:
            raise ValueError(f"stimulus_type must be one of {self.permissible_stimulus_types}")

        if stimulus_type.lower() == 'closedloopdensenoise':
            stim_name_func = lambda x: f'closedloopdensenoise{x}'
            alias_func = lambda x:  f"closedloopdensenoise{x}_cldn{x}_{x}closedloopdensenoise"

            self.stimulus_info_dict: Dict[str, Any] = dict(
                stim_name_func=stim_name_func, 
                alias_func=alias_func, pix_n_x=20, pix_n_y=15, 
                pix_scale_x_um=30, pix_scale_y_um=30, 
                stim_trace=None, 
                skip_duplicates=True
            )
        elif stimulus_type.lower() == 'closedloopchirp':
            raise NotImplementedError(f"stimulus type {stimulus_type} not implemented yet")
        
        elif stimulus_type.lower() == 'closedloopmousecamera':
            stim_name_func = lambda x: f'closedloopmousecamera{x}'
            alias_func = lambda x:  f"closedloopmousecamera{x}_clmc{x}_{x}closedloopmousecamera"

            self.stimulus_info_dict: Dict[str, Any] = dict(
                stim_name_func=stim_name_func, 
                alias_func=alias_func,
                stim_trace=None, 
                skip_duplicates=True,
                ntrigger_rep=123,
                isrepeated=False,
                framerate= 30,
            )
        else:
            raise ValueError(f"stimulus type {stimulus_type} not recognized mut be part either closedloopdensenoise, chirp or closedloopmousecamera")

    def load_config(self) -> None:
        """
        load config file
        """
        # Load configuration for user
        dj.config.load(self.config_file)
        dj.config['schema_name'] = self.schema_name
        sleep(self.sleep_time_between_table_ops)

        # TODO setup log
        print("schema_name:", dj.config['schema_name'])
        dj.conn()
        sleep(self.sleep_time_between_table_ops)



    def initial_schema_activation(self) -> None:
        """
        setup schema in djimaging
        """

        # schema should availible from the import
        activate_schema(schema=schema, create_schema=True, create_tables=True)
        sleep(self.sleep_time_between_table_ops)
        
        # store the schema that we activated
        self.schema = schema
        

    def set_metadata(self) -> None:

        # make sure tables are empty 

        if len(UserInfo()) > 0:
            if input("UserInfo not empty, clear all tables of schema (yes/no))") == "yes":
                self.clear_schema_tabels()
            else:
                raise ValueError('clear tables before continuing')

        UserInfo().upload_user(self.userinfo)
        sleep(self.sleep_time_between_table_ops)

        RawDataParams().add_default()
        sleep(self.sleep_time_between_table_ops)

        
        RawDataParams().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        sleep(self.sleep_time_between_table_ops)


    def add_cldn_stimulus(self) -> None:

        # add stim_trace to dict
        current_stimulus_path = os.path.join(self.home_directory, self.dir_where_new_stim_appear,f"closedloopdensenoise{self.iteration}.h5")
        with h5py.File(current_stimulus_path, "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)

        
        # add to dict if it agrees wiith test fraction 
        assert noise_stimulus.shape[0] * self.test_fraction % self.stimulus_shape[0] == 0 , f"Test stimulus shape {noise_stimulus.shape[0] * self.test_fraction} not divisible by {self.stimulus_shape[0]}"
        assert noise_stimulus.shape[0] * (1 - self.test_fraction) % self.stimulus_shape[0] == 0 , f"Train stimulus shape {noise_stimulus.shape[0] * (1 - self.test_fraction)} not divisible by {self.stimulus_shape[0]}"
        self.stimulus_info_dict['stim_trace'] = noise_stimulus
        
        stim_kwargs = self.stimulus_info_dict.copy()
        stim_kwargs['stim_name'] = stim_kwargs['stim_name_func'](self.iteration)
        stim_kwargs['alias'] = stim_kwargs['alias_func'](self.iteration)
        stim_kwargs.pop('stim_name_func')
        stim_kwargs.pop('alias_func')

        # add to database
        Stimulus().add_noise(**stim_kwargs)
        

    def add_clmc_stimulus(self) -> None:
        
        # add stim_trace to dict
        # TODO: add stim trace
        current_stimulus_path = None
        self.stimulus_info_dict['stim_trace'] = None

        print("stimulus trace not implemented yet !!!!!!!!!!!")
        
        stim_kwargs = self.stimulus_info_dict.copy()
        stim_kwargs['stim_name'] = stim_kwargs['stim_name_func'](self.iteration)
        stim_kwargs['alias'] = stim_kwargs['alias_func'](self.iteration)
        stim_kwargs.pop('stim_name_func')
        stim_kwargs.pop('alias_func')

        Stimulus().add_stimulus(**stim_kwargs)




    def upload_iteration_data(self) -> None:
        

        # TODO: rescan vermeiden, scant alle 
        if self.iteration == 0:
            Experiment().rescan_filesystem(verboselvl=3)
            sleep(self.sleep_time_between_table_ops)


        Field().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        
        
        if self.stimulus_type.lower() == 'closedloopdensenoise':
            self.add_cldn_stimulus()
        
        elif self.stimulus_type.lower() == 'closedloopchirp':
            # TODO: allow chirp for roi mask and classification
            raise NotImplementedError (f'stimulus type {self.stimulus_type} not implemented')   
        
        elif self.stimulus_type.lower() == 'closedloopmousecamera':
            self.add_clmc_stimulus()
        else:
            raise ValueError(f"stimulus type {self.stimulus_type} not recognized mut be closedloopdensenoise, chirp or closedloopmousecamera")

        sleep(self.sleep_time_between_table_ops)

        Presentation().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)
    
    def clean_up(self) -> None:
        """
        clean up tables instance back to initial state
        """
        self.clear_schema_tabels()

        self.iteration = 0

    def clear_schema_tabels(self) -> None:

        UserInfo().delete()
        sleep(self.sleep_time_between_table_ops)
        
        Experiment().delete()
        sleep(self.sleep_time_between_table_ops)

        Field().delete()
        sleep(self.sleep_time_between_table_ops)

        Stimulus().delete()
        sleep(self.sleep_time_between_table_ops)

        RoiMask().delete()
        sleep(self.sleep_time_between_table_ops)

        Roi().delete()
        sleep(self.sleep_time_between_table_ops)

        Traces().delete()
        sleep(self.sleep_time_between_table_ops)

        Presentation().delete()
        sleep(self.sleep_time_between_table_ops)

        RawDataParams().delete()
        sleep(self.sleep_time_between_table_ops)

        PreprocessParams().delete()
        sleep(self.sleep_time_between_table_ops)
        
        PreprocessTraces().delete()
        






    
    ######################################################################### Main functions ###############################################################################
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

        if self.debug:
            RoiMask().plot1()
            save_path = self.plotting.get("save_path", "plotting")
            plt.gcf().savefig(os.path.join(self.repo_directory,save_path, f"roi_mask_{self.iteration}.pdf"))



    def add_iteration_traces_and_preprocess(self) -> None:
        

        Traces().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)
        PreprocessParams().add_default(skip_duplicates=True)
        sleep(self.sleep_time_between_table_ops)
        PreprocessTraces().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

    def save_iteration_data_for_openretina(self,trace_type: str = "smoothed_trace") -> None:
        """
        Retrieve informaiton for training an openretina model
        """

        assert trace_type in ["smoothed_trace", "pp_trace"]


        # 1) for neural response
        iter_key = dict(cond1=f'iter{self.iteration}')

        trace = (PreprocessTraces() & iter_key).fetch(trace_type)

        # assume these values are the same for each field
        trace_t0, trace_dt = (PreprocessTraces() & iter_key & dict(roi_id=1) ).fetch1("pp_trace_t0", "pp_trace_dt")

        # reshape and tests
        trace = np.concatenate(trace).reshape(trace.shape[0], -1)
        trace_times = np.arange(trace[1].size) * trace_dt + trace_t0

        # 2) for stimulus
        triggertimes = (Presentation() & iter_key & dict(roi_id=1)).fetch1("triggertimes") 
        stim_dt = np.mean(np.diff(triggertimes))
        
        # 3) make stimuli and response have same sampling frequency
        nr_ca_samples_per_stimulus_trigger =  int(stim_dt / trace_dt)

        response = []
        for trigg_time in triggertimes:

            closest_idx = np.argmin(np.abs(trace_times - trigg_time))
            response.append(trace[:,closest_idx:closest_idx + nr_ca_samples_per_stimulus_trigger])
        response = np.concatenate(response, axis=1)

        # 4) create train and test split
        shuffled_idx = np.random.permutation(len(triggertimes))
        test_idx = shuffled_idx[:int(len(shuffled_idx) * self.test_fraction)]
        train_idx = shuffled_idx[int(len(shuffled_idx) * self.test_fraction) : ]
        
        test_response = response[:,test_idx]
        train_response = response[:,train_idx]
        test_stimulus = self.stimulus_info_dict['stim_trace'][test_idx][np.newaxis, ...]
        train_stimulus = self.stimulus_info_dict['stim_trace'][train_idx][np.newaxis, ...]

        # tests for stimulus shape 
        assert len(triggertimes) % self.stimulus_shape[0] == 0, f"triggertimes not divisible by stimulus shape {self.stimulus_shape[0]}"
        assert test_response.shape[1] == test_stimulus.shape[1], f"test response and stimulus not same shape {test_response.shape} {test_stimulus.shape}"
        assert train_response.shape[1] == train_stimulus.shape[1], f"train response and stimulus not same shape {train_response.shape} {train_stimulus.shape}"

        assert len(test_stimulus.shape) == 4
        assert len(train_stimulus.shape) == 4
        
        # 5) save to directory
        save_file_name: str = os.path.join(self.openretina_processed_data_path, f"openretina_data_iter{self.iteration}.pkl")
        with open(save_file_name, "wb") as f:
            pickle.dump(
                   {"test_response": test_response,
                "train_response": train_response,
                "test_stimulus": test_stimulus,
                "train_stimulus": train_stimulus,
                },
                f,
             )
            
        print(f"Saved openretina data to {save_file_name}")




    def process_data(self,connect_and_activate: bool = True,save_iteration_data_for_openretina: bool = True):
        """
        main function =
        Call this function to run the preprocessor on each iteration of the loop
        """
        

        if self.iteration == 0:
            if connect_and_activate:
                self.load_config()
                sleep(self.sleep_time_between_table_ops)
                self.initial_schema_activation()
                sleep(self.sleep_time_between_table_ops)
            self.set_metadata()
        
        self.upload_iteration_data()
        self.add_iteration_rois()
        self.add_iteration_traces_and_preprocess()

        if save_iteration_data_for_openretina:
            self.save_iteration_data_for_openretina()

        self.iteration += 1

        
###############################################################################################################################################################
