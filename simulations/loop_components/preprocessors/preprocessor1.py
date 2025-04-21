
import os
import datajoint as dj
import subprocess
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any


from djimaging.utils.dj_utils import activate_schema

# This imports `schema`. TODO: change this to general closde loop schema
from djimaging.user.ssuhai.schemas.ssuhai_schema import UserInfo, \
    Experiment, Field, Stimulus, RoiMask, Roi, Traces, \
    Presentation, RawDataParams, \
    schema



class Preprocessor1:
    """
    This class 
    1. takes recording files (in the correct folder structure)
    2. sets up a schema in djimaging to store them if there does not exist one 
    3. 

    """

    def __init__(self,
                 username: str,
                 home_directory: str,
                 path_to_djimaging_rel_to_home: str,
                 path_to_djconfig_rel_to_home: str,
                 userinfo: dict,
                 sleep_time_between_table_ops: int  = 1,
                 ):
        """

        """


        """ Should I put these in configs?
        """
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
        self.userinfo: dict = userinfo        

        self.schema_name = f"ageuler_{self.username}_test"

        self.iteration: int = 0

        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops



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



    def upload_iteration_data(self) -> None:
        
        # TODO: rescan vermeiden, scant alle 
        if self.iteration == 0:
            Experiment().rescan_filesystem(verboselvl=3)
            sleep(self.sleep_time_between_table_ops)


        Field().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)


        #TODO: this is dummy stimulus, need to add my own
        Stimulus().add_stimulus(stim_name=f'closedloop{self.iteration}', alias=f"closedloop{self.iteration}_cl{self.iteration}_{self.iteration}closedloop", isrepeated=True, ntrigger_rep=2,
                                trial_info=[1, 2], skip_duplicates=True)
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


    def add_iteration_traces(self) -> None:
        

        Traces().populate(processes=20, display_progress=True)
        sleep(self.sleep_time_between_table_ops)


    def process_data(self,connect_and_activate: bool = True,return_traces: bool =False) -> None:
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
        self.add_iteration_traces()

        self.iteration += 1

        return (Traces() & dict(cond1=f'iter{self.iteration - 1}')).fetch() if return_traces else None
###############################################################################################################################################################
