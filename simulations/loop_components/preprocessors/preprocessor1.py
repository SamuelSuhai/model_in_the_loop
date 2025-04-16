
import os
import datajoint as dj
import subprocess
from djimaging.utils.dj_utils import activate_schema
import warnings
warnings.simplefilter("ignore", FutureWarning)

# This imports `schema`. TODO: change this to general closde loop schema
from djimaging.user.ssuhai.schemas.ssuhai_schema import *


class Preprocessor1:
    """
    This class 
    1. takes recording files (in the correct folder structure)
    2. sets up a schema in djimaging to store them if there does not exist one 
    3. 

    """

    def __init__(self,
                 path_to_djimaging_rel_to_home: str,
                 userinfo: dict,
                 ):
        """

        """


        """ Should I put these in configs?
        """
        # check who is running command
        self.username: str = subprocess.check_output(["whoami"]).decode().strip()

        # get home directory
        self.home_directory: str = os.path.expanduser("~")

        # check if the path to djimaging is correct
        self.path_to_djimaging: str = os.path.join(self.home_directory, path_to_djimaging_rel_to_home)
        if not os.path.exists(self.path_to_djimaging):
            raise ValueError(f"Path to djimaging does not exist: {self.path_to_djimaging}")
        
        # Set djimaging condfig file
        self.config_file:str = f'{self.home_directory}/datajoint/dj_{self.username}_conf.json'
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'

        # information for UserInfo table
        self.userinfo: dict = userinfo        

        self.schema_name = f"ageuler_{self.username}_test"

        self.iteration_nr = 0

    def load_config(self):
        """
        load config file
        """
        # Load configuration for user
        dj.config.load(self.config_file)
        dj.config['schema_name'] = self.schema_name


        # TODO setup log
        print("schema_name:", dj.config['schema_name'])
        dj.conn()


    def initial_schema_activation(self):
        """
        setup schema in djimaging
        """

        # schema should availible from the import
        activate_schema(schema=schema, create_schema=True, create_tables=True)

        # store the schema that we activated
        self.schema = schema


        # TODO: link the tables with attribute of the class

    def set_metadata(self):
        UserInfo().upload_user(self.userinfo)

        if len(RawDataParams()) == 0:
            RawDataParams().add_default()
        
        RawDataParams().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))



    def get_iteration_rois(self):
        """
        draw rois on the data from one loop iteration
        TODO: add option to take the rois from last iteration
        """
        RoiMask().rescan_filesystem(verboselvl=3)
        missing_fields = RoiMask().list_missing_field()
        assert missing_fields, "No missing fields found. Please check the ROI mask table."
        field_key = missing_fields[0]

        # TODO
        # somehow there is an error in getting some property of the loaded smp file object 
        # but the error occurs in the SMH class. Some dict key is not found.
        roi_canvas = RoiMask().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        roi_canvas.start_gui()
        if input("Done with Roi checking? (yes/no))") != "yes":
            raise ValueError('Enter yes if you wish to continue.')
        roi_canvas.insert_database(roi_mask_tab=RoiMask, field_key=field_key)

    def add_iteration_traces(self):
        Roi().populate(processes=20, display_progress=True)
        Traces().populate(processes=20, display_progress=True)

    def upload_iteration_data(self):
        

        if self.iteration_nr == 0:
            Experiment().rescan_filesystem(verboselvl=3)


        Field().rescan_filesystem(verboselvl=3)


        #TODO: this is dummy stimulus, need to add my own
        Stimulus().add_stimulus(stim_name='closedloop1', alias="closedloop1_cl1_firstclosedloop", isrepeated=True, ntrigger_rep=2,
                                trial_info=[1, 2], skip_duplicates=True)


        Presentation().populate(processes=20, display_progress=True)



    def main(self):
        """
        Call this function to run the preprocessor on each iteration of the loop
        """
        if self.iteration_nr == 0:
            self.load_config()
            self.initial_schema_activation()
            self.set_metadata()
        
        self.upload_iteration_data()
        self.get_iteration_rois()
        self.add_iteration_traces()
        self.iteration_nr += 1
        


    def clear_schema_tabels(self):

        UserInfo().delete()
        Experiment().delete()
        Field().delete()
        Stimulus().delete()
        RoiMask().delete()
        Roi().delete()
        Traces().delete()
        Presentation().delete()
        RawDataParams().delete()
        RawData().delete()
        

    