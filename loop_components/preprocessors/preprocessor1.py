
import os
import datajoint as dj
import subprocess
from djimaging.utils.dj_utils import activate_schema

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
                

        self.schema_name = f"ageuler_{self.username}_test"

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


    def draw_rois(self):
        """
        draw rois on the recordings
        """
        pass

    def add_recordings(self):
        """
        add recordings to the schema if it exists. Used for further loop iterations. 
        """
        pass

    