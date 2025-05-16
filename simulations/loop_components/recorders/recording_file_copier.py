import os
import glob
import shutil
from time import sleep
import numpy as np  

class RecordingFileCopier:
    """
    Coplies smp smh and ini filed to an experiment dir to simulate new data coming in with each loop iteration
    The fiels are renamed to make each loop iteration a uniqule level of cond1.
    """

    def __init__(self,
                 repo_directory: str, 
                 stimulus_type: str,
                 sleep_time_between_file_ops: float = 0.1,
                 debug: bool = True,
                 ) -> None:
        
        
        self.repo_directory: str = repo_directory
        self.stimulus_type: str = stimulus_type
        self.sleep_time_between_file_ops: float = sleep_time_between_file_ops
        self.debug: bool = debug
        
        # name of experiment directory
        self.exp_num: str = str(1)
        self.exp_date: str = '20250412'
        self.new_file_name_base: str = "M1_LR_GCL0"
        self.stimulus_name_to_abbreviation: dict = {
            "closedloopdensenoise": "cldn",
            "closedloopmousecamera": "clmc",
            "closedloopchirp": "clchirp",
        }
        assert self.stimulus_type in self.stimulus_name_to_abbreviation.keys(), f"Stimulus type {self.stimulus_type} not in {self.stimulus_name_to_abbreviation.keys()}."

        self.iteration: int = 0


        # where to find files
        self.source_dir = os.path.join( self.repo_directory, 'data','recordings','static_test_data')
        assert os.path.exists(self.source_dir), f"Source directory {self.source_dir} does not exist."

        # store where to find the smp smh and ini files for a certain stimulus type
        self.ini_file, self.source_smp_file, self.source_smh_file = '','',''
        for file in os.listdir(self.source_dir):
            if self.stimulus_type in file:
                if file.endswith('.smp'):
                    self.source_smp_file: str = file
                elif file.endswith('.smh'):
                    self.source_smh_file: str = file
            elif file.endswith('.ini'):
                self.ini_file : str = file
        assert self.source_smp_file != '', f"Source directory {self.source_dir} does not contain any smp file for stimulus type {self.stimulus_type}."
        assert self.source_smh_file != '', f"Source directory {self.source_dir} does not contain any smh file for stimulus type {self.stimulus_type}."
        assert self.ini_file != '', f"Source directory {self.source_dir} does not contain any ini file{self.stimulus_type}."
               

        # where to create experiment dir and subdirs and copy files to
        self.target_dir: str = os.path.join(self.repo_directory, 'data','recordings','updated_loop_data',self.exp_date)
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
            

        

    def create_experiment_dir_structure(self) -> None:
        """
        create Experiment directory structure
        """

        # create experiment dir
        try:
            os.mkdir(os.path.join(self.target_dir, self.exp_num))
        except FileExistsError:
            print(f"Experiment directory {self.exp_num} already exists. Cleaning up.")
            self.clean_up()
            os.mkdir(os.path.join(self.target_dir, self.exp_num))

        # subdirs for stimuli and raw data
        os.mkdir(os.path.join(self.target_dir, self.exp_num, "Pre"))
        os.mkdir(os.path.join(self.target_dir, self.exp_num, "Raw"))

    def copy_ini(self) -> None:
        """
        copy ini file to the target directory structure of the experiment
        """
        source_path = os.path.join(self.source_dir, self.ini_file)
        target_path = os.path.join(self.target_dir, self.exp_num,self.ini_file)
        shutil.copy(source_path, target_path)


    def copy_to_dir_structure(self) -> None:
        """
        copy files to the target directory structure of the experiment. It renames them such that
        the stimulus is at loc 4 and the loop iteration is at loc 5 of the filename.
        """
        stimulus_abbreviation = self.stimulus_name_to_abbreviation[self.stimulus_type]
        new_file_full_name_woending = f'{self.new_file_name_base}_{stimulus_abbreviation}{self.iteration}_iter{self.iteration}'
        
        for sourcefilename in [self.source_smp_file, self.source_smh_file]:
            _, ext = os.path.splitext(sourcefilename)
            # copy the file to the target directory
            source_path = os.path.join(self.source_dir, sourcefilename)
            target_path = os.path.join(self.target_dir, self.exp_num, "Raw", new_file_full_name_woending + ext)
            shutil.copy(source_path, target_path)

 
    def clean_up(self) -> None:
        """
        clean up the experiment directory and set instance back to initial state
        """
        # remove the experiment directory
        if self.debug:
            user_input = input(f"removing experiment directory {self.exp_num} from {self.target_dir}.\nEnter 'yes' to continue.'yes' to continue.")
            if user_input == "yes":
                shutil.rmtree(os.path.join(self.target_dir, self.exp_num))
            else:
                raise ValueError(f"User input {user_input} is not 'yes'.")
        else:
            shutil.rmtree(os.path.join(self.target_dir, self.exp_num))

        self.iteration = 0

    def record(self) -> None:
        '''
        This simulates the recroding process. 
    
        '''

        if self.iteration == 0:
            # copy ini file into experiment dir
            self.create_experiment_dir_structure()
            self.copy_ini()
        sleep(self.sleep_time_between_file_ops)

        self.copy_to_dir_structure()
        sleep(self.sleep_time_between_file_ops)
        
        self.iteration += 1
          