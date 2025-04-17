import os
import shutil
from time import sleep

class FileCopier:
    """
    Coplies smp smh and ini filed to an experiment dir to simulate new data coming in with each loop iteration
    The fiels are renamed to make each loop iteration a uniqule level of cond1.
    """

    def __init__(self, 
                 source_dir: str,
                 source_exp_dir:str, 
                 source_smp_file: str,
                 source_smh_file: str,
                 ini_file: str,
                 target_dir: str,
                 new_file_name_base: str, # TODO make this from souce smp smh file just remove last parts
                 sleep_time_between_file_ops: float = 0.1,
                 ) -> None:
        
        
        # where to find files
        self.source_dir = source_dir
        self.source_exp_dir = source_exp_dir
        self.source_smp_file = source_smp_file
        self.source_smh_file = source_smh_file
        self.ini_file = ini_file
        assert os.path.exists(self.source_dir), f"Source directory {self.source_dir} does not exist."


        # where to create experiment dir and subdirs and copy files to
        self.target_dir: str = target_dir
        self.exp_num: str = str(1)
        assert os.path.exists(self.target_dir), f"Target directory {self.target_dir} does not exist."
        self.new_file_name_base: str = new_file_name_base
        
        self.iteration = 0
        self.sleep_time_between_file_ops = sleep_time_between_file_ops

    def create_experiment_dir_structure(self) -> None:
        """
        create Experiment directory structure
        """
        # TODO: make exp_num something in a config file

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
        source_path = os.path.join(self.source_dir, self.exp_num, self.ini_file)
        target_path = os.path.join(self.target_dir, self.exp_num, self.ini_file)
        shutil.copy(source_path, target_path)


    def copy_to_dir_structure(self) -> None:
        """
        copy files to the target directory structure of the experiment. It renames them such that
        the stimulus is at loc 4 and the loop iteration is at loc 5 of the filename.
        TODO: what position in file name could be read from config
        """

        new_file_full_name = f'{self.new_file_name_base}_cl{self.iteration}_iter{self.iteration}'

        # copy smp file into Raw
        source_path = os.path.join(self.source_dir, self.source_exp_dir, "Raw",self.source_smp_file)
        target_path = os.path.join(self.target_dir, self.exp_num, "Raw", new_file_full_name + '.smp')
        shutil.copy(source_path, target_path)

        # copy smh file into Raw
        source_path = os.path.join(self.source_dir, self.source_exp_dir, "Raw",self.source_smh_file)
        target_path = os.path.join(self.target_dir, self.exp_num, "Raw", new_file_full_name + '.smh')
        shutil.copy(source_path, target_path)
 
    def clean_up(self) -> None:
        """
        clean up the experiment directory and set instance back to initial state
        """
        # remove the experiment directory
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
          