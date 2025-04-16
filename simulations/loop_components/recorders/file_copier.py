import os
import shutil
class FileCopier:
    """
    A simple file mover that moves recording files from one directory to another.
    """

    def __init__(self, 
                 source_dir: str,
                 source_exp_dir:str, 
                 source_smp_file: str,
                 source_smh_file: str,
                 source_stim_file: str,
                 ini_file: str,
                 target_dir: str):
        
        
        # where to find files
        self.source_dir = source_dir
        self.source_exp_dir = source_exp_dir
        self.source_smp_file = source_smp_file
        self.source_smh_file = source_smh_file
        self.source_stim_file = source_stim_file
        self.ini_file = ini_file
        assert os.path.exists(self.source_dir), f"Source directory {self.source_dir} does not exist."


        # where to output files
        self.target_dir = target_dir

        # create 
        os.mkdir(self.target_dir)

        
        self.iteration = 0

    @staticmethod
    def create_target_dir_structure(target_dir: str,exp_num: int) -> None:
        """
        create target dir structure
        """

        # create experiment dir
        os.mkdir(os.path.join(target_dir, str(exp_num)))

        # subdirs for stimuli and raw data
        os.mkdir(os.path.join(target_dir, str(exp_num), "Pre"))
        os.mkdir(os.path.join(target_dir, str(exp_num), "Raw"))

    def copy_to_dir_structure(self, exp_num:int) -> None:
        """
        copy files to the target directory structure of the experiment
        """
        # copy ini file TODO: the ini file is copied with same name, this could give problems 
        source_path = os.path.join(self.source_dir, self.source_exp_dir, self.ini_file)
        target_path = os.path.join(self.target_dir, str(exp_num), self.ini_file)
        shutil.copy(source_path, target_path)

        # copy smp and smh files into Raw: smp  
        source_path = os.path.join(self.source_dir, self.source_exp_dir, "Raw",self.source_smp_file)
        target_path = os.path.join(self.target_dir, str(exp_num), "Raw",self.source_smp_file)
        shutil.copy(source_path, target_path)

        # copy file into Raw: smh
        source_path = os.path.join(self.source_dir, self.source_exp_dir, "Raw",self.source_smh_file)
        target_path = os.path.join(self.target_dir, str(exp_num), "Raw", self.source_smh_file)
        shutil.copy(source_path, target_path)

        # copy stimulus files into Pre
        source_path = os.path.join(self.source_dir, self.source_exp_dir, "Pre",self.source_stim_file)
        target_path = os.path.join(self.target_dir, str(exp_num), "Pre",self.source_stim_file)
        shutil.copy(source_path, target_path)
        

    def record(self):
        '''
        This simulates the recroding process. It creates a new experiment dir. 
    
        '''

        self.iteration += 1

        # create target dir structure
        self.create_target_dir_structure(self.target_dir, self.iteration)

        # copy files into exp dir 
        self.copy_to_dir_structure(self.iteration)

          