import os
import shutil
class FileMover:
    """
    A simple file mover that moves recording files from one directory to another.
    """

    def __init__(self, 
                 source_dir: str, 
                 source_smp_file: str, 
                 # TODO 
                 target_dir: str):
        self.source_dir = source_dir
        
        # where to output files
        self.target_dir = target_dir
        assert os.path.exists(self.source_dir), f"Source directory {self.source_dir} does not exist."
        
        # make sure folder is empty
        assert not os.path.exists(self.target_dir), f"Target directory {self.target_dir} already exists."

        self.iteration = 0

    def copy_file(self, new_filename: str, old_filename: str) -> None:
        """
        copy file with new name 

        Args:
            filename (str): The name of the file to move.
        """


        source_path = os.path.join(self.source_dir, old_filename)
        target_path = os.path.join(self.target_dir, new_filename)

        assert os.path.exists(self.source_dir), f"Source directory {self.source_dir} does not exist."
        assert os.path.exists(self.target_dir), f"Target directory {self.target_dir} does not exist."

        shutil.copy(source_path, target_path)

    def record(self):

        # create dirs in target dir
         pass     