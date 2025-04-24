import os 
import shutil

class NoiseFileCopier:
    def __init__(self, 
                 home_directory,
                 source_path, 
                 dir_where_new_stim_appear,
                 debug: bool = False,
   ):
        
        self.home_directory = home_directory
        self.source_path = source_path
        self.dir_where_new_stim_appear = dir_where_new_stim_appear


        self.source_path = os.path.join(home_directory,source_path)
        self.destination_path_prefix = os.path.join(home_directory,dir_where_new_stim_appear,"closedloopdensenoise")
        self.iteration = 0
        
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source path {self.source_path} does not exist")
        os.makedirs(os.path.join(self.home_directory,self.dir_where_new_stim_appear), exist_ok=True)


        self.debug = debug

    def stimulate(self):

        destination_path = self.destination_path_prefix + f"{self.iteration}.h5"
        shutil.copy(self.source_path, destination_path)
        print(f"Copied noise file from {self.source_path} to {destination_path}")
        self.iteration += 1

    def clean_up(self):
        """
        clean up the destination directory
        """
        full_dir_where_new_stim_appear = os.path.join(self.home_directory,self.dir_where_new_stim_appear)
        for file in os.listdir(full_dir_where_new_stim_appear):
            if file.endswith(".h5"):
                os.remove(os.path.join(full_dir_where_new_stim_appear, file))
                print(f"Removed {file} from {os.path.join(full_dir_where_new_stim_appear)}")

        self.iteration = 0 
