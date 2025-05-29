import os 
import shutil

class StimulusFileCopier:

    permissible_stimulus_types = ["closedloopdensenoise", "closedloopmousecamera","closedloopchirp"]

    def __init__(self, 
                repo_directory,
                stimulus_type: str,
                debug: bool = True,
   ):   
        if not stimulus_type in self.permissible_stimulus_types:
            raise ValueError(f"stimulus_type must be one of {self.permissible_stimulus_types}")
        
        self.stimulus_type = stimulus_type
        self.repo_directory = repo_directory
        self.source_path = os.path.join(repo_directory, 'data', 'stimuli','static_test_data',stimulus_type + '.h5')
        self.dir_where_new_stim_appear = os.path.join(repo_directory, 'data', 'stimuli','updated_loop_data')

        self.iteration = 0
        
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source path {self.source_path} does not exist")
        
        os.makedirs(self.dir_where_new_stim_appear, exist_ok=True)
        self.debug = debug
        
        if os.listdir(self.dir_where_new_stim_appear) != []:
            self.clean_up()

    def stimulate(self):

        destination_path = self.dir_where_new_stim_appear + f"/{self.stimulus_type}{self.iteration}.h5"
        shutil.copy(self.source_path, destination_path)
        if self.debug:
            print(f"Copied noise file from {self.source_path} to {destination_path}")
        self.iteration += 1

    def clean_up(self):
        """
        clean up the destination directory
        """

        for file in os.listdir(self.dir_where_new_stim_appear):
            if file.endswith(".h5"):
                os.remove(self.dir_where_new_stim_appear + f"/{file}")
                if self.debug:
                    print(f"Removed {file} from {self.dir_where_new_stim_appear}")
            else:
                if self.debug:
                    print(f"File {file} is not a .h5 file, skipping")
        self.iteration = 0 
