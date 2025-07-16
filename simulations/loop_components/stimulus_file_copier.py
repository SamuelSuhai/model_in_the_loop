import os 
import shutil
import glob 


def create_directory_structure(base_directory: str,date: int ,experiment: int,):
    """Creates folder structure for DJ"""

    for subfolder in ["Raw","Pre"]:
        to_create = os.path.join(base_directory,str(date),str(experiment),subfolder)
        os.makedirs(to_create, exist_ok=True)



def copy_stim_files(recording_files_dir: str,destination_base: str, date: int, experiment: int) -> None:
    """
    Copies all smp, smh and ini files from recording dir to a dir structure that one can use in DJ.
    """
    all_files_in_dir = os.listdir(recording_files_dir)

    for filename in all_files_in_dir:

        if not os.path.isfile(os.path.join(recording_files_dir, filename)):
            continue


        # Get the full source path when needed
        source_file = os.path.join(recording_files_dir, filename)
        

        # Split filename and extension
        name_parts = filename.split('.')
        if len(name_parts) < 2:
            continue  # Skip files without extensions
            
        stim_file, ending = name_parts[0], name_parts[1]
     
        
        # first deal with smp and smh files 
        if ending in ["smp", "smh"]:
            new_stim_file = stim_file + "_iter0" if not "iter" in stim_file else stim_file
            new_path_full = os.path.join(destination_base, str(date), str(experiment), "Raw", new_stim_file + "." + ending)
        elif ending == "ini":
            new_path_full = os.path.join(destination_base,str(date), str(experiment),stim_file + "." + ending)
        else:
            raise ValueError(f"Unknown file ending {ending} for file {filename}")

        # Copy the files with different endings
        shutil.copy(source_file, new_path_full)
        print(f"Copied file from {source_file} to {new_path_full}")



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
