from typing import Any, Callable, Dict, List, Optional
import os
from djimaging.utils.mask_utils import to_roi_mask_file

import os 
import shutil
from typing import List, Optional   


def create_directory_structure(base_directory: str,date: int ,experiment: int,):
    """Creates folder structure for DJ"""

    for subfolder in ["Raw","Pre"]:
        to_create = os.path.join(base_directory,str(date),str(experiment),subfolder)
        os.makedirs(to_create, exist_ok=True)



def copy_rec_files(recording_files_dir: str,
                    destination_base: str, 
                    date: int, 
                    experiment: int,
                    permissible_stimulus_types: List[str] = ["chirp","dn","mb"] + [f"mc{str(i)}" for i in range(0,21)],
                    full_dummy_ini_dir: Optional[str] = None,
                    ) -> None:
    """
    Copies all smp, smh and ini files from recording dir to a dir structure that one can use in DJ.
    """
    all_files_in_dir = os.listdir(recording_files_dir)


    # filter files
    filtered_files_in_dir = []
    for filename in all_files_in_dir:

        # wrong ending 
        if not filename.endswith(('.smp', '.smh', '.ini')):
            continue
        
        # Check if the file is a permissible stimulus type
        file_info_list = filename.split('.')[0].split('_')
        file_info_list = [info.lower() for info in file_info_list]  # Normalize to lowercase
        if not any(stimulus_type in file_info_list for stimulus_type in permissible_stimulus_types):
            print(f"SKIPPING File {filename}: does not match any permissible stimulus type.")
            continue

        filtered_files_in_dir.append(filename)

    

    for filename in filtered_files_in_dir:

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
        
        # check if file already exists
        if os.path.exists(new_path_full):
            print(f"SKIPPING File {new_path_full} already exists, skipping.")
            continue

        # Copy the files with different endings
        shutil.copy(source_file, new_path_full)
        print(f"COPIED file from {source_file} to {new_path_full}")
    

    # deal with the case of missing ini file
    has_ini_file = any(file.endswith('.ini') for file in filtered_files_in_dir)
    
    if not has_ini_file:

        if full_dummy_ini_dir is None:
            raise ValueError("full_dummy_ini_dir must be provided if no ini file is found")

        full_dummy_ini_file_path = os.path.join(full_dummy_ini_dir, "dummy.ini")
        dummy_ini_dest = os.path.join(destination_base, str(date), str(experiment), f"{str(date)}_left.ini")

        shutil.copy(full_dummy_ini_file_path, dummy_ini_dest)
        print(f"NO INI FILE found.\nCOPIED dummy ini file from {full_dummy_ini_file_path} to {dummy_ini_dest}")



def clear_roi_field_field(Presentation, field_key: Dict[str, Any], safemode: bool = True) -> None:
    # rm roi dirs        
    # # remove any roi masks saved in the directory
    all_field_presentation_files = (Presentation & field_key).fetch("pres_data_file")
    all_roi_mask_files = [to_roi_mask_file(file, roi_mask_dir="ROIs") for file in all_field_presentation_files]
    if len(all_roi_mask_files) == 0:
        print("No ROI mask files found to delete.")
        return
    # prompt user to confirm deletion
    if safemode:
        confirm = input(f"Are you sure you want to remove ROI maskfiles \n{"\n".join(all_roi_mask_files)} ROI mask files? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return
    # remove all roi mask files
    for file in all_roi_mask_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed file: {file}")
