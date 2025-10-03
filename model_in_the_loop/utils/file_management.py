from typing import Any, Callable, Dict, List, Optional
import os
from djimaging.utils.mask_utils import to_roi_mask_file
from omegaconf import DictConfig
import os 
import shutil
from typing import List, Optional   
import datetime


def create_directory_structure(base_directory: str,
                               experiment: int = 1,):
    """Creates folder structure for DJ"""

    date = datetime.date.today().strftime('%Y%m%d')


    for subfolder in ["Raw","Pre"]:
        to_create = os.path.join(base_directory,date,str(experiment),subfolder)
        os.makedirs(to_create, exist_ok=True)



def copy_rec_files(recording_files_dir: str,
                    destination_base: str, 
                    date: str | None = None, 
                    experiment: int  = 1,
                    permissible_stimulus_types: List[str] = ["chirp","dn","mb"] + [f"mc{str(i)}" for i in range(0,21)],
                    full_dummy_ini_dir: Optional[str] = None,
                    ini_file_static_args: Optional[Dict[str,Any]] = None,
                    ) -> None:
    """
    Copies all smp, smh and ini files from recording dir to a dir structure that one can use in DJ.
    """
    if isinstance(ini_file_static_args, DictConfig):
        ini_file_static_args = dict(ini_file_static_args)

    all_files_in_dir = os.listdir(recording_files_dir)
    if date is None:
        date = datetime.date.today().strftime('%Y%m%d')


    # filter files
    filtered_files_in_dir = []
    for filename in all_files_in_dir:

        # wrong ending 
        if not filename.endswith(('.smp', '.smh', '.ini')):
            continue
        
        # Check if the file is a permissible stimulus type
        file_info_list = filename.split('.')[0].split('_')
        file_info_list = [info.lower() for info in file_info_list]  # Normalize to lowercase
        
        if "lr" in file_info_list:
            eye = "left"
        elif "rr" in file_info_list:
            eye = "right"
        else:
            raise ValueError(f"File {filename} does not specify eye (lr or rr) in its name. Need this to get eye info")


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
            new_path_full = os.path.join(destination_base, date, str(experiment), "Raw", new_stim_file + "." + ending)
        elif ending == "ini":
            new_path_full = os.path.join(destination_base,date, str(experiment),stim_file + "." + ending)
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
        dummy_ini_dest = os.path.join(destination_base, date, str(experiment), f"{date}_{eye}.ini")
        
        # add date and eye to ini file
        modificatoins = ini_file_static_args if ini_file_static_args is not None else {}
        modificatoins.update({"string_date":  datetime.date.today().strftime('%Y-%m-%d'),
                              "string_eye": eye,
                              })
        print(f"Copying modified ini file because no ini file found in data dump ...")

        modify_ini_file_preserving_format(full_dummy_ini_file_path, dummy_ini_dest, modifications=modificatoins, verbose=True)




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


def rm_all_experiment_dirs(data_dir: str, safemode: bool = True) -> None:
    """
    Remove experiment dirs in data_dir
    """
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Nothing to delete.")
        return
    
    # get experiminet subdirs 
    experiment_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if len(experiment_dirs) == 0:
        print(f"No experiment directories found in {data_dir}. Nothing to delete.")
        return
    else:
        for exp_dir in experiment_dirs:
            # prompt user to confirm deletion
            if safemode:
                confirm = input(f"Are you sure you want to remove {exp_dir}? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("Deletion cancelled.")
                    return
            shutil.rmtree(exp_dir)
            print(f"Removed directory and all its contents: {exp_dir}")

def clear_data_dump_dir(data_dump_dir: str, safemode: bool = True) -> None:
    """
    Remove files in data dump dir
    """
    if not os.path.exists(data_dump_dir):
        raise ValueError(f"Directory {data_dump_dir} does not exist. Nothing to delete.")
    
    # if empty do nothin
    if len(os.listdir(data_dump_dir)) == 0:
        print(f"Directory {data_dump_dir} is already empty. Nothing to delete.")
        return

    if safemode:
        confirm = input(f"Are you sure you want to remove all files in {data_dump_dir}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return


    # remove all files in data dump dir
    for filename in os.listdir(data_dump_dir):
        file_path = os.path.join(data_dump_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                print(f"Skipping directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")




def modify_ini_file_preserving_format(input_path: str, 
                                      output_path: str, 
                                      modifications: Dict[str, Any],
                                      verbose = False) -> None:
    """
    Load an INI file, modify its entries while preserving comments and formatting,
    and save to a new location.
    
    Args:
        input_path (str): Path to the input INI file
        output_path (str): Path where the modified INI file will be saved
        modifications (Dict[str, Any]): Dictionary of modifications where:
            - Keys are parameter names (e.g., "string_eye", "string_date")
            - Values are the new values to set
    
    Example:
        modify_ini_file_preserving_format(
            "dummy.ini",
            "modified.ini",
            {
                "string_date": "2025-09-15",
                "string_userName": "newexperimenter",
                "string_eye": "right"
            }
        )
    """
    # Read the original file as text
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Track current section for line-by-line processing
    current_section = None
    modified_lines = []
    
    # Process each line
    for line in lines:
        line = line.rstrip('\n')
        
        # Check if this is a section header
        if line.strip().startswith('[') and line.strip().endswith(']'):
            current_section = line.strip()[1:-1]
            modified_lines.append(line)
            continue
            
        # Check if this is a key-value pair
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            
            # Check if this key should be modified
            if key in modifications:
                new_value = str(modifications[key])
                modified_lines.append(f"{key}={new_value}")
                if verbose:
                    print(f"Modified {key}: {value.strip()} -> {new_value}")
                continue
        
        # If not modified, keep the original line
        modified_lines.append(line)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the modified content to the output file
    with open(output_path, 'w') as f:
        f.write('\n'.join(modified_lines))
    if verbose:
        print(f"Modified INI file (with preserved formatting) saved to: {output_path}")