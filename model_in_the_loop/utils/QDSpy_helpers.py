import os 
import yaml
import shutil


def read_metadata(local_subdir_full: str) -> dict:
    files = os.listdir(local_subdir_full)
    all_yaml_files = [f for f in files if f.endswith('.yaml')]
    assert len(all_yaml_files) == 1, f"More than one yaml file found in '{local_subdir_full}': {all_yaml_files}"
    yaml_file_name = all_yaml_files[0]
    
    # Create absolute path by joining directory and file name
    yaml_file_path = os.path.join(local_subdir_full, yaml_file_name)
    
    print(f"READING CONFIG FROM {yaml_file_path}")
    with open(yaml_file_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    return metadata



def copy_stim_dir_to_local(stimulus_output_dir: str, remote_subdir_base: str, local_stimulus_dir:str) -> None:
  """ copies entire subdir from remote stimulus dir to loacal"""
  abs_remote_subdir = os.path.join(stimulus_output_dir, remote_subdir_base)
  abs_local_subdir = os.path.join(local_stimulus_dir, remote_subdir_base)
  if os.path.exists(abs_local_subdir):
    raise FileExistsError(f"Local subdir '{abs_local_subdir}' already exists. Remove it .")
  print(f"Copying entire directory from '{abs_remote_subdir}' to '{abs_local_subdir}'")
  shutil.copytree(abs_remote_subdir, abs_local_subdir)
  


def get_latest_remote_stimulus_subdir(stimulus_output_dir: str) -> str:
    """Returns the most recently modified directory (base name) in stimulus_output_dir."""
    directories = [os.path.join(stimulus_output_dir, d) for d in os.listdir(stimulus_output_dir) 
                  if os.path.isdir(os.path.join(stimulus_output_dir, d))]
    
    if not directories:
        raise FileNotFoundError(f"No directories found in '{stimulus_output_dir}'")
    
    # most recent mod dir
    latest_dir_path = max(directories, key=os.path.getmtime)
    latest_dir_name = os.path.basename(latest_dir_path)

    
    return latest_dir_name


def check_remote_files(remote_subdir) -> None:
  """checks if there is exaclty one yaml file and at least one avi file in the remote dir. returns a list of absolute avi file paths and the absolute yaml file path."""

  files = os.listdir(remote_subdir)

  avi_files = [f for f in files if f.endswith('.avi')]
  yaml_files = [f for f in files if f.endswith('.yaml')]


  if not avi_files:
    raise FileNotFoundError(f"No .avi files found in '{remote_subdir}'")
  if not yaml_files:
    raise FileNotFoundError(f"No .yaml files found in '{remote_subdir}'")
  if len(yaml_files) > 1:
    raise ValueError(f"Multiple .yaml files found in '{remote_subdir}': {yaml_files}")