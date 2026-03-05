# Model-in-the-loop: A Pipeline for Same-Session Validation of Retinal Digital Twins for Calcium Imaging

This repository accompanies my [MSc Thesis](https://www.overleaf.com/read/zbkkjfzzfjsz#2ac696).  
It builds on [djimaging](https://github.com/eulerlab/djimaging), [open-retina](https://github.com/open-retina/open-retina) (see D’Agostino, Zenkel, et al., 2025), and [QDSpy](https://github.com/eulerlab/QDSpy). Parts of the code are copied from or modified versions of code found in these repositories.

The goal of this repository is to enable semi-automatic closed-loop experiments in which digital twins can be trained and tested within a single experiment session.

---

# Installation

1. Clone the following repositories:
   - [djimaging](https://github.com/eulerlab/djimaging)  
   - [open-retina](https://github.com/open-retina/open-retina)  
   - [ScanM support](https://github.com/eulerlab/ScanM_support)

2. Update the paths in the `requirements.txt` file so that they point to the cloned repositories.

3. Create an environment and install the required packages.

4. Adjust `config.yaml`. The following fields must be updated:  
   `repo_directory`, `home_directory`, `dj_config_directory`, `recording_files_dir`, `stimulus_output_dir`, `set_cache_dir_openretina`, `load_model_path`.  
   See the comments in the file for guidance.

5. Update the username in `config/DJ.yaml`. You can also adjust parameters of the djimaging tables here.

---

# Description

**core/**  
Contains `DJComputeWrappers`, schemas, and GUI-related code.

**data/**  
Directory where new data should be placed (`data/new_data_dump`).

**models/**  
Pretrained models used for refinement.

**utils/**  
Helper scripts, including:
- `transform_to_avi_stimulus.py`: functions for converting optimized stimuli to `.avi`, saving stimuli and metadata in an iteration folder.  
- `model_trainig.py`: utilities for refinement (training-related helpers).  
- `QDSpy_helpers/`: functions imported by the Stimulus PC for reading metadata.
