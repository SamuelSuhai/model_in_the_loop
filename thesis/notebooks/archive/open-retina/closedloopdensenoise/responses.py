import pickle 
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.utils.file_utils import get_cache_directory
from .constants import (PATH_IN_CACHE, STIMULUS_SHAPE)



def load_all_responses(
        stim_type: str = "closedloopdensenoise",
        iteration: int = 0,
):
    

    if stim_type != "closedloopdensenoise":
        raise NotImplementedError(f"Stimulus type {stim_type} not implemented.")
    
    cache_dir = get_cache_directory()
    full_data_path = os.path.join(cache_dir, PATH_IN_CACHE, f"openretina_data_iter{iteration}.pkl")
    with open(full_data_path, "rb") as f:
        data = pickle.load(f)

    # standardize the responses
    data["train_response"] = (data["train_response"] - np.mean(data["train_response"])) / np.std(data["train_response"])
    data["test_response"] = (data["test_response"] - np.mean(data["test_response"])) / np.std(data["test_response"])

    iteration_response = ResponsesTrainTestSplit(
        train=data["train_response"],
        test=data["test_response"],
        stim_id=stim_type,
    )
    
    return {f"iter{iteration}": iteration_response}


    

        
