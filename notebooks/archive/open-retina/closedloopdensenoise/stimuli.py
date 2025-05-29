import pickle 
import os
from typing import Any, Dict, List, Tuple

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies
from openretina.utils.file_utils import get_cache_directory
from .constants import (PATH_IN_CACHE, STIMULUS_SHAPE)



def load_all_stimuli(
        stim_type: str = "closedloopdensenoise",
        iteration: int = 0,
        normalize_stimuli: bool = True,
):
    
    if stim_type != "closedloopdensenoise":
        raise NotImplementedError(f"Stimulus type {stim_type} not implemented.")

    cache_dir = get_cache_directory()
    full_data_path = os.path.join(cache_dir, PATH_IN_CACHE, f"openretina_data_iter{iteration}.pkl")
    with open(full_data_path, "rb") as f:
        data = pickle.load(f)
    
    train_video = data["train_stimulus"]
    test_video = data["test_stimulus"]

    # import numpy as np
    # train_video = np.repeat(train_video, 10, axis=1)
    # test_video = np.repeat(test_video, 10, axis=1)

    if normalize_stimuli:
        train_video, test_video, norm_dict = normalize_train_test_movies(train_video, test_video)
    else:
        norm_dict = {"norm_mean": None, "norm_std": None}

    iteration_stimulus =  MoviesTrainTestSplit(
                    train=train_video,
                    test=test_video,
                    stim_id=stim_type,
                    random_sequences=None,
                    norm_mean=norm_dict["norm_mean"],
                    norm_std=norm_dict["norm_std"],
                )
    
    return {f"iter{iteration}": iteration_stimulus}
