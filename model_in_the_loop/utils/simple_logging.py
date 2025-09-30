from functools import wraps
from time import perf_counter
import datetime
from typing import Callable, Any,Tuple, Dict
import numpy as np
import os

def time_it(func: Callable[..., Any],eval_folder) -> Callable[..., Any]:

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        time_before = perf_counter()
        out = func (*args, **kwargs)
        time_after = perf_counter()
        elapsed = time_after - time_before
        print(f"Function {func.__name__} took {elapsed:.6f} seconds to execute.")
        log_message = f"Function {func.__name__} took {elapsed:.6f} seconds to execute.\n"
        with open(eval_folder + "/function_timing_log.txt", "a") as log_file:
            log_file.write(log_message)
        return out
    return wrapper

def log(msg: str,online_experiment_dir = None) -> None:
    
    if online_experiment_dir is None:
        repo_dir = os.environ.get("MITL_REPO_DIRECTORY",os.getcwd())
        online_experiment_dir = os.path.join(repo_dir,"model_in_the_loop/outputs/logs")
    
    day_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # add folder and file if they dont exists
    save_dir = os.path.join(online_experiment_dir, day_str)
    log_file_path = f"{online_experiment_dir}/{day_str}/log.txt"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # write start message
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Log started on {day_str}\n")
    
    # append message to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
        
