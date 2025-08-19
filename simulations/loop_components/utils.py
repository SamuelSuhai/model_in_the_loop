from functools import wraps
from time import perf_counter
import datetime
from typing import Callable, Any,Tuple, Dict
import numpy as np
import os

EVAL_FOLDER = "/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/data/evaluation"
ONLINE_EXPERIMENT_LOG_DIR = "/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/logs/online_experiments"

def time_it(func: Callable[..., Any]) -> Callable[..., Any]:

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        time_before = perf_counter()
        out = func (*args, **kwargs)
        time_after = perf_counter()
        elapsed = time_after - time_before
        print(f"Function {func.__name__} took {elapsed:.6f} seconds to execute.")
        log_message = f"Function {func.__name__} took {elapsed:.6f} seconds to execute.\n"
        with open(EVAL_FOLDER + "/function_timing_log.txt", "a") as log_file:
            log_file.write(log_message)
        return out
    return wrapper

def log(msg: str) -> None:
    day_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # add folder and file if they dont exists
    save_dir = os.path.join(ONLINE_EXPERIMENT_LOG_DIR, day_str)
    log_file_path = f"{ONLINE_EXPERIMENT_LOG_DIR}/{day_str}/log.txt"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # write start message
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Log started on {day_str}\n")
    
    # append message to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
        


# def extract_hoefling_format_data_from_db(dj_table_holder: DJTableHolder) -> Dict[str, Dict] | None:
#         """ 
#         Extracts all the data for the model from the database. 
#         """
        
#         # get the key for the iteration
#         key = (self.field_table - self).proj().fetch(as_dict=True)[0]
#         if self.ignore_iteration_as_primary_key:
#             key.pop("cond1") # remove iteration dependence because we want to get all stimuli

#         # get all iteration data in long format:
#         all_iter_data = self.get_query(key)

#         if len(all_iter_data) == 0:
#             return None


#         # Get the data and build the dictionary
#         session_data_dict = self.from_query_to_data_dict(all_iter_data)


#         # get the session name from the key
#         session_name = "online_session_" + self.get_field_key(
#                 session_data_dict["field"],
#                 session_data_dict["eye"],
#                 session_data_dict["date"],
#             )
#         if self.insert_real_data:
#             self.insert1({**key, 'session_name': session_name ,'session_data_dict': session_data_dict})
#         else:
#             self.insert1({**key, 'session_name': session_name, 'session_data_dict': "NOT UPLOADED"})
        
#         return {session_name:session_data_dict}