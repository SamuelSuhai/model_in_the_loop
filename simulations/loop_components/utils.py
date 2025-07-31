from functools import wraps
from time import perf_counter
from typing import Callable, Any,Tuple, Dict
import numpy as np

EVAL_FOLDER = "/gpfs01/euler/User/ssuhai/GitRepos/simulation_closed_loop/data/evaluation"

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