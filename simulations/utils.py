from functools import wraps
from time import perf_counter
from typing import Callable, Any

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