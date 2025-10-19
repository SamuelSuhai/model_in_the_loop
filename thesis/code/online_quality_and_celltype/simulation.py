import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import thesis.code.online_quality_and_celltype.utils as ut


@dataclass
class TimingConfig:
    T_total: float
    t_stim: float
    t_pipeline: float
    t_rest: float
    t_switch: float

@dataclass
class ModelConfig:
    target_type: int
    p_types: np.ndarray
    confusion: np.ndarray
    decision_threshold: int = 1
    rng_seed: Optional[int] = None

def draw_field_true_counts_multinomial(N_total: int, p_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(N_total, p_types)



def draw_field_true_counts_poisson(lambda_total: float, p_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lambdas = lambda_total * p_types
    return rng.poisson(lambdas)

def draw_online_pred_count_for_target(n_true: np.ndarray, confusion: np.ndarray, target_type: int, rng: np.random.Generator) -> int:
    """
    returnd online pipeline predictions (int) predicted nr of target cells.
    K: nr of cell types
    n_true: (K,) array of true counts of each type in the field
    confusion: (K, ?) confusion matrix where confusion[i,j] is the probability of Offline i, online j (row normalized) so p(online=j | offline=i)
    """
    K = n_true.shape[0]
    count = 0
    for i in range(K): # loop over celltypes
        # probabiiliy of perdicting target_type given true type i
        p_pred_target_given_true = confusion[i, target_type]

        # skip when no true cells of type i or p_hit is 0
        if n_true[i] > 0 and p_pred_target_given_true > 0:

            # draw the number of times  equal to the nr of cells in the field
            count += rng.binomial(n_true[i], p_pred_target_given_true)
    return int(count)

def field_yield_true_target(n_true: np.ndarray, target_type: int) -> int:
    # how many true target cells are in the field
    return int(n_true[target_type])

def run_online(timing: TimingConfig,
               model: ModelConfig,
               draw_field_fn: Callable[..., np.ndarray],
               draw_field_args: Tuple,
               rng: np.random.Generator) -> Dict[str, str |float]:
    t = 0.0
    yield_cells = 0
    fields_visited = 0
    false_positive_fields = 0

    while t + timing.t_stim + timing.t_pipeline + timing.t_switch <= timing.T_total:
        fields_visited += 1

        # generate true cell counts for field
        n_true = draw_field_fn(*draw_field_args, rng=rng)

        # get predicted
        pred_target = draw_online_pred_count_for_target(n_true, model.confusion, model.target_type, rng)

        # time spent in chrip/mb and pipeline
        t += (timing.t_stim + timing.t_pipeline)

        # still time and predicted desired nr of target cells
        if pred_target >= model.decision_threshold and (t + timing.t_rest) <= timing.T_total:

            # true nr of target cells in field
            true_yield = field_yield_true_target(n_true, model.target_type)
            yield_cells += true_yield


            if pred_target > 0 and true_yield == 0:
                false_positive_fields += 1

            # time spend with rest of field experiment
            t += timing.t_rest

        #  time the field switching takes
        if t + timing.t_switch > timing.T_total:
            break
        t += timing.t_switch

    return dict(strategy="online",
                yield_cells=yield_cells,
                fields_visited=fields_visited,
                false_positive_fields=false_positive_fields,
                time_used=t)

def run_offline(timing: TimingConfig,
                model: ModelConfig,
                draw_field_fn: Callable[..., np.ndarray],
                draw_field_args: Tuple,
                rng: np.random.Generator) -> Dict[str, float | str]:
    t = 0.0
    yield_cells = 0
    fields_visited = 0

    # Start a field only if you can fully complete stim + rest for it
    while t + timing.t_stim + timing.t_rest <= timing.T_total:
        fields_visited += 1

        n_true = draw_field_fn(*draw_field_args, rng=rng)
        yield_cells += field_yield_true_target(n_true, model.target_type)

        # Pay the full recording cost for this field
        t += (timing.t_stim + timing.t_rest)

        # If there’s time, switch to the next field; otherwise we’re done (final field)
        if t + timing.t_switch <= timing.T_total:
            t += timing.t_switch
        else:
            break

    return dict(strategy="offline",
                yield_cells=yield_cells,
                fields_visited=fields_visited,
                time_used=t)


def simulate(num_trials: int,
             timing: TimingConfig,
             model: ModelConfig,
             generator: str = "multinomial",
             N_total: Optional[int] = None,
             lambda_total: Optional[float] = None) -> pd.DataFrame:
    """
    num_trials: number of trials to simulate
    timing: TimingConfig object with timing parameters
    model: ModelConfig object with model parameters
    generator: "multinomial" or "poisson"
    N_total: total number of cells per field for multinomial generator

    """

    assert generator in ("multinomial", "poisson"), "generator must be 'multinomial' or 'poisson'"

    if generator == "multinomial":
        assert isinstance(N_total, int) and N_total >= 0, "Provide N_total >= 0 for multinomial generator"
        draw_field_fn = draw_field_true_counts_multinomial
        draw_field_args = (N_total, model.p_types)
    else:
        assert isinstance(lambda_total, (int, float)) and lambda_total >= 0, "Provide lambda_total >= 0 for poisson generator"
        draw_field_fn = draw_field_true_counts_poisson
        draw_field_args = (float(lambda_total), model.p_types)

    rng = np.random.default_rng(model.rng_seed)
    records = []
    for r in range(num_trials):
        trial_seed = None if model.rng_seed is None else (model.rng_seed + r + 1)
        trial_rng = np.random.default_rng(trial_seed)
        online = run_online(timing, model, draw_field_fn, draw_field_args, trial_rng)
        offline = run_offline(timing, model, draw_field_fn, draw_field_args, trial_rng)
        records.append(online); records.append(offline)
    return pd.DataFrame.from_records(records)


def summarize_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_strategy = df.groupby("strategy").agg(
        mean_yield=("yield_cells", "mean"),
        median_yield=("yield_cells", "median"),
        std_yield=("yield_cells", "std"),
        mean_fields=("fields_visited", "mean"),
        mean_time=("time_used", "mean")
    ).reset_index()
    by_strategy["cells_per_time"] = by_strategy["mean_yield"] / by_strategy["mean_time"].replace(0, np.nan)
    try:
        mean_yield_online = by_strategy.loc[by_strategy["strategy"] == "online", "mean_yield"].values[0]
        mean_yield_off   = by_strategy.loc[by_strategy["strategy"] == "offline", "mean_yield"].values[0]
        mean_time_online = by_strategy.loc[by_strategy["strategy"] == "online", "mean_time"].values[0]
        mean_time_off    = by_strategy.loc[by_strategy["strategy"] == "offline", "mean_time"].values[0]
        delta = pd.DataFrame([{
            "metric": "mean_yield_gain (online - offline)",
            "value": mean_yield_online - mean_yield_off
        }, {
            "metric": "cells_per_time_online",
            "value": mean_yield_online / mean_time_online if mean_time_online > 0 else np.nan
        }, {
            "metric": "cells_per_time_offline",
            "value": mean_yield_off / mean_time_off if mean_time_off > 0 else np.nan
        },{
            "metric":  "percentage_gain",
            "value": 100.0 * (mean_yield_online - mean_yield_off) / mean_yield_off if mean_yield_off > 0 else np.nan
        }])
    except Exception:
        delta = pd.DataFrame([{"metric": "mean_yield_gain (online - offline)", "value": np.nan},
                              {"metric": "cells_per_time_online", "value": np.nan},
                              {"metric": "cells_per_time_offline", "value": np.nan}])
    return by_strategy, delta



def get_all_sim_data(offline2online_celltype_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

    df_prepared, applied_thresholds = ut.prepare_celltype_data(
    offline2online_celltype_df,
    offline_col='offline_cell_type',
    online_col='online_cell_type',
    max_type=32,
    nan_strategy='group',
)

    # Step 2: Create the confusion matrices
    confusion_counts, confusion_probs = ut.create_confusion_matrices(
        df_prepared,
        'offline_cell_type' + '_grouped',
        'online_cell_type' + '_grouped',
        max_type =32
    )
    counts = confusion_counts.to_numpy()

    # probability of finding celltype in offline analysis
    # sum over columns ie marginalize over online cell types
    prob_type = counts.sum(axis=1) / counts.sum()

    C = confusion_probs.to_numpy()

    return C, prob_type

def wrapper_sim(target, prob_type, C,stim_min =4, rest_min=25, switch_min=2, pipeline_min=1):
    # Problem setup
    K = len(prob_type)
    p_types = prob_type

    if isinstance(target,int):
        target_list = [target]
    else:
        target_list = target




    timing = TimingConfig(
        T_total= 7 *(stim_min + rest_min) * 60,   #  Time for 7 fields
        t_stim= stim_min * 60 , # 4 min
        t_pipeline= pipeline_min *60, # 1 min
        t_rest=rest_min * 60, # rest of stimuli (25 min)
        t_switch= switch_min * 60# time to select new field (2 min)
    )
    model = ModelConfig(
        target_type=None,
        p_types=p_types,
        confusion=C,
        decision_threshold=1,
        rng_seed=42
    )

    type_results = []

    for _target in target_list:
        model.target_type = _target

        # Run simulation (multinomial with fixed cells-per-field)
        df = simulate(
            num_trials=1000,
            timing=timing,
            model=model,
            generator="multinomial",
            N_total=100
        )



        by_strategy = df.groupby("strategy").agg(
            mean_yield=("yield_cells", "mean"),
            median_yield=("yield_cells", "median"),
            std_yield=("yield_cells", "std"),
            mean_fields=("fields_visited", "mean"),
            mean_time=("time_used", "mean")
        ).reset_index()
        
        # df["precentage_gain"] = df.apply(lambda row: 100 * row[]

        online_yield = by_strategy.loc[by_strategy["strategy"] == "online", "mean_yield"].item()
        offline_yield = by_strategy.loc[by_strategy["strategy"] == "offline", "mean_yield"].item()
        percentage_gain = 100 *  online_yield / offline_yield - 100 if offline_yield > 0 else np.nan
        
        type_results.append(
            {
                "target_type_idx": _target,
                "online_yield": online_yield,
                "offline_yield": offline_yield,
                "percentage_gain": percentage_gain,
                "target_fraction": p_types[_target],
                "target_hit_prob": C[_target,_target]
            }


        )
    df = pd.DataFrame(type_results)

    return df