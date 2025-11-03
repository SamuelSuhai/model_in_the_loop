import os
from collections.abc import Iterable
from numbers import Integral
from thesis.code.analysis_closed_loop_experiments.rf_mei_test.rf_mei_test_schema import *
from thesis.code.analysis_closed_loop_experiments.rf_mei_test.rf_mei_test_tables import CIRCLE_TYPES
import thesis.code.analysis_closed_loop_experiments.rf_mei_test.utils.plot_utils as pu


from thesis.code.plot.style import get_palette
import datajoint as dj
from omegaconf import DictConfig, ListConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

MULTIPROCESSING_THREADS = 1

USERINFO_MAP = {
    "FloSalmon":
    {
        'experimenter': 'Salmon',
        'data_dir': '/gpfs01/euler/data/Data/Suhai/thesis/dj/FloSalomon'
    },
    "FloDeja":
    {
        'experimenter': 'Deja',
        'data_dir': '/gpfs01/euler/data/Data/Suhai/thesis/dj/FloDeja'
    }
}

def populate_user(recordings ="FloDeja",clear=False):
    
    if clear:
        UserInfo().delete()
        return
    

    userinfo = {
    **USERINFO_MAP[recordings],
    'animal_loc': 1,
    'region_loc': 2,
    'field_loc': 3,
    'stimulus_loc': 4,
    'cond1_loc': 5,
    'cond2_loc': 6,}

    UserInfo().upload_user(userinfo)

    

    assert os.path.isdir(userinfo['data_dir'])



def populate_cascade_params(clear=False) -> None:

    if clear:
        CascadeTraceParams().delete()
        CascadeParams().delete()
        return
    
     # spike estimation

    CascadeTraceParams().add_default(stim_names=['mouse_cam','optstim'],skip_duplicates=True)
    CascadeParams().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel',skip_duplicates=True) #

def populate_experiment_field_presentation(clear=False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:
    """
    Upload metadata for the current iteration, including experiments, fields, and presentations.
    """


    if clear:
        ## delete these tables
        Experiment().delete(safemode=safemode)
        OpticDisk().delete(safemode=safemode)
        Field().delete(safemode=safemode)
        RelativeFieldLocation().delete(safemode=safemode)
        Presentation().delete(safemode=safemode)
        return
    
    Experiment().rescan_filesystem(verboselvl=3)
    
    OpticDisk().populate(processes=processes, display_progress=display_progress)



    Field().rescan_filesystem(verboselvl=3)
    RelativeFieldLocation().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)
    Presentation().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)

def add_rois_and_traces(clear=False,scan_for_existing = True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:
    if clear:
        Roi().delete()
        Traces().delete()
        PreprocessTraces().delete()
        return
    if scan_for_existing:
        RoiMask().rescan_filesystem(verboselvl=3)
        
    
    Roi().populate(processes=processes, display_progress=display_progress)
    Traces().populate(processes=processes, display_progress=display_progress)
    PreprocessTraces().populate(processes=processes, display_progress=display_progress)


def add_params(table_parameters, recordings = "FloDeja",clear=False) -> None:
    if clear:
        RawDataParams().delete()
        PreprocessParams().delete()
        ClassifierTrainingData().delete()
        ClassifierMethod().delete()
        Classifier().delete()
        DNoiseTraceParams().delete()
        STAParams().delete()
        SplitRFParams().delete()
        CascadeTraceParams().delete()
        CascadeParams().delete()
        return
    
    RawDataParams().add_default()
    
    RawDataParams().update1(dict(
            experimenter=USERINFO_MAP[recordings]['experimenter'],
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
    preprocess_params =table_parameters.get("PreprocessParams", {})
    
    # add preprocess params for own stimuli
    preprocess_params[0]["stim_names"].extend(['circle'])
    preprocess_params[1]["stim_names"].extend(['optstim'])
    
    



    if isinstance(preprocess_params, ListConfig):
        for params in preprocess_params:
            PreprocessParams().add_default(**params,skip_duplicates=True)
    elif isinstance(preprocess_params, DictConfig):
        PreprocessParams().add_default(**preprocess_params,skip_duplicates=True)
    else:
        raise ValueError(f"Expected preprocess_params to be DictConfig or ListConfig, got {type(preprocess_params)}")
    print("preprocessing params:\n",preprocess_params)

    

    # Celltype assignment
    ClassifierTrainingData().add_default(skip_duplicates=True)
    ClassifierMethod().add_default(skip_duplicates=True)
    Classifier().populate()

    # rf estimation 
    dense_noise_parmas = table_parameters.get("DNoiseTraceParams", {})
    DNoiseTraceParams().add_default(**dense_noise_parmas)


    STAParams().add_default()
    SplitRFParams().add_default()



    # spike estimation
    CascadeTraceParams().add_default(stim_names=['mouse_cam','optstim'])
    CascadeParams().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel')


def populate_cascade_tabs(clear=False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:

    if clear:
        CascadeTraces().delete(safemode=safemode)
        CascadeSpikes().delete(safemode=safemode)
    
        return
    CascadeTraces().populate(processes=processes, display_progress=display_progress)
    CascadeSpikes().populate(processes=processes, display_progress=display_progress)

def populate_DN(clear =False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:
    if clear:
        DNoiseTrace().delete(safemode=safemode)
        STA().delete(safemode=safemode)
        SplitRF().delete(safemode=safemode)
        FitGauss2DRF().delete(safemode=safemode)
        return
    
    DNoiseTrace().populate(processes=processes, display_progress=display_progress)
    STA().populate(processes=processes, display_progress=display_progress)
    SplitRF().populate(processes=processes, display_progress=display_progress)
    FitGauss2DRF().populate(processes=processes, display_progress=display_progress)


def populate_averages(clear=False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:
    if clear:
        Averages().delete(safemode=safemode)
        return
    Averages().populate(processes=processes, display_progress=display_progress)



def populate_celltype_assignment(clear=False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True) -> None:
    if clear:
        ChirpQI().delete(safemode=safemode)
        OsDsIndexes().delete(safemode=safemode)
        Baden16Traces().delete(safemode=safemode)
        CelltypeAssignment().delete(safemode=safemode)
        return
    ChirpQI().populate(processes=processes, display_progress=display_progress)
    OsDsIndexes().populate(processes=processes, display_progress=display_progress)
    Baden16Traces().populate(processes=processes, display_progress=display_progress)
    CelltypeAssignment().populate(processes=processes, display_progress=display_progress)

def add_all_stimuli(table_parameters) -> None:

    import h5py

    with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
        noise_stimulus = f['stimulusarray'][:].T.astype(int)
    noise_stimulus = noise_stimulus*2-1
    
    Stimulus().add_nostim(skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
    Stimulus().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)

    
    Stimulus().add_noise(**table_parameters.Stimulus.noise, stim_trace=noise_stimulus)

        
    Stimulus().add_movingbar(skip_duplicates=True)
    
    Stimulus().add_stimulus(
        stim_name="mouse_cam", 
        alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
        stim_family="natural", 
        framerate=30.0,  
        stim_path="",
        ntrigger_rep=123,
        unique_alias=True,
        skip_duplicates=True,
    )


def add_own_stimuli(clear = False) -> None:
    if clear:
        Stimulus().delete()
        return
    
    Stimulus().add_stimulus(stim_name='circle', 
                            alias="rf_rf1_rf2_rf3_rf4_rf5", 
                            isrepeated=True, 
                            ntrigger_rep=1,
                            trial_info=None, 
                            stim_dict={
                                "size_large": 100,
                                "size_small": 50,
                            },
                            skip_duplicates=True)
    Stimulus().add_stimulus(stim_name='optstim', 
                            alias="mei_mei1_mei2_mei3_mei4_mei5", 
                            isrepeated=True, 
                            ntrigger_rep=1,
                            trial_info=None,
                            skip_duplicates=True)
        


def load_dj_config(config_file: str,schema_name: str) -> None:
    # Load configuration for user
    dj.config.load(config_file)

    dj.config['schema_name'] = schema_name
    dj.config['enable_python_native_blobs'] = True
    dj.config["display.limit"] = 20
    

    print("schema_name:", dj.config['schema_name'])
    dj.conn()


def get_field_keys_with_stim(stim_name):
    if isinstance(stim_name,str):
        stim_query = f"stim_name = '{stim_name}'"
    all_stim_fields = (Field().proj() & (Presentation().proj() & stim_query)).fetch(as_dict=True)
    return all_stim_fields



def prepclassifier_and_activate(output_folder):
    
    
    from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
    prepare_dj_config_rgc_classifier(output_folder)


    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=schema, create_schema=True, create_tables=True)

    


def plot_roi_pp_trace_for_stim(roi_id,stim_name,field_key,cond2_restriction = {}):
    (PreprocessTraces() & field_key & dict(roi_id=roi_id) & dict(stim_name=stim_name) & cond2_restriction).plot1()
    fig = plt.gcf()
    return fig

def plot_all_rois_pp_trace_for_stim(
                                    stim_name,
                                    field_key,
                                    cond2_restriction = {},
                                    pp_trace_tab = PreprocessTraces,
                                    stimulus_presentation_info_table = StimulusPresentationInfo,
                                    offline2online_roi_id_tab = Offline2OnlineRoiId):
    
    restricted_pp_traces = (pp_trace_tab() & offline2online_roi_id_tab() & stimulus_presentation_info_table() & field_key & dict(stim_name=stim_name) & cond2_restriction)
    all_rois = restricted_pp_traces.fetch("roi_id")
    for roi in all_rois:
        plot_roi_pp_trace_for_stim(roi,stim_name,field_key,cond2_restriction)
    



def populate_snippets(clear=False,processes=MULTIPROCESSING_THREADS, display_progress=True):
    if clear:
        Snippets().delete()
        return
    
    Snippets().populate(processes=processes, display_progress=display_progress)


def populate_stimulus_presentation_info(clear = False):
    if clear:
        StimulusPresentationInfo().delete()
        return
    
    StimulusPresentationInfo().populate()

def populate_single_snippets(clear=False,safemode=True,processes=MULTIPROCESSING_THREADS, display_progress=True):
    if clear:
        SingleSnippet().delete(safemode=safemode)
        return
    SingleSnippet().populate(processes=processes, display_progress=display_progress)

def load_rf_mask_saved_in_dict(rf_mask_file):

    import pickle
    with open(rf_mask_file, 'rb') as f:
        rf_mask_dict = pickle.load(f)
    return rf_mask_dict

def highlight_roi_in_mask(roi_mask,roi_id,ax = None,alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots()
    other_rois = (roi_mask !=roi_id) & (roi_mask != 1)
    mask = (roi_mask==-roi_id).astype(float) * 100.0 + (other_rois).astype(float) * 50
    
    ax.imshow(np.rot90(mask,k=1),cmap='Reds',alpha=alpha)
    return ax



def fetch_trace_trigger_triggerinfo(preprocess_traces_table,
                                    presentation_table,
                                    stimulus_presentation_info_table,
                                    online2offline_roi_id_table,
                                    field_roi_cond2_key: Dict[str,Any]):

    # join tables 
    full_query = (preprocess_traces_table * presentation_table * stimulus_presentation_info_table * online2offline_roi_id_table) & field_roi_cond2_key
    assert len(full_query) == 1, f"Expected one entry for {field_roi_cond2_key}, got {len(full_query)}"

    pp_trace_t0, pp_trace_dt, pp_trace, _ = full_query.fetch1(
        "pp_trace_t0", "pp_trace_dt", "pp_trace", "smoothed_trace")
    triggertimes = full_query.fetch1("triggertimes")

    pp_trace_times = np.arange(pp_trace.size) * pp_trace_dt + pp_trace_t0

    triggeridx2positions = full_query.fetch1("triggeridx2positions")
    triggeridx2online_roi_id = full_query.fetch1("triggeridx2online_roi_id")
    triggeridx2stim_type = full_query.fetch1("triggeridx2stim_type")
    true_online_roi_id = full_query.fetch1("true_online_roi_id")
    return pp_trace_times,pp_trace,triggertimes,(triggeridx2positions,triggeridx2online_roi_id,triggeridx2stim_type),true_online_roi_id


def get_field_roi_cond2_key(offline2online_roi_id_table,field_key,roi_id,con2_value = "control"):
    key = (offline2online_roi_id_table & field_key & dict(roi_id=roi_id)).fetch1().copy()
    key["cond2"] = con2_value
    return key


def get_trigger2rf_center_dist(triggeridx2positions,roi_position):
    triggeridx2rf_center_dist = np.linalg.norm(np.array([triggeridx2positions]).squeeze() - roi_position,axis = 1)
    return triggeridx2rf_center_dist

def get_roi_position(triggeridx2positions,triggeridx2online_roi_id,true_online_roi_id):
    idx = np.where(np.array(triggeridx2online_roi_id) == true_online_roi_id)[0][0]
    roi_position = triggeridx2positions[idx]
    return roi_position
    




def fetch_and_plot_trace_trigger_triggerinfo(preprocess_traces_table,
                                             presentation_table,
                                             stimulus_presentation_info_table,
                                             offline2online_roi_id_table, 
                                             key,
                                             ax = None,
                                             stim_type = None,
                                             add_dist_text = True,
                                             ):



    pp_trace_times,pp_trace,triggertimes,\
        (triggeridx2positions,triggeridx2online_roi_id,triggeridx2stim_type),\
            true_online_roi_id = fetch_trace_trigger_triggerinfo(preprocess_traces_table,
                                                            presentation_table,
                                                            stimulus_presentation_info_table,
                                                            offline2online_roi_id_table,
                                                            key)
    
    # get the roi_positoi
    roi_position = get_roi_position(triggeridx2positions,triggeridx2online_roi_id,true_online_roi_id)
    
    triggeridx2rf_center_dist = get_trigger2rf_center_dist(triggeridx2positions,roi_position)
    triggeridx2hilightalpha =  triggeridx2rf_center_dist / np.max(triggeridx2rf_center_dist)
    if  add_dist_text:
        triggeridx2txt = [str(pos) +"\ndist:" + str(int(dist)) for pos,dist in zip(triggeridx2positions,triggeridx2rf_center_dist)]
    else:
        triggeridx2txt = None
    
    
    if ax is None:
        fig, ax = plt.subplots(figsize =(10, 5))
    
    # restrict to elypse type
    if stim_type is not None:
        assert stim_type in CIRCLE_TYPES, f"stim_type {stim_type} not in {CIRCLE_TYPES}"
        elypse_triggeridx = [i for i, t in enumerate(triggeridx2stim_type) if t == stim_type]
        first_trigger = triggertimes[elypse_triggeridx[0]]
        last_trigger = triggertimes[elypse_triggeridx[-1]]
        ax.set_xlim(first_trigger - 1, last_trigger + 2)

        # filter
    
    ax = pu.plot_trace_trigger_triggerinfo(pp_trace_times,
                                        pp_trace,
                                        triggertimes,
                                        triggeridx2hilightalpha,
                                        ax,
                                        triggeridx2txt)
    
    # set legend lower right
    ax.legend(loc='lower right')

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fluorescence [a.u.]")
    ax.set_title(f"Roi Id {key['roi_id']} circle type {stim_type}")

    return ax


def wrapper_fetch_and_plot_trace_trigger_triggerinfo_all_rois(
        offine2online_roi_id_table,
        pp_trace_table,
        presentation_table,
        stimulus_presentation_info_tablele,
        field_cond2_key,
):
    offline_roi_ids = (offine2online_roi_id_table() & field_cond2_key).fetch("roi_id")
    nrow = len(offline_roi_ids)
    ncol = 4
    fig,ax = plt.subplots(nrow,ncol,figsize=(25,5*nrow),)
 
    for i,roi_id in enumerate(offline_roi_ids,):
        field_roi_cond2_key = {** field_cond2_key,"roi_id": roi_id}
        for i_circle, stim_type in enumerate(CIRCLE_TYPES):
            fetch_and_plot_trace_trigger_triggerinfo(
                preprocess_traces_table = pp_trace_table(),
                presentation_table = presentation_table(),
                stimulus_presentation_info_table  = stimulus_presentation_info_tablele(),
                offline2online_roi_id_table = offine2online_roi_id_table(), 
                key = field_roi_cond2_key,
                stim_type=stim_type,
                ax = ax[i,i_circle],
            )

def set_xaxis_lim_to_stim_type_interval(ax: plt.Axes,
                               triggeridx2stim_type: List[str],
                               triggertimes: np.ndarray,
                               stim_type,
                               pad_left = 0.1,
                               pad_ritht = 0.1) -> plt.Axes:
    elypse_triggeridx = [i for i, t in enumerate(triggeridx2stim_type) if t == stim_type]
    first_trigger = triggertimes[elypse_triggeridx[0]]
    last_trigger = triggertimes[elypse_triggeridx[-1]]
    ax.set_xlim(first_trigger - pad_left, last_trigger + pad_ritht)
    return ax



def plot_single_roi_trace_trigger_bg_stim(
        offline2online_roi_id_table,
        pp_trace_table,
        presentation_table,
        stimulus_presentation_info_table,
        field_cond2_key,
        stim_type,
        roi_id,
        stim_onset_delay=0.95,
        ax = None,):
    """
    For one roi:
    - fetch trace, trigger times, trigger info
    - plot trace with background and stimulus periods highlighted
    """

    if ax is None:
        fig, ax = plt.subplots()


    key = {** field_cond2_key,"roi_id": roi_id}
    pp_trace_times,pp_trace,triggertimes,\
    (triggeridx2positions,triggeridx2online_roi_id,triggeridx2stim_type),\
        true_online_roi_id = fetch_trace_trigger_triggerinfo(pp_trace_table,
                                                        presentation_table,
                                                        stimulus_presentation_info_table,
                                                        offline2online_roi_id_table,
                                                        key)
    
    # get stimulus onset times 
    stim_onset_times = triggertimes + stim_onset_delay


    # restrict x axiis to elypse type
    assert stim_type in CIRCLE_TYPES, f"stim_type {stim_type} not in {CIRCLE_TYPES}"
    ax = set_xaxis_lim_to_stim_type_interval(
        ax,
        triggeridx2stim_type,
        triggertimes,
        stim_type,
        pad_left = 0.15,
        pad_ritht = -0.1,
    )

    
    ax = pu.plot_trace_trigger_bg_stim(trace_times = pp_trace_times,
                                        trace= pp_trace,
                                        triggertimes = triggertimes,
                                        stim_onset_times= stim_onset_times,
                                        ax = ax,
                                        bg_color='gray',
                                        stim_color='yellow',
                                        bg_kwargs={"alpha":0.3},
                                        stim_kwargs={"alpha":0.3}
                                        )
    
    # remove spines
    sns.despine(ax=ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fluorescence [a.u.]")

    
    # legend
    ax = pu.add_trigger_bg_stim_legend(ax)


    return ax



def fetch_rois_snippets_df(field_key,
                           roi_id,
                           stim_name ='circle',
                           cond2 =None):
    """
    
    """


    if isinstance(roi_id,Iterable) and not isinstance(roi_id,(str,bytes)):
        query_str_roi = f"roi_id in {tuple(map(int,roi_id))}"
    else: 
        query_str_roi = f"roi_id='{roi_id}'" 

    if cond2 is not None:
        assert isinstance(cond2,(str,int)), f"Expected cond2 to be str, int got {type(cond2)}"
        query_str_roi = query_str_roi + f" AND cond2 = '{cond2}'"
    
    stim_query = f"stim_name = '{stim_name}'"

    # query fo
    query = (SingleSnippet() * Offline2OnlineRoiId() * StimulusPresentationInfo() * OnlineInferredRFPosition())
    
    rois_snippets_df = pd.DataFrame((query & field_key & query_str_roi & stim_query).fetch(as_dict=True))
    rois_snippets_df = rois_snippets_df[["roi_id",
                                   "true_online_roi_id",
                                   "stimulus_type",
                                   "single_snippet",
                                   "x_pos","y_pos",
                                   "single_snippet_t0",
                                   "online_roi_id",
                                   "x_rf","y_rf",
                                   "is_first_pres_of_stimulus",
                                   "cond2"]]

    return rois_snippets_df
    


def add_distance_to_snippet_df(snippet_df):

    def _calc_pres_dist_from_rf_position(x_rf,y_rf,
                                         x_pos,y_pos):
        return np.linalg.norm(np.array([x_rf,y_rf]) - np.array( [x_pos,y_pos]))

    snippet_df["distance"] = snippet_df.apply(lambda row: _calc_pres_dist_from_rf_position(row["x_rf"],row["y_rf"], row["x_pos"],row["y_pos"]),axis = 1)
    return snippet_df

def drop_first_presentation_of_stim_type(single_snippet_df, verbose=True):
    """
    Drops is_first_pres_of_stimulus == 1 rows.
    """
    to_remove = single_snippet_df["is_first_pres_of_stimulus"] == 1
    if verbose:
        print(f"Dropping {to_remove.sum()} rows with is_first_pres_of_stimulus == 1")
    
    
    single_snippet_df = single_snippet_df[single_snippet_df["is_first_pres_of_stimulus"] == 0]
    return single_snippet_df

def add_bsl_corrected_snippets(single_snippet_df,
                               method= "first",
                               first_n_frames = 60):

    if method == "first":
        func = lambda x: x - x[0]
    elif method == "min":
        func = lambda x: x - np.min(x[:first_n_frames])
    elif method == "median":
        func = lambda x: x - np.median(x[:first_n_frames])
    else:
        raise ValueError

    single_snippet_df[f"single_snippet_bsl_corrected_{method}"] = single_snippet_df["single_snippet"].apply(func)
    return single_snippet_df

def fetch_all_fields_with_stm(stim_name):

    return (Field().proj() & (Presentation().proj() & dict(stim_name=stim_name))).fetch(as_dict=True)


def average_df_over_colvalues(
        df: pd.DataFrame,
        cols_to_gb: List[str],
        cols_to_average: List[str],
        verbose = True,
    ) -> pd.DataFrame:
    """
    For each unique combination of values in cols_to_gb, average the values in cols_to_average"""

    # groupby all columns except cols_to_average_over and cols_to_average
    
    print(f"Grouping by columns: {cols_to_gb}, reducing df cols over {set(df.columns) - set(cols_to_gb)}")
    for c in cols_to_gb:
        assert not isinstance(df[c].iloc[0], np.ndarray), f"Column {c} is not suitable for grouping, contains ndarray."

    grouped_df = df.groupby(cols_to_gb,as_index=False)
    
    if verbose:
        print(f"Number of unique groups: {len(grouped_df)}")
    
    agg_func = lambda x: np.mean(np.stack(x.to_list()),axis=0)

    average_df = grouped_df[cols_to_average].agg(
        func = agg_func
    )

    if verbose:
        print(f"Averaged df over columns: {cols_to_average}, resulting df has {len(average_df)} rows. Differece in rows: {len(df) - len(average_df)}")
    return average_df



def get_mean_snippet_df(field_key,
                        roi_id,
                        bsl_correction_method = "first",
                        stim_name ='circle',
                        cond2_value=None,
                        drop_first_presentation=True,
                        verbose=True):
    rois_snippets_df = fetch_rois_snippets_df(field_key,
                                              roi_id=roi_id,
                                                stim_name =stim_name,
                                              cond2=cond2_value)


    if bsl_correction_method is not None:
        rois_snippets_df = add_bsl_corrected_snippets(rois_snippets_df,method=bsl_correction_method)
        snippet_col_name = f"single_snippet_bsl_corrected_{bsl_correction_method}"
    else:
        snippet_col_name = "single_snippet"
    rois_snippets_df = add_distance_to_snippet_df(rois_snippets_df)
    
    
    if drop_first_presentation:
        rois_snippets_df = drop_first_presentation_of_stim_type(rois_snippets_df)

    # keep necesary cols
    rois_snippets_df = rois_snippets_df[[c for c in rois_snippets_df.columns if "single_snippet" in c or c in ["roi_id",
                                        "stimulus_type",
                                        "online_roi_id",
                                        "distance",
                                        "cond2"]]]
    if verbose:
        print(f"Filtered rois_snippets_df to columns: {rois_snippets_df.columns.tolist()}")

    # average away
    cols_iding_rows = ["roi_id",
                       "stimulus_type",
                       "distance"]
    n_row_before = len(rois_snippets_df)
    average_df = average_df_over_colvalues(rois_snippets_df,
                                           cols_to_keep=cols_iding_rows,
                                           cols_to_average=[snippet_col_name])
    assert len(average_df) == n_row_before / rois_snippets_df["cond2"].nunique(), f"Unexpected number of rows after averaging.\n Before: {n_row_before}, after: {len(average_df)}, expected: {n_row_before / rois_snippets_df["cond2"].nunique()}"

    # sort by distance
    sorted_avg_df = average_df.sort_values("distance",ascending=True)

    print(f"output df col names: {sorted_avg_df.columns.tolist()}")
    return sorted_avg_df, snippet_col_name



def get_str_from_field_key(field_key,only_these=["exp_num","field"]):
    return "_".join([f"{k}-{v}" for k,v in field_key.items() if k in only_these])


def get_snippet_trace_data(df,snippet_col_name,stim_name):


    stim_df = df[df["stimulus_type"] == stim_name]
    snippet_trace_list = stim_df[snippet_col_name].to_list()
    single_snippet_dt = 1/60
    snippet_presentation_distances = stim_df["distance"].to_list()
    return snippet_trace_list,single_snippet_dt, snippet_presentation_distances



def polarity_func(stim_max_loc_df):
    assert ["is_second_half_peak","stimulus_type"] == list(stim_max_loc_df.columns)

    peak_on_small_second_half = stim_max_loc_df[stim_max_loc_df["stimulus_type"] == "on_small"]["is_second_half_peak"].iloc[0]
    peak_on_big_second_half = stim_max_loc_df[stim_max_loc_df["stimulus_type"] == "on_big"]["is_second_half_peak"].iloc[0]

    peak_off_small_second_half = stim_max_loc_df[stim_max_loc_df["stimulus_type"] == "off_small"]["is_second_half_peak"].iloc[0]
    peak_off_big_second_half = stim_max_loc_df[stim_max_loc_df["stimulus_type"] == "off_big"]["is_second_half_peak"].iloc[0]

    if peak_on_small_second_half and peak_on_big_second_half:
        return "on"
    elif peak_off_small_second_half and peak_off_big_second_half:
        return "off"
    else:
        return "other"

def get_polariy_of_cell(df,snippet_col_name,polarity_func):

    new_df = df.copy()

    # apply new column where entries are max of snippet_col_name
    new_df["max_in_snippet"] = new_df[snippet_col_name].apply(lambda x: np.max(x))
    new_df["max_snippet_frame"] = new_df[snippet_col_name].apply(lambda x: np.argmax(x))

    
    # group by roi_id, stimulus_type and  Get the frame of the max for the snippet with the highest peak
    idx_highest_snippet = new_df.groupby(["roi_id","stimulus_type"])["max_in_snippet"].idxmax()
    top_snippets_df = new_df.loc[idx_highest_snippet].reset_index(drop=True)


    # apply new column that says if peak in first or second half of snippet
    lengths = top_snippets_df[snippet_col_name].apply(len)
    assert lengths.nunique() == 1, "Snippet lengths vary across rows."
    L = lengths.iloc[0]
    top_snippets_df["is_second_half_peak"] =  top_snippets_df["max_snippet_frame"] > (L // 2)
    
    # group by roi_id and apply polarity func
    polarity_df = top_snippets_df.groupby("roi_id",as_index=False).apply(
        func = lambda subdf: pd.Series({
            "polarity": polarity_func(subdf[["is_second_half_peak","stimulus_type"]])
            })
    )

    return polarity_df





def fetch_celltype_df(field_key,
                      ):
    
    query = (CelltypeAssignment() & field_key)
    celltype_df = pd.DataFrame((query.proj(
        "roi_id","celltype"
    )).fetch(as_dict=True))
    celltype_df = celltype_df[["roi_id","celltype"]]
    # turn celltype int
    celltype_df["celltype"] = celltype_df["celltype"].astype(int)
    celltype_df["roi_id"] = celltype_df["roi_id"].astype(int)
    return celltype_df



def add_snippet_scalar_value(full_df,measure,snippet_col_name):
    stim_window_start_fr = len(full_df[snippet_col_name].iloc[0]) // 2

    if measure == "response_magnitude":
        full_df["response_magnitude"] = full_df[snippet_col_name].apply(lambda x: np.max(x[stim_window_start_fr:]) if np.abs(np.max(x[stim_window_start_fr:])) > np.abs(np.min(x[stim_window_start_fr:])) else np.min(x[stim_window_start_fr:]))
    
    elif measure == "response_mean":
        # as alternative: mean value during stim pres
        full_df["response_mean"] = full_df[snippet_col_name].apply(lambda x: np.mean(x[stim_window_start_fr:]))

    else:
        raise ValueError(f"Unknown measure {measure}")



def fetch_and_format_data_for_snippet_analysis(
        field_key: Dict[str,Any] | List[Dict[str,Any]],
        roi_id_list,
        bsl_correction_method = "median",

        polarity = "on",
        stimulus_type = "on_small",
        measure = "response_mean",
        drop_first_presentation = True,
        cond2_value = None,
        ) -> pd.DataFrame:
    

    if isinstance(field_key,dict):
        field_key = [field_key]
    
    print(f"No of fields passed: {len(field_key)}")

    if not isinstance(roi_id_list[0],(list,np.ndarray)):
        roi_id_list = [roi_id_list]
        assert len(field_key) == 1, "If roi_id_list is a single list of ints, field_key must be a single dict."
    nr_rois_per_field = [len(rois) for rois in roi_id_list]
    print(f"Nr of rois per field: {nr_rois_per_field}")

    full_df_list = []
    for fk,roi_ids in zip(field_key,roi_id_list,strict=True):
        # fetch and process df
        mean_df,snippet_col_name = get_mean_snippet_df(fk,
                                                       roi_ids,
                                                       bsl_correction_method=bsl_correction_method,
                                                       drop_first_presentation = drop_first_presentation,cond2_value=cond2_value)

        # add polarity of cell
        polarity_df = get_polariy_of_cell(mean_df,snippet_col_name,polarity_func)
        assert len(polarity_df) == len(np.unique(polarity_df["roi_id"])), f"Length of celltype_df {len(polarity_df)} does not match number of unique roi_ids in full_df {len(np.unique((polarity_df['roi_id'])))}"

        full_df = mean_df.merge(polarity_df,on="roi_id",how="left")

        # filter by polarity and stimulus type
        n_before_filtering = len(full_df)
        full_df = full_df[full_df["stimulus_type"] == stimulus_type]
        assert len(full_df) == n_before_filtering / 4, f"After filtering by polarity {polarity} and stimulus_type {stimulus_type}, expected number of rows to be quartered. Before: {n_before_filtering}, after: {len(full_df)}"
        
        # filter polarity 
        full_df = full_df[full_df["polarity"] == polarity]

        # add celltype info
        celltype_df = fetch_celltype_df(fk)
        celltype_df = celltype_df[celltype_df["roi_id"].isin(full_df["roi_id"])]
        assert len(celltype_df) == len(np.unique(full_df["roi_id"])), f"Length of celltype_df {len(celltype_df)} does not match number of unique roi_ids ({len(np.unique((full_df['roi_id'])))}"
        full_df = full_df.merge(celltype_df,on="roi_id",how="left")
        full_df_list.append(full_df)

    full_df = pd.concat(full_df_list)
    add_snippet_scalar_value(full_df,measure,snippet_col_name)


    return full_df










def wrapper_scatter_response_distance_celltype(
        field_key: Dict[str,Any] | List[Dict[str,Any]],
        roi_id_list: List[int] | List[List[int]],
        bsl_correction_method = "median",
        plot_kwargs = {},
        show_legend=False,
        polarity = "on",
        stimulus_type = "on_small",
        measure = "response_mean",
        drop_first_presentation = True,
        ax = None,
        cond2_value = None,
        ):
    

    full_df = fetch_and_format_data_for_snippet_analysis(field_key,
                                                         roi_id_list,
                                                         bsl_correction_method=bsl_correction_method,
                                                            polarity=polarity,
                                                            stimulus_type=stimulus_type,
                                                            measure=measure,
                                                            drop_first_presentation=drop_first_presentation,
                                                            cond2_value=cond2_value,
                                                         )
    


    if ax is None:
        fig, ax = plt.subplots()
    

    scatter_kwargs = plot_kwargs.get("scatter_kwargs",{})
    single_group_fit_kwargs = plot_kwargs.get("single_group_fit_kwargs",{})
    
    overall_fit_kwargs = plot_kwargs.get("overall_fit_kwargs",{})
    if measure == "response_mean":
        ylabel = "Mean fluorescence change [a.u.]" 
    elif measure == "response_magnitude":
        ylabel = "Peak fluorescence change [a.u.]"
    else:
        ylabel = measure

    celltypes = list(map(int,np.unique(full_df["celltype"])))

    if plot_kwargs.get("use_celltype_cmap",False) is True:
        color_map = pu.get_celltype_alpha_cmap(celltypes=celltypes)
    else:
        palette = sns.color_palette("tab10", n_colors=len(celltypes))
        color_map = {celltype: palette[i] for i,celltype in enumerate(celltypes)}
    
    ax = pu.plot_mulit_group_scatter_fits(full_df=full_df,
                                       x = "distance",
                                        y=measure,
                                        ax=ax,
                                        hue="celltype",
                                        xlabel="Distance to RF center [μm]",
                                        color_map=color_map,
                                        ylabel=ylabel,
                                        scatter_kwargs=scatter_kwargs,
                                        single_group_fit_kwargs=single_group_fit_kwargs,
                                        overall_fit_kwargs=overall_fit_kwargs,
                                        show_legend=show_legend,
                                  )
    import matplotlib.lines as mlines

    # add thin line proxy artist for legend
    proxy_single_reg= mlines.Line2D([], [], color='grey', linestyle='-', linewidth=0.3, label="celltype fit")
    handles, labels = ax.get_legend_handles_labels()
    
    # combine existing handles with new proxy artist
    handles.append(proxy_single_reg)
    labels.append(proxy_single_reg.get_label())
    
    # recreate legend with all handles
    ax.legend(handles=handles,labels=labels,ncols=2)


    return ax






def wrapper_scatter_response_distance(
        field_key,
        roi_id_list,
        bsl_correction_method = "median",
        plot_kwargs = {},
        show_legend=False,
        polarity = "on",
        stimulus_type = "on_small",
        measure = "response_magnitude",
        drop_first_presentation = True,
        add_celltype_info = False,
        ax = None,
        cond2_value = None,
    ):

    # fetch and process df
    mean_df,snippet_col_name = get_mean_snippet_df(field_key,
                                                   roi_id_list,
                                                   bsl_correction_method=bsl_correction_method,
                                                   drop_first_presentation = drop_first_presentation,cond2_value=cond2_value)

    # add polarity of cell
    polarity_df = get_polariy_of_cell(mean_df,snippet_col_name,polarity_func)
    full_df = mean_df.merge(polarity_df,on="roi_id",how="left")

    # filter by polarity and stimulus type
    full_df = full_df[(full_df["polarity"] == polarity) & (full_df["stimulus_type"] == stimulus_type)]

    # add celltype info
    if add_celltype_info:
        celltype_df = fetch_celltype_df(field_key)
        full_df = full_df.merge(celltype_df,on="roi_id",how="left")

    # get one scalar value per row (snippet): 
    # the response increase from baseline i.e. the most extreme value of the snippet during stim pres
    add_snippet_scalar_value(full_df,measure,snippet_col_name)


    palette = sns.color_palette("tab20", n_colors=len(roi_id_list))
    color_map = {roi_id: palette[i] for i,roi_id in enumerate(roi_id_list)}
    ax = pu.plot_mulit_group_scatter_fits(full_df=full_df,
                                    x = "distance",
                                    y=measure,

                                    ax=ax,
                                    hue="roi_id",
                                    xlabel="Distance to RF center [μm]",
                                    ylabel="Mean fluorescence change [a.u.]",

                                  color_map = color_map,
                                  scatter_kwargs=plot_kwargs.get("scatter_kwargs",{}),
                                  single_group_fit_kwargs=plot_kwargs.get("single_group_fit_kwargs",{}),
                                  overall_fit_kwargs=plot_kwargs.get("overall_fit_kwargs",{}),
                                  show_legend=show_legend,
                                  )

    return ax



def wrapper_plot_one_roi_ordered_snippets(
        field_key,
        roi_id,
        bsl_correction_method = "first",
        stim_name = "off_big",
        plot_kwargs = {},
        snippet_vline = True,
        time_buffer_between_snippets = 0,
        show_legend=False,
        highlight_bg_times = (0,0.95),
        highlighted_stim_times = (0.95,2),
        ax = None,
        cond2_value = None,
    ):  


    # fetch and process df
    mean_df,snippet_col_name = get_mean_snippet_df(field_key,
                                                   roi_id,
                                                   bsl_correction_method=bsl_correction_method,
                                                   cond2_value=cond2_value)


    snippet_trace_list,single_snippet_dt ,\
        snippet_presentation_distances= get_snippet_trace_data(mean_df,
                                                                snippet_col_name,
                                                                stim_name=stim_name)

    # get distance related x tick labels
    x_tick_lables = list(map(lambda dist: f"{dist:.0f}",snippet_presentation_distances))



    # plot snippets by distance             
    pu.plot_ordered_snippets(snippet_trace_list = snippet_trace_list,
                             highlight_bg_times = highlight_bg_times,
                             highlight_bg_patch_kwargs = {"alpha":0.3},
                            highlight_stim_times = highlighted_stim_times,
                            highlight_stim_patch_kwargs = {"alpha":0.3},
                            snippet_vline = snippet_vline,
                            single_snippet_dt = single_snippet_dt,
                            time_buffer_between_snippets = time_buffer_between_snippets,
                            x_tick_lables = x_tick_lables,
                            x_ticks_kwargs = {"rotation":45},
                            plot_kwargs = plot_kwargs,show_legend =show_legend,ax = ax)
        
def wrapper_plot_one_roi_successive_snippets(
            field_key,
            roi_id,
            bsl_correction_method = "first",
            cl_stim_family = "optstim",
            plot_kwargs = {},
            time_buffer_between_snippets = 0,
            show_legend=False,
            ax = None,
            cond2_value = None,
            drop_first_presentation = True,
            ):

    # fetch and process df
    mean_df,snippet_col_name = get_mean_snippet_df(field_key,
                                                   roi_id,bsl_correction_method=bsl_correction_method,
                                                    stim_name=cl_stim_family,
                                                    drop_first_presentation = drop_first_presentation,
                                                    keep_and_agg_more_cols = ["single_snippet_t0"],
                                                   cond2_value=cond2_value)
    print(f"Cols of mean df: {mean_df.columns.tolist()}")
    # select only the snippets with zero distance 
    sub_df =mean_df[mean_df["distance"] == 0]

    # group by t0
    print(f"Cols of sub df: {sub_df.columns.tolist()}")
    sub_df = sub_df.sort_values("single_snippet_t0",ascending=True)
    snippet_trace_list = sub_df[snippet_col_name].to_list()
    print([len(s) for s in snippet_trace_list])
    single_snippet_dt = 1/60

    # x tick labels are stimulus type
    x_tick_lables = sub_df["stimulus_type"].to_list()

    # plot them
    pu.plot_ordered_snippets(snippet_trace_list,
                          single_snippet_dt,
                            time_buffer_between_snippets = time_buffer_between_snippets,
                            x_tick_lables = x_tick_lables,
                            x_ticks_kwargs = {"rotation":45},
                            plot_kwargs = plot_kwargs,
                            show_legend =show_legend,
                            ax = ax)


def wrapper_plot_rois_list_ordered_snippets(
        field_key,
        roi_id_list,
        bsl_correction_method = "first",
        stim_name = "off_big",
        plot_kwargs = {},
        time_buffer_between_snippets = 0,
        ax = None,
        cond2_value = None,
    ):


    # fetch and process df
    # plot each roi
    n_col = np.sqrt(len(roi_id_list)).astype(int)
    n_row = (np.ceil(len(roi_id_list)/n_col)).astype(int)
    if ax is None:
        fig, ax = plt.subplots(n_row,n_col,figsize=(n_col*5,n_row*5))
        ax = ax.flatten()
    else:
        assert len(ax) >= len(roi_id_list), f"Expected ax to have at least {len(roi_id_list)} subplots, got {len(ax)}"
    for i,roi_id in enumerate(roi_id_list):
        print(f"Plotting roi {roi_id} ({i+1}/{len(roi_id_list)})")
        wrapper_plot_one_roi_ordered_snippets(
            field_key=field_key,
            roi_id=roi_id,
            bsl_correction_method=bsl_correction_method,
            stim_name=stim_name,
            plot_kwargs = plot_kwargs,
            time_buffer_between_snippets = time_buffer_between_snippets,
            ax = ax[i],
            cond2_value = cond2_value,
        )
        ax[i].set_title(f"Roi {roi_id}")
