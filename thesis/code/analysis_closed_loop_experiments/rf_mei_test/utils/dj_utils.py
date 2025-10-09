import os
from thesis.code.analysis_closed_loop_experiments.rf_mei_test.rf_mei_test_schema import *
import datajoint as dj
from omegaconf import DictConfig, ListConfig


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

def populate_experiment_field_presentation(clear=False,safemode=True) -> None:
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
    
    OpticDisk().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)



    Field().rescan_filesystem(verboselvl=3)
    RelativeFieldLocation().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)
    Presentation().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)

def add_rois_and_traces(clear=False) -> None:
    if clear:
        Roi().delete()
        Traces().delete()
        PreprocessTraces().delete()
        return
    
    Roi().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)
    Traces().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)
    PreprocessTraces().populate(processes=MULTIPROCESSING_THREADS, display_progress=True)


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
    preprocess_params[0]["stim_names"].extend(['circle','optstim'])

    



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
    CascadeTraceParams().add_default(stim_names=['mouse_cam'])
    CascadeParams().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel')


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
                            isrepeated=False, 
                            ntrigger_rep=0,
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


def prepclassifier_and_activate(output_folder):
    
    
    from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
    prepare_dj_config_rgc_classifier(output_folder)


    from djimaging.utils.dj_utils import activate_schema
    activate_schema(schema=schema, create_schema=True, create_tables=True)

    