import os
import datajoint as dj
import subprocess
import shutil
import warnings
warnings.simplefilter("ignore", FutureWarning)
from time import sleep 
from typing import List, Dict, Any, Tuple, Callable,Optional
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from djimaging.utils import plot_utils
from djimaging.utils.dj_utils import get_primary_key
from .utils import time_it
from .model_to_stimulus import (load_stimuli, preprocess_for_openretina,
                                train_model_online,generate_meis_with_n_random_seeds,
                                decompose_mei, get_model_mei_response
            )





class DJTableHolder:
    """
    A class that 
    1) Stores the tables so other compute classes can call operations on them 
    2) activates the schema and established a connection to the database.
    3) Calls a setup functio that populates the parameters tabeles
    """

    def __init__(self,
                 username: str,
                 home_directory: str,
                 repo_directory: str,
                 dj_config_directory: str,
                 rgc_output_directory: str,
                 data_subfolders: Dict[str, str],
                 
                 userinfo: dict,

                 table_parameters: dict,

                 sleep_time_between_table_ops: int  = 1,
                 debug: bool = False,
                 plot_results: bool = False,

                 ):
        """
        Store information needed to run table ops
        """
        self.iteration: int = 0
        self.debug: bool = debug
        self.multiprocessing_threads: int = 20 if not debug else 1 # this is so I can go in with debugger wo problems othrewise get problems
        self.plot_results: bool  = plot_results
        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops
        self.tables = {}

        # check who is running command
        self.username: str = username

        self.table_parameters: dict = table_parameters

        # get home directory
        self.home_directory: str = home_directory
        self.config_file: str = os.path.join(self.home_directory,dj_config_directory, f'dj_{self.username}_conf.json')
        self.rgc_output_folder = os.path.join(home_directory, rgc_output_directory)
        self.data_subfolders = data_subfolders

        # get repo directory
        self.repo_directory: str = repo_directory

        # information for UserInfo table
        self.userinfo: dict = userinfo
        self.schema_name = f"ageuler_{self.username}_closed_loop"

        # some tests idk if necessary
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()
        assert self.home_directory == os.path.expanduser("~")
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'
        assert os.path.exists(userinfo["data_dir"]), f"Path to data dir does not exist: {username["data_dir"]}"
        assert os.path.isdir(self.rgc_output_folder), f'Set path to output directory: {self.rgc_output_folder}'
    
    
    def __call__(self, table_name):
        
        """
        Access tables directly using the instance as a callable
        """
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} not found in loaded tables")
        return self.tables[table_name]

    def load_tables(self):
        """
        Import and load all database tables
        """
        

        from djimaging.user.ssuhai.schemas.ssuhai_schema_closed_loop import (
                UserInfo, Experiment, OpticDisk,RelativeFieldLocation, RelativeRoiLocation,
                Field, Stimulus, RoiMask, Roi, Traces,
                Presentation, RawDataParams, PreprocessParams, PreprocessTraces,
                Snippets, Averages, ChirpQI, OsDsIndexes,
                ClassifierMethod, ClassifierTrainingData, Classifier,
                Baden16Traces, CelltypeAssignment,
                CascadeTraceParams, CascadeParams, CascadeTraces, CascadeSpikes,
                # RF
                DNoiseTraceParams,DNoiseTrace,STAParams,STA, SplitRFParams,SplitRF,
                PeakSTAPosition,
                OpenRetinaHoeflingFormat, schema,
            )
        
        self.schema = schema

        # Store tables in dictionary
        self.tables = {
            'UserInfo': UserInfo,
            'Experiment': Experiment,
            'Field': Field,
            'OpticDisk': OpticDisk,
            'RelativeFieldLocation': RelativeFieldLocation,
            'RelativeRoiLocation': RelativeRoiLocation,
            'Stimulus': Stimulus,
            'RoiMask': RoiMask,
            'Roi': Roi,
            'Traces': Traces,
            'Presentation': Presentation,
            'RawDataParams': RawDataParams,
            'PreprocessParams': PreprocessParams,
            'PreprocessTraces': PreprocessTraces,
            'Snippets': Snippets,
            'Averages': Averages,
            'ChirpQI': ChirpQI,
            'OsDsIndexes': OsDsIndexes,
            'ClassifierMethod': ClassifierMethod,
            'ClassifierTrainingData': ClassifierTrainingData,
            'Classifier': Classifier,
            'Baden16Traces': Baden16Traces,
            'CelltypeAssignment': CelltypeAssignment,
            'CascadeTraceParams': CascadeTraceParams,
            'CascadeParams': CascadeParams,
            'CascadeTraces': CascadeTraces,
            'CascadeSpikes': CascadeSpikes,
            
            # RF 
            'DNoiseTraceParams': DNoiseTraceParams,
            'DNoiseTrace': DNoiseTrace,
            'STAParams': STAParams,
            'STA': STA,
            'SplitRFParams': SplitRFParams,
            'SplitRF': SplitRF,
            'PeakSTAPosition': PeakSTAPosition,

            

            # OpenRetina
            'OpenRetinaHoeflingFormat': OpenRetinaHoeflingFormat
        }

        sleep(self.sleep_time_between_table_ops)
        from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
        prepare_dj_config_rgc_classifier(self.rgc_output_folder)

        from djimaging.utils.dj_utils import activate_schema

        activate_schema(schema=self.schema) #, create_schema=True, create_tables=True)
        sleep(self.sleep_time_between_table_ops)


            
            

    def load_config(self) -> None:
        """
        load config file 
        """
        # Load configuration for user
        dj.config.load(self.config_file)

        dj.config['schema_name'] = self.schema_name
        dj.config['enable_python_native_blobs'] = True
        dj.config["display.limit"] = 20
        sleep(self.sleep_time_between_table_ops)

        print("schema_name:", dj.config['schema_name'])
        dj.conn()
        sleep(self.sleep_time_between_table_ops)
        
        

    def set_params_and_userinfo(self) -> None:

        # make sure tables are empty 
        if len(self('UserInfo')()) == 0:
            self('UserInfo')().upload_user(self.userinfo)
            sleep(self.sleep_time_between_table_ops)

        self('RawDataParams')().add_default()
        sleep(self.sleep_time_between_table_ops)

        # TODO extract hard coded values to config
        self('RawDataParams')().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        sleep(self.sleep_time_between_table_ops)

        preprocess_params =self.table_parameters.get("PreprocessParams", {})
        print("preprocessing params:\n",preprocess_params)
        self('PreprocessParams')().add_default(**preprocess_params)
   
        

        # Celltype assignment
        self('ClassifierTrainingData')().add_default(skip_duplicates=True)
        self('ClassifierMethod')().add_default(skip_duplicates=True)
        self('Classifier')().populate()

        # rf estimation 
        dense_noise_parmas = self.table_parameters.get("DNoiseTraceParams", {})
        self('DNoiseTraceParams')().add_default(**dense_noise_parmas)


        self('STAParams')().add_default()
        self('SplitRFParams')().add_default()



        # spike estimation
        self('CascadeTraceParams')().add_default(stim_names=['mouse_cam'])
        self('CascadeParams')().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel') # for spike estimation itself

    def add_all_stimuli(self) -> None:
        import h5py

        with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)
        noise_stimulus = noise_stimulus*2-1
        
        
        self('Stimulus')().add_nostim(skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)

        
        self('Stimulus')().add_noise(**self.table_parameters.Stimulus.noise, stim_trace=noise_stimulus,)
            
        self('Stimulus')().add_movingbar(skip_duplicates=True)
        
        self('Stimulus')().add_stimulus(
            stim_name="mouse_cam", 
            alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
            stim_family="natural", 
            framerate=30.0,  
            stim_path="",
            ntrigger_rep=123,
            unique_alias=True,
            skip_duplicates=True,
        )

        

    def setup(self) -> None:
        """
        Wrapper for the things we can do before the first iteration. 
        These are: 
        1) Connecting to the database
        2) Loading the tables
        3) Activating the schema
        4) setting the information of the stimuli used
        5) setting geenral pipeline parameters and user infor
        """
        
        self.load_config()
        sleep(self.sleep_time_between_table_ops)
        
        
        self.load_tables()
        sleep(self.sleep_time_between_table_ops)


        self.add_all_stimuli()
        sleep(self.sleep_time_between_table_ops)
        
        self.set_params_and_userinfo()
        sleep(self.sleep_time_between_table_ops)



    def clear_tables(self,target: str = "all", field_key : Dict[str,str] | None = None) -> None:
        """
        Clear tables.
        This is useful to start a new iteration with a clean slate.
        """
        if target == "all":
            # delete entries of userinfo will result in all being deleted
            self('UserInfo')().delete()

            # delete Stimulus, all params tables and  classifier tables as these do not depend on userinfo
            self('Stimulus')().delete()
            self('RawDataParams')().delete()
            self('PreprocessParams')().delete()
            self('ClassifierMethod')().delete()
            self('Classifier')().delete()
            self('ClassifierTrainingData')().delete()
            self('CascadeTraceParams')().delete()
            self('CascadeParams')().delete()
            self('DNoiseTraceParams')().delete()

            self('STAParams')().delete()
            self('SplitRFParams')().delete()

        
        elif target == "experiment":
            # delete all entries of the current experiment
            self('Experiment')().delete()

        elif target == "field":
            # delete all entries of the current field
            if field_key is None:
                raise ValueError("field_key must be provided when target is 'field'")
            (self('Field')() & field_key).delete()
        


######################################################## For Table operations up to Traces ##########################################################
class Preprocessor:
    """ Class that hold table operations for 
    1) loading data into database
    2) addin ROIs
    3) adding traces"""

    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the Preprocessor with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder


    def upload_iteration_metadata(self) -> None:
        """
        Upload metadata for the current iteration, including experiments, fields, and presentations.
        """
        # TODO: rescan vermeiden, scant alle 
        self.dj_table_holder('Experiment')().rescan_filesystem(verboselvl=3)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)
        
        self.dj_table_holder('OpticDisk')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)



        self.dj_table_holder('Field')().rescan_filesystem(verboselvl=3)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('RelativeFieldLocation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('Presentation')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)
        
    def add_iteration_roi_mask(self) -> None:
        """
        Add ROIs for the current iteration by drawing and inserting them into the database.
        NOTE that this will not be called from the GUI, but its for when we use this from the command line
        """
        self.dj_table_holder('RoiMask')().rescan_filesystem(verboselvl=3)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        missing_fields = self.dj_table_holder('RoiMask')().list_missing_field()
        assert len(missing_fields) == 1 , f"Expecting exatly no missing fields but found {len(missing_fields)}"
        field_key = missing_fields[0]
        sleep(self.dj_table_holder.sleep_time_between_table_ops)


        roi_canvas = self.dj_table_holder('RoiMask')().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        #TODO : REALLY UNSURE IF THIS DOES WHAT I WANT 
        # roi_canvas.exec_autorois_all_stacks()
        roi_canvas.exec_autorois_all()
        roi_canvas.exec_save_to_file()

        # add to database
        roi_canvas.insert_database(roi_mask_tab=self.dj_table_holder('RoiMask'), field_key=field_key)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)


        if self.dj_table_holder.plot_results:
            self.dj_table_holder('RoiMask')().plot1()
            plt.gcf().savefig(os.path.join(self.dj_table_holder.repo_directory,"figures",f"roi_mask_{field_key}.png"), dpi=300, bbox_inches='tight')
    
    def add_iteration_rois(self) -> None:
        self.dj_table_holder('Roi')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

    def add_iteration_traces(self) -> None:
        """
        Populate the Traces and PreprocessTraces tables for the current iteration.
        """
        self.dj_table_holder('Traces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)

        self.dj_table_holder('PreprocessTraces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        sleep(self.dj_table_holder.sleep_time_between_table_ops)



#################################################### DJ Compute Wrappers ###############################################


class DJComputeWrapper(ABC):
    """
    ABC for DJ wrappers: They group table operations together and have 
    functions that allow them to be inetrated into the GUI.
    The idea is that this makes the whole pipeline more modualr and expandable: users can create their own wrappers that can be used in the GUI."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the wrapper, used to identify it in the GUI.
        """
        pass


    @abstractmethod
    def plot1(self, roi_key: Dict[str, Any]) -> None:
        """
        Plot the first roi of the wrapper. This is called in the gui.
        """
        pass

    @abstractmethod
    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        """
        Plot an overview of the rois of the wrapper. This is called in the gui.
        """
        pass

    @abstractmethod
    def check_requirements(self) -> bool:
        """
        Check if the required tables are populated in the database.
        """
        pass

    @abstractmethod
    def compute_analysis(self) -> None:
        """
        Compute the main analysis for the wrapper. This is called in the gui. """
        pass

class QualityAndTypeWrapper(DJComputeWrapper):
    """Groups table operations for quality metrics and cell type assignment."""
    
    # a color map used to get an roi 2 color mapping based on the g name
    g_name_to_rgb255 = {
        "OFF": np.array([255, 0, 0]),          # Red
        "ON-OFF": np.array([0, 255, 0]),      # Green
        "Fast ON": np.array([64, 224, 208]),  # Turquoise
        "Slow ON": np.array([0, 0, 255]),     # Blue
        "Uncertain RGCs": np.array([128, 0, 128]),  # Purple
        "ACs": np.array([255, 255, 255])        # white
    }



    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the QualityAndType wrapper with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder
        self.requires_tables = [
            'Snippets',
            'Averages',
            ]

    @property
    def name(self) -> str:
        return "Quality and Type"
    
    def add_quality_metrics(self,field_key) -> None:
        """
        Populate the ChirpQI and OsDsIndexes tables for quality metrics.
        """
        if len(self.dj_table_holder('ChirpQI')() & field_key) == 0:
            
            self.dj_table_holder('ChirpQI')().populate(display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)
            
        if len(self.dj_table_holder('OsDsIndexes')() & field_key) == 0:
            self.dj_table_holder('OsDsIndexes')().populate(display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)

    def add_celltype_assignments(self, field_key) -> None:
        """
        Populate the Baden16Traces and CelltypeAssignment tables for cell type assignments.
        """

        if len(self.dj_table_holder('Baden16Traces')() & field_key) == 0:
            self.dj_table_holder('Baden16Traces')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        
        if len(self.dj_table_holder('CelltypeAssignment')() & field_key) == 0:
            self.dj_table_holder('CelltypeAssignment')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

    def compute_analysis(self, field_key = {},progress_callback: Optional[Callable]  = None) -> None:
        """
        Compute quality metrics and cell type assignment.
        """
        if progress_callback is not None:
            progress_callback(0)

        self.check_requirements(field_key)
        if progress_callback is not None:
            progress_callback(30)

        self.add_quality_metrics(field_key)
        if progress_callback is not None:
            progress_callback(60)

        self.add_celltype_assignments(field_key)
        if progress_callback is not None:
            progress_callback(100)



    
    @staticmethod
    def g_to_type_name(g: int) -> str:

        if g in range(1,10):
            return "OFF"
        elif g in range(10,15):
            return "ON-OFF"
        elif g in range(15,21):
            return "Fast ON"
        elif g in range(21,29):
            return "Slow ON"
        elif g in range(29,33):
            return "Uncertain RGCs"
        elif g >= 33:
            return "ACs"
        else:
            raise ValueError(f"Unknown group {g} for g_to_type_name")

    
    def g_to_rgb255(self,g: int) -> np.ndarray:
        """Converts the G (celltype) to a superclass type name and from there the designated RGB255 color."""

        g_name = self.g_to_type_name(g)
        return self.g_name_to_rgb255[g_name]


    # range mapping from quality index to alpha value 
    
    def qi2alpha255(self,qi: float,alpha_min = 0.1,alpha_max = 0.6) -> float:
        assert 0 <= qi <= 1, "Quality index must be between 0 and 1"
        return (alpha_min + (qi * (alpha_max - alpha_min))) * 255


    
    def text1(self,roi_id: int, field_key = {}, ) -> str:
        """
        Get the text for the requested roi.
        """
        roi_restriction = {'roi_id': roi_id}
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        ori_dir_qi_table = self.dj_table_holder('OsDsIndexes')() & field_key
        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key

        d_qi = (ori_dir_qi_table & roi_restriction).fetch("d_qi").item()
        qidx_chirp = (chirp_qi_table & roi_restriction).fetch("qidx").item()

        celltype_data = (celltype_table & roi_restriction).fetch1()
        confidence_scores = celltype_data['confidence']

        # Get top 3 with scores
        top_3_indices = confidence_scores.argsort()[-3:][::-1]
        top_3_groups = top_3_indices + 1 # assume index based group assignment
        top_3_scores = confidence_scores[top_3_indices]
        top_3_group_names = [self.g_to_type_name(g) for g in top_3_groups]

        assert celltype_data["celltype"] == top_3_groups[0], "Top group does not match celltype assignment"

        text = "Quality Metrics:\n"
        text += f"Chirp QI: {qidx_chirp:.2f}\n"
        text += f"d_qi: {d_qi:.2f}\n\n"

        text += "Top 3 types:\n"
        for g,name,score in zip(top_3_groups, top_3_group_names,top_3_scores):
            text += f"{g} ({name}): {score:.2f}\n"
        return text


    def plot_g_name_legend(self, ax, y_start=0.05, fontsize=10):
        """
        Plots a legend mapping g_name to RGB color below the text area,
        with rectangles drawn just before the labels.
        """
        y_step = 0.07
        rect_x = 0.05  # x-position for rectangles
        text_x = rect_x + 0.07  # x-position for text, just after rectangle

        for i, (g_name, rgb) in enumerate(self.g_name_to_rgb255.items()):
            color = tuple(rgb / 255)
            y = y_start + i * y_step
            # Draw colored rectangle
            ax.add_patch(plt.Rectangle((rect_x, y), 0.06, 0.05, color=color, transform=ax.transAxes, clip_on=False))
            # Draw label just after rectangle
            ax.text(text_x, y + 0.025, f"{g_name}", fontsize=fontsize, color='black',
                    verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)
        
    def plot1(self, roi_id, stim_name= "gChirp", field_key = {}, xlim=None,show_fig = True) -> None:
        """
        Plot the Averages of the requested roi.
        """
        import matplotlib.gridspec as gridspec


        single_averages_table = (self.dj_table_holder('Averages')() & field_key & {'roi_id': roi_id, 'stim_name': stim_name})
        snippets_table = self.dj_table_holder('Snippets')() & field_key


        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (snippets_table & single_averages_table).fetch1(
            'snippets_t0', 'snippets_dt', 'snippets', 'triggertimes_snippets')

        average, average_norm, average_t0, average_dt, triggertimes_rel = \
            single_averages_table.fetch1('average', 'average_norm', 'average_t0', 'average_dt', 'triggertimes_rel')

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)
        average_times = np.arange(len(average)) * average_dt + average_t0

        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 4))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])
        
        # Text area (spans both rows, left column)
        text_ax = fig.add_subplot(gs[:, 0])
        text_ax.axis('off')
        
        # Plot areas (right column)
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        
        # plotting
        ax1.plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        ax1.set(ylabel='trace', xlim=xlim)
        
        plot_utils.plot_trace_and_trigger(
            ax=ax2, time=average_times, trace=average,
            triggertimes=triggertimes_rel, trace_norm=average_norm)
        ax2.set(xlabel='rel. to trigger')
        
        # Add text to dedicated text area
        text = self.text1(roi_id,field_key=field_key)
        text_ax.text(0.05, 0.95, text, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    transform=text_ax.transAxes)
        
        # add colors for g_name
        self.plot_g_name_legend(text_ax, y_start=0.05, fontsize=10)

        # set tite
        fig.suptitle(f"ROI {roi_id} - {stim_name}", fontsize=16)
        
        if show_fig:
            plt.show()

    
    
    def check_requirements(self, field_key) -> None:
        """
        Check if the required tables are populated in the database.
        """
        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & field_key) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
                sleep(self.dj_table_holder.sleep_time_between_table_ops)

    def get_roi2rgb_and_alpha_255_map(self,field_key: Dict[str, Any]) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Get two mappings: one for roi to rgb based on celltype and one roi to alpha based on chirpQI
        

        """
        
        # get celltype for each roi 
        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key
        celltype_data = celltype_table.fetch('roi_id', 'celltype',as_dict=True)
        roi2rgb255 = {data['roi_id']: self.g_to_rgb255(data['celltype']) for data in celltype_data}

        # get chirpQI for each roi
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        chirp_qi_data = chirp_qi_table.fetch('roi_id', 'qidx', as_dict=True)
        roi2alpha = {data['roi_id']: self.qi2alpha255(data['qidx']) for data in chirp_qi_data}

        return roi2rgb255, roi2alpha
         
    
    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        """
        Plot an overview of the rois of the wrapper.
        """
        pass


class STAWrapper(DJComputeWrapper):
    """Groups table operations for STA estimation."""

    def __init__(self, dj_table_holder: DJTableHolder):
        """
        Initialize the STA wrapper with a DJTableHolder instance.
        """
        self.dj_table_holder = dj_table_holder
        self.requires_tables = [
            
        ]

    @property
    def name(self) -> str:
        return "STA"
    
    def compute_analysis(self, field_key = {},progress_callback: Optional[Callable] = None) -> None:
        """
        Compute the STA analysis.
        """
        if len(self.dj_table_holder('STA')() & field_key) == 0:
            if progress_callback is not None:
                progress_callback(0)
            
            self.dj_table_holder('DNoiseTrace')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
            sleep(self.dj_table_holder.sleep_time_between_table_ops)

            if progress_callback is not None:
                progress_callback(30)
            self.dj_table_holder('STA')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
            sleep(self.dj_table_holder.sleep_time_between_table_ops)

            if progress_callback is not None:
                progress_callback(80)
            self.dj_table_holder('SplitRF')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
            sleep(self.dj_table_holder.sleep_time_between_table_ops)

            self.dj_table_holder('PeakSTAPosition')().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

            if progress_callback is not None:
                progress_callback(100)

    def check_requirements(self, field_key) -> None:
        """
        Check if the required tables are populated in the database.
        """
        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & field_key) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
                sleep(self.dj_table_holder.sleep_time_between_table_ops)
    
    
    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        pass
        
    def plot1(self,roi_id: int,field_key={},show = True) -> None:

        restricted_split_rf = (self.dj_table_holder('SplitRF')() & field_key & {'roi_id': roi_id})

        assert len(restricted_split_rf) == 1, f"Expected exactly one SplitRF for roi_id {roi_id}, found {len(restricted_split_rf)}"

        # plot it and 
        key = get_primary_key(table=restricted_split_rf, key=None)

        rf_time = restricted_split_rf.fetch1_rf_time(key=key)
        srf, trf, peak_idxs = (restricted_split_rf & key).fetch1("srf", "trf", "trf_peak_idxs")

        fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex='col')

        ax = axs[0]
        plot_utils.plot_srf(srf, ax=ax)
        ax.set_title(f'sRF for ROI {roi_id}')

        # add x mark at the peak position in index coordinates
        x,y = self.dj_table_holder("PeakSTAPosition")().get_rf_spatial_max_indices(srf)
        
        # flip x and y becaus in QDSpy dense noise the axis 0 is x so I called it x but its axis 0. 
        ax.plot(y,x, 'x', color='black', markersize=10, label='Peak Position')
        ax.legend()

        ax = axs[1]
        plot_utils.plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)

        if show:
            plt.show()






class RandomSeedMEIWrapper(DJComputeWrapper):

    def __init__(self,dj_table_holder,
                 model_configs,
                 seeds: List[int],
                ) -> None:
        
        self.dj_table_holder = dj_table_holder

        self.requires_tables = [
            "CascadeTraces",
            "CascadeSpikes",
        ]
        self.model_configs = model_configs

        # to store the data: the key is the index in the readout and not the roi_id
        self.neuron_seed_mei_dict = {}
        self.neuron_seed_mei_responses = {}
        self.neuron_seed_decomposed_meis = {}
        self.model = None

        # these are the roi ids that were kept after filtering
        self.rois_after_filtering: List[int] = []

        self.display_channel = 1 # UV channel 

        self.seeds = seeds
        self.colors = plt.cm.nipy_spectral(np.linspace(0, 1,len(self.seeds))) 

        # make sure openretina table is empty 
        assert len(self.dj_table_holder('OpenRetinaHoeflingFormat')()) == 0, "OpenRetinaHoeflingFormat table is not empty. Please clear it before using this wrapper."

    def plot_seed_respones(self,neuron_id: int,ax: plt.Axes, optimization_window= (10,20),response_window = (21,50)):
        """
        Plots the responses of all seeds for a single neuron.
        all_meis_responses: has the structure {seed: response}
        optimization_window: what part of the  repsonse of the neuron was optimized. Then the frame of the start of the repsonse is added during plotting
        """
        response_start, respones_end = response_window
    
        all_meis_responses: Dict[int, np.ndarray] = self.neuron_seed_mei_responses[neuron_id]

        x = np.arange(response_start, respones_end + 1)
        for i,(seed, response) in enumerate(all_meis_responses.items()):
            ax.plot(x,response, label=f"Seed {seed}", color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')

        ax.set_xlabel("Time (frames)")
        ax.set_xlim(0, respones_end)
        ax.set_ylabel("Response")
        
        # Highlight the optimization window
        if optimization_window is not None:
            start , end  = optimization_window
            start += response_start
            end += response_start
            ax.axvspan(start , end, color='yellow', alpha=0.3, label='Optimization Window')
        ax.legend(ncol=2, fontsize=6)


    def plot_temporal_kernels(self,neuron_id: int, ax: plt.Axes) -> None:
        """
        Plots the temporal kernels of all seeds for a single neuron.
        """
        seed_neuron_decomposed_meis = self.neuron_seed_decomposed_meis[neuron_id]
        for i,(seed, decomposition) in enumerate(seed_neuron_decomposed_meis.items()):
            temporal_kernel = decomposition['temporal_kernels'][self.display_channel]
            ax.plot(temporal_kernel, label=f'Seed {seed}', color=self.colors[i], linestyle='-' if seed % 2 == 0 else '--')
        
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Temporal Kernel')
        ax.legend()

    def plot1(self,roi_id: int, field_key: Dict[str,Any] = {}) -> None:

        if roi_id not in self.rois_after_filtering:
            print("ROI ID not found in filtered rois. Available IDs:", self.rois_after_filtering)
            return 
        
        # find neuron_id for roi_id
        neuron_id = self.rois_after_filtering.index(roi_id)


        # plot temporal kernels in a line plot
        fig,axs = plt.subplots(1,2,figsize=(10, 5))
        self.plot_temporal_kernels(neuron_id, ax=axs[0])
        axs[0].set_title(f"Temporal Kernels for ROI {roi_id} (neuron idx {neuron_id})")

    

        # plot responses 
        self.plot_seed_respones(neuron_id, ax=axs[1], optimization_window=(10,20), response_window=(21,50))
        axs[1].set_title(f"Responses to MEIs")

        plt.show()


    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        pass

    
    @property
    def name(self) -> str:
        return "Random Seed MEI"

    def check_requirements(self, field_key, progress_callback: Optional[Callable]) -> None:
        """
        Check if the required tables are populated in the database.
        """

        progress: int = 0 
        
        if progress_callback is not None:
            progress_callback(0)

        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & field_key) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
                sleep(self.dj_table_holder.sleep_time_between_table_ops)

                progress += 15
                if progress_callback is not None:
                    progress_callback(progress)


    def compute_analysis(self, field_key = {},progress_callback: Optional[Callable] = None) -> None:

        # extract data in hoefling format from DB 
        if len(self.dj_table_holder('OpenRetinaHoeflingFormat')() & field_key) == 0:
            if progress_callback is not None:
                progress_callback(0)

            self.check_requirements(field_key,progress_callback= progress_callback)

            if progress_callback is not None:
                progress_callback(30)
            ## model training 
            self.session_dict_raw = self.dj_table_holder('OpenRetinaHoeflingFormat')().extract_data()
            
            # preprocess and filter further 
            movies_dict = load_stimuli(self.model_configs)

            neuron_data_dict = preprocess_for_openretina(self.session_dict_raw,self.model_configs)
    
            # load and refine model
            self.model = train_model_online(self.model_configs,neuron_data_dict,movies_dict)
            

            ## MEI generatio 
            new_session_id = list(self.session_dict_raw.keys())[0]

            # for debug: check roi_id to neuron_id mapping
            self.rois_after_filtering: List[int] = neuron_data_dict[new_session_id].session_kwargs["roi_ids"].tolist()

            # the neuron_id is the index in the readout I think, so we need a mapping betwen roi_id and neuron_id
            neurons_ids_to_analyze = list(range(len(self.rois_after_filtering)))

            if progress_callback is not None:
                progress_callback(70)
            self.neuron_seed_mei_dict = generate_meis_with_n_random_seeds(
                                        model = self.model,
                                        new_session_id = new_session_id,
                                        random_seeds =self.seeds,
                                        neuron_ids_to_analyze = neurons_ids_to_analyze, # NOTE: this will optimize each id individually 
                                        set_model_to_eval_mode = False, # model in training mode for noisy MEIs
                                    )

            

            ## generate responses for the MEIs and decompose them
            self.neuron_seed_mei_responses = {neuron_id: {} for neuron_id in self.neuron_seed_mei_dict.keys()}
            self.neuron_seed_decomposed_meis = {neuron_id: {} for neuron_id in self.neuron_seed_mei_dict.keys()}
            for neuron_id,seed_dict in self.neuron_seed_mei_dict.items():
                for seed,mei in seed_dict.items():

                    # responses 
                    response = get_model_mei_response(model = self.model,
                                                      mei=mei,
                                                      session_id = new_session_id,
                                                      neuron_id = neuron_id,)
                    self.neuron_seed_mei_responses[neuron_id][seed] = response

                    # decompose the MEIs
                    temporal_kernels, spatial_kernels, stimulus_time = decompose_mei(stimulus = mei.detach().cpu().numpy())
                    self.neuron_seed_decomposed_meis[neuron_id][seed] = {
                        "temporal_kernels": temporal_kernels,
                        "spatial_kernels": spatial_kernels,
                        "stimulus_time": stimulus_time,
                    }
            if progress_callback is not None:
                progress_callback(100)




class OpenRetinaWrapper:
    """
    After init call setup once and the process_iteration_data in each iteration."""

    def __init__(self,
                 username: str,
                 home_directory: str,
                 repo_directory: str,
                 dj_config_directory: str,
                 rgc_output_directory: str,
                 data_subfolders: Dict[str, str],
                 
                 userinfo: dict,

                 table_parameters: dict,

                 sleep_time_between_table_ops: int  = 1,
                 debug: bool = False,
                 plot_results: bool = False,

                 ):
        """
        Store information needed to run table ops
        """
        self.iteration: int = 0
        self.debug: bool = debug
        self.multiprocessing_threads: int = 20 if not debug else 1 # this is so I can go in with debugger wo problems othrewise get problems
        self.plot_results: bool  = plot_results
        self.sleep_time_between_table_ops: int = sleep_time_between_table_ops
        self.tables = {}

        # check who is running command
        self.username: str = username

        self.table_parameters: dict = table_parameters

        # get home directory
        self.home_directory: str = home_directory
        self.config_file: str = os.path.join(self.home_directory,dj_config_directory, f'dj_{self.username}_conf.json')
        self.rgc_output_folder = os.path.join(home_directory, rgc_output_directory)
        self.data_subfolders = data_subfolders

        # get repo directory
        self.repo_directory: str = repo_directory

        # information for UserInfo table
        self.userinfo: dict = userinfo
        self.schema_name = f"ageuler_{self.username}_closed_loop"

        # some tests idk if necessary
        assert self.username == subprocess.check_output(["whoami"]).decode().strip()
        assert self.home_directory == os.path.expanduser("~")
        assert os.path.isfile(self.config_file), f'Set the path to your config file: {self.config_file}'
        assert os.path.exists(userinfo["data_dir"]), f"Path to data dir does not exist: {username["data_dir"]}"
        assert os.path.isdir(self.rgc_output_folder), f'Set path to output directory: {self.rgc_output_folder}'
  
    def __call__(self, table_name):
        """
        Access tables directly using the instance as a callable
        """
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} not found in loaded tables")
        return self.tables[table_name]

    def load_tables(self):
        """
        Import and load all database tables
        """
        

        from djimaging.user.ssuhai.schemas.ssuhai_schema_closed_loop import (
                UserInfo, Experiment, OpticDisk,RelativeFieldLocation, RelativeRoiLocation,
                Field, Stimulus, RoiMask, Roi, Traces,
                Presentation, RawDataParams, PreprocessParams, PreprocessTraces,
                Snippets, Averages, ChirpQI, OsDsIndexes,
                ClassifierMethod, ClassifierTrainingData, Classifier,
                Baden16Traces, CelltypeAssignment,
                CascadeTraceParams, CascadeParams, CascadeTraces, CascadeSpikes,
                # RF
                DNoiseTraceParams,DNoiseTrace,STAParams,STA, SplitRFParams,SplitRF,
                PeakSTAPosition,
                OpenRetinaHoeflingFormat, schema,
            )
        
        self.schema = schema
        # Store tables in dictionary
        self.tables = {
            'UserInfo': UserInfo,
            'Experiment': Experiment,
            'Field': Field,
            'OpticDisk': OpticDisk,
            'RelativeFieldLocation': RelativeFieldLocation,
            'RelativeRoiLocation': RelativeRoiLocation,
            'Stimulus': Stimulus,
            'RoiMask': RoiMask,
            'Roi': Roi,
            'Traces': Traces,
            'Presentation': Presentation,
            'RawDataParams': RawDataParams,
            'PreprocessParams': PreprocessParams,
            'PreprocessTraces': PreprocessTraces,
            'Snippets': Snippets,
            'Averages': Averages,
            'ChirpQI': ChirpQI,
            'OsDsIndexes': OsDsIndexes,
            'ClassifierMethod': ClassifierMethod,
            'ClassifierTrainingData': ClassifierTrainingData,
            'Classifier': Classifier,
            'Baden16Traces': Baden16Traces,
            'CelltypeAssignment': CelltypeAssignment,
            'CascadeTraceParams': CascadeTraceParams,

            'CascadeParams': CascadeParams,
            'CascadeTraces': CascadeTraces,
            'CascadeSpikes': CascadeSpikes,
            
            # RF 
            'DNoiseTraceParams': DNoiseTraceParams,
            'DNoiseTrace': DNoiseTrace,
            'STAParams': STAParams,
            'STA': STA,
            'SplitRFParams': SplitRFParams,
            'SplitRF': SplitRF,
            'PeakSTAPosition': PeakSTAPosition,

            

            # OpenRetina
            'OpenRetinaHoeflingFormat': OpenRetinaHoeflingFormat
        }

        sleep(self.sleep_time_between_table_ops)
        from djimaging.tables.classifier.rgc_classifier import prepare_dj_config_rgc_classifier
        prepare_dj_config_rgc_classifier(self.rgc_output_folder)

        from djimaging.utils.dj_utils import activate_schema

        activate_schema(schema=self.schema, create_schema=True, create_tables=True)
        sleep(self.sleep_time_between_table_ops)


            
            

    def load_config(self) -> None:
        """
        load config file 
        """
        # Load configuration for user
        dj.config.load(self.config_file)

        dj.config['schema_name'] = self.schema_name
        dj.config['enable_python_native_blobs'] = True
        dj.config["display.limit"] = 20
        sleep(self.sleep_time_between_table_ops)

        print("schema_name:", dj.config['schema_name'])
        dj.conn()
        sleep(self.sleep_time_between_table_ops)
        
        

    def set_params_and_userinfo(self) -> None:

        # make sure tables are empty 
        if len(self('UserInfo')()) == 0:
            self('UserInfo')().upload_user(self.userinfo)
            sleep(self.sleep_time_between_table_ops)

        self('RawDataParams')().add_default()
        sleep(self.sleep_time_between_table_ops)

        # TODO extract hard coded values to config
        self('RawDataParams')().update1(dict(
            experimenter='closedlooptest',
            raw_id=int(1),
            from_raw_data=int(1),
            igor_roi_masks='no',
            ))
        sleep(self.sleep_time_between_table_ops)

        preprocess_params =self.table_parameters.get("PreprocessParams", {})
        self('PreprocessParams')().add_default(**preprocess_params)
   
        

        # Celltype assignment
        self('ClassifierTrainingData')().add_default(skip_duplicates=True)
        self('ClassifierMethod')().add_default(skip_duplicates=True)
        self('Classifier')().populate()

        # rf estimation 
        dense_noise_parmas = self.table_parameters.get("DNoiseTraceParams", {})
        self('DNoiseTraceParams')().add_default(**dense_noise_parmas)


        self('STAParams')().add_default()
        self('SplitRFParams')().add_default()



        # spike estimation
        self('CascadeTraceParams')().add_default(stim_names=['mouse_cam'])
        self('CascadeParams')().add_default(model_name = 'Global_EXC_7.8125Hz_smoothing200ms_causalkernel') # for spike estimation itself


        

    def setup(self) -> None:
        """
        Wrapper for the things we can do before the first iteration. 
        These are: 
        1) Connecting to the database
        2) Loading the tables
        3) Activating the schema
        4) setting the information of the stimuli used
        5) setting geenral pipeline parameters and user infor
        """
        
        self.load_config()
        sleep(self.sleep_time_between_table_ops)
        
        
        self.load_tables()
        sleep(self.sleep_time_between_table_ops)


        self.add_all_stimuli()
        sleep(self.sleep_time_between_table_ops)
        
        self.set_params_and_userinfo()
        sleep(self.sleep_time_between_table_ops)

##################################################################### During iteration functions ##############################################################################
    @time_it
    def upload_iteration_metadata(self) -> None:
        

        # TODO: rescan vermeiden, scant alle 
        if self.iteration == 0:
            self('Experiment')().rescan_filesystem(verboselvl=3)
            sleep(self.sleep_time_between_table_ops)
            
            self('OpticDisk')().populate(processes=self.multiprocessing_threads, display_progress=True)



        self('Field')().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        self('RelativeFieldLocation')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('Presentation')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)
 
        

    @time_it
    def add_iteration_rois(self) -> None:
        """
  
        """
        self('RoiMask')().rescan_filesystem(verboselvl=3)
        sleep(self.sleep_time_between_table_ops)

        missing_fields = self('RoiMask')().list_missing_field()
        assert len(missing_fields) == 1 , f"Expecting exatly no missing fields but found {len(missing_fields)}"
        field_key = missing_fields[0]
        sleep(self.sleep_time_between_table_ops)


        roi_canvas = self('RoiMask')().draw_roi_mask(field_key=field_key, canvas_width=30)
        
        #TODO : REALLY UNSURE IF THIS DOES WHAT I WANT 
        # roi_canvas.exec_autorois_all_stacks()
        roi_canvas.exec_autorois_all()
        roi_canvas.exec_save_to_file()

        # add to database
        roi_canvas.insert_database(roi_mask_tab=self('RoiMask'), field_key=field_key)
        sleep(self.sleep_time_between_table_ops)
        self('Roi')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        if self.plot_results:
            self('RoiMask')().plot1()
            plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"roi_mask_{field_key}.png"), dpi=300, bbox_inches='tight')

    @time_it
    def add_iteration_traces(self) -> None:
        

        self('Traces')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('PreprocessTraces')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)


    @time_it
    def add_trace_reformatting(self) -> None:
        
        
        self('Snippets')().populate(processes=self.multiprocessing_threads, display_progress=True)
        
        sleep(self.sleep_time_between_table_ops)
        self('Averages')().populate(processes=self.multiprocessing_threads, display_progress=True)

    @time_it
    def add_quality_metrics(self) -> None:

        self('ChirpQI')().populate(display_progress=True, processes=self.multiprocessing_threads)
        self('OsDsIndexes')().populate(display_progress=True, processes=self.multiprocessing_threads)

    @time_it
    def add_sta(self) -> None:

        self('DNoiseTrace')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

        self('STA')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

    @time_it
    def add_spikes(self) -> None:

        self('CascadeTraces')().populate()
        sleep(self.sleep_time_between_table_ops)

        # TODO: see if we can only do this for mouse cam??? bc takes a lot of time
        self('CascadeSpikes')().populate()


    def add_all_stimuli(self) -> None:
        import h5py

        with h5py.File("/gpfs01/euler/data/Resources/Stimulus/noise.h5", "r") as f:
            noise_stimulus = f['stimulusarray'][:].T.astype(int)
        noise_stimulus = noise_stimulus*2-1
        
        
        self('Stimulus')().add_nostim(skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=1000, stim_name='gChirp', alias="chirp_gchirp_globalchirp", skip_duplicates=True)
        self('Stimulus')().add_chirp(spatialextent=300, stim_name='lChirp', alias="lchirp_localchirp", skip_duplicates=True)

        
        self('Stimulus')().add_noise(**self.table_parameters.Stimulus.noise, stim_trace=noise_stimulus,)
        
        # self('Stimulus')().update1(dict(stim_name='densnoise',stim_family='noise', 
        #               stim_dict=dict( pix_scale_x_um=40, pix_scale_y_um=40, pix_n_x=20, pix_n_y=15, framerate = 5)))
        
        self('Stimulus')().add_movingbar(skip_duplicates=True)
        
        self('Stimulus')().add_stimulus(
            stim_name="mouse_cam", 
            alias="mc00_mc01_mc02_mc03_mc04_mc05_mc06_mc07_mc08_mc09_mc10_mc11_mc12_mc13_mc14_mc15_mc16_mc17_mc18_mc19_mc00bd_mc01bd_mc02bd_mc03bd_mc04bd_mc05bd_mc06bd_mc07bd_mc08bd_mc09bd_mc10bd_mc11bd_mc12bd_mc13bd_mc14bd_mc15bd_mc16bd_mc17bd_mc18bd_mc19bd", 
            stim_family="natural", 
            framerate=30.0,  
            stim_path="",
            ntrigger_rep=123,
            unique_alias=True
        )

    
    @time_it
    def add_celltype_assignments(self) -> None:

        self('Baden16Traces')().populate(display_progress=True, processes=self.multiprocessing_threads)
        self('CelltypeAssignment')().populate(display_progress=True)
        
        if self.plot_results:
            for threshold_confidence in [0, 0.25, 0.5]:
                self('CelltypeAssignment')().plot(threshold_confidence=threshold_confidence)
                plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')
                
                # self('CelltypeAssignment')().plot_features(threshold_confidence= threshold_confidence)
                # plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_features_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')
                
                self('CelltypeAssignment')().plot_group_traces(threshold_confidence= threshold_confidence)
                plt.gcf().savefig(os.path.join(self.repo_directory,"figures",f"celltype_assignment_traces_confidence_threshold_{threshold_confidence}.png"), dpi=300, bbox_inches='tight')

            
        # if self.debug:
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.0)
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.25)
        #     self('CelltypeAssignment')().plot(threshold_confidence=0.5)

    @time_it
    def add_peak_sta_positions(self) -> None:

        self('PeakSTAPosition')().populate(processes=self.multiprocessing_threads, display_progress=True)
        sleep(self.sleep_time_between_table_ops)

       
    
    @time_it
    def extract_data(self) -> Dict[str,Dict[str, Any]] | None:
        
        session_dict = self('OpenRetinaHoeflingFormat')().extract_data()
        return session_dict

    

    def compute_quality_and_type(self, compute_or_delete = "compute") -> None:

        if compute_or_delete == "compute":

            # check if rois have been extracted
            if len(self('Roi')()) == 0:
                self('Roi')().populate(processes=self.multiprocessing_threads, display_progress=True)
            
            # check if traces tables are populated 
            if len(self('PreprocessTraces')()) == 0:
                self('Traces')().populate(processes=self.multiprocessing_threads, display_progress=True)
                sleep(self.sleep_time_between_table_ops)

                self('PreprocessTraces')().populate(processes=self.multiprocessing_threads, display_progress=True)
                sleep(self.sleep_time_between_table_ops)

            # check if snipptes table is provided
            if len(self('Snippets')()) == 0:
                self('Snippets')().populate(processes=self.multiprocessing_threads, display_progress=True)
                sleep(self.sleep_time_between_table_ops)
                self('Averages')().populate(processes=self.multiprocessing_threads, display_progress=True)

            # add the quality metrics and type assignments
            self('ChirpQI')().populate(display_progress=True, processes=self.multiprocessing_threads)
            self('OsDsIndexes')().populate(display_progress=True, processes=self.multiprocessing_threads)

            # add type assignments
            self('Baden16Traces')().populate(display_progress=True, processes=self.multiprocessing_threads)
            self('CelltypeAssignment')().populate(display_progress=True)
   
            
        elif compute_or_delete == "delete":
            # delete the quality metrics and type assignments
            self('ChirpQI')().delete()
            self('OsDsIndexes')().delete()

            # delete type assignments
            self('Baden16Traces')().delete()
            self('CelltypeAssignment')().delete()

        else:
            raise ValueError("compute_or_delete must be either 'compute' or 'delete'")

    def compute_sta(self, compute_or_delete = "compute") -> None:
        
        if compute_or_delete == "compute":
            # add DN traces
            self('DNoiseTrace')().populate(processes=self.multiprocessing_threads, display_progress=True)
            sleep(self.sleep_time_between_table_ops)

            # add STA traces
            self('STA')().populate(processes=self.multiprocessing_threads, display_progress=True)
            sleep(self.sleep_time_between_table_ops)

            # Split RF
            self('SplitRF')().populate(processes=self.multiprocessing_threads, display_progress=True)

        elif compute_or_delete == "delete":
            # delete DN traces
            self('DNoiseTrace')().delete()
            sleep(self.sleep_time_between_table_ops)

            # delete STA traces
            self('STA')().delete()
            sleep(self.sleep_time_between_table_ops)

            # delete Split RF
            self('SplitRF')().delete()
        else:
            raise ValueError("compute_or_delete must be either 'compute' or 'delete'")            

    def process_iteration_data(self,) -> Dict[str,Dict[str, Any]] | None:
        """
        main function =
        Call this function to run the pipeline on each iteration of the loop
        """
        
        
        
        self.upload_iteration_metadata()
        self.add_iteration_rois()

        self.add_iteration_traces()
        self.add_trace_reformatting()

        self.add_quality_metrics()
        self.add_celltype_assignments()

        self.add_sta()
        self.add_peak_sta_positions()


        self.add_spikes()
        iteration_data = self.extract_data()

        self.iteration += 1

        return iteration_data


        
##################################################################  Other utils  #############################################################################################
    
    def clean_up(self, at_processing_stage = 'setup') -> None:
            """
            Clear schema tables.
            
            Args:
                all: If True, delete all tables. If False, only delete tables modified in each iteration.
            """


            self.iteration = 0
            


            # Tables that are modified in each iteration
            iteration_modified_table_names = [
                'RoiMask',
                'Roi', 
                'Traces',
                'PreprocessTraces',
                'Presentation',
                'PeakSTAPosition',
               
                'DNoiseTrace',
                'STA',

                'CascadeTraces',
                'CascadeSpikes',
            ]
            
            data_extraction_table_names = [
                'OpenRetinaHoeflingFormat',
            ]
            # Tables that are static across iterations
            static_table_names = [
                'UserInfo',
                'Experiment',
                'Field',
                'Stimulus', 
                'RawDataParams',
                'PreprocessParams',
                'Snippets',
                'Averages',
                'ChirpQI',
                'OsDsIndexes',
                'ClassifierMethod',
                'ClassifierTrainingData',
                'Classifier',
                'CascadeTraceParams',
                'CascadeParams',
                'Baden16Traces',
                'CelltypeAssignment',
                'DNoiseTraceParams',
                'STAParams',
            ]
            
            # Tables to delete based on 'all' parameter
            if at_processing_stage == 'setup':
                table_names_to_delete = static_table_names + iteration_modified_table_names + data_extraction_table_names
            elif at_processing_stage == 'iteration':
                table_names_to_delete =  iteration_modified_table_names + data_extraction_table_names
            elif at_processing_stage == 'data_extraction':
                table_names_to_delete = data_extraction_table_names
            else:
                raise ValueError("Invalid processing stage. Choose from 'setup', 'iteration', or 'data_extraction'.")
             
            # Delete tables with pause between each operation
            for table_name in table_names_to_delete:
                if self(table_name) is not None:
                    print(f"Deleting table: {table_name}")
                    
                    self(table_name)().delete()
                sleep(self.sleep_time_between_table_ops / 5)
            
            # check if there are ROIs dir in recording dir with Pre and Raw and delete it if so
            
            roi_dir = os.path.join(self.userinfo['data_dir'], str(self.data_subfolders["day"]),str(self.data_subfolders["experiment"]),'ROIs')
            if os.path.exists(roi_dir) and at_processing_stage != 'data_extraction':
                print(f"Deleting ROIs directory: {roi_dir}")
                user_ok = input(f"Are you sure you want to delete this directory? (y/n): ")
                if user_ok.lower() == 'y':
                    shutil.rmtree(roi_dir, ignore_errors=False)

            
