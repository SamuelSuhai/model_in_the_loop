from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .base import DJComputeWrapper, DJTableHolder
from djimaging.utils import plot_utils


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
        self.dj_table_holder('ChirpQI')().populate(field_key,display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)
            
   
        self.dj_table_holder('OsDsIndexes')().populate(field_key,display_progress=True, processes=self.dj_table_holder.multiprocessing_threads)

    def add_celltype_assignments(self, field_key) -> None:
        """
        Populate the Baden16Traces and CelltypeAssignment tables for cell type assignments.
        """
        self.dj_table_holder('Baden16Traces')().populate(field_key, processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        self.dj_table_holder('CelltypeAssignment')().populate(field_key, processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

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

    def get_roi_ids_passing_criterion(self, 
                                      field_key: Dict[str, Any],
                                      d_qi_min: float, 
                                      qidx_min: float,
                                      celltypes: List[int] = list(range(99)),
                                      classifier_confidence: float = 0.0,
                                      verbose = True) -> List[int]:
        """
        For a field key it looks in the tables ChirpQI and OsDsIndexes for the d_qi and qidx values. Takes rois that pass either chrip or ori_dir quality index as passing.
        """
         
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        ori_dir_qi_table = self.dj_table_holder('OsDsIndexes')() & field_key
        n_rois_before = len(chirp_qi_table)

        if len(chirp_qi_table) == 0 or len(ori_dir_qi_table) == 0:
            raise ValueError("ChirpQI or OsDsIndexes table is empty for the given field_key")
        
        # Fetch all roi_ids and their corresponding d_qi and qidx values
        roi_ids_chirp, qidx_values = chirp_qi_table.fetch('roi_id', 'qidx')
        chirp_data_dict = {roi_id: qidx for roi_id, qidx in zip(roi_ids_chirp, qidx_values)}
        
        roi_ids_mb,d_qi_values = ori_dir_qi_table.fetch('roi_id', 'd_qi')
        ori_dir_data_dict = {roi_id: d_qi for roi_id, d_qi in zip(roi_ids_mb, d_qi_values)}

        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key
        if len(celltype_table) == 0:
            raise ValueError("CelltypeAssignment table is empty for the given field_key")
        
        # create a criterion
        roi_ids_celltype,celltype_data,confidence_scors = celltype_table.fetch('roi_id','celltype','max_confidence')
        celltype_data_dict = {roi_id: celltype for roi_id, celltype in zip(roi_ids_celltype, celltype_data)}
        
        #  get the confidence of assigned group which is the max 
        confidence_data_dict = {roi_id: confidence for roi_id, confidence in zip(roi_ids_celltype, confidence_scors)}
        # if verbose:
        #     print(f"Number after filtering rois based on classifier confidence {sum(confidence_scors > classifier_confidence)}.")
        #     print(f"Number after filtered rois based on celltypes {sum([t in celltypes for t in celltype_data])}.")
        #     print(f"Number after filtered rois based on Chirp MB QI {sum((qidx_values >qidx_min) | (d_qi_values > d_qi_min) )}.")

        # Filter roi_ids based on the criteria
        passing_roi_ids = []
        for roi_id in np.unique(np.concatenate([roi_ids_mb, roi_ids_chirp, roi_ids_celltype])):
            d_qi = ori_dir_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            qidx = chirp_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            confidence = confidence_data_dict.get(roi_id, 0.0)  # Default to 0.0 if not found
            celltype = celltype_data_dict.get(roi_id, -1) 
            
            if (d_qi >= d_qi_min or qidx >= qidx_min) and (confidence >= classifier_confidence) and celltype in celltypes:
                passing_roi_ids.append(roi_id)

        n_rois_after = len(passing_roi_ids)
        if verbose:
            print(f"Found {n_rois_after} rois passing the criterion out of {n_rois_before} rois.\
                ({d_qi_min=}, chrip {qidx_min=}, {celltypes=}, {classifier_confidence=})")
        
        # turn int 
        passing_roi_ids =list(map(int,passing_roi_ids))

        return passing_roi_ids
    
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

        snippets_restricted = snippets_table & single_averages_table
        if len(snippets_restricted) == 0 or len(single_averages_table) == 0:
            print(f"No data found for ROI {roi_id} and stim {stim_name}.")
            return
        

        snippets_t0, snippets_dt, snippets, triggertimes_snippets = snippets_restricted.fetch1(
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
            
            # populate the necessary tables
            self.dj_table_holder(table_name)().populate(field_key,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
            if len(self.dj_table_holder(table_name)() & field_key) == 0:
                raise ValueError(f"Required table {table_name} is empty for the given field_key after population. \
                                    Please ensure that the prerequisite data is available. If you enconter this in the GUI, make sure you \
                                    Hit insert to db first.")

    def get_roi2rgb_and_alpha_255_map(self,
                                      field_key: Dict[str, Any],
                                      all_roi_ids: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Get two mappings: one for roi to rgb based on celltype and one roi to alpha based on chirpQI.
        all_roi_ids is a list of roi_ids that are in the canvas so we cover all rois in the cavas with the color code.

        """
        
        # get celltype for each roi 
        celltype_table = self.dj_table_holder('CelltypeAssignment')() & field_key
        celltype_data = celltype_table.fetch('roi_id', 'celltype',as_dict=True)
        roi2rgb255 = {data['roi_id']: self.g_to_rgb255(data['celltype']) for data in celltype_data}

        # get chirpQI for each roi
        chirp_qi_table = self.dj_table_holder('ChirpQI')() & field_key
        chirp_qi_data = chirp_qi_table.fetch('roi_id', 'qidx', as_dict=True)
        roi2alpha = {data['roi_id']: self.qi2alpha255(data['qidx']) for data in chirp_qi_data}

        # ensure all roi_ids are covered
        for roi_id in all_roi_ids:
            if roi_id not in roi2rgb255:
                # default color for missing roi_ids
                roi2rgb255[roi_id] = np.array([128, 128, 128])
            if roi_id not in roi2alpha:
                # default alpha for missing roi_ids
                roi2alpha[roi_id] = self.qi2alpha255(0.0)
        

        return roi2rgb255, roi2alpha
         
    def clear_tables(self, field_key,safemode=True) -> None:
        """
        Clear tables related to quality metrics and cell type assignment.
        """
        tables_to_clear = [
            'ChirpQI',
            'OsDsIndexes',
            'Baden16Traces',
            'CelltypeAssignment',
        ]

        for table_name in tables_to_clear:
            (self.dj_table_holder(table_name)() & field_key).delete(safemode=safemode)


    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        """
        Plot an overview of the rois of the wrapper.
        """
        pass
