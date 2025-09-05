import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional
from .base import DJComputeWrapper, DJTableHolder
from djimaging.utils import plot_utils
from model_in_the_loop.utils.datajoiont_utils import get_rois_in_field_restriction_str
from djimaging.utils.dj_utils import get_primary_key

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

    def clear_tables(self, field_key,safemode=True) -> None:
        """
        Clear tables related to STA analysis.
        """
        tables_to_clear = [
            'DNoiseTrace',
            'STA',
            'SplitRF',
            'FitGauss2DRF',
        ]

        for table_name in tables_to_clear:
            (self.dj_table_holder(table_name)() & field_key).delete(safemode=safemode)
    
    def compute_analysis(self, 
                         field_key = {},
                         roi_id_subset: Optional[List[int]] = None ,
                         progress_callback: Optional[Callable] = None) -> None:
        """
        Compute the STA analysis.
        """
        complete_restriction = get_rois_in_field_restriction_str(field_key, roi_id_subset)
        
        if progress_callback is not None:
            progress_callback(0)
        
        self.dj_table_holder('DNoiseTrace')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        

        if progress_callback is not None:
            progress_callback(30)
        self.dj_table_holder('STA')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

        if progress_callback is not None:
            progress_callback(80)
        self.dj_table_holder('SplitRF')().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)

        self.dj_table_holder("FitGauss2DRF")().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
        

        if progress_callback is not None:
            progress_callback(100)

    def check_requirements(self, 
                           field_key,
                           roi_id_subset: Optional[List[int]] = None ,) -> None:
        """
        Check if the required tables are populated in the database.
        """
        complete_restriction = get_rois_in_field_restriction_str(field_key, roi_id_subset)

        for table_name in self.requires_tables:
            if len(self.dj_table_holder(table_name)() & complete_restriction) == 0:
                
                # populate the necessary tables
                self.dj_table_holder(table_name)().populate(complete_restriction,processes=self.dj_table_holder.multiprocessing_threads, display_progress=True)
    
    def get_roi_ids_passing_criterion(self, 
                                      field_key: Dict[str, Any], 
                                      rf_qidx_min: float = 0.5,) -> List[int]:
        """
        Looks at the FitGauss2DRF table and returns the roi_ids that have a qidx value above the given threshold.
        """

        fit_gauss_table = self.dj_table_holder('FitGauss2DRF')() & field_key

        if len(fit_gauss_table) == 0:
            raise ValueError("FitGauss2DRF table is empty for the given field_key")
        
        # Fetch all roi_ids and their corresponding qidx values
        roi_ids, qidx_values = fit_gauss_table.fetch('roi_id', 'rf_qidx')

        # Filter roi_ids based on the criteria
        passing_roi_ids = [
            roi_id for roi_id, qidx in zip(roi_ids, qidx_values, strict=True) if qidx >= rf_qidx_min
        ]

        return passing_roi_ids


    def plot_roi_overview(self, roi_keys: List[Dict[str, Any]]) -> None:
        pass
        
    def plot1(self,roi_id: int,field_key={},axs = None, show = True) -> None:

        restricted_split_rf = (self.dj_table_holder('SplitRF')() & field_key & {'roi_id': roi_id})
        if len(restricted_split_rf) == 0:
            print(f"No RF computed for roi_id {roi_id}.")
            return
        elif len(restricted_split_rf) > 1:
            raise ValueError(f"Expected exactly one SplitRF for roi_id {roi_id}, found {len(restricted_split_rf)}")

        # plot it and 
        key = get_primary_key(table=restricted_split_rf, key=None)

        rf_time = restricted_split_rf.fetch1_rf_time(key=key)
        srf, trf, peak_idxs = (restricted_split_rf & key).fetch1("srf", "trf", "trf_peak_idxs")
        
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex='col')

        ax = axs[0]
        plot_utils.plot_srf(srf, ax=ax)
        ax.set_title(f'sRF for ROI {roi_id}')

        # add x mark at the peak position in index coordinates

        fit_restricted = (self.dj_table_holder("FitGauss2DRF")() & field_key & {'roi_id': roi_id})
        assert len(fit_restricted) == 1, f"Expected exactly one FitGauss2DRF for roi_id {roi_id}, found {len(fit_restricted)}"
        gauss_fit = fit_restricted.fetch1("srf_params")
        
        # the flip is because the axis 0 is in QDSpy dense noise x but in the fit tables its y
        x = gauss_fit['y_mean']
        y = gauss_fit['x_mean']
        
        # flip x and y becaus in QDSpy dense noise the axis 0 is x so I called it x but its axis 0. 
        ax.plot(y,x, 'x', color='black', markersize=10, label='Mean Position')
        ax.legend()

        ax = axs[1]
        plot_utils.plot_trf(trf, t_trf=rf_time, peak_idxs=peak_idxs, ax=ax)

        if show:
            plt.show()
