# Import GUI components
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .utils_gui import plot_stack_and_rois
from djimaging.utils.dj_utils import get_primary_key
import datajoint as dj

class PipelineGUIBase:
    """Base class for all GUI components"""
    def __init__(self, openretina_wrapper):
        self.wrapper = openretina_wrapper
        self.widgets = {}
        self.outputs = {}
    
    def create_layout(self):
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement create_layout")
    
    def update_data(self):
        """Override in subclasses"""
        raise NotImplementedError("Subclasses must implement update_data")



class VisualizationGUI(PipelineGUIBase):
    """Visualization GUI for the pipeline with ROI interaction"""
    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.current_visualization = None
        self.selected_roi = None
        self.roi_data = None
        self.field_data = None
        self.current_fig = None  # Store current figure for updates
        
        # Create the widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create all widgets for the GUI"""
        # Visualization outputs
        self.outputs['roi_plot'] = widgets.Output()
        self.outputs['quality_text'] = widgets.Output()
        self.outputs['celltype_text'] = widgets.Output()
        self.outputs['chirp_plot'] = widgets.Output()
        self.outputs['temporal_sta'] = widgets.Output()
        self.outputs['spatial_sta'] = widgets.Output()
        
        # Button to refresh/update visualization
        self.widgets['refresh'] = widgets.Button(
            description='Refresh Data',
            button_style='primary',
            tooltip='Update the visualization with current data'
        )
        
        # ROI selector dropdown (will be populated with available ROIs) - now synchronized with clicks
        self.widgets['roi_selector'] = widgets.Dropdown(
            options=[],
            description='Select ROI:',
            disabled=True
        )
        
        # Add instruction text
        self.widgets['instruction'] = widgets.HTML(
            value="<i>Click on an ROI in the visualization to select it, or use the dropdown menu.</i>"
        )
        
        # Wire up the callbacks
        self.widgets['refresh'].on_click(self.on_refresh_click)
        self.widgets['roi_selector'].observe(self.on_roi_selection_change, names='value')
        
    def on_roi_selection_change(self, change):
        """Handle ROI selection changes from dropdown"""
        if change['new'] is not None and change['new'] != self.selected_roi:
            self.selected_roi = change['new']
            self.update_roi_info_panels()
            self.update_roi_visualization()
    
    def on_roi_click_selection(self, roi_id):
        """Handle ROI selection from clicking on visualization"""
        if roi_id != self.selected_roi:
            self.selected_roi = roi_id
            # Update dropdown to match click selection
            self.widgets['roi_selector'].value = roi_id
            self.update_roi_info_panels()
            self.update_roi_visualization()

    def on_refresh_click(self, button):
        """Handle refresh button clicks"""
        self.load_data_from_wrapper()
        self.update_roi_dropdown()
        self.update_roi_visualization()
        if self.selected_roi is not None:
            self.update_roi_info_panels()
    
    def load_data_from_wrapper(self,key: Optional[Dict[str, Any]] = None):
        """Load data from the OpenRetina wrapper"""
    
        roi_mask_table = self.wrapper('RoiMask')()
        presentation_table = self.wrapper('Presentation')()
        field_table = self.wrapper('Field')()
        userinfo_table = self.wrapper('UserInfo')()

        key = get_primary_key(table=roi_mask_table.proj() * presentation_table.proj(), key=key)
        npixartifact, scan_type = (field_table & key).fetch1('npixartifact', 'scan_type')
        data_name, alt_name = (userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        try:
            alt_ch_average = (presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1(
                'ch_average')
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)

        
        # Get field data
        self.field_data = {
                'main_ch_average': main_ch_average,
                'alt_ch_average': alt_ch_average,
                'scan_type': scan_type,
                'npixartifact': npixartifact,
            }
        
        # Get ROI mask
        roi_mask = (roi_mask_table.RoiMaskPresentation & key).fetch1('roi_mask')

        self.roi_data = {
            'roi_mask': roi_mask,
            'roi_ids': - np.unique(roi_mask[roi_mask < 0])
        }
        
        print(f"Loaded {len(self.roi_data['roi_ids'])} ROIs")

    
    def update_roi_dropdown(self):
        """Update the ROI selector dropdown with available ROIs"""
        if self.roi_data is not None:
            roi_options = [(f"ROI {roi_id}", roi_id) for roi_id in self.roi_data['roi_ids']]
            self.widgets['roi_selector'].options = roi_options
            self.widgets['roi_selector'].disabled = False
            if len(roi_options) > 0:
                self.widgets['roi_selector'].value = roi_options[0][1]
                self.selected_roi = roi_options[0][1]
        else:
            self.widgets['roi_selector'].options = []
            self.widgets['roi_selector'].disabled = True
            self.selected_roi = None
    
    def update_roi_visualization(self):
        """Update the ROI mask visualization"""
        with self.outputs['roi_plot']:
            self.outputs['roi_plot'].clear_output(wait=True)
            
            
            self.current_fig, ax = plot_stack_and_rois(
                main_ch_average=self.field_data['main_ch_average'],
                alt_ch_average=self.field_data['alt_ch_average'],
                roi_mask=self.roi_data['roi_mask'],
                roi_ch_average=self.field_data['main_ch_average'],
                scan_type=self.field_data['scan_type'],
                npixartifact=self.field_data['npixartifact'],
                highlight_roi= [self.selected_roi],
                figsize=(8, 6)
            )
            
            # Add title with instructions
            title = 'ROI Visualization - Click on ROIs to Select'
            if self.selected_roi is not None:
                title += f' (Selected: ROI {self.selected_roi})'
            ax.set_title(title)
            
            # Simplified and more robust click handler
            def on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                
                try:
                    # Get the ROI mask data
                    roi_mask = self.roi_data['roi_mask']
                    
                    # Get the extent used in the plot_stack_and_rois function
                    if self.field_data['scan_type'] == 'xy':
                        extent = (roi_mask.shape[0] / 2, -roi_mask.shape[0] / 2,
                                roi_mask.shape[1] / 2, -roi_mask.shape[1] / 2)
                    elif self.field_data['scan_type'] == 'xz':
                        extent = (roi_mask.shape[0] / 2, -roi_mask.shape[0] / 2,
                                -roi_mask.shape[1] / 2, roi_mask.shape[1] / 2)
                    
                    # Convert click coordinates to array indices
                    x_range = extent[1] - extent[0]  # right - left
                    y_range = extent[3] - extent[2]  # top - bottom
                    
                    # Map to array coordinates
                    x_norm = (event.xdata - extent[0]) / x_range
                    y_norm = (event.ydata - extent[2]) / y_range
                    
                    # Convert to array indices (note: roi_mask is transposed in plot function)
                    x_idx = int(x_norm * roi_mask.shape[1])
                    y_idx = int((1 - y_norm) * roi_mask.shape[0])  # Flip y because of origin='lower'
                    
                    # Check bounds
                    if (0 <= x_idx < roi_mask.shape[1] and 0 <= y_idx < roi_mask.shape[0]):
                        clicked_roi = roi_mask[y_idx, x_idx]
                        print(f"Clicked at pixel ({x_idx}, {y_idx}), ROI value: {clicked_roi}")
                        
                        # ROI mask uses negative values, so convert to positive
                        if clicked_roi < 0:  # Valid ROI
                            roi_id = abs(clicked_roi)
                            print(f"Selected ROI {roi_id}")
                            self.on_roi_click_selection(roi_id)
                        else:
                            print("Clicked on background")
                    else:
                        print(f"Click outside bounds: ({x_idx}, {y_idx})")
                        
                except Exception as e:
                    print(f"Error in click handler: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Connect the event with proper backend handling
            try:
                # Disconnect any existing handlers first
                self.current_fig.canvas.mpl_disconnect('all')
            except:
                pass
                
            # Connect the new handler
            cid = self.current_fig.canvas.mpl_connect('button_press_event', on_click)
            print(f"Connected click handler with ID: {cid}")
            
            plt.tight_layout()
            plt.show()
            
            # Re-enable interactive mode
            plt.ion()
                

    def update_roi_info_panels(self):
        """Update all ROI information panels"""
        if self.selected_roi is None:
            return
        
        self.update_quality_indices()
        self.update_celltype_info()
        self.update_chirp_traces()
        self.update_temporal_sta()
        self.update_spatial_sta()
    
    def update_quality_indices(self):
        """Update quality indices text panel"""
        with self.outputs['quality_text']:
            self.outputs['quality_text'].clear_output(wait=True)

            
            quality_data = self.get_quality_data(self.selected_roi)

            quality_text = "" 
            for key, value in quality_data.items():
                quality_text += f"{key}: {value:.2f}\n"

    def update_celltype_info(self):
        """Update cell type information panel"""
        with self.outputs['celltype_text']:
            self.outputs['celltype_text'].clear_output(wait=True)

                
            cell_type_data: Dict[str, np.ndarray] = self.get_celltype_data(self.selected_roi)

            celltype_text = "Top 3 Types:\n"
            for ctype, prob in zip(cell_type_data["top_3_groups"], cell_type_data["top_3_scores"]):
                celltype_text += f"{ctype}: {prob:.2f}\n"
            
            print(celltype_text)

    def update_chirp_traces(self):
        """Update chirp traces plot"""
        with self.outputs['chirp_plot']:
            self.outputs['chirp_plot'].clear_output(wait=True)
           
            chirp_data = self.get_chirp_data(self.selected_roi)

    
    def update_temporal_sta(self):
        """Update temporal STA plot"""
        with self.outputs['temporal_sta']:
            self.outputs['temporal_sta'].clear_output(wait=True)
            
          
            temporal_sta_data = self.get_temporal_sta_data(self.selected_roi)
               
    
    def update_spatial_sta(self):
        """Update spatial STA plot"""
        with self.outputs['spatial_sta']:
            self.outputs['spatial_sta'].clear_output(wait=True)
           
            spatial_sta_data = self.get_spatial_sta_data(self.selected_roi)
            
    def get_quality_data(self, roi_id):
        """Get quality indices for a specific ROI"""
        chirp_qi_table = self.wrapper('ChirpQI')()
        ori_dir_qi_table = self.wrapper('OsDsIndexes')()
        print(roi_id)

        d_qi = (ori_dir_qi_table & dict(roi_id=roi_id)).fetch("d_qi").item()
        qidx_chirp = (chirp_qi_table & dict(roi_id=roi_id)).fetch("qidx").item()

        return {"d_qi": d_qi, "qidx_chirp": qidx_chirp}

    def get_celltype_data(self, roi_id):
        """Get celltype assignment data for a specific ROI"""
        
        if roi_id is None:
            return None

        celltype_table = self.wrapper('CelltypeAssignment')()
        celltype_data = (celltype_table & dict(roi_id=roi_id)).fetch1()
        confidence_scores = celltype_data['confidence']

        # Get top 3 with scores
        top_3_indices = confidence_scores.argsort()[-3:][::-1]
        top_3_groups = top_3_indices + 1 # assume index based group assignment
        top_3_scores = confidence_scores[top_3_indices]

        assert celltype_data["celltype"] == top_3_groups[0], "Top group does not match celltype assignment"

        return {"top_3_groups": top_3_groups, "top_3_scores": top_3_scores}
        
    def get_chirp_data(self, roi_id):
        """Get chirp response data for a specific ROI"""
        
        return None
    
    def get_temporal_sta_data(self, roi_id):
        """Get temporal STA data for a specific ROI"""

        return None
    
    def get_spatial_sta_data(self, roi_id):
        """Get spatial STA data for a specific ROI"""

        return None
    
    def create_layout(self):
        """Create the main layout for the visualization GUI"""
        # Control panel at the top
        controls = widgets.VBox([
            widgets.HBox([
                self.widgets['refresh'],
                self.widgets['roi_selector']
            ]),
            self.widgets['instruction']
        ])
        
        # Left side: ROI visualization (larger)
        left_panel = widgets.VBox([
            widgets.HTML("<h3>ROI Visualization</h3>"),
            self.outputs['roi_plot']
        ])
        
        # Right side: ROI information panels
        # Top row: Quality indices and cell types
        info_top = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h4>Quality Indices</h4>"),
                self.outputs['quality_text']
            ]),
            widgets.VBox([
                widgets.HTML("<h4>Cell Types</h4>"),
                self.outputs['celltype_text']
            ])
        ])
        
        # Middle row: Chirp traces
        info_middle = widgets.VBox([
            widgets.HTML("<h4>Chirp Response</h4>"),
            self.outputs['chirp_plot']
        ])
        
        # Bottom row: STA plots
        info_bottom = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h4>Temporal STA</h4>"),
                self.outputs['temporal_sta']
            ]),
            widgets.VBox([
                widgets.HTML("<h4>Spatial STA</h4>"),
                self.outputs['spatial_sta']
            ])
        ])
        
        # Right panel combining all info panels
        right_panel = widgets.VBox([
            info_top,
            info_middle,
            info_bottom
        ])
        
        # Main layout with better proportions for click interaction
        main_content = widgets.HBox([
            left_panel,
            right_panel
        ])
        
        # Full layout with controls at top
        full_layout = widgets.VBox([
            widgets.HTML("<h2>Pipeline ROI Analyzer</h2>"),
            controls,
            main_content
        ])
        
        return full_layout
    
    def start_gui(self):
        """Start the GUI and load initial data"""
        # Load data on startup
        self.load_data_from_wrapper()
        self.update_roi_dropdown()
        self.update_roi_visualization()
        if self.selected_roi is not None:
            self.update_roi_info_panels()
        
        return self.create_layout()