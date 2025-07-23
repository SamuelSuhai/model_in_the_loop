import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple

from djimaging.utils.math_utils import normalize_zero_one


def plot_stack_and_rois(main_ch_average, scan_type='xy', roi_mask=None, roi_ch_average=None, npixartifact=0,
               title='', figsize=(6, 4), highlight_roi=[], fig=None, ax=None, gamma=0.5) -> Tuple[plt.Figure, plt.Axes]:
    if roi_mask is None or roi_mask.size == 0:
        raise ValueError("roi_mask is required for this simplified plot function")

    if roi_ch_average is None:
        roi_ch_average = main_ch_average

    # normalize and gamma correction
    roi_ch_average = normalize_zero_one(roi_ch_average) ** gamma
    main_ch_average = normalize_zero_one(main_ch_average) ** gamma

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    fig.suptitle(title)

    if scan_type == 'xy':
        ax.set(xlabel='relY [pixel]', ylabel='relX [pixel]')
        extent = (main_ch_average.shape[0] / 2, -main_ch_average.shape[0] / 2,
                  main_ch_average.shape[1] / 2, -main_ch_average.shape[1] / 2)
    elif scan_type == 'xz':
        ax.set(xlabel='relY [pixel]', ylabel='relZ [pixel]')
        extent = (main_ch_average.shape[0] / 2, -main_ch_average.shape[0] / 2,
                  -main_ch_average.shape[1] / 2, main_ch_average.shape[1] / 2)
    else:
        raise ValueError(f'Unknown scan_type: {scan_type}')

    rois = -roi_mask.astype(float).T

    _roi_ch_average = roi_ch_average.copy()
    _roi_ch_average[:npixartifact] = np.nan

    ax.imshow(_roi_ch_average.T, cmap='viridis', origin='lower', extent=extent)
    rois_us = np.repeat(np.repeat(rois, 10, axis=0), 10, axis=1)

    rois_to_plot = np.unique(rois[rois > 0])

    
    # roi_us_bool = rois_us > 0
    # ax.contour(rois_us, extent=extent, levels= list(rois_to_plot), alpha=0.8, cmap= 'jet', linewidths=0.5)

    # # plot filled special one
    # ax.contourf(rois_us == highlight_roi, extent=extent, levels=[1 - 1e-3, 1 + 1e-3], alpha=0.3, colors='red')

    for roi in rois_to_plot:
        bool_mask = rois_us == roi
        _rois_us = bool_mask.astype(int) 

        if roi in highlight_roi:
            color = 'red'
        else:
            color = 'blue'

        ax.contour(_rois_us, extent=extent, levels=[1 - 1e-3], alpha=0.8, colors=color)
        
        y_coords, x_coords = np.where(bool_mask)
        
        if len(y_coords) > 0 and len(x_coords) > 0:
            # Calculate center in upsampled coordinates
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            
            # Convert to plot coordinates using the extent
            x_plot = extent[0] + (extent[1] - extent[0]) * center_x / _rois_us.shape[1]
            y_plot = extent[2] + (extent[3] - extent[2]) * center_y / _rois_us.shape[0]
            
            # Add text with ROI ID at the center
            ax.text(x_plot, y_plot, f"{int(roi)}", 
                   ha='center', va='center', color="white", fontweight='bold', fontsize=8,
                   )#bbox=dict(facecolor=color, alpha=0.2, pad=1, boxstyle='round'))



    plt.tight_layout()
    return fig, ax