import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Union, Tuple

from djimaging.utils.math_utils import normalize_zero_one


def plot_stack_and_rois(main_ch_average, alt_ch_average, scan_type='xy', roi_mask=None, roi_ch_average=None, npixartifact=0,
               title='', figsize=(6, 4), highlight_roi=[], fig=None, ax=None, gamma=1.) -> Tuple[plt.Figure, plt.Axes]:
    if roi_mask is None or roi_mask.size == 0:
        raise ValueError("roi_mask is required for this simplified plot function")

    if roi_ch_average is None:
        roi_ch_average = main_ch_average

    # normalize and gamma correction
    roi_ch_average = normalize_zero_one(roi_ch_average) ** gamma

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
    vmin = np.min(rois)
    vmax = np.max(rois)


    rois_to_plot = np.unique(rois[rois > 0])

    

    for roi in rois_to_plot:
        _rois_us = (rois_us == roi).astype(int) * roi

        if roi in highlight_roi:
            print(f"Highlighting ROI {roi}")
            ax.imshow(_rois_us, extent=extent, vmin=vmin, vmax=vmax, alpha=0.5, cmap='jet')
        else:
            ax.contour(_rois_us, extent=extent, vmin=vmin, vmax=vmax, levels=[roi - 1e-3], alpha=0.8, cmap='jet')

    plt.tight_layout()
    return fig, ax