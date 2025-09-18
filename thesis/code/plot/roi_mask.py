import numpy as np
import seaborn as sns
from djimaging.utils.image_utils import rotate_image, rescale_image


def plot_roi_mask_on_stack(ax, roi_mask, ch_average, roi_id, angle_deg=0., npixartifact=0, upscale=5, pad_square=True):
    if pad_square:
        w, h = ch_average.shape
        assert w == 64
        assert h == 16

        w, h = roi_mask.shape
        assert w == 64, roi_mask.shape
        assert h == 16, roi_mask.shape

        ch_average_ = np.full((64, 64), np.nan)
        ch_average_[:, 24:40] = ch_average
        ch_average = ch_average_

        roi_mask_ = np.full((64, 64), np.nan)
        roi_mask_[:, 24:40] = roi_mask
        roi_mask = roi_mask_

    roi_mask = -roi_mask.copy().astype(float)
    roi_mask[:npixartifact, :] = -1
    roi_mask = np.repeat(np.repeat(roi_mask, upscale, axis=0), upscale, axis=1)
    roi_mask[roi_mask == -1] = np.nan
    roi_mask = rotate_image(roi_mask.copy(), angle_deg, order=1)

    ch_average = rotate_image(rescale_image(ch_average.copy(), upscale, order=0), angle_deg, order=1)

    extent = (0, 64, 0, 64)
    cmap = sns.color_palette("gray", as_cmap=True)
    cmap.set_bad('w')
    ax.imshow(ch_average.T, cmap=cmap, origin='lower', extent=extent)
    ax.contour((roi_mask == roi_id).T, levels=[0.999], colors=['c'], origin='lower', linewidths=[0.5], extent=extent)
    ax.axis('off')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)


def plot_roi_mask_filled(ax, roi_mask, order=None, **kwargs):
    roi_mask = roi_mask.copy()
    roi_mask[roi_mask == 1] = 0
    roi_mask = np.abs(roi_mask)

    if order is not None:
        mapping = {old_id: new_id for new_id, old_id in enumerate(order)}
        mapping[0] = 0
        roi_mask = (np.vectorize(mapping.get))(roi_mask)

    roi_mask = roi_mask.astype(float)
    roi_mask[roi_mask == 0] = np.nan

    ax.imshow(roi_mask, vmin=0, vmax=np.nanmax(roi_mask), origin='lower', **kwargs, interpolation='None')
    ax.axis('off')
