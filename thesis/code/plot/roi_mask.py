import numpy as np
import seaborn as sns
from djimaging.utils.image_utils import rotate_image, resize_image,color_image


def plot_roi_mask_on_stack(ax, 
                           roi_mask, 
                           ch_average, 
                           roi_ids, 
                           gamma=0.4,
                            upscale=2,
                            roi_color=(1,0,0),
                            roi_alpha = 0.5):

    image = ch_average.copy()
    w,h = image.shape[0], image.shape[1]
    assert w == 64 and h == 64, "input image should be 64x64"
    assert (roi_mask >= 0).all(), "roi_mask should be non-negative"

    roi_mask = roi_mask.copy()
    roi_mask = np.repeat(np.repeat(roi_mask, upscale, axis=0), upscale, axis=1)

    scale = (upscale,upscale)
    output_shape = np.ceil(np.asarray(image.shape) * np.asarray(scale)).astype('int')
    resized_image = resize_image(image, output_shape=output_shape, order=1)

    color_img = color_image(image, cmap='gray', gamma=gamma,alpha=255).transpose(1,0,2)

    extent = (0, 64, 0, 64)
    cmap = sns.color_palette("gray", as_cmap=True)
    cmap.set_bad('w')
    ax.imshow(color_img, cmap=cmap, origin='lower', extent=extent)

    # Create a combined mask for all ROIs
    combined_mask = np.zeros_like(roi_mask, dtype=bool)
    for roi_id in roi_ids:
        is_roi = (roi_mask == roi_id)
        combined_mask = combined_mask | is_roi
    
    # Create RGBA overlay for all ROIs with the same color
    roi_overlay = np.zeros((*combined_mask.shape, 4), dtype=float)
    roi_overlay[..., :3] = np.array(roi_color)  # RGB channels
    roi_overlay[..., 3] = combined_mask.astype(float) * 0.5  # Alpha channel (50% transparency)
    
    # Plot all ROIs
    ax.imshow(roi_overlay.transpose(1, 0, 2), origin='lower', extent=extent,alpha=roi_alpha)

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
