import os
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# Color palette
TEXT_WIDTH = 5.2  # inch
FULLPAGE_WIDTH = 7.5  # inch
POSTERFIG_WIDTH = 14.46  # inch
FIGURE_HEIGHT = 1.2  # inch
FIGURE_MAX_HEIGHT = 8.75  # inch


def set_default_params(notebook_dpi=120, kind='paper',
                       text_width=None, fullpage_width=None, figure_height=None, figure_max_height=None):
    if text_width is not None:
        global TEXT_WIDTH
        TEXT_WIDTH = text_width
    if fullpage_width is not None:
        global FULLPAGE_WIDTH
        FULLPAGE_WIDTH = fullpage_width
    if figure_height is not None:
        global FIGURE_HEIGHT
        FIGURE_HEIGHT = figure_height
    if figure_max_height is not None:
        global FIGURE_MAX_HEIGHT
        FIGURE_MAX_HEIGHT = figure_max_height
    set_rc_params(notebook_dpi=notebook_dpi, kind=kind)


def set_rc_params(notebook_dpi=120, kind='paper'):
    sns.set_context('paper')
    sns.set_style('ticks')
    plt.style.use(f'{os.path.abspath(os.path.dirname(__file__))}/mplstyles/{kind}.mplstyle')
    plt.rcParams['figure.figsize'] = (FULLPAGE_WIDTH, FULLPAGE_WIDTH / 3)
    plt.rcParams['figure.dpi'] = notebook_dpi  # only affects the notebook

def figsize_from_frac(w,h):
    return(FULLPAGE_WIDTH*w, FULLPAGE_WIDTH*h)

def set_rc_poster_params():
    set_rc_params()
    plt.rcParams['figure.figsize'] = (POSTERFIG_WIDTH, POSTERFIG_WIDTH / 2)


def show_saved_figure(fig):
    fig.savefig('.temp.jpg', dpi=600)
    plt.figure(figsize=(10, 10), facecolor=(0.5, 0.5, 0.5, 0.5))

    im = plt.imread('.temp.jpg')

    if np.any(im[0, :] < 255) or np.any(im[-1, :] < 255) or np.any(im[:, 0] < 255) or np.any(im[:, -1] < 255):
        print('Warning: Figure is probably clipped!')

    plt.imshow(im, aspect='equal')
    plt.axis('off')
    plt.show()

    from os import remove as removefile
    removefile('.temp.jpg')


def tight_layout(h_pad=1, w_pad=1, rect=(0, 0, 1, 1), pad=None):
    """tigh layout with different default"""
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad, pad=pad or 2. / plt.rcParams['font.size'], rect=rect)


def subplots(nrows=1, ncols=4, xsize='text', ysizerow='auto', yoffsize=0.0, figsize='auto', **kwargs):
    """Like plt.subplots, but with auto size."""

    # Get auto fig size.
    if figsize == 'auto':
        if xsize == 'text':
            xsize = TEXT_WIDTH
        elif xsize == 'fullwidth':
            xsize = FULLPAGE_WIDTH
        else:
            assert isinstance(xsize, (float, int)), "x not in {'text', 'fullwidth', float}"

        if ysizerow == 'auto':
            ysizerow = FIGURE_HEIGHT

        ysize = ysizerow * nrows + yoffsize

        if ysize > FIGURE_MAX_HEIGHT:
            print(f"ysize: {ysize} > {FIGURE_MAX_HEIGHT}")

        figsize = (xsize, ysize)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor='w', **kwargs)
    return fig, axs


def auto_subplots(n_plots, max_ny_sb=20, max_nx_sb=4, xsize='text', ysizerow='auto', yoffsize=0.0,
                  allow_fewer=True, rm_unused=True, **kwargs):
    """Create subplots with auto infer n_rows and cols. Otherwise similar to subplots"""

    ncols = np.min([max_nx_sb, n_plots])
    nrows = np.min([max_ny_sb, int(np.ceil([n_plots / ncols]))])

    if not allow_fewer: assert ncols * nrows >= n_plots

    fig, axs = subplots(ncols=ncols, nrows=nrows, xsize=xsize, ysizerow=ysizerow, yoffsize=yoffsize, **kwargs)

    if n_plots > 1:
        if rm_unused and axs.size > n_plots:
            for i in np.arange(n_plots, axs.size):
                axs.flat[i].axis('off')

    return fig, axs


def iterate_axes(axs):
    """Make axes iterable, independent of type.
    axs (list of matplotlib axes or matplotlib axis) : Axes to apply function to.
    """

    if isinstance(axs, list):
        return axs
    elif isinstance(axs, np.ndarray):
        return axs.flatten()
    else:
        return [axs]


def int_format_ticks(axs, which='both'):
    """Use integer ticks for integers."""
    from matplotlib.ticker import FuncFormatter

    def int_formatter(x):
        if x.is_integer():
            return str(int(x))
        else:
            return f"{x:g}"

    formatter = FuncFormatter(int_formatter)
    if which in ['x', 'both']:
        for ax in iterate_axes(axs):
            ax.xaxis.set_major_formatter(formatter)
    if which in ['y', 'both']:
        for ax in iterate_axes(axs):
            ax.yaxis.set_major_formatter(formatter)


def scale_ticks(axs, scale, x=True, y=False):
    ticks = ticker.FuncFormatter(lambda xi, pos: '{0:g}'.format(xi * scale))
    for ax in iterate_axes(axs):
        if x:
            ax.xaxis.set_major_formatter(ticks)
        if y:
            ax.yaxis.set_major_formatter(ticks)


def move_xaxis_outward(axs, scale=3):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs):
        ax.spines['bottom'].set_position(('outward', scale))


def move_yaxis_outward(axs, scale=3):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs):
        ax.spines['left'].set_position(('outward', scale))


def adjust_log_tick_padding(axs, pad=2.1):
    """ Change tick padding for all log scaled axes.
    Parameters:
    axs (array or list of matplotlib axes) : Axes to apply function to.
    pad (float) : Size of padding.
    """

    for ax in iterate_axes(axs):
        if ax.xaxis.get_scale() == 'log':
            ax.tick_params(axis='x', which='major', pad=pad)
            ax.tick_params(axis='x', which='minor', pad=pad)

        if ax.yaxis.get_scale() == 'log':
            ax.tick_params(axis='y', which='major', pad=pad)
            ax.tick_params(axis='y', which='minor', pad=pad)


def set_labs(axs, xlabs=None, ylabs=None, titles=None, panel_nums=None, panel_num_space=0, panel_num_va='bottom',
             panel_num_pad=0, panel_num_y=None, panel_loc='left'):
    """Set labels and titles for all given axes.
    Parameters:

    axs : array or list of matplotlib axes.
        Axes to apply function to.

    xlabs, ylabs, titles : str, list of str, or None
        Labels/Titles.
        If single str, will be same for all axes.
        Otherwise should have same length as axes.

    """

    for i, ax in enumerate(iterate_axes(axs)):
        if xlabs is not None:
            if isinstance(xlabs, str):
                xlab = xlabs
            else:
                xlab = xlabs[i]
            ax.set_xlabel(xlab)

        if ylabs is not None:
            if isinstance(ylabs, str):
                ylab = ylabs
            else:
                ylab = ylabs[i]
            ax.set_ylabel(ylab)

        if titles is not None:
            if isinstance(titles, str):
                title = titles
            else:
                title = titles[i]
            ax.set_title(title)

        if panel_nums is not None:
            if panel_nums == 'auto':
                panel_num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
            elif isinstance(panel_nums, str):
                panel_num = panel_nums
            else:
                panel_num = panel_nums[i]
            ax.set_title(panel_num + panel_num_space * ' ', loc=panel_loc, fontweight='bold', ha='right',
                         va=panel_num_va,
                         pad=panel_num_pad, y=panel_num_y)


def left2right_ax(ax):
    """Create a twin axis, but remove all duplicate spines.
    Parameters:
    ax (Matplotlib axis) : Original axis to create twin from.
    Returns:
    ax (Matplotlib axis) : Twin axis with no duplicate spines.
    """

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax = ax.twinx()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def move_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0: box[0] += dx
        if dy != 0: box[1] += dy
        ax.set_position(box)


def change_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0:
            box[2] += dx
        if dy != 0:
            box[3] += dy
        ax.set_position(box)


def align_x_box(ax, ref_ax):
    """Change offset of box"""
    box = np.array(ax.get_position().bounds)
    ref_box = np.array(ref_ax.get_position().bounds)
    box[0] = ref_box[0]
    box[2] = ref_box[2]
    ax.set_position(box)


def align_box_left(ax, ref_ax):
    """Change offset of box"""
    box = np.array(ax.get_position().bounds)
    ref_box = np.array(ref_ax.get_position().bounds)
    box[0] = ref_box[0]
    ax.set_position(box)


def idx2color(idx, colorpalette='tab10', isscatter=False):
    if idx is None:
        c = (0, 0, 0)
    else:
        c = ColorConverter.to_rgb(sns.color_palette(colorpalette).as_hex()[idx])
    if isscatter:
        c = np.atleast_2d(np.asarray(c))
    return c


def text2mathtext(txt):
    txt = txt.replace('^', '}^\mathrm{')
    txt = txt.replace('_', '}_\mathrm{')
    txt = txt.replace(' ', '} \mathrm{')
    txt = txt.replace('-', '\mathrm{-}')
    return r"$\mathrm{" + txt + "}$"


def get_legend_handle(**kwargs):
    """Make method legend on given axis."""
    return Line2D([0], [0], **kwargs)


def get_legend_handles(markers, colors, lss, **kwargs):
    """Get handle for list of markers, colors and linestyles"""
    legend_handles = []
    for marker, color, ls in zip(markers, colors, lss):
        legend_handles.append(get_legend_handle(marker=marker, color=color, ls=ls, **kwargs))
    return legend_handles


def row_title(ax, title, pad=70, size='large', ha='left', va='center', **kwargs):
    """Create axis row title using annotation"""
    ax.annotate(title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=size, ha=ha, va=va, **kwargs)


def grid(ax, axis='both', major=True, minor=False, **kwargs):
    """make grid on axis"""
    for axi in iterate_axes(ax):
        if major:
            axi.grid(True, axis=axis, which='major', alpha=.3, c='k',
                     lw=plt.rcParams['ytick.major.width'], zorder=-10000, **kwargs)
        if minor:
            axi.grid(True, axis=axis, which='minor', alpha=.3, c='gray',
                     lw=plt.rcParams['ytick.minor.width'], zorder=-20000, **kwargs)


def make_share_xlims(axs, symmetric=False, xlim=None):
    """Use xlim lower and upper bounds for all axes."""
    if xlim is None:
        xlb = np.min([ax.get_xlim()[0] for ax in iterate_axes(axs)])
        xub = np.max([ax.get_xlim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            xlim = (xlb, xub)
        else:
            xlim = (-np.max(np.abs([xlb, xub])), np.max(np.abs([xlb, xub])))
    for ax in iterate_axes(axs): ax.set_xlim(xlim)


def make_share_ylims(axs, symmetric=False, ylim=None):
    """Use ylim lower and upper bounds for all axes."""
    if ylim is None:
        ylb = np.min([ax.get_ylim()[0] for ax in iterate_axes(axs)])
        yub = np.max([ax.get_ylim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            ylim = (ylb, yub)
        else:
            ylim = (-np.max(np.abs([ylb, yub])), np.max(np.abs([ylb, yub])))
    for ax in iterate_axes(axs):
        ax.set_ylim(ylim)


def plot_scale_bar(ax, x0, y0, size, pad=0, text=None, orientation='h', color='k', text_ha='center', text_va='top',
                   text_x_pad=0):
    if orientation[0] == 'h':
        ax.plot([x0, x0 + size], [y0, y0], solid_capstyle='butt', clip_on=False, color=color)
        if text is not None:
            ax.text(x0 + size / 2 + text_x_pad, y0 + pad, text, va=text_va, ha=text_ha, color=color)
    else:
        ax.plot([x0, x0], [y0, y0 + size], solid_capstyle='butt', clip_on=False, color=color)
        if text is not None:
            ax.text(x0 + + size, y0 + size / 2, text, va='center', ha='left', color=color)


def lines(ax, ts, orientation='v', **kwargs):
    kws = dict(color='k', ls='--', lw=0.5, alpha=1., clip_on=False, zorder=-1)
    if len(kwargs) > 0:
        kws.update(kwargs)

    for t in ts:
        if orientation == 'v':
            ax.axvline(t, **kws)
        elif orientation == 'h':
            ax.axhline(t, **kws)
        else:
            raise ValueError(orientation)


def var_to_label(var):
    label_dict = {
        'd_dist_to_soma': 'Distance to soma [µm]',
        'rf_cdia_um': 'RF size [µm]',
        'stim1_stim2_cc': 'Correlation (local-global)',
    }
    if var not in label_dict.keys():
        warnings.warn('Label for {var} not found!')
    return label_dict.get(var, var)


def data_to_range(data, q=0.01):
    """Compute a plotting range for data"""
    v_min = np.nanmin(data)
    v_max = np.nanmax(data)
    v_rng = v_max - v_min
    return v_min - q * v_rng, v_max + q * v_rng


def shrink_axis(ax, factor_x=0.7, factor_y=0.8):
    pos = ax.get_position()

    # Calculate new position
    new_width = pos.width * factor_x
    new_height = pos.height * factor_y
    new_left = pos.x0 + (pos.width - new_width)
    new_bottom = pos.y0 + (pos.height - new_height)

    # Set the new position
    ax.set_position([new_left, new_bottom, new_width, new_height])
