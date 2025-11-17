import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple,Iterable
import matplotlib.lines as mlines
import thesis.code.plot.style as styler
from collections import Counter
from djimaging.utils.plot_utils import plot_trace_and_trigger




def make_plot_df(df, only_order_n=None):
    """For CI analysis of fits, 
    Prepare dataframe with midpoint and error sizes, optionally filter poly_power."""
    if only_order_n is not None:
        df = df[df["poly_power"] == only_order_n].copy()
    df = df.copy()
    df["mid"] = (df["low"] + df["high"]) / 2
    df["err_low"] = df["mid"] - df["low"]
    df["err_high"] = df["high"] - df["mid"]
    return df



def plot_points_and_ci(df,column,
                       ax,
                       dodge=0.3,
                       colors = sns.color_palette("tab10"),
                       xtick_angle: float =0.0,):

    
    colvals = df[column].unique()

    offsets = np.linspace(-dodge/2, dodge/2, len(colvals))
    x_ticks = []
    for i,level in enumerate(colvals):
        sub = df[df[column] == level]
        x = i + offsets[i]
        ax.errorbar([x], sub["mid"], yerr=[sub["err_low"], sub["err_high"]],
                    fmt="o", color=colors[i % len(colors)], 
                    capsize=3, )
        x_ticks.append(x)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(colvals, rotation=xtick_angle)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.3)
    sns.despine(ax=ax)
    return ax


def generate_hierarchical_colors(supergroup_sizes):
    """
    AI: Generate hierarchical colors for a list of supergroup sizes.
    
    Parameters
    ----------
    supergroup_sizes : list[int]
        Number of subgroups (or cells) inside each supergroup.
        
    Returns
    -------
    colors : list[list]
        colors[i][j] is the color for subgroup j of supergroup i.
    base_colors : list
        One representative color per supergroup (useful for regression lines).
    """

    # A set of visually distinct sequential colormaps
    available_cmaps = [
        plt.cm.Reds, # off 
        plt.cm.Greens,
        plt.cm.GnBu,
        plt.cm.Blues,
        plt.cm.Greys,
        plt.cm.Purples,
        plt.cm.Oranges,
        plt.cm.PuBu,
        plt.cm.YlOrBr,
        plt.cm.PuRd,
    ]
    
    # Pick as many as needed, cycling if supergroups > number of available maps
    base_cmaps = [
        available_cmaps[i % len(available_cmaps)]
        for i in range(len(supergroup_sizes))
    ]
    
    all_colors = []
    base_colors = []

    for sg_idx, n_subgroups in enumerate(supergroup_sizes):
        cmap = base_cmaps[sg_idx]

        # avoid extremes → keep shades
        shades = np.linspace(0.2, 1, n_subgroups)[::-1]

        subgroup_colors = [cmap(s) for s in shades]
        all_colors.extend(subgroup_colors)

        # representative color for the whole supergroup (e.g., mid shade)
        base_colors.append(cmap(0.6))

    return all_colors, base_colors





def plot_conf_intervals(df, ax=None, cmap_by = "celltype", legend_str=None, dodge=0.3,figsize =(4,4)):
    """Plot confidence intervals per celltype and poly_power."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    celltypes = sorted(df["celltype"].unique())
    levels = sorted(df["poly_power"].unique())
    if len(levels) == 1:
        dodge = 0
    
    # sort by celltype 
    df = df.sort_values(by="celltype")

    if cmap_by == "celltype":
        # Create direct mapping of celltypes to colors
        # NOTE sorry hard coded no time
        colors, _ = generate_hierarchical_colors([4,0,0,3,1,3])
    elif cmap_by == "poly_power":
        if len(levels) == 1:
            colors = ["black"]
        else:
            colors = sns.color_palette("viridis", len(levels))
    else:
        raise ValueError("cmap_by must be 'celltype' or 'poly_power'")
       
        
    color_map = dict(zip(levels, colors))
    offsets = np.linspace(-dodge/2, dodge/2, len(levels))

    for i, lvl in enumerate(levels):
        sub = df[df["poly_power"] == lvl]
        xs = [celltypes.index(ct) + offsets[i] for ct in sub["celltype"]]
        for x, ct, mid, err_low, err_high in zip(xs, sub["celltype"], 
                                                sub["mid"], sub["err_low"], 
                                                sub["err_high"]):
            color_idx = celltypes.index(ct)
            ax.errorbar([x], [mid], yerr=[[err_low], [err_high]],
                       fmt="o", color=colors[color_idx % len(colors)], 
                       capsize=3, )#label=str(lvl) if ct == celltypes[0] else "")

    # add vline at zero
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.3)

    ax.set_xticks(range(len(celltypes)))
    ax.set_xticklabels(celltypes)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Estimate (± 95% CI)")
    sns.despine(ax=ax)
    if len(levels) > 1:
        labels = [legend_str.get(l, str(l)) if legend_str else str(l) for l in levels]
        ax.legend(title="poly_power", labels=labels)
    return fig,ax


def add_mulitgroup_proxy_legend(ax: plt.Axes,
                                dot_label: str,
                                full_reg_label: str,
                                single_reg_label: str,
                                **legend_kwargs) -> plt.Axes:
    # add projx scatter artist: one scatter dot in grey saying "single rgc response (one color = one rgc)"

    # Create a single grey dot as a legend handle
    proxy_dot = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', label=dot_label)
    proxy_full_reg= mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label=full_reg_label)
    proxy_single_reg= mlines.Line2D([], [], color='grey', linestyle='-', linewidth=0.3, label=single_reg_label)

    # Add the legend
    ax.legend(handles=[proxy_dot,
                       proxy_full_reg,
                       proxy_single_reg],
            frameon=legend_kwargs.get("frameon",False),
            bbox_to_anchor=legend_kwargs.get("bbox_to_anchor",(1.0, 1.0)),
            loc=legend_kwargs.get("loc",'upper right'),
            **legend_kwargs)

    return ax


def plot_trace_trigger_triggerinfo(trace_times,
                                   trace,
                                   triggertimes,
                                   triggeridx2hilightalpha: List[float],
                                   ax,
                                   triggeridx2txt= None,):
    """
    plots traces and triggers. Highlighted by triggeridx2hilightalpha (). 
    if triggeridx2txt is given, it will also add text to the highlighted triggers.
    """

    plot_trace_and_trigger(
        trace_times,
        trace,
        triggertimes,
        ax = ax
    )
    # add highlights
    if triggeridx2hilightalpha is not None:
        for triggeridx, alpha in enumerate(triggeridx2hilightalpha):
            if alpha > 0 and triggeridx < len(triggertimes)-1:
                ax.axvspan(triggertimes[triggeridx], triggertimes[triggeridx +1], color='yellow', alpha=alpha)
            if triggeridx2txt is not None:
                ax.text(triggertimes[triggeridx], np.max(trace), triggeridx2txt[triggeridx], color='blue', fontsize=4,clip_on=True)

    return ax


def plot_trace_trigger_bg_stim(trace_times: np.ndarray,
                                trace: np.ndarray,
                                triggertimes: np.ndarray,
                                stim_onset_times: np.ndarray,
                                ax,
                                bg_color='gray',
                                stim_color='yellow',
                                bg_kwargs={},
                                stim_kwargs={}
    ):
    """
    Plots trace and triggers, highlighting background and stimulus periods.
    Assumes the order of the stimulus is trigger, then background, then stimulus.
    """
    assert len(triggertimes) == len(stim_onset_times), "triggertimes and stim_onset_times must be of same length"


    plot_trace_and_trigger(
        trace_times,
        trace,
        triggertimes,
        ax = ax
    )

    # add vspans for bg and stim
    for i, trigger_time in enumerate(triggertimes):
        stim_onset_time = stim_onset_times[i]
        # background period
        ax.axvspan(trigger_time, stim_onset_time, color=bg_color, **bg_kwargs)
        # stimulus period
        if i < len(triggertimes) - 1:
            next_trigger_time = triggertimes[i + 1]
            ax.axvspan(stim_onset_time, next_trigger_time, color=stim_color, **stim_kwargs)
    return ax
    



def plot_mulit_group_scatter_fits(full_df: pd.DataFrame,
                                    x: str,
                                  y: str,
                                  ax: plt.Axes,
                                  hue : str,
                                  xlabel: str,
                                  ylabel: str,
                                  color_map: Dict[Any,str | np.ndarray] | None = {},
                                  legend_title: str = "",
                                  scatter_kwargs: Dict[str,Any]={},
                                  single_group_fit_kwargs: Dict[str,Any]={},
                                  overall_fit_kwargs: Dict[str,Any]={},
                                  show_legend: bool = False) -> plt.Axes:
        

    # First make the scatter plot

   
    sns.scatterplot(
        data=full_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        palette = color_map,
        **scatter_kwargs
    )
    
    # Then add polynomial fits for each hue value
    for hue_val in full_df[hue].unique():
        hue_val_data = full_df[full_df[hue] == hue_val]
        if len(hue_val_data) >= 3:  # Need at least 3 points for a quadratic fit
            line_color = "gray" if color_map is None else color_map.get(hue_val,"gray")
            line_kws = single_group_fit_kwargs.get("line_kws",
                                                   {"linewidth":0.3,
                                                    "color": line_color,
                                                   })
            sns.regplot(
                x=x, 
                y=y,
                data=hue_val_data,
                ax=ax,
                order=single_group_fit_kwargs.get("order",2),
                scatter=False,
                ci = single_group_fit_kwargs.get("ci",None),
                line_kws=line_kws,
                scatter_kws =single_group_fit_kwargs.get("scatter_kws",{"legend":False}),
                label=single_group_fit_kwargs.get("label",None),
            )
    # add one overall polynomial regression line: black thick
    sns.regplot(
        x=x, 
        y=y,
        data=full_df,
        ax=ax,
        order=overall_fit_kwargs.get("order",2),
        scatter=False,
        ci = overall_fit_kwargs.get("ci",None),
        line_kws=overall_fit_kwargs.get("line_kws",{"linewidth":1.5,"color": "black"}),
        scatter_kws = overall_fit_kwargs.get("scatter_kws",{"legend":False}),
        label="overall fit",
    )


    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend(title=legend_title, loc='upper right',ncol=2)
    else:
        ax.legend_.remove()


    return ax

def add_trigger_bg_stim_legend(ax: plt.Axes) -> plt.Axes:
    # remove legend and add own with proxy artists
    ax.legend().remove()

    # add patch for background and stimulus and red line for trigger

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='gray', edgecolor='gray', alpha=0.3, label='Background period'),
        Patch(facecolor='yellow', edgecolor='yellow', alpha=0.3, label='Stimulus period'),
        Line2D([0], [0], color='red', lw=2, label='Trigger')
    ]
    ax.legend(handles=legend_elements, loc='center',bbox_to_anchor=(0.5, 1.1),ncol=3)
    return ax

def plot_sparse_snippets(snippet_trace_list,
                         single_snippet_dt,
                         snippet_t0s: List[float],
                         ax =None,
                         plot_kwargs = {},):
    """
    Can be used to plot snippets in snippet_trace_list,
    snippet_t0s: list of start times for each snippet
    """
    if ax is None:
        fig, ax = plt.subplots()
    # create time ves
    single_snippet_time_vec = np.arange(len(snippet_trace_list[0])) * single_snippet_dt

    for i, snippet in enumerate(snippet_trace_list):
        time_axis = single_snippet_time_vec + snippet_t0s[i]
        ax.plot(time_axis, snippet, **plot_kwargs)
    return ax


def plot_ordered_snippets(snippet_trace_list,
                            single_snippet_dt,
                            single_snippet_time_shift: float= 0,
                            highlight_bg_times: Tuple[float,float] = [],
                            highlight_bg_patch_kwargs: Dict[str,Any] = {},
                            highlight_stim_times: Tuple[float,float] = [],
                            highlight_stim_patch_kwargs: Dict[str,Any] = {},
                            snippet_vline = False,
                            time_buffer_between_snippets = 0,
                            ax =None,
                            plot_kwargs = {},
                            x_tick_lables = None,
                            x_ticks_kwargs = {},
                            show_legend=False):
    """
    Can be used to plot snippets in snippet_trace_list,
    highilight_bg_times: start stop time relative to snippet start
    highilight_stim_times: start stop time relative to snippet start
    """
   
    if ax is None:
        fig, ax = plt.subplots()
    
    # create time ves
    single_snippet_time_vec = np.arange(len(snippet_trace_list[0])) * single_snippet_dt


    # for vline
    concatenated_traces = np.concatenate(snippet_trace_list)
    vmin = np.nanmin(concatenated_traces)
    vmax = np.nanmax(concatenated_traces)
    vrng = vmax - vmin

    t0 = 0
    x_tick_vals = []
    for i, snippet in enumerate(snippet_trace_list):
        time_axis = single_snippet_time_vec + t0 + single_snippet_time_shift

        ax.plot(time_axis, snippet, **plot_kwargs)
        

        # add x tick at center of snippet
        x_tick_vals.append(t0 + single_snippet_time_vec[len(single_snippet_time_vec)//2])

        # add highlighted bg period
        if highlight_bg_times:
            ax.axvspan(t0 + highlight_bg_times[0],
                       t0 + highlight_bg_times[1],
                       color='gray',
                       **highlight_bg_patch_kwargs)
        # add highlighted stim period
        if highlight_stim_times:
            ax.axvspan(t0 + highlight_stim_times[0],
                       t0 + highlight_stim_times[1],
                       color='yellow',
                       **highlight_stim_patch_kwargs)
        if snippet_vline:

            ax.vlines(t0, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', zorder=-2)
        

        # increment time
        t0 = time_axis[-1] + time_buffer_between_snippets
    
    # set xlim
    tmax = t0 - time_buffer_between_snippets
    ax.set_xlim(0, tmax)

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)

    # add x ticks with  labels
    if x_tick_lables is not None:
        ax.set_xticks(x_tick_vals)
        ax.set_xticklabels(x_tick_lables, **x_ticks_kwargs)

    ax.set_xlabel('Distance to RF center [μm]')
    ax.set_ylabel('Fluorescence [a.u.]')
    
    if show_legend:
        ax = add_trigger_bg_stim_legend(ax)
    return ax


def plot_snippets_subplots(snippet_trace_list1: List[np.ndarray],
                            snippets_trace_list2: List[np.ndarray],
                            times: np.ndarray,
                            axes: Optional[np.ndarray[plt.Axes]] = None,
                            text1: str ="",
                            text2: str ="",
    ) -> Tuple[ np.ndarray]:
    """
    Plots two lists of snippets in two axes, index determines color from tab 10.
    texts are added to the top left of each subplot.
    
    Args:
        snippet_trace_list1: List of numpy arrays containing first set of snippets
        snippets_trace_list2: List of numpy arrays containing second set of snippets
        times: Time points for x-axis
        axes: Optional array of two matplotlib axes for plotting
        text1: Text to add to first subplot
        text2: Text to add to second subplot
    
    Returns:
        tuple: (fig, axes) - figure and axes array containing the plots
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    
    # Get color palette
    colors = sns.color_palette("tab10", n_colors=max(len(snippet_trace_list1), len(snippets_trace_list2)))
    
    # Plot first set of snippets
    for i, snippet in enumerate(snippet_trace_list1):
        axes[0].plot(times, snippet, color=colors[i])
    axes[0].text(0.02, 0.98, text1, 
                 transform=axes[0].transAxes,
                 verticalalignment='top',
                 horizontalalignment='left')
    
    # Plot second set of snippets
    for i, snippet in enumerate(snippets_trace_list2):
        axes[1].plot(times, snippet, color=colors[i])
    axes[1].text(0.02, 0.98, text2,
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 horizontalalignment='left')
    
    # Remove top and right spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    return  axes


def get_celltype_alpha_cmap(celltypes: List[int]) -> Dict[int,np.ndarray]:
    """
    Takes in a list of ints (celltpyes) and colors by supergroup using styler.group2supergroup_color
    then it adds an alpha value to the numpy array to maxiamlly distinguish the values within a supergroup.
    Uses styler.RGC_GROUP_GROUP_ID_TO_CLASS_NAME for grouping
    starts at alpha = 1, if there is another celltype its 1 and 0.5 etc ....
    """
    ct_df = pd.DataFrame({"group":celltypes})
    ct_df["supergroup"] = ct_df["group"].map(styler.RGC_GROUP_GROUP_ID_TO_CLASS_NAME)
    ct_df["rgb"] = ct_df["group"].map(styler.group2supergroup_color)
    
    # Assign alpha values within each supergroup
    def assign_alpha(group):
        n = len(group)
        if n == 1:
            alpha_values = [1.0]
        else:
            # Create linearly spaced alpha values from 1.0 to 0.5
            alpha_values = np.linspace(1.0, 0.2, n)
        
        # Create a copy and assign alpha values properly
        group = group.copy()
        group.loc[:, "alpha"] = alpha_values
        return group
    
    ct_df = ct_df.groupby("supergroup", group_keys=False).apply(assign_alpha)
    
    # Create RGBA color map
    color_map = {}
    for _, row in ct_df.iterrows():
        celltype = row["group"]
        rgb = row["rgb"]
        alpha = row["alpha"]
        # Append alpha to RGB to create RGBA
        rgba = np.append(rgb, alpha)
        color_map[celltype] = rgba
    
    return color_map


def normalize_arrays(arrays: Iterable[np.ndarray],type:str = "single") -> List[np.ndarray]:
    """
    Normalizes the arrays in the list to 0..1 range. If type is "single", each array is normalized independently. if type is "joint", all arrays are normalized jointly.
    """
    if type == "single":
        normalized_arrays = []
        for array in arrays:
            arr_min = np.nanmin(array)
            arr_max = np.nanmax(array)
            if np.isfinite(arr_min) and np.isfinite(arr_max) and arr_max > arr_min:
                norm_array = (array - arr_min) / (arr_max - arr_min)
            else:
                norm_array = np.zeros_like(array)
            normalized_arrays.append(norm_array)
        return normalized_arrays
    elif type == "joint":
        all_data = np.concatenate([array.ravel(order='C') for array in arrays])
        vmin = np.nanmin(all_data) if np.any(np.isfinite(all_data)) else 0.0
        vmax = np.nanmax(all_data) if np.any(np.isfinite(all_data)) else 0.0

        normalized_arrays = []
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            scale = vmax - vmin
            for array in arrays:
                norm_array = (array - vmin) / scale
                normalized_arrays.append(norm_array)
        else:
            for array in arrays:
                norm_array = np.zeros_like(array)
                normalized_arrays.append(norm_array)
        return normalized_arrays



def plot_2d_array_comparison(
        array1: List[np.ndarray],
        array2: List[np.ndarray],
        axes: np.ndarray[plt.Axes],
        array_colors : Tuple[str, str],
        cmap: str = "RdBu_r",
        gap_width: int = 2,  # Width of gap between sections (in pixels)
        norm_type: str | None = None,
    ):
    """

    """
    assert len(array1) == len(array2), "Lists must have the same length"
    n_pairs = len(array1)
    assert axes.shape == (n_pairs,), f"Axes array must have shape ({n_pairs},) but got {axes.shape}"



    # --- Plot ---
    for idx in range(n_pairs):

        ax = axes[idx]

        array1_data = np.asarray(array1[idx])
        array2_data  = np.asarray(array2[idx])

        if array1_data.shape != array2_data.shape:
            raise ValueError(f"Shape mismatch at pair {idx}: {array1_data.shape} vs {array2_data.shape}")

        # normalize 
        if norm_type:
            assert norm_type in ["single","joint"], "norm_type must be 'single' or 'joint'"
            array1_norm, array2_norm = normalize_arrays([array1_data, array2_data], type=norm_type)
        else:
            array1_norm = array1_data
            array2_norm = array2_data
        

        h, w = array1_norm.shape
        gap = np.zeros((h, gap_width))

        # Build concatenated image: offline | gap | online | [gap | comparison]
        borders = [
            ("array1", (0, 0, w, h),     array_colors[0]),
            ("array2",  (w + gap_width, 0, w, h), array_colors[1]),
        ]


        combined_img = np.concatenate([array1_norm, gap, array2_norm], axis=1)
        im = ax.imshow(combined_img, cmap=cmap, vmin=0, vmax=1, origin='upper')
        last_main_im = im

        # Borders
        for _, (x0, y0, ww, hh), color in borders:
            rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), ww, hh,
                                 edgecolor=color, facecolor='none', linewidth=6)
            ax.add_patch(rect)

        # Ticks & limits
        ax.set_xticks([])
        ax.set_yticks([])
        right_edge = (2*w + gap_width - 0.5)
        ax.set_xlim(-0.5, right_edge)
        ax.set_ylim(h - 0.5, -0.5)

    return axes





def plot_2time_series(time_series: np.ndarray,
                     axes: np.ndarray[plt.Axes] | None = None,
                     palette: Tuple[str, str] = ("green", "purple"),
                     labels: Tuple[str,  str] = ("Green channel", "UV channel")
                     ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots time series comparison in subplots.
    
    Args:
        time_series: Array of shape (n_rois, 2, n_timepoints) containing pairs of time series
        axes: Optional array of matplotlib axes to plot on. If None, new axes will be created
        
    Returns:
        tuple: (fig, axes) containing the figure and axes array
    """
    
    MODEL_FRAMERATE = 30  # fps
    n_rois, n_series, n_timepoints = time_series.shape
    assert n_series == 2, f"time_series must have shape (n_rois, 2, n_timepoints) but got {time_series.shape}"

    # Create time vector
    time = np.arange(n_timepoints) / MODEL_FRAMERATE  # in seconds

    # Create subplots
    if axes is None:
        fig, axes = plt.subplots(n_rois, 1, figsize=(5, 0.5 * n_rois), sharex=True)
        if n_rois == 1:
            axes = [axes]
    else:
        fig = axes[0].figure
        assert len(axes) == n_rois, "Number of provided axes must match number of ROIs"

    # get palette

    for i, ax in enumerate(axes):
        ax.plot(time, time_series[i, 0], color=palette[0], lw=1, 
                )
        ax.plot(time, time_series[i, 1], color=palette[1], lw=1, 
               )



        sns.despine(ax=ax)

        ax.set_xlabel("Time [s]")



    # Figure-level legend above all traces
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.05))

    return fig, axes