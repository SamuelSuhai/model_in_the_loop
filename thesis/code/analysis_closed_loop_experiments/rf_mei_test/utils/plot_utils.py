import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple,Iterable
import matplotlib.lines as mlines
import thesis.code.plot.style as styler
from collections import Counter




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


def plot_conf_intervals(df, ax=None, cmap_by = "celltype", legend_str=None, dodge=0.3,figsize =(4,4)):
    """Plot confidence intervals per celltype and poly_power."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    celltypes = sorted(df["celltype"].unique())
    levels = sorted(df["poly_power"].unique())
    if len(levels) == 1:
        dodge = 0
    
    
    if cmap_by == "celltype":
        # Create direct mapping of celltypes to colors
        palette = sns.color_palette("tab10", n_colors=len(celltypes))
        colors = palette  # Use palette directly
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
    ax.set_xlabel("Celltype")
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
            loc=legend_kwargs.get("loc",'upper right')
            **legend_kwargs)

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
        ax.legend(title=legend_title, loc='upper right')
    else:
        ax.legend_.remove()


    return ax

def plot_ordered_snippets(snippet_trace_list,
                            single_snippet_dt,
                            time_buffer_between_snippets = 0,
                            ax =None,
                            plot_kwargs = {},
                            stim_onset_patch_kwargs = {},
                            x_tick_lables = None,
                            x_ticks_kwargs = {},
                            show_legend=False):
    """
    Can be used to plot snippets in snippet_trace_list,
    """

    if ax is None:
        fig, ax = plt.subplots()
    single_snippet_time_vec = np.arange(len(snippet_trace_list[0])) * single_snippet_dt
    t0 = 0
    x_tick_vals = []
    for i, snippet in enumerate(snippet_trace_list):
        time_axis = single_snippet_time_vec + t0

        ax.plot(time_axis, snippet,color = plot_kwargs.get("color","blue"))
        

        # add x tick at center of snippet
        x_tick_vals.append(t0 + single_snippet_time_vec[len(single_snippet_time_vec)//2])

        # add patch from half snippet to end
        stim_onset_patch_start = t0 + single_snippet_time_vec[len(single_snippet_time_vec)//2]
        stim_onset_patch_end = t0 + single_snippet_time_vec[-1]
        ax.axvspan(stim_onset_patch_start, stim_onset_patch_end,
                    color=stim_onset_patch_kwargs.get("color","yellow"), 
                    alpha=stim_onset_patch_kwargs.get("alpha",0.3),
                    )

        t0 = time_axis[-1] + time_buffer_between_snippets

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)

    # add x ticks with distance labels
    ax.set_xticks(x_tick_vals)
    if x_tick_lables is not None:
        ax.set_xticklabels(x_tick_lables, **x_ticks_kwargs)
    else:
        ax.set_xticklabels([ i for i in range(len(snippet_trace_list))])    

    ax.set_xlabel('Distance to RF center [μm]')
    ax.set_ylabel('Fluorescence [a.u.]')
    
    if show_legend:
        axvspan_label=stim_onset_patch_kwargs.get("label","Stimulus presentation")

        # add proxy artists
        from matplotlib.patches import Patch
        proxy_lines = [Patch(facecolor=stim_onset_patch_kwargs.get("color","yellow"), alpha=stim_onset_patch_kwargs.get("alpha",0.3), label=axvspan_label)]
        ax.legend(handles=proxy_lines,
                  frameon=False, 
                bbox_to_anchor=(1.0, 1.15),  # (x, y) position relative to the plot

                  loc='upper right')

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
