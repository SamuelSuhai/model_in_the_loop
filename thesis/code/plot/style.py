import seaborn as sns
import numpy as np
from .labeling import RGC_GROUP_GROUP_ID_TO_CLASS_NAME
PALETTE = 'tab10'

online_offline_dict = {
    'online': "dodgerblue",
    'offline': "black",
}

supergroup2color = {
    "OFF": np.array([255, 0, 0]) / 255,          # Red
    "ON-OFF": np.array([0, 255, 0]) / 255,      # Green
    "Fast ON": np.array([64, 224, 208]) / 255,  # Turquoise
    "Slow ON": np.array([0, 0, 255]) / 255,     # Blue
    "Uncertain RGC": np.array([128, 0, 128]) / 255,  # Purple
    "AC": np.array([0, 0, 0]) / 255,       # white
    }

group2supergroup_color = {group: supergroup2color[supergroup] for group,supergroup in RGC_GROUP_GROUP_ID_TO_CLASS_NAME.items()}
    


def get_palette(_type):
    if _type == 'online_offline':
        return online_offline_dict
    elif _type == 'group':
        return group2supergroup_color
    elif _type == 'supergroup':
        return supergroup2color
    else:
        raise NotImplementedError("")


def get_group_color(group, indicator):
    return get_palette(indicator)[group]



def set_legend_side_labels(ax):
    for t in ax.get_legend().texts:
        t.set_text(_label_dict.get(t.get_text(), t.get_text()))


def set_side_xlabels(ax):
    ax.set_xticklabels([_label_dict.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])

