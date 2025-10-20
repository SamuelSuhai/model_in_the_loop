import seaborn as sns

PALETTE = 'tab10'

online_offline_dict = {
    'online': "dodgerblue",
    'offline': "black",
}



def get_palette(_type):
    if _type == 'online_offline':
        return online_offline_dict
    else:
        raise NotImplementedError("")


def get_group_color(group, indicator):
    return get_palette(indicator)[group]



def set_legend_side_labels(ax):
    for t in ax.get_legend().texts:
        t.set_text(_label_dict.get(t.get_text(), t.get_text()))


def set_side_xlabels(ax):
    ax.set_xticklabels([_label_dict.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
