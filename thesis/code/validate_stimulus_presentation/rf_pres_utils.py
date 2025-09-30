import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


def load_rf_data(save_dir: str):
    with open(os.path.join(save_dir,'fit_gauss_2d_rf_20200226.pkl'),'rb') as f:
        FitGauss2DRF = pickle.load(f)

    with open(os.path.join(save_dir,'split_rf_20200226.pkl'),'rb') as f:
        SplitRF = pickle.load(f)
    
    return FitGauss2DRF, SplitRF



def get_means_from_srf_df (FitGauss2DRF:pd.DataFrame, roi_id:int) -> tuple[float,float]:
    params = FitGauss2DRF[FitGauss2DRF["roi_id"]==roi_id]["srf_params"].item()
    
    y_mean,x_mean = params["y_mean"], params["x_mean"]
    return y_mean, x_mean

def plot_srf_pos(roi_id:int ,
                 FitGauss2DRF:pd.DataFrame,
                 SplitRF:pd.DataFrame,
                 noise_stimulus: np.ndarray,
                 rf_kwargs = {},
                 ax = None):

    
    y_mean,x_mean = get_means_from_srf_df(FitGauss2DRF, roi_id)

    # (dj_table_holder('SplitRF')() & {'roi_id': roi_id}).plot1()
    sRF = SplitRF[SplitRF["roi_id"]==roi_id]["srf"].item()

    print("srf table x mean", x_mean,"\nsrf table y mean", y_mean)
    if ax is None:
        fig,ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(noise_stimulus[0],origin='lower',cmap="viridis")
    
    if 'alpha' not in rf_kwargs:
        rf_kwargs['alpha'] = 0.5
    ax.imshow(sRF, origin='lower', **rf_kwargs, cmap='grey')
    ax.scatter(x_mean,y_mean, color='red', s=20, label='Peak Position')

    # take out ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # no spines 
    for spine in ax.spines.values():
        spine.set_visible(False)