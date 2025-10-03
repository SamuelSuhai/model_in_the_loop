import matplotlib.pyplot as plt



def show_all_rois_plot(dj_table_holder,wrapper,field_key):


    # different way of getting rois for different wrappers
    if wrapper.name == "Random Seed MEI":
        rois = wrapper.mei_data_container["roi_id"].unique()

        # we have 4 col per row 
        n_rows = len(rois)
        n_cols = 4
        fig,axs = plt.subplots(n_rows,n_cols,figsize=(n_cols*5,n_rows*5))
        for row_idx,roi_id in enumerate(rois):
            axes_reshpaped = axs[row_idx,:].reshape(2,2)
            wrapper.plot1(roi_id=roi_id,
                          field_key=field_key,
                          axs=axes_reshpaped,show=False)
    
    elif wrapper.name == "STA":
        rois = (dj_table_holder("STA")() & field_key).proj().fetch("roi_id")
        n_rois = len(rois)
        n_cols = 2
        n_rows = n_rois
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        for i, roi_id in enumerate(rois):
            wrapper.plot1( roi_id=roi_id, axs=[axs[i,0],axs[i,1]],show=False)
    else:
        raise ValueError("Wrapper type not supported for show_all_rois_plot")


    
    return fig, axs

