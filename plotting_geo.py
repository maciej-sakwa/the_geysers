import scipy.signal as sgn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#General plot parameters
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 1.5
colors = ['#5DADE2', '#52BE80', '#CD6155']


# Updated
def plot_means_subplots(time_series:np.ndarray, cluster_labels:list, plotted_clusters:list) -> None:

    # Parameters and label definitions
    mpl.rcParams['font.size'] = 14
    arr_labels = np.array(cluster_labels)
    legend_labels = [f'DTH-{i}' for i in plotted_clusters]
    years = np.arange(2006, 2017, 1)
    tick_labels = [f'\'{item[-2:]}' for item in map(str, years)]
    i = 0
    x = range(time_series.shape[1])

    # Plot definition
    fig, axs = plt.subplots(len(plotted_clusters))
    for cluster, ax in zip(plotted_clusters, axs.ravel()):
        # Normalization and mean calculation
        time_extract = time_series[arr_labels == cluster]
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)
        std = np.std(time_extract_norm.T, axis=0) / np.sqrt(np.shape(time_extract)[0])
    

        # Mean and STD plots with fill between
        ax.plot(time_norm, label='{}'.format(legend_labels[i]), c='k')
        ax.plot(time_norm + std, "k--")
        ax.plot(time_norm - std, 'k--')
        ax.fill_between(x, time_norm - std, time_norm + std, color='#a7c8e2')
        ax.legend(loc='upper left', framealpha=1)

        # X-axis settings
        ax.set_xticks(np.arange(0, 121, 12))
        ax.set_xticklabels(tick_labels)
        ax.set_xlim([0, 125])
        ax.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
        i += 1

    # axs[0].set_yticks([0, 0.2])
    # axs[1].set_yticks([0, 0.5])
    # axs[2].set_yticks([0, 0.5])
    # Plot area settings
    fig.supylabel('Normalized density [p.u.]')
    fig.set_size_inches(8, 7.5)
    ax.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
    plt.xlabel('Observation date [y]')
    plt.tight_layout()
    plt.show()


# Updated
def b_values(df_catalogue:pd.DataFrame, mag_borders:list, plotted_clusters:list) -> list:

    mag_cluster = []
    hist_magnitudes = []

    # magnitudes of the three clusters
    for i, cluster in enumerate(plotted_clusters):
        mag_cluster.append(np.array(df_catalogue[df_catalogue.cluster == cluster]['Magnitude']))
        
        hist_magnitudes_cluster, bin_edges_magnitudes = np.histogram(mag_cluster[i], np.arange(-1, 5, 0.2))
        hist_magnitudes.append(hist_magnitudes_cluster)

    indices_for_bvalue = (bin_edges_magnitudes[:-1] >= mag_borders[0]) & (bin_edges_magnitudes[:-1] <= mag_borders[1])

    # Plot  
    plt.figure(figsize=(8, 4))  
    for i, cluster in enumerate(plotted_clusters):
        plt.plot(bin_edges_magnitudes[:-1] + 0.1, np.log10(hist_magnitudes[i]), label=f'Cluster {i}', c=colors[i])
    plt.plot([mag_borders[0], mag_borders[0]], [0, 5], 'k')
    plt.plot([mag_borders[1], mag_borders[1]], [0, 5], 'k')

    # Visuals 
    plt.xlim([0, 4.5])
    plt.ylim([0, 5])
    plt.xlabel('Earthquake magnitude')
    plt.ylabel('Occurrences [log10]')
    plt.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Balculation of b value
    b_value_cluster_3, _ = np.polyfit(bin_edges_magnitudes[:-1][indices_for_bvalue],
                                      np.log10(hist_magnitudes[0][indices_for_bvalue]), 1)
    b_value_cluster_4, _ = np.polyfit(bin_edges_magnitudes[:-1][indices_for_bvalue],
                                      np.log10(hist_magnitudes[1][indices_for_bvalue]), 1)
    b_value_cluster_5, _ = np.polyfit(bin_edges_magnitudes[:-1][indices_for_bvalue],
                                      np.log10(hist_magnitudes[2][indices_for_bvalue]), 1)

    return [b_value_cluster_3, b_value_cluster_4, b_value_cluster_5]


# TODO Update function
def ICA(folder_path = r'C:\Users\macie\Desktop\Fellowship\The Geysers\code\for_mauro',
        data_folder = r'C:\Users\ws777\Desktop\Fellowship\The Geysers\code\data', 
        ICA_plot:bool=False, ICA_FFT_plot:bool=False) -> None:


    file_ind_components = os.path.join(folder_path, r'ICs.csv')
    file_water_global = os.path.join(data_folder, r'water_injecion_GLOBAL.csv')
    ics = np.array(pd.read_csv(file_ind_components, index_col=[0]))

    water_global = pd.read_csv(file_water_global, names=['date', 'injection'])
    water_global['year'] = water_global.apply(lambda row: row.date[-4:], axis=1)
    water_global['month'] = water_global.apply(lambda row: row.date[4:6], axis=1)
    water_yearly_arr = water_global.groupby(['year'])['injection'].sum().to_numpy()

    years = np.arange(2006, 2017, 1)
    x_water = np.arange(6, 127, 12)
    water_yearly_arr[-1] = np.average(water_yearly_arr)

    if ICA_plot:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
        for i in range(3):
            ax.plot(ics[:, i], c=colors[i], label=f'IC{i+1}')
        ax.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')

        ax.set_xlabel('Observation [y]')
        ax.set_ylabel('Time series [p.u.]')
        ax.set_ylim([-4, 6])
        ax.set_yticks([-4, -2, 0, 2, 4, 6])
        ax.set_yticklabels(['', '-2', '0', '2', '4', ''])
        ax.set_xticks(np.arange(0, 121, 12))
        ax.set_xticklabels([f'\'{item[-2:]}' for item in map(str, years)])
        ax.set_xlim([0, 125])
        ax_due = ax.twinx()
        ax_due.plot(0.5*(water_global['injection']/water_global['injection'].mean() - 1), linestyle='dotted',
                    c='#c7c7c7', label = 'Monthly injections')
        ax_due.plot(x_water, (water_yearly_arr/np.average(water_yearly_arr)) - 1,
                    c='k', linestyle="dashed", label='Yearly water injections')
        ax_due.set_ylim([-0.2, 0.3])
        ax_due.set_yticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
        ax_due.set_yticklabels(['', '-0.1', '0.0', '0.1', '0.2', ''])
        ax_due.set_ylabel('Water Injections Normalized [p.u.]')
        ax.legend(loc='upper left')
        ax_due.legend(loc='upper right')
        plt.show()

    if ICA_FFT_plot:
        y_fft_water = water_global['injection'].to_numpy()
        x_fft = np.fft.fftfreq(len(ics[:, 0]), 1)
        plt.figure(figsize=(8, 4))
        for i in range(3):
            plt.plot(np.log10(x_fft[1:len(ics[:, i])//2]), np.log10(np.abs(np.fft.fft(ics[:, i])[1:len(ics[:, i])//2])),
                    c=colors[i], label=f'IC{i + 1}')
        plt.plot(np.log10(x_fft[1:len(y_fft_water)//2]),
                np.log10(np.abs(np.fft.fft(y_fft_water)[1:len(y_fft_water)//2])),
                    c='k', label='Monthly injections', linestyle='dashed')
        ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        tick_labels = [str(item) if "1" in str(item) else "" for item in ticks]
        log_10_ticks = np.log10(ticks)
        plt.xticks(log_10_ticks, tick_labels)
        plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], labels=["", "0.1", "", "1", "", "10", "", "100"])
        plt.xlim([-2.25, -0.25])
        plt.ylim([-1.5, 2])
        plt.xlabel('Frequency [1/m]')
        plt.ylabel('Fast Fourier Transform')
        plt.legend(loc='lower left')
        plt.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
        plt.tight_layout()
        plt.show()


    return


# Check alignment
def plot_cluster_nodes(df_catalogue:pd.DataFrame, df_density:pd.DataFrame,  geo_bounds: dict, wells:np.ndarray = None, plotted_clusters:list = [3,4,5]) -> None:

    # Add a final element at the end of each file
    x_contour = np.arange(geo_bounds['long_min'], geo_bounds['long_max'] + geo_bounds['step_l'], geo_bounds['step_l'])
    y_contour = np.arange(geo_bounds['lat_min'], geo_bounds['lat_max'] + geo_bounds['step_l'], geo_bounds['step_l'])
    z_contour = - np.arange(geo_bounds['depth_min'], geo_bounds['depth_max'] + geo_bounds['step_d'], geo_bounds['step_d'])


    x_cat = df_catalogue['Longitude']
    y_cat = df_catalogue['Latitude']
    z_cat = df_catalogue['Depth']

    H_cat, xedges_cat, yedges_cat = np.histogram2d(np.array(x_cat), np.array(y_cat), bins=(x_contour, y_contour))
    H_cat_depth, xedges_cat_depth, zedges_cat_depth = np.histogram2d(np.array(x_cat), -np.array(z_cat), bins=(x_contour,np.flip(z_contour)))

    X_cat, Y_cat = np.meshgrid(xedges_cat, yedges_cat)
    X_cat_depth, Z_cat_depth = np.meshgrid(xedges_cat_depth, zedges_cat_depth)

    # Setup the plot
    mpl.rcParams['font.size'] = 14
    fig, axes = plt.subplots(nrows=2, ncols=len(plotted_clusters), 
                             figsize=(len(plotted_clusters)*5 + 1, 9), sharey='row',
                             gridspec_kw={'width_ratios': [1, 1, 1.2], "height_ratios":[1, 0.9]})
    
    columns = [f'Cluster {item}' for item in plotted_clusters]

    # loop over the cluster
    for i, cl in enumerate(plotted_clusters):

        sub_df_catalogue = df_catalogue[df_catalogue.cluster == cl].copy()
        sub_df_density = df_density[df_density.cluster == cl].copy()
        sequence_x_vals = sub_df_catalogue.Longitude
        sequence_y_vals = sub_df_catalogue.Latitude
        sequence_z_vals = sub_df_catalogue.Depth

        # Bins and contours definitions
        H, xedges, yedges = np.histogram2d(sequence_x_vals, sequence_y_vals, bins=(x_contour, y_contour))
        Hz, xedges, zedges = np.histogram2d(sequence_x_vals, -sequence_z_vals, bins=(x_contour, np.flip(z_contour)))
        X, Y = np.meshgrid(xedges, yedges)
        X1, Z = np.meshgrid(xedges, zedges)

        # Findint the highest value
        max_value_array = sub_df_density.groupby(['index_1D'])['density'].max().values
        max_value_index_array = sub_df_density.groupby(['index_1D'])['density'].max().index.to_numpy()
        max_value_index = np.argmax(max_value_array)

        # Convert the coordinates to list
        max_value_coords = sub_df_density[sub_df_density.index_1D == max_value_index_array[max_value_index]][['lat', 'long']].mean().to_list()


        # Lat-long graph
        axes[0, i].contour(X_cat[:-1, :-1]+(geo_bounds['step_l']/2), Y_cat[:-1, :-1]+(geo_bounds['step_l']/2), np.log10(H_cat.T), 15, cmap='Greys', alpha = 0.5)
        plot_latlong = axes[0, i].contourf(X[:-1, :-1]+(geo_bounds['step_l']/2), Y[:-1, :-1]+(geo_bounds['step_l']/2), H.T, 10, cmap='Reds')
        if wells is not None:
            axes[0, i].scatter(wells[:, 0]+(geo_bounds['step_l']/2), wells[:, 1]-(geo_bounds['step_l']/2), s=20, label='Injection Wells')
        axes[0, i].scatter(x=max_value_coords[1]+(geo_bounds['step_l']/2), y=max_value_coords[0]+(geo_bounds['step_l']/2), s=50, marker='1', c='#18edd4', label = 'Max. density')

        # Visuals 
        axes[0, i].set_xlabel('Longitude')
        axes[0, i].legend(loc="upper right")
        axes[0, i].set_xlim(-122.9, -122.65)
        axes[0, i].set_xticks(np.arange(-122.9, -122.61, 0.05))
        axes[0, i].set_xticklabels(["-122.9", "", "-122.8", "", "-122.7", ""])
        axes[0, i].grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
        axes[0, i].set_title(f'{columns[i]}')


        # Depth-long graph
        axes[1, i].contour(X_cat_depth[:-1, :-1]+(geo_bounds['step_l']/2), Z_cat_depth[:-1, :-1]+(geo_bounds['step_d']/2), np.log10(H_cat_depth.T), 15,
                           cmap='Greys')
        plot_longdepth = axes[1, i].contourf(X1[:-1, :-1]+(geo_bounds['step_l']/2), Z[:-1, :-1]+(geo_bounds['step_d']/2), Hz.T, 10, cmap='Reds')

        # Visuals 
        axes[1, i].set_xlabel('Longitude')
        axes[1, i].set_xlim(-122.9, -122.65)
        axes[1, i].set_xticks(np.arange(-122.9, -122.61, 0.05))
        axes[1, i].set_xticklabels(["-122.9", "", "-122.8", "", "-122.7", ""])
        axes[1, i].grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')

    fig.colorbar(plot_latlong, ax=axes[0, i])
    fig.colorbar(plot_longdepth, ax=axes[1, i])

    # Ax settings
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_ylim(38.7, 38.9)
    axes[0, 0].set_yticks(np.arange(38.7, 38.91, 0.05))
    axes[0, 0].set_yticklabels(["38.7", "", "38.8", "", "38.9"])

    axes[1, 0].set_ylabel('Depth [km]')
    axes[1, 0].set_ylim(-6, 0)
    axes[1, 0].set_yticks(np.arange(-6, 1, 1))
    axes[1, 0].set_yticklabels(["-6", "", "-4", "", "-2", "", "0"])

    # Final settings
    plt.tight_layout()
    plt.show()

    return


# TODO Update function
def coulomb(df_density:pd.DataFrame, coulomb_stress_ratios: np.ndarray, plotted_clusters:list) -> None:

    nodes = np.sort(np.unique(df_density.index_1D))
    
    nodes_cluster = []
    indices_nodes_cluster = []
    hist_ratios_max = []

    for i, cluster in enumerate(plotted_clusters):
        nodes_cluster.append(np.array(np.unique(df_density[df_density.cluster==cluster]['index_1D'])))
        indices_nodes_cluster.append(np.arange(len(nodes))[np.isin(nodes, nodes_cluster[i])])
        hist_ratios_max_single, bin_edges_ratios_max = np.histogram(np.log10(coulomb_stress_ratios)[indices_nodes_cluster[i]], np.arange(0, 1, 0.1))
        hist_ratios_max.append(hist_ratios_max_single)

    
    # Plot 
    plt.figure(figsize=(8, 4))
    for i, cluster in range(plotted_clusters):
        plt.plot(bin_edges_ratios_max[:-1]+0.05, hist_ratios_max[i]/max(hist_ratios_max[i]), label=f'Cluster {cluster}', c=colors[i])

    # Visuals 
    plt.xlabel('Ratio of Coulomb stress drop [log10]')
    plt.xticks([0, 0.3, 0.6, 0.778,  0.9, 1],(['', '2', '4', '6', '8', '10']))
    plt.ylabel('Frequency')
    plt.xlim([0, 1])
    plt.ylim([0, 1.2])
    plt.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


# Check alignment
def cc_plots(geo_bounds: dict, df_catalogue: pd.DataFrame, df_density: pd.DataFrame, timeseries: np.array, ics: pd.DataFrame, wells:np.ndarray) -> None:

    # Add a final element at the end of each file
    x_contour = np.append(np.sort(np.unique(df_density.long)), np.sort(np.unique(df_density.long))[-1] + geo_bounds['step_l'])
    y_contour = np.append(np.sort(np.unique(df_density.lat)), np.sort(np.unique(df_density.lat))[-1] + geo_bounds['step_l'])
    z_contour = -np.append(np.sort(np.unique(df_density.depth)), np.sort(np.unique(df_density.depth))[-1] + geo_bounds['step_d'])

    nodes = np.sort(np.unique(df_density.index_1D))

    # find the node with the largest CC in time history with the ICs
    # initialize CC with the three ICs
    ccs_with_ics = np.zeros([len(nodes), np.shape(ics)[1]])

    longs_nodes = []
    lats_nodes = []
    depths_nodes = []
    cluster_nodes = []

    # loop over the grid nodes
    print('Run over the nodes..')
    for i, node in enumerate(nodes):

        df_node = df_density[df_density.index_1D == node].copy()

        # A series with all the same values. Setting to '0' is the same as '1,2,' ect...
        cluster_nodes.append(np.array(df_node['cluster'])[0])
        longs_nodes.append(np.array(df_node['long'])[0])
        lats_nodes.append(np.array(df_node['lat'])[0])
        depths_nodes.append(np.array(df_node['depth'])[0])

        # extract time series of the node
        timeseries_node = timeseries[node]

        # loop over the ICs
        for en_ic in range(ics.shape[1]):
            ic = np.array(ics.iloc[:, en_ic])
            C = sgn.correlate(timeseries_node - np.mean(timeseries_node), ic)
            ccs_with_ics[i, en_ic] = max(C) / (np.linalg.norm(timeseries_node - np.mean(timeseries_node)) * np.linalg.norm(ic))


    # Set up meshgrid and catalogue histogram
    latitudes_array = np.array(df_catalogue.Latitude)
    longitudes_array = np.array(df_catalogue.Longitude)
    H_cat, xedges_cat, yedges_cat = np.histogram2d(longitudes_array, latitudes_array, bins=(x_contour, y_contour))
    X_mesh, Y_mesh = np.meshgrid(xedges_cat, yedges_cat)

    # Indices of the max nodes
    ind_max_ic1 = np.nanargmax(ccs_with_ics[:,0])
    ind_max_ic2 = np.nanargmax(ccs_with_ics[:,1])
    ind_max_ic3 = np.nanargmax(ccs_with_ics[:,2])

    # Location of the max nodes
    latitude_max = [lats_nodes[ind_max_ic1], lats_nodes[ind_max_ic2], lats_nodes[ind_max_ic3]]
    longitude_max = [longs_nodes[ind_max_ic1], longs_nodes[ind_max_ic2], longs_nodes[ind_max_ic3]]
    depth_max = [depths_nodes[ind_max_ic1], depths_nodes[ind_max_ic2], depths_nodes[ind_max_ic3]]

    histograms_max_cc_depth = []

    # Create depth layer histogram
    for i, depth in enumerate(depth_max):
        x_cat = df_catalogue[df_catalogue['Depth']==depth]['Longitude']
        y_cat = df_catalogue[df_catalogue['Depth']==depth]['Latitude']
        histogram_cat, xedges_cat, yedges_cat = np.histogram2d(np.array(x_cat), np.array(y_cat), bins=(x_contour, y_contour))
        histograms_max_cc_depth.append(histogram_cat)

    # Set up the plot area       
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 5), sharey=True, sharex=True, gridspec_kw={'width_ratios': [1, 1, 1.2]})
    plot_titles = ['(a)', '(b)', '(c)']

    # Plot
    for i, depth in enumerate(depth_max):

        axes[i].scatter(wells[:, 0], wells[:, 1], s=20, alpha=0.3, label='Injection wells')
        contour = axes[i].contour(X_mesh[1:, 1:]-0.0125, Y_mesh[1:, 1:]-0.0125, np.log10(histograms_max_cc_depth[i].T), 10, cmap='Greys', alpha=.5)
        axes[i].grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
        cc_plot = axes[i].scatter(np.array(longs_nodes)[np.array(depths_nodes)==depth] + 0.0125,
                                    np.array(lats_nodes)[np.array(depths_nodes)==depth]+ 0.0125,
                                    c = ccs_with_ics[np.array(depths_nodes)==depth, i], 
                                    s=200, cmap='Reds',
                                    vmin=0, vmax=1, 
                                    label=f"Cross correlation IC{i+1}")
        
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].legend(loc="upper right")
        axes[i].set_title(plot_titles[i])
        fig.colorbar(contour)

    # Ax settings
    axes[0].set_ylim(38.7, 38.9)
    axes[0].set_xlim(-122.9, -122.65)
    axes[0].set_xticks(np.arange(-122.9, -122.61, 0.05))
    axes[0].set_xticklabels(["-122.9", "", "-122.8", "", "-122.7", ""])
    axes[0].set_yticks(np.arange(38.7, 38.91, 0.05))
    axes[0].set_yticklabels(["38.7", "", "38.8", "", "38.9"])


    fig.colorbar(cc_plot, ax=axes[2])
    plt.tight_layout()
    plt.show()



    # years = np.arange(2006, 2017, 1)

    # plt.figure(figsize=(8,4))
    # plt.plot(df_time_series.iloc[nodes[ind_max_ic1], :], label='NM-CC1', c=colors[0])
    # plt.plot(df_time_series.iloc[nodes[ind_max_ic2], :], label='NM-CC2', c=colors[1])
    # plt.plot(df_time_series.iloc[nodes[ind_max_ic3], :], label='NM-CC3', c=colors[2])

    # plt.xticks(np.arange(0, 121, 12), labels=[f'\'{item[-2:]}' for item in map(str, years)])
    # plt.xlim([0, 125])
    # # plt.ylim([0, 5])
    # plt.ylabel('Normalized density [p.u.]')
    # plt.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
    # plt.xlabel('Observation date [y]')
    # plt.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()


    return



def dth_fft(time_series:np.ndarray, cluster_labels:list, plotted_clusters:list, injections: pd.DataFrame) -> None:

    # Parameters definitions
    arr_labels = np.array(cluster_labels)

    y_fft_water = injections['injection'].to_numpy()/injections['injection'].max()
    x_fft = np.fft.fftfreq(np.shape(time_series)[1], 1)
    plt.figure(figsize=(8, 4))

    for i, cluster in enumerate(plotted_clusters):

        # Normalization and mean calculation
        time_extract = time_series[arr_labels == cluster]
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)
        
        plt.plot(np.log10(x_fft[1:len(time_norm)//2]), np.log10(np.abs(np.fft.fft(time_norm)[1:len(time_norm)//2])),
                    c=colors[i], label=f'FFT of DTH-{cluster}')
        
    # Injections
    plt.plot(np.log10(x_fft[1:len(y_fft_water)//2]), np.log10(np.abs(np.fft.fft(y_fft_water)[1:len(y_fft_water)//2])),
                    c='k', label='Monthly injections', linestyle='dashed')
    
    # Visuals
    ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tick_labels = [str(item) if "1" in str(item) else "" for item in ticks]
    log_10_ticks = np.log10(ticks)
    plt.xticks(log_10_ticks, tick_labels)
    plt.yticks([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], labels=["", "0.01", "", "0.1", "", "1", "", "10", "", "100"])
    plt.xlim([-2.25, -0.25])
    plt.ylim([-2.5, 2])
    plt.xlabel('Frequency [1/m]')
    plt.ylabel('Fast Fourier Transform')
    plt.legend(loc='lower left')
    plt.grid(visible=True, axis='both', alpha=0.3, which='major', c='#dbdbdb')
    plt.tight_layout()
    plt.show()