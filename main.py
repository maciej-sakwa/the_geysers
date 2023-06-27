import os
import numpy as np
import pandas as pd

import data_preparation
import plotting_geo




def main():
    
    # Density dataset generation
    geo_bound = {'lat_min': 38.7,
                 'lat_max': 38.9,
                 'long_min': -122.95,
                 'long_max': -122.65,
                 'depth_min': 0,
                 'depth_max': 14,
                 'step_l': 0.025,
                 'step_d': 0.25}


   # Load files for plotting 

    folder_path_mauro = r'../for_mauro'
    folder_path_geysers = r'../data/the_geysers/'

    wells = np.loadtxt(os.path.join(folder_path_mauro, 'Wells_bentz.txt'))
    coulomb_max = np.loadtxt(os.path.join(folder_path_mauro, 'coulomb_max.txt'))
    time_series = np.array(pd.read_csv(os.path.join(folder_path_geysers, 'time_series.csv'), header=None))
    df_reduced_catalogue = pd.read_csv(os.path.join(folder_path_geysers, 'reduced_catalogue_clustered.csv'), index_col=[0])
    df_density_summary = pd.read_csv(os.path.join(folder_path_mauro, 'full_results_new_matrix.csv'), index_col=[0])
    ics = pd.read_csv(os.path.join(folder_path_mauro, 'ICs.csv'), index_col=[0])
    
    water_injections = pd.read_csv(os.path.join(folder_path_geysers, 'water_injecion_GLOBAL.csv'), names=['date', 'injection'])
    water_injections['year'] = water_injections.apply(lambda row: row.date[-4:], axis=1)
    water_injections['month'] = water_injections.apply(lambda row: row.date[4:6], axis=1)

    df_labels = pd.read_csv(os.path.join(folder_path_geysers, 'cluster_labels'), index_col = [0])
    cluster_labels = np.array([i[0] for i in df_labels.values])
    unique_labels = np.sort(np.unique(cluster_labels))


    print('datasets loaded')

    # time_series_not_zero = time_series[np.sum(time_series, axis = 1) > 0]
    # cluster_labels_not_zero = cluster_labels[np.sum(time_series, axis = 1) > 0]
    # plotting_geo.plot_means_subplots(time_series, cluster_labels, plotted_clusters = [5, 4, 3])
    # b_values = plotting_geo.b_values(df_reduced_catalogue, mag_borders = [1.7, 3], plotted_clusters=[2, 1])  
    plotting_geo.plot_cluster_nodes(df_reduced_catalogue, df_density=df_density_summary, df_labels=df_labels, 
                                     geo_bounds=geo_bound, wells=wells, plotted_clusters=[5, 4, 3])
    # plotting_geo.plot_cluster_nodes_single(df_reduced_catalogue, df_density=df_density_summary, df_labels=df_labels, 
    #                                 geo_bounds=geo_bound, wells=wells, cluster=2)
    # plotting_geo.dth_fft(time_series, cluster_labels, plotted_clusters=[3, 4, 5], injections = water_injections, time_plot=True, psd_plot=False, freq_plot=False)

    # coulomb_max_txt, density data instead of reduced catalogue
    # plotting_geo.coulomb(df_reduced_catalogue, coulomb_max, plotted_clusters = [3,4,5])
    # ICA, water injections
    # plotting_geo.ICA(ICA_FFT_plot=True, ICA_plot=True)

    #timeseries, fullresuls, reduced catalogue, ics, wells
    #plotting_geo.cc_plots(geo_bound, df_reduced_catalogue, df_density_summary, time_series, ics, wells)
    # catalogue = data_preparation.Catalogue(catalogue=df_reduced_catalogue, geo_bounds=geo_bound)
    # slope_means, slope_stds = catalogue.b_value_error(mag_borders=(1.8, 2.4))
    # print(slope_stds)
    # print(slope_means)

if __name__ == '__main__':
    main(do_clustering=False, generate_dataset=False)
