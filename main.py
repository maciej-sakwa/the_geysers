import os
import pickle
from typing import Any, List, Union

import numpy as np
import pandas as pd

import data_preparation
import plotting_geo




def main(generate_dataset=True, do_clustering=True):
    
    # Density dataset generation
    geo_bound = {'lat_min': 38.7,
                 'lat_max': 38.9,
                 'long_min': -122.95,
                 'long_max': -122.65,
                 'depth_min': 0,
                 'depth_max': 14,
                 'step_l': 0.025,
                 'step_d': 0.25}

    if generate_dataset:

        # Load catalogue
        df_catalogue_full = pd.read_csv('../data/catalogue.csv', index_col=[0])

        # Reduce the catalogue to the area of interest and index cube nodes with lat_id, long_id, depth_id
        catalogue = data_preparation.Catalogue(df_catalogue_full, geo_bound)
        catalogue.index_cubes()
        df_catalogue = catalogue.reduced_catalogue

        # Perform aggregation based on the cube index
        df_density = catalogue.compile_dataset()

        # Add 1D index to each record of the density dataset
        catalogue.reduced_catalogue['index_1D'] = \
            catalogue.reduced_catalogue.apply(lambda row: data_preparation.index_1d(row, geo_bound['lat_max'], geo_bound['long_max']), axis=1)
        catalogue.reduced_catalogue = catalogue.reduced_catalogue.sort_values(by=['index_1D'])

        catalogue.reduced_catalogue.to_csv('../data/reduced_catalogue.csv')
        print('Dataset saved.')

    # # Load distance matrix


    # Perform HAC clustering - save results as PD DF
    if do_clustering:
        with open('../data/distance_matrix_ok.pickle', 'rb') as pickle_file:
            mat_distances = pickle.load(pickle_file)
        print('mat_distances - dataset loaded.')
        labels, linkage = clustering.perform_hac(mat_distances, n_cluster=5, criterion='maxclust', method='ward')
        unique_labels = np.unique(labels)
        cluster_labels = pd.DataFrame(data=labels, columns=['cluster'])
        cluster_labels.to_csv('../data/cluster_labels.csv')
    else:
        cluster_labels = pd.read_csv('../data/cluster_labels')
        unique_labels = np.unique(np.array(cluster_labels['cluster']))


    # Load files for plotting 

    folder_path_mauro = r'../for_mauro'

    wells = np.loadtxt(os.path.join(folder_path_mauro, 'Wells_bentz.txt'))
    coulomb_max = np.loadtxt(os.path.join(folder_path_mauro, 'coulomb_max.txt'))
    time_series = np.array(pd.read_csv('../data/time_series.csv', header=None))
    df_reduced_catalogue = pd.read_csv('../data/reduced_catalogue_clustered.csv', index_col=[0])
    df_density_summary = pd.read_csv(os.path.join(folder_path_mauro, 'full_results_new_matrix.csv'), index_col=[0])
    ics = pd.read_csv(os.path.join(folder_path_mauro, 'ICs.csv'), index_col=[0])

    print('datasets loaded')


    # plotting_geo.plot_means_subplots(time_series, cluster_labels)
    # b_values = plotting_geo.b_values(df_reduced_catalogue, mag_borders = [1.7, 3], plotted_clusters=[3, 4, 5])  
    # plotting_geo.plot_cluster_nodes(df_reduced_catalogue, df_density_summary, wells)

    # coulomb_max_txt, density data instead of reduced catalogue
    # plotting_geo.coulomb(df_reduced_catalogue, coulomb_max, plotted_clusters = [3,4,5])
    # ICA, water injections
    # plotting_geo.ICA(ICA_FFT_plot=True, ICA_plot=True)

    #timeseries, fullresuls, reduced catalogue, ics, wells
    plotting_geo.cc_plots(geo_bound, df_reduced_catalogue, df_density_summary, time_series, ics, wells)

if __name__ == '__main__':
    main(do_clustering=False, generate_dataset=False)
