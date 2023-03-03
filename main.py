import math
import os
import pickle
import random
from typing import Any, List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import clustering
import data_preparation
import plotting_geo
from GPU_cross_correlation import GPUTensorCC


def main(generate_dataset=True, load_timeseries=True, do_clustering=True):
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
        catalogue_geysers_full = pd.read_csv('../data/catalogue.csv')

        selected_data = catalogue_geysers_full[(catalogue_geysers_full['Latitude'] >= geo_bound['lat_min']) &
                                               (catalogue_geysers_full['Latitude'] <= geo_bound['lat_max']) &
                                               (catalogue_geysers_full['Longitude'] >= geo_bound['long_min']) &
                                               (catalogue_geysers_full['Longitude'] <= geo_bound['long_max'])]

        catalogue_geysers_indexed = data_preparation.index_cubes(selected_data, geo_bound,
                                                                 step_l=geo_bound['step_l'],
                                                                 step_d=geo_bound['step_d'])

        catalogue_geysers = data_preparation.compile_dataset(catalogue_geysers_indexed)


        # Add 1D index to each record of the density dataset
        lat_max = catalogue_geysers['lat_id'].max()
        long_max = catalogue_geysers['long_id'].max()
        catalogue_geysers['index_1D'] = \
            catalogue_geysers.apply(lambda row: data_preparation.index_1d(row, lat_max, long_max), axis=1)
        catalogue_geysers = catalogue_geysers.sort_values(by=['index_1D'])

        catalogue_geysers.to_csv('../data/results_20062016.csv')
        print('Dataset saved.')

    # Time series generation
    if load_timeseries:
        # time_series = data_preparation.time_series(catalogue_geysers)
        # np.savetxt("../data/time_series_2km.csv", time_series, delimiter=",")
        time_series = np.array(pd.read_csv('../data/time_series.csv', header=None))
        print('time_series - dataset loaded.')

    # # Load distance matrix
    # with open('../data/distance_matrix_ok.pickle', 'rb') as pickle_file:
    #     mat_distances = pickle.load(pickle_file)
    #     print('mat_distances - dataset loaded.')

    # Perform HAC clustering - save results as PD DF
    # if do_clustering:
    #     labels, linkage = clustering.perform_hac(mat_distances, n_cluster=5, criterion='maxclust', method='ward')
    #     unique_labels = np.unique(labels)
    #     cluster_labels = pd.DataFrame(data=labels, columns=['cluster'])
    #     cluster_labels.to_csv('../data/cluster_labels.csv')
    else:
        cluster_labels = pd.read_csv('../data/cluster_labels')
        unique_labels = np.unique(np.array(cluster_labels['cluster']))

    # # Merge results with original DF
    # catalogue_geysers = pd.read_csv('../data/results_20062016.csv')
    # catalogue_geysers_clustered = catalogue_geysers.merge(cluster_labels, left_on='index_1D', right_index=True)
    # catalogue_geysers_clustered.to_csv('../data/full_results_new_matrix.csv')

    df_catalogue = pd.read_csv('../data/reduced_catalogue_clustered.csv', index_col=[0])
    print('catalogue_geysers_clustered - dataset loaded.')

    # bvalue_means, bvalue_stds = data_preparation.b_value_error(df_catalogue)
    #
    # print(f'Mean b-value over iterations: {bvalue_means}')
    # print(f'B-value error over iterations: {bvalue_stds}')
    #
    # plt.errorbar([3, 4, 5], bvalue_means, yerr=bvalue_stds, marker='D', markersize=5, linestyle='None')
    # plt.xticks([3, 4, 5], [f'Cluster {item}' for item in [3, 4, 5]])
    # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    # plt.show()


    for index, file in enumerate(os.listdir('../data/ffts')):
        fft = np.loadtxt(os.path.join('../data/ffts', file))
        x_lr = -np.log10(1 * np.arange(0, len(fft))[1:] / (2 * len(fft)))
        y_lr = np.log10(fft[1:])
        plt.plot(x_lr, y_lr, label=f"FFT {index}")

        lin_reg = linregress(x_lr, y_lr)
        print(f"Slope for fft of IC{index+1} is equal {lin_reg.slope}")

    plt.xscale('log')
    plt.yscale('log')
    plt.show()


    n_clust=np.arange(3, 11, 1)
    silhuette = [0.41, 0.1675, 0.4477, 0.439, 0.425084, 0.341536, 0.335546, 0.333744]

    fig_bar = plt.bar(n_clust, silhuette, width = 0.95, color='#a7c8e2')
    plt.grid(visible=True, axis='y', alpha=0.3, which='major', c='#dbdbdb')
    plt.ylabel('Silhouette coefficient')
    plt.xlabel('Number of clusters')
    plt.tight_layout
    plt.show()

    plotting_geo.plot_means_subplots(time_series, cluster_labels)




if __name__ == '__main__':
    main(do_clustering=False, load_timeseries=True, generate_dataset=False)
