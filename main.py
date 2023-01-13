from typing import List, Union, Any
import numpy as np
import pandas as pd
import plotting_geo
import data_preparation
import matplotlib.pyplot as plt
from GPU_cross_correlation import GPUTensorCC
import pickle
import clustering


def main(generate_dataset=True, load_timeseries=True, do_clustering=True, load_injections=True):
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

    # Load distance matrix
    with open('../data/distance_matrix_ok.pickle', 'rb') as pickle_file:
        mat_distances = pickle.load(pickle_file)
        print('mat_distances - dataset loaded.')

    # Perform HAC clustering - save results as PD DF
    if do_clustering:
        labels, linkage = clustering.perform_hac(mat_distances, n_cluster=5, criterion='maxclust', method='ward')
        unique_labels = np.unique(labels)
        cluster_labels = pd.DataFrame(data=labels, columns=['cluster'])
        cluster_labels.to_csv('../data/cluster_labels.csv')
    else:
        cluster_labels = pd.read_csv('../data/cluster_labels')
        unique_labels = np.unique(np.array(cluster_labels['cluster']))

    # Merge results with original DF
    catalogue_geysers = pd.read_csv('../data/results_20062016.csv')
    catalogue_geysers_clustered = catalogue_geysers.merge(cluster_labels, left_on='index_1D', right_index=True)
    catalogue_geysers_clustered.to_csv('../data/full_results_new_matrix.csv')

    # catalogue_geysers_clustered = pd.read_csv('../data/full_results_norm_dist_n_5')
    print('catalogue_geysers_clustered - dataset loaded.')

    if load_injections:
        # Load injections
        injections = pd.read_csv('../data/injections.csv', names=['year', 'month', 'Prati9', 'Prati29', 'sum'])
        injections = injections[injections.year < 2017]
        injections['month_id'] = injections.apply(lambda row: int((row.year - 2006) * 12 + row.month), axis=1)
        print('injections - dataset loaded.')
        # injections_history = pd.read_csv('../data/water_injection.csv', names=['year', 'injection'])
        # injections_history['month_id'] = injections_history.apply(lambda row: int((row['year'] - 2006) * 12))
        # injections_history_selected = injections_history[2006 <= injections_history.year <= 2016]
        # coordinates = pd.read_csv('../data/injection_wells_coordinates.csv', names=['long', 'lat'])

    means, means_std, pca = clustering.perform_pca(time_series, cluster_labels)
    np.savetxt('../data/mixtures_new.csv', means, delimiter=',')
    np.savetxt('../data/mixtures_std_new.csv', means_std, delimiter=',')
    # pca_fitted_std = pca.fit_transform(means_std)
    # for i in range(len(pca_fitted_std.T)):
    #     plt.plot(pca_fitted_std.T[i], label='Principal component {}'.format(i))
    # plt.legend()
    # plt.show()

    plotting_geo.volume_plot(catalogue_geysers_clustered[catalogue_geysers_clustered['cluster']>2], geo_bound)



if __name__ == '__main__':
    main(do_clustering=True, load_timeseries=True, generate_dataset=False, load_injections=False)
