from typing import List, Union, Any
import numpy as np
import pandas as pd
import plotting_geo
import data_preparation
import matplotlib.pyplot as plt
from GPU_cross_correlation import GPUTensorCC
import pickle
import clustering


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

        # lat_range = np.arange(geo_bound['lat_min'], geo_bound['lat_max'] + geo_bound['step_l'], geo_bound['step_l'])
        # long_range = np.arange(geo_bound['long_min'], geo_bound['long_max'] + geo_bound['step_l'], geo_bound['step_l'])
        # depth_range = np.arange(geo_bound['depth_min'], geo_bound['depth_max'] + geo_bound['step_d'], geo_bound['step_d'])
        #
        # catalogue_geysers['lat'] = catalogue_geysers.apply(lambda row: lat_range[int(row['lat_id'])], axis=1)
        # catalogue_geysers['long'] = catalogue_geysers.apply(lambda row: long_range[int(row['long_id'])], axis=1)
        # catalogue_geysers['depth'] = catalogue_geysers.apply(lambda row: depth_range[int(row['depth_id'])], axis=1)

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
        time_series = np.array(pd.read_csv('../data/time_series.csv'))
        print('time_series - dataset loaded.')

    # cc_single = GPUTensorCC(time_series)
    # single_ccm_matrix, lag_matrix = cc_single.get_result()
    # with open('../data/ccm.csv', 'wb') as file:
    #     pickle.dump(single_ccm_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('../data/ccm.csv', 'rb') as pickle_file:
    #     single_ccm_matrix = pickle.load(pickle_file)
    # # Fill nan with 1 - the nan values are given for 0 value time series
    # single_ccm_matrix = np.nan_to_num(single_ccm_matrix, nan=1)
    with open('../data/mat_distances_norm.pkl', 'rb') as pickle_file:
        mat_distances = pickle.load(pickle_file)
        print('mat_distances - dataset loaded.')
    
    # Perform HAC clustering - save results as PD DF
    if do_clustering:
        labels, linkage = clustering.perform_hac(mat_distances, n_cluster=5, criterion='maxclust', method='ward')
        unique_labels = np.unique(labels)
        cluster_labels = pd.DataFrame(data=labels, columns=['cluster'])
        cluster_labels.to_csv('../data/cluster_labels')
    else:
        cluster_labels = pd.read_csv('../data/cluster_labels')
        unique_labels = np.unique(np.array(cluster_labels['cluster']))

    means, pca = clustering.perform_pca(time_series, cluster_labels)

    # Merge results with original DF
    # catalogue_geysers = pd.read_csv('../data/results_20062016.csv')
    # catalogue_geysers_clustered = catalogue_geysers.merge(cluster_labels, left_on='index_1D', right_index=True)
    # catalogue_geysers_clustered.to_csv('../data/full_results_norm_dist_n_5_test')
    catalogue_geysers_clustered = pd.read_csv('../data/full_results_norm_dist_n_5')
    print('catalogue_geysers_clustered - dataset loaded.')
    injections = pd.read_csv('../data/injections.csv', names=['year', 'month', 'Prati9', 'Prati29', 'sum'])
    injections = injections[injections.year<2017]
    injections['month_id'] = injections.apply(lambda row: int((row.year - 2006) * 12 + row.month), axis=1)

    # # Plotting
    # plotting_geo.plot_cluster(catalogue_geysers_clustered, skip=[1,2,3])
    # plotting_geo.plot_average(time_series, cluster_labels)
    # plotting_geo.plot_all(time_series, cluster_labels, cluster=4)
    # for label in unique_labels:
    #     plotting_geo.plot_average_single(time_series, cluster_labels, cluster=label)
    plotting_geo.contour_plot_cluster(catalogue_geysers_clustered, 2)


    # fig, axs = plt.subplots(pca.n_components, sharex='all')
    # for i, ax in zip(range(pca.n_components), axs.ravel()):
    #     ax.plot(pca.components_[i], label='Component {}'.format(i+1))
    #     ax.legend(loc='upper right')
    # plt.show()

    # plt.plot(injections['month_id'],injections['sum'], label='Sum')
    # # plt.plot(injections['Prati9'], label='Prati9')
    # # plt.plot(injections['Prati29'], label='Prati29')
    # plt.legend()
    # plt.show()
    # a=1

    # fig, ax = plt.subplots(5, sharex='all')
    # ax[0].plot(-pca.components_[0], label='Component {}'.format(i+1))
    # ax[1].plot(pca.components_[1], label='Component {}'.format(i + 1))
    # ax[2].plot(injections['month_id'],injections['sum'], label='Sum')
    # ax[3].plot(means[0], label='Cluster 4')
    # ax[4].plot(means[1], label='Cluster 5')
    # plt.legend()
    # plt.show()
    # a=1

if __name__ == '__main__':

    main(do_clustering=False, load_timeseries=True, generate_dataset=False)