from typing import List, Union, Any
import numpy as np
import pandas as pd
import plotting_geo
import data_preparation
import matplotlib.pyplot as plt
from GPU_cross_correlation import GPUTensorCC
import pickle
import clustering


def main(**kwargs):


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
    if timeseries:
        time_series = data_preparation.time_series(catalogue_geysers)
        np.savetxt("../data/time_series_2km.csv", time_series, delimiter=",")
        with open('../data/mat_distances_norm.pkl', 'rb') as pickle_file:
            mat_distances = pickle.load(pickle_file)
        time_series = np.array(pd.read_csv('../data/time_series.csv'))

    # cc_single = GPUTensorCC(time_series)
    # single_ccm_matrix, lag_matrix = cc_single.get_result()
    # with open('../data/ccm.csv', 'wb') as file:
    #     pickle.dump(single_ccm_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('../data/ccm.csv', 'rb') as pickle_file:
    #     single_ccm_matrix = pickle.load(pickle_file)
    # # Fill nan with 1 - the nan values are given for 0 value time series
    # single_ccm_matrix = np.nan_to_num(single_ccm_matrix, nan=1)
    
    # Perform HAC clustering - save results as PD DF
    if do_clustering:
        labels, linkage = clustering.perform_hac(mat_distances, n_cluster=5, criterion='maxclust')
        unique_labels = np.unique(labels)
        cluster_labels = pd.DataFrame(data=labels, columns=['cluster'])


    # Merge results with original DF
    # catalogue_geysers_clustered = catalogue_geysers.merge(cluster_labels, left_on='index_1D', right_index=True)
    # catalogue_geysers_clustered.to_csv('../data/full_results_norm_dist_n_5')

    catalogue_geysers = pd.read_csv('../data/full_results_norm_dist_n_5')
    print('Dataset loaded.')

    # for clust in unique_labels:
    #     time_series_cluster = time_series[labels == clust, :]
    #     for i in range(5, 15):
    #         plt.plot(time_series_cluster[i])
    #     plt.xlabel('Observation month')
    #     plt.ylabel('Density [per node]')
    #     plt.title('Example time series for cluster '+str(clust))
    #     plt.show()

if __name__ == '__main__':
    main(do_clustering=False, timeseries=False, generate_dataset=False)