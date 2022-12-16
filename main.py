from typing import List, Union, Any
import numpy as np
import pandas as pd
import plotting_geo
import data_preparation
import matplotlib.pyplot as plt
from GPU_cross_correlation import GPUTensorCC
import pickle
import clustering

if __name__ == '__main__':

    # density dataset generation

    # geo_bound = {'lat_min': 38.7,
    #              'lat_max': 38.9,
    #              'long_min': -122.95,
    #              'long_max': -122.65,
    #              'depth_min': 0,
    #              'depth_max': 14}
    #
    # catalogue_geysers_full = pd.read_csv('../data/catalogue.csv')
    #
    # catalogue_geysers = catalogue_geysers_full[(catalogue_geysers_full['Latitude'] >= geo_bound['lat_min']) &
    #                                             (catalogue_geysers_full['Latitude'] <= geo_bound['lat_max']) &
    #                                             (catalogue_geysers_full['Longitude'] >= geo_bound['long_min']) &
    #                                             (catalogue_geysers_full['Longitude'] <= geo_bound['long_max'])]
    #
    # catalogue_geysers_indexed = data_preparation.index_cubes(catalogue_geysers, geo_bound, step_l=0.025, step_d=2)
    # final_dataset = data_preparation.compile_dataset(catalogue_geysers_indexed)
    # final_dataset.to_csv('../data/results_indexed_20062016_2km.csv')
    catalogue_geysers = pd.read_csv('../data/results_indexed_20062016_2km.csv')
    
    # Add 1D index to each record of the density dataset
    lat_max = catalogue_geysers['lat_id'].max()
    long_max = catalogue_geysers['long_id'].max()
    catalogue_geysers['index_1D'] = catalogue_geysers.apply(lambda row: data_preparation.index_1d(row, lat_max, long_max),
                                                            axis=1)
    catalogue_geysers = catalogue_geysers.sort_values(by=['index_1D'])

    # time series generation

    # time_series = data_preparation.time_series(catalogue_geysers)
    # np.savetxt("../data/time_series_2km.csv", time_series, delimiter=",")
    with open('../data/mat_distances_norm.pkl', 'rb') as pickle_file:
        mat_distances = pickle.load(pickle_file)
    timeseries = np.array(pd.read_csv('../data/time_series.csv'))

    # cc_single = GPUTensorCC(timeseries)
    # single_ccm_matrix, lag_matrix = cc_single.get_result()
    # with open('../data/ccm.csv', 'wb') as file:
    #     pickle.dump(single_ccm_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../data/ccm.csv', 'rb') as pickle_file:
        single_ccm_matrix = pickle.load(pickle_file)


    # Fill nan with 1 - the nan values are given for 0 value time series 
    single_ccm_matrix = np.nan_to_num(single_ccm_matrix, nan=1)
    
    # Perform HAC clustering - save results as PD DF
    labels = clustering.perform_hac(mat_distances)
    unique_labels = np.unique(labels)
    cluster_labels = pd.DataFrame(data=labels, columns = ['cluster'])


    # Merge results with original DF
    catalogue_geysers_clustered = catalogue_geysers.merge(cluster_labels, left_on='index_1D', right_index=True)
    print(unique_labels)
    for clust in unique_labels:
        time_series_cluster = timeseries[labels == clust, :]
        for i in range(5, 15):
            plt.plot(time_series_cluster[i])
        plt.xlabel('Observation month')
        plt.ylabel('Density [per node]')
        plt.title('Example time series for cluster '+str(clust))
        plt.show()