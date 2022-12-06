from typing import List, Union, Any

import pandas as pd
import plotting_geo
import data_preparation
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # geo_bound = {'lat_min': 38.7,
    #              'lat_max': 38.9,
    #              'long_min': -122.95,
    #              'long_max': -122.65,
    #              'depth_min': 0,
    #              'depth_max': 15}
    #
    # catalogue_geysers_full = pd.read_csv('../data/catalogue.csv')
    #
    # catalogue_geysers = catalogue_geysers_full[(catalogue_geysers_full['Latitude'] >= geo_bound['lat_min']) &
    #                                             (catalogue_geysers_full['Latitude'] <= geo_bound['lat_max']) &
    #                                             (catalogue_geysers_full['Longitude'] >= geo_bound['long_min']) &
    #                                             (catalogue_geysers_full['Longitude'] <= geo_bound['long_max'])]
    #
    # catalogue_geysers_indexed = data_preparation.index_cubes(catalogue_geysers, geo_bound, step_l=0.025, step_d=0.25)
    # final_dataset = data_preparation.compile_dataset(catalogue_geysers_indexed)
    # final_dataset.to_csv('../data/results_indexed_20062016.csv')

    # catalogue_geysers = pd.read_csv('../data/results_indexed_20062016.csv')
    #
    # data_preparation.time_series(catalogue_geysers)

    timeseries = pd.read_csv('../data/time_series.csv')
    plt.plot(timeseries.iloc[51,:])
    plt.plot(timeseries.iloc[75,:])
    plt.plot(timeseries.iloc[74, :])
    plt.show()
