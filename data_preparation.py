import numpy as np
import pandas as pd


"""
This library contains functions used in preparation of datasets used for The Geysers activity. The functions are:
    - density_calculation - deprecated function
    - index_cubes - find the cube (x,y,z) coordinates indexes for all entries in pandas dataset
    - find_closest - find the closest value (or index) from a sorted dataset
    - compile_dataset - compiles the density dataset

"""


def density_calculation(dataset, geo_bound, step_l, step_d):
    """
    Deprecated function

    :param dataset: pandas DF; catalogue geysers file
    :param geo_bound: dictionary; geometric boundaries
    :param step_l: latitude and longitude step
    :param step_d: depth step
    :return:
    """

    years = dataset.year.unique()
    months = dataset.month.unique()
    lat_range = np.arange(geo_bound['lat_min'], geo_bound['lat_max'], step_l)
    long_range = np.arange(geo_bound['long_min'], geo_bound['long_max'], step_l)
    depth_range = np.arange(geo_bound['depth_min'], geo_bound['depth_max'], step_d)
    table_row = []
    results = []

    for y in years:
        print(str(y))
        data_extracted_year = dataset[(dataset['year'] == y)]
        for m in months:
            print(str(m))
            data_extracted_month = data_extracted_year[(data_extracted_year['month'] == m)]
            for lat in lat_range:
                data_extracted_pos = data_extracted_month[(data_extracted_month['Latitude'] >= lat) &
                                                          (data_extracted_month['Latitude'] < (lat + step_l))]
                for long in long_range:
                    data_extracted_pos = data_extracted_pos[(data_extracted_pos['Longitude'] >= long) &
                                                            (data_extracted_pos['Longitude'] < (long + step_l))]
                    for dep in depth_range:
                        data_extracted = data_extracted_pos.loc[(data_extracted_pos['Depthkm'] >= dep) &
                                                                (data_extracted_pos['Depthkm'] < (dep + step_d))]
                        density = len(data_extracted)

                        magmax = data_extracted
                        table_row = [y, m, lat, lat + step_l, long, long + step_l, dep, dep + step_d,
                                     density]

                        results.append(table_row)

    results_columns = ['year', 'month', 'lat', 'lat_step', 'long', 'long_step', 'depth', 'depth_step', 'density']

    results_df = pd.DataFrame(results, columns=results_columns)
    results_df.to_csv('../data/results20062016.csv')
    print('Done')


def index_cubes(dataset, geo_bound, step_l, step_d):
    """
    Add x,y,z (Latitude, Longitude, Depth) cube indices to the dataset column, based on the determined step

    :param dataset: pandas DF; catalogue geysers file
    :param geo_bound: dictionary; geometric boundaries
    :param step_l: latitude and longitude step
    :param step_d: depth step
    :return:
    """

    # Preparation of the cube grid [min, max+step, step]
    lat_range = np.arange(geo_bound['lat_min'], geo_bound['lat_max'] + step_l, step_l)
    long_range = np.arange(geo_bound['long_min'], geo_bound['long_max'] + step_l, step_l)
    depth_range = np.arange(geo_bound['depth_min'], geo_bound['depth_max'] + step_d, step_d)

    # Adding cube x,y,z indexes to the dataframe
    dataset['lat_id'] = dataset.apply(lambda row: find_closest(lat_range, row, "Latitude"), axis=1)
    dataset['long_id'] = dataset.apply(lambda row: find_closest(long_range, row, "Longitude"), axis=1)
    dataset['depth_id'] = dataset.apply(lambda row: find_closest(depth_range, row, "Depthkm"), axis=1)

    return dataset


def find_closest(array, row, col_id):
    """
    Find the closest lower value in a sorted array (possibly there is an easier way to do it)

    :param array: array to search (sorted)
    :param row: df row to use
    :param col_id: column name to search for the value
    :return: array value
    """

    for i in range(len(array)):
        if i + 1 == len(array):
            # Change return to 'i' if ID is needed instead of value
            return i

        if (array[i + 1] - row[col_id]) >= 0:
            return i

    return float('NaN')


def compile_dataset(dataset):
    """
    Compile density dataset based on indexed dataset

    :param dataset: Dataset with location indexes from index_cubes
    :return: Density dataset
    """
    # Find all time steps
    year = dataset.year.unique()
    month = dataset.month.unique()

    # Final dataset preallocation
    dataset_full = pd.DataFrame()

    # Create an aggregated dataframe for each month
    for y in year:
        print("Calculating year: " + str(y))
        for m in month:
            print("Month " + str(m))
            # Extract exact month data
            dataset_month = dataset[(dataset.year == y) & (dataset.month == m)]

            # Data aggregation by location indexes - Magnitude count (density) and mean is found
            dataset_loc = dataset_month.groupby(['lat_id', 'long_id', 'depth_id'])['Magnitude'].agg(['count', 'mean', 'max', 'min'])

            # Dataframe tidy up
            dataset_reset = dataset_loc.reset_index()
            time = {'year': y, 'month': m}
            dataset_reset = dataset_reset.assign(**time)


            # Concat
            dataset_full = pd.concat([dataset_full, dataset_reset], ignore_index=True)

    dataset_full = dataset_full.rename(columns={'count': 'density', 'mean': 'mag_mean', 'max': 'mag_max', 'min': 'mag_min'})

    return dataset_full


def index_1d(row, max_lat, max_long):
    lat_id = row['lat_id']
    long_id = row['long_id']
    depth_id = row['depth_id']
    return lat_id + max_lat * long_id + max_lat * max_long * depth_id


def index_3d(index, max_lat, max_long):
    lat_id = index % max_lat
    long_id = (index / max_lat) % max_long
    depth_id = index / (max_lat * max_long)

    return [lat_id, long_id, depth_id]


def time_series(density_dataset):

    lat_max = density_dataset['lat_id'].max()
    long_max = density_dataset['long_id'].max()

    print(lat_max, long_max)

    density_dataset['index_1D'] = density_dataset.apply(lambda row: index_1d(row, lat_max, long_max), axis=1)
    density_dataset['month_id'] = density_dataset.apply(lambda row: (row.year - 2006)*12 + row.month, axis=1)

    a=1