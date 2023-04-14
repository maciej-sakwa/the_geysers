import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('data_prep')
log.setLevel(logging.DEBUG)


def find_closest_ind(array:np.ndarray, row:pd.Series, col_id:int) -> int:

    for i in range(len(array)):
        if i + 1 == len(array):
            # Change return to 'i' if ID is needed instead of value
            return i

        if (array[i + 1] - row[col_id]) >= 0:
            return i

    return float('NaN')


def index_1d(row:pd.Series, max_lat:int, max_long:int) -> int:
    
    """
    Adds a 1D index for a gicen row based on the dimensions of the dataset
    :param row: Pandas DF row
    :param max_lat: max latitude in the dataset
    :param max_long: max longitude in the dataset
    :return: index
    """

    # Extract data
    lat_id = row['lat_id']
    long_id = row['long_id']
    depth_id = row['depth_id']

    #Retrun index
    return int(lat_id + max_lat * long_id + max_lat * max_long * depth_id)


class Catalogue:
    def __init__(self, catalogue, geo_bounds):
        self.catalogue = catalogue
        self.geo_bounds = geo_bounds
        self.reduced_catalogue = self.catalogue[(self.catalogue['Latitude'] >= self.geo_bounds['lat_min']) &
                                               (self.catalogue['Latitude'] <= self.geo_bounds['lat_max']) &
                                               (self.catalogue['Longitude'] >= self.geo_bounds['long_min']) &
                                               (self.catalogue['Longitude'] <= self.geo_bounds['long_max'])].copy()
        
        
        # Define the bins [min, max+step, step]
        self.lat_range = np.arange(self.geo_bounds['lat_min'], self.geo_bounds['lat_max'] + self.geo_bounds['step_l'], self.geo_bounds['step_l'])
        self.long_range = np.arange(self.geo_bounds['long_min'], self.geo_bounds['long_max'] + self.geo_bounds['step_l'], self.geo_bounds['step_l'])
        self.depth_range = np.arange(self.geo_bounds['depth_min'], self.geo_bounds['depth_max'] + self.geo_bounds['step_d'], self.geo_bounds['step_d'])
    
    def __getitem__(self, index):
        return self.reduced_catalogue.iloc[index, :].copy()


    def __del__(self):
        logging.warning('Instance destroyed')

    def index_cubes(self) -> None:
        """
        Add x,y,z (Latitude, Longitude, Depth) cube indices to the dataset column, based on the determined step
        """
        logging.info('Indexing cubes...')
        # Adding cube x,y,z indexes to the dataframe
        self.reduced_catalogue['lat_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.lat_range, row, "Latitude"), axis=1)
        self.reduced_catalogue['long_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.long_range, row, "Longitude"), axis=1)
        self.reduced_catalogue['depth_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.depth_range, row, r"Depthkm"), axis=1)
        
        return

    def compile_dataset(self):
        """
        Compile density dataset based on indexed dataset
        :return: Density dataset
        """
        logging.info('Compiling density dataset...')
        # Find all time steps
        year = self.reduced_catalogue.year.unique()
        month = self.reduced_catalogue.month.unique()

        # Final dataset preallocation
        df_density = pd.DataFrame()

        # Create an aggregated dataframe for each month
        for y in year:
            for m in month:
                logging.debug(f"Calculating month: {m}/{y}")
                # Extract exact month data
                dataset_month = self.reduced_catalogue[(self.reduced_catalogue.year == y) & (self.reduced_catalogue.month == m)]

                # Data aggregation by location indexes - Magnitude count (density) and mean is found
                dataset_loc = dataset_month.groupby(['lat_id', 'long_id', 'depth_id'])['Magnitude'].agg(
                    ['count', 'mean', 'max', 'min']).reset_index()

                # Dataframe tidy up
                time = {'year': y, 'month': m}
                dataset_loc = dataset_loc.assign(**time)

                # Concat
                df_density = pd.concat([df_density, dataset_loc], ignore_index=True)

        logging.info('Density dataset compiled')
        self.df_density = df_density.rename(
            columns={'count': 'density', 'mean': 'mag_mean', 'max': 'mag_max', 'min': 'mag_min'})

        return self.df_density
    

    #paste clustering here
    def time_series(self):
        """
        Create a time series array based on generated density dataset
        :param density_dataset: The indexed density dataset
        :return: Time series
        """
        lat_max = max(self.lat_range)
        long_max = max(self.long_range)

        self.df_density['index_1D'] = self.df_density.apply(lambda row: index_1d(row, lat_max, long_max), axis=1)
        self.df_density['month_id'] = self.df_density.apply(lambda row: int((row.year - 2006) * 12 + row.month), axis=1)

        # preallocation of results - in the output array each row corresponds to one cube, and each columt to one month
        result = np.zeros((len(self.lat_range)*len(self.long_range)*len(self.depth_range), self.df_density['month_id'].max()))

        for i in range(self.df_density['index_1D'].max()):
            logging.debug(f"Time series: {i}")

            # Skip loop if there is no density in given cube
            if self.df_density[self.df_density['index_1D'] == i] is None:
                continue

            selected_data = self.df_density[self.df_density['index_1D'] == i]

            for ii in range(self.df_density['month_id'].max()):

                if selected_data[selected_data['month_id'] == ii].empty:
                    continue
                else:
                    month_data = selected_data[selected_data['month_id'] == ii]

                    result[i, ii] = month_data.iloc[0, 4]
        
        self.time_series = result.astype('int32')

        return self.time_series
    
    
    
    
    def perform_hac(matrix, method='ward', threshold=0.15, n_cluster=5, criterion='distance'):
        """
        Perform HAC clustering with scipy algorithm based on the distance matrix
        """


        cond_matrix = distance.squareform(matrix)
        Z = hierarchy.linkage(cond_matrix, method=method)
        # threshold_dist = np.amax(hierarchy.cophenet(Z)) * threshold
        # labels = hierarchy.fcluster(Z, t=threshold_dist, criterion=criterion)
        labels = hierarchy.fcluster(Z, t=n_cluster, criterion=criterion)

        # Clustering scores
        # ch = calinski_harabasz_score(cond_matrix, labels)
        sil = silhouette_score(matrix, labels)
        print("Score: {}".format(sil))

        return labels, Z


    def b_value_error(self, mag_borders=(1.8, 3.0)):
        """
        Calculates b-value means and std error for the chosen clusters with bootstraping
        :param df_catalogue: the geysers catalogue with cluster labels
        """

        # Bins and clusters definitions
        bin_size = 0.2
        clusters = [3, 4, 5]
        bins = np.arange(-1, 5, bin_size)
        bin_labels = [f'{item: .1f}' for item in bins]

        # Result bins
        slope_means = []
        slope_stds = []

        for item in clusters:

            # Result bins
            slopes = []

            # Filter the catalogue for the Earthquakes belonging to chosen cluster
            df_filtered = self.reduced_catalogue.query(f'cluster == {item}').copy()
            # Cut the dataset into bins based on the Magnitude, add the bin label as new column (an extract it as np array)
            magnitudes_filtered = pd.cut(df_filtered['Magnitude'], bins, labels=bin_labels[:-1]).to_numpy()

            # Bootstrap iterations
            for i in range(100):
                # Randomly select the magnitudes with replacement, save as df to use groupby later
                magnitudes_random = np.random.choice(magnitudes_filtered, size=len(magnitudes_filtered), replace=True)
                df_magnitudes_random = pd.DataFrame(magnitudes_random, columns=['Mag_dist'])

                # Create a logarithmic distribution
                log_distribution = np.log10(df_magnitudes_random.groupby(['Mag_dist'])['Mag_dist'].count())

                x_lr = np.arange(mag_borders[0], mag_borders[1] + bin_size, bin_size)
                y_lr = log_distribution[int(mag_borders[0] / bin_size)-1: int(mag_borders[1] / bin_size)].to_numpy()
                lin_reg = linregress(x=x_lr, y=y_lr)

                slopes.append(-lin_reg.slope)

            # Save results
            slope_means.append(np.mean(slopes))
            slope_stds.append(np.std(slopes))

        return slope_means, slope_stds

    






