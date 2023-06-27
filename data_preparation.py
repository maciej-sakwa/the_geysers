import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


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

    # Extract data
    lat_id = row['lat_id']
    long_id = row['long_id']
    depth_id = row['depth_id']

    #Retrun index
    return int(lat_id + max_lat * long_id + max_lat * max_long * depth_id)

class Catalogue:
    def __init__(self, catalogue, geo_bounds):
        self.df_catalogue = catalogue
        self.geo_bounds = geo_bounds
        self.reduced_catalogue = self.df_catalogue[(self.df_catalogue['latitude'] >= self.geo_bounds['lat_min']) &
                                               (self.df_catalogue['latitude'] <= self.geo_bounds['lat_max']) &
                                               (self.df_catalogue['longitude'] >= self.geo_bounds['long_min']) &
                                               (self.df_catalogue['longitude'] <= self.geo_bounds['long_max'])].copy()
        
        
        # Define the bins [min, max+step, step]
        self.lat_range = np.arange(self.geo_bounds['lat_min'], self.geo_bounds['lat_max'] + self.geo_bounds['step_l'], self.geo_bounds['step_l'])
        self.long_range = np.arange(self.geo_bounds['long_min'], self.geo_bounds['long_max'] + self.geo_bounds['step_l'], self.geo_bounds['step_l'])
        self.depth_range = np.arange(self.geo_bounds['depth_min'], self.geo_bounds['depth_max'] + self.geo_bounds['step_d'], self.geo_bounds['step_d'])

        self.df_density = None
        self.df_density_periods = None
        self.time_series_array = None
        self.number_of_nodes = None
    


    def __getitem__(self, index):
        return self.reduced_catalogue.iloc[index, :].copy()

    def __del__(self):
        logging.warning('Instance destroyed')

    def __len__(self):
        return len(self.reduced_catalogue)

    def index_cubes(self) -> None:
        """
        Add x,y,z (Latitude, Longitude, Depth) cube indices to the dataset column, based on the determined step
        """
        logging.info('Indexing cubes...')
        # Adding cube x,y,z indexes to the dataframe
        self.reduced_catalogue['lat_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.lat_range, row, "latitude"), axis=1)
        self.reduced_catalogue['long_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.long_range, row, "longitude"), axis=1)
        self.reduced_catalogue['depth_id'] = self.reduced_catalogue.apply(lambda row: find_closest_ind(self.depth_range, row, "depth"), axis=1)
        
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
                dataset_loc = dataset_month.groupby(['lat_id', 'long_id', 'depth_id'])['magnitude'].agg(
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
    

    def compile_dataset_days(self):
        """
        Compile density dataset based on indexed dataset
        :return: Density dataset
        """
        logging.info('Compiling density dataset...')
        # Find all time steps
        periods = self.reduced_catalogue.day_id.unique()

        # Final dataset preallocation
        df_density = pd.DataFrame()

        # Create an aggregated dataframe for each month
        for p in periods:
            logging.debug(f"Calculating period: {p}")
            # Extract exact month data
            dataset_month = self.reduced_catalogue[(self.reduced_catalogue.day_id == p)]

            # Data aggregation by location indexes - Magnitude count (density) and mean is found
            dataset_loc = dataset_month.groupby(['lat_id', 'long_id', 'depth_id'])['magnitude'].agg(
                    ['count', 'mean', 'max', 'min']).reset_index()

            # Dataframe tidy up
            time = {'period': p}
            dataset_loc = dataset_loc.assign(**time)

            # Concat
            df_density = pd.concat([df_density, dataset_loc], ignore_index=True)

        logging.info('Density dataset compiled')
        self.df_density = df_density.rename(
            columns={'count': 'density', 'mean': 'mag_mean', 'max': 'mag_max', 'min': 'mag_min'})


        return self.df_density
    
    
    def time_series(self):
        """
        Create a time series array based on generated density dataset
        :param density_dataset: The indexed density dataset
        :return: Time series
        """
        lat_max = len(self.lat_range)
        long_max = len(self.long_range)

        logging.info('Adding 1D index...')
        self.df_density['index_1D'] = self.df_density.apply(lambda row: index_1d(row, lat_max, long_max), axis=1)
        self.df_density['month_id'] = self.df_density.apply(lambda row: int((row.year - 2006) * 12 + row.month), axis=1)

        # preallocation of results - in the output array each row corresponds to one cube, and each columt to one month
        result = np.zeros((len(self.lat_range)*len(self.long_range)*len(self.depth_range), self.df_density['month_id'].max()))

        logging.info('Generating time series...')
        for i in tqdm(range(self.df_density['index_1D'].max())):
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

                    result[i, ii] = month_data.iloc[0, 3]
        
        self.time_series_array = result.astype('int32')
        

        return self.time_series_array
    
  
    def distance_matrix(self, normalization = True):

        # cant be just pdist?


        if self.time_series_array is None:
            raise ValueError('Time series not compiled')

        # Pre-allocation
        self.number_of_nodes = len(self.time_series_array)
        operational_array = self.time_series_array
        distance_matrix = np.zeros([self.number_of_nodes, self.number_of_nodes])
        
        if normalization:

            # Extract series where the density is higher than 0
            divider = np.max(operational_array, axis = 1)
            divider[divider == 0] = 1
            operational_array = np.divide(operational_array, divider.reshape(self.number_of_nodes, 1))


        # Calculate distance matric
        logging.info(f'Compiling distance matrix:')
        for i in tqdm(range(self.number_of_nodes)):
                
            for j in np.arange(i + 1,self.number_of_nodes):

                signal_i = operational_array[i, :]
                signal_j = operational_array[j, :]

                if np.array_equal(signal_i, signal_j):
                    continue

                dist = np.linalg.norm(signal_i - signal_j)

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        
        return distance_matrix

    
    def calculate_mixtures(self):

        mixtures = []
        mixtures_std = []
        # loop over the clusters
        unique_clusters = np.unique(self.df_density.cluster)

        for uc in unique_clusters:
            df_sum_one_cluster = self.df_density [self.df_density.cluster == uc]
                # loop pver the grid nodes of the clsuter
            list_signals_one_cl_one_node = []
            for gnc in np.unique(df_sum_one_cluster.index_1D):
                df_sum_one_cluster_one_node = df_sum_one_cluster [df_sum_one_cluster.index_1D == gnc]
                norm_factor = max(df_sum_one_cluster_one_node.density)
                list_signals_one_cl_one_node.append(self.time_series_array[gnc,:]/norm_factor)
            array_list_signals_one_cl_one_node = np.array(list_signals_one_cl_one_node)
            mean_mixture = np.average(array_list_signals_one_cl_one_node, axis=0)
            mixtures.append(mean_mixture)
            std_obj = StandardScaler()
            mean_mixture_std = std_obj.fit_transform(mean_mixture.reshape(-1,1))
            mixtures_std.append(mean_mixture_std[:,0])

        return np.array(mixtures).T, np.array(mixtures_std).T

    
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


    def b_value_error(self, mag_borders=(1.8, 3.0), boot_iter = 100):
        """
        Calculates b-value means and std error for the chosen clusters with bootstraping
        :param df_catalogue: the geysers catalogue with cluster labels
        """

        # Bins and clusters definitions
        bin_size = 0.2
        clusters = [1]
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
            for i in range(boot_iter):
                # Randomly select the magnitudes with replacement, save as df to use groupby later
                magnitudes_random = np.random.choice(magnitudes_filtered, size=len(magnitudes_filtered), replace=True)
                df_magnitudes_random = pd.DataFrame(magnitudes_random, columns=['Mag_dist'])

                # Create a logarithmic distribution
                log_distribution = np.log10(df_magnitudes_random.groupby(['Mag_dist'])['Mag_dist'].count())

                x_lr = np.arange(mag_borders[0], mag_borders[1] + bin_size, bin_size)
                y_lr = log_distribution[int(mag_borders[0] / bin_size) - 1: int(mag_borders[1] / bin_size) +1].to_numpy()
                lin_reg = linregress(x=x_lr, y=y_lr)

                slopes.append(-lin_reg.slope)

            # Save results
            slope_means.append(np.mean(slopes))
            slope_stds.append(np.std(slopes))

        return slope_means, slope_stds

    






