from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np



def density(dataset, year, month):
    """
    Plots a 3d graph of seismic activity density in the studied 'cube'

    :param dataset:    pandas dataframe; contains all the data
    :param year:    int; the year in which you plot, year has to be contained in dataset
    :param month:   int; the year in which you plot, year has to be contained in dataset
    :return:
    """

    selected_data = dataset[(dataset['year'] == year) & (dataset['month'] == month) & (dataset['density'] > 1)]
    lat = np.array(selected_data['lat_id'])
    long = np.array(selected_data['long_id'])
    depth = np.array(selected_data['depth_id'])


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    geo_dens = ax.scatter3D(lat, long, depth, c=selected_data['density'])
    plt.colorbar(geo_dens)
    plt.title('3D density graph for: ' + str(month) +', ' + str(year))
    plt.tight_layout()
    plt.show()


def time(dataset, geo_bound):
    """

    :param dataset:
    :param geo_bound:
    :return:
    """

    selected_data = dataset[(dataset['lat_id'] >= geo_bound['lat_min']) &
                            (dataset['lat_id'] <= geo_bound['lat_max']) &
                            (dataset['long_id'] >= geo_bound['long_min']) &
                            (dataset['long_id'] <= geo_bound['long_max']) &
                            (dataset['depth_id'] >= geo_bound['depth_min']) &
                            (dataset['depth_id'] <= geo_bound['depth_max']) &
                            (dataset['density'] > 10)]


    print(selected_data.head(10))

    plt.scatter(x=selected_data.index, y=selected_data['density'])
    plt.show()

    #x