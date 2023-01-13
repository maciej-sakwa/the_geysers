from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_preparation import find_closest_ind
from PIL import Image
import glob
import numpy as np
import os
import plotly.graph_objects as go

#General plot parameters
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams['ases.spines.right'] = False



def density(dataset, year, month):
    """
    Plots a 3d graph of seismic activity density in the studied 'cube'

    :param dataset:    pandas dataframe; contains all the data
    :param year:    int; the year in which you plot, year has to be contained in dataset
    :param month:   int; the year in which you plot, year has to be contained in dataset
    :return:
    """

    selected_data = dataset[(dataset['year'] == year) & (dataset['month'] == month)]
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


def contour_plot(density_dataset, d):

    depth_levels = np.sort(density_dataset.depth_id.unique())
    print(depth_levels)
    density_level = density_dataset[density_dataset['depth_id'] == depth_levels[d]]
    years = density_dataset.year.unique()
    dir_path='../images'
    top = density_level['density'].max()
    bottom = density_level['density'].min()

    for year in years:
        yearly_data = density_level.groupby(['lat_id', 'long_id', 'year'])['density'].sum().reset_index()
        selected_data = yearly_data[yearly_data['year'] == year].reset_index()
        density_matrix = np.zeros((selected_data.lat_id.max() + 1, selected_data.long_id.max() + 1))

        for index, row in selected_data.iterrows():
            i = row['lat_id']
            j = row['long_id']
            density_matrix[i, j] = row['density']

        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(density_matrix)
        plt.colorbar(cp)
        plt.title('Contour plot for depth: ' + str(2 * depth_levels[d]) + ' to: ' + str(2 * depth_levels[d] + 2)
                  + ', year: ' +str(year))
        file_path = 'depth_'+str(2*depth_levels[d])
        file_name = 'contour_' + str(2* depth_levels[d]) + '_' + str(year)
        file_path = os.path.join(dir_path, file_path, file_name)
        plt.savefig(file_path)


def make_gif(frame_folder, depth: int):
    """
    Make gif out of density contour plots

    :param frame_folder: path to folder containing contour plots
    :param depth: int, selected depth level
    :return:
    """

    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("depth_"+str(depth)+".gif", format="GIF", append_images=frames, save_all=True, duration=200, loop=0)


def plot_cluster(dataset, skip=[0]):
    plt.style.use('ggplot')
    selected_data = dataset[(dataset['density']>0)]
    plot_data = selected_data.groupby(['lat', 'long', 'depth'])['cluster'].mean()
    plot_data = plot_data.reset_index()

    clusters = np.unique(np.array(plot_data['cluster']))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for c in clusters:

        if c in skip:
            continue

        cluster_data = plot_data[plot_data['cluster']==c]
        lat_c = np.array(cluster_data['lat'])
        long_c = np.array(cluster_data['long'])
        depth_c = np.array(cluster_data['depth'])

        ax.scatter3D(lat_c, long_c, -depth_c, label="Cluster {}".format(c))

    plt.title('3D cluster graph')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_average(timeseries, labels, skip=[0]):
    plt.style.use('ggplot')
    arr_labels = np.array(labels['cluster'])
    unique_labels = np.unique(arr_labels)

    for ul in unique_labels:

        if c in skip:
            continue

        time_extract = timeseries[arr_labels == ul]
        print(len(time_extract))
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)

        plt.plot(time_norm, label='Cluster {}'.format(ul))

    plt.legend()
    plt.title('Averaged normalised time series for created clusters')
    plt.xlabel('Month')
    plt.ylabel('Density [p.u.]')
    plt.show()


def plot_average_single(timeseries, labels, cluster):
    """

    """
    arr_labels = np.array(labels['cluster'])
    x = range(timeseries.shape[1])


    time_extract = timeseries[arr_labels == cluster]
    time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
    time_norm = np.average(time_extract_norm.T, axis=0)
    std = np.std(time_extract_norm.T, axis=0)

    fig, ax = plt.subplots()
    ax.plot(time_norm, label='Mean for cluster {}'.format(cluster), c='k')
    ax.plot(time_norm+std, "k--", label='Standard deviation')
    ax.plot(time_norm-std, 'k--')
    ax.fill_between(x, time_norm-std, time_norm+std, color='#56B4E9', alpha=0.4)

    plt.legend()
    fig.set_size_inches(10, 5)
    plt.title('Average normalised time series for cluster {}'.format(cluster))
    plt.xlabel('Observation month')
    plt.ylabel('Density [p.u.]')
    plt.show()


def contour_plot_cluster(dataset, depth):

    dataset_depth = dataset[dataset['depth'] == depth]
    cluster_matrix = np.zeros((dataset_depth.lat_id.max() + 1, dataset_depth.long_id.max() + 1))

    for index, row in dataset_depth.iterrows():
        i = int(row['lat_id'])
        j = int(row['long_id'])
        cluster_matrix[i, j] = row.loc['cluster']

    fig, ax = plt.subplots(1, 1)
    cp = ax.scatter(x=dataset_depth['lat_id'], y=dataset_depth['long_id'], c=dataset_depth['cluster'])
    plt.colorbar(cp)
    plt.show()


    # # Plotting
    # plotting_geo.plot_cluster(catalogue_geysers_clustered, skip=[1,2,3])
    # plotting_geo.plot_average(time_series, cluster_labels)
    # plotting_geo.plot_all(time_series, cluster_labels, cluster=4)
    # for label in unique_labels:
    #     plotting_geo.plot_average_single(time_series, cluster_labels, cluster=label)
    # plotting_geo.contour_plot_cluster(catalogue_geysers_clustered, 2)

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

    # fig, ax = plt.subplots(3, sharex='all')
    # ax[0].plot(-pca.components_[0], label='Component 1')
    # # ax[0].plot(pca.components_[1], label='Component 2')
    # ax[0].legend()
    # ax[1].plot(injections['month_id'], injections['sum'], label='Sum')
    # ax[1].plot(injections['month_id'], injections['Prati9'], label='Prati9')
    # ax[1].plot(injections['month_id'], injections['Prati29'], label='Prati29')
    # ax[1].legend()
    # ax[2].plot(means[0], label='Cluster 4')
    # ax[2].plot(means[1], label='Cluster 5')
    # ax[2].legend()
    # plt.show()


def volume_plot(density_df, geo_bound):

    density_grouped = density_df.groupby(['lat', 'long', 'depth'])['cluster'].mean()
    density_selected = density_grouped.reset_index()

    lat_unique = np.sort(density_selected['lat'].unique())
    long_unique = np.sort(density_selected['long'].unique())
    depth_unique = np.sort(density_selected['depth'].unique())

    density_selected['lat_id'] = density_selected.apply(lambda row: find_closest_ind(lat_unique, row, "lat"), axis=1)
    density_selected['long_id'] = density_selected.apply(lambda row: find_closest_ind(long_unique, row, "long"), axis=1)
    density_selected['depth_id'] = density_selected.apply(lambda row: find_closest_ind(depth_unique, row, "depth"), axis=1)


    X, Y, Z = np.meshgrid(lat_unique, long_unique, depth_unique)
    values = np.zeros((len(lat_unique), len(long_unique), len(depth_unique)))
    print(range(len(lat_unique)))
    print(range(len(long_unique)))
    print(range(len(depth_unique)))

    for x in range(len(lat_unique)):
        for y in range(len(long_unique)):
            for z in range(len(depth_unique)):
                data = density_selected[(density_selected['lat_id'] == x) &
                                        (density_selected['long_id'] == y) &
                                        (density_selected['depth_id'] == z)]
                if data.shape[0] == 0:
                    continue
                else:
                    row = data.to_numpy()
                    values[x, y, z] = row[0, 3]
                    print(str(values[x, y, z]), str(row[0,3]))



    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=-Z.flatten(),
        value=values.flatten(),
        opacity=0.5,
        isomin=0,
        isomax=5,
        surface_count=6,
        colorbar_nticks = 6
    ))

    fig.show()
