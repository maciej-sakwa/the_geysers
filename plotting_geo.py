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


def plot_means_subplots(time_series, cluster_labels) -> None:
    """
    Plots the Means and STD graph of the the chosen domains

    :param time_series:
    :param cluster_labels:

    """

    # Parameters and label definitions
    mpl.rcParams['font.size'] = 14
    arr_labels = np.array(cluster_labels['cluster'])
    clusters = [3, 4, 5]
    legend_labels = ['DTH-A', 'DTH-B', 'DTH-C']
    years = np.arange(2006, 2017, 1)
    tick_labels = [f'\'{item[-2:]}' for item in map(str, years)]
    i = 0
    x = range(time_series.shape[1])

    # Plot definition
    fig, axs = plt.subplots(3)
    for cluster, ax in zip(clusters, axs.ravel()):
        # Normalization and mean calculation
        time_extract = time_series[arr_labels == cluster]
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)
        std = np.std(time_extract_norm.T, axis=0)

        # Mean and STD plots with fill between
        ax.plot(time_norm, label='{}'.format(legend_labels[i]), c='k')
        ax.plot(time_norm + std, "k--", label='SD')
        ax.plot(time_norm - std, 'k--')
        ax.fill_between(x, time_norm - std, time_norm + std, color='#a7c8e2')
        ax.legend(loc='upper left', framealpha=1)

        # X-axis settings
        ax.set_xticks(np.arange(0, 121, 12))
        ax.set_xticklabels(tick_labels)
        ax.set_xlim([0, 125])
        i += 1

    # Plot area settings
    fig.supylabel('Normalized density [p.u.]')
    fig.set_size_inches(8, 7.5)
    plt.xlabel('Observation date [y]')
    plt.tight_layout()
    plt.show()


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


  # fig, ax = plt.subplots(4)
    # for i in range(3):
    #     ax[0].plot(np.array(ind_components.iloc[:, i+1]).T, label='IC: {}'.format(i+1))
    # ax[0].set_title('Independent Components')
    # ax[0].legend(loc='upper right')
    #
    # ax[1].set_title('Injection Rates')
    # ax[1].plot(injections_local.month_id, injections_local.injections/injections_local.injections.max(),
    #            label='Injections_LOCAL')
    # ax[1].plot(injections_global.month_id, injections_global.injections/injections_global.injections.max(),
    #            label='Injections_GLOBAL')
    # ax[1].legend(loc='upper right')
    #
    # ax[2].set_title('Injection FFT')
    # ax[2].plot(xf_local, np.abs(fft_local)[:sample_size_local//2], label='Injections_FFT_LOCAL')
    # ax[2].plot(xf_global, np.abs(fft_global)[:sample_size_global//2], label='Injections_FFT_GLOBAL')
    # ax[2].legend(loc='upper right')
    # years = np.arange(2006,2017,1)
    # heights = injections_yearly[(injections_yearly['date']>2005) & (injections_yearly['date']<2017)].injections
    # ax[3].bar(years, height=heights)
    # ax[3].set_title('Yeary Injections')
    #
    # plt.show()



    # n_clust=np.arange(3, 11, 1)
    # silhuette = [0.41, 0.1675, 0.4477, 0.439, 0.425084, 0.341536, 0.335546, 0.333744]
    #
    # fig_bar = plt.bar(n_clust, silhuette, width = 0.95, color='#a7c8e2')
    # plt.ylabel('Silhouette coefficient')
    # plt.xlabel('Number of clusters')
    # plt.tight_layout
    # plt.show()