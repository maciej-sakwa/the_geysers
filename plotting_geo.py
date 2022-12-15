from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import glob
import numpy as np
import os

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


def make_gif(frame_folder, depth):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("depth_"+str(depth)+".gif", format="GIF", append_images=frames, save_all=True, duration=200, loop=0)