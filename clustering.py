from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import FastICA, PCA
from scipy.cluster import hierarchy
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt


def perform_hac_sk(ccm, criterion='single'):
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=500,
                                         metric='euclidean',
                                         linkage='ward').fit(cond_matrix)

    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [clustering.children_, clustering.distances_, counts]
    ).astype(float)

    hierarchy.dendrogram(linkage_matrix, truncate_mode="level", p=7)
    plt.title('Dendrogram for % clusters' % n_clusters)
    plt.show()
    return clustering.labels_


def perform_hac(matrix, method='ward', threshold=0.15, n_cluster=5, criterion='distance'):
    """

    """
    cond_matrix = distance.squareform(matrix)
    Z = hierarchy.linkage(cond_matrix, method=method)
    # threshold_dist = np.amax(hierarchy.cophenet(Z)) * threshold
    # labels = hierarchy.fcluster(Z, t=threshold_dist, criterion=criterion)
    labels = hierarchy.fcluster(Z, t=n_cluster, criterion=criterion)

    # # Plot dendrogram
    # hierarchy.dendrogram(Z, truncate_mode="level", p=8)
    # plt.title('Dendrogram for {} clusters'.format(n_cluster))
    # plt.ylabel('Cophenetic distance')
    # plt.xlabel('Nodes')
    # plt.show()

    # Clusterig scores
    # ch = calinski_harabasz_score(cond_matrix, labels)
    sil = silhouette_score(matrix, labels)
    print("Score: {}".format(sil))

    return labels, Z


def perform_ica(time_series, labels, n_components=2, main_clusters=[4,5]):
    """


    """

    if not n_components == len(main_clusters):
        print('Error! Number of components mismatched with chosen clusters.')
        return

    # Extract unique labels from labels
    arr_labels = np.array(labels['cluster'])
    unique_labels = np.unique(arr_labels)

    # Pre define averages matrix
    averages = []
    averages_ica = []

    # Calculate normalized mean for all clusters
    for ul in unique_labels:

        # Extract and transform the data from time series dataset
        # Calculate a normalized average for members of a given cluster
        time_extract = time_series[arr_labels == ul]
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)
        # Append to averages list
        averages.append(time_norm)


    averages_arr = np.array(averages)
    averages = np.delete(averages_arr, 0, 1)

    # Extract the chosen clusters
    for x in main_clusters:
        index = x-1
        averages_ica.append(averages[index])



    transformer = FastICA(n_components=2, random_state=0, whiten='arbitrary-variance')
    averages_transformed = transformer.fit_transform(np.array(averages_ica))
    print('Result')
    print(averages_transformed)
    print('Components')
    print(transformer.components_)
    print('Mixing matrix')
    print(transformer.mixing_)

    components = transformer.components_
    for c in range(len(components)):
        plt.plot(components[c], label='Components {}'.format(c+1))

    # plt.yscale('log')
    plt.legend()
    plt.show()

    for c in range(len(components)):
        plt.plot(averages_transformed[c], label='Components {}'.format(c+1))
    # plt.yscale('log')
    plt.legend()
    plt.show()

    a=1


def perform_pca(time_series, labels, n_components=2, main_clusters=[4,5]):
    """


    """

    if not n_components == len(main_clusters):
        print('Error! Number of components mismatched with chosen clusters.')
        return

    # Extract unique labels from labels
    arr_labels = np.array(labels['cluster'])
    unique_labels = np.unique(arr_labels)

    # Pre define averages matrix
    averages = []
    averages_ica = []

    # Calculate normalized mean for all clusters
    for ul in unique_labels:

        # Extract and transform the data from time series dataset
        # Calculate a normalized average for members of a given cluster
        time_extract = time_series[arr_labels == ul]
        time_extract_norm = time_extract.T / np.max(time_extract, axis=1)
        time_norm = np.average(time_extract_norm.T, axis=0)
        # Append to averages list
        averages.append(time_norm)


    averages_arr = np.array(averages)
    averages = np.delete(averages_arr, 0, 1)

    # Extract the chosen clusters
    for x in main_clusters:
        index = x-1
        averages_ica.append(averages[index])



    transformer = PCA(n_components=2, random_state=0, whiten=False)
    averages_transformed = transformer.fit_transform(np.array(averages_ica))
    print('Result')
    print(averages_transformed)
    print('Components')
    print(transformer.components_)

    return averages_ica, transformer
