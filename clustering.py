from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score
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
