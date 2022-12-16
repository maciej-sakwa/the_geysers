from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import numpy as np
import matplotlib.pyplot as plt


def perform_hac(ccm, criterion='single'):

    distance = 1 - ccm
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=20,
                                         metric='euclidean',
                                         linkage='ward').fit(ccm)

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
    
    # dendrogram(linkage_matrix, truncate_mode="level", p=5)
    # plt.show()
    return clustering.labels_
