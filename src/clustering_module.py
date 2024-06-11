"""
Module still under construction, not properly tested


"""


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import MDS
from scipy.cluster import hierarchy
from scipy.spatial import distance
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





class ScipyHierarchy():
    def __init__(self, distance_matrix, reduce = False,  **kwargs) -> None:
        self.distance_matrix = distance_matrix
        self.threshold = kwargs.get('threshold', 5)
        self.method = kwargs.get('method', 'ward')
        self.metric = kwargs.get('metric', 'euclidean')
        self.combination_dictionary = kwargs.get('combinations_dictionary', None)

        # Operational variables
        self.labels_ = None
        self.reduced_distance_matrix = None
        self.linkage = None

        # If the matrix is a dimentionality square matrix - reduce dimensionality
        if (self.distance_matrix.shape[0] == self.distance_matrix.shape[1]) & reduce:
            self.reduce_distance_matrix()
        elif self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            self.reduced_distance_matrix = self.distance_matrix

        # Update the init parameters
        if self.combination_dictionary is not None:
            for key, value in self.combination_dictionary.items():
                setattr(self, key, value)

        # Pick the clustering criterion based on the threshold
        if self.threshold > 1:
            self.criterion = 'maxclust'
        else:
            self.criterion = 'distance'
        

    def cluster(self):
        # Condence the distance matrix if it is symmetric (squareform)
        if np.all(np.abs(self.distance_matrix - self.distance_matrix.T) < 1e-5):
            condenced_matrix = distance.squareform(self.distance_matrix)
        else: 
            condenced_matrix = self.distance_matrix

        #Perform clustering
        self.linkage = hierarchy.linkage(condenced_matrix, method=self.method,  metric=self.metric)
        self.labels_ = hierarchy.fcluster(self.linkage, t=self.threshold, criterion=self.criterion)
        
        
    def score(self):
        
        if self.labels_ is None:
            raise ValueError('Clustering not performed')
        
        scores = []
        
        # Find silhouette score
        scores = silhouette_score(self.distance_matrix, self.labels_)

        # Find CH and DB scores if feature matrix is passed
        if self.reduced_distance_matrix is not None:
            scores.append(calinski_harabasz_score(self.distance_matrix, self.labels_))
            scores.append(davies_bouldin_score(self.distance_matrix, self.labels_))

        return scores
    

    def dendrogram(self):

        plt.figure(figsize=(8, 6))
        hierarchy.dendrogram(self.linkage)
        plt.ylabel('Copthenetic distance')
        plt.xlabel('Cluster')
        plt.show()

    def reduce_distance_matrix(self):

        # Reduce matrices if distance matrix is passed
        embedding = MDS(n_components=3, random_state=0, dissimilarity='precomputed', max_iter=100, normalized_stress='auto')
        self.reduced_distance_matrix = embedding.fit_transform(self.distance_matrix)


class EvaluateClustering():
    def __init__(self, clustering, X):
        self.clustering = clustering # clustering object to be evaluated
        self.X = X # feature space or distance matrix

    def grid_search_clustering(self, parameters: dict) -> pd.DataFrame:

        # Set up the grid search matrix
        # Create a list of key-value pairs using nested list comprehension
        combinations_list = [[key, val] for key, val in parameters.items()]
    
        # Create a dictionary of key-value combinations using dictionary comprehension
        combinations_dict = {key: product([key], val) for key, val in combinations_list}
    
        # Compute cross key combinations. Output is a list of dictionaries with parameters to set in clustering
        combinations = list(product(*combinations_dict.values()))
        combinations = [dict((x,y) for x, y in combination) for combination in combinations]

        df_scored = pd.DataFrame()

        # Iterate through the combination dictionaries and check the score
        for comb in tqdm(combinations):
            keys = list(comb.keys())
            values = list(comb.values())

            row = pd.DataFrame([values], columns=keys)

            evaluated_clustering = self.clustering(self.X, combinations_dictionary = comb)
            evaluated_clustering.cluster()
            scores = evaluated_clustering.score()

            if isinstance(scores, list):
                row['score_sil'] = scores[0]
                row['score_ch'] = scores[1]
                row['score_db'] = scores[2]
            else:
                row['score_sil'] = scores

            df_scored = pd.concat([df_scored, row], ignore_index=True)

        return df_scored
    

class DistanceMatrix():
    def __init__():
        pass

