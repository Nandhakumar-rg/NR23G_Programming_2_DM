import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

def data_index_function(data, I, J):
    """
    Calculate the dissimilarity (Euclidean distance) between two clusters.

    :param data: Dataset from which distances are computed.
    :param I: First cluster of indices.
    :param J: Second cluster of indices.
    :return: Dissimilarity between clusters I and J.
    """
    # Extract points for each cluster
    cluster_I = data[I, :]
    cluster_J = data[J, :]

    # Calculate all pairwise distances between points in clusters I and J
    distances = scipy.spatial.distance.cdist(cluster_I, cluster_J, metric='euclidean')

    # Return the minimum distance (single linkage)
    return np.min(distances)

def compute():
    answers = {}

    # A. Load the dataset
    mat = io.loadmat('hierarchical_toy_data.mat')
    data = mat['X']  # Assuming the data is stored under the key 'X'
    answers["3A: toy data"] = data

    # B. Create a linkage matrix and plot a dendrogram
    Z = linkage(data, method='single')
    answers["3B: linkage"] = Z
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
      # Saving the plot as an image
    answers["3B: dendogram"] = plt.savefig('dendrogram.png')

    # C. Find the iteration where clusters {I={8,2,13}} and {J={1,9}} were merged
    # Assuming the solution involves manual inspection or a programmed approach
    # This placeholder assumes manual inspection or a specific logic to identify the iteration
    answers["3C: iteration"] = -1  # Placeholder

    # D. Assign the function for calculating dissimilarity
    answers["3D: function"] = data_index_function

    # E. List all the clusters available at the time of merging
    # Placeholder for manual inspection or specific logic to determine the clusters
    answers["3E: clusters"] = [{0, 0}, {0, 0}]  # Placeholder

    # F. Comment on the dendrogram phenomenon
    answers["3F: rich get richer"] = "Placeholder for your explanation based on the dendrogram."

    return answers

if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
