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
    toy_data = io.loadmat('hierarchical_toy_data.mat')  
    answers["3A: toy data"] = toy_data

    data = toy_data['X'] if 'X' in toy_data else toy_data[list(toy_data.keys())[-1]]

    # B. Create a linkage matrix and plot a dendrogram
    Z = linkage(data, method='single')
    answers["3B: linkage"] = Z
    plt.figure(figsize=(10, 7))
    dendrogram_Z = dendrogram(Z)
    plt.savefig('dendrogram.png')  
    answers["3B: dendogram"] = dendrogram_Z

    # C. Find the iteration where clusters {I={8,2,13}} and {J={1,9}} were merged

    I={8,2,13}
    J={1,9}
    for i,(i1,i2,_,_)in enumerate(Z):
        if set([int(i1),int(i2)])==I.union(J):
            answers["3C: iteration"]=i
            break

    # D. Assign the function for calculating dissimilarity
    answers["3D: function"] = data_index_function

    # E. List all the clusters available at the time of merging
 
    answers["3E: clusters"] = [{0, 0}, {0, 0}]  

    # F. Comment on the dendrogram phenomenon
    answers["3F: rich get richer"] = "The dendrogram exhibits a tendency for larger clusters to absorb smaller ones, particularly observable in the left-hand side (blue clusters) where we see broader merges at higher linkage distances. This suggests a 'rich get richer' phenomenon, where some clusters grow by continuously merging with smaller clusters or individual points, resulting in a less balanced cluster size distribution towards the end of the agglomerative process."


    return answers

if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
