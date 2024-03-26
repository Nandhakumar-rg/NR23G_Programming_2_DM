import myplots as myplt
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
from sklearn.cluster import AgglomerativeClustering, KMeans
import pickle
import utils as u

def fit_kmeans(data, n_clusters):
    """
    Fits k-means to the data and returns the labels.

    :param data: Dataset to cluster.
    :param n_clusters: Number of clusters to fit.
    :return: Predicted labels from k-means clustering.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)

    return kmeans.labels_

def compute():
    answers = {}

    # Datasets
    n_samples = 100
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    blobs_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=42)

    datasets_list = [
        (noisy_circles, 'noisy_circles'),
        (noisy_moons, 'noisy_moons'),
        (blobs_varied, 'blobs_varied'),
        (aniso, 'aniso'),
        (varied, 'varied')
    ]

    answers["1A: datasets"] = {name: data for data, name in datasets_list}

    k_values = [2, 3, 5, 10]
    kmeans_results = {}
    for data, name in datasets_list:
        results = {}
        for k in k_values:
            labels = fit_kmeans(data[0], k)
            results[k] = labels
        kmeans_results[name] = (data, results)

    answers["1B: fit_kmeans"] = fit_kmeans
    answers["1C: cluster successes"] = {"nc": [2], "nm": [2, 3], "b": [3, 5, 10]}
    answers["1C: cluster failures"] = ["bvv", "add"]
    answers["1D: datasets sensitive to initialization"] = ["nm", "bvv"] 

    # Generate plots
    myplt.plot_part1C(kmeans_results, 'kmeans_clusters.pdf')

    return answers

if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
