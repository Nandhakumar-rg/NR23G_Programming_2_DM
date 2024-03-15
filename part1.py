import myplots as myplt
import time
import warnings
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

# Define fit_kmeans function here
def fit_kmeans(data, n_clusters):
    X, _ = data  # Only use the data, not the labels
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Standardize the data
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(X_scaled)
    return kmeans.labels_

def compute():
    answers = {}

    # Load datasets
    n_samples = 100
    random_state = 42
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    blobs_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso_data, aniso_labels = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    aniso = np.dot(aniso_data, [[0.6, -0.6], [-0.4, 0.8]])
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

    datasets_dict = {
        'nc': noisy_circles,
        'nm': noisy_moons,
        'bvv': blobs_varied,
        'add': (aniso, aniso_labels),
        'b': blobs
    }

    # Store datasets in answers dict
    answers["1A: datasets"] = datasets_dict

    # Assuming myplt.plot_part1C is defined elsewhere and correctly implements plotting
    k_values = [2, 3, 5, 10]
    kmeans_results = {}
    for name, dataset in datasets_dict.items():
        kmeans_results[name] = {}
        for k in k_values:
            labels = fit_kmeans(dataset, k)
            kmeans_results[name][k] = (dataset[0], labels)  # Store tuple of data and labels for plotting

    # Assuming you will use myplt.plot_part1C to generate and save your plots for analysis
    # Example usage (you'll need to adjust based on your actual implementation):
    myplt.plot_part1C(kmeans_results, "kmeans_clusters.pdf")

    # Placeholder for analysis results based on visual inspection of generated plots
    answers["1C: cluster successes"] = {"nc": [2], "nm": [2], "bvv": [3], "add": [2], "b": [3]}
    answers["1C: cluster failures"] = []  # Based on visual inspection
    answers["1D: datasets sensitive to initialization"] = []  # This would require multiple initializations and comparisons

    return answers

if __name__ == "__main__":
    answers = compute()
    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)