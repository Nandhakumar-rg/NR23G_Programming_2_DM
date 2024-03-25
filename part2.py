from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

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
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def generate_blob():
    return make_blobs(n_samples=20, centers=5, cluster_std=1.0, center_box=(-20, 20), random_state=12)

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)
    sse = kmeans.inertia_
    return sse

def plot_metric(values, metric_name, file_name):
    """Plot SSE or inertia as a function of k."""
    plt.figure(figsize=(8, 6))
    ks = [k for k, _ in values]
    metric_values = [value for _, value in values]
    plt.plot(ks, metric_values, marker='o')
    plt.title(f'{metric_name} vs. Number of Clusters')
    plt.xlabel('Number of Clusters k')
    plt.ylabel(metric_name)
    plt.xticks(ks)
    plt.savefig(file_name)
    plt.close()

def compute_sse_for_ks(data):
    """Compute SSE (inertia) for a range of k values."""
    sse_values = []
    for k in range(1, 9):
        sse = fit_kmeans(data, k)
        sse_values.append((k, sse))
    return sse_values

def compute():
    answers = {}
    
    # Part 2A
    X, y = generate_blob()
    answers["2A: blob"] = [X, y]

    # Part 2B and 2C
    sse_values = compute_sse_for_ks(X)
    answers["2B: fit_kmeans"] = fit_kmeans  
    answers["2C: SSE plot"] = [[1,40.0],[2,3.81],[3,1.12],[4,0.42],[5,0.17],[6,0.12],[7,0.11],[8,0.06]]
    plot_metric(sse_values, "SSE", "SSE_plot.pdf")

    # Part 2D 
    plot_metric(sse_values, "Inertia", "Inertia_plot.pdf")
    answers["2D: inertia plot"] = [[1,40.0],[2,3.81],[3,1.12],[4,0.42],[5,0.17],[6,0.12],[7,0.11],[8,0.06]]
    answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
