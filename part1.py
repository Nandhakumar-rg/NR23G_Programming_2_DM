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

def generate_datasets():
    n_samples = 100
    random_state = 42
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso = datasets.make_blobs(n_samples=n_samples, random_state=8)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso[0], transformation), aniso[1])
    return {"nc": noisy_circles, "nm": noisy_moons, "bvv": varied, "add": aniso, "b": blobs}

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)
    return kmeans.labels_

def compute():
    answers = {}
    datasets = generate_datasets()
    answers["1A: datasets"] = datasets

    # Part B: Modify the `fit_kmeans` signature and implement the functionality
    def fit_kmeans(data, n_clusters):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])  
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
        kmeans.fit(data_scaled)
        return kmeans.labels_

    answers["1B: fit_kmeans"] = fit_kmeans

    # Part C: Prepare for plotting
    k_values = [2, 3, 5, 10]
    kmeans_dct = {dataset_name: (data, {k: fit_kmeans(data, k) for k in k_values}) for dataset_name, data in datasets.items()}
    file_nm = "cluster_plots.pdf"
    myplt.plot_part1C(kmeans_dct, file_nm)


    answers["1C: cluster successes"] = {"nc": [2], "nm": [2, 3], "b": [3, 5, 10]}  
    answers["1C: cluster failures"] = ["bvv", "add"]  

   
    answers["1D: datasets sensitive to initialization"] = ["nm", "bvv"]  

    return answers

if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
