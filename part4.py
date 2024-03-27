import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

def fit_hierarchical_cluster(data, linkage_type, n_clusters):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = AgglomerativeClustering(linkage=linkage_type, n_clusters=n_clusters)
    model.fit(data_scaled)
    return model.labels_

def fit_modified(data, method):
    Z = linkage(data, method=method)
    distances = Z[:, 2]
    distance_differences = np.diff(distances)
    max_diff_idx = np.argmax(distance_differences)
    cut_off_distance = (distances[max_diff_idx] + distances[max_diff_idx + 1]) / 2
    return Z, cut_off_distance

def compute():
    answers = {}
    dct = answers["4A: datasets"] = {}
    nc = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    nm = make_moons(n_samples=100, noise=0.05, random_state=42)
    bvv = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    add = make_blobs(n_samples=100, random_state=42)
    b = make_blobs(n_samples=100, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = (np.dot(add[0], transformation), add[1])
    answers["4A: datasets"] = {
        'nc': nc,
        'nm' : nm,
        'bvv' : bvv,
        'add' : add,
        'b' : b
    }
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster
    datasets = answers["4A: datasets"]
    linkage_methods = ['single', 'complete', 'ward', 'average']
    clustering_outcomes = {}
    plot_fig, plot_axes = plt.subplots(len(datasets), len(linkage_methods), figsize=(20, 15))
    for dataset_idx, (name_of_dataset, (dataset_features, _)) in enumerate(datasets.items()):
        norm_features = StandardScaler().fit_transform(dataset_features)
        for linkage_idx, method in enumerate(linkage_methods):
            cluster_labels = fit_hierarchical_cluster(norm_features, method, 2)
            plot_axes[dataset_idx, linkage_idx].scatter(norm_features[:, 0], norm_features[:, 1], c=cluster_labels, s=50, cmap='viridis', edgecolors='k')
            plot_axes[dataset_idx, linkage_idx].set_title(f'{name_of_dataset} - {method}')
            if name_of_dataset not in clustering_outcomes:
                clustering_outcomes[name_of_dataset] = {}
            clustering_outcomes[name_of_dataset][method] = cluster_labels
    plot_fig.tight_layout()
    plot_fig.savefig('Hierarchical_Clustering_Results.pdf')
    dct = answers["4B: cluster successes"] = ["nc", "nm"]
    datasets = answers["4A: datasets"]
    linkage_strategies = ['ward', 'complete', 'average', 'single']
    with PdfPages('HierarchicalClustering_PartC_Plots.pdf') as pdf_book:
        for name, (data, labels) in datasets.items():
            normalizer = StandardScaler()
            normalized_data = normalizer.fit_transform(data)
            fig, axis = plt.subplots(1, len(linkage_strategies), figsize=(20, 5))
            fig.suptitle(f'Cluster Analysis: {name}', fontsize=16)
            for idx, strategy in enumerate(linkage_strategies):
                linkage_matrix, optimal_cutoff = fit_modified(normalized_data, strategy)
                identified_clusters = fcluster(linkage_matrix, optimal_cutoff, criterion='distance')
                axis[idx].scatter(normalized_data[:, 0], normalized_data[:, 1], c=identified_clusters, cmap='viridis', edgecolors='k', s=50)
                axis[idx].set_title(f"{name} using {strategy}")
                axis[idx].set_xlabel('Dimension 1')
                axis[idx].set_ylabel('Dimension 2')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf_book.savefig(fig)
            plt.close()
    dct = answers["4C: modified function"] = fit_modified
    return answers

if __name__ == "__main__":
   answers = compute()

with open("part4.pkl", "wb") as f:
    pickle.dump(answers, f)
