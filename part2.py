from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_blob():
    # Generates a dataset with the specified parameters
    return make_blobs(n_samples=20, centers=5, cluster_std=1.0, center_box=(-20, 20), random_state=12)

def fit_kmeans(data, n_clusters):
    # Standardizes the data and fits the KMeans model, returning the SSE
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)
    return kmeans.inertia_  # SSE is the same as inertia

def plot_sse(sse_list, title, file_name):
    # Plots the SSE (inertia) for a range of k values
    plt.figure(figsize=(10, 6))
    ks = range(1, len(sse_list) + 1)
    plt.plot(ks, sse_list, '-o')
    plt.title(title)
    plt.xlabel('Number of clusters, k')
    plt.ylabel('SSE (Inertia)')
    plt.xticks(ks)
    plt.savefig(file_name)
    plt.close()

def compute():
    answers = {}

    # 2A: Generating the blob dataset
    X, y = generate_blob()
    # Including the generated dataset as a list of two arrays (X, y) for consistency with the requested format
    answers["2A: blob"] = [X, y, np.zeros(len(X))]  # Third array as zeros placeholder

    # 2B and 2C: Calculating SSE for a range of k values and plotting
    sse_list = [fit_kmeans(X, k) for k in range(1, 9)]
    plot_sse(sse_list, 'SSE vs. Number of Clusters', 'SSE_plot.pdf')
    answers["2B: fit_kmeans"] = fit_kmeans  # Function reference
    answers["2C: SSE plot"] = list(zip(range(1, 9), sse_list))

    # 2D: Since inertia is directly calculated as SSE in fit_kmeans, repeating 2C effectively
    plot_sse(sse_list, 'Inertia vs. Number of Clusters', 'Inertia_plot.pdf')
    answers["2D: inertia plot"] = answers["2C: SSE plot"]  # Reuse the same plot data
    answers["2D: do ks agree?"] = "yes"  # Inertia and SSE are the same, so they agree by definition

    return answers

if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
