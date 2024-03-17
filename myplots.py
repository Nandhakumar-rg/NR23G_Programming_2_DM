import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_part1C(kmeans_dct, file_nm):
    """
    Plots the results of k-means clustering for different datasets and values of k.
    
    Arguments:
    kmeans_dct -- A dictionary where the key is the dataset name and the value is a tuple. 
                  The first element of the tuple is the dataset points X, 
                  and the second element is a dictionary with keys as values of k 
                  and values as the corresponding labels from k-means clustering.
    file_nm -- Name of the PDF file to save the plots.
    """
    nb_datasets = len(kmeans_dct)
    nb_ks = 4  # Assuming you're plotting for k values: 2, 3, 5, 10
    
    # Create a PDF pages object to save plots
    with PdfPages(file_nm) as pdf:
        # Iterate through each dataset
        for dataset_name, (X, k_results) in kmeans_dct.items():
            # Create a new figure for each dataset
            fig, axs = plt.subplots(nb_ks, 1, figsize=(8, 8), sharex=True, sharey=True)
            
            # Make sure axs is an array even when nb_ks is 1
            if nb_ks == 1:
                axs = [axs]
            
            for i, (k, labels) in enumerate(k_results.items()):
                ax = axs[i]
                # Scatter plot of dataset points colored by k-means labels
                scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="viridis", alpha=0.5)
                ax.set_title(f"{dataset_name}, k={k}")
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                # Create a legend for the clusters
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
            
            plt.tight_layout()
            pdf.savefig(fig)  # Save the current figure to the PDF
            plt.close(fig)  # Close the figure to free memory

