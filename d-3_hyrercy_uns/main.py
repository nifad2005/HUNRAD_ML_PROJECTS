import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Dummy Data Generation
def make_blobs_data(n_samples=300, cluster_std=0.5, random_state=0):
    X, _ = make_blobs(n_samples=n_samples,centers=3, cluster_std=cluster_std, random_state=random_state)
    return X

# Plotting layout
def plot_dendrogram():
    X = make_blobs_data()
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], s=30, c='blue', marker='*', label='Points are here.')
    plt.title('Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_hierarchical_clustering():
    X = make_blobs_data()
    linked = linkage(X, method='ward')
    plt.figure(figsize=(12, 7))
    dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
    plt.title('Cluster Dendrogram')
    plt.xlabel('Cluster Index')
    plt.ylabel('Distance')
    plt.show()

# Clustering the data using AgglomerativeClustering
def clustering_using_agglomerativeClustering():
    X = make_blobs_data()
    cluster = AgglomerativeClustering(n_clusters=3)
    labels = cluster.fit_predict(X)
    print("Labels:\n", labels)
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title('Agglomerative Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    print(__name__)
    clustering_using_agglomerativeClustering()      