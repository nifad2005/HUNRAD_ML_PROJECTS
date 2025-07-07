import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


X , _ = make_blobs(n_samples=300, centers=3, cluster_std=1, random_state=0)

outliers = np.array([[10, 10], [12, 12], [14, 14]])
X = np.vstack((X, outliers))


kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)   
print("Cluster Centers:\n", kmeans.cluster_centers_)

y_kmeans = kmeans.predict(X)

print("Labels:\n", y_kmeans)