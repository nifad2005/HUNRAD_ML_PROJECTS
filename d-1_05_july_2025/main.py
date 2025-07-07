import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs


X , _ = make_blobs(n_samples=250, centers=4, cluster_std=0.50, random_state=0)


plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], s=50, c='red' )
plt.title("Randomly Generated Data🤷.")
plt.xlabel("Feature 1 ")
plt.ylabel("Feature 2 ")
plt.show()


k_means = KMeans(n_clusters=4, random_state=0, n_init=10)
k_means.fit(X)
labels = k_means.predict(X)

count = np.bincount(labels)
print(count)