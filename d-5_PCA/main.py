import numpy as np 
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

def generate_data():
    X, _ = make_blobs(n_samples=300, centers=4, n_features=3, cluster_std=0.60, random_state=0)
    return X

def pca_fun():
    X = generate_data()
    pca = PCA(n_components=2)  
    X_reduced = pca.fit_transform(X)
    print("Original shape:", X)


if __name__ == "__main__":
    pca_fun()


