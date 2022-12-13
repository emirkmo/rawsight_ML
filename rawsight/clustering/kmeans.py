import numpy as np

from ..distances import combination_distance


def find_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = combination_distance(X, centroids)
    return np.argmin(dist, axis=1)


def random_centroids(X: np.ndarray, k: int) -> np.ndarray:
    return X[np.random.choice(X.shape[0], k, replace=False)]


def compute_centroids(X: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Returns:
    centroids (ndarray): (K, n) Newly computed centroids"""
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(X[idx == i], axis=0)
    return centroids


def kmeans(X: np.ndarray, k: int, max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]:
    centroids = random_centroids(X, k)
    idx = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        idx = find_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return centroids, idx
