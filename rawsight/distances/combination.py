import numpy as np

from .distance import Distance
from .metrics import euclidean


def combination_distance(
    X: np.ndarray, Y: np.ndarray, distance: Distance = euclidean
) -> np.ndarray:
    """Compute the pairwise distance between two sets of vectors Assuming len(Y) << len(X)
    More efficient calculation can be found in scipy.spatial.distance.cdist"""

    dist = np.vstack([distance(X, y) for y in Y]).T
    return dist
