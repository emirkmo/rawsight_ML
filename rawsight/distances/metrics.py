import numpy as np
from numpy.typing import ArrayLike


def euclidean(x1: ArrayLike, x2: ArrayLike) -> float | np.ndarray:
    return np.linalg.norm(x1 - x2, axis=1)
