import numpy as np
from numpy.typing import ArrayLike


def get_n_features(x: ArrayLike) -> int:
    if len(x.shape) == 1:
        return 1
    return x.shape[1]


def get_n_classes(x: ArrayLike) -> int:
    if len(x.shape) == 1:
        return 1
    return x.shape[0]


def transpose(x: ArrayLike) -> ArrayLike:
    if np.isscalar(x):
        return x
    return x.T
