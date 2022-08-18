import numpy as np
from numpy.typing import ArrayLike


def add_poly_features(x: ArrayLike, degrees: int = 1):
    """Add polynomial features to x.
    x is an array of input features, where its shape must be m * n where n is
    the number of features.
    degrees is the number of polynomial degrees to add.
    """
    x = np.array(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if degrees < 1 or not np.isfinite(degrees):
        raise ValueError("degrees must be positive definite and atleast 1.")
    if degrees == 1:
        return x
    return np.hstack([x ** i for i in range(1, degrees + 1)])
