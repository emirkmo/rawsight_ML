from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class Distance(Protocol):
    def __call__(self, x1: ArrayLike, x2: ArrayLike) -> float | np.ndarray:
        ...


class DistanceFunc(Protocol):
    def __call__(
        self, x1: np.ndarray, x2: np.ndarray, distance: Distance
    ) -> np.ndarray:
        ...
