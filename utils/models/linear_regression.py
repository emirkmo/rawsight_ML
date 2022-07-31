import numpy as np
from numpy.typing import NDArray, ArrayLike
from .model import BaseLinearModel
from typing import Optional, Callable


def linear(x: NDArray, w: ArrayLike, b: float) -> NDArray | float:
    return np.dot(x, w) + b


class LinearModel(BaseLinearModel):

    def __init__(self, w: ArrayLike = (1,), b: float = 0, n_features: int = 1,
                 verify_inputs: bool = True, verify_params: bool = True):
        """w should have the same length as n_features.
        Will be automatically done if w has length 1 else will raise.
        verify_inputs will coerce x and raise if x has wrong shape,
        set verify_inputs to false in order to disable this check,
        useful to reduce overhead when iterating.
        """
        super().__init__(w=w, b=b, n_features=n_features, verify_inputs=verify_inputs,
                         verify_params=verify_params)

    def evaluate(self, x: NDArray) -> NDArray | float:
        """x is array of input X, where its shape must be m * n where n is
        the number of features."""
        x = self.get_x_as_array(x) if self.verify_inputs else x
        w = self.get_w_as_array()
        return linear(x, w, self.b)

    def partial_derivatives(self, x: NDArray):
        return self.dw(x), self.db()


