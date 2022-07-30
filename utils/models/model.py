from typing import Protocol
from numpy.typing import NDArray, ArrayLike
import numpy as np

class Model(Protocol):
    def __call__(self, x: NDArray) -> NDArray:
        ...

    def partial_derivatives(self, x: NDArray) -> NDArray:
        ...

    def evaluate(self, x: NDArray) -> ArrayLike | float:
        ...

class BaseModelMixin:

    def __init__(self, w: ArrayLike, b: float, n_features: int):
        self.w = verify_w(w, n_features)
        self.b = b
        self.n = n_features
        self.verify = verify_inputs

    def add_feature(self, w_0=1):
        """add another input feature with initial guess w_0."""
        self.w.append(w_0)
        self.n += 1

    def _verify_input_dimension(self, x):
        return x.shape[-1] == self.n or (self.n == 1 and len(x.shape) == 1)

    def get_w_as_array(self) -> float | NDArray:
        return np.array(self.w) if len(self.w) > 1 else self.w[0]

    def get_x_as_array(self, x: NDArray) -> NDArray:
        """x is array of input X, where its shape must be m * n where n is
        the number of features."""
        x = np.array(x)
        if not self._verify_input_dimension(x):
            raise ValueError(f"x must be m * n, but was was {x.shape}."
                             f" n is number of features and is currently: {self.n}.")
        return x

    def evaluate(self, x: NDArray) -> NDArray | float:
        """x is array of input X, where its shape must be m * n where n is
        the number of features."""
        x = self.get_x_as_array(x) if self.verify else x
        w = self.get_w_as_array()
        return np.dot(x, w) + self.b

    @staticmethod
    def dw(x: NDArray):
        """partial derivative with respect to w"""
        return x

    @staticmethod
    def db(x: Optional[NDArray] = None):
        """partial derivative with respect to b"""
        return 1.

    @property
    def parameters(self):
        return np.array(self.w), np.array(self.b)

    @parameters.setter
    def parameters(self, parameters):
        w, self.b = parameters
        self.w = self.verify_w(w, self.n)

    def partial_derivatives(self, x: NDArray):
        return self.dw(x), self.db()

    def __call__(self, x):
        return self.evaluate(x)

    def __repr__(self):
        return f"LinearModel(w={self.w}, b={self.b}, n_features={self.n})"


def verify_w(w: ArrayLike, n: int) -> list[float]:
    """ w is array of weights,
    where its shape must be n * 1 or 1 * n where n is
    number of features. Return w as a stack (list) for
    easier appending of features.
    Args:
        w (ArrayLike): array of weights
        n (int): number of features

    Returns:
        list: list of weights
    """
    if getattr(w, '__iter__', None) is None:
        w = np.atleast_1d(w)
    if n < 1 or not np.isfinite(n):
        raise ValueError("n_features must be positive definite and atleast 1.")
    if n == 1:
        return [w[0]]
    if len(w) in (1, n):
        return list(np.ones(n) * w)
    raise ValueError(f"w was {w}, but should be length of n_features or 1.")

