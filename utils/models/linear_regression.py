from typing import Optional
import numpy as np
from numpy.typing import NDArray
from .model import Model


class LinearModel(Model):

    def __init__(self, w=(1,), b=0, n_features=1):
        """w should have the same length as n_features.
        Will be automatically done if w has length 1 else will raise."""
        self.w = self.verify_w(w, n_features)
        self.b = b
        self.n = n_features

    @staticmethod
    def verify_w(w, n):
        if getattr(w, '__iter__', None) is None:
            w = np.atleast_1d(w)
        if n < 1 or not np.isfinite(n):
            raise ValueError("n_features must be positive definite and atleast 1.")
        if n == 1:
            return [w[0]]
        if len(w) in (1, n):
            return list(np.ones(n) * w)
        raise ValueError(f"w was {w}, but should be length of n_features or 1.")

    def add_feature(self, w_0=1):
        """add another input feature with initial guess w_0."""
        self.w.append(w_0)
        self.n += 1

    def _verify_input_dimension(self, x):
        if len(x.shape) == 1:
            return self.n == 1
        return x.shape[1] == self.n

    def get_w_x_as_array(self, x: NDArray):
        """x is array of input X, where its shape must be m * n where n is
        the number of features."""
        w = np.array(self.w)
        x = np.array(x)
        if not self._verify_input_dimension(x):
            raise ValueError(f"x must be m * n, but was was {x.shape}."
                             f" n is number of features and is currently: {self.n}.")
        return w, x

    def evaluate(self, x: NDArray):
        """x is array of input X, where its shape must be m * n where n is
        the number of features."""
        w, x = self.get_w_x_as_array(x)
        return w * x + self.b

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
