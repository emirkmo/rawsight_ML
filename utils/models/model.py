from typing import Protocol, Optional, Sequence, Iterable
from numpy.typing import NDArray, ArrayLike
import numpy as np


class Model(Protocol):
    def __call__(self, x: NDArray) -> NDArray:
        ...

    def partial_derivatives(self, x: NDArray) -> NDArray:
        ...

    def evaluate(self, x: NDArray) -> ArrayLike | float:
        ...

    # noinspection PyPropertyDefinition
    @property
    def parameters(self) -> ArrayLike:
        ...

    @parameters.setter
    def parameters(self, param: ArrayLike):
        ...

    def predict(self, x: NDArray) -> ArrayLike | float:
        ...


class BaseLinearModel:

    def __init__(self, w: ArrayLike = (1,), b: float = 0, n_features: int = 1,
                 verify_inputs: bool = True, verify_params: bool = True):
        """
        BaseLinearModel is a mixin class for Linear, Polynomial, and Logistic Regression.
        w should have the same length as n_features, will be automatically done if w has length 1
        else will raise.
        :param w: weight vector
        :param b: bias
        :param n_features: number of features
        :param verify_inputs: coerce x and raise if x has wrong shape, set verify_inputs to false
        :param verify_params: coerce w and b and raise if w has wrong shape, set verify_params to false
        """
        self.w = self.verify_w(w, n_features)
        self.b = b
        self.n = n_features
        self.verify_inputs = verify_inputs
        self.verify_params = verify_params

    @staticmethod
    def verify_w(w: ArrayLike, n: int) -> list[float]:
        """ Shape of w must be n * 1 or 1 * n where n is
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

    def evaluate(self, x: NDArray) -> ArrayLike | float:
        raise NotImplementedError("evaluate must be implemented in main model class, not this mixin.")

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
        self.w = self.verify_w(w, self.n) if self.verify_params else w

    def partial_derivatives(self, x: NDArray):
        return self.dw(x), self.db()

    def __call__(self, x):
        return self.evaluate(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(w={self.w}, b={self.b}, n_features={self.n})"

    def predict(self, x: NDArray) -> ArrayLike | float:
        return self.evaluate(x)
