import numpy as np
from typing import Protocol, Sequence, Iterable
from numpy.typing import ArrayLike


# noinspection PyPropertyDefinition
class Regularization(Protocol):

    def __init__(self, param: ArrayLike | Iterable, n_samples: int, lamb: float = 1):
        """

        Parameters
        ----------
        param : ArrayLike | Sequence
            weights
        lamb : float, optional
            the regularization parameter.
        """
        raise NotImplementedError("Regularization is a Protocol class.")

    @property
    def cost(self) -> ArrayLike:
        ...

    @property
    def gradient(self) -> ArrayLike:
        ...


class SquaredSumRegularization:
    def __init__(self, param: ArrayLike, n_samples: int, lamb: float = 1):
        """
        Squared sum regularization of the weights
        times lambda divided by the number of features.
        Parameters
        ----------
        param : ArrayLike
            weights to be regularized.
        lamb : float, optional
            the regularization parameter.

        """
        self.lamb = lamb
        self.param = np.atleast_1d(param)
        self.n_samples = n_samples

    @property
    def cost(self) -> float:
        return self.lamb * np.sum(self.param ** 2) / self.n_samples / 2

    @property
    def gradient(self) -> ArrayLike:
        return self.lamb * self.param / self.n_samples
