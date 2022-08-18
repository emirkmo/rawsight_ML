from typing import Any

import numpy as np
from numpy import ScalarType
from numpy.typing import ArrayLike, NDArray
from .cost_function_factory import CostFunction, NDArrayInt
from .regularization import SquaredSumRegularization


def _least_squares_cost(fy: ArrayLike, y: ArrayLike) -> float:
    """ Simple squared residual cost function for LinearModel, vectorized form.
    """
    return np.sum((fy - y) ** 2) / 2 / len(y)


def _least_squares_cost_gradient(fy: ArrayLike, y: ArrayLike) -> ArrayLike:
    """ Simple squared residual cost function for LinearModel, vectorized form.
    """
    return (fy - y) / len(y)


def _logistic_cost(fy: ArrayLike, y: ArrayLike) -> float:
    """ Simple negative log cost function for LogisticModel, vectorized form.
    """
    return np.sum(-y * np.log(fy) - (1 - y) * np.log(1 - fy)) / len(y)


def _logistic_cost_gradient(fy: ArrayLike, y: ArrayLike) -> ArrayLike:
    """ Simple negative log cost function for LogisticModel, vectorized form.
    """
    return _least_squares_cost_gradient(fy, y)


def _cross_entropy_loss(fy: NDArray, y: NDArrayInt) -> NDArray:
    """ Simple negative log cost function for LogisticModel, vectorized form.
    """
    return -np.log(fy[range(y.shape[0]), y])


def _cross_entropy_cost(fy: NDArray, y: NDArrayInt) -> float:
    """ Simple negative log cost function for LogisticModel, vectorized form.
    """
    loss = _cross_entropy_loss(fy, y)
    return np.sum(loss)/y.shape[0]


def _cross_entropy_cost_gradient(fy: NDArray, y: NDArrayInt) -> ArrayLike:
    """ Simple negative log cost function for LogisticModel, vectorized form.
    """
    p = fy.copy()
    p[range(y.shape[0]), y] -= 1
    return p/y.shape[0]

# def _cross_entropy_loss(fy: NDArray, y: NDArrayInt) -> NDArray:
#     """ Simple negative log cost function for LogisticModel, vectorized form.
#     """
#     return -np.log(fy)
#
#
# def _cross_entropy_cost(fy: NDArray, y: NDArrayInt) -> float:
#     """ Simple negative log cost function for LogisticModel, vectorized form.
#     """
#     loss = _cross_entropy_loss(fy, y)
#     return np.sum(loss)/y.shape[0]
#
#
# def _cross_entropy_cost_gradient(fy: NDArray, y: NDArrayInt) -> ArrayLike:
#     """ Simple negative log cost function for LogisticModel, vectorized form.
#     """
#     return (fy - 1) / len(y)


least_squares_cost_function = CostFunction(_least_squares_cost, _least_squares_cost_gradient, regularization=None)
regularized_least_squares_cost_function = CostFunction(_least_squares_cost, _least_squares_cost_gradient,
                                                       regularization=SquaredSumRegularization)

logistic_cost_function = CostFunction(cost_function=_logistic_cost, gradient=_logistic_cost_gradient)
regularized_logistic_cost_function = CostFunction(cost_function=_logistic_cost, gradient=_logistic_cost_gradient,
                                                  regularization=SquaredSumRegularization)

cross_entropy_cost_function = CostFunction(cost_function=_cross_entropy_cost, gradient=_cross_entropy_cost_gradient)
regularized_cross_entropy_cost_function = CostFunction(cost_function=_cross_entropy_cost,
                                                       gradient=_cross_entropy_cost_gradient,
                                                       regularization=SquaredSumRegularization)