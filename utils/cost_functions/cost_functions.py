import numpy as np
from numpy.typing import ArrayLike
from .cost_function_factory import CostFunction
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


least_squares_cost_function = CostFunction(_least_squares_cost, _least_squares_cost_gradient)
regularized_least_squares_cost_function = CostFunction(_least_squares_cost, _least_squares_cost_gradient,
                                                       regularization=SquaredSumRegularization)

logistic_cost_function = CostFunction(cost_function=_logistic_cost, gradient=_logistic_cost_gradient)
regularized_logistic_cost_function = CostFunction(cost_function=_logistic_cost, gradient=_logistic_cost_gradient,
                                                  regularization=SquaredSumRegularization)