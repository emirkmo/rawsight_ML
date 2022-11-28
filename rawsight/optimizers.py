from typing import Callable

import numpy as np

from rawsight import CostFunction, Model

__all__ = ["batch_gradient_descent", "regularized_batch_gradient_descent", "Optimizer"]

Optimizer = Callable[[np.ndarray, np.ndarray, Model, CostFunction, ...], Model]


def batch_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    cost_function: CostFunction,
    learning_rate: float,
    max_iter: int,
) -> Model:
    """
    Gradient descent algorithm for linear regression.
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        model    : callable initialized with parameters
        cost_function : callable that computes the cost function for linear regression
        learning_rate : learning rate for gradient descent
        max_iter : maximum number of iterations to run gradient descent
    Returns:
        model : callable initialized with parameters that minimize the cost function
    """
    # initialize model parameters
    parameters = np.array(model.parameters, dtype=object)

    # run gradient descent
    niter = 0
    while niter < max_iter:
        # compute gradient
        gradients = cost_function.compute_gradient(x, y, model, lamb=0)

        # update model & parameters
        for i in range(len(parameters)):
            parameters[i] = parameters[i] - learning_rate * gradients[i]
        model.parameters = parameters

        niter += 1
    return model


def regularized_batch_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    cost_function: CostFunction,
    learning_rate: float,
    max_iter: int,
    regularization_param: float = 1,
) -> Model:
    """
    Gradient descent algorithm for linear regression.
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        model Model: callable initialized with parameters
        cost_function CostFunction: callable that computes the cost function for linear regression
        learning_rate float: learning rate for gradient descent
        max_iter int: maximum number of iterations to run gradient descent
        regularization_param float: regularization parameter
    Returns:
        model : callable initialized with parameters that minimize the cost function

    """
    # initialize model parameters
    parameters = list(model.parameters)
    # run gradient descent
    niter = 0
    while niter < max_iter:
        # compute gradient
        gradients = cost_function.compute_gradient(
            x, y, model, lamb=regularization_param
        )
        # update model & parameters
        for i in range(len(parameters)):
            parameters[i] = parameters[i] - learning_rate * gradients[i]
        model.parameters = parameters

        niter += 1
    return model
