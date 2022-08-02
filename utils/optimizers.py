import numpy as np
from utils import Model, CostFunction


def batch_gradient_descent(x: np.ndarray, y: np.ndarray, model: Model, cost_function: CostFunction,
                           learning_rate: float, max_iter: int) -> Model:
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
        gradients = np.array(cost_function.compute_gradient(x, y, model), dtype=object)

        # update model & parameters
        parameters = parameters - learning_rate * gradients
        model.parameters = parameters

        niter += 1
    return model


def regularized_batch_gradient_descent(x: np.ndarray, y: np.ndarray, model: Model, cost_function: CostFunction,
                                       learning_rate: float, max_iter: int, regularization_param: float = 1) -> Model:
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
        gradients = np.array(cost_function.compute_gradient(x, y, model, lamb=regularization_param), dtype=object)

        # update model & parameters
        parameters = parameters - learning_rate * gradients
        model.parameters = parameters

        niter += 1
    return model
