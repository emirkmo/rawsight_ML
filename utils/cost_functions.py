import numpy as np
from typing import Callable, Protocol
from utils.input_validation import transpose

class CostFunction(Protocol):
    def __call__(self, x, y, model):
        ...

    def compute_cost(self, x, y, model):
        ...

    def compute_gradient(self, x, y, model):
        ...


def least_squares_cost(fy, y):
    """ Simple squared residual cost function for LinearModel, vectorized form.
    """
    return np.sum((fy - y) ** 2) / 2 / len(y)


def least_squares_cost_gradient(fy, y):
    """ Simple squared residual cost function for LinearModel, vectorized form.
    """
    return (fy - y) / len(y)


class LeastSquaresCostFunction:
    """
    CostFunction is a class that encapsulates the cost function for linear regression.
    It is initialized with a callable cost function.
    """

    def __init__(self, cost_function: Callable = least_squares_cost, gradient: Callable = least_squares_cost_gradient):
        self.cost_function = cost_function
        self.gradient = gradient

    def compute_gradient(self, x, y, model):
        """
        Computes the partial derivative of the cost function with respect to the model parameters.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          model    : callable initialized with parameters

        Returns
            grad_theta_i (ndarray (n,)): The partial derivative of the cost function with respect to the model parameters
        """
        fy = model(x)
        pds = model.partial_derivatives(x)
        return tuple([np.sum(self.gradient(fy, y) * transpose(partial), axis=-1) for partial in pds])

    def compute_cost(self, fy, y):
        """
        Computes the cost function for linear regression.

        Args:
          fy (ndarray (m,)): model target values
          y (ndarray (m,)): data target values

        Returns
            total_cost (float): The cost of using model with current parameters as
            the parameters for linear regression to fit the data points in x and y
        """
        return self.cost_function(fy, y)

    def __call__(self, x, y, model):
        """
        Computes the cost function for linear regression.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          model    : callable initialized with parameters

        Returns
            total_cost (float): The cost of using model with current parameters as
            the parameters for linear regression to fit the data points in x and y
        """
        fy = model(x)
        return self.compute_cost(fy, y)
