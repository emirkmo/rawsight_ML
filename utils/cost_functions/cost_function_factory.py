from utils.input_validation import transpose
from utils.models import Model
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from abc import ABC, abstractmethod

_CostFunctionCallable = Callable[[ArrayLike, ArrayLike], float]
_CostFunctionGradientCallable = Callable[[ArrayLike, ArrayLike], ArrayLike]


class AbstractCostFunction(ABC):

    @abstractmethod
    def __init__(self, cost_function: _CostFunctionCallable, gradient: _CostFunctionGradientCallable):
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def __call__(self, x: NDArray, y: ArrayLike, model: Model):
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def compute_cost(self, fy: ArrayLike, y: ArrayLike):
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def compute_gradient(self, x: NDArray, y: ArrayLike, model: Model):
        raise NotImplementedError("CostFunction is an abstract class")


class CostFunction(AbstractCostFunction):
    """
    CostFunction factory object with callable cost and gradient functions.

    CostFunctionFactory is initialized with a callable cost function and gradient.
    The callable cost function takes in the model output and the target values and
    returns the cost.
    The callable gradient takes in the model output and the target values and
    returns the partial derivative of the cost function with respect to the model parameters.

    The cost function is used to compute the cost of the model.
    The current implementation is valid for linear regression (and logistic and polynomial)
    since they can be reduced to the same form.
    """

    def __init__(self, cost_function: _CostFunctionCallable, gradient: _CostFunctionGradientCallable):
        self.cost_function = cost_function
        self.gradient = gradient

    def compute_gradient(self, x: NDArray, y: ArrayLike, model: Model):
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

    def compute_cost(self, fy: ArrayLike, y: ArrayLike):
        """
        Computes the cost function.

        Args:
          fy (ndarray (m,)): model target values
          y (ndarray (m,)): data target values

        Returns
            total_cost (float): The cost of using model with current parameters as
            the parameters for linear regression to fit the data points in x and y
        """
        return self.cost_function(fy, y)

    def __call__(self, x: NDArray, y: ArrayLike, model: Model):
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
