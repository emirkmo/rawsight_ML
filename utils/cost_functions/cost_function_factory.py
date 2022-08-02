from utils.input_validation import transpose
from utils.models import Model
from typing import Callable, Optional, Type
import numpy as np
from numpy.typing import ArrayLike, NDArray
from abc import ABC, abstractmethod
from .regularization import Regularization

_CostFunctionCallable = Callable[[ArrayLike, ArrayLike], float]
_CostFunctionGradientCallable = Callable[[ArrayLike, ArrayLike], ArrayLike]


class AbstractCostFunction(ABC):

    @abstractmethod
    def __init__(self, cost_function: _CostFunctionCallable, gradient: _CostFunctionGradientCallable,
                 regularization: Optional[Type[Regularization]] = None):
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def __call__(self, x: NDArray, y: ArrayLike, model: Model, lamb: float = 1.) -> ArrayLike:
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def compute_cost(self, fy: ArrayLike, y: ArrayLike) -> float:
        raise NotImplementedError("CostFunction is an abstract class")

    @abstractmethod
    def compute_gradient(self, x: NDArray, y: ArrayLike, model: Model, lamb: float = 1.) -> tuple[ArrayLike, ...]:
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

    def __init__(self, cost_function: _CostFunctionCallable, gradient: _CostFunctionGradientCallable,
                 regularization: Optional[Type[Regularization]] = None):
        self.cost_function = cost_function
        self.gradient = gradient
        self.regularization = regularization
        self.regularize_intercept = False  # by convention intercept is not regularized,
        # we skip last element of gradient assuming it to be the intercept unless this is `True.`

    def compute_gradient(self, x: NDArray, y: ArrayLike, model: Model, lamb: float = 1.) -> tuple[ArrayLike, ...]:
        """
        Computes the partial derivative of the cost function with respect to the model parameters.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          model    : callable initialized with parameters
          lamb (float): regularization parameter, default 1.

        Returns
            grad_theta_i (ndarray (n,)): The partial derivative of the cost function with
            respect to the model parameters. Usually this is dj_dw, dj_db, but it can be
            generalized to i = 0, ..., n-1 parameters of the model, called theta here.

        """
        fy = model(x)
        pds = model.partial_derivatives(x)

        if self.regularization is None:
            return tuple([np.sum(self.gradient(fy, y) * transpose(partial), axis=-1) for partial in pds])

        if self.regularize_intercept:
            r = self.regularization(model.parameters, len(y), lamb).gradient
            return tuple([np.sum(self.gradient(fy, y) * transpose(partial), axis=-1) + r for partial in pds])

        dj_wb = [np.sum(self.gradient(fy, y) * transpose(partial), axis=-1) for partial in pds]
        dj_wb[:-1] += self.regularization(model.parameters[:-1], len(y), lamb).gradient
        return tuple(dj_wb)

    def compute_cost(self, fy: ArrayLike, y: ArrayLike) -> float:
        """
        Computes the cost function (without regularization).

        Args:
          fy (ndarray (m,)): model target values
          y (ndarray (m,)): data target values

        Returns
            total_cost (float): The cost of using model with current parameters
        """
        return self.cost_function(fy, y)

    def __call__(self, x: NDArray, y: ArrayLike, model: Model, lamb: float = 1) -> float:
        """
        Computes the cost function.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          model    : callable initialized with parameters
          lamb (float): regularization parameter

        Returns
            total_cost (float): The cost of using model with current parameters as
            the parameters for linear regression to fit the data points in x and y
        """
        fy = model(x)

        cost = self.compute_cost(fy, y)
        if self.regularization is None:
            return cost

        if self.regularize_intercept:
            return cost + self.regularization(model.parameters, len(y), lamb).cost
        return cost + self.regularization(model.parameters[:-1], len(y), lamb).cost
