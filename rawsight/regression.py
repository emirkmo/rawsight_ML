from typing import Callable, Optional, Sequence, Any

from numpy.typing import ArrayLike, NDArray

from rawsight.cost_functions.cost_functions import regularized_cross_entropy_cost_function
from rawsight.models import LogisticModel, Model, LinearModel
from rawsight.cost_functions import (
    regularized_logistic_cost_function,
    CostFunction,
    regularized_least_squares_cost_function,
)
from rawsight.optimizers import regularized_batch_gradient_descent, Optimizer
from rawsight import get_n_features
from rawsight.scoring import accuracy
from rawsight.models.logistic import LogisticMapper, sigmoid
from rawsight.models.softmax import softmax, stable_softmax, SoftmaxModel
from enum import Enum
import numpy as np


class Regression:
    """Regression factory class"""

    _default_hyperparams = {"learning_rate": 0.1, "max_iter": 1000}

    def __init__(
        self,
        cost_function: CostFunction,
        model: Model,
        optimizer: Optimizer,
        hyperparams: Optional[dict[str, float]] = None,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        **unused_kwargs: Any,
    ):
        self.model = model
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.hyperparams = (
            self._default_hyperparams if hyperparams is None else hyperparams
        )
        self.x = x
        self.y = y

    def _xy(
        self, x: Optional[NDArray] = None, y: Optional[NDArray] = None
    ) -> tuple[NDArray, NDArray]:
        return self.x if x is None else x, self.y if y is None else y

    def fit(self, x: Optional[NDArray] = None, y: Optional[NDArray] = None) -> None:
        x, y = self._xy(x, y)
        self.model = self.optimizer(
            x, y, self.model, self.cost_function, **self.hyperparams
        )

    def predict(self, x: Optional[NDArray] = None, thresh: bool = False) -> ArrayLike:
        x = self.x if x is None else x
        return self.model.predict(x, thresh=thresh)

    def score(
        self,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        thresh: bool = False,
    ) -> float:
        x, y = self._xy(x, y)
        return accuracy(self.predict(x, thresh=thresh), y)

    def cost(self, x: Optional[NDArray] = None, y: Optional[NDArray] = None) -> float:
        x, y = self._xy(x, y)
        return self.cost_function(x, y, self.model)

    def partial_derivatives(
        self, x: Optional[NDArray] = None, y: Optional[NDArray] = None
    ) -> Sequence[NDArray | float]:
        x, y = self._xy(x, y)
        return self.model.partial_derivatives(x)

    def params(self) -> ArrayLike:
        return self.model.parameters


class LinearRegression(Regression):
    """Linear Regression with L2 regularization"""

    def __init__(
        self,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        w: Optional[ArrayLike] = None,
        b: Optional[float] = None,
        n_features: Optional[int] = None,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        regularization_param: float = 0.1,
        **unused_kwargs: Any,
    ):
        if x is None and n_features is None:
            raise ValueError("Either x or n_features must be provided")
        self.n_features = n_features if n_features is not None else get_n_features(x)
        self.w_init = np.zeros(self.n_features) if w is None else w
        self.b_init = 0 if b is None else b
        pars_dict = {"w": self.w_init, "b": self.b_init, "n_features": self.n_features}
        hyperparams = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "regularization_param": regularization_param,
        }
        super().__init__(
            cost_function=regularized_least_squares_cost_function,
            model=LinearModel(**pars_dict),
            optimizer=regularized_batch_gradient_descent,
            hyperparams=hyperparams,
            x=x,
            y=y,
        )


class LogisticRegression(Regression):
    """Logistic Regression with L2 regularization"""

    def __init__(
        self,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        w: Optional[ArrayLike] = None,
        b: Optional[float] = None,
        n_features: Optional[int] = None,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        regularization_param: float = 0.1,
        threshold: float = 0.5,
        activation_function: LogisticMapper = sigmoid,
        **unused_kwargs: Any,
    ):
        if x is None and n_features is None:
            raise ValueError("Either x or n_features must be provided")
        self.n_features = n_features if n_features is not None else get_n_features(x)
        self.w_init = np.zeros(self.n_features) if w is None else w
        self.b_init = 0 if b is None else b
        pars_dict = {
            "w": self.w_init,
            "b": self.b_init,
            "n_features": self.n_features,
            "threshold": threshold,
            "activation_function": activation_function,
        }
        hyperparams = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "regularization_param": regularization_param,
        }
        super().__init__(
            cost_function=regularized_logistic_cost_function,
            model=LogisticModel(**pars_dict),
            optimizer=regularized_batch_gradient_descent,
            hyperparams=hyperparams,
            x=x,
            y=y,
        )

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for the activation function for logistic regression"""
        self.model.threshold = threshold


class SoftmaxRegression(Regression):
    """Logistic Regression with L2 regularization"""

    def __init__(
        self,
        x: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        w: Optional[ArrayLike] = None,
        b: Optional[float] = None,
        n_features: Optional[int] = None,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        regularization_param: float = 0.1,
        threshold: float = 0.5,
        activation_function: LogisticMapper = stable_softmax,
        **unused_kwargs: Any,
    ):
        if x is None and n_features is None:
            raise ValueError("Either x or n_features must be provided")
        self.n_features = n_features if n_features is not None else get_n_features(x)
        self.w_init = np.zeros(self.n_features) if w is None else w
        self.b_init = 0 if b is None else b
        pars_dict = {
            "w": self.w_init,
            "b": self.b_init,
            "n_features": self.n_features,
            "threshold": threshold,
            "activation_function": activation_function,
        }
        hyperparams = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "regularization_param": regularization_param,
        }
        super().__init__(
            cost_function=regularized_cross_entropy_cost_function,
            model=SoftmaxModel(**pars_dict),
            optimizer=regularized_batch_gradient_descent,
            hyperparams=hyperparams,
            x=x,
            y=y,
        )

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for the activation function for logistic regression"""
        self.model.threshold = threshold


class RegressionTypes(Enum):
    linear = "linear"
    logistic = "logistic"
    softmax = "softmax"

    def get_regressor(self, **kwargs: Any) -> Regression:
        if self.name == "linear":
            return LinearRegression(**kwargs)
        elif self.name == "logistic":
            return LogisticRegression(**kwargs)
        elif self.name == "softmax":
            kwargs["y"] = kwargs["y"].astype(np.int)
            kwargs["activation_function"] = stable_softmax
            return SoftmaxRegression(**kwargs)


def run_regression(
    regression_type: RegressionTypes | str,
    x: NDArray,
    y: NDArray,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    w: ArrayLike = (0,),
    b: NDArray | float = 0,
    regularization_param: float = 0.1,
    activation_function: LogisticMapper = sigmoid,
    threshold: float = 0.5,
    verbose: bool = True,
    fit: bool = True,
) -> Regression:
    """Run a linear or logistic regression algorithm"""
    regression = RegressionTypes(regression_type).get_regressor(
        x=x,
        y=y,
        w=w,
        b=b,
        n_features=get_n_features(x),
        learning_rate=learning_rate,
        max_iter=max_iter,
        regularization_param=regularization_param,
        activation_function=activation_function,
        threshold=threshold,
    )
    if not fit:
        return regression

    if verbose:
        print(f"Initial cost: {regression.cost()}")
    regression.fit()
    if verbose:
        print(f"Final cost: {regression.cost()}")
    return regression
