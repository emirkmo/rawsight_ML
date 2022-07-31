from numpy.typing import ArrayLike, NDArray
from utils.models import LogisticModel, Model
from utils.cost_functions import logistic_cost_function
from utils.optimizers import batch_gradient_descent
from utils import get_n_features
import numpy as np


def run_logistic_regression(x: NDArray, y: NDArray, learning_rate: float = 0.1,
                            max_iter: int = 1000, w: ArrayLike = (0,), b: float = 0) -> Model:
    model = LogisticModel(w=w, b=b, n_features=get_n_features(x))
    print(f"Initial cost: {logistic_cost_function(x, y, model)}")
    model = batch_gradient_descent(x, y, model, logistic_cost_function, learning_rate, max_iter)
    print(f"Final cost: {logistic_cost_function(x, y, model)}")
    return model


def main():
    x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = run_logistic_regression(x_train, y_train, max_iter=10000)
    print(model)
    print("Expected parameters from lab: w:[5.28 5.08], b:-14.222409982019837")

    print("-"*100)
    # Dataset2
    x_train = np.array([0., 1, 2, 3, 4, 5])
    y_train = np.array([0,  0, 0, 1, 1, 1])

    model = run_logistic_regression(x_train, y_train, learning_rate=0.1, max_iter=10000, w=0, b=0)
    print(model)
    print("Expected parameters from lab: w:[5.12], b:[-12.61]")


if __name__=='__main__':
    print("-" * 100)
    main()
