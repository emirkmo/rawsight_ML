from typing import Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike

from rawsight.cost_functions import NDArrayInt
from rawsight.models.logistic import LogisticModel, LogisticMapper, LogisticNNModel


# from .model import BaseLinearModel, BaseNeuralNetLinearModel, Model


def softmax(x: NDArray) -> NDArray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def stable_softmax(x: NDArray) -> NDArray:
    """Compute softmax values for each sets of scores in x. Stable because
    each row is max normalized before summation, avoiding infinity. However,
    small numbers around zero are not avoided."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class SoftmaxModel(LogisticNNModel):
    def __init__(
        self,
        w: NDArray,
        b: NDArray,
        n_features: int = 1,
        activation_function: LogisticMapper = stable_softmax,
        threshold: float = 0.5,
    ):
        super().__init__(
            w=w,
            b=b,
            n_features=n_features,
            activation_function=activation_function,
            threshold=threshold,
        )

    def partial_derivatives(self, x: NDArray):
        return self.dw(x), self.db()

    def predict(self, x: NDArray, thresh: bool = True) -> ArrayLike | NDArray:
        fy = self.evaluate(x)
        if thresh:
            return np.argmax(fy > self.threshold, axis=-1)
        return np.argmax(fy, axis=-1)


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    print(x.shape)
    print(softmax(x))
    print(np.sum(softmax(x), axis=-1, keepdims=True))
    print(stable_softmax(x))
    print(np.sum(stable_softmax(x), axis=-1, keepdims=True))

    dataset_1 = {
        "X_tmp": np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]),
        "y_tmp": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        "w_tmp": np.ones((2, 2)),
        "b_tmp": np.array([-1.0, -1.0]),
        "lambda_tmp": 0.7,
    }

    x = dataset_1["X_tmp"]
    y = dataset_1["y_tmp"].astype(int)
    w = dataset_1["w_tmp"]
    b = dataset_1["b_tmp"]

    s = SoftmaxModel(
        w=np.array([[-0.95079089, 0.57670977], [-0.95079089, 0.57670977]]),
        b=np.array([1.15495607, -3.15495607]),
        n_features=2,
    )

    # x2 = np.array([[1, 2, 3, 6]])
    # print(x2.shape)
    # print(softmax(x2))
    # x3 = np.array([1, 2, 3, 6])
    # print(x3.shape)
    # print(softmax(x3))
    # x4 = np.array([[1], [2], [3], [6]])
    # print(x4.shape)
    # print(softmax(x4))
