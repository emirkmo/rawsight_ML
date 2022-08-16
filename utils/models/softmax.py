import numpy as np
from numpy.typing import NDArray
#from .model import BaseLinearModel, BaseNeuralNetLinearModel, Model


def softmax(x: NDArray) -> NDArray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1, keepdims=True)


if __name__=='__main__':
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    print(x.shape)
    print(softmax(x))
    x2 = np.array([[1, 2, 3, 6]])
    print(x2.shape)
    print(softmax(x2))
    x3 = np.array([1, 2, 3, 6])
    print(x3.shape)
    print(softmax(x3))
    x4 = np.array([[1], [2], [3], [6]])
    print(x4.shape)
    print(softmax(x4))
