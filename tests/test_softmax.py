from utils.models.softmax import softmax
import numpy as np


def test_softmax_shape():
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    assert softmax(x).shape == (3, 4)

    x2 = np.array([[1, 2, 3, 6]])
    assert softmax(x2).shape == (1, 4)
    x3 = np.array([1, 2, 3, 6])
    assert softmax(x3).shape == (4, )
    x4 = np.array([[1], [2], [3], [6]])
    assert softmax(x4).shape == (4, 1)


def test_softmax_values():
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    assert np.allclose(
        softmax(x), np.array(
            [
            [0.00626879, 0.01704033, 0.04632042, 0.93037047],
            [0.03205860, 0.08714432, 0.23688282, 0.64391426],
            [0.03205860, 0.08714432, 0.23688282, 0.64391426]
            ]
        )
    )


if __name__ == '__main__':
    test_softmax_shape()
    test_softmax_values()
    print("All tests passed")
