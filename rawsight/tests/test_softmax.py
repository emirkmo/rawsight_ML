from rawsight.models.softmax import softmax, stable_softmax
from rawsight.regression import run_regression
import numpy as np
import pytest


@pytest.mark.parametrize("function", [softmax, stable_softmax])
def test_softmax_shape(function):
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    assert function(x).shape == (3, 4)
    x2 = np.array([[1, 2, 3, 6]])
    assert function(x2).shape == (1, 4)
    x3 = np.array([1, 2, 3, 6])
    assert function(x3).shape == (4,)
    x4 = np.array([[1], [2], [3], [6]])
    assert function(x4).shape == (4, 1)


@pytest.mark.parametrize("function", [softmax, stable_softmax])
def test_softmax_values(function):
    x = np.array([[1, 2, 3, 6], [4, 5, 6, 7], [7, 8, 9, 10]])
    assert np.allclose(
        function(x),
        np.array(
            [
                [0.00626879, 0.01704033, 0.04632042, 0.93037047],
                [0.03205860, 0.08714432, 0.23688282, 0.64391426],
                [0.03205860, 0.08714432, 0.23688282, 0.64391426],
            ]
        ),
    )


def test_logistic_regression_with_softmax():
    np.random.seed(1)
    dataset_1 = {
        "X_tmp": np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]),
        "y_tmp": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        "w_tmp": np.ones((2, 2)),
        "b_tmp": np.array([-1.0, -1.0]),
        "lambda_tmp": 0.7,
    }

    regression = run_regression(
        regression_type="softmax",
        x=dataset_1["X_tmp"],
        y=dataset_1["y_tmp"].astype(int),
        w=dataset_1["w_tmp"],
        b=dataset_1["b_tmp"],
        max_iter=10000,
        regularization_param=dataset_1["lambda_tmp"],
        learning_rate=0.01,
    )

    regression2 = run_regression(
        regression_type="logistic",
        x=dataset_1["X_tmp"],
        y=dataset_1["y_tmp"].astype(int),
        w=dataset_1["w_tmp"][0],
        b=dataset_1["b_tmp"][0],
        max_iter=10000,
        regularization_param=dataset_1["lambda_tmp"],
        learning_rate=0.01,
    )

    regression2.set_threshold(0.5)
    regression.set_threshold(0.5)

    assert regression.score(thresh=True) == regression2.score(thresh=True)
    assert np.allclose(
        regression.predict(thresh=True), regression2.predict(thresh=True)
    )
    assert np.allclose(regression.predict(thresh=True), dataset_1["y_tmp"])
    assert np.shape(regression.model.evaluate(dataset_1["X_tmp"])) == (
        dataset_1["y_tmp"].shape[0],
        dataset_1["w_tmp"].shape[-1],
    )


def test_softmax_3d():
    from sklearn.datasets import make_blobs

    X_train, y_train = make_blobs(
        n_samples=1000, n_features=2, centers=3, cluster_std=1.0, random_state=1
    )

    regression = run_regression(
        regression_type="softmax",
        x=X_train,
        y=y_train.astype(int),
        w=np.ones([3, 2]),
        b=np.ones(3),
        max_iter=20000,
        regularization_param=0.1,
        learning_rate=0.01,
    )
    print(regression.model.evaluate(X_train))
    print(regression.predict(thresh=True))
    print(y_train)
    assert True


if __name__ == "__main__":
    test_softmax_shape()
    test_softmax_values()
    test_logistic_regression_with_softmax()
    test_softmax_3d()
    print("All tests passed")
