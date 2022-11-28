"""Logistic regression tests.
TODO: Change tests to use the new LogisticRegression class and run_regression function.
"""
from utils.cost_functions import (
    regularized_logistic_cost_function,
    logistic_cost_function,
)
from utils.models import LogisticModel
from utils.input_validation import get_n_features
from utils.regression import run_regression, LogisticRegression as MyLogisticRegression

# from Course1.Week3.logistic_regression import run_logistic_regression
import numpy as np
from sklearn.linear_model import LogisticRegression
import pytest

np.random.seed(1)
dataset_1 = {
    "X_tmp": np.array(
        [
            [
                4.17022005e-01,
                7.20324493e-01,
                1.14374817e-04,
                3.02332573e-01,
                1.46755891e-01,
                9.23385948e-02,
            ],
            [
                1.86260211e-01,
                3.45560727e-01,
                3.96767474e-01,
                5.38816734e-01,
                4.19194514e-01,
                6.85219500e-01,
            ],
            [
                2.04452250e-01,
                8.78117436e-01,
                2.73875932e-02,
                6.70467510e-01,
                4.17304802e-01,
                5.58689828e-01,
            ],
            [
                1.40386939e-01,
                1.98101489e-01,
                8.00744569e-01,
                9.68261576e-01,
                3.13424178e-01,
                6.92322616e-01,
            ],
            [
                8.76389152e-01,
                8.94606664e-01,
                8.50442114e-02,
                3.90547832e-02,
                1.69830420e-01,
                8.78142503e-01,
            ],
        ]
    ),
    "y_tmp": np.array([0, 1, 0, 1, 0]),
    "w_tmp": np.array(
        [-0.40165317, -0.07889237, 0.45788953, 0.03316528, 0.19187711, -0.18448437]
    ),
    "b_tmp": 0.5,
    "lambda_tmp": 0.7,
}

dataset_2 = {
    "X": np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]),
    "y": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    "w": (1, 1),
    "b": -1,
}


def test_regularized_logistic_cost_function():
    X_tmp = dataset_1["X_tmp"]
    y_tmp = dataset_1["y_tmp"]
    w_tmp = dataset_1["w_tmp"]
    b_tmp = dataset_1["b_tmp"]
    lambda_tmp = dataset_1["lambda_tmp"]
    model = LogisticModel(w=w_tmp, b=b_tmp, n_features=get_n_features(X_tmp))
    totcost = regularized_logistic_cost_function(X_tmp, y_tmp, model, lambda_tmp)
    assert 0.69 > totcost > 0.68


def test_logistic_cost_function():
    X_tmp = dataset_1["X_tmp"]
    y_tmp = dataset_1["y_tmp"]
    w_tmp = dataset_1["w_tmp"]
    b_tmp = dataset_1["b_tmp"]
    model = LogisticModel(w=w_tmp, b=b_tmp, n_features=get_n_features(X_tmp))
    totcost = logistic_cost_function(X_tmp, y_tmp, model)
    assert 0.66 > totcost > 0.65


@pytest.fixture
def regularized_logistic_regression():
    regression = run_regression(
        "logistic",
        dataset_2["X"],
        dataset_2["y"],
        0.3,
        10000,
        dataset_2["w"],
        dataset_2["b"],
        1,
    )
    # my_model = run_logistic_regression(dataset_2["X"], dataset_2["y"],
    #                                    learning_rate=0.1, max_iter=20000, w=dataset_2["w"],
    #                                    b=dataset_2["b"], cost_function=regularized_logistic_cost_function)
    my_model = regression.model
    return my_model  # swap to yield if adding cleanup.


def test_logistic_regression_against_sklearn(regularized_logistic_regression):
    np.random.seed(1)
    my_model = regularized_logistic_regression
    w, b = my_model.parameters
    w = np.round(w, 2)
    b = np.round(b, 1)
    lr_model = LogisticRegression()
    lr_model.fit(dataset_2["X"], dataset_2["y"])

    assert np.all(np.round(lr_model.coef_, 2) == w)
    assert np.round(lr_model.intercept_, 1) == b


def test_logistic_regression_against_tensorflow():
    import tensorflow as tf
    from datasets import load_tumor_simple
    from utils.regression import LogisticRegression

    dataset = load_tumor_simple()

    logistic_layer = tf.keras.layers.Dense(
        units=1, input_dim=1, activation="sigmoid", name="logistic"
    )
    p_log = logistic_layer(dataset.X_train)
    logistic_regression = LogisticRegression(
        x=dataset.X_train,
        y=dataset.y_train,
        w=logistic_layer.get_weights()[0],
        b=logistic_layer.get_weights()[1],
    )

    assert all(abs(logistic_regression.predict() - p_log.numpy().reshape(-1)) <= 0.01)


if __name__ == "__main__":
    test_regularized_logistic_cost_function()
    test_logistic_cost_function()
    # my_model = run_logistic_regression(dataset_2["X"], dataset_2["y"],
    #                                    learning_rate=0.3, max_iter=10000, w=dataset_2["w"],
    #                                    b=dataset_2["b"], cost_function=regularized_logistic_cost_function)
    test_logistic_regression_against_sklearn()
    test_logistic_regression_against_tensorflow()
