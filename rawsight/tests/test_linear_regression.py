import numpy as np

from datasets import load_housing_data, Dataset
from typing import Callable
from sklearn.linear_model import SGDRegressor
from rawsight.cost_functions import least_squares_cost_function
from rawsight.cost_functions.cost_functions import _least_squares_cost
from rawsight.optimizers import batch_gradient_descent, regularized_batch_gradient_descent
from rawsight.models import LinearModel, Model
from rawsight.regression import run_regression
from functools import partial
import pytest

Optimizer = Callable[..., Model]


def run_linear_regression(dataset: Dataset, optimizer: Optimizer) -> Model:
    # regression = run_regression("linear", dataset.X_train, dataset.y_train optimizer)
    model = LinearModel(w=1, b=0, n_features=dataset.X_train.shape[1])
    return optimizer(dataset.X_train, dataset.y_train, model=model)


@pytest.fixture
def housing_data() -> Dataset:
    dataset = load_housing_data()
    dataset.normalize_features()
    yield dataset


@pytest.fixture
def sgdr(housing_data):
    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(housing_data.X_train, housing_data.y_train)
    return sgdr


@pytest.fixture
def linear_regression(housing_data) -> Model:
    bg = partial(
        batch_gradient_descent,
        cost_function=least_squares_cost_function,
        learning_rate=0.03,
        max_iter=20000,
    )
    model = run_linear_regression(housing_data, bg)
    return model


@pytest.fixture
def linear_regression_regularized(housing_data) -> Model:
    bgr = partial(
        regularized_batch_gradient_descent,
        cost_function=least_squares_cost_function,
        learning_rate=0.01,
        max_iter=20000,
        regularization_param=0.15,
    )
    model = run_linear_regression(housing_data, bgr)
    return model


def compare_to_sklearn(input_model: Model, sgdr_: SGDRegressor) -> None:
    print(input_model.parameters[0])
    assert isinstance(input_model.parameters[0], np.ndarray)
    assert pytest.approx(input_model.parameters[0], abs=1) == np.array(sgdr_.coef_)
    assert pytest.approx(np.atleast_1d(input_model.parameters[1]), abs=1) == np.array(
        sgdr_.intercept_
    )


def test_linear_regression_parameters_vs_sklearn(
    linear_regression, linear_regression_regularized, sgdr
):
    compare_to_sklearn(input_model=linear_regression, sgdr_=sgdr)
    compare_to_sklearn(input_model=linear_regression_regularized, sgdr_=sgdr)


def test_regularized_better(
    housing_data, sgdr, linear_regression, linear_regression_regularized
):
    model = linear_regression
    model_reg = linear_regression_regularized

    assert isinstance(model, LinearModel)
    assert isinstance(model_reg, LinearModel)

    model_cost = _least_squares_cost(model.parameters[0], sgdr.coef_)
    model_reg_cost = _least_squares_cost(model_reg.parameters[0], sgdr.coef_)
    assert model_reg_cost <= model_cost


def test_parity_to_tensorflow():
    import tensorflow as tf
    from datasets import load_tumor_simple
    from rawsight.regression import LinearRegression

    dataset = load_tumor_simple()

    linear_layer = tf.keras.layers.Dense(
        units=1, input_dim=1, activation="linear", name="linear"
    )
    p_lin = linear_layer(dataset.X_train)
    linear_regression = LinearRegression(
        x=dataset.X_train,
        y=dataset.y_train,
        w=linear_layer.get_weights()[0],
        b=linear_layer.get_weights()[1],
    )

    assert all(abs(linear_regression.predict().reshape(-1, 1) - p_lin.numpy()) <= 0.01)


if __name__ == "__main__":
    test_linear_regression_parameters_vs_sklearn()
    test_regularized_better()
    test_parity_to_tensorflow()
