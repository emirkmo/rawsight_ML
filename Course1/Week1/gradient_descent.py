import numpy as np
from utils import OLS, LeastSquaresCostFunction, CostFunction, get_n_features


def batch_gradient_descent(x: np.ndarray, y: np.ndarray, model: OLS, cost_function: CostFunction,
                           learning_rate: float, max_iter: int) -> OLS:
    """
    Gradient descent algorithm for linear regression.
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        model    : callable initialized with parameters
        cost_function : callable that computes the cost function for linear regression
        learning_rate : learning rate for gradient descent
        max_iter : maximum number of iterations to run gradient descent
    Returns:
        model : callable initialized with parameters that minimize the cost function
    """
    # initialize model parameters
    parameters = np.array(model.parameters, dtype=object)

    # run gradient descent
    niter = 0
    while niter < max_iter:
        # compute gradient
        gradients = np.array(cost_function.compute_gradient(x, y, model))

        # update model & parameters
        parameters = parameters - learning_rate * gradients
        model.parameters = parameters

        niter += 1
    return model


def main():
    x_train = np.array([1.0, 2.0])  # features
    y_train = np.array([300.0, 500.0])  # target value

    model = OLS(w=100, b=0.5)
    cost_function = LeastSquaresCostFunction()

    initial_cost = cost_function(x_train, y_train, model)
    print(f"Initial cost: {initial_cost}, initial model: {model}")

    model = batch_gradient_descent(x_train, y_train, model, cost_function, learning_rate=0.1, max_iter=500)

    final_cost = cost_function(x_train, y_train, model)
    print(f"Final cost: {final_cost}, final model: {model}")


if __name__ == '__main__':
    main()
